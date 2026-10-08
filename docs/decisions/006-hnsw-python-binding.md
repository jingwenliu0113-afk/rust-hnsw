# 006：HNSW 的 Python binding 與 efSearch 介面

日期：2026-10-08
狀態：採用

## 背景

002 固定了 batch interface，但把「efSearch 等查詢參數怎麼傳」留到 HNSW adapter 出現時決定。
現在 Rust 的 HNSW insert / search / save / load 都已完成（004、005），要建 HNSW 的 Python binding，
並讓之後的 hnswlib / FAISS adapter 走同一個介面。

## 決策

### 介面擴充（`benchmarks/interface.py`）

- 新增 `HnswAnnIndex(AnnIndex)` protocol，多一個方法：`set_ef_search(ef: int) -> None`。
- `search(queries, k)` 的簽名不變，使用最近一次 `set_ef_search` 的值。
- **ef 必須明確設定**：沒有預設值。在 `set_ef_search` 之前呼叫 `search` 會回 `RuntimeError`。
  三套實作的預設 ef 各不相同（依記憶，hnswlib 為 10、FAISS 為 16；**未確認**），依賴預設值會讓比較不對等。
- `set_ef_search` 先檢查再存下，失敗時保留前一個值：
  - 非整數（含 `None`、float）回 `TypeError`（`operator.index`）。
  - `ef < 1` 回 `ValueError`。
- 實際使用的 ef：
  - Rust 依 004 使用 `max(efSearch, k)`。
  - hnswlib / FAISS 在 `ef < k` 時的行為**未確認**，可能與 Rust 不同；寫 adapter 時以原始碼或實測確認，並記入 decision log。
- **parameter sweep 一律使用 `ef ≥ k`**，讓上述差異不影響比較。
  由 runner 在計時前檢查 sweep config 的每一組 (ef, k)，不符即拒絕執行；`adapter.search` 不做這項檢查，避免增加計時路徑的成本。

### Rust binding（`rust_hnsw.HnswIndex`）

- 建構：`HnswIndex(dim, m=None, ef_construction=None, seed=None)`。
  省略或傳 `None` 時使用 `HnswParams::default()`（m=16、ef_construction=200、seed=100），預設值只存在 Rust 一處。
- 方法：
  - `add(data)`。
  - `search(queries, k, ef_search)` → `(ids: int64, distances: float32)`。
  - `save(path)`、`HnswIndex.load(path)`（staticmethod）。
- 唯讀屬性：`dim`、`m`、`ef_construction`、`seed`、`max_level`（空 index 為 `None`）；`len()`。
- 輸入規則同 002：只收 float32、C-contiguous 的 2-D array；add、search、save、load 期間釋放 GIL。
- Rust 的 `PADDING_ID` 轉成 `-1`，distance 維持 `+inf`（002）。
- binding 的 `search` 與 Rust 核心語意相同，不額外檢查 `ef_search`；「ef ≥ 1」由 adapter 負責。
- 錯誤對應：
  - `HnswError` → `ValueError`。
  - `PersistError::Io` → PyO3 依 `io::ErrorKind` 轉成對應的 `OSError` 子類（例如 `FileNotFoundError`）。
  - 其他 `PersistError`（格式不符、損壞、截斷）→ `ValueError`。
  - Python 參數的型別或範圍轉換失敗（例如負數的 k），由 PyO3 回 `TypeError` / `OverflowError`，不重新對應。

### Adapter（`benchmarks/adapters.py`）

- `RustHnsw(m, ef_construction, seed)`：
  - 三個參數都必須是整數（`operator.index`）；`None` 或 float 立即 `TypeError`，不默默套用 binding 的預設值。
  - `build`：建 index 並 add。
  - `set_ef_search`：檢查後存下 ef。
  - `search`：一次 FFI 呼叫。

## 為什麼

- efSearch 是 sweep 的主要變數，三套實作必須用同一種方式設定。
- 不把 ef 放進 `search` 的參數：hnswlib / FAISS 的原生 API 都是「先設定、再搜尋」；
  若 adapter 在每次 search 內設定，會把設定成本算進計時。
- 不提供預設 ef：避免某個 adapter 不小心用了自己的預設值。

## 放棄的替代方案

- **`search(queries, k, **params)`**：每次 search 都要解析參數，adapter 間行為不易統一。
- **把 ef 當成 build 參數**：違反 hnswlib / FAISS 的語意，也無法對同一個 index 做 sweep。
- **adapter 提供預設 ef**：三套實作的預設不同，容易默默造成不對等的比較。

## 對實驗的影響

- 正式 sweep 的流程：
  1. build，計時 build time。
  2. 對每個 ef：`set_ef_search`（不計時）→ warm-up → 計時的 search。
- 與 002 的計時協定一致。

### 寫 hnswlib / FAISS adapter 前必須確認（目前都未確認）

以下都必須在看到任何 benchmark 結果之前確認，並以 decision log 記錄處理方式：

- **結果不足 k**：依記憶，hnswlib 的 `knn_query` 只要有任何 query 的結果不足 k，就會對整批 raise；FAISS 補齊時的 id 與距離值也需確認。
  - 先用 `tests/test_hnsw_binding.py` 的 `duplicate_data(12, 300, 6)` 與 k = 80 實測。
  - 若會 raise，不得改成 per-query 的 Python 迴圈（違反 002）。
- **ef 與 k 的關係**以及**預設值**（見上）。
- **呼叫成本不只取決於 Python 層**：002 寫「三套實作付出相同的 Python / FFI 呼叫成本」，實際上只有 Python 層相同。
  - FFI 層各不相同：PyO3、pybind11、SWIG。
  - 本專案在 FFI 內完成 id 轉換，hnswlib / FAISS 的 adapter 則需要在 Python 端做 numpy 後處理。
  - 因此 runner 必須對每個 adapter 分別量測空呼叫成本（包含後處理），並與 P50 / P95 一起報告。
  - adapter 的後處理應盡量 zero-copy 或 in-place（建議，非硬性規則）。

## 審查摘要（2026-10-08）

實作後以兩個角度審查（介面與公平性、強健性與測試鑑別力），各配一個反駁者，共 15 個 findings。
沒有發現實作與 002 / 006 不符的錯誤，修正集中在測試鑑別力與文件。

**測試補強**（反駁者以 mutation 重現：拿掉對應行為後原測試仍全綠）：

- GIL 釋放：背景 thread 執行時，main thread 必須能繼續執行。
- efSearch 確實傳進核心：recall 必須隨 ef 提高；`ef < k` 時恰好提高到 k，不會更大。
- sweep 語意：同一個 adapter 以非單調順序反覆 `set_ef_search`，每次都生效；設定失敗時保留前一個值。
- `max_level` 以獨立計算的常數斷言。

**文件**：建構子簽名改為實際的 `None` 預設；把 hnswlib / FAISS 的行為標為未確認。

**駁回**：

- PyO3 例外類別一致化。
- 並行存取語意（屬 Optional）。
- bytes 路徑。
- adapter 契約測試參數化（等第二個 adapter 出現再說）。
