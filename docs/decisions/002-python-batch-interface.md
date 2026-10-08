# 002：統一 Python batch interface 與計時協定

日期：2026-10-08
狀態：採用

## 背景

SPEC 要求 Rust 實作、hnswlib、FAISS HNSW 三者都從同一層 Python batch interface 被測，
避免不同 FFI / 呼叫開銷污染比較。
需要先固定：介面長相、輸入輸出格式、執行緒數、QPS 與 latency 怎麼量。

## 決策

### 介面（`benchmarks/interface.py` 的 `AnnIndex`）

每個實作寫一個 adapter，只提供：

- `name: str`
- `build(data) -> None`
- `search(queries, k) -> (ids, distances)`

調整 efSearch 等查詢參數的方法，等 HNSW adapter 出現時再加入介面（另寫 decision log）。

### 輸入

- harness 在**計時之外**呼叫一次 `prepare_vectors`：
  轉成 float32、C-contiguous、2-D，檢查 finite 與非零 row，並做 L2 正規化。
- 三套實作吃同一份 unit vectors。
- Rust binding 只收 float32 C-contiguous array，不做隱性轉型或複製；格式不符直接報錯。

### 輸出

- `ids`：int64，shape `(num_queries, k)`。
- `distances`：float32，shape `(num_queries, k)`，為 cosine distance（`1 − ip`）。
- 每個 row 依距離升冪排序，row 內 id 不重複。
- 結果不足 k 時，尾端補 `id = -1`、`distance = +inf`。
- 各 adapter 負責轉換成這個格式，例如：
  - FAISS `METRIC_INNER_PRODUCT` 回傳的是相似度，要轉成 `1 − ip`。
  - hnswlib 的 label 是 uint64，要轉成 int64。
- `check_search_result` 驗證以上所有規則。

### 呼叫方式

- 每次 `search` 是**一次 FFI 呼叫處理整批 queries**；adapter 內不得有 per-query 的 Python 迴圈。
- Rust binding 在 search 期間釋放 GIL（與 hnswlib 一致）。

### 執行緒

- Core 階段 build 與 search **一律單執行緒**。
  - hnswlib：`set_num_threads(1)` / `num_threads=1`。
  - FAISS：`omp_set_num_threads(1)`。
  - Rust 實作本身是單執行緒。
- parallel construction / concurrent queries 屬 Optional，不在 Core。

### 計時協定

- **QPS**：整批 queries 一次 `search` 呼叫的 wall time 換算。
- **P50 / P95 latency**：以 batch size = 1 逐筆呼叫**同一個介面**量測。
  三套實作付出相同的 Python / FFI 呼叫成本。
- warm-up 與重跑次數在 benchmark runner 實作時固定，並寫入每次 benchmark 的 config 紀錄。
- 正式 latency / throughput benchmark 必須透過 `./tools/with-mac-lock` 執行（見 `MACHINE_POLICY.md`）。

### Rust binding 技術

PyO3 + rust-numpy + maturin，Python module 名稱為 `rust_hnsw`。

## 為什麼

- 若各自用原生 API 測，FFI 與資料轉換成本不同，會污染比較。
- 用 per-query Python 迴圈量 QPS，結果會被 Python overhead 主導。
- 在 Rust 內部量單筆 latency 無法對 hnswlib / FAISS 做同等量測；
  用相同的 batch-size-1 呼叫路徑，至少讓三者的 overhead 一致。
- 正規化放在計時之外並只做一次，避免某個實作因為內部多做正規化而吃虧或佔便宜。

## 放棄的替代方案

- **各實作用原生 API 直接測**：FFI 差異污染比較。
- **per-query 迴圈量 QPS**：被 Python overhead 主導。
- **Rust 內部計時 latency**：baseline 無法同等量測。
- **Core 就用多執行緒**：屬 Optional，且會讓比較多一個變數。

## 對實驗的影響

- Rust 實作的 insert / query 內部仍會再正規化一次（對 unit vector 幾乎是 no-op，但有 O(d) 成本）。
  這是對自己實作略為不利的 overhead，保留並在報告中註明；
  若 profiling 顯示它佔比顯著，再以新的 decision log 處理。
- hnswlib 的 space 選擇（`cosine` 會在內部再正規化一次、`ip` 不會）在寫 adapter 時決定，
  原則是：三者對同一份 unit vectors 做相同的 `1 − ip` 計算，不讓任一方做多餘的正規化以外的額外工作。
- latency 數字包含 Python / FFI 呼叫成本；報告時要註明，並在 runner 中量一次空呼叫成本作參考。
