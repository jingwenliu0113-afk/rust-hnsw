# NEXT

## 目前位置

**Core / Day 1 已完成**，下一步進入 Week 1（HNSW skeleton）。
尚未進 Advanced（Gate 條件都還沒有資料可判斷），沒有做任何 Optional。

## 上次完成（2026-10-08）

### 環境

- 安裝 Rust stable 1.99.0（官方 rustup，`--no-modify-path`，未改任何 shell 設定檔）。
- 專案內 `.venv`（uv，CPython 3.13.9），依賴固定在 `requirements-dev.txt`
  （maturin 1.15.0、numpy 2.5.3、pytest 9.1.1）。沒有動到 Anaconda base env。

### Day 1 五項

1. **Rust workspace**：`crates/hnsw`（純 Rust）＋ `python`（PyO3 binding crate）。
   `cargo test` 預設只跑 `crates/hnsw`。
2. **cosine distance 與 exact brute-force**：`crates/hnsw/src/distance.rs`、`exact.rs`。
   f64 正規化後存 f32；exact search 以 f64 累加、依 `(distance, id)` 排序。
3. **correctness tests**（全部先寫測試、確認紅燈後才實作）：
   - Rust 34 個：distance 14、exact 20。
     包含單位圓人工可驗證案例、tie-break、atomic add、錯誤處理、
     與獨立完整排序參考實作比對（含大量重複向量）。
   - Python 50 個：binding、interface、metrics。
     包含與 numpy 參考實作比對、Fortran-order / strided array 被拒絕。
   - mutation 檢查：拿掉 id tie-break 會被「大量重複向量」測試抓到。
4. **Python binding skeleton**：`rust_hnsw.ExactIndex(dim)`，`add` / `search` / `len` / `dim`。
   只收 float32 C-contiguous 2-D array，search 期間釋放 GIL。
5. **固定 benchmark interface**：`benchmarks/interface.py`（`AnnIndex` Protocol、
   `prepare_vectors`、`check_search_result`）、`benchmarks/adapters.py`（`RustExact`）、
   `benchmarks/metrics.py`（`recall_at_k`、`tie_aware_recall_at_k`）。

### Decision logs

- 001 cosine 距離定義、正規化與 tie-break。
- 002 統一 Python batch interface 與計時協定（單執行緒；QPS 用整批呼叫、P50/P95 用 batch size 1）。
- 003 Recall@K 定義：主要為 id-based；tie-aware（eps = 1e-6）只作診斷。在看到任何結果之前登記。

### 驗證

`./scripts/test.sh` 全部通過：rustfmt、clippy `-D warnings`、cargo test（34）、maturin build、pytest（50）。

## 下一步（Week 1，依序）

1. **HNSW skeleton：insert + search**（`crates/hnsw/src/hnsw.rs`，單執行緒）。
   - 先寫 decision log：neighbor selection（simple 或 heuristic）、
     level 分佈（`mL = 1/ln(M)`）、layer 0 的 `M0 = 2M`、seed 與 insertion order 的控制、
     結果不足 k 時的 `-1` padding。
   - 先寫測試：
     - ef ≥ n 的小型案例應與 exact 完全一致。
     - 人工可驗證的小圖。
     - synthetic 資料上 recall 隨 efSearch 整體改善、無系統性惡化（只作 correctness）。
2. **save / load**：檔案格式先寫 decision log，再寫 round-trip 測試。
3. **binding 加 `HnswIndex`**；`AnnIndex` 加查詢參數（efSearch）的介面，需另寫 decision log。
4. **安裝 hnswlib、faiss-cpu 並寫 adapters**：依 002，單執行緒、同一份 unit vectors、輸出轉成統一格式。
5. **第一個小型真實資料集**：選擇與切分需先寫 decision log。
   候選為 ann-benchmarks 原生 angular 的資料集（例如 GloVe-100-angular）。
6. **第一張 recall vs efSearch 圖。**

## Blocked

沒有 hard blocker。

## 需要使用者決定或留意

- 已依使用者決定，`rust-hnsw/` 為獨立 git repo（branch `main`）。
- 使用者已把 `source "$HOME/.cargo/env"` 加到 `~/.zshrc`；`scripts/test.sh` 也會自動載入。
- 本次沒有跑任何正式 benchmark，也沒有取用 Mac heavy lock 或遠端機器（都不需要）。

## SPEC 與現況差異

目前沒有發現差異。
