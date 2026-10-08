# 009：profiling 方法（Gate 證據的產生方式）

日期：2026-10-08
狀態：採用

## 背景

SPEC 的 Gate 要求「profiler 已顯示時間主要花在哪」。Advanced 的 optimization 也必須先有 hypothesis，而 hypothesis 來自 profiling。
因此 profiling 的工具與流程要先固定，避免看到結果後才挑選對自己有利的量法。

## 決策

- **工具**：macOS 內建的 `sample`。不需要 sudo 或另外安裝，取樣間隔 1 ms，可以同時看到 Rust 與 C++ 的符號。
- **工作負載**：`benchmarks/profile.py`，透過與 benchmark 相同的 adapter（`RustHnsw`、`Hnswlib`、`FaissHnsw`）重複執行 build 或 search。
  - 兩個階段分開取樣。
  - 參數一律明確給定（007）。
  - 不輸出任何 latency / QPS 數字。
- **流程**：`scripts/profile.sh <名稱> <秒數> -- <profile.py 參數>`。
  - 工作負載印出 `READY` 後才開始取樣，避免把資料載入或 build 混進 search 的取樣。
  - 原始報告與 top-of-stack 排行存在 `results/profiles/`。
- **主要呈現**：top-of-stack（self time）排行，由 `benchmarks/profile_report.py` 解析，Rust v0 符號整理成可讀路徑。
  完整 call graph 保留在原始報告中。
- **作為 Gate 證據的 profiling**：
  - 使用正式的真實資料集。
  - 在 `./tools/with-mac-lock` 下執行，並記錄 git commit 與參數。
  - Rust 與 hnswlib（以及 FAISS）在**相近的 Recall@K** 下比較（依 007 §4，不用相同的 ef）。
- **synthetic 資料的 profile** 只用來驗證工具本身，不作為證據，也不寫進任何結論。

## 已知限制

- 被 inline 的函式算在呼叫者身上。例如 Rust 的 cosine distance 會算進 `search_layer`。
  若需要拆開，再另以 debug line info 搭配 `xctrace`（Time Profiler）補充，屆時另寫 decision log。
- batch search 是單一 FFI 呼叫，Python 直譯器的取樣占比很小；仍會出現在排行中，不另外過濾。

## 為什麼

- `sample` 是零安裝、零權限的取樣 profiler，三套實作都能用同一個方法量測。
- 透過 adapter 取樣，量到的是 benchmark 實際執行的程式路徑。
- 證據的條件（真實資料、lock、相近 recall）先寫死，避免事後挑選。

## 放棄的替代方案

- **dtrace / `cargo flamegraph`**：在 macOS 需要 sudo 或調整 SIP。
- **samply**：需要另外安裝。
- **只 profile Rust**：Gate 要比較的是與 hnswlib 的差距來源，必須兩邊用同一方法。
