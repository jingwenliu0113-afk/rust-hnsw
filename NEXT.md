# NEXT

## 目前位置

**Core / Week 1 進行中**：

- 已完成：HNSW insert / search（004）、save / load（005）、Python binding 與 efSearch 介面（006）、hnswlib / FAISS adapters（007）。
- 進行中：benchmark runner（decision 008 草稿審查中）。

尚未進 Advanced（Gate 條件都還沒有資料可判斷），沒有做任何 Optional。

## 工作方式（使用者 2026-10-08 指示）

- 自主推進 Core：完成一項 → 測試 → 更新 decision log / NEXT → 乾淨的 commit → 直接進下一項。
- 可以 commit，不得 push / 建 PR / 改 remote。
- 只在真正的 blocker 才停下來問，例如：大型資料或模型下載、credential、費用、改變已凍結的 protocol 或 metric。

## Commits

- `ddcb27d chore: establish core baseline`
- `7961673 feat(hnsw): add HNSW insert/search with correctness tests`
- `e12fe07 docs: define Docker resource policy`
- `50f469a feat(hnsw): add versioned save/load`
- `f134ff3 feat(python): add HNSW binding and efSearch interface`
- `feat(benchmarks): add hnswlib and FAISS adapters with measured semantics`（本次）

## 最近完成：hnswlib / FAISS adapters（decision 007）

- 版本 pin：hnswlib 0.8.0（本機以 `-O3 -march=native` 編譯）、faiss-cpu 1.15.1（NEON wheel）。
- **先實測再寫**：`tests/test_third_party_semantics.py`（22 個）釘住原生語意。
  - hnswlib：回傳 `1 − ip` distance；`ef` 實際為 `max(ef, k)`；結果不足 k 時整批 raise；`knn_query` 不檢查維度。
  - FAISS：回傳 similarity；efSearch < k 時結果不同，efSearch ≪ k 時會補 -1；padding 為 `−FLT_MAX`；OpenMP 是 per-thread 設定；
    新點在 layer 0 選最多 2M 個鄰居；插入順序不是 row 順序。
- **adapters**（`benchmarks/adapters.py`）：`RustHnsw`、`Hnswlib`、`FaissHnsw`。
  - build 時強制輸入為 unit vectors。
  - 轉換一律在 adapter：similarity → distance、`−FLT_MAX` → `+inf`、uint64 → int64。
  - hnswlib 結果不足時轉成 `IncompleteResultsError`，不逐筆重試。
  - k / 維度檢查與 Rust 一致；單執行緒（FAISS 每次 search 都設定）。
- **契約與語意等價測試**：`tests/test_adapters.py`（56 個），包括：
  - 三者回報同一種距離。
  - 高 ef 的 recall ≥ 0.99；小圖與 exact 一致。
  - sweep 語意、每次 search 只有一次原生呼叫。
  - 以「其他 thread 的 CPU time」驗證單執行緒，並注入多執行緒預設與跨 thread 呼叫。
- **審查**：兩個角度各配一個反駁者，共 25 個 findings，主要修正見 007 文末。
  14 個 adapter mutant（含逐筆查詢、OpenMP 未設定、未檢查 unit norm 等）都會被測試抓到。
- **驗證**：`./scripts/test.sh` 全部通過，Rust 100 個、pytest 154 個。

## 下一步（依序）

1. **benchmark runner**（進行中）：
   - decision 008 草稿已寫好、審查中；審查後定稿，先寫測試再實作。
   - 重點：每個 build 在獨立子 process；計時前驗證 config（`ef ≥ k`、參數為整數），以及 queries 的 dtype、layout 與 unit norm（007 §1）。
   - QPS 重跑 5 次、latency 重跑 3 次；Recall@1 取 k = 10 的第一欄（已實測三者等價，要以測試釘住）。
   - 每個 adapter 分別量空呼叫成本；formal 模式必須在 `./tools/with-mac-lock` 下、git clean、低負載。
   - 先只用 synthetic 資料跑 smoke 模式驗證 runner 本身。
2. **profiling 準備**（Gate 需要「profiler 已顯示時間主要花在哪」）：
   - 本機有 `xctrace`（Time Profiler）與 `sample`，不需安裝。
   - 準備對 Rust search / build 做 profiling 的流程。
3. **第一個真實資料集**（需要使用者決定，見 Blocked）。
4. **第一張 recall vs efSearch 圖**（依 007：跨實作只在相近的 Recall@K 下比較）。

## Blocked / 需要使用者決定

- **資料集下載**（尚未提出）：SPEC 需要「標準 ANN 真實資料集」與「真實文本的現代 embedding」。
  兩者都需要下載資料（數百 MB）或 embedding 模型，依使用者規則要先詢問。
  runner 與 profiling 準備完成後提出具體方案。

## 需要留意

- 本次沒有跑任何正式 benchmark，也沒有取用 Mac heavy lock 或遠端機器（都不需要）。
  審查 agent 的實驗都在 scratchpad 副本進行。
- session 中多次觀察到 incident-investigator 持有 heavy lock（process 存活，不是 stale lock）；本專案沒有碰它。
- 本 Project 沒有使用 Docker。

## SPEC / decision 與現況差異

- 004 只規定 M ≥ 2；005 補上 `MAX_M = 2^31 − 1`。
- 004 的「hnswlib 尚未實測」已由 007 回填。
- 其他沒有發現差異。
