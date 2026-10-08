# NEXT

## 目前位置

**Core / Week 1 進行中**：

- 已完成：HNSW insert / search（004）、save / load（005）、Python binding 與 efSearch 介面（006）、
  hnswlib / FAISS adapters（007）、benchmark runner（008，smoke 已驗證）、比較規則的事先登記（010）。
- 下一步：runner 的獨立審查與修正 → 分析 / 報告腳本（依 010）→ 第一個真實資料集（需要使用者決定）。

尚未進 Advanced（Gate 條件都還沒有資料可判斷），沒有做任何 Optional。

## 工作方式（使用者 2026-10-08 指示）

- 自主推進 Core：完成一項 → 測試 → 更新 decision log / NEXT → 乾淨的 commit → push → 直接進下一項。
- `origin` = GitHub Private repo（使用者建立）。測試通過的乾淨 commit 可直接 push；禁止 force push、改寫 history、改 visibility。
- 只在真正的 blocker 才停下來問，例如：大型資料或模型下載、credential、費用、改變已凍結的 protocol 或 metric。

## Commits

- `ddcb27d chore: establish core baseline`
- `7961673 feat(hnsw): add HNSW insert/search with correctness tests`
- `e12fe07 docs: define Docker resource policy`
- `50f469a feat(hnsw): add versioned save/load`
- `f134ff3 feat(python): add HNSW binding and efSearch interface`
- `7fdf8d0 feat(benchmarks): add hnswlib and FAISS adapters with measured semantics`
- `5625cc9 feat(benchmarks): add sample-based profiling workflow`
- `feat(benchmarks): add benchmark runner with pre-registered protocol`（本次）

## 最近完成：benchmark runner（decision 008、010）

- **008 草稿經三個角度審查**（量測方法、指標涵蓋、工程實作），57 個 findings 各有一位驗證者重現或反駁；主要修正見 008 文末。
  - 最重要的一項：草稿「Recall@1 取 k = 10 的第一欄、與 k = 1 等價」在重複向量上不成立（FAISS 第一欄是 tie 中 id 最大者），
    改為每個 ef 另跑一次不計時的 k = 1 search。先前 NEXT 寫的「已實測三者等價」是錯的（只在沒有 tie 的資料上成立）。
- **010**：比較規則（id-based Recall@10、目標 0.90 / 0.95 / 0.99、frontier 讀值不內插）、variance 設計與彙整、scaling、
  被標記點的處理，都在看到任何結果之前登記。
- **runner**（`benchmarks/run.py`、`run_child.py`、`config.py`、`environment.py`、`evaluate.py`、`protocol.py`、`datasets.py`）：
  - 每個（build、實作）在全新的子 process 中量測，只 import 受測的 library；recall 由父 process 在子 process 結束後計算。
  - 執行順序交錯並輪換實作；shuffle 的 id 以 `np.where(ids >= 0, perm[ids], -1)` 映射回來。
  - formal 的前置檢查：lock 祖先、git 乾淨、venv、版本 pin、binding 的編譯資訊（`rust_hnsw.BUILD_INFO`，由新的 `python/build.rs` 記錄）、
    binding 晚於 HEAD、已登記的真實資料集；執行中每個子 process 前重新確認。
  - 子 process 的失敗分類（exit_nonzero / signal / timeout）、crash 前的點保留、SIGTERM 中止仍寫出 run.json。
- **測試**：config 驗證、前置檢查（通過與拒絕兩條路徑）、指標計算、stub adapter 的整合測試（id 映射、Recall@1 來源、
  結果不足 k 不重試、crash / timeout / SIGTERM、formal 檢查）、三個真實 adapter 的 smoke run，以及只讀輸出檔重算全部指標。
  另以 `test_third_party_semantics.py` 釘住 FAISS 在 efC < M 時語意不同、tie 時 k = 1 ≠ 第一欄。
- **驗證**：`./scripts/test.sh` 全部通過，Rust 100 個、pytest 283 個。

## 下一步（依序）

1. **runner 的獨立審查**（ultracode）：以 mutation 檢查測試鑑別力，並逐條對照 008 / 010；修正後另一個 commit。
2. **分析 / 報告腳本**（依 010）：從 run.json 產生 frontier 讀值、variance 彙整、recall vs efSearch 圖；先用 smoke 輸出驗證。
3. **閒置 load 的量測**（008 §2）：在沒有 lock holder、沒有其他重工作時量 10 分鐘，確認 4.0 的門檻；不符就另寫 decision log。
4. **第一個真實資料集**（需要使用者決定，見 Blocked）：決定後寫資料集 decision（來源、授權、SHA-256、子集規則），登記到 `benchmarks/datasets.py`。
5. **第一張 recall vs efSearch 圖與正式 run**（formal config 先 commit，依 010 列出 variance 格點）。
6. **Gate 用的 profiling**（009：真實資料、lock 下、相近 recall）。

## Blocked / 需要使用者決定

- **資料集下載**（尚未提出）：SPEC 需要「標準 ANN 真實資料集」與「真實文本的現代 embedding」。
  兩者都需要下載資料（數百 MB）或 embedding 模型，依使用者規則要先詢問。分析腳本完成後一次提出具體方案。

## 需要留意

- **機器負載**：本 session 多次觀察到其他 Project 在不取 lock 的情況下跑 CPU 重工作（text-to-sql、agent-gym 各數個 100% CPU 的 process），
  1 分鐘 load 約 8.4；incident-investigator 也長時間持有 heavy lock（process 存活，不是 stale lock）。
  正式 run 會被 008 §2 的 load 門檻擋下（等待最多 10 分鐘後中止），這是預期行為；本專案沒有碰它們。
- 本次沒有跑任何正式 benchmark，也沒有取用 Mac heavy lock 或遠端機器。smoke 輸出全部標記 `load`（因為上述負載），只用來驗證 runner。
- 本 Project 沒有使用 Docker。

## SPEC / decision 與現況差異

- 004 只規定 M ≥ 2；005 補上 `MAX_M = 2^31 − 1`。
- 004 的「hnswlib 尚未實測」已由 007 回填。
- 007 §3「不產生 recall」已補記：Recall@1 另以 k = 1 search 取得（008 §7）。
- MACHINE_POLICY 的範例指令改為 `.venv/bin/python -m benchmarks.run`（只改範例，不改規則；`python benchmarks/run.py` 會 import 失敗）。
- 其他沒有發現差異。
