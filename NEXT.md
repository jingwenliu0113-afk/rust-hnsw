# NEXT

## 目前位置

**Core / Week 1 進行中**：

- 已完成：HNSW insert / search（004）、save / load（005）、Python binding 與 efSearch 介面（006）、
  hnswlib / FAISS adapters（007）、benchmark runner（008，含實作後審查）、比較規則（010）、報告腳本。
- 使用者已決定資料集（2026-10-08）：標準 ANN 用 `glove-100-angular`；真實文字 workload 用公開文字 corpus + `google/embeddinggemma-2`。
- 資料集 decision 011 已寫好；GloVe loader（`benchmarks/prepare_glove.py`）與文字 workload 的流程
  （`text_corpus.py`、`embed_text.py`、`prepare_text.py`）已實作並測試。
- 資料準備程式經過兩個角度的審查（19 個 findings，採用 8 個、駁回 11 個，摘要見 011 文末）。最重要的一項：embedding 改用 batch size 1，
  因為原本的重現性檢查在 batch 32 下必然失敗（padding 造成約 2e-7 的差），第一次文字 pilot 已因此中止。
- 進行中：以修正後的程式重新產生 EmbeddingGemma 2 的 pilot（10,000 base + 1,000 query）與 glove-100-angular（新的 manifest 格式，並順便確認重跑結果逐位元相同）。

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
- `2b783bd feat(benchmarks): add benchmark runner with pre-registered protocol`
- `f4500ba feat(benchmarks): add report generation for the pre-registered comparison rules`
- `beb8fd5 docs: record dataset proposal and idle-load attempt in NEXT`
- `0cd4e50 fix(benchmarks): harden runner after implementation review`
- `4478926 feat(datasets): add GloVe and EmbeddingGemma 2 dataset preparation`
- `fix(datasets): make embeddings independent of batch composition and bind frozen ground truth to the vectors`（本次）

## 最近完成：資料準備程式的審查（decision 011 補記）

- 兩個角度（資料正確性、與 runner 的整合），19 個 findings，各有一位驗證者重現或反駁；採用 8 個（實際 6 項修正）、駁回 11 個。
- 修正：
  - embedding 改用 batch size 1（不 padding，向量與批次組成無關；batch 32 下前 32 筆單獨重新 encode 會差約 2e-7，已以 mutation 確認新測試能抓到）。
  - frozen ground truth 使用前另外比對 train / queries 的 SHA-256（原本只改 queries 第 100 列之後，smoke 仍會沿用過期的 gt）。
  - 前處理改為先截斷再過濾；pin tokenizers / safetensors / huggingface-hub 並在執行時比對；
    manifest 記錄 `git status` 逐行清單、Python 版本、`rust_hnsw.BUILD_INFO` 與 Unicode 資料庫版本。
- 測試：pytest 408 個（另 .venv-embed 16 個）。

## 最近完成：runner 實作後的審查

- 三個角度（是否忠實落實 008 / 010、mutation testing、強健性），56 個 findings，各有一位驗證者重現或反駁；20 個照原建議、23 個修改、13 個駁回。摘要見 008 文末。
- 主要修正：binding 新舊判斷改為也比對 tracked 原始碼的 mtime、formal config 必須已 commit、重新確認移到 load 等待之後並在最後再確認一次、
  三個擴充模組都比對、thermal 解析改用 pmset 的實際格式（原本在本機永遠不會觸發）、新增 `memory`（phys_footprint）與 `power` 標記和 AC 電源檢查、
  中止時保留進行中子 process 的已完成點、SIGHUP、收尾不會失敗、不記錄 `CARGO_*` 與看起來像機密的變數、build 層級的 load 標記。
- 測試：補上 mutation 存活的部分（呼叫序列、計時邊界、記憶體時點、每一種標記的正向測試、formal 檢查的接線）。pytest 共 367 個。

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

## 最近完成：報告腳本（decision 010）

- `benchmarks/report.py`：讀一或多個 run.json（同 mode、同資料集、同協定版本才可合併），輸出 report.json、report.md 與圖：
  - frontier 讀值（每個目標取 QPS 最高、P50 / P95 / index memory 最低，附 (M, efC, ef)；不內插、不外推，未達到就記「未達到」）；
  - 被標記點排除前後的讀值不同時兩者都列；hnswlib 整批失敗的點不在 frontier，Rust / FAISS 補齊的點標記「不足 k」；
  - QPS 比值（只在兩者都達到目標時計算）、seed / order variance（median 與 min–max，保留原始值，FAISS 另註）、scaling、recall vs efSearch 檢查；
  - 圖：每個 (n, M, efC) 的 recall vs efSearch（跨 seed min–max）、每個 n 的 recall–QPS 散佈。
- 新依賴：`matplotlib==3.11.2`（pin 在 requirements-dev.txt）。
- 測試：`tests/test_report.py` 以手寫的 run 紀錄逐條驗證 010 的規則，並對真正的 runner 輸出跑一次。pytest 共 298 個。

## 下一步（依序）

1. （已完成）runner 的獨立審查與修正。
2. （已完成）分析 / 報告腳本。
3. **閒置 load 的量測**（008 §2）：在沒有 lock holder、沒有其他重工作時量 10 分鐘，確認 4.0 的門檻；不符就另寫 decision log。
   - 2026-10-08 19:00 前後嘗試：incident-investigator 持有 heavy lock（soak），text-to-sql、agent-gym 各有 100% CPU 的 process，1 分鐘 load 約 7.8–8.2，無法量閒置基準。暫時 blocked，之後再試（只影響 formal run，不影響其他 Core 工作）。
4. **資料集（使用者已決定，decision 011）**：
   - 程式與測試已完成：GloVe 轉換、frozen ground truth（runner 以 SHA-256 加前 100 個 query 重算驗證後使用）、
     manifest 登記、文字 corpus 的前處理與切分（以文章為單位、加鹽 SHA-256 排序、完全相同的文字只留一份）、
     EmbeddingGemma 2 的 embedding（固定 revision、fp32、CPU、8 threads、拒絕 fp16、逐位元重現檢查）。
   - glove-100-angular 已登記（manifest 與 `glove-correctness.toml` 同一個 commit）。以修正後的程式重跑一次，
     train / queries / neighbors 與三個 frozen ground truth 共 6 個檔案的 SHA-256 與第一次完全相同（準備約 466 秒，peak RSS 約 3.1 GB）。
   - pilot 以修正後的程式重跑（batch size 1）；manifest 與 `text-pilot-correctness.toml` 放在同一個 commit。
   - 之後：在兩個資料集上跑 correctness 用的 smoke run（recall vs efSearch，不作效能結論）。
   - pilot 之後：另寫 decision 決定正式文字 workload 的規模（N 與登記的 n）。
5. **第一張 recall vs efSearch 圖與正式 run**（formal config 先 commit，依 010 列出 variance 格點）。
6. **Gate 用的 profiling**（009：真實資料、lock 下、相近 recall）。

## Blocked / 需要使用者決定

- 目前沒有需要使用者決定的事項。資料集已於 2026-10-08 決定（見「目前位置」）。

## 需要留意

- **機器負載**：本 session 多次觀察到其他 Project 在不取 lock 的情況下跑 CPU 重工作（text-to-sql、agent-gym 各數個 100% CPU 的 process），
  1 分鐘 load 約 8.4；incident-investigator 也長時間持有 heavy lock（process 存活，不是 stale lock）。
  正式 run 會被 008 §2 的 load 門檻擋下（等待最多 10 分鐘後中止），這是預期行為；本專案沒有碰它們。
- 本次沒有跑任何正式 benchmark，也沒有取用 Mac heavy lock 或遠端機器。smoke 輸出全部標記 `load`（因為上述負載），只用來驗證 runner。
- 本 Project 沒有使用 Docker。
- 文字 embedding 在獨立的 `.venv-embed`（torch 2.14.1、transformers 5.19.0、sentence-transformers 6.1.0）產生，與 benchmark 用的 `.venv` 分開；
  模型以匿名方式從 Hugging Face 下載（不使用任何 token），快取在 `~/.cache/huggingface`。

## SPEC / decision 與現況差異

- 004 只規定 M ≥ 2；005 補上 `MAX_M = 2^31 − 1`。
- 004 的「hnswlib 尚未實測」已由 007 回填。
- 007 §3「不產生 recall」已補記：Recall@1 另以 k = 1 search 取得（008 §7）。
- 008 §4「每個 n 都在 run 中重算 ground truth」由 011 §1 修訂：有 frozen ground truth 的 n，驗證後使用；其他 n 照舊計算。
- MACHINE_POLICY 的範例指令改為 `.venv/bin/python -m benchmarks.run`（只改範例，不改規則；`python benchmarks/run.py` 會 import 失敗）。
- 其他沒有發現差異。
