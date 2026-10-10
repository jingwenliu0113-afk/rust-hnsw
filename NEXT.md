# NEXT

## 目前狀態（2026-10-10 18:10 台灣時間 / 10:10 UTC，使用者再次暫停重負載）

時間一律註明時區（台灣時間 = UTC+8）。

- **暫停與恢復**：
  - 2026-10-10 01:50（台灣）使用者第一次暫停重負載；10:10（台灣，02:10 UTC）使用者確認可以恢復。`prereg-gate` 併回 main（ce0317a）、`./scripts/test.sh` 全部通過。
  - 2026-10-10 17:58（台灣，09:58 UTC）使用者要用 Mac，**再次暫停**：以 SIGTERM 中止進行中的 `text-formal-1`，等使用者通知。
    本專案沒有任何程式在跑，也沒有持有 heavy lock。
- **formal run 進度**（協定 008-v2，高效能電力模式、Apple 70W）：
  - 閒置 load 量測 `results/idle_load/20261009T080239Z.json`：`ok`（p95 + 1.0 = 3.35 ≤ 4.0，沒有重工作）。之後的批次以 `--skip-idle` 沿用。
  - **GloVe 8 份全部 complete**：1 `20261009T080239Z-formal`、2 `20261009T091611Z-formal`、3 `20261009T101347Z-formal`、4 `20261009T105616Z-formal`、
    5 `20261010T021933Z-formal`、6 `20261010T025425Z-formal`、7 `20261010T034635Z-formal`、8 `20261010T040529Z-formal`。
    被排除（`--excluded`）：`20261009T113943Z-formal`（GloVe 5 第一次，macOS mediaanalysisd 約 270% CPU → load gate 中止）。
  - **文字**：`text-formal-1` 第一次 `20261010T083544Z-formal` 因使用者暫停被中止（aborted，18 個子 process 跑到第 15 個）→ 整份重跑，報告以 `--excluded` 列出。
    文字 1–4 都還要跑（約 4–5 h），之後是 Gate profiling（約 1–1.5 h）。
  - 恢復時：先 `./scripts/test.sh`（本 commit 之後 binding 早於 HEAD），再
    `./scripts/formal_launcher.sh results/batches/text-2.log 28800 --skip-idle benchmarks/configs/text-formal-1-m16-order.toml … text-formal-4-highm-scaling.toml`。
- **GloVe 正式報告**（`results/reports/glove-100-angular/`，8 個 run，所有 013 §5 的檢查通過）。重點（n = 100,000，Recall@10）：
  - **recall 隨 ef**：三個實作在每個 (n, M, efC) 都沒有系統性惡化，最大相鄰下降 0。Rust 與 hnswlib 在相同 (M, efC, ef) 的 recall 幾乎相同
    （例如 (16, 200) ef 304：0.9568 對 0.9567）。
  - **frontier QPS 讀值**（010 §2）：0.90 / 0.95 / 0.99 時 hnswlib ÷ rust = 1.32 / 1.24 / 1.35，faiss ÷ rust = 2.02 / 2.23 / 1.64。
  - **015 的 Gate 判讀（事先登記）**：主要目標 0.95 為 `within_quantization`（rust 的 QPS 範圍 [2537, 3234]、hnswlib [3135, 4192]，重疊約 3%）；
    0.90 為 `uncertain`（QPS 範圍分開，但交點前後的 seed 範圍沒有分開）；0.99 為 `within_quantization`。依 015，GloVe 上「有差距」**不成立**。
  - **看到結果之後的觀察（不是事先登記的判讀，只供使用者參考）**：variance 格點上，Rust 與 hnswlib 在**同一個 ef** 達到目標、recall 相同到小數第 3–4 位，
    5 組比較（0.90 的 (16, 200) ef 152、(48, 200) ef 64；0.95 的 (16, 200) ef 304、(48, 200) ef 128；0.99 的 (48, 200) ef 384）
    hnswlib 的 seed 範圍都整段高於 Rust，median 比值 1.22–1.38。015 的 QPS 範圍規則刻意保守（容許 Rust 的交點落在更小的 ef），所以沒有判為 `gap`。
  - **build time**（(16, 200)，5 seeds 的 median）：rust 31.4 s、hnswlib 22.9 s、faiss 14.4 s。
  - **index memory（phys_footprint）**：100k 時 rust 約 103–119 MiB、hnswlib 67–92、faiss 51–76；但 300k 時 rust 186、hnswlib 190、faiss 158。
    100k 的 Rust 讀值可能包含 allocator 保留下來的暫時副本（B1 的同一個機制，未驗證），看到結果之後的推測，不寫進結論。
  - **被標記而沒有主要讀值的格點**：1,183,514 的三個 build 全部 `background`（其他專案的 python 間歇 ≥ 50%）、M8 的 Rust build `background`、
    300k 的 Rust ef 512 / 768 有 `load`。這些格點只有「包含被標記點的讀值」（報告另列，010 §3）。
- **decision 015** 與相關程式已在 main（事先登記的時間證據：分支 `prereg-gate` 的 7a8c86a、1fa384b，早於任何正式報告）。

## 目前位置

**Core：除了文字 workload 的正式 run（暫停中）、Gate profiling 與文字報告之外都已完成；GloVe 的正式 run 與報告已完成**：

- 已完成：HNSW insert / search（004）、save / load（005）、Python binding 與 efSearch 介面（006）、
  hnswlib / FAISS adapters（007）、benchmark runner（008，含實作後審查）、比較規則（010）、報告腳本。
- 使用者已決定資料集（2026-10-08）：標準 ANN 用 `glove-100-angular`；真實文字 workload 用公開文字 corpus + `google/embeddinggemma-2`。
- 資料集 decision 011 已寫好；GloVe loader（`benchmarks/prepare_glove.py`）與文字 workload 的流程
  （`text_corpus.py`、`embed_text.py`、`prepare_text.py`）已實作並測試。
- 資料準備程式經過兩個角度的審查（19 個 findings，採用 8 個、駁回 11 個，摘要見 011 文末）。最重要的一項：embedding 改用 batch size 1，
  因為原本的重現性檢查在 batch 32 下必然失敗（padding 造成約 2e-7 的差），第一次文字 pilot 已因此中止。
- 兩個資料集已登記：`glove-100-angular`（重跑後 6 個檔案逐位元相同）與 `simplewiki-embeddinggemma2-pilot`。
- **真實資料上的 correctness 已通過**（SPEC Core 3、Week 1 第 6、7 項；smoke 模式，不作效能結論）：
  - GloVe n = 100k、M = 16、efC = 200、ef 10–256（`results/runs/20261008T142627Z-smoke`）：三者 Recall@10 都單調上升，ef = 10 約 0.55、ef = 256 約 0.95；
    Rust 與 hnswlib 的曲線幾乎重疊（每個 ef 的差最大約 0.0014）。
  - 文字 pilot n = 10k（`results/runs/20261008T143953Z-smoke`）：三者都單調上升，ef = 64 約 0.98、ef = 256 約 0.999。
  - 兩者都沒有結果不足 k、沒有重跑或 batch / single 不一致，frozen ground truth 驗證後使用。圖在各 run 的 `report/`（results/ 不進 Git）。
- decision 012：正式文字 workload 為 `simplewiki-embeddinggemma2-100k`（base 100,000、query 2,000、登記 n = 10k / 30k / 100k）。
- decision 013：正式格點已採用（12 份 config、約 19.2 h），GloVe 的 8 份已 commit。
- 2026-10-08 23:37 前後 session 結束，背景工作收到 SIGTERM：
  - GloVe 成本探測記為 aborted（`results/runs/20261008T144333Z-smoke`）：n = 100k、M = 48、efC = 400 三個實作已完整量完；
    n = 1,183,514 只完成 hnswlib 的 build。後半段以 `glove-cost-probe-full.toml` 重跑。
  - 100k 文字資料集在 embedding 階段中斷（只寫出 jsonl），從頭重跑（embedding 與批次組成無關，重跑結果應逐位元相同於未中斷的版本）。
  - 中斷前機器的 15 分鐘 load 約 69、swap 使用約 2.6 GB（主要是其他專案的模擬器與虛擬機器）。
- **`simplewiki-embeddinggemma2-100k` 已登記**（012 補記）：前 1,000 個 query 與 pilot 逐位元相同；correctness smoke 通過（ef 192 約 0.993）。
  文字 4 份 formal config 已 commit（timeout 依 013 §6 檢查後不需上調）。**12 份 formal config 全部已 commit。**

尚未進 Advanced（Gate 的讀法已在 decision 015 事先登記，但還沒有正式結果可判斷），沒有做任何 Optional。

## 工作方式（使用者 2026-10-08 指示）

- 自主推進 Core：完成一項 → 測試 → 更新 decision log / NEXT → 乾淨的 commit → push → 直接進下一項。
- `origin` = GitHub Private repo（使用者建立）。測試通過的乾淨 commit 可直接 push；禁止 force push、改寫 history、改 visibility。
- （2026-10-09 使用者離開前指示）需要使用者的事項（決策、下載 / 費用批准、API key、credential、sudo、人工標註、修改 frozen protocol、
  進入 Advanced / Optional）**不等待**：記到下方「Blocked / 需要使用者」後跳過，改做其他 Core 工作。heavy lock 被占用、機器負載過高時也一樣。
  不得為了繼續而放寬門檻、改 metric / split / frozen protocol、產生付費用量或使用 credential。只有在所有可做的 Core 工作都完成或全部 blocked 時才停下。

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
- `9c2fe2a fix(datasets): make embeddings independent of batch composition and bind frozen ground truth to the vectors`
- `ead811f fix(datasets): record the preparation environment when preparation starts`
- `433a61f feat(datasets): register glove-100-angular with frozen ground truth`
- `3882eca feat(datasets): register the EmbeddingGemma 2 text pilot`
- `1bf8545 docs: decide the formal text workload scale and add the GloVe cost probe`
- `793edc9 feat(benchmarks): profile registered datasets and record the profiling context`
- `693ba51 chore(benchmarks): rerun the full-size half of the GloVe cost probe`
- `ddf0fd1 docs: record the reusable-page bias in index memory as a blocked decision`
- `04a68d8 feat(benchmarks): add the idle-load baseline measurement required before formal runs`
- `f37ce6c docs: update README with real-data correctness and the new tools`
- `cc900e5 feat(benchmarks): report index footprint next to index memory`
- `2c8543f feat(benchmarks): record power mode, adapter wattage and battery state`
- `8958f33 docs: narrow the deferral to formal, performance-sensitive Mac experiments`
- `746c1d0 docs: pre-register the formal grid (decision 013) and add the GloVe formal configs`
- `fbe7f40 feat(benchmarks): add the pre-merge checks and auxiliary scaling reading required by 013`
- `7580224 docs: add the formal-run runbook to NEXT`
- `89b5f09 fix(benchmarks): close gate-tool gaps found in review`
- `e0529a0 docs: defer the idle-load baseline until the formal power conditions are decided`
- `16c9c4a feat(datasets): register the formal text workload and its formal configs`
- `7dd11a5 docs: summarize what needs the user in NEXT`
- `b73300d feat(benchmarks): apply the user's formal-run conditions and phys_footprint index memory (decision 014)`
- `f077de9 fix(benchmarks): sample background CPU during children and harden the formal-run conditions (014 addendum)`
- 分支 `prereg-gate`（2026-10-10 fast-forward 併回 main）：
  - `7a8c86a docs: pre-register the Gate evidence reading and Gate profiling (decision 015)`
  - `1fa384b fix(benchmarks): read the Gate gap with QPS brackets and check every profiling condition (015 review)`
  - `chore(scripts): add the formal batch and launcher scripts; update NEXT`（本次）

## 最近完成：Gate 證據的事先登記（decision 015，分支 `prereg-gate`）

- 在看任何正式效能結果之前固定：
  - **差距的判讀**：主要目標 Recall@10 0.95、n = 100,000。各實作的 QPS 範圍以「第一次達標的 ef」與「前一個 ef」夾住真正的交點（不內插）；
    範圍分開、而且每個登記的 variance 格點上較快一方最慢的 seed 仍快於較慢一方最快的 seed（各 ≥ 3 個未標記的 build），才判為 `gap` / `rust_faster`。
    兩個資料集的所有組合都事先寫明意義；`within_quantization` 等不代表「很接近 hnswlib」。是否進 Advanced 由使用者決定。
  - **Gate profiling**：(100k, 16, 200) seed 1，各實作在正式 run 中 Recall@10 ≥ 0.95 的最小 ef（任一未達時三者一起改用 0.90），
    search 30 秒、build 300 秒；取樣後核對參數、lock、formal 條件、git、量測程式、BUILD_INFO、版本與 recall（`benchmarks/gate_profile.py`）。
- 初稿（4/3 的比值門檻）經獨立審查：門檻與自己的理由矛盾（梯子有一半是 ×1.5）等 10 個 findings 全部採用，見 015 文末。
- 以 GloVe 1–4 的真正紀錄乾跑新的報告程式，只檢查結構，沒有看任何效能數字。
- `scripts/formal_batch.sh`、`scripts/formal_launcher.sh`：原本放在 session 的 scratchpad，收進 repo，並以假的 python 測試流程。
- 測試：分支上 `test_report.py`、`test_gate_profile.py`、`test_formal_scripts.py`、`test_profile_driver.py` 的相關部分通過；
  mutation 都被抓到。worktree 沒有 `.venv`，`profile.sh` 的兩個測試在 worktree 內無法執行；完整的 `./scripts/test.sh` 留到併回 main 時跑。

## 最近完成：gate 工具的審查（008 / 009 / 010 / 013 補記）

- 三個角度（程式正確性、decision 忠實度、測試鑑別力），findings 由驗證者重現或反駁（tests 角度的驗證者因 session 用量上限中斷，17 個 mutation findings 由主 session 判斷，大多採用）。
- 主要修正：
  - `profile.sh` 等 READY 超過 600 秒時會靜默取樣錯誤的階段 → 沒有上限地等，沒有 READY 就失敗。
  - `idle_load` 會把受污染的量測判成 `exceeds` → 5 分鐘預備期、三條重工作規則、逐樣本紀錄（008 補記）。
  - `report.py` 可以只交比較有利的 run → formal 報告必須恰好包含 013 §5 的每份 config 各一個 run，並掃描沒交出的 formal run（013 補記）；
    合併時比對資料集檔案的 SHA-256 與 ground truth 來源。
  - B1 的措辭改為「決定之前不產生 formal 報告」，footprint 輔助讀值在 010 補記登記。
- 測試：pytest 467 個。

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
   - 工具已完成（2026-10-09）：`./tools/with-mac-lock .venv/bin/python -m benchmarks.idle_load`。每 15 秒取樣、共 10 分鐘，
     輸出 P50 / P95 / 最大值與「P95 + 1.0」對門檻 4.0 的判定；其他 process 平均 ≥ 50% CPU 時判為 `not_idle`（008 §2 的「沒有其他重工作」），
     結果寫到 `results/idle_load/`。只記錄、不改門檻；`exceeds` 時依 008 §2 需要另寫 decision 修訂門檻（屬於修改門檻，記到 Blocked 等使用者）。
   - 2026-10-09 00:50 檢查：本專案的 100k embedding、agent-gym 與另一個 python3 各占約 100–180% CPU，heavy lock 由 llm-serving-lab 持有 → 仍 blocked。
4. **資料集（使用者已決定，decision 011）**：
   - 程式與測試已完成：GloVe 轉換、frozen ground truth（runner 以 SHA-256 加前 100 個 query 重算驗證後使用）、
     manifest 登記、文字 corpus 的前處理與切分（以文章為單位、加鹽 SHA-256 排序、完全相同的文字只留一份）、
     EmbeddingGemma 2 的 embedding（固定 revision、fp32、CPU、8 threads、拒絕 fp16、逐位元重現檢查）。
   - glove-100-angular 已登記（manifest 與 `glove-correctness.toml` 同一個 commit）。以修正後的程式重跑一次，
     train / queries / neighbors 與三個 frozen ground truth 共 6 個檔案的 SHA-256 與第一次完全相同（準備約 466 秒，peak RSS 約 3.1 GB）。
   - pilot 已登記（manifest 與 `text-pilot-correctness.toml` 同一個 commit）：base 10,000、query 1,000、frozen ground truth n = 10,000；
     batch size 1 下 base 約 1,124 秒、query 約 108 秒（機器 load 約 9–12，約 9 筆/秒），token 數最多 386、平均約 98，去重移除 6 段。
   - （已完成）兩個資料集上的 correctness run 與第一張 recall vs efSearch 圖（見「目前位置」）。
   - （已完成）decision 012：正式文字 workload 的規模；`simplewiki-embeddinggemma2-100k` 已產生並登記（012 補記）。
5. **正式格點（decision 013，已採用）**：
   - 以三個獨立設計（成本最小、統計穩健、涵蓋最廣）加三位評審（合規、成本、完整性）定稿；三位評審都以「成本最小」為基礎，
     再移植 M8、GloVe (48, 200) 的 5-seed variance、1,183,514 加 768 / 1218 / 1536、文字 M16 上界 512 等。
   - 12 份 config、38 個 build、負載下上限約 19.2 h（GloVe 11.3 h、文字 7.9 h），每份 0.3–2.6 h。
   - GloVe 的 8 份已 commit；文字的 4 份逐字登記在 013 附錄，與 `simplewiki-embeddinggemma2-100k` 的 manifest 同一個 commit 加入。
   - **第一次 formal run 之前還必須完成**：
     - （已完成，013 補記）`report.py` 的四項檢查：拒絕跨 run 重複的 build、拒絕 aborted / invalid / error 的 run（以 `--excluded` 列出）、
       合併前的版本一致檢查、輔助的 (16, 200)-only scaling 讀值。
     - （已完成）文字 4 份 config 的 commit（與 manifest 同一個 commit）。
     - （已完成）閒置 load 量測（`ok`）、B1、B2、使用者解除 10-09 晚上的限制。
   - **正式 run 的執行手冊**（使用者解除限制、B1 / B2 決定、閒置 load 量測通過之後；每一步都在 repo root）：
     1. 確認電力條件（decision 014 與補記）：自動或高效能電力模式（一份報告只用一種）、Apple 原廠 70W 變壓器、AC；背景沒有其他 heavy workload。
     2. 確認 git 乾淨、文字 4 份 config 已 commit；`./scripts/test.sh`（重新編譯 binding，formal 前置檢查要求 binding 晚於 HEAD）。
     3. 第一次之前：`./tools/with-mac-lock .venv/bin/python -m benchmarks.idle_load`（含 5 分鐘預備期，共約 15 分鐘；008 補記），結果必須是 `ok`。
        `not_idle`：等其他工作結束再量。`exceeds`：先看紀錄中逐樣本的 %CPU 確認沒有漏判，再記到 Blocked 由使用者決定是否修訂門檻。
     **正式 run 進行中不得在 repo 內建立、修改或刪除任何檔案（含 untracked 檔案與 `.claude/`）、不得 commit、不得跑 pytest / 編譯等重工作**：
     每個子 process 前的重新確認會檢查 HEAD 與 `git status`（變了就 invalid），背景 heavy workload 會讓 gate 等待、或讓子 process 被標記 `background`。
     其他 Core 工作在 scratchpad 進行；NEXT.md 等文件的更新留到兩個 run 之間再 commit，之後要重新 `./scripts/test.sh` 讓 binding 晚於 HEAD。
     4. 依 013 §5 的建議順序，一次一份：`./tools/with-mac-lock .venv/bin/python -m benchmarks.run benchmarks/configs/<config>.toml --mode formal`
        （GloVe 1 → 8、文字 1 → 4，可交錯）。lock 拿不到（exit 75）就改天再跑，不得刪 lock。
     5. 某份 config 的 run 若是 aborted / invalid / error：整份重跑，原本的 run 之後以 `--excluded` 列出；incomplete 照實保留、不重跑。
     6. 每個資料集全部 config 跑完、而且 B1 已決定之後，才產生正式報告：
        `.venv/bin/python -m benchmarks.report <該資料集全部 run 目錄> --out results/reports/<資料集> [--excluded <失敗的 run>]`。
        報告會拒絕：缺少或重複的 formal config、同一份 config 另外有沒交出的 complete / incomplete run、沒以 --excluded 列出的失敗 run（013 補記）。
   - **恢復正式 run 的步驟**（使用者解除暫停之後；log 一律寫在 `results/batches/`，不得寫在 repo 的 tracked 路徑）：
     1. main 沒有新的 commit 時：`git merge --ff-only prereg-gate`；`./scripts/test.sh` 全部通過（重新編譯 binding）；push main。
     2. `mkdir -p results/batches && ./scripts/formal_launcher.sh results/batches/glove-2.log 28800 --skip-idle`
        `benchmarks/configs/glove-formal-5-m48-seed-5-efc100.toml … glove-formal-8-full.toml`（背景執行；LAUNCHED 之後看 log 的 RUN_END / STOP / ALL_DONE）。
     3. GloVe 的 ALL_DONE 之後：同樣以 `--skip-idle` 跑 `text-formal-1 … 4`。任一 STOP：先看 run.json 的 status_reasons，再整份重跑該 config。
     4. GloVe 報告：`.venv/bin/python -m benchmarks.report <GloVe 8 個 run> --out results/reports/glove-100-angular --excluded results/runs/20261009T113943Z-formal`
        （加上之後其他失敗的 run）；文字同理。
6. **Gate 用的 profiling**（009：真實資料、lock 下、相近 recall；參數與核對見 decision 015 §2）。
   - 兩個資料集的全部 formal config 完成之後：`gate_profile plan <run 目錄> > results/profiles/gate-<資料集>-plan.sh`、
     `./tools/with-mac-lock bash results/profiles/gate-<資料集>-plan.sh`、`gate_profile check <run 目錄>`（全部 ok 才作為證據）。
   - Gate 的判讀（報告的「Gate 讀值」一節）與 profiling 的 top-of-stack 摘要寫進 NEXT，交給使用者決定是否進 Advanced。
   - 工具已補齊（009 補記）：`profile.py --dataset <名稱> --n <n>` 與取樣前寫出的 JSON（git、BUILD_INFO、lock 狀態、Recall@10）。
     已以文字 pilot 端到端驗證（無 lock，只驗證工具，不作證據）。
   - 待正式 run 的結果出來後，依相近的 Recall@10 為各實作選 ef，在 heavy lock 下取樣。

## Deferred（已解除）

- 2026-10-10 01:50 前後：使用者要先跑重要工作，暫停本專案所有重負載（formal run、Gate profiling、`./scripts/test.sh` 的編譯）；04:00 前後使用者確認可以恢復。

- 2026-10-09 01:00 使用者暫停正式、performance-sensitive 的 Mac 實驗（自動電力模式、68W）；09:30 解除，並以 decision 014 固定正式 run 的條件。

## Blocked / 需要使用者決定

### 目前需要使用者

- **解除重負載的暫停**（2026-10-10 17:58 台灣時間）：解除後先 `./scripts/test.sh`，再重跑文字 1–4（見「目前狀態」）。
- **GloVe 1,183,514 這一格沒有主要讀值**：三個實作的 build 都被標記 `background`（其他專案的 python 沒有取 lock 就跑）。010 §3 不允許重跑 complete 的 run。
  是否在看過結果之後補跑 `glove-formal-8`（需另寫 decision，標明是看過結果之後、只因 hygiene 標記），由使用者決定；預設照實報告「只有含標記點的讀值」。
- **Gate 的判讀**：GloVe 依 015 為 `within_quantization`（不成立），但同 ef、同 recall 的比較一致顯示 hnswlib 快 1.22–1.38 倍（看到結果之後的觀察）。
  文字的正式報告與 Gate profiling 完成後，一起交給使用者決定是否進 Advanced。

### B1.（已決定，decision 014）index memory 的量測定義：phys_footprint 為主、rss 為輔

- **問題**：008 §8 規定 `index_memory_bytes` = build 後的 rss − baseline rss，recall–memory frontier 一律用它。
  macOS 上，釋放的大型配置會以 reusable page 留在 rss 裡（不算 phys_footprint），直到系統有記憶體壓力才回收；正式 run 要求機器閒置，這些 page 最不會被回收。
- **證據**（`scripts/probe_reusable_memory.py`，2026-10-09 兩次實測）：釋放 451 MiB 的暫時副本後 Δrss ≈ +451–470 MiB、Δfootprint ≈ 0–14 MiB；
  呼叫 `malloc_zone_pressure_relief` 後 Δrss 回到約 0–19 MiB。numpy 的對照組結果相同，所以是 allocator 的普遍行為，不是 Rust 特有。
- **對結果的影響**：Rust 的 `HnswIndex.add` 為了「任一列不合法就整批不插入」先建立一份完整的正規化副本（`unit_rows`），add 結束時釋放。
  完整 GloVe（1,183,514）的 smoke 探測中，Rust 的 rss 增量 1,186 MiB、footprint 增量 738 MiB，差 448 MiB ≈ 一份資料；n = 100k 時沒有出現差距。
  照現行定義，Rust 在大 n 的 recall–memory frontier 會被多算約一份資料。現有的 `memory` 標記只抓 rss **低估**（壓縮），抓不到這種**高估**。
- **選項**：
  1. （建議）`index_memory_bytes` 改用 phys_footprint 的增量（含被壓縮的 page、不含 reusable page）；rss 增量保留為輔助欄位。
  2. 量 baseline 與 after_build 之前先呼叫 `malloc_zone_pressure_relief`，繼續用 rss（會改變被量測時 allocator 的狀態，但對三者對稱）。
  3. 維持現狀，在報告中註明 Rust 在大 n 時被多算。
- **為什麼不擋正式 run**：run.json 本來就同時記錄 rss（`index_memory_bytes`）與 footprint（`index_footprint_bytes`），
  這個決定只影響報告的 memory frontier 採用哪個欄位；但必須在看到正式結果**之前**決定，否則會變成看結果選指標。
  **B1 決定之前，不產生、不查看任何 formal 報告**（formal run 可以先跑；010 補記）。
  報告已加上 footprint 的輔助欄位（frontier 與 scaling；主要讀值仍是 rss，未改 008 §8），已在 010 補記登記；B1 決定後依決定保留或移除。
- **另外的觀察（不是量測問題）**：同一份暫時副本讓 Rust build 期間的 peak RSS 多出一份資料（真實的實作特性，照實報告）；可能是 Advanced A 的假設之一，Core 不改。

### B2.（已決定，decision 014）正式 run 的電力條件：自動電力模式、原廠 70W AC、電池下降 ≥ 3 點作廢

- 008 的 formal 前置檢查只確認「接的是 AC 電源」（並在執行中標記 `power`）。但電力模式與供電也會改變 CPU 效能：
  2026-10-08 23:37 的 run 紀錄是「高效能」（powermode 2）、電池 92%；2026-10-09 01:00 為「自動」（powermode 0）、68W 變壓器、電池 56% 充電中。
- **建議**：在第一次 formal run 之前登記固定的電力條件，formal 前置檢查與每個子 process 前都確認，例如：
  powermode = 2（高效能）、變壓器瓦數 ≥ 某個值（由你指定可用的變壓器）、電池不在充電中（或電量 ≥ 某個值）；不符就拒絕 / 標記 `power`。
- 目前先只**記錄**（不改任何規則）：metadata 另外記下 `pmset -g ac`（變壓器瓦數）與解析後的 powermode。
- 這一項與 Deferred 一起處理：解除限制時，請一併決定正式 run 要用的電力條件。

### B3.（已解決，使用者處理）系統的 Xcode 在 2026-10-09 01:05 被更新後，`/usr/bin/git` 與 `cc` 無法執行

- `/Applications/Xcode.app` 在 01:05–01:06 被更換（不是本專案做的）。新的 `libxcrun.dylib` 是 arm64、系統的 `xcrun` 需要 arm64e，
  所以經過 xcrun 的 `/usr/bin/git`、`/usr/bin/cc`（cargo 連結也會用到）都報 `unable to load libxcrun`。
- 繞道（不改系統設定）：本專案的指令一律加 `export DEVELOPER_DIR=/Library/Developer/CommandLineTools`，改用已安裝的 Command Line Tools
  （git 2.54.0、Apple clang 21.0.0）。pytest 423 個在此設定下全部通過。
- 修正系統需要 sudo（例如 `sudo xcode-select -s /Library/Developer/CommandLineTools`，或修復 / 重新安裝 Xcode），由你決定，我沒有動。
- 對正式 run 的影響：formal 前的 `./scripts/test.sh` 與 runner 內的 git 檢查也要在同樣的環境變數下執行；連結器版本不影響 rustc 的程式碼產生。

## 需要留意

- **機器負載**：本 session 多次觀察到其他 Project 在不取 lock 的情況下跑 CPU 重工作（text-to-sql、agent-gym 各數個 100% CPU 的 process），
  1 分鐘 load 約 8.4；incident-investigator 也長時間持有 heavy lock（process 存活，不是 stale lock）。
  正式 run 會被 008 §2 的 load 門檻擋下（等待最多 10 分鐘後中止），這是預期行為；本專案沒有碰它們。
- 正式 run 自 2026-10-09 起在 Mac heavy lock 下進行（見「目前狀態」）；沒有使用遠端機器。之前的 smoke 輸出多半標記 `load`（因為上述負載），只用來驗證 runner 與 correctness。
- macOS 自己的背景工作（mediaanalysisd 的照片分析，約 270% CPU）也會觸發 load gate 而中止 formal run；關掉它屬於系統設定，本專案不動，遇到時照 013 §5 整份重跑。
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
