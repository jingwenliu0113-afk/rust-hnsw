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
- 進行中：產生 100k 文字資料集（背景，約 3 小時）；GloVe 完整大小的成本探測（smoke），作為正式格點 decision 的依據。

尚未進 Advanced（Gate 條件都還沒有資料可判斷），沒有做任何 Optional。

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
- `feat(benchmarks): add the pre-merge checks and auxiliary scaling reading required by 013`（本次）

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
   - （已完成）decision 012：正式文字 workload 的規模。進行中：產生 `simplewiki-embeddinggemma2-100k`，完成後與它的 manifest 一起 commit。
5. **正式格點（decision 013，已採用）**：
   - 以三個獨立設計（成本最小、統計穩健、涵蓋最廣）加三位評審（合規、成本、完整性）定稿；三位評審都以「成本最小」為基礎，
     再移植 M8、GloVe (48, 200) 的 5-seed variance、1,183,514 加 768 / 1218 / 1536、文字 M16 上界 512 等。
   - 12 份 config、38 個 build、負載下上限約 19.2 h（GloVe 11.3 h、文字 7.9 h），每份 0.3–2.6 h。
   - GloVe 的 8 份已 commit；文字的 4 份逐字登記在 013 附錄，與 `simplewiki-embeddinggemma2-100k` 的 manifest 同一個 commit 加入。
   - **第一次 formal run 之前還必須完成**：
     - （已完成，013 補記）`report.py` 的四項檢查：拒絕跨 run 重複的 build、拒絕 aborted / invalid / error 的 run（以 `--excluded` 列出）、
       合併前的版本一致檢查、輔助的 (16, 200)-only scaling 讀值。
     - （可以自己做）文字 4 份 config 的 commit（等 manifest）。
     - （Deferred / Blocked）閒置 load 量測、B1、B2、使用者解除今晚的限制。
6. **Gate 用的 profiling**（009：真實資料、lock 下、相近 recall）。
   - 工具已補齊（009 補記）：`profile.py --dataset <名稱> --n <n>` 與取樣前寫出的 JSON（git、BUILD_INFO、lock 狀態、Recall@10）。
     已以文字 pilot 端到端驗證（無 lock，只驗證工具，不作證據）。
   - 待正式 run 的結果出來後，依相近的 Recall@10 為各實作選 ef，在 heavy lock 下取樣。

## Deferred（使用者 2026-10-09 約 01:00 指示並更正範圍，等使用者明確解除）

- 今晚 Mac 使用「自動」電力模式（pmset powermode 0）、只有 68W 供電（70W USB-C 變壓器，電池充電中）：
  **只暫停正式、performance-sensitive 的 Mac 實驗**；其他 heavy 工作（資料準備、embedding、smoke 探測、測試、開發）照常，commit / push 照常。
- 因此 deferred（不等待，改做其他 Core 工作）：
  - 所有 formal run（GloVe 與文字的 formal config 會照常寫好並 commit，但不執行）。
  - lock 下作為 Gate 證據的 profiling。
- 閒置 load 量測（`benchmarks.idle_load`）本身不是效能量測，但它是 formal run 的前置條件，而且需要 heavy lock 空出來、沒有其他重工作：
  目前 lock 由 llm-serving-lab 持有、本專案的 embedding 也在跑，所以照第 3 項的狀態繼續 blocked；條件具備時可以量（結果另外標註當時的電力狀態）。

## Blocked / 需要使用者決定

### B1. index memory 的量測定義（修改凍結的 008 §8，需要你決定；不擋正式 run）

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
  若正式 run 在你決定之前完成：報告的 memory 讀值兩種都列（rss 與 footprint），選擇的依據只能是上面這份與正式結果無關的證據。
  報告已加上 footprint 的輔助欄位（frontier 與 scaling；主要讀值仍是 rss，未改 008 §8）。
- **另外的觀察（不是量測問題）**：同一份暫時副本讓 Rust build 期間的 peak RSS 多出一份資料（真實的實作特性，照實報告）；可能是 Advanced A 的假設之一，Core 不改。

### B2. 正式 run 的電力條件（新增前置條件屬於修改 008，需要你決定）

- 008 的 formal 前置檢查只確認「接的是 AC 電源」（並在執行中標記 `power`）。但電力模式與供電也會改變 CPU 效能：
  2026-10-08 23:37 的 run 紀錄是「高效能」（powermode 2）、電池 92%；2026-10-09 01:00 為「自動」（powermode 0）、68W 變壓器、電池 56% 充電中。
- **建議**：在第一次 formal run 之前登記固定的電力條件，formal 前置檢查與每個子 process 前都確認，例如：
  powermode = 2（高效能）、變壓器瓦數 ≥ 某個值（由你指定可用的變壓器）、電池不在充電中（或電量 ≥ 某個值）；不符就拒絕 / 標記 `power`。
- 目前先只**記錄**（不改任何規則）：metadata 另外記下 `pmset -g ac`（變壓器瓦數）與解析後的 powermode。
- 這一項與 Deferred 一起處理：解除限制時，請一併決定正式 run 要用的電力條件。

### B3. 系統的 Xcode 在 2026-10-09 01:05 被更新後，`/usr/bin/git` 與 `cc` 無法執行（需要你處理系統設定；本專案有繞道，不擋工作）

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
