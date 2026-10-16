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

## 補記（2026-10-08，在任何 Gate 證據之前）

原本的工作負載只能讀任意 `.npy`（會重新正規化、不能取子集），也沒有記錄 git commit、參數與 recall，不足以落實上面「作為 Gate 證據」的三個條件。補上：

- `benchmarks.profile --dataset <名稱> --n <n>`：與 runner 相同的登記向量（不重新正規化）與前 n 列子集（008 §4），ground truth 也用相同規則
  （有 frozen 檔就驗證後使用，否則以 ExactIndex 計算）。
- 取樣前寫出 `results/profiles/<名稱>.json`（`scripts/profile.sh` 自動傳入 `--meta`）：參數、資料集檔案的 SHA-256、ground truth 來源、
  git commit 與 status、`rust_hnsw.BUILD_INFO`、套件版本、是否由祖先 process 持有 heavy lock、load average、機器資訊。
- search 階段另以一次不計時的 k = 10 search 計算 Recall@10（id-based，003），記在同一個 JSON。
  「相近的 Recall@K」以這個值判斷：各實作的 ef 由正式 run 的結果選出，取樣時再確認一次。
- 仍然不輸出任何 latency / QPS 數字。

## 補記（2026-10-09，gate 工具的審查）

- `scripts/profile.sh` 原本最多等 600 秒的 READY，逾時會**靜默地**對 READY 之前的階段（資料載入、frozen GT 驗證、build）取樣並以 0 結束。
  完整 GloVe 的 build 單一實作就約 600 秒，所以這條路徑在 Gate 證據的實際使用中走得到。改為沒有上限地等到 READY；
  workload 沒有印出 READY 就結束時，以非 0 結束，絕不取樣。
- （014 補記）作為 Gate 證據的取樣（`--dataset` 且祖先持有 heavy lock）在印出 READY 之前檢查 formal run 的電力條件，
  並等待 load 與背景 heavy workload 的 gate（最多 10 分鐘）；結果寫進取樣前的 JSON（`evidence_conditions`）。

## 補記（2026-10-09）：Gate profiling 的參數

- 格點、目標 recall、ef 的選法、取樣時間與取樣後的 recall 核對，在看任何正式效能結果之前由 decision 015 §2 固定（`benchmarks/gate_profile.py`）。

## 補記（2026-10-16，Gate profiling 之後）：沒有符號的列與 FAISS 的歸屬

- `profile_report.py` 原本只解析「`symbol (in image) count`」的列，沒有符號的列（`??? (in libfaiss.dylib) load address … + 0xOFF [0x…] count`、
  `??? (in <unknown binary>) [0x…] count`）被**默默略過**，而且比例只以解析到的列為分母。結果：FAISS GloVe search 的排行是空的，
  文字 search 的排行把 18 個 malloc 樣本當成 100%（實際總數約 2.2 萬）。Rust 與 hnswlib 的 build 也各有少量位址列被略過。
- 修正：沒有符號的列照樣保留（以映像檔內的位移或位址表示），任何無法解析的列一律報錯；`gate_profile check` 另外要求排行不得為空。
  12 份排行以修正後的程式從**原始取樣檔**重新產生（取樣本身沒有問題，不重新取樣）。
- pip 安裝的 `faiss-cpu`（`libfaiss.dylib`）沒有符號表。FAISS 的歸屬改以 `nm -n` 的 text 符號近似：每個位移歸給它**之前最近的符號**。
  這是看過取樣之後才加的、近似的解讀：被 inline 或沒有匯出的函式會算到前一個符號，只用來看大類，不作精確結論。
- SPEC 的 Gate 以 hnswlib 為準；Rust 與 hnswlib 都有完整的符號。

## 補記（2026-10-17，Advanced C 的實作之後、任何真實資料上的量化結果之前）：C 的取樣

decision 019 §8 的 profiling 沿用本文件的工具與 015 §2 的證據條件。不同之處：

- **函式邊界**：新臂 F、S 以 `#[inline(never)]` 拆出兩個距離 kernel（`rhnsw_kernel_fp32`、`rhnsw_kernel_sq8`）。
  - 三個走訪函式也是 `#[inline(never)]`：否則會被整個 inline 進 `QuantIndex::search_batch`，走訪會落到「其他」。
  - 這只改函式邊界，不改算術；binding 中有獨立符號，由 `nm` 測試保證。
- **分類規則**：在取樣之前固定（019 §8），實作於 `benchmarks/quant_profile.py`，並以測試逐條釘住。
- **FAISS 不取樣**：沒有符號表，依上一則補記只能近似歸屬。
- **R 的機器碼**：C 的 binary 中，Core 的 `search_layer` 把 visited set 的 `HashMap::insert` inline 進來了，Gate profiling 的 binary 則沒有。
  - 原始碼沒有改；這是加入 quant 模組後 codegen 的改變（019 的補記第 5 節）。
  - 所以 C 中 R 的類別占比與 Gate profiling 的排行不能直接對照：C5 預期會不一致，原因已知。
