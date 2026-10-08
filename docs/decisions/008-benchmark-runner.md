# 008：benchmark runner 與量測協定的落實

日期：2026-10-08
狀態：採用（草稿經一輪獨立審查修訂，見文末；在看到任何正式結果之前定稿）

## 背景

002 定了計時協定的原則（QPS 用整批呼叫、P50 / P95 用 batch size 1、單執行緒、warm-up、重跑）。
006 / 007 補了 efSearch 介面、各 adapter 的語意，以及「runner 在計時前檢查 ef ≥ k」「每個 adapter 分別量空呼叫成本」。
SPEC 的指標：Recall@1、Recall@10、P50、P95、QPS、build time、peak RSS、index size、scaling with N、seed / insertion order variance。

runner 把這些落實成程式，只產生**每個量測點的原始數字**。
怎麼從原始數字得出比較結論（目標 recall、取點、彙整 variance、scaling）由 010 事先登記。

這裡的每個數字（重跑次數、warm-up、指標定義、門檻）都在看到任何正式結果之前固定；
之後改動任何一項都必須另寫 decision log，並說明是在看到結果之前還是之後。

## 決策

### 1. 執行模式與前置檢查

- **formal**：產生可以進報告的數字。開始前全部檢查通過才執行，任一項不符就拒絕：
  1. **lock**：讀 `/tmp/ai-portfolio-mac-heavy.lock/owner.txt` 的 `pid=` 行（缺少或不是整數就拒絕）。
     該 pid 必須是 runner 自己或任一祖先 process（`with-mac-lock` 以子 process 執行命令；接受「自己」是為了容許日後改成 `exec`）。
  2. **git 乾淨**：`git status --porcelain=v1 --untracked-files=all` 輸出為空（未被 ignore 的 untracked 檔也算 dirty）。
  3. **interpreter**：`sys.prefix` 是 repo 的 `.venv`。
  4. **版本**：hnswlib、faiss-cpu、numpy、maturin 的已安裝版本（`importlib.metadata.version`）與 `requirements-dev.txt` 的 pin 完全相同（007 的語意是在這些版本上實測的）。
  5. **Rust binding 的編譯資訊**（見 §10、§11）：`opt_level == "3"`、`debug_assertions` 為 false；
     rustflags 除了 maturin 為 macOS 擴充模組加上的 `-C link-arg=…` 之外為空；沒有以 `CARGO_PROFILE_RELEASE_*` 環境變數覆寫 profile。
  6. **binding 的來源**：`rust_hnsw` 擴充模組的 mtime 晚於 HEAD 的 commit 時間，**也晚於每個 tracked 的 Rust / binding 原始碼檔**
     （`crates/`、`python/`、`Cargo.toml`、`Cargo.lock`）。git 的 checkout、stash、還原都會更新檔案 mtime，
     所以「改了原始碼、編譯、再還原成乾淨的 HEAD」會被擋下。已知限制：從另一份 checkout 編譯、保留 mtime 複製進同一個 `.venv` 的 binary 偵測不到。
  7. **資料集**：必須是資料集 decision 已登記的真實資料集，檔案 SHA-256 相符。synthetic 資料一律拒絕（SPEC §5）。
  8. **config 已 commit**：config 檔位於 `benchmarks/configs/`，而且內容與 HEAD 中的版本完全相同（repo 外或被 ignore 的路徑都拒絕）。
     run.json 記錄 config 的 repo 相對路徑與 blob SHA。
  9. **電源**：使用 AC 電源（本機是筆電；電池供電或低電量模式會改變 CPU 頻率）。
- **formal 的固定流程**：在乾淨的 HEAD 上先跑 `./scripts/test.sh`（含 `maturin develop --release`）並全部通過，
  等 load 降回門檻以下，再執行：

  ```bash
  ./tools/with-mac-lock .venv/bin/python -m benchmarks.run <config> --mode formal
  ```

  runner 自己不編譯：編譯會推高 load，與 §2 的門檻衝突。
- **formal 執行期間的重新確認**：每個子 process 啟動前（**在 load 等待通過之後**）與最後一個子 process 結束後，
  重讀 owner.txt（內容必須與開始時相同）、確認 HEAD 沒變、工作目錄仍乾淨、三個擴充模組（`rust_hnsw`、hnswlib、faiss）的 SHA-256 都沒變。
  任一項不符（包括讀檔失敗，例如擴充模組重新安裝期間短暫不存在）就停止，run 標記為 `invalid`。
  （formal run 可能長達數小時，其他 session 依規定可以在不取 lock 的情況下改檔或重新編譯；load 等待期間正是最可能發生的時候。）
- **smoke**：只用來驗證 runner 本身。不檢查也不需要 lock，前置條件只記錄不強制；輸出標記 `"mode": "smoke"`，不得用於報告。
  smoke 與 formal 走**同一個量測函式、同一組協定常數**，只差在前置檢查與 config。

### 2. 機器負載（協定常數，不放在 config）

- 門檻：1 分鐘 load average ≤ **4.0**（暫定）。
  - 本機為 8 個 P core + 4 個 E core；runner 自己的前一個子 process 會在 1 分鐘 load 中殘留約 1.0。
  - 定稿時無法量到閒置基準：當下有其他 Project 持有 heavy lock，另有多個不需 lock 的 CPU 重工作在跑（load 約 8.4）。
    **第一次 formal run 之前**，必須在沒有 lock holder、沒有其他重工作時量 10 分鐘的閒置 load；
    若「閒置 load 的 P95 + 1.0」> 4.0，先以新的 decision log 修訂門檻，再跑 formal。
- 每個子 process 啟動前檢查：超過門檻就等待（每 15 秒檢查一次），最多 10 分鐘；仍超過就中止整個 run（`aborted`），不降低門檻繼續。
- 每個子 process 前後、build 前後與每個 ef 點前後都記錄 load average；結束時 1 分鐘 load 超過門檻的點（與 build）標記 `load`（見 §9）。
- smoke 只記錄，不等待、不中止。

### 3. config

TOML 檔（`tomllib`），放在 `benchmarks/configs/`。formal 用的 config 必須在 run 之前 commit（git 乾淨的檢查會確保這點）。

```toml
schema_version = 1
dataset = "<資料集登記名稱>"
implementations = ["rust-hnsw", "hnswlib", "faiss-hnsw"]
ef_search = [10, 20, 40, 80]
child_timeout_s = 3600

[[sweep]]
n = [100000]
m = [16]
ef_construction = [200]
seeds = [1, 2, 3]
shuffles = [101, 102]
```

- 每個 `[[sweep]]` 對 `n × m × ef_construction` 展開，每組產生以下 build（**不做 seed × order 的全交叉**）：
  - seed variance：`seeds` 中的每個 seed，搭配原始順序。
  - order variance：`seeds[0]` 搭配 `shuffles` 中的每個 permutation seed。
- 各 sweep 展開後的 build（n、M、efC、seed、order）不得重複。
- `ef_search` 與 `implementations` 全 run 共用；runner 把 `ef_search` 由小到大排序。
- **驗證（計時前，不符就拒絕）**：
  - 只允許上面列出的 key，其他一律拒絕（包括 `k`：k 是協定常數；也擋掉拼字錯誤）。
  - 整數一律以 `type(v) is int` 檢查（TOML 的 `true` 不得當成 1）。
  - 每個清單非空、沒有重複（`shuffles` 可以是空的）。
  - `2 ≤ m ≤ 2^31 − 1`（005）；`ef_construction ≥ m`；每個 `ef_search ≥ k`（006）；`k ≤ n ≤` train 筆數；
    seed 與 shuffle 在 `[0, 2^63 − 1]`；`child_timeout_s ≥ 1`；`schema_version == 1`。
  - `implementations` 只能是 `rust-hnsw`、`hnswlib`、`faiss-hnsw`。
  - query 數 ≥ 100（§6 的 warm-up 常數固定為 100，不隨資料改變）。
- **`ef_construction ≥ m` 的理由**：實測 Rust 與 hnswlib 實際使用 `max(efC, M)`，FAISS 沒有這條規則；
  efC < M 時三者的建構語意不同（`tests/test_third_party_semantics.py` 釘住）。與 `ef ≥ k` 同理，在計時前擋下。

### 4. 資料子集、插入順序與 ground truth

- **子集**：大小為 n 的子集一律取 prepared train 的**前 n 列**，所有實作、seed、order 共用；不同 n 的子集因此是巢狀的。
  抽樣規則（例如以固定 seed 重排後存檔，讓「前 n 列」等於隨機子集）由該資料集的 decision 在準備資料檔時落實，runner 不自行決定。
- **所有 n 使用同一組 queries。**
- **插入順序**：原始順序就是子集的列順序。shuffle 以 `np.random.default_rng(shuffle_seed).permutation(n)` 產生 `perm`，
  父 process 把 `subset[perm]` 寫成獨立的 `.npy`，子 process 只載入打亂後的那一份（不會多一份資料進入 peak RSS）。
- **id 映射**：父 process 在計時外以 `np.where(ids >= 0, perm[ids], -1)` 映射回子集的列號，再算 recall。
  不能寫成 `perm[ids]`：numpy 會把 `-1` 當成最後一個元素，padding 就變成合法 id。
- FAISS 的實際插入順序本來就不是 row 順序（007 §5），所以它的 order variance 意義不同，依 010 分開呈現。
- **ground truth**：每個 n 都在該子集上由 `ExactIndex` 重新計算 k = 10（001 的 f64 與 tie-break）。
  所有 ground truth 都在第一個子 process 啟動前算完，ExactIndex 隨即釋放；子 process 執行期間父 process 只阻塞等待。
- 資料集自帶的 ground truth 只在「資料集宣告 metric 為 angular / cosine、n 等於完整 train、同一組 queries」時比對一致率；
  其他情況記為 `not_applicable` 並附原因。只作 sanity check，不作正式 ground truth。
- **資料集檢查**（003、004）：父 process 對每個 n 的子集，在 `prepare_vectors` 之後以 f32 bit pattern 計算
  「至少與另一列完全相同的列比例」，並以 f64 計算「exact 第 1 名與第 2 名距離差 ≤ `TIE_EPS` 的 query 數」（rank 1 有 tie）。
  兩者與 `TIE_EPS` 一起寫入 run.json。

### 5. process 隔離與執行順序

- 每個（build、實作）在**獨立的子 process** 中量測：`subprocess.run([sys.executable, "-m", "benchmarks.run_child", spec])`，cwd 為 repo root。
  - 全新的 interpreter（不用 fork），不繼承父 process 已 import 的 library。
  - 所有子 process 使用同一份環境變數，不額外設定執行緒相關的變數（adapter 本身負責單執行緒，007 §6；強制設定反而會掩蓋 adapter 的失效）。
    實際傳給子 process 的環境中，`OMP_*`、`KMP_*`、`RUSTFLAGS`、`PYTHON*`、`VIRTUAL_ENV`、`DYLD_*`、`Malloc*`、`MALLOC_*`、`VECLIB_*`、`OPENBLAS_*`
    寫入 run.json（會影響執行緒、allocator 或動態連結）；名稱含 TOKEN、SECRET、PASSW、CREDENTIAL、AUTH、`_KEY` 的不記錄，其他變數也不記錄。
    不記錄 `CARGO_*`：不影響已編好的子 process，而且可能含有 registry token。
  - 結果不從 stdout 解析：每完成一個 ef 點，子 process 就在父 process 指定的 JSONL 檔 append 一行並 flush，原始陣列另存 `.npz`。
    stdout / stderr 導到該子 process 的 log 檔。
- **執行順序**：外層迴圈為 build（依 config 展開的順序），內層為實作；第 g 個 build 把實作清單輪換 g mod（實作數）位。
  時間漂移（熱狀態、背景負載）因此不會與「實作」重疊。完整順序寫入 run.json。
- **失敗**：`subprocess.run(..., timeout=child_timeout_s)`。
  - 分類：returncode > 0 → `exit_nonzero`；< 0 → `signal`；逾時 → `timeout`（子 process 被 kill）；
    returncode 為 0 但沒有 end 紀錄 → `missing_output`；中止時正在執行 → `aborted`。
  - crash 或中止前已完成的 ef 點保留（id 照常映射），該子 process 標記 `partial`；未映射的原始檔一律刪除。不自動重跑。
  - 任何子 process 失敗時，run 狀態為 `incomplete`，runner 以非零碼結束，但仍寫出完整的 run.json。
  - 收到 SIGINT / SIGTERM / SIGHUP 時 kill 子 process，寫出 `aborted` 狀態。
  - 收尾（寫出 run.json）期間暫時忽略這三個訊號，寫完才還原；run.json 以暫存檔加 rename 原子寫入；
    收尾時的紀錄（結束時的 snapshot、檔案 SHA-256）失敗只記錄錯誤，不影響寫出 run.json。
  - `IncompleteResultsError` 是量測結果，不是失敗（§6）。

### 6. 子 process 內的量測流程

1. **import**：只 import 受測的那一個 library（`rust_hnsw` / `hnswlib` / `faiss`），然後才載入 data 與 queries（完整載入記憶體）。
2. **Python 迴圈成本**（只作參考）：以與 latency 相同的計時方式呼叫空函式 1000 次，記錄 P50 / P95；
   另以 10^6 次呼叫的總時間除以次數記錄平均 ns。`time.get_clock_info("perf_counter")` 寫入 metadata（本機刻度約 41.7 ns）。
3. **library warm-up**（不計時）：以同一個 adapter 類別、同一組 M / efC / seed，用 data 的前 16 列建 index，做一次 k = 1、ef = 1 的 search，然後釋放。
   讓 timed build 不含 import 與一次性初始化。
4. **空呼叫成本**（006，只作參考，不從 P50 / P95 扣除）：
   - 以 **n = 1**（data 的第 1 列）建 index，維度等於資料集維度，k = 1、ef = 1。
   - search 只剩 1 次 d 維距離計算，不含 graph traversal；量到的是每次呼叫都要付的固定成本與 O(d) 的輸入處理、後處理。
   - 呼叫路徑與 latency 相同（計時外預先建立的 1×d view，計時只包住 `adapter.search`）。100 次 warm-up 後取 1000 個樣本，斷言結果沒有 `-1`。
5. 釋放上面的 index，`gc.collect()`，斷言受測 library 已在 `sys.modules`、另外兩個不在，記錄 **baseline**（`ps -o rss=` 與 `ru_maxrss`）。
6. **build**：`adapter.build(data)`，以 `perf_counter_ns` 計時 → **build time**（不含 import 與一次性初始化）。
   build 一返回就記錄 `ru_maxrss` → **peak RSS**，以及 `ps -o rss=` → after_build。
7. **對每個 `ef_search`（由小到大）**：
   1. `set_ef_search(ef)`（不計時），記錄 load average 與時間戳。
   2. **warm-up**：k = 10 整批 search 1 次（不計時）。
   3. 先 `gc.collect()` 再 `gc.disable()`，涵蓋第 4、5 步，結束後才 `gc.enable()`。
   4. **QPS**：k = 10 整批 search 重跑 5 次，每次以 `perf_counter_ns` 計時；QPS = nq / 秒數。
      第 1 次的 ids 存檔供 Recall@10；另比對 warm-up 與 5 次的 ids，記錄不一致的 query 數（單執行緒下預期為 0）。
   5. **latency**：計時外先建立 nq 個 1×d 的 view（切自 C-contiguous array，本身就是 C-contiguous）。
      前 100 個 query 做一次 warm-up；接著重跑 3 次，每次依資料集順序走過全部 query，每筆只計時 `adapter.search(view, 10)`。
      第 1 次重跑的逐筆 ids 在計時之後存下，與整批結果逐列比對，記錄不一致的 query 數（預期為 0）。
   6. **Recall@1 的來源**：同一個 ef 再做 1 次**不計時**的 k = 1 整批 search（見 §7）。
   7. 記錄 build、QPS 區段、latency 區段各自「其他 thread 的 CPU time」（`process_time` 增量減 `thread_time` 增量，007 §6）。
      超過 2% + 1 ms 的區段標記 `hygiene: multithread`。
   8. **`IncompleteResultsError`**（hnswlib，007 §3）：k = 10 的 warm-up 或任何計時呼叫拋出時，該點的 Recall@10、QPS、P50 / P95 都標記為 incomplete，
      不保留任何已量到的計時，不重試、不 fallback；仍然做第 6 步的 k = 1 search（它有自己的 incomplete 標記）。
8. 所有 ef 點結束後記錄 `ru_maxrss` → peak_after_search（只作參考）。
9. **原生序列化大小**（最後才做，避免存檔的 I/O 落在計時區間；Rust 的 `save_to_path` 會 fsync，另外兩者不會）：
   各 adapter 的 `save_native(path)`（Rust `save`、hnswlib `save_index`、FAISS `write_index`）寫到 repo 外的暫存目錄，取檔案大小後刪除。

percentile 一律 `np.percentile(x, q, method="linear")`。所有記憶體欄位以 bytes 儲存、欄位名稱帶單位；`ps` 的 rss 以 1024 bytes 為單位，乘以 1024。

### 7. Recall@1 取自另外一次 k = 1 search

草稿原本取 k = 10 結果的第一欄，並宣稱「與 k = 1 的答案等價」。審查以重複向量實測推翻：

- 有完全同距離（重複向量）時，第一欄的 id 取決於各 library 的輸出排序：Rust 與 hnswlib 在回傳集合內依 (distance, id) 升冪，
  FAISS 多為 id 降冪（007）。在 200 個方向 × 15 份副本的資料上，hnswlib / FAISS 有八成以上的 query「k = 1 的答案 ≠ k = 10 的第一欄」，
  id-based Recall@1 兩種取法相差可達 0.5 以上（hnswlib 偏高、FAISS 幾乎為 0）；Rust 兩者永遠相同。所有差異都在距離完全相同的 tie 上。
- 因此 **Recall@1（id-based 與 tie-aware）取自每個 ef 一次不計時的 k = 1 整批 search**，ef 與 k = 10 相同。
  - 它對應 SPEC 的「1-NN 查詢的品質」，不受 k = 10 輸出排序的影響；hnswlib 在 k = 10 不足時，k = 1 仍可得到 Recall@1。
  - 它在 tie 上**同樣是任意的**（各 library 在 tie 中選哪一個沒有保證），並不比第一欄「更正確」。
    tie 的影響由 003 的 tie-aware Recall@1 並列呈現，並依 §4 報告 rank 1 有 tie 的 query 數。
  - 額外記錄「k = 1 的答案 ≠ k = 10 第一欄」的 query 數，作為 tie 的說明材料（成本為零）。
- QPS、P50、P95 只在 k = 10 量測。
- 測試：無 tie 的資料上三個 adapter 的第一欄 id 等於 k = 1 的結果；重複向量資料上第一欄的距離等於 k = 1 的距離（差異只來自 tie 順序）；
  `test_third_party_semantics.py` 釘住「FAISS 在 tie 資料上 k = 1 ≠ 第一欄」；runner 測試以 stub adapter 確認 Recall@1 來自 k = 1 的結果。

### 8. 記憶體與 index size

| 欄位 | 定義 | 用途 |
| --- | --- | --- |
| `peak_rss_bytes` | build 一返回時的 `ru_maxrss` | SPEC 的 peak RSS。包含 interpreter、library 與輸入陣列（n·d·4 bytes） |
| `peak_rss_minus_baseline_bytes` | 上值減 baseline 的 `ru_maxrss` | build 期間增加的高水位 |
| `index_memory_bytes` | build 後的 `ps` rss 減 baseline 的 `ps` rss | **SPEC 的 index size**；recall–memory frontier 一律用它 |
| `serialized_bytes` | 原生格式的檔案大小 | 只作輔助欄位 |
| `peak_rss_after_search_bytes` | 所有 ef 點結束後的 `ru_maxrss` | 只作參考 |
| `index_footprint_bytes` | build 後的 phys_footprint 減 baseline（`proc_pid_rusage`） | 輔助欄位，用於 `memory` 標記 |

- recall 一律在父 process、子 process 結束後計算（`tie_aware_recall_at_k` 會把資料轉成 float64，n·d·8 bytes），
  子 process 不載入 ground truth，peak RSS 不會被 harness 主導。
- **檔案大小不能比較記憶體**：Rust 的格式（005）只存實際的鄰居數；hnswlib 與 FAISS 存固定容量的 slot（未用的格子也寫出，hnswlib 每點另有 8 bytes 的 label）；
  Rust 在記憶體中的 per-node `Vec<Vec<u32>>` 開銷也不在檔案裡。實測 n = 5000、d = 32 時檔案大小 Rust 最小，方向可能與記憶體相反。
- macOS 的 rss 不含被壓縮或 swap 出去的 page，而記憶體壓力在 macOS 上主要以壓縮呈現：
  - 子 process 在 baseline 與 after_build 都記錄 phys_footprint（含被壓縮的 page）。footprint 增量比 rss 增量多出 2% + 1 MiB 以上時，
    代表 build 期間有 page 被壓縮、`index_memory_bytes` 被低估，該 build 標記 `memory`。
  - 父 process 在每個子 process 前後記錄 `sysctl vm.swapusage` 與 `vm_stat` 的 compressor 頁數；swap 使用量增加的子 process 標記 `swap`。
  - page 大小 16 KiB；allocator 的雜訊由跨 seed 的分布呈現（010）。

### 9. hygiene 紀錄與標記

- **run 開始與結束**：CPU 型號、P / E core 數（`hw.perflevel0/1.physicalcpu`）、RAM、OS 版本、`pmset -g` 全文（含 powermode）、`pmset -g batt`、
  `pmset -g therm`、CPU 使用率最高的 15 個 process、load average、swap。
- **每個子 process 前後**：時間戳、load average、`pmset -g therm`、swap 與 compressor 頁數。
- **每個 ef 點前後**：時間戳與 load average（`os.getloadavg()`，不另開 process）。
- 標記（事先登記；被標記的點照常報告，**不會被悄悄重跑**，處理方式見 010）。分成子 process、build、點三層，報告時合併到點上：
  - `load`：該點（或 build）結束時 1 分鐘 load 超過 §2 的門檻。
  - `thermal`：子 process 前後的 `pmset -g therm` 出現等級不為 0 的 thermal / performance warning（實際格式為 `Thermal Warning Level = N`）、
    無法解析等級的 warning 訊息，或 CPU speed / scheduler limit < 100。
  - `swap`、`memory`：§8。
  - `power`：子 process 前或後不是 AC 電源。
  - `multithread`：§6.7.7。
  - `nondeterministic`：§6.7.4 或 §6.7.5 的不一致數 > 0。

### 10. metadata（每次 run 一份）

- §9 的機器資訊；Python（`sys.version`、`sys.executable`）、numpy、hnswlib、faiss-cpu、maturin 的版本（`importlib.metadata`；`hnswlib.__version__` 不存在）。
- `faiss.get_compile_options()`（由父 process 取得）。
- **Rust binding 的編譯資訊**：binding 的 `build.rs` 在編譯時記錄 profile、opt-level、debug、rustflags、`CARGO_PROFILE_*` 環境變數與 `rustc -V`，
  以 `rust_hnsw.BUILD_INFO` 公開（另含 `cfg!(debug_assertions)`）。runtime 讀到的 shell `RUSTFLAGS` 不代表 `.so` 的編譯設定，所以不用它。
- 三個擴充模組（`rust_hnsw`、hnswlib、faiss 的 `.so`）的路徑與 SHA-256。
  hnswlib 的 `-O3 -march=native` 是依其 `setup.py` 推論（007），不是觀察到的，照實註明；不記錄 runtime 的 `c++ --version`（不一定是當初編譯 hnswlib 的 compiler）。
- 007 §8 的 distance kernel 事實（hnswlib 在 arm64 上為純量 reduction、FAISS 為 NEON wheel、Rust 為純量迴圈）。
- git commit、`git status` 全文（smoke 也記錄）、config 全文與 SHA-256、資料集檔案的 SHA-256、mode、時間。
- **協定常數**：集中在 `benchmarks/protocol.py`，以 `protocol_version = "008-v1"` 整份寫入 run.json（002 要求 warm-up 與重跑次數寫入每次的紀錄）。
  有測試斷言這些值與本文件相同；改動時測試會紅燈，提醒要另寫 decision log。

### 11. Rust 的編譯設定

- formal 量測使用 `maturin develop --release` 的預設設定（opt-level 3，沒有 `target-cpu=native`，沒有 LTO），一律以 `./scripts/test.sh` 編譯。
- §1 強制 opt-level、debug assertions、rustflags 與 `CARGO_PROFILE_RELEASE_*` 環境變數。LTO 與 codegen-units 另由 tracked 的 `Cargo.toml` 決定
  （目前沒有 `[profile]` 段落，由 git 乾淨確保）。已知限制：使用者層級的 cargo 設定檔（例如 `~/.cargo/config.toml`）中的 `[profile]` 無法從 binary 得知，
  formal 前要人工確認沒有這類設定。
- 這與 004「baseline 刻意不優化」一致。aarch64 上 NEON 本來就是 baseline 指令集，`-march=native` 主要差在較新的擴充。
- 本專案的距離計算是純量 f32 迴圈（SIMD 屬 Optional），編譯旗標的差異連同 metadata 一起報告。

### 12. 輸出

`results/runs/<UTC 時間>-<mode>/`（與 009 的 `results/profiles/` 分開；`results/` 不進版控）：

- `run.json`：status（`complete` / `incomplete` / `aborted` / `invalid`，以及 runner 本身發生非預期例外時的 `error`：仍寫出 run.json 並附 traceback，
  例外重新拋出）與原因、protocol、config、metadata、子集資訊、執行順序、
  每個子 process 的狀態與 hygiene 紀錄、每個量測點的摘要，以及所有附檔的相對路徑與 SHA-256。
- `points/<point_id>.npz`：`ids_k10`（int64 [nq, 10]，已映射回子集列號）、`ids_k1`、`qps_ns`（[5]）、`latency_ns`（[3, nq]）。
- `children/<child_id>/`：spec、JSONL、log，以及空呼叫與 Python 迴圈的原始樣本。
- `gt/n<n>.npy`：每個 n 的 ground truth ids。
- 打亂後的資料 `.npy` 與存檔用的暫存檔放在 repo 外的暫存目錄（`tempfile.mkdtemp()`），每個 build 結束後與 run 結束時（含失敗）清除。
- 有測試只讀 run.json 與附檔重算所有指標，結果必須與摘要完全一致。

## 為什麼

- 子 process 隔離是量 peak RSS 唯一乾淨的方法，也避免實作之間互相污染；recall 移到父 process，peak RSS 才不會被 harness 的 f64 複本主導。
- 重跑次數、warm-up、門檻、指標定義先固定並寫進每份輸出，避免看到結果後才調整。
- formal 的前置檢查與執行中的重新確認，把 EXPERIMENT_RULES 從文字變成程式檢查，也讓「可以追溯到 commit」對實際被量測的 binary 成立。
- 交錯執行並輪換實作，讓時間漂移不會變成實作之間的系統性差異。
- 被標記的點照常報告、不悄悄重跑，避免事後挑選。

## 放棄的替代方案

- **同一個 process 依序量測三套實作**：peak RSS 無法區分，且有交互污染。
- **使用資料集自帶的 ground truth**：metric、tie-break 與精度規則可能與 001 不同；N 子集上也不適用。
- **Recall@1 取 k = 10 的第一欄**：tie 下不等價，會把各 library 的輸出排序混入主要指標（§7）。
- **以原生檔案大小作為 index size**：比的是三種格式而不是三個 index，而且偏向本專案（§8）。
- **在子 process 內計算 recall**：tie-aware 的 f64 複本會主導 peak RSS。
- **runner 在 formal 開始前自行編譯 binding**：編譯推高 load，與門檻衝突；改為驗證編譯資訊與來源。
- **load 門檻放在 config**：每次 run 都能自行放寬，與協定凍結矛盾。
- **以 calibration 偏離就作廢並重跑**：會引入「重跑哪些」的事後選擇；改為標記並照常報告。
- **CPU time / wall time 比值驗證單執行緒**：受負載影響（007 已放棄）；沿用「其他 thread 的 CPU time」。
- **強制設定 `OMP_NUM_THREADS=1`**：會掩蓋 adapter 在單執行緒上的失效。

## 對實驗的影響

- 所有正式數字都來自 formal 模式，可以追溯到 git commit、被量測的 binary、config 與 dataset checksum。
- 結果不足 k 的點會被明確標記，不會被補救；hnswlib 的這類點沒有 QPS / latency / Recall@10。
- recall–memory frontier 一律使用 `index_memory_bytes`；`serialized_bytes` 不得用於跨實作的記憶體結論。
- 第一次 formal run 之前必須先完成：資料集 decision（登記名稱與 SHA-256、子集規則）、閒置 load 的量測（§2）、formal config 的 commit。
- 協定常數（QPS 5 次、latency 3 次、整批 warm-up 1 次、單筆 warm-up 100 筆、空呼叫 100 + 1000 次、k = 10、load 門檻與等待上限、
  percentile 方法、gc 處理、query 數下限）自本文件定稿起固定；之後修改需要新的 decision log。

## 審查摘要（2026-10-08）

草稿以三個角度審查（量測方法、指標涵蓋、工程實作），每個 finding 都有一位驗證者重現或反駁，共 57 個 findings：17 個照原建議、38 個修改後採用、2 個駁回。

**主要修正：**

- **Recall@1**（三個角度都列為 blocker）：草稿「k = 10 第一欄等價於 k = 1」只在沒有 tie 時成立，改為另外一次 k = 1 search（§7）。
  驗證者對「第一欄還是 k = 1」意見相反：一方認為 k = 1 對應 1-NN 查詢、且不受輸出排序與 hnswlib 整批失敗影響；
  另一方指出 k = 1 在 tie 上同樣任意。採用前者，並把後者的論點寫進 §7（兩者都不比對方「正確」，tie 的影響由 tie-aware 並列呈現）。
- **peak RSS 被 harness 主導**：recall 移到父 process；peak RSS 定義在 build 一返回時；import 移到 baseline 之前（FAISS 的 import 約 20 ms、+16 MB）。
- **index size**：檔案大小改為輔助欄位，index size 改用 RSS 增量（§8）。
- **空呼叫成本**：16 點 index 仍有 traversal，三者的排序會隨維度反轉；改為 n = 1。
- **efC ≥ M 的驗證**：實測 FAISS 在 efC < M 時語意不同。
- **子集、插入順序與 id 映射**：草稿完全沒有定義；`perm[ids]` 會把 `-1` 變成合法 id。
- **binary 的來源**：審查期間就觀察到另一個 session 重新編譯了 `.venv` 中的 `.so`；改為 build.rs 記錄編譯資訊，並在執行中重新確認 SHA-256。
- **執行順序、load 門檻的位置與時點、thermal、子 process 的失敗處理、config 驗證（bool、未知 key、重複）**。
- **比較規則與 variance 的彙整**：改由 010 事先登記，runner 只輸出原始數字。

**駁回：**

- 「FAISS 的插入順序差異沒有記錄」：007 已記錄並有測試。
- 「比對 warm-up 與重跑結果以偵測 OpenMP 設定錯誤」：FAISS 每個 query 獨立計算，執行緒數改變不影響結果，抓不到；
  決定性的比對仍保留，但理由是偵測非決定性的 bug（§6.7.4）。

## 實作後的審查（2026-10-08，在任何正式結果之前）

runner 實作完成後，以三個角度審查（是否忠實落實 008 / 010、mutation testing 檢查測試鑑別力、強健性與 formal 的可行性），
每個 finding 都有一位驗證者重現或反駁，共 56 個 findings：20 個照原建議、23 個修改後採用、13 個駁回。依此修訂本文件（上面各節已更新）：

- **前置檢查的漏洞**：
  - 「`.so` 晚於 HEAD 且 git 乾淨」推不出「`.so` 由 HEAD 編出」（改原始碼、編譯、再還原就能通過）→ 改為也要晚於每個 tracked 原始碼檔（§1.6）。
  - formal config 可以放在 repo 外或被 ignore 的路徑 → 新增 §1.8。
  - 重新確認排在 load 等待之前、最後一個子 process 之後沒有確認、只比對 rust_hnsw → 都已修正。
  - `CARGO_PROFILE_RELEASE_*` 覆寫看不到 → build.rs 記錄並拒絕（§11）。
- **thermal 標記在本機永遠不會觸發**：pmset 的實際格式是 `Thermal Warning Level = N`（大寫），原本的比對區分大小寫；等級 0 代表正常（§9）。
- **記憶體壓力**：macOS 主要以壓縮而不是 swap 呈現 → 新增 phys_footprint 與 `memory` 標記（§8）。新增 `power` 標記與 AC 電源的前置檢查。
- **中止與收尾**：中止時進行中的子 process 不會出現在 run.json、SIGHUP 沒有處理、收尾的任何失敗都會讓 run.json 寫不出來 → 都已修正（§5、§12）。
- **機密**：`CARGO_*` 前綴可能寫出 registry token → 改為不記錄，並以名稱過濾機密（§5）。
- **測試鑑別力**：mutation 發現 hygiene 標記、formal 檢查的接線、008 §6 的呼叫順序與次數、記憶體的量測時點、計時區間的邊界都沒有測試保護。
  已補上：以追蹤用的 stub 斷言完整的呼叫序列（包括 gc 狀態），以假時鐘斷言計時區間只包住 `adapter.search`，
  以配置記憶體的 stub 斷言 baseline 與 peak RSS 的時點，以注入的 snapshot 與 stub 正向測試每一種標記。
- **父 process 的記憶體**：tie-aware recall 不再把整份 data 轉成 f64，原始陣列在 prepare 之後即釋放。

**駁回的主要項目**：要求 runner 對單一 config 強制 010 的最低 variance 次數（會擋下合理的分批 run）、
比對 lock owner 的 process 名稱或啟動時間（綁死 with-mac-lock 的格式，且情境需要 stale lock 加上 pid 重用）、
每個子 process 結束就寫一次 run.json（收尾已改為不會失敗，SIGKILL / 斷電屬極端情況）、
以及 ground truth 計算期間無法立即中止（此時還沒有任何量測，不會遺失資料）。
