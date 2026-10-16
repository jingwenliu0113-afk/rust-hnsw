# 007：hnswlib / FAISS adapters 與實測語意

日期：2026-10-08
狀態：採用（實作後經一輪獨立審查修訂，見文末）

## 背景

SPEC Core 4 要求以 hnswlib 與 FAISS HNSW 作為成熟的 ANN baseline，並走 002 / 006 的同一層 Python batch interface。
006 的審查指出幾項 hnswlib / FAISS 的行為「只能依記憶、未確認」。

使用者要求：

- 先實測第三方 library 的實際語意，不只依文件假設。
- 三套實作使用同一批 float32、L2 正規化的 unit vectors。
- 明確記錄各自回傳 similarity 或 distance；轉換統一在 adapter 層做。
- 兩者都固定單執行緒，並以測試驗證設定真的生效。
- 實測 `ef < k`、結果不足 k、重複向量、tie 順序、padding 的 id 與 distance。
- 參數一律明確傳入、不依賴 library default。
- 結果不足 k 時不得改成 per-query 迴圈。
- 版本要 pin 死。

## 實測環境

- hnswlib **0.8.0**：PyPI 上只有 sdist，由 uv 在本機以 clang 21 編譯。依其 `setup.py`，旗標為 `-O3 -march=native`（本機支援 native）。
- faiss-cpu **1.15.1**：官方 wheel，`faiss.get_compile_options()` 為 `OPTIMIZE DD ARM_NEON MAC_METAL`。
- numpy 2.5.3、CPython 3.13.9、macOS arm64（Apple M4 Pro，12 cores）。

第三方欄位中，標 **[測試]** 的項目由 `tests/test_third_party_semantics.py` 釘住；標 **[原始碼]** 的項目依原始碼行號確認。
library 升級後若測試紅燈，代表語意改變，必須回頭修訂本文件與 adapters。

## 實測語意

| 項目 | 本專案 Rust（004） | hnswlib 0.8.0 | FAISS 1.15.1 `IndexHNSWFlat` |
| --- | --- | --- | --- |
| 建構 | `HnswIndex(dim, m, efC, seed)` | `Index(space="ip", dim)` + `init_index(...)` | `IndexHNSWFlat(d, M, METRIC_INNER_PRODUCT)` |
| 回傳的量 | distance `1 − dot`，升冪 | distance `1 − ip`，升冪 [測試] | **similarity** `ip`，**降冪** [測試] |
| id 型別 | int64（binding 轉換） | **uint64** [測試] | int64 [測試] |
| 預設參數 | M=16、efC=200（binding） | M=16、efC=200、ef=10 [測試] | efConstruction=40、efSearch=16 [測試] |
| 預設執行緒 | 單執行緒 | `hardware_concurrency`（本機 12），per-index [測試：> 1] | OpenMP，**per-thread** 設定；新 thread 的預設是 `OMP_NUM_THREADS` 或核心數（本機 12） [測試] |
| 實際使用的 ef | `max(ef, k)` | `max(ef, k)`（`hnswalg.h:1308`） [原始碼、測試：ef=1、5、10 結果相同] | **不等於** `max(ef, k)`：efSearch=1、5、10 結果都不同；efSearch ≪ k 時一般資料也會回傳少於 k 個結果 [測試] |
| 結果不足 k | 補 `-1` / `+inf` | **整批 raise** `RuntimeError("Cannot return the results in a contiguous 2D array. Probably ef or M is too small")` [測試] | 補 id `-1`，D 為 **`−FLT_MAX`**（不是 `−inf`） [測試] |
| k > n | `ValueError` | 同上，整批 raise [測試] | 補 `-1` / `−FLT_MAX` [測試] |
| k = 0 | `ValueError` | 回傳 shape `(q, 0)` | `AssertionError` |
| query 維度不符 | `ValueError` | **不檢查**：依 index 的 dim 讀每列，維度不足時越界讀取（`knnQuery` 沒有檢查；`add_items` 有） [測試] | 只有 Python wrapper 的 `assert` [測試] |
| query dtype / layout 不符 | `TypeError` / `ValueError` | 靜默轉型並複製（`array_t` forcecast，`bindings.cpp:617`） | 靜默轉型並複製（wrapper 的 `ascontiguousarray`） |
| 同距離（tie）的順序 | `(distance, id)` 升冪（保證） | 回傳集合內 `(distance, label)` 升冪（`searchKnn` 的 result heap 加上 `knnQuery` 從尾端回填，保證）；邊界 tie 時選進哪些點不保證 [原始碼、測試] | 多數為 id 降冪，但會混合；完全相同的向量算出的距離可能差 1 ulp，所以重複向量不一定是 tie [測試只釘住 2-D fixture] |
| 各層容量 | layer 0 為 2M、其他層 M | `maxM0_ = 2M`（`hnswalg.h:113`） [原始碼、測試] | `cum_nneighbor_per_level` = [0, 2M, 3M, …] [測試] |
| **新點在 layer 0 選的鄰居數** | **M** | **M** | **最多 2M**：512 維隨機資料上，每個點的 layer-0 degree 都 > M [測試] |
| level 分佈 | `mL = 1/ln M` | `mult_ = 1/ln M`（`hnswalg.h:142`） [原始碼、測試] | `assign_probas[0] = 1 − 1/M` [測試] |
| level 亂數 seed | `seed` | `random_seed`（`init_index` 參數） | 預設固定；可用 `hnsw.rng = faiss.RandomGenerator(seed)` 設定，**只影響 level** [測試] |
| **插入順序** | row 順序 | row 順序 | **不是 row 順序**：先依 level 分組，組內以內部固定的亂數重排；最高層有多個點時，entry 不一定是 id 最小者 [測試] |
| 決定性（單執行緒） | 是 | 是（同 seed） [測試] | 是（同 seed） [測試] |

### 大量完全重複的向量（synthetic stress）

設定：`duplicate_data(seed, 300, 6)`、efC=20、ef=300，2 組資料 × 2 個 build seed。統計每個 query 可回傳的有效結果數。
hnswlib 的數字以逐 query 探測「可回傳的最大 k」取得，不是 adapter 的行為。

| M | Rust | FAISS（min / median） | hnswlib（min / median） |
| --- | --- | --- | --- |
| 4 | 恆為 2M + 1 = 9 | 10–18 / 18–24 | 12–227 / 48–266 |
| 8 | 恆為 2M + 1 = 17 | 24–42 / 26–42 | 42–292 / 156–292 |

- 三者在大量重複向量上都可能結果不足 k。
- **graph 斷裂最嚴重的是本專案的 Rust**：004 的 `(distance, id)` tie-break 讓副本都指向 id 最小的那批（見 004「實作後的觀察」）。
- hnswlib 的整批 raise 是 API「全有或全無」的結果，不代表它的 graph 較差；它連通較好的原因（同距離時 heap 順序不固定）是推測，尚未驗證。
- 這是 synthetic stress 的觀察，不作 transferable 結論。Core 階段不改演算法（依 004）。

## 決策

### 1. 同一份輸入，build 時強制檢查

三套實作都吃 `prepare_vectors` 產生的同一份 float32、C-contiguous、L2 正規化的 unit vectors（002）。

- hnswlib 用 `space="ip"`，不用 `"cosine"`。`"cosine"` 會在每次 `knn_query` 內再正規化並複製一次，等於在計時路徑上不利於 hnswlib；對 unit vectors，兩者的 labels 相同，距離差約 1e-7 量級（f32 rounding，與資料有關）。
- FAISS 用 `METRIC_INNER_PRODUCT`。
- 本專案的 Rust 會在內部再正規化一次。這是 002 已記錄、對自己略為不利的 overhead。
- **hnswlib 與 FAISS 的 cosine 等價只在 unit-norm 輸入時成立**，而 Rust 的內部正規化會掩蓋這個錯誤：
  - 審查實測：norm 只偏差 0.9–1.1 時，Rust 的 Recall@10 仍是 1.0，兩個 baseline 降到 0.776，而且 `check_search_result` 抓不到。
  - 因此**三個 adapter 都在 build 時檢查**：float32、C-contiguous，且每列 `|‖v‖² − 1| ≤ 1e-5`；不符就 `ValueError`。
  - 成本 O(n·d)，三者相同，不在 search 的計時路徑上。
- queries 的 dtype、layout 與 unit norm 由 runner 在計時前檢查一次（008），adapter 的 search 不重複檢查。
  若 runner 只跑 hnswlib 或 FAISS，這個檢查可避免靜默轉型的成本進入計時路徑。

### 2. 回傳值的轉換一律在 adapter 層

統一輸出 002 的格式：ids 為 int64、distances 為 float32 cosine distance、升冪、不足 k 時補 `-1` / `+inf`。runner 不做任何轉換或猜測。

- **Rust**：binding 已輸出統一格式，不轉換。
- **hnswlib**：distance 已是 `1 − ip`，不轉換；labels 以 `labels.view(np.int64)` zero-copy 轉成 int64（id < 2^63）。
- **FAISS**：
  - similarity 轉 distance：`np.subtract(1, D, out=D)`（in-place）。f32 rounding 是單調的，升冪不變。
  - padding：`I == -1` 的位置把 distance 設為 `+inf`。原始的 `−FLT_MAX` 轉換後會變成約 3.4e38，不是 `+inf`。

### 3. 結果不足 k

- Rust 與 FAISS 補齊，adapter 轉成 `-1` / `+inf`。
- hnswlib 整批 raise，adapter **不改成 per-query 迴圈，也不做任何 fallback**（使用者要求；改變 execution model 違反 002）。
  - 實作：只有 hnswlib 的這一條訊息會轉成專案的 `IncompleteResultsError`（`RuntimeError` 子類），讓 runner 不必比對 library 的錯誤字串；其他錯誤原樣拋出。
  - 測試以呼叫記錄器斷言：每次 search 只有一次原生呼叫，失敗後也沒有逐筆重試。
  - runner 遇到 `IncompleteResultsError`，把該（實作、資料集、參數、k）點記為「結果不足 k」，不產生 recall / latency（008）。
    008 定稿後補記：Recall@1 另以一次 k = 1 的 search 取得（008 §7），不受 k = 10 整批失敗影響。
  - 這個不對稱必須在報告中呈現，不得隱藏。

### 4. ef 與 k

- Rust、hnswlib 實際使用 `max(ef, k)`；FAISS 在 `efSearch < k` 時結果不同，efSearch ≪ k 時還會回傳少於 k 個結果。
- adapter **不模擬** `max(ef, k)`，不改變 FAISS 的原生語意。
- 依 006，parameter sweep 一律 `ef ≥ k`，由 runner 在計時前檢查。這是**必要條件**，不只是建議。
- `ef ≥ k` 只保證三者的 **search loop** 語意一致。**同一組 M / efConstruction 建出的 graph 並不等價**：
  FAISS 的新點在 layer 0 最多選 2M 個鄰居，Rust 與 hnswlib 只選 M 個。審查實測，同 (M=8, efC=40, ef) 下 FAISS 的 recall 穩定高 0.02–0.05；
  在同一張 FAISS graph 上改用 hnswlib 式的 search loop，recall 不變，所以差異來自建構，不是搜尋。

### 5. 參數一律明確傳入

`M`、`efConstruction`、`seed` 在建構 adapter 時給定，必須是整數（`operator.index`）；`efSearch` 由 `set_ef_search` 給定，沒有預設值（006）。

- hnswlib：`init_index(max_elements=n, ef_construction=efC, M=M, random_seed=seed)`；`set_ef(ef)`。
- FAISS：`IndexHNSWFlat(d, M, METRIC_INNER_PRODUCT)`；add 之前設定 `hnsw.efConstruction = efC` 與 `hnsw.rng = RandomGenerator(seed)`；`hnsw.efSearch = ef`。
- 三者的 RNG 不同，同一個 seed 在不同實作中不會產生相同的 level。seed variance 是各實作自己的 variance。
- FAISS 的 seed 只控制 level；插入順序由 FAISS 內部決定（見表格），所以 **insertion order variance 對 FAISS 的意義與另外兩者不同**：
  打亂 row 順序會改變結果，但不等於改變插入順序。報告時要分開說明。

### 6. 單執行緒

- **hnswlib**：`set_num_threads(1)`，並在 `add_items` / `knn_query` 明確傳 `num_threads=1`。
- **FAISS**：OpenMP 的 nthreads 是 **per-thread** 設定，只對呼叫的 thread 生效。
  - 審查實測：在 main thread 設成 1 之後，從另一個 thread 呼叫 search 仍會用 12 個執行緒。
  - adapter 在 build 與**每次 search 開頭**都呼叫 `omp_set_num_threads(1)`，與 hnswlib 每次呼叫都傳 `num_threads=1` 對稱，成本約數十 ns。
- **測試驗證設定真的生效**：
  - 量測「其他 thread 消耗的 CPU time」，即 `process_time` 的增量減去 `thread_time` 的增量。單執行緒時這個值為 0，與 wall time 和機器負載無關，所以不需要 skip。
  - 斷言 build 與 search 期間其他 thread 的 CPU time ≤ 2% + 1 ms。
  - 量測前主動把 OpenMP 與 hnswlib index 的執行緒數設成 4，模擬環境預設是多執行緒的情況。
  - FAISS 另測「在 main thread build、在另一個 thread search」。
  - 正對照：在 4 個執行緒下，其他 thread 的 CPU time 必須 > 30%，證明量測有鑑別力。

### 6.5 query 的 k 與維度由 adapter 檢查

- 原生行為不一：hnswlib 的 `knn_query` 不檢查維度（會越界讀取），FAISS 只用 `assert`；k 的行為三者也不同（見表格）。
- hnswlib 與 FAISS adapter 在 `search` 開頭檢查維度與 `1 ≤ k ≤ n`，不符就 `ValueError`，與 Rust binding 一致。
  這是 O(1) 的整數比較，避免記憶體越界讀取，也避免 runner 把設定錯誤誤記成「結果不足 k」。
- k 以 `operator.index` 轉成 Python int（FAISS 的 SWIG 介面不接受 `np.int64`）。

### 6.6 計時路徑上 adapter 的額外工作

- **Rust**：Python 層沒有額外工作；檢查與轉換都在 FFI 內。
- **hnswlib**：維度與 k 檢查、`labels.view(np.int64)`。
- **FAISS**：維度與 k 檢查、`omp_set_num_threads(1)`、SWIG wrapper 本身（`ascontiguousarray`、配置輸出陣列、`assert`）、`np.subtract`、padding mask。

依 006，runner 的空呼叫基準必須呼叫完整的 `adapter.search`，並在 lock 下量測。本文件不記錄未持 lock 量到的數字。

### 7. tie 的順序不做標準化

三者在同距離時的順序不同。adapter 不重新排序：重新排序會在計時路徑上加入排序成本，也會改變原生行為。
這對 id-based Recall@K 的影響由 003 的 tie-aware 診斷量化；該診斷以 f64 重算距離，不受 FAISS 的 ulp 差異影響。

### 8. 編譯旗標與 distance kernel 要記錄

- hnswlib：本機以 `-O3 -march=native` 編譯。但 **arm64 上 hnswlib 0.8.0 沒有手寫 SIMD**：
  `USE_SSE` / `USE_AVX` 只在 x86 定義，`InnerProduct` 是純量迴圈。
  反組譯 `InnerProductDistance` 可見乘法被自動向量化（`fmul.4s`），加總仍是依序的純量 `fadd`（沒有 `-ffast-math`）。
- FAISS：官方 NEON wheel。
- 本專案 Rust：release 預設（沒有 `target-cpu=native`），依 004 是純量迴圈。

benchmark metadata 必須記錄以上事實，包括「hnswlib distance kernel：arm64 上為純量 reduction」。
兩者 kernel 的實際 codegen 與差距歸因，以 Gate 要求的 profiling 確認，本文件不預先下結論。
Rust 是否要以 `target-cpu=native` 編譯屬於 benchmark protocol，在 008 決定。

### 9. 版本 pin

`requirements-dev.txt` 固定 `hnswlib==0.8.0`、`faiss-cpu==1.15.1`。

## 為什麼

- 依實測而不是文件寫 adapter：
  - FAISS 的 `ef < k` 語意、`−FLT_MAX` padding、per-thread 的 OpenMP 設定、layer 0 鄰居數、插入順序，都和事前的預期不同。
  - hnswlib 整批 raise，確認了 006 審查的風險；`knn_query` 不檢查維度，是寫測試時才發現的。
- 轉換集中在 adapter：runner 只看到一種格式，避免每個呼叫端各自猜 similarity 或 distance。
- 不模擬、不 fallback：benchmark 量的是各 library 的原生行為；任何補救都會改變 execution model 或計時路徑。
- build 時強制 unit norm：本專案的實作恰好最能容忍錯誤輸入，不檢查會讓比較默默偏向自己。

## 放棄的替代方案

- **hnswlib 用 `space="cosine"`**：每次查詢多一次正規化與複製。
- **hnswlib 結果不足 k 時逐筆查詢**：違反 002 與使用者要求。
- **FAISS adapter 模擬 `max(ef, k)`**：會改變 FAISS 的原生語意。
- **adapter 依 (distance, id) 重新排序 tie**：增加計時路徑的成本，也改變原生行為。
- **只在 build / set_ef_search 設定 FAISS 執行緒數**：OpenMP 是 per-thread 設定，換 thread 呼叫 search 就失效。
- **在 search 內檢查 query 的 dtype / layout**：每次呼叫增加約 100 ns；改由 runner 在計時前檢查一次。
- **以 CPU time / wall time 比值驗證單執行緒**：受機器負載影響，忙碌時必須 skip；改用其他 thread 的 CPU time。

## 對實驗的影響

- 正式比較只使用 `ef ≥ k` 的點（006），由 runner 強制。
- **跨實作只在相近的 Recall@K 下比較**（SPEC §4）。同 (M, efC, ef) 的 recall-vs-efSearch 曲線與 build time 不能直接對照，因為三者建出的 graph 不等價（§4）。
- 在大量重複向量的資料上，三者都可能結果不足 k；hnswlib 的點會整批標記，不會被補救。graph 斷裂最嚴重的是本專案的 Rust（見重複向量一節）。
- FAISS 的 tie 順序與另外兩者不同，需同時報告 003 的 tie-aware 診斷。
- insertion order variance 對 FAISS 的意義不同，要分開報告（§5）。
- 三者的 SIMD / 編譯旗標事實見 §8，要連同結果一起報告。

## 審查摘要（2026-10-08）

實作後以兩個角度審查（語意與公平性、測試鑑別力與 flaky 風險），各配一個反駁者，共 25 個 findings，都附實際重現。

**實作修正：**

- FAISS 的 OpenMP 設定改為每次 search 都設定（per-thread）。
- 三個 adapter 在 build 時強制 unit norm。
- hnswlib / FAISS 的 k 檢查與 Rust 一致。

**文件修正：**

- 改正「process 全域」「tie 順序不保證」「hnswlib 有 SIMD 優勢」等錯誤敘述。
- 補上重複向量、layer 0 鄰居數、插入順序、efSearch ≪ k 補 -1、計時路徑額外工作等實測。

**測試補強**（反駁者以 mutation 重現：拿掉對應行為後原測試仍全綠）：

- 每次 search 只有一次原生呼叫，hnswlib 失敗後沒有逐筆重試。
- 單執行緒驗證改量其他 thread 的 CPU time，並注入多執行緒的預設值；另測 FAISS 跨 thread 呼叫。
- 測 hnswlib 用 ip space、FAISS 用 INNER_PRODUCT。
- efSearch 改用不等於任何 library 預設的值（37）。
- 參數傳遞測試改用 200 筆 query，並以 white-box 斷言確認參數真的設進 library。

**兩位反駁者意見相反的一項**：search 時是否檢查 query 的 dtype / layout。
折衷為由 runner 在計時前檢查一次（§1），adapter 的 search 不檢查。

## 補記（2026-10-17，Advanced C 的實作之後、任何真實資料上的量化結果之前）：`IndexHNSWSQ` 與 Q 的移植

decision 019 的 `faiss-hnsw-sq8`（Q）把 `FaissHnsw` 建出的 Flat graph 移植進 `IndexHNSWSQ`，不用 FAISS 以量化距離建圖。
細節見 019 的補記（2026-10-17）第 2 節；以下是 `tests/test_third_party_semantics.py` 新增的 [測試] 事實：

- **SQ storage**：
  - 必須先 train，否則 add 拋出 `is_trained` 錯誤。
  - QT_8bit、RS_minmax、`rangestat_arg = 0` 時，`trained` 等於逐維的 min 與 max − min。
  - vdiff = 0 的維度 code 為 0、解碼為 vmin；n = 1 時重建與原向量逐位元相同。
- **search 的輸出**：
  - similarity 可略大於 1（實測約 1.0039），也可小於 −1。adapter 以 `np.clip(1 − ip, 0, 2)` 轉成距離，這是 Q 在計時路徑上比 `FaissHnsw` 多出的一步。
  - padding 是 −1 / −FLT_MAX，adapter 改成 +inf。
  - `efSearch` 的語意與 Flat 相同；同 seed 時結果是決定性的。
  - 在 `omp_set_num_threads(1)` 下，train、add、移植與 search 都是單執行緒。
- **移植**：
  - 複製 `hnsw`（levels、offsets、neighbors、entry_point、max_level）並設 `ntotal`；storage train、add 之後，設 `is_trained = True`。
  - 只 train storage 時，外層 `is_trained` 仍是 False。
  - 移植是深複製，改動或釋放 Flat 都不影響 Q；`write_index` / `read_index` 之後 graph 與結果逐位元相同。
- **定案**：移植在 synthetic 測試上全部通過，判定可行。019 §3 的備案（原生 `IndexHNSWSQ`）不採用，之後也不得改用。
