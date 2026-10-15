# 015：Gate 證據的讀法與 Gate profiling 的參數（事先登記）

日期：2026-10-09
狀態：採用（在看任何正式效能結果之前；是否進 Advanced 仍由使用者決定）

## 背景

SPEC 的 Gate 有四個條件：correctness 通過；Recall 隨 efSearch 整體改善或飽和，沒有系統性惡化；同 recall 下與 hnswlib 有差距；
profiler 已顯示時間主要花在哪。另有一個特例：Core 已很接近 hnswlib 時，不必硬做 optimization attribution，Advanced 改聚焦 vector quantization frontier。

既有的 decision 沒有固定以下兩件事：

- 010、013 規定了 frontier 讀值、QPS 比值與 seed 範圍，但沒有規定**怎樣才算「有差距」**。
- 009 規定了 profiling 的工具與證據條件（正式資料集、heavy lock、相近的 Recall@K），但沒有規定**哪個格點、哪個 recall、ef 怎麼選、取樣多久、取樣後怎麼核對**。

兩者都會直接改變 Gate 的結論，必須在看到正式結果之前固定。

撰寫時的狀態：GloVe 的 formal config 1–4 已完成，5 因背景的 macOS 系統工作（mediaanalysisd）被 load gate 中止（013 §5，整份重跑）。
撰寫者只看過這些 run 的狀態、hygiene 標記與背景工作的診斷紀錄，**沒有看過任何 formal run 的 recall、QPS、latency、build time 或記憶體**，
也還沒有產生任何 formal 報告。

另外補上 013 §3 已規定、但 `report.py` 還沒有輸出的讀值：主要結論附上的 seed 範圍。

## 決策

### 1. Gate 的讀值（由每個資料集的正式報告產生，不另外量測）

| SPEC 條件 | 讀值 |
| --- | --- |
| correctness 通過 | `./scripts/test.sh` 全部通過（含小型人工可驗證案例，004）；真實資料的 correctness run（NEXT「目前位置」） |
| Recall 隨 efSearch 整體改善 / 飽和 | 正式報告的 010 §6 表：rust-hnsw 在每個 (n, M, efC) 都沒有系統性惡化 |
| 同 recall 下與 hnswlib 有差距 | 本節下列的判讀 |
| profiler 已顯示時間主要花在哪 | 第 2 節的取樣 |

**差距的判讀**（每個資料集各自判讀；主要 n = 100,000；`report.py` 的 `gate`）：

- **主要目標：Recall@10 = 0.95**。0.90、0.99 照同樣規則計算並列出，只作輔助。
  理由：0.95 是三個目標的中間值，兩個資料集的格點都預期能達到；GloVe 的 0.99 只有 M48 一個格點，文字的 0.99 不確定（013 §2）。
- **QPS 範圍（處理 ef 梯子的量化，不內插）**：frontier 讀值取「達標點中 QPS 最高者」，但真正的交點在某兩個量測的 ef 之間，
  梯子間距 ×1.33–1.5 讓讀值最多低估約 25–33%（013 §2），而且兩個實作的低估程度不同。因此對每個實作算出一個範圍：
  - 下界 = frontier 的 QPS 讀值（010 §2）。
  - 上界 = 每個 (M, efC) 格點中，recall 第一次達標的 ef **前一個** ef 的 QPS，取所有格點的最大值。QPS 隨 ef 遞減，真正的交點不會比它快。
    格點的第一個 ef 就已達標時，交點可能在更小的 ef，上界未知；格點在 ef 清單內沒有達標時，交點在最大 ef 之後，以最大 ef 的 QPS 為上界。
  - 假設 recall 不會在相鄰兩個量測的 ef 之間先超過目標再掉回去。
- **方向**：hnswlib 的下界 > rust-hnsw 的上界 → hnswlib 較快；rust-hnsw 的下界 > hnswlib 的上界 → rust-hnsw 較快；範圍重疊 → 無法區分。
- **seed 範圍（處理 seed 與 run 之間的變異）**：方向成立時，每個 013 §3 附上的 variance 格點
  （GloVe 0.90 / 0.95 為 (16, 200) 與 (48, 200)、0.99 為 (48, 200)；文字三個目標都是 (16, 200)）都要分開：
  較快一方在它第一個達標 ef 上**最慢**的 seed，仍快於較慢一方在交點**之前**那個 ef（沒有達標時為最大 ef）上**最快**的 seed。
  兩個點各至少要有 3 個未標記的 build（variance 格點有 5 個 seed）；格點缺資料、上界未知或 build 不足都算沒有分開。
- **判讀**：
  - `gap`（有差距，hnswlib 較快）：QPS 範圍分開、方向為 hnswlib 較快，而且每個附上的 variance 格點的 seed 範圍都分開。
  - `rust_faster`：方向相反，條件相同。
  - `within_quantization`：QPS 範圍重疊，在 ef 梯子的量化之內無法區分。
  - `uncertain`：QPS 範圍分開，但某個附上的 variance 格點沒有分開（seed 範圍重疊、缺資料、上界未知或 build 不足）。
  - `undetermined`：任一方沒有達到目標。
- **被標記的 build**：主要判讀排除有 hygiene 標記的 build（010 §2）；另以全部 build 再判讀一次，兩者不同時並列（010 §3），不自行擇一。
- **兩個資料集合起來**（事先登記全部組合）：
  - SPEC 的「同 recall 下與 hnswlib 有差距」**只在主要目標判讀為 `gap` 的資料集上成立**。兩個都是 `gap`：條件成立。只有一個是 `gap`：
    只在該資料集成立，照實寫明；Advanced 若進行，以該資料集為主。
  - 沒有任何資料集是 `gap`：條件不成立。`within_quantization`、`uncertain`、`undetermined` 都**不**代表「很接近 hnswlib」；
    是否適用 SPEC 的特例（很接近、改聚焦 quantization），需要看 QPS 範圍與 seed 範圍本身，由使用者判斷。`rust_faster` 照實報告。
- P50 / P95、index memory、build time 的比值照常列在報告中，作為選擇 Advanced 假設的參考，不用於上述判讀。
  FAISS 同樣列出；SPEC 的 Gate 只以 hnswlib 為準。
- **是否進 Advanced、做哪些假設（最多三個），由使用者決定**。本文件只固定讀法，記到 NEXT 的「需要使用者」。

### 2. Gate profiling 的參數

- **格點**：每個資料集 n = 100,000、(M, efC) = (16, 200)、seed 1、原始插入順序。這是兩個資料集共同的中心格點與完整 variance 格點（013 §1），
  每個 build 的成本也最低。
- **ef**：各實作各自選（007 §4：三者的 graph 不等價，不用相同的 ef）。在正式 run 的這個 build 上，Recall@10 ≥ 0.95 的最小 ef。
  任一實作在 ef 清單內沒有達到 0.95 時，**三個實作一起**改用 0.90，並在計畫中註明，讓取樣仍在相近的 recall（009）。
  用 seed 1 這個 build 本身的 recall，而不是跨 seed 的 median：被取樣的 index 就是這個 build。
  recall 不受 hygiene 標記影響，所以被標記的 build 也照樣用來選 ef。
- **階段與時間**：每個實作取樣 search 30 秒、build 300 秒（`sample` 每 1 ms 一次，009）。build 階段持續重建；
  依 013 的成本模型，文字 100k (16, 200) 的 build 三個實作合計約 591 秒（負載下上限）。
  單一實作的 build 若超過 300 秒，取樣只涵蓋 build 的前段，照實註明，不延長。
- **順序**：同一資料集先 search、後 build；同一階段依 rust-hnsw、hnswlib、faiss-hnsw。兩個資料集各 6 次取樣。
- **條件**：`./tools/with-mac-lock` 下執行；formal 的電力條件與 load / 背景 gate（014 補記）；git 乾淨、binding 晚於 HEAD 與每個 tracked 原始碼檔
  （`profile.py` 作為證據取樣時檢查，不符就拒絕）；HEAD 與 formal run 之間的實作與量測程式（`report.py` 的 `MEASUREMENT_PATHS`）沒有差異。
- **計畫**：`benchmarks/gate_profile.py plan` 從正式 run 選 ef 並產生指令，寫在 `results/profiles/` 下（不讓 repo 變 dirty）。
  計畫一開始就拒絕覆寫已存在的同名輸出：上一次嘗試的檔案要先移開，不得混進新的證據。
- **核對**（`gate_profile.py check`，每個實作、兩個階段都核對）：取樣前紀錄、`sample` 原始報告與排行都存在，而且取樣晚於紀錄；
  紀錄的參數與計畫完全相同；持有 heavy lock、通過 formal 條件；git 乾淨；紀錄的 commit 與每個 formal run 的 commit 之間實作與量測程式沒有差異；
  `rust_hnsw.BUILD_INFO` 與套件版本和 formal run 相同；search 階段的 Recall@10 與正式 run 同一個 build 的選定點完全相同
  （同一個 `metrics.recall_at_k`，三個實作的 build 在相同 seed 下都是確定的）。不符時照實記錄，該取樣不作為 Gate 證據，直到查明原因。
- **呈現**：009 的 top-of-stack（self time）排行，每個實作、每個階段列出前 20 名，原始報告保留在 `results/profiles/`。
  把函式歸類成「距離計算、graph 走訪、visited set、heap、記憶體配置、Python / FFI」等類別是看過取樣之後的解讀，照實標明，原始排行才是證據。
  Rust 的距離計算被 inline 進 `search_layer`（009 已知限制），需要拆開時依 009 另寫 decision。
- 不輸出任何 latency / QPS 數字（009）。

### 3. `report.py` 補上 013 §3 的 seed 範圍與 Gate 讀值

- frontier 的每個讀值附上該點的跨 seed min–max（013 §3 第 1 點）；表格在 build 數 ≥ 2 時列出範圍與 build 數。
- `variance_ranges`：每個目標、每個 variance 格點、每個實作，在自己的最小達標 ef 上的 Recall@10、QPS、P50、P95 跨 seed min–max；
  格點在 ef 清單內沒有達到目標時改用最大 ef，標明「與讀值不在同一 recall 區間」（013 §3 第 2 點與 fallback）；
  缺資料時列出，不略過。主要版本排除有 hygiene 標記的 build（010 §2），另以全部 build 計算，不同時並列（010 §3）。
  表中註明：其他 n 只有 1 個 seed，以本表作為代理範圍（013 §3）。
- `gate`：第 1 節的判讀（`qps_bracket` 與各 variance 格點的 seed 範圍），報告另列一節，標明「建議判讀，是否進 Advanced 由使用者決定」。
- 這些是新增的輸出欄位，不改變任何既有讀值的定義；`report.py` 不在 `MEASUREMENT_PATHS` 中，formal run 之間不受影響。

## 為什麼

- 判讀規則、格點與 ef 的選法在看到正式結果之前固定，Gate 的結論就不能是看了數字之後挑出來的；所有組合的意義也事先寫明。
- QPS 範圍直接用量到的相鄰 ef 夾住交點，既不內插（010 §2），也不必假設一個固定的量化倍數；兩個實作各自的量化誤差都被涵蓋。
- seed 範圍要求「較快一方最慢的 seed 快於較慢一方最快的 seed」，而且兩邊都取對自己最不利的那一側的 ef，避免 seed 或 run 之間的漂移被當成差距。
- profiling 只取一個格點、一個 recall：Gate 要回答的是「時間主要花在哪」，不是 profile 的 scaling；兩個資料集都取，因為維度（100 與 768）會改變距離計算的占比。

## 放棄的替代方案

- **以讀值比值的固定門檻判讀**（初稿用 4/3）：梯子有一半的間距是 ×1.5，4/3 不足以涵蓋量化誤差；而且量化誤差在兩個方向都有，
  比值低於門檻也不代表「很接近」。改用上面的 QPS 範圍。
- **以 10% 作為差距門檻**：遠小於讀值的量化誤差。
- **只看 seed 範圍是否分開**：variance 格點用同一條梯子，量化誤差相同，seed 範圍分開不能排除量化造成的差距。
- **在相鄰 ef 之間內插出相同 recall 的 QPS**：010 §2 不內插。
- **所有實作用相同的 ef 取樣**：三者的 graph 不等價，相同 ef 的 recall 不同（007 §4），違反 009 的「相近的 Recall@K」。
- **各實作各自 fallback 到 0.90**：一個實作在 0.91、另外兩個在 0.95 取樣，recall 不相近。
- **以跨 seed 的 median 選 ef**：被取樣的是 seed 1 這個 build，用它本身的 recall 才能在取樣時逐位元核對。
- **在每個格點、每個目標都取樣**：成本是數十倍，而 Gate 只需要「時間主要花在哪」。

## 對實驗的影響

- Gate profiling 在**兩個資料集的全部 formal config 都完成之後**進行（ef 從正式 run 選出），使用同一個 heavy lock 的條件。
- 正式報告多出「主要結論附上的 seed 範圍」與「Gate 讀值」兩節；README 的結論依 013 §3 附上這些範圍。
- Gate 的判讀結果與 profiling 摘要記到 NEXT，交由使用者決定是否進 Advanced。

## 初稿的審查與修正（同日，仍在看任何正式效能結果之前）

初稿（commit 7a8c86a，只在分支上）由一位獨立審查者檢查程式與規則，用合成資料重現邊界情況，並做 18 個 mutation，其中 9 個存活。
10 個 findings 全部採用：

- **門檻與理由矛盾（高）**：初稿以 hnswlib ÷ rust 的讀值比值 ≥ 4/3 判為有差距，但梯子有一半的間距是 ×1.5。
  反例：兩者真實速度相同，rust 的交點剛好在 64 之上（讀值取 96）、hnswlib 在 64 達標，讀值比值約 1.43、seed 範圍也分開 → 誤判為有差距。
  改為第 1 節的 QPS 範圍（`qps_bracket`），這個反例現在判為 `within_quantization`（有測試）。
- **`no_gap` 被過度解讀**：量化誤差在兩個方向都有，rust 快 4 倍也會被標成「沒有差距」。改為對稱的 `gap` / `rust_faster` / `within_quantization`，
  並明定這些都不代表「很接近 hnswlib」。
- **缺資料的 variance 格點被靜默略過**、**排除標記後只剩 1 個 build 仍判為分開**、**沒有並列包含標記點的判讀**：
  改為逐一檢查登記的格點、每側至少 3 個未標記的 build、主要與全部 build 兩種判讀不同時並列。
- **`check` 沒有落實 §2**：初稿只核對 search 的 recall 與 lock。改為兩個階段都核對，並檢查輸出檔、參數、git、量測程式、BUILD_INFO、版本；
  `profile.py` 作為證據取樣時拒絕 dirty 的 checkout 與過舊的 binding；計畫拒絕覆寫上一次的輸出。
- **fallback 讓取樣的 recall 不相近**：改為三個實作一起改用 0.90。
- **README 的流程讓 repo 變 dirty**：計畫改寫在 `results/profiles/` 下。
- **只登記了兩種一致的結果**：第 1 節寫明所有組合的意義。
- **呈現**：seed 範圍表標示「不足 k」、註明其他 n 的代理範圍。
- 補上測試，涵蓋存活的 mutation（median 與 min / max 落在目標兩側、邊界相等、錯的 n、缺格點、build 指令的完整參數等），
  修正後的 mutation 都被測試抓到。

## 補記（2026-10-15，任何 Gate profiling 之前）：取樣與正式 run 比對的程式範圍

- §2 原本要求「HEAD 與 formal run 之間的實作與量測程式（`report.py` 的 `MEASUREMENT_PATHS`）沒有差異」。decision 017、018 修改了 runner 的有效性規則
  （`run.py`、`protocol.py`、`environment.py`），GloVe 的 formal run 是在修改之前，照原文 GloVe 的 Gate profiling 永遠無法通過核對。
- 改為比對**被取樣的程式**：`gate_profile.PROFILED_PATHS`（`crates/`、`python/`、`Cargo.toml`、`Cargo.lock`、`requirements-dev.txt`、
  `benchmarks/adapters.py`、`interface.py`、`datasets.py`、`metrics.py`）。runner 的有效性規則不在 profiling 的執行路徑上，不影響「時間花在哪」。
  GloVe formal run（f077de9 起）到 017 之後的 HEAD，這些路徑沒有任何差異（有測試）。
- 電力條件依 017、018（不看電量；70W 或 140W）。
