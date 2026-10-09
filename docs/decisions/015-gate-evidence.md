# 015：Gate 證據的讀法與 Gate profiling 的參數（事先登記）

日期：2026-10-09
狀態：採用（在看任何正式效能結果之前；是否進 Advanced 仍由使用者決定）

## 背景

SPEC 的 Gate 有四個條件：correctness 通過；Recall 隨 efSearch 整體改善或飽和，沒有系統性惡化；同 recall 下與 hnswlib 有差距；
profiler 已顯示時間主要花在哪。另有一個特例：Core 已很接近 hnswlib 時，不必硬做 optimization attribution，Advanced 改聚焦 vector quantization frontier。

既有的 decision 沒有固定以下兩件事：

- 010、013 規定了 frontier 讀值、QPS 比值與 seed 範圍，但沒有規定**多大的比值算「有差距」、多小算「很接近」**。
- 009 規定了 profiling 的工具與證據條件（正式資料集、heavy lock、相近的 Recall@K），但沒有規定**哪個格點、哪個 recall、ef 怎麼選、取樣多久**。

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

**差距的判讀**（每個資料集各自判讀；主要 n = 100,000）：

- **主要目標：Recall@10 = 0.95**。0.90、0.99 照同樣規則計算並列出，只作輔助。
  理由：0.95 是三個目標的中間值，兩個資料集的格點都預期能達到；GloVe 的 0.99 只有 M48 一個格點，文字的 0.99 不確定（013 §2）。
- 比值 r = hnswlib 的 QPS ÷ rust-hnsw 的 QPS，取 010 §2 的 frontier 讀值（`report.py` 的 `ratios`）。
- seed 範圍：013 §3 的 variance 格點（GloVe 0.90 / 0.95 為 (16, 200) 與 (48, 200)、0.99 為 (48, 200)；文字三個目標都是 (16, 200)），
  各實作在自己的最小達標 ef 上的跨 seed min–max。「分開」= hnswlib 最慢的 seed 仍快於 rust-hnsw 最快的 seed。
- 判讀：
  - **有差距**（`gap`）：r ≥ 4/3，而且每個附上的 variance 格點上 seed 範圍都分開。
  - **沒有超過量化誤差的差距**（`no_gap`）：r < 4/3，包含 rust-hnsw 較快。
  - **不確定**（`uncertain`）：r ≥ 4/3，但某個格點的 seed 範圍重疊，或該格點在 ef 清單內沒有達到目標（013 §3 的 fallback）。
  - **無法判定**（`undetermined`）：任一方沒有達到目標。
- 4/3 的理由：013 §2 的 ef 梯子間距為 ×1.33–1.5，單一 QPS 讀值最多可能低估約 25–33%。比值低於 4/3 時，差距無法與讀值本身的量化誤差區分。
  門檻在看到結果之前固定；不以內插縮小量化誤差（010 §2）。
- 兩個資料集合起來：
  - 都是 `gap`：SPEC 的「同 recall 下與 hnswlib 有差距」成立。
  - 都是 `no_gap`：適用 SPEC 的特例（很接近 hnswlib）。
  - 其他組合：不自行解讀，連同所有讀值交給使用者判斷。
- P50 / P95、index memory、build time 的比值照常列在報告中，作為選擇 Advanced 假設的參考，不用於上述判讀。
  FAISS 的比值同樣列出；SPEC 的 Gate 只以 hnswlib 為準。
- **是否進 Advanced、做哪些假設（最多三個），由使用者決定**。本文件只固定讀法，記到 NEXT 的「需要使用者」。

### 2. Gate profiling 的參數

- **格點**：每個資料集 n = 100,000、(M, efC) = (16, 200)、seed 1、原始插入順序。這是兩個資料集共同的中心格點與完整 variance 格點（013 §1），
  每個 build 的成本也最低。
- **ef**：各實作各自選（007 §4：三者的 graph 不等價，不用相同的 ef）。在正式 run 的這個 build 上，Recall@10 ≥ 0.95 的最小 ef；
  ef 清單內都沒有達到時改用 0.90，並在計畫中註明。用 seed 1 這個 build 本身的 recall，而不是跨 seed 的 median：被取樣的 index 就是這個 build。
  recall 不受 hygiene 標記影響，所以被標記的 build 也照樣用來選 ef。
- **階段與時間**：每個實作取樣 search 30 秒、build 300 秒（`sample` 每 1 ms 一次，009）。build 階段持續重建；依 013 的成本模型，文字 100k (16, 200) 的 build 三個實作合計約 591 秒（負載下上限）。
  單一實作的 build 若超過 300 秒，取樣只涵蓋 build 的前段，照實註明，不延長。
- **順序**：同一資料集先 search、後 build；同一階段依 rust-hnsw、hnswlib、faiss-hnsw。兩個資料集各 6 次取樣。
- **條件**：`./tools/with-mac-lock` 下執行；formal 的電力條件與 load / 背景 gate（014 補記，`profile.py` 的 `evidence_conditions`）；
  git 乾淨、binding 晚於 HEAD；HEAD 與 formal run 之間的實作與量測程式（`report.py` 的 `MEASUREMENT_PATHS`）沒有差異。
- **核對**：取樣前寫出的 JSON 中，search 階段的 ef 與 Recall@10 必須與正式 run 同一個 build 的選定點完全相同
  （同一個 `metrics.recall_at_k`，三個實作的 build 在相同 seed 下都是確定的），而且持有 heavy lock、通過 formal 條件。
  不符時照實記錄，該取樣不作為 Gate 證據，直到查明原因。
- **工具**：`benchmarks/gate_profile.py`。`plan` 從正式 run 選 ef 並產生指令，`check` 做上述核對。
- **呈現**：009 的 top-of-stack（self time）排行，每個實作、每個階段列出前 20 名，原始報告保留在 `results/profiles/`。
  把函式歸類成「距離計算、graph 走訪、visited set、heap、記憶體配置、Python / FFI」等類別是看過取樣之後的解讀，照實標明，原始排行才是證據。
  Rust 的距離計算被 inline 進 `search_layer`（009 已知限制），需要拆開時依 009 另寫 decision。
- 不輸出任何 latency / QPS 數字（009）。

### 3. `report.py` 補上 013 §3 的 seed 範圍與 Gate 讀值

- frontier 的每個讀值附上該點的跨 seed min–max（013 §3 第 1 點）；表格在 build 數 ≥ 2 時列出範圍與 build 數。
- 新的 `variance_ranges`：每個目標、每個 variance 格點、每個實作，在自己的最小達標 ef 上的 Recall@10、QPS、P50、P95 跨 seed min–max；
  格點在 ef 清單內沒有達到目標時改用最大 ef，標明「與讀值不在同一 recall 區間」（013 §3 第 2 點與 fallback）。
  與主要讀值一樣，排除有 hygiene 標記的 build（010 §2）。
- 新的 `gate`：第 1 節的判讀，報告另列一節，標明「建議判讀，是否進 Advanced 由使用者決定」。
- 這些是新增的輸出欄位，不改變任何既有讀值的定義；`report.py` 不在 `MEASUREMENT_PATHS` 中，formal run 之間不受影響。

## 為什麼

- 判讀門檻、格點與 ef 的選法在看到正式結果之前固定，Gate 的結論就不能是看了數字之後挑出來的。
- 差距的門檻綁在 013 §2 已記錄的讀值量化誤差上，不是任意的百分比；同時要求 seed 範圍分開，避免 run 之間的漂移被當成差距。
- profiling 只取一個格點、一個 recall：Gate 要回答的是「時間主要花在哪」，不是 profile 的 scaling；兩個資料集都取，因為維度（100 與 768）會改變距離計算的占比。

## 放棄的替代方案

- **以 10% 作為差距門檻**：小於 013 §2 記錄的讀值量化誤差（25–33%），會把量化誤差當成差距。
- **只看 seed 範圍是否分開，不設比值門檻**：seed 範圍很窄時，幾個百分點的差距也會被判為「有差距」，但那仍在讀值的量化誤差之內。
- **在相鄰 ef 之間內插出相同 recall 的 QPS**：010 §2 不內插。
- **所有實作用相同的 ef 取樣**：三者的 graph 不等價，相同 ef 的 recall 不同（007 §4），違反 009 的「相近的 Recall@K」。
- **以跨 seed 的 median 選 ef**：被取樣的是 seed 1 這個 build，用它本身的 recall 才能在取樣時逐位元核對。
- **在每個格點、每個目標都取樣**：成本是數十倍，而 Gate 只需要「時間主要花在哪」。

## 對實驗的影響

- Gate profiling 在**兩個資料集的全部 formal config 都完成之後**進行（ef 從正式 run 選出），使用同一個 heavy lock 的條件。
- 正式報告多出「主要結論附上的 seed 範圍」與「Gate 讀值」兩節；README 的結論依 013 §3 附上這些範圍。
- Gate 的判讀結果與 profiling 摘要記到 NEXT，交由使用者決定是否進 Advanced。
