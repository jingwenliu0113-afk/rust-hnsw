# 010：比較規則與 variance 彙整（事先登記）

日期：2026-10-08
狀態：採用（在看到任何正式結果之前預先登記）

## 背景

SPEC 要求「正式 performance comparison 要在相近 Recall@K 下比較」，並報告 scaling with N 與 seed / insertion order variance。
EXPERIMENT_RULES 要求 same recall region、多次重跑、報告 variance。

008 的 runner 只輸出每個量測點的原始數字。從原始數字到「慢 X 倍」這類結論之間，還有很多選擇：
用哪個 recall、目標水準、怎麼從離散的 ef 取點、達不到目標怎麼辦、variance 怎麼彙整。
這些選擇會直接改變結論，所以必須在看到任何正式結果之前固定（008 審查 F07、F08、F12）。

## 決策

### 1. 比較用的 recall

- **id-based Recall@10**（003 的主要指標）。
- 目標水準：**0.90、0.95、0.99**。
- tie-aware Recall@10 與 Recall@1 並列報告，不用於取點。

### 2. 主要比較：recall 對效能的 frontier

- 對每個（資料集、n、實作），以所有量測的 (M, efC, ef) 點組成 frontier。
  - 每個點的數值取**跨 seed 的 median**（只用原始插入順序的 build；見第 4 點）。
  - 參與 median 的是該點 k = 10 status 為 ok 的 build；主要讀值另排除有 hygiene 標記的 build，用剩下的 build 計算 median。
    每個讀值附上參與的 build 數。（2026-10-08 補記，在任何正式結果之前：原文沒有規定部分 seed 被標記或失敗時怎麼算。）
  - 不比較「同一組 (M, efC, ef)」：三者建出的 graph 不等價（007 §4）。同 (M, efC) 的 recall-vs-ef 曲線只作輔助，不用於效能結論。
- 對每個目標水準，在「median Recall@10 ≥ 目標」的點中，分別取：
  - QPS 最高（recall–throughput）；
  - P50 最低、P95 最低（recall–latency）；
  - `index_memory_bytes` 最低（recall–memory，008 §8）。
  每個讀值都附上該點的 (M, efC, ef)。
- **不內插、不外推**。沒有任何點達到目標時記為「未達到」，不得以最接近的點代替。
- 實作間的比值（例如 hnswlib 的 QPS ÷ Rust 的 QPS）只在兩者都達到目標時計算。
- 看到結果之後增加 ef、M、efC 的格點，必須另寫 decision log，並說明是在看到結果之後。

### 3. 結果不足 k 與被標記的點

- Rust 與 FAISS 補 `-1` 的點（`incomplete_queries > 0`）**納入** frontier，並加上標記：`-1` 依 003 已算成未命中，recall 如實反映缺少的結果。
- hnswlib 整批失敗的點沒有 QPS / latency / Recall@10，不存在於 frontier（007 §3）。
- 所有結果不足 k 的點另外列表呈現，並列出三者在同一組參數上的狀態（007 第 3 點的不對稱）。
- 有 hygiene 標記（008 §9：`load`、`thermal`、`swap`、`memory`、`power`、`multithread`、`nondeterministic`；子 process、build、點三層的標記都算）的點
  **不進入**主要比較，另外列表呈現；若排除它們會改變某個目標的讀值，同時報告排除前後的讀值。scaling 的 build 指標同樣處理。
- 失敗或被中止的子 process（008 §5）照實列出；不得以重跑取代已完成的 formal run，除非該 run 為 `invalid`、`aborted` 或 `error`，且要記錄原因。

### 4. seed 與 insertion order variance

- **設計**（008 §3 的 config 展開方式）：
  - seed variance：原始插入順序，`seeds` 中的每個 seed。
  - order variance：`seeds[0]`，加上 `shuffles` 中的每個 permutation。
  - 不做全交叉。三套實作使用同一份清單。
- **最低次數**：每個資料集至少有一組事先列在 formal config 中的 (n, M, efC)，跑 **≥ 5 個 seed** 與 **≥ 5 個 shuffle**；
  其他格點至少 1 個 seed。哪些格點跑完整的 variance，在 formal config commit 時決定，不得事後挑選。
- **彙整**：保留每個 build 的原始值；每個（實作、n、M、efC、ef）的每個指標報告跨 build 的 **median 與 min–max**。
  - QPS 用每個 build 5 次重跑的中位數；P50 / P95 用每個 build 合併 3 次重跑的樣本計算。
  - 同一個 process 內的重跑只反映短期計時雜訊，不得當成實作之間比較的不確定度；實作間比較的不確定度用跨 seed 的範圍。
- **不得只呈現單一 seed 的結果**。主要結論（例如「慢 X 倍」）必須附上 variance 格點上的 seed 範圍。
- FAISS 的 order variance 與另外兩者意義不同（007 §5），分開呈現，不並列比較。

### 5. scaling with N

- 子集依 008 §4（巢狀、同一組 queries、每個 n 各自的 ground truth）。
- 報告兩類指標：
  - 固定 (M, efC) 下的 build time、peak RSS、`index_memory_bytes`。
  - 第 1、2 點規則下、各目標 recall 的 QPS、P50、P95（每個 n 各自讀 frontier）。**不報固定 ef 下的值**：固定 ef 時 recall 會隨 N 改變。

### 6. recall 隨 efSearch 的檢查（SPEC Core 3）

對每個（實作、資料集、n、M、efC），以跨 seed 的 median Recall@10：

- **系統性惡化**：最大 ef 的 recall 低於最小 ef 的 recall。出現時視為 correctness 問題，要先調查，不進入比較。
- 相鄰 ef 之間的下降不要求嚴格單調（SPEC）；報告最大的相鄰下降幅度，不另設門檻。

## 為什麼

- frontier 讀值不需要內插，比「相同 (M, efC) 下比較」少一個自由度，也符合 007 實測的 graph 不等價。
- 目標水準與取點規則在資料集與結果出現之前固定，避免選一個對自己有利的 recall 區間。
- 被標記的點照常呈現、主要比較排除，並在會改變結論時同時報告兩種讀值，避免看到結果後才決定要不要納入。

## 放棄的替代方案

- **在相鄰兩點之間對 (recall, log QPS) 內插**：多一個模型假設，也讓「達到目標」的判定變得不明確。
- **只在相同 (M, efC) 下比較**：三者的 graph 不等價（007 §4）。
- **只使用三者都完整的點**：會丟掉大量有效資料；Rust / FAISS 的 `-1` 已算成未命中，不會偏袒它們。
- **seed × order 全交叉**：成本是 S × P 倍，而兩種 variance 分開量就足以回答 SPEC 的問題。
- **每個指標報 mean ± std**：分布未知、樣本數少；median 與 min–max 搭配原始值更直接。

## 對實驗的影響

- 第一次 formal run 之前，formal config 必須列出完整 variance 的格點與 ef 格點，並 commit。
- 報告的主要圖表依第 2 點產生；任何偏離本文件的分析都必須另寫 decision log，並標明是在看到結果之後。

## 補記（2026-10-09，在任何正式結果之前）：index footprint 的輔助讀值

- 報告在 recall–memory frontier 與 scaling 的表中，另列 `index_footprint_bytes`（phys_footprint 的增量）的讀值，作為輔助欄位。
  原因：macOS 上已釋放的大型配置仍算在 rss（reusable page），Rust 在大 n 時的 `index_memory_bytes` 會被多算約一份資料（NEXT.md B1）。
- 主要讀值不變：仍依 008 §8 / 本文件第 2 點使用 `index_memory_bytes`。是否改用 footprint 由使用者決定（B1），必須在看到任何正式結果之前決定；
  B1 決定之前，不產生、不查看任何 formal 報告（formal run 可以先跑）。決定之後依決定保留或移除這個輔助欄位。

## 補記（2026-10-09）：B1 已由使用者決定

- 上一則補記的 footprint 輔助讀值改為**主要讀值**：recall–memory frontier（第 2 點）與 scaling（第 5 點）的 index memory 以
  `index_footprint_bytes`（phys_footprint 增量）為準，`index_memory_bytes`（rss 增量）改為輔助讀值（decision 014）。
- 第 3 點的 hygiene 標記清單加入 `background`（014 與補記：子 process 期間，以累積 CPU 時間的區間平均判定有背景 heavy workload）。
