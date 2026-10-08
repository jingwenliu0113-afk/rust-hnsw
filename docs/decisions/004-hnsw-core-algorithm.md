# 004：HNSW Core 演算法細節與 baseline 表示法

日期：2026-10-08
狀態：採用

## 背景

HNSW 原論文（Malkov & Yashunin）有幾處留給實作者決定，hnswlib 與 FAISS 也各有細節差異。
這些選擇會影響 recall 與速度。

研究問題是「哪些實作決策決定 recall / 延遲 / 記憶體取捨」。
因此演算法層面要盡量和 hnswlib 一致，讓同 recall 下的差距主要來自實作與表示法。

## 決策

### 參數

- `M`：layer ≥ 1 每個點最多的鄰居數，必須 `M ≥ 2`（`mL = 1/ln(M)` 需要 `M > 1`）。
- `M0 = 2M`：layer 0 的鄰居上限。
- `efConstruction`：必須 ≥ 1；實際使用 `max(efConstruction, M)`（同 hnswlib）。
- `seed`：level 亂數的種子。

### 層級

- `level = floor(−ln(U) · mL)`，`mL = 1/ln(M)`，`U ∈ (0, 1]`。
- 亂數用 crate 內建的 SplitMix64，以 `seed` 初始化，依插入順序抽取。
- 相同 seed 加上相同插入順序，產生完全相同的 graph。

### 插入（Algorithm 1）

1. 高於新點 level 的各層：greedy search（ef = 1）往下走。
2. 新點 level 以下的各層，由上往下：
   - `search_layer(efConstruction)` 找候選。
   - 用 heuristic 選 **M** 個鄰居（layer 0 也是 M；M0 只是 layer 0 的容量上限，同論文與 hnswlib）。
   - 建立雙向連結。
   - 下一層的 entry 為這一層找到的最近點（同 hnswlib；論文寫法是整個候選集合）。
3. 反向連結：鄰居清單未滿就直接加入；
   已滿則把「現有鄰居 + 新點」以 heuristic 重新挑選 `Mmax` 個（layer 0 為 M0，其他層為 M）。
4. 新點 level 高於目前最高層時，成為新的 entry point。

### 鄰居選擇 heuristic（Algorithm 4，hnswlib 變體）

- 候選依到 base 的距離升冪處理。
- 候選 `c` 若與任一已選點 `s` 滿足 `dist(c, s) < dist(c, base)`，就丟掉。
- 不做 `extendCandidates`、不做 `keepPrunedConnections`（hnswlib 預設皆不做）。
- 候選數 < M 時直接全收，不跑 heuristic（hnswlib 與 FAISS 都有這個 shortcut）。

### 搜尋（Algorithm 2、5）

- 上層用 greedy（ef = 1），layer 0 用 `ef = max(efSearch, k)`，回傳前 k 個。
- `efSearch` 是每次 search 的參數，不是 index 狀態。
- `search_layer` 在「最近的候選比結果中最遠的還遠」時停止。
- 所有比較都以 `(distance, id)` 排序，結果完全決定性。
  hnswlib 部分 priority queue 只比較距離，同距離時的順序沒有明確定義；
  兩者只在距離完全相同（例如重複向量）時可能不同。
- 距離依 001：unit vectors 上的 f32 `1 − dot`。

### k 與結果不足

- 要求 `1 ≤ k ≤ n`，否則回傳錯誤（同 exact）。
- graph 中可到達的點少於 k 時，結果會不足 k。
  Rust 批次結果以 `PADDING_ID = u32::MAX`、`distance = +inf` 補齊；
  Python binding 轉成 002 規定的 `-1` / `+inf`。

### Baseline 表示法（刻意不優化）

- 向量：一個連續的 `Vec<f32>`（row-major unit vectors）。
- graph：每個 node 各自一個 `Vec<Vec<u32>>`（per-node allocation）。
- visited set：每次搜尋新建 std `HashSet<u32>`。
- candidates / results：std `BinaryHeap`。
- 距離：純量 f32 迴圈，不手寫 SIMD（SIMD 屬 Optional）。
- 單執行緒。

## 為什麼

- 演算法細節對齊 hnswlib，同 recall 下的差距才能歸因到實作與表示法。
- Advanced A / B 需要一個**未優化的起點**。
  visited list pool、contiguous graph layout、prefetch 等優化都要等 Gate 通過後，
  先寫 hypothesis，再以 ablation 驗證。
- 決定性（seed + insertion order）是報告 seed / insertion order variance 的前提。
- 自帶 PRNG：外部 crate 升級時，亂數序列可能改變。

## 放棄的替代方案

- **Simple selection（直接取最近 M 個）**：hnswlib / FAISS 預設都用 heuristic，改用 simple 會讓比較混入演算法差異。
- **keepPrunedConnections / extendCandidates**：hnswlib 預設不用。
- **下一層用整個候選集合當 entry（論文寫法）**：與 hnswlib 不一致。
- **一開始就用 visited epoch list 或 arena layout**：baseline 會失去「未優化起點」的角色，
  也違反「optimization 先寫 hypothesis」。
- **用 `rand` crate**：版本更新可能改變亂數序列。

## 對實驗的影響

- 預期 Core 比 hnswlib 慢，尤其是 SipHash 的 `HashSet` visited set 與 per-node allocation。
  這是預期內的 baseline，差距要靠 profiling 解釋（Gate 條件）。
- seed 與 insertion order 都可控，variance 實驗可直接做。
- id 等於插入順序。若 harness 為了 insertion order 實驗打亂資料，必須自行把 id 映射回原始 row，
  才能與 ground truth 比對。

## 實作後的觀察（2026-10-08）

**大量完全重複的向量會讓 layer 0 斷裂。**

- 測試設定：300 個點，5 個方向各重複 60 次，M = 4（M0 = 8）。
  ef = 300 ≥ n 時，從 entry 只到得了 9 個點。
- 機制：
  1. heuristic 的條件是 `dist(c, s) < dist(c, base)`。與 base 完全相同的候選距離為 0，`0 < 0` 不成立，
     所以副本之間永遠不會互相 prune。
  2. 鄰居清單一滿，就被同方向的副本佔滿，跨方向的連結被 prune 掉。
  3. 同距離時依 id 小者優先，所有副本都指向同一批 id 最小的副本，可到達的集合只剩 1 + 8 = 9。
- 原本測試假設「ef ≥ n 時結果等於 exact」。這個假設只在 n ≤ M0 + 1（layer 0 從不 prune）時有保證。
  因此改為只斷言 graph 不變量、結果格式與補齊規則，並在測試註解中說明原因。
- hnswlib 的 heuristic 判斷條件相同（strict `<`），理論上也會被副本佔滿鄰居清單；**尚未實測**。
  - 補記（2026-10-08）：已於 007 實測。同一類資料上，Rust 的可到達點數恆為 2M + 1，FAISS 與 hnswlib 都較多；
    graph 斷裂最嚴重的是本專案的 Rust。Core 階段仍不改演算法。
- 對實驗的意義：真實資料集的重複率（003 已要求報告）不只影響 recall 的計算，也可能影響 graph 連通性。
  載入真實資料集時，要同時報告重複率，並觀察是否出現結果不足 k（padding）的 query。
  Core 階段不針對這點改演算法；若要處理，需另寫 decision log。
