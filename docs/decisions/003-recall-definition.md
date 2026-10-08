# 003：Recall@K 定義（主要指標）與 tie-aware 診斷

日期：2026-10-08
狀態：採用（在看到任何 benchmark 結果之前預先登記）

## 背景

SPEC 要求 Recall@1 與 Recall@10。
Recall 有兩種常見定義：

- **id-based**：返回的 id 與 exact top-K 的 id 交集。hnswlib / FAISS 文獻常用。
- **distance-based**：返回項的距離只要不超過 exact 第 K 近的距離（加上容忍值）就算命中。
  ann-benchmarks 用這種，容忍值為 1e-3。

真實資料集常有重複或幾乎重複的向量。
這時 id-based 會因為 tie 而低估 recall：返回了一個同樣近、但 id 不同的點也算錯。

## 決策

### 主要指標：id-based Recall@K

對每個 query：

```text
recall_q = |返回的前 K 個 id ∩ exact 前 K 個 id| / K
```

再對所有 queries 取平均。

- exact 前 K 依 001 的 `(distance, id)` 排序。
- 補齊用的 `-1` 永遠不算命中。
- Recall@1 與 Recall@10 都用此定義。
- **所有主要結論只用這個指標。**

### 次要診斷：tie-aware Recall@K

對每個 query：

```text
threshold_q = exact_dist(query, exact 第 K 近的 id) + eps
recall_q    = |{返回的前 K 個有效 id 中，exact_dist ≤ threshold_q 者}| / K
```

- `eps = 1e-6`（cosine distance 的絕對值）。
- 所有距離都由 harness 用 numpy f64 在 unit vectors 上重新計算，
  **不使用各實作自己回報的距離**，避免不同實作的 f32 rounding 影響判定。
- 一律與主要指標同時報告，只用來量化 tie 對 id-based Recall 的影響，不取代主要指標。

### 資料集檢查

載入真實資料集時，計算並報告完全重複向量的比例。

## 為什麼

- id-based 是 SPEC 字面上的 Recall@K，也是 hnswlib / FAISS 文獻的慣例。
- 只有 id-based 時，tie 造成的誤差無法與真正的搜尋錯誤分開；
  tie-aware 診斷可以把兩者拆開。
- ann-benchmarks 的 1e-3 對 cosine distance 太寬鬆，會把真正的錯誤也算成命中；
  這裡的 eps 只用來吸收不同 f64 計算路徑（例如 numpy matmul 與 Rust 依序累加）的 rounding 差異。

## 放棄的替代方案

- **只用 distance-based 當主要指標**：寬鬆程度取決於 eps，不同 eps 結論不同。
- **只用 id-based、不做診斷**：資料有重複時無法解釋 recall 天花板。
- **使用各實作回報的距離判定**：不同實作的 f32 誤差會讓判定不公平。

## 對實驗的影響

- 圖表與主要結論使用 id-based Recall@K。
- 若某資料集的 tie-aware 與 id-based 差距明顯，需在報告中說明原因（通常是重複向量）。
- 若之後要改動任何 Recall 定義或 eps，必須另寫 decision log，並說明是在看到結果之前還是之後。
