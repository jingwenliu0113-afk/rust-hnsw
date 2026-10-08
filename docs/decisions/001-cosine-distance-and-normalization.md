# 001：Cosine 距離定義、正規化與 tie-break

日期：2026-10-08
狀態：採用

## 背景

SPEC 固定距離為 cosine，但沒有定義：

- distance 的確切公式與範圍。
- 何時正規化、用什麼精度。
- 零向量、NaN / Inf 怎麼處理。
- 同距離（tie）時誰排前面。
- exact search（ground truth）用什麼精度累加。

這些都會直接影響 ground truth 與 Recall，必須在寫任何 benchmark 之前固定。

## 決策

1. **距離公式**：`d(a, b) = 1 − a·b / (‖a‖‖b‖)`，範圍 `[0, 2]`，越小越近。
2. **正規化時機**：向量在 insert 與 query 時各正規化一次。
   norm 與除法用 f64 計算，結果存成 f32 unit vector。
   之後 index 內部距離一律是 `1 − dot(u, v)`。
3. **非法輸入直接拒絕（回傳錯誤）**，不默默丟掉或補值：
   - 零向量（norm 為 0）。
   - 含 NaN / Inf 的向量。
   - 維度不符、dim = 0。
   - 批次寫入時任一 row 不合法，整批都不寫入（atomic）。
4. **ID**：依插入順序從 0 開始連續編號，型別 `u32`。
5. **Exact search（ground truth）**：
   - 在已正規化的 f32 向量上，用 f64 累加 dot。
   - 排序 key 為 `(distance, id)` 升冪；同距離時 id 小者優先。
   - 回傳時距離轉為 f32。
6. **k 的範圍**：exact search 要求 `1 ≤ k ≤ n`，否則回傳錯誤。

## 為什麼

- 與兩個 baseline 的語意一致：
  hnswlib `cosine` space 內部也是先正規化再算 `1 − ip`；
  FAISS 做 cosine 的標準作法是 `normalize_L2` + `METRIC_INNER_PRODUCT`。
- 只正規化一次，避免每次距離計算都重算 norm。
- f64 累加讓 ground truth 少受 rounding 影響，減少「假 tie」。
- 固定 tie-break 讓 ground truth 可重現，人工驗證案例也有唯一正解。
- 明確拒絕零向量：cosine 對零向量沒有定義，任何補值都是任意選擇。

## 放棄的替代方案

- **每次距離計算都算 norm**：較慢，且與 baseline 的作法不同。
- **f32 ground truth**：與 ANN 同精度，但 near-tie 更多，ground truth 較不穩定。
- **零向量定義成距離 1**：結果會依賴一個任意的特例。
- **k > n 時用 -1 補齊**：exact search 中 k > n 屬於誤用，直接報錯較誠實。
  （ANN 實作回傳不足 k 時的補齊規則見 002。）

## 對實驗的影響

- 所有實作看到的是同一批 unit vectors（見 002）。
- ground truth 用 f64、ANN 用 f32，near-tie 附近仍可能有極小差異；
  其對 Recall 的影響用 003 的 tie-aware 診斷量化。
