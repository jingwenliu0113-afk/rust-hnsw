# 012：正式文字 workload 的規模

日期：2026-10-08
狀態：採用（在任何正式結果之前；pilot 只跑過 smoke 模式的 correctness run）

## 背景

011 §3.5 規定 pilot 之後另寫 decision，決定正式文字 workload 的 N 與登記的 n。使用者的規則：先做小型 pilot，不要直接生成 1M embeddings。

pilot（`simplewiki-embeddinggemma2-pilot`，base 10,000、query 1,000）的流程面結果：

- **前處理與切分**：10,000 個 base passage 來自 4,250 篇文章；token 數最多 386、平均約 98；完全相同的文字移除 6 段。
  整個 corpus 有 241,787 篇文章，其中 147,792 篇至少有一個 passage，去重前共 353,615 個 passage，所以 N 的上限約 35 萬。
- **embedding**：batch size 1、fp32、CPU 8 threads（011 補記）。機器 load 約 9–12 時，base 10,000 筆 1,124 秒、query 1,000 筆 108 秒，約 9 筆/秒；
  逐位元重現檢查通過。
- **資料特性**（診斷，不是指標）：query 到最近 base 的 cosine distance 中位數 0.283（第 1 / 99 百分位 0.148 / 0.338），
  沒有任何 query 小於 1e-3，所以沒有近似重複的 self-match。第 10 近與最近距離的比值中位數 1.15（GloVe n = 100k 為 1.17），
  鄰居之間的區分度與 GloVe 相近，不是容易的 workload。
- **correctness run**（smoke，n = 10,000、M = 16、efC = 200、ef 10–256，`results/runs/20261008T143953Z-smoke`）：
  三個實作的 Recall@10 都隨 ef 單調上升（最大相鄰下降 0），ef = 10 約 0.79、ef = 64 約 0.98、ef = 256 約 0.999；
  frozen ground truth 驗證後使用，沒有結果不足 k、沒有重跑不一致。計時是 smoke（load 約 9），不作任何效能結論。

## 決策

- **名稱**：`simplewiki-embeddinggemma2-100k`。
- **規模**：base N = 100,000、query Q = 2,000。前處理、切分、輸入格式、模型與 runtime 完全沿用 011 §3（不變）。
- **登記的 n**（frozen ground truth；scaling with N 的格點）：10,000、30,000、100,000。
- **產生方式**：與 pilot 相同，單一 process 依序產生（`benchmarks.prepare_text --n-base 100000 --n-queries 2000 --gt-n 10000 --gt-n 30000 --gt-n 100000`）。
  資料準備不需要 Mac heavy lock（011 背景）。

## 為什麼

- **N = 100,000**：
  - 768 維 × 100k 的 float32 向量約 307 MB，記憶體規模與 GloVe（100 維）的 300k 相近；大 N 的 scaling 由 GloVe（到 1,183,514）負責。
    文字 workload 回答的是「高維、現代 embedding 分布」下的取捨，10k–100k 一個數量級足以看出 N 的趨勢。
  - embedding 成本：以 pilot 的速度約 3 小時，可以在 heavy lock 被占用時於背景完成。
  - 只用 corpus 約 3 成的 passage，日後若要擴大仍有空間。
- **Q = 2,000**：
  - 768 維的 search 比 GloVe 的 100 維慢，每個 ef 點的計時（008：5 次 QPS 重跑、3 次 latency 重跑，再加 k = 1 search）與 Q 成正比。
    Q = 10,000（與 GloVe 相同）會讓文字 workload 的正式 run 時間多 5 倍。
  - Recall@1 在 0.99 附近的抽樣標準誤：Q = 1,000（pilot）約 0.003，Q = 2,000 約 0.002；Recall@10 的樣本數再多 10 倍（同一 query 內相關，實際較小）。
    0.99 目標附近，Q = 2,000 足以區分相鄰的 ef 點。
- **前綴穩定**：base 依文章雜湊順序排列，embedding 又與批次組成無關（011 補記），所以 Q 不變時，日後以更大的 N 重新準備，
  前 100,000 列會逐位元相同，只需要多算新增的 passage。query 的順序同樣固定：pilot 的 1,000 個 query 就是這裡前 1,000 個 query。

## 放棄的替代方案

- **N = 1,000,000**：使用者明示先不要；Simple English Wikipedia 也只有約 35 萬個 passage。
- **N = 300,000**：embedding 約 9–10 小時，而大 N 的 scaling 已由 GloVe 涵蓋。
- **Q = 1,000（與 pilot 相同）**：0.99 附近的抽樣誤差較大；多 1,000 個 query 的 embedding 成本只有約 2 分鐘。
- **多個 process 平行產生 embedding**：batch size 1 下理論上逐位元相同，但多一層需要驗證的機制；目前的成本可以接受。

## 對實驗的影響

- 正式文字 workload 的結論只代表「Simple English Wikipedia 的 passage、EmbeddingGemma 2 的 768 維 document embedding、held-out 文件查詢」（011）。
- pilot 資料集保留在 registry，只用於 correctness 與流程驗證，不進入正式比較。
- 正式格點（M、efC、ef、variance）另寫 decision。

## 補記（2026-10-09）：產生結果

- 依本文件產生並登記（manifest 在 `benchmarks/manifests/simplewiki-embeddinggemma2-100k.json`）：base 100,000、query 2,000、
  frozen ground truth n = 10,000 / 30,000 / 100,000；完全相同的文字移除 112 段；token 數最多 449、平均約 98。
- embedding：base 約 10,333 秒、query 約 170 秒（batch size 1、fp32、CPU；期間機器 load 約 4–9）。兩個 job 的逐位元重現檢查都通過。
- **前綴穩定已實測**：本資料集的前 1,000 個 query 向量與 pilot 的 1,000 個 query 向量**逐位元相同**，passage id 也相同；
  證實 embedding 與批次組成、job 大小無關（011 補記）。
- correctness smoke（`text-correctness.toml`，n = 100,000、(16, 200)）：三個實作的 Recall@10 都隨 ef 單調上升，
  ef 10 約 0.77、ef 64 約 0.97、ef 192 約 0.993、ef 512 約 0.999；沒有結果不足 k、沒有重跑不一致。計時是 smoke，不作效能結論。
- 中途一次被 session 結束中斷（只寫出 jsonl），從頭重跑。
