# Rust HNSW 實驗規範

## Benchmark Hygiene

正式 benchmark 時：

- 不得同時跑 OTel Demo、training 或其他重負載。
- 記錄：
  - machine
  - CPU
  - RAM
  - OS
  - power state
  - background load
  - compiler version
  - git commit
  - benchmark config
- 若可觀察，記錄 thermal throttling / temperature。

## Fairness

- 三套 ANN implementation 走同一層 Python batch interface。
- same vectors
- same metric
- same k
- same recall region
- same thread count
- warm-up
- 多次重跑

## Dataset

- 真實 ANN dataset 為主。
- 真實 text embedding 為第二組。
- 隨機向量不拿來做主要 transferable claim。

## 報告

- 不只比 latency。
- 必須同時報 recall。
- seed 與 insertion order variance 要報。
- optimization 必須先寫 hypothesis。
- Advanced 做滿三個主要實驗即封版。
