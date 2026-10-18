# Decision Logs

只有會影響架構、資料格式、評測協定、benchmark 公平性或研究問題的決策才需要寫 decision log。

命名建議：

```text
001-xxxx.md
002-xxxx.md
```

每篇至少包含：

- 背景
- 決策
- 為什麼
- 放棄的替代方案
- 對實驗的影響

## 索引（001–019）

- 標題逐字取自各檔的第一行，日期取自各檔的「日期」行。
- 「補記」一欄列出文末補記標題中的日期；同一天有多則時註明則數。
- decision 的內文為繁體中文。文中的「使用者」指專案作者（the owner）。

| decision | 日期 | 補記 |
| --- | --- | --- |
| [001：Cosine 距離定義、正規化與 tie-break](001-cosine-distance-and-normalization.md) | 2026-10-08 | — |
| [002：統一 Python batch interface 與計時協定](002-python-batch-interface.md) | 2026-10-08 | 2026-10-17 |
| [003：Recall@K 定義（主要指標）與 tie-aware 診斷](003-recall-definition.md) | 2026-10-08 | — |
| [004：HNSW Core 演算法細節與 baseline 表示法](004-hnsw-core-algorithm.md) | 2026-10-08 | — |
| [005：HNSW index 檔案格式（save / load）](005-index-file-format.md) | 2026-10-08 | — |
| [006：HNSW 的 Python binding 與 efSearch 介面](006-hnsw-python-binding.md) | 2026-10-08 | — |
| [007：hnswlib / FAISS adapters 與實測語意](007-third-party-adapters.md) | 2026-10-08 | 2026-10-17 |
| [008：benchmark runner 與量測協定的落實](008-benchmark-runner.md) | 2026-10-08 | 2026-10-09（2 則）、2026-10-17 |
| [009：profiling 方法（Gate 證據的產生方式）](009-profiling-method.md) | 2026-10-08 | 2026-10-08、2026-10-09（2 則）、2026-10-16、2026-10-17 |
| [010：比較規則與 variance 彙整（事先登記）](010-comparison-rules.md) | 2026-10-08 | 2026-10-09（2 則）、2026-10-17 |
| [011：資料集（glove-100-angular 與 EmbeddingGemma 2 文字 workload）](011-datasets.md) | 2026-10-08 | 2026-10-08 |
| [012：正式文字 workload 的規模](012-text-workload-scale.md) | 2026-10-08 | 2026-10-09 |
| [013：正式 run 的格點（formal config）](013-formal-grid.md) | 2026-10-09 | 2026-10-09（4 則）、2026-10-17 |
| [014：正式 run 的電力條件、背景 workload 與 index memory 指標（使用者決定）](014-formal-run-conditions-and-index-memory.md) | 2026-10-09 | 2026-10-09、2026-10-11、2026-10-17 |
| [015：Gate 證據的讀法與 Gate profiling 的參數（事先登記）](015-gate-evidence.md) | 2026-10-09 | 2026-10-15 |
| [016：文字 workload 的電力模式不一致——作廢 text-formal-2、3，全部以高效能模式重跑（使用者決定）](016-text-power-mode-voided-runs.md) | 2026-10-11 | — |
| [017：取消電量下降的作廢規則（使用者決定，協定 008-v3）](017-remove-battery-drop-rule.md) | 2026-10-11 | — |
| [018：允許 Apple 原廠 140W 變壓器（使用者決定，協定 008-v4）](018-allow-140w-adapter.md) | 2026-10-15 | — |
| [019：Advanced C：vector quantization 的假設、表示與評測協定（事先登記）](019-advanced-c-vector-quantization.md) | 2026-10-16 | 2026-10-17（2 則）、2026-10-19 |

閱讀入口：

- 事先登記的比較規則：010。Gate 證據的讀法：015。Advanced C 的假設、協定、結果與封版：019（2026-10-19 的補記）。
- 依 015 事先登記的讀法，Gate 條件 3「同 recall 下與 hnswlib 有差距」不成立。
  2026-10-16 進入 Advanced C，是專案作者依 SPEC 特例所做的決定，不代表 Gate 通過。
