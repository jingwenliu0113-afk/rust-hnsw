# 正式結果（Formal results）

> English: this page transcribes the key tables of the four formal benchmark reports, value for value, so that readers of the public repository can see every verdict. The reports themselves are generated locally and are not in Git; their SHA-256 digests are listed below. The page is in Traditional Chinese, like the decision logs.

本文件把四份正式報告的關鍵內容照原值轉錄，讓沒有 `results/` 的讀者也看得到完整的判定。表中的數字逐字取自報告（含報告的有效位數與 `median [min, max]（builds）` 的寫法）；除了明確標為「換算」的單位換算，沒有重新計算或四捨五入。需要精簡時只省略列或欄，並在表前註明。每一節都標明數字的來源報告。

- 判定規則都在看任何正式效能數字之前登記：比較規則 [010](decisions/010-comparison-rules.md)、Gate 的讀法 [015](decisions/015-gate-evidence.md)（寫於 GloVe 的 formal 1–4 完成之後、看任何效能數字或報告之前，015 照實記錄）、Advanced C 的假設與判定 [019](decisions/019-advanced-c-vector-quantization.md)（在任何量化結果之前）。
- 本文件只轉錄，不改任何判定。報告與本文件不一致時，以報告為準，本文件是錯的（`tests/test_results_doc.py` 在本機有報告時逐項比對）。

## 1. 來源與核對（Sources）

- Core 的兩份報告由 `benchmarks/report.py`、Advanced C 的兩份由 `benchmarks/quant_report.py`，從正式（formal）run 產生。
- `results/` 不進版控（decision [008](decisions/008-benchmark-runner.md) §12；`.gitignore` 排除 `/results/`）。報告與原始 run（`run.json`、每個量測點的檔案）都不在 repo 中；其中 `run.json` 與 `report.json` 含有本機路徑與其他程序的命令列。因此只有 repo 的讀者無法重產這些報告，也無法重產正式數字（數字與機器相關）。
- 下表是本機報告檔的 SHA-256，供持有報告的人核對本文件轉錄的是哪一份。

<!-- check:sha256 -->
| 檔案 File | SHA-256 |
| --- | --- |
| `results/reports/glove-100-angular/report.md` | `83036d815674cb52af2f12964f4d4b3c65fef9313411709b7b13f8212093c58f` |
| `results/reports/glove-100-angular/report.json` | `6a7911d993071f842e6b4f0bb7383688caa3f2f0ec5215b16860f5e33ac81efc` |
| `results/reports/simplewiki-embeddinggemma2-100k/report.md` | `bcb93ff37faf28961f76a265281a7d683638e779a85d17b1a7d336d6dde788f2` |
| `results/reports/simplewiki-embeddinggemma2-100k/report.json` | `2c47be75f7a54f4b8b3a77875227ab808de1805bb6c6c1bbf0f124606bee7c7e` |
| `results/reports/quant-glove/report.md` | `d77058ce700aabbaf94954beed1f217bb6b33faff36450aa57f28d6c9931b15f` |
| `results/reports/quant-glove/report.json` | `0f1030630ae99f58ce8a49fd902f6bdaa29bc785144ce4dbc73758c2d17fe936` |
| `results/reports/quant-text/report.md` | `3675cf6131fad662e670fb27ad758431258bb7a1052c4077a1e025155bbdfccb` |
| `results/reports/quant-text/report.json` | `fe751acbb0d14e10c2d2f260e30e8a0ccaeb1150ac6172170616470ff76c6f92` |

### 1.1 產生報告的 run（取自各報告的「合併的 run」段）

commit 與協定取自各 run 的 `run.json`（metadata 的 git commit；git 狀態都是乾淨的）。

<!-- check:runs -->
| 報告 Report | run | 狀態 Status | config | 協定 Protocol | commit |
| --- | --- | --- | --- | --- | --- |
| glove-100-angular | `20261009T080239Z-formal` | complete | `benchmarks/configs/glove-formal-1-m16-order.toml` | 008-v2 | `f077de9` |
| glove-100-angular | `20261009T091611Z-formal` | complete | `benchmarks/configs/glove-formal-2-m16-seeds-efc400.toml` | 008-v2 | `f077de9` |
| glove-100-angular | `20261009T101347Z-formal` | complete | `benchmarks/configs/glove-formal-3-m48-seeds-1-2.toml` | 008-v2 | `f077de9` |
| glove-100-angular | `20261009T105616Z-formal` | complete | `benchmarks/configs/glove-formal-4-m48-seeds-3-4.toml` | 008-v2 | `f077de9` |
| glove-100-angular | `20261010T021933Z-formal` | complete | `benchmarks/configs/glove-formal-5-m48-seed-5-efc100.toml` | 008-v2 | `ce0317a` |
| glove-100-angular | `20261010T025425Z-formal` | complete | `benchmarks/configs/glove-formal-6-m32-300k.toml` | 008-v2 | `ce0317a` |
| glove-100-angular | `20261010T034635Z-formal` | complete | `benchmarks/configs/glove-formal-7-m8.toml` | 008-v2 | `ce0317a` |
| glove-100-angular | `20261010T040529Z-formal` | complete | `benchmarks/configs/glove-formal-8-full.toml` | 008-v2 | `ce0317a` |
| simplewiki-embeddinggemma2-100k | `20261015T191333Z-formal` | complete | `benchmarks/configs/text-formal-1-m16-order.toml` | 008-v4 | `4acf799` |
| simplewiki-embeddinggemma2-100k | `20261015T210125Z-formal` | complete | `benchmarks/configs/text-formal-2-m16-seeds-efc400.toml` | 008-v4 | `4acf799` |
| simplewiki-embeddinggemma2-100k | `20261015T222221Z-formal` | complete | `benchmarks/configs/text-formal-3-m8-efc100.toml` | 008-v4 | `4acf799` |
| simplewiki-embeddinggemma2-100k | `20261015T225545Z-formal` | complete | `benchmarks/configs/text-formal-4-highm-scaling.toml` | 008-v4 | `4acf799` |
| quant-glove | `20261016T203802Z-formal` | complete | `benchmarks/configs/quant-glove-seed-1.toml` | 008-v5 | `2c4e520` |
| quant-glove | `20261016T211434Z-formal` | complete | `benchmarks/configs/quant-glove-seed-2.toml` | 008-v5 | `2c4e520` |
| quant-glove | `20261016T214152Z-formal` | complete | `benchmarks/configs/quant-glove-seed-3.toml` | 008-v5 | `2c4e520` |
| quant-glove | `20261016T220514Z-formal` | complete | `benchmarks/configs/quant-glove-seed-4.toml` | 008-v5 | `2c4e520` |
| quant-glove | `20261016T221833Z-formal` | complete | `benchmarks/configs/quant-glove-seed-5.toml` | 008-v5 | `2c4e520` |
| quant-text | `20261016T223724Z-formal` | complete | `benchmarks/configs/quant-text-seed-1.toml` | 008-v5 | `2c4e520` |
| quant-text | `20261016T230246Z-formal` | complete | `benchmarks/configs/quant-text-seed-2.toml` | 008-v5 | `2c4e520` |
| quant-text | `20261016T232100Z-formal` | complete | `benchmarks/configs/quant-text-seed-3.toml` | 008-v5 | `2c4e520` |
| quant-text | `20261016T234053Z-formal` | complete | `benchmarks/configs/quant-text-seed-4.toml` | 008-v5 | `2c4e520` |
| quant-text | `20261016T235454Z-formal` | complete | `benchmarks/configs/quant-text-seed-5.toml` | 008-v5 | `2c4e520` |

- Advanced C 的報告另外以 Core 的 4 個 run 作為 C6、C7 的對照：`20261009T080239Z-formal`、`20261009T091611Z-formal`（GloVe）、`20261015T191333Z-formal`、`20261015T210125Z-formal`（文字）。
- 所有正式 run 都在同一台機器上（`run.json`：Apple M4 Pro、macOS 27.0.1、高效能電力模式；GloVe 的 Core run 用 Apple 70W 變壓器，文字與 Advanced C 的 run 用 140W，decision 018）。比較用的 baseline 版本 pin 為 hnswlib 0.8.0、faiss-cpu 1.15.1（decision [007](decisions/007-third-party-adapters.md) §9）。

## 2. 讀法（Glossary）

- **Recall@10**：id-based，主要指標（003、010 §1）。讀值取跨 seed 的 median；不內插、不外推；frontier 讀值是 median Recall@10 ≥ 目標的點中 QPS 最高者（P50、P95、記憶體各自取最佳的點，所以 `@` 後的參數可能不同）。
- **`median [min, max]（N builds）`**：跨 seed 的 median 與範圍；frontier 表在只有一個 build 時不列範圍。
- **index memory**：`phys_footprint` 的增量（decision 014），是主要記憶體指標；Advanced C 稱為 M_014。
- **hygiene 標記**（被標記的點或 build 不進主要讀值，另列「包含被標記點」的讀值，010 §2、§3）：
  - `load`：量測點或 build 結束時，1 分鐘 load average 超過 4.0。
  - `background`：子 process 執行期間，runner 的背景取樣看到其他 process 的持續高 CPU（同一 pid 整段平均 ≥ 50%、同一 command 合計 ≥ 100%，或同一 pid 連續 4 個區間 ≥ 50%；decision 014 補記）。標記只說明「有不屬於本次 run 的負載」，不說明那是什麼工作。
- **Gate 的判讀代碼**（015 §1；Advanced C 的 quantization frontier 沿用同一套規則，019 §6）：
  - `gap`：QPS 範圍分開、hnswlib 較快，而且每個附上的 variance 格點的 seed 範圍都分開；`rust_faster`：方向相反。Advanced C（019 §6）的對應代碼為 `sq8_faster` / `fp32_faster`；`within_quantization`、`uncertain`、`undetermined` 沿用相同代碼。
  - `within_quantization`：兩者的 QPS 範圍重疊，在 **ef 梯子的量化**之內無法區分。這裡的 quantization 指 ef 只量了離散的幾個值，與 Advanced C 的 vector quantization（SQ8）無關。
  - `uncertain`：QPS 範圍分開，但某個 variance 格點的 seed 範圍沒有分開（或資料不足）。
  - `undetermined`：任一方沒有達到目標。
  - 015 明定：`within_quantization`、`uncertain`、`undetermined` 都**不**代表「很接近 hnswlib」。
- **QPS 範圍**（015 §1）：下界是 frontier 讀值；上界是各 (M, efC) 格點第一次達標的 ef 的**前一個** ef 的 QPS（取最大）。
- **Advanced C 的符號**（019 §2）：R = `rust-hnsw`（Core，錨點）、F = `rust-hnsw-fp32g`（FP32，新的搜尋路徑）、S = `rust-hnsw-sq8`（SQ8，與 F 同一張圖）、P = `faiss-hnsw`、Q = `faiss-hnsw-sq8`。e\* 是判定用的 ef（GloVe 304、文字 48）。ρ_eval = ρ_QPS × D_S / D_F，即 S 與 F 每秒完成的距離評估數之比。**乾淨配對**：F 與 S 的子 process、build 與該點都沒有 hygiene 標記。L_total = R_F − R_S、L_trav = T_F − T_S（走訪造成）、L_rank = T_S − R_S（排序造成）、δ_F = T_F − R_F。

## 3. Core：glove-100-angular（d = 100）

來源：`results/reports/glove-100-angular/report.md`（協定 008-v2；run 見 §1.1）。主要 n = 100,000。

### 3.1 Frontier（n = 100,000）

每列照抄報告的 frontier 表；省略輔助的 index RSS 欄。QPS 為每秒查詢數，P50 / P95 為單一查詢 latency。

<!-- check:core-frontier:glove-100-angular -->
| 實作 Implementation | 目標 Target | QPS | P50 (µs) | P95 (µs) | index memory，phys_footprint (MiB) | 排除標記點是否改變讀值 |
| --- | --- | --- | --- | --- | --- | --- |
| faiss-hnsw | 0.90 | 9423 [8787, 9479]（5 builds） @ M=48, efC=200, ef=64 | 114.1 [113.5, 120.3]（5 builds） @ M=48, efC=200, ef=64 | 142.8 [135.4, 164.3]（5 builds） @ M=16, efC=200, ef=128 | 50.9 @ M=8, efC=200, ef=512 | 否 |
| faiss-hnsw | 0.95 | 5650 @ M=32, efC=200, ef=128 | 184.3 @ M=32, efC=200, ef=128 | 222.6 @ M=32, efC=200, ef=128 | 55.8 @ M=16, efC=400, ef=256 | 否 |
| faiss-hnsw | 0.99 | 1654 @ M=32, efC=200, ef=384 | 560.6 @ M=32, efC=200, ef=384 | 701.2 @ M=32, efC=200, ef=384 | 69.6 @ M=32, efC=200, ef=384 | 否 |
| hnswlib | 0.90 | 6168 @ M=16, efC=400, ef=128 | 171.6 @ M=16, efC=400, ef=128 | 220.8 @ M=16, efC=400, ef=128 | 66.9 @ M=16, efC=100, ef=192 | 是 |
| hnswlib | 0.95 | 3135 [3032, 3153]（5 builds） @ M=48, efC=200, ef=128 | 317.8 @ M=16, efC=400, ef=256 | 394.3 @ M=16, efC=400, ef=256 | 66.9 @ M=16, efC=100, ef=384 | 是 |
| hnswlib | 0.99 | 1363 @ M=32, efC=200, ef=384 | 796.8 @ M=32, efC=200, ef=384 | 1064.3 @ M=32, efC=200, ef=384 | 80.3 @ M=32, efC=200, ef=384 | 否 |
| rust-hnsw | 0.90 | 4656 [4011, 4972]（5 builds） @ M=48, efC=200, ef=64 | 218.6 [205.7, 231.6]（5 builds） @ M=48, efC=200, ef=64 | 266.3 @ M=16, efC=400, ef=128 | 102.8 @ M=16, efC=100, ef=192 | 是 |
| rust-hnsw | 0.95 | 2537 [2334, 2706]（5 builds） @ M=48, efC=200, ef=128 | 413.9 [407.8, 450.0]（5 builds） @ M=48, efC=200, ef=128 | 490.9 @ M=16, efC=400, ef=256 | 102.8 @ M=16, efC=100, ef=384 | 是 |
| rust-hnsw | 0.99 | 1008 @ M=32, efC=200, ef=384 | 990.8 @ M=32, efC=200, ef=384 | 1213.2 @ M=32, efC=200, ef=384 | 115.6 @ M=32, efC=200, ef=384 | 否 |

包含被標記點的讀值（只列出與主要讀值不同者，010 §3；n = 100,000）：

<!-- check:core-frontier-flagged:glove-100-angular -->
| 實作 | 目標 | QPS | P50 (µs) | P95 (µs) | index memory (MiB) |
| --- | --- | --- | --- | --- | --- |
| hnswlib | 0.90 | 6168 @ M=16, efC=400, ef=128 | 171.6 @ M=16, efC=400, ef=128 | 220.8 @ M=16, efC=400, ef=128 | 63.2 @ M=8, efC=200, ef=384 |
| hnswlib | 0.95 | 3135 [3032, 3153]（5 builds） @ M=48, efC=200, ef=128 | 317.8 @ M=16, efC=400, ef=256 | 394.3 @ M=16, efC=400, ef=256 | 63.2 @ M=8, efC=200, ef=1024 |
| rust-hnsw | 0.90 | 4656 [4011, 4972]（5 builds） @ M=48, efC=200, ef=64 | 218.6 [205.7, 231.6]（5 builds） @ M=48, efC=200, ef=64 | 266.3 @ M=16, efC=400, ef=128 | 94.0 @ M=8, efC=200, ef=384 |
| rust-hnsw | 0.95 | 2537 [2334, 2706]（5 builds） @ M=48, efC=200, ef=128 | 413.9 [407.8, 450.0]（5 builds） @ M=48, efC=200, ef=128 | 490.9 @ M=16, efC=400, ef=256 | 94.0 @ M=8, efC=200, ef=1024 |

### 3.2 Frontier QPS 點的 Recall@10 與 Recall@1

取自報告的 seed variance 表（原始插入順序），對應上表 QPS 讀值的 (M, efC, ef)；recall 不受 hygiene 標記影響。

<!-- check:core-recall:glove-100-angular -->
| 實作 | 目標 | M | efC | ef | builds | Recall@10 | Recall@1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| faiss-hnsw | 0.90 | 48 | 200 | 64 | 5 | 0.9124 [0.9120, 0.9131] | 0.9465 [0.9447, 0.9477] |
| faiss-hnsw | 0.95 | 32 | 200 | 128 | 1 | 0.9505 [0.9505, 0.9505] | 0.9736 [0.9736, 0.9736] |
| faiss-hnsw | 0.99 | 32 | 200 | 384 | 1 | 0.9922 [0.9922, 0.9922] | 0.9970 [0.9970, 0.9970] |
| hnswlib | 0.90 | 16 | 400 | 128 | 1 | 0.9001 [0.9001, 0.9001] | 0.9316 [0.9316, 0.9316] |
| hnswlib | 0.95 | 48 | 200 | 128 | 5 | 0.9624 [0.9623, 0.9625] | 0.9825 [0.9822, 0.9830] |
| hnswlib | 0.99 | 32 | 200 | 384 | 1 | 0.9921 [0.9921, 0.9921] | 0.9958 [0.9958, 0.9958] |
| rust-hnsw | 0.90 | 48 | 200 | 64 | 5 | 0.9126 [0.9121, 0.9128] | 0.9472 [0.9467, 0.9481] |
| rust-hnsw | 0.95 | 48 | 200 | 128 | 5 | 0.9624 [0.9622, 0.9626] | 0.9828 [0.9824, 0.9832] |
| rust-hnsw | 0.99 | 32 | 200 | 384 | 1 | 0.9921 [0.9921, 0.9921] | 0.9958 [0.9958, 0.9958] |

### 3.3 QPS 比值（相對 rust-hnsw；兩者都達到目標才計算）

<!-- check:core-ratios:glove-100-angular -->
| n | 目標 | faiss-hnsw ÷ rust-hnsw | hnswlib ÷ rust-hnsw |
| --- | --- | --- | --- |
| 100000 | 0.90 | 2.02 | 1.32 |
| 100000 | 0.95 | 2.23 | 1.24 |
| 100000 | 0.99 | 1.64 | 1.35 |
| 300000 | 0.90 | 1.45 | 1.27 |
| 300000 | 0.95 | — | — |
| 300000 | 0.99 | — | — |
| 1183514 | 0.90 | — | — |
| 1183514 | 0.95 | — | — |
| 1183514 | 0.99 | — | — |

### 3.4 Gate 讀值（decision 015；n = 100,000）

照抄報告的 Gate 表。「主要」= 015 登記的主要目標 0.95；0.90、0.99 只作輔助。報告把這張表標為建議判讀；是否進 Advanced 由專案作者決定（015 §1）。

<!-- check:core-gate:glove-100-angular -->
| 目標 | 主要 | rust-hnsw QPS 範圍 | hnswlib QPS 範圍 | variance 格點的 seed 範圍 | 判讀 | 包含被標記 build 的判讀 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.90 | — | [4656, 5923] | [6168, 7725] | (16, 200)：hnswlib ef 152 4825（4 builds） vs rust-hnsw ef 128 5071（5 builds） → 未分開；(48, 200)：hnswlib ef 64 5291（5 builds） vs rust-hnsw ef 48 6318（5 builds） → 未分開 | `uncertain`：不確定（seed 範圍未分開或資料不足） | 相同 |
| 0.95 | 是 | [2537, 3234] | [3135, 4192] | — | `within_quantization`：在 ef 梯子的量化範圍內，無法區分 | 相同 |
| 0.99 | — | [1008, 1606] | [1363, 2349] | — | `within_quantization`：在 ef 梯子的量化範圍內，無法區分 | 相同 |

### 3.5 seed variance：主要結論附上的 seed 範圍（013 §3）

n = 100,000，各實作在自己的最小達標 ef；排除被標記的 build；`median [min, max]`。完整的 seed variance 表（每個 ef）在報告中。

<!-- check:core-seed-ranges:glove-100-angular -->
| 目標 | (M, efC) | 實作 | ef | builds | Recall@10 | QPS | P50 (µs) | P95 (µs) | 與讀值同一 recall 區間 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | (16, 200) | faiss-hnsw | 128 | 5 | 0.9028 [0.9003, 0.9030] | 8934 [7811, 9204] | 118.0 [112.2, 129.5] | 142.8 [135.4, 164.3] | 是 |
| 0.90 | (16, 200) | hnswlib | 152 | 4 | 0.9115 [0.9114, 0.9118] | 5459 [4825, 5999] | 193.3 [169.7, 207.5] | 249.3 [208.9, 271.0] | 是 |
| 0.90 | (16, 200) | rust-hnsw | 152 | 5 | 0.9117 [0.9114, 0.9122] | 4000 [3853, 4398] | 252.7 [233.4, 263.5] | 310.8 [294.2, 324.7] | 是 |
| 0.90 | (48, 200) | faiss-hnsw | 64 | 5 | 0.9124 [0.9120, 0.9131] | 9423 [8787, 9479] | 114.1 [113.5, 120.3] | 145.2 [143.8, 155.8] | 是 |
| 0.90 | (48, 200) | hnswlib | 64 | 5 | 0.9127 [0.9123, 0.9128] | 5663 [5291, 5707] | 184.0 [182.3, 192.8] | 240.5 [237.2, 257.2] | 是 |
| 0.90 | (48, 200) | rust-hnsw | 64 | 5 | 0.9126 [0.9121, 0.9128] | 4656 [4011, 4972] | 218.6 [205.7, 231.6] | 285.4 [268.0, 311.7] | 是 |
| 0.95 | (16, 200) | faiss-hnsw | 304 | 5 | 0.9584 [0.9576, 0.9588] | 3621 [3239, 3741] | 285.6 [276.2, 314.0] | 334.7 [325.0, 382.4] | 是 |
| 0.95 | (16, 200) | hnswlib | 304 | 5 | 0.9567 [0.9567, 0.9568] | 2970 [2778, 3265] | 342.5 [311.9, 347.0] | 420.5 [381.1, 450.9] | 是 |
| 0.95 | (16, 200) | rust-hnsw | 304 | 4 | 0.9568 [0.9565, 0.9570] | 2150 [1992, 2310] | 467.5 [444.1, 513.8] | 538.2 [509.3, 614.8] | 是 |
| 0.95 | (48, 200) | faiss-hnsw | 128 | 5 | 0.9620 [0.9619, 0.9628] | 4901 [4503, 4946] | 216.2 [214.9, 246.8] | 266.0 [263.0, 327.3] | 是 |
| 0.95 | (48, 200) | hnswlib | 128 | 5 | 0.9624 [0.9623, 0.9625] | 3135 [3032, 3153] | 337.1 [331.1, 347.1] | 432.0 [417.2, 442.1] | 是 |
| 0.95 | (48, 200) | rust-hnsw | 128 | 5 | 0.9624 [0.9622, 0.9626] | 2537 [2334, 2706] | 413.9 [407.8, 450.0] | 511.1 [493.2, 585.2] | 是 |
| 0.99 | (48, 200) | faiss-hnsw | 384 | 5 | 0.9951 [0.9950, 0.9954] | 1628 [1482, 1765] | 636.4 [593.0, 663.0] | 749.6 [690.1, 805.5] | 是 |
| 0.99 | (48, 200) | hnswlib | 384 | 5 | 0.9952 [0.9951, 0.9952] | 1196 [1168, 1209] | 876.0 [865.5, 910.5] | 1084.8 [1075.3, 1273.4] | 是 |
| 0.99 | (48, 200) | rust-hnsw | 384 | 5 | 0.9952 [0.9951, 0.9952] | 918 [893, 975] | 1119.2 [1046.3, 1165.0] | 1354.1 [1266.5, 1493.3] | 是 |

包含被標記 build 的 seed 範圍（只列出與上表不同者）：

<!-- check:core-seed-ranges-flagged:glove-100-angular -->
| 目標 | (M, efC) | 實作 | ef | builds | Recall@10 | QPS | P50 (µs) | P95 (µs) | 與讀值同一 recall 區間 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | (16, 200) | hnswlib | 152 | 5 | 0.9114 [0.9114, 0.9118] | 5473 [4825, 5999] | 195.8 [169.7, 207.5] | 250.2 [208.9, 271.0] | 是 |
| 0.95 | (16, 200) | rust-hnsw | 304 | 5 | 0.9567 [0.9565, 0.9570] | 2120 [1954, 2310] | 481.0 [444.1, 513.8] | 555.7 [509.3, 648.6] | 是 |

### 3.6 insertion order variance（(16, 200)、seed 1）

原始順序加 shuffles 101–105，共 6 個 build 的 Recall@10 `median [min, max]`（省略報告中的 QPS 欄）。FAISS 的實際插入順序不是 row 順序（007 §5）：打亂 row 只改變 id 與 level 的對應，意義與另外兩者不同，不並列比較。

<!-- check:core-order:glove-100-angular -->
| ef | rust-hnsw Recall@10 | hnswlib Recall@10 | faiss-hnsw Recall@10（不並列比較） |
| --- | --- | --- | --- |
| 64 | 0.8289 [0.8276, 0.8302] | 0.8291 [0.8281, 0.8294] | 0.8351 [0.8340, 0.8358] |
| 96 | 0.8712 [0.8698, 0.8725] | 0.8711 [0.8695, 0.8716] | 0.8767 [0.8750, 0.8778] |
| 128 | 0.8973 [0.8966, 0.8980] | 0.8971 [0.8966, 0.8977] | 0.9021 [0.9006, 0.9034] |
| 152 | 0.9117 [0.9107, 0.9121] | 0.9113 [0.9101, 0.9121] | 0.9158 [0.9146, 0.9169] |
| 192 | 0.9284 [0.9282, 0.9300] | 0.9284 [0.9279, 0.9299] | 0.9325 [0.9317, 0.9333] |
| 256 | 0.9468 [0.9463, 0.9476] | 0.9469 [0.9462, 0.9476] | 0.9496 [0.9490, 0.9505] |
| 304 | 0.9563 [0.9557, 0.9570] | 0.9563 [0.9556, 0.9569] | 0.9584 [0.9579, 0.9589] |
| 384 | 0.9676 [0.9668, 0.9683] | 0.9678 [0.9667, 0.9684] | 0.9683 [0.9677, 0.9689] |

### 3.7 recall 隨 efSearch（010 §6）

<!-- check:core-recall-vs-ef:glove-100-angular -->
報告的表共 24 列（3 個實作 × 8 個 (n, M, efC)），**每一列都是「系統性惡化：否」、最大相鄰下降 0.0000**。(n, M, efC)：(100000, 8, 200)、(100000, 16, 100)、(100000, 16, 200)、(100000, 16, 400)、(100000, 32, 200)、(100000, 48, 200)、(300000, 16, 200)、(1183514, 16, 200)。

### 3.8 scaling：(M, efC) = (16, 200)

每個 build 計一次、排除被標記的 build。三者建出的 graph 不等價，同 (M, efC) 的 build time 只作描述，不能直接對照（007）。完整的 scaling 表（每個 (M, efC)）在報告中。

<!-- check:core-scaling:glove-100-angular -->
| 實作 | n | build (s) | peak RSS (MiB) | index memory，phys_footprint (MiB) | builds | 被標記的 build |
| --- | --- | --- | --- | --- | --- | --- |
| faiss-hnsw | 100000 | 14.408 [13.797, 15.949] | 157.7 [156.9, 159.6] | 58.2 [55.5, 59.3] | 5 | 0 |
| faiss-hnsw | 300000 | 69.066 [69.066, 69.066] | 354.9 [354.9, 354.9] | 158.4 [158.4, 158.4] | 1 | 0 |
| faiss-hnsw | 1183514 | — | — | — | 0 | 1 |
| hnswlib | 100000 | 22.918 [21.030, 25.076] | 157.7 [157.4, 158.3] | 68.1 [67.6, 68.6] | 5 | 0 |
| hnswlib | 300000 | 96.226 [96.226, 96.226] | 369.1 [369.1, 369.1] | 190.0 [190.0, 190.0] | 1 | 0 |
| hnswlib | 1183514 | — | — | — | 0 | 1 |
| rust-hnsw | 100000 | 31.415 [26.511, 32.803] | 189.6 [188.0, 190.7] | 104.9 [103.4, 105.3] | 5 | 0 |
| rust-hnsw | 300000 | 126.022 [126.022, 126.022] | 465.5 [465.5, 465.5] | 186.3 [186.3, 186.3] | 1 | 0 |
| rust-hnsw | 1183514 | — | — | — | 0 | 1 |

### 3.9 被標記的點與沒有主要讀值的格點

n = 300,000 與 1,183,514 只有 seed 1 的一個 build、(16, 200) 一個格點。1,183,514 的三個 build 全部被標記為 `background`，所以這一格**沒有主要讀值**，只有「包含被標記點」的讀值；300,000 的 rust-hnsw 在 0.95 也是如此（ef 512、768 有 `load`）。

<!-- check:core-other-n:glove-100-angular -->
| 實作 | n | 目標 | 主要讀值 QPS | 排除標記點是否改變讀值 | 包含被標記點的 QPS |
| --- | --- | --- | --- | --- | --- |
| faiss-hnsw | 300000 | 0.90 | 2684 @ M=16, efC=200, ef=256 | 否 | （與主要讀值相同） |
| faiss-hnsw | 300000 | 0.95 | 1311 @ M=16, efC=200, ef=512 | 否 | （與主要讀值相同） |
| faiss-hnsw | 300000 | 0.99 | 未達到 | 否 | （與主要讀值相同） |
| faiss-hnsw | 1183514 | 0.90 | 未達到 | 是 | 1400 @ M=16, efC=200, ef=384 |
| faiss-hnsw | 1183514 | 0.95 | 未達到 | 是 | 484 @ M=16, efC=200, ef=1024 |
| faiss-hnsw | 1183514 | 0.99 | 未達到 | 否 | （與主要讀值相同） |
| hnswlib | 300000 | 0.90 | 2347 @ M=16, efC=200, ef=256 | 否 | （與主要讀值相同） |
| hnswlib | 300000 | 0.95 | 1344 @ M=16, efC=200, ef=512 | 否 | （與主要讀值相同） |
| hnswlib | 300000 | 0.99 | 未達到 | 否 | （與主要讀值相同） |
| hnswlib | 1183514 | 0.90 | 未達到 | 是 | 1401 @ M=16, efC=200, ef=384 |
| hnswlib | 1183514 | 0.95 | 未達到 | 是 | 576 @ M=16, efC=200, ef=1024 |
| hnswlib | 1183514 | 0.99 | 未達到 | 否 | （與主要讀值相同） |
| rust-hnsw | 300000 | 0.90 | 1854 @ M=16, efC=200, ef=256 | 否 | （與主要讀值相同） |
| rust-hnsw | 300000 | 0.95 | 未達到 | 是 | 915 @ M=16, efC=200, ef=512 |
| rust-hnsw | 300000 | 0.99 | 未達到 | 否 | （與主要讀值相同） |
| rust-hnsw | 1183514 | 0.90 | 未達到 | 是 | 1079 @ M=16, efC=200, ef=384 |
| rust-hnsw | 1183514 | 0.95 | 未達到 | 是 | 439 @ M=16, efC=200, ef=1024 |
| rust-hnsw | 1183514 | 0.99 | 未達到 | 否 | （與主要讀值相同） |

被標記的點共 74 個，依 build 整理如下（逐點清單在報告的「被標記的點」段）：

<!-- check:core-flagged:glove-100-angular -->
| 實作 | n | (M, efC) | seed | shuffle | 被標記的 ef 與標記 |
| --- | --- | --- | --- | --- | --- |
| hnswlib | 100000 | (16, 200) | 1 | 104 | ef 128、152、192：`load` |
| faiss-hnsw | 100000 | (16, 200) | 1 | 104 | ef 64、96、128、152：`background`；ef 192、256、304、384：`background`, `load` |
| rust-hnsw | 100000 | (16, 200) | 1 | 104 | ef 64、96、128、152、192、256、304、384：`load` |
| faiss-hnsw | 100000 | (16, 200) | 1 | 105 | ef 304：`load` |
| rust-hnsw | 100000 | (16, 200) | 1 | 105 | ef 64、192、256、304、384：`background`, `load`；ef 96、128、152：`background` |
| rust-hnsw | 100000 | (16, 200) | 5 | 原始 | ef 304：`load` |
| hnswlib | 100000 | (16, 200) | 5 | 原始 | ef 152：`load` |
| rust-hnsw | 100000 | (16, 100) | 1 | 原始 | ef 128：`load` |
| faiss-hnsw | 300000 | (16, 200) | 1 | 原始 | ef 384：`load` |
| rust-hnsw | 300000 | (16, 200) | 1 | 原始 | ef 512、768：`load` |
| rust-hnsw | 100000 | (8, 200) | 1 | 原始 | ef 128、192、256、384、512、768、1024：`background`, `load` |
| hnswlib | 100000 | (8, 200) | 1 | 原始 | ef 128、192、256、384、512、768、1024：`load` |
| faiss-hnsw | 100000 | (8, 200) | 1 | 原始 | ef 128、1024：`load` |
| rust-hnsw | 1183514 | (16, 200) | 1 | 原始 | ef 192、256、384、768、1024、1218：`background`；ef 512、1536：`background`, `load` |
| hnswlib | 1183514 | (16, 200) | 1 | 原始 | ef 192、256、384、512、768、1024、1218、1536：`background`, `load` |
| faiss-hnsw | 1183514 | (16, 200) | 1 | 原始 | ef 192、256、384、768：`background`, `load`；ef 512、1024、1218、1536：`background` |

### 3.10 被排除的 run

<!-- check:core-excluded:glove-100-angular -->
| run | 狀態 | 原因（報告原文） |
| --- | --- | --- |
| `20261009T113943Z-formal` | aborted | aborted: load average above 4.0 or background heavy workload [] for 600 s (008 §2, 014) |

這是 GloVe formal 5 的第一次執行，被 macOS 的背景系統工作（mediaanalysisd）觸發 load gate 中止；整份 config 重跑，資料不合併（013 §5、015 的背景）。

## 4. Core：simplewiki-embeddinggemma2-100k（d = 768）

來源：`results/reports/simplewiki-embeddinggemma2-100k/report.md`（協定 008-v4；run 見 §1.1）。主要 n = 100,000。

### 4.1 Frontier（n = 100,000）

每列照抄報告的 frontier 表；省略輔助的 index RSS 欄。QPS 為每秒查詢數，P50 / P95 為單一查詢 latency。

<!-- check:core-frontier:simplewiki-embeddinggemma2-100k -->
| 實作 Implementation | 目標 Target | QPS | P50 (µs) | P95 (µs) | index memory，phys_footprint (MiB) | 排除標記點是否改變讀值 |
| --- | --- | --- | --- | --- | --- | --- |
| faiss-hnsw | 0.90 | 10290 @ M=8, efC=200, ef=48 | 98.6 @ M=8, efC=200, ef=48 | 129.3 @ M=8, efC=200, ef=48 | 306.4 @ M=8, efC=200, ef=48 | 否 |
| faiss-hnsw | 0.95 | 6811 @ M=16, efC=100, ef=48 | 151.6 @ M=16, efC=100, ef=48 | 182.0 @ M=16, efC=100, ef=48 | 306.4 @ M=8, efC=200, ef=96 | 否 |
| faiss-hnsw | 0.99 | 2692 [2689, 2699]（4 builds） @ M=16, efC=200, ef=128 | 380.1 [379.8, 380.2]（4 builds） @ M=16, efC=200, ef=128 | 436.6 [435.7, 437.2]（4 builds） @ M=16, efC=200, ef=128 | 306.4 @ M=8, efC=200, ef=384 | 是 |
| hnswlib | 0.90 | 3509 @ M=8, efC=200, ef=64 | 284.8 @ M=8, efC=200, ef=64 | 361.8 @ M=8, efC=200, ef=64 | 316.4 @ M=8, efC=200, ef=64 | 否 |
| hnswlib | 0.95 | 2541 @ M=16, efC=100, ef=48 | 400.3 @ M=16, efC=100, ef=48 | 489.8 @ M=16, efC=100, ef=48 | 316.4 @ M=8, efC=200, ef=128 | 否 |
| hnswlib | 0.99 | 844 @ M=32, efC=200, ef=96 | 1203.2 @ M=32, efC=200, ef=96 | 1468.3 @ M=32, efC=200, ef=96 | 316.4 @ M=8, efC=200, ef=384 | 否 |
| rust-hnsw | 0.90 | 3203 @ M=16, efC=100, ef=32 | 310.6 @ M=16, efC=100, ef=32 | 397.2 @ M=8, efC=200, ef=64 | 310.4 @ M=8, efC=200, ef=64 | 否 |
| rust-hnsw | 0.95 | 2387 @ M=16, efC=100, ef=48 | 422.9 @ M=16, efC=100, ef=48 | 520.4 @ M=16, efC=100, ef=48 | 310.4 @ M=8, efC=200, ef=128 | 否 |
| rust-hnsw | 0.99 | 934 @ M=48, efC=200, ef=64 | 1082.6 @ M=48, efC=200, ef=64 | 1358.1 @ M=48, efC=200, ef=64 | 310.4 @ M=8, efC=200, ef=384 | 否 |

包含被標記點的讀值（只列出與主要讀值不同者，010 §3；n = 100,000）：

<!-- check:core-frontier-flagged:simplewiki-embeddinggemma2-100k -->
| 實作 | 目標 | QPS | P50 (µs) | P95 (µs) | index memory (MiB) |
| --- | --- | --- | --- | --- | --- |
| faiss-hnsw | 0.99 | 2690 [2458, 2699]（5 builds） @ M=16, efC=200, ef=128 | 380.2 [379.8, 415.2]（5 builds） @ M=16, efC=200, ef=128 | 437.0 [435.7, 492.0]（5 builds） @ M=16, efC=200, ef=128 | 306.4 @ M=8, efC=200, ef=384 |

### 4.2 Frontier QPS 點的 Recall@10 與 Recall@1

取自報告的 seed variance 表（原始插入順序），對應上表 QPS 讀值的 (M, efC, ef)；recall 不受 hygiene 標記影響。

<!-- check:core-recall:simplewiki-embeddinggemma2-100k -->
| 實作 | 目標 | M | efC | ef | builds | Recall@10 | Recall@1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| faiss-hnsw | 0.90 | 8 | 200 | 48 | 1 | 0.9057 [0.9057, 0.9057] | 0.9335 [0.9335, 0.9335] |
| faiss-hnsw | 0.95 | 16 | 100 | 48 | 1 | 0.9552 [0.9552, 0.9552] | 0.9740 [0.9740, 0.9740] |
| faiss-hnsw | 0.99 | 16 | 200 | 128 | 5 | 0.9906 [0.9900, 0.9909] | 0.9965 [0.9950, 0.9970] |
| hnswlib | 0.90 | 8 | 200 | 64 | 1 | 0.9225 [0.9225, 0.9225] | 0.9525 [0.9525, 0.9525] |
| hnswlib | 0.95 | 16 | 100 | 48 | 1 | 0.9519 [0.9519, 0.9519] | 0.9710 [0.9710, 0.9710] |
| hnswlib | 0.99 | 32 | 200 | 96 | 1 | 0.9932 [0.9932, 0.9932] | 0.9960 [0.9960, 0.9960] |
| rust-hnsw | 0.90 | 16 | 100 | 32 | 1 | 0.9221 [0.9221, 0.9221] | 0.9520 [0.9520, 0.9520] |
| rust-hnsw | 0.95 | 16 | 100 | 48 | 1 | 0.9516 [0.9516, 0.9516] | 0.9710 [0.9710, 0.9710] |
| rust-hnsw | 0.99 | 48 | 200 | 64 | 1 | 0.9902 [0.9902, 0.9902] | 0.9935 [0.9935, 0.9935] |

### 4.3 QPS 比值（相對 rust-hnsw；兩者都達到目標才計算）

<!-- check:core-ratios:simplewiki-embeddinggemma2-100k -->
| n | 目標 | faiss-hnsw ÷ rust-hnsw | hnswlib ÷ rust-hnsw |
| --- | --- | --- | --- |
| 10000 | 0.90 | 4.52 | 1.03 |
| 10000 | 0.95 | 4.28 | 1.04 |
| 10000 | 0.99 | 3.94 | 1.05 |
| 30000 | 0.90 | 3.05 | 1.06 |
| 30000 | 0.95 | 2.89 | 1.06 |
| 30000 | 0.99 | 3.51 | 1.07 |
| 100000 | 0.90 | 3.21 | 1.10 |
| 100000 | 0.95 | 2.85 | 1.06 |
| 100000 | 0.99 | 2.88 | 0.90 |

### 4.4 Gate 讀值（decision 015；n = 100,000）

照抄報告的 Gate 表。「主要」= 015 登記的主要目標 0.95；0.90、0.99 只作輔助。報告把這張表標為建議判讀；是否進 Advanced 由專案作者決定（015 §1）。

<!-- check:core-gate:simplewiki-embeddinggemma2-100k -->
| 目標 | 主要 | rust-hnsw QPS 範圍 | hnswlib QPS 範圍 | variance 格點的 seed 範圍 | 判讀 | 包含被標記 build 的判讀 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.90 | — | [3203, 3920] | [3509, 4243] | — | `within_quantization`：在 ef 梯子的量化範圍內，無法區分 | 相同 |
| 0.95 | 是 | [2387, 3203] | [2541, 3401] | — | `within_quantization`：在 ef 梯子的量化範圍內，無法區分 | 相同 |
| 0.99 | — | [934, 1171] | [844, 1176] | — | `within_quantization`：在 ef 梯子的量化範圍內，無法區分 | 相同 |

### 4.5 seed variance：主要結論附上的 seed 範圍（013 §3）

n = 100,000，各實作在自己的最小達標 ef；排除被標記的 build；`median [min, max]`。完整的 seed variance 表（每個 ef）在報告中。

<!-- check:core-seed-ranges:simplewiki-embeddinggemma2-100k -->
| 目標 | (M, efC) | 實作 | ef | builds | Recall@10 | QPS | P50 (µs) | P95 (µs) | 與讀值同一 recall 區間 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | (16, 200) | faiss-hnsw | 32 | 4 | 0.9276 [0.9245, 0.9304] | 8703 [8648, 8712] | 118.2 [117.8, 118.4] | 145.7 [144.2, 147.9] | 是 |
| 0.90 | (16, 200) | hnswlib | 32 | 5 | 0.9244 [0.9236, 0.9254] | 3276 [3266, 3281] | 305.6 [304.2, 306.1] | 382.8 [380.3, 388.4] | 是 |
| 0.90 | (16, 200) | rust-hnsw | 32 | 4 | 0.9240 [0.9235, 0.9255] | 3103 [3084, 3110] | 321.9 [320.3, 324.3] | 406.4 [403.8, 416.8] | 是 |
| 0.95 | (16, 200) | faiss-hnsw | 48 | 4 | 0.9582 [0.9540, 0.9583] | 6282 [6248, 6299] | 163.1 [162.7, 163.8] | 192.4 [191.0, 193.2] | 是 |
| 0.95 | (16, 200) | hnswlib | 48 | 5 | 0.9538 [0.9528, 0.9546] | 2439 [2303, 2445] | 413.6 [412.0, 428.7] | 503.5 [500.8, 523.4] | 是 |
| 0.95 | (16, 200) | rust-hnsw | 48 | 4 | 0.9542 [0.9537, 0.9548] | 2299 [2285, 2301] | 439.6 [438.4, 441.2] | 529.3 [525.8, 534.3] | 是 |
| 0.99 | (16, 200) | faiss-hnsw | 128 | 4 | 0.9906 [0.9900, 0.9909] | 2692 [2689, 2699] | 380.1 [379.8, 380.2] | 436.6 [435.7, 437.2] | 是 |
| 0.99 | (16, 200) | hnswlib | 192 | 5 | 0.9929 [0.9929, 0.9930] | 799 [746, 799] | 1271.8 [1270.4, 1331.2] | 1532.6 [1530.9, 1630.7] | 是 |
| 0.99 | (16, 200) | rust-hnsw | 192 | 4 | 0.9930 [0.9929, 0.9930] | 743 [743, 744] | 1362.2 [1359.9, 1363.0] | 1640.0 [1635.5, 1646.2] | 是 |

包含被標記 build 的 seed 範圍（只列出與上表不同者）：

<!-- check:core-seed-ranges-flagged:simplewiki-embeddinggemma2-100k -->
| 目標 | (M, efC) | 實作 | ef | builds | Recall@10 | QPS | P50 (µs) | P95 (µs) | 與讀值同一 recall 區間 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | (16, 200) | faiss-hnsw | 32 | 5 | 0.9273 [0.9245, 0.9304] | 8697 [7925, 8712] | 118.4 [117.8, 129.4] | 146.1 [144.2, 161.8] | 是 |
| 0.90 | (16, 200) | rust-hnsw | 32 | 5 | 0.9244 [0.9235, 0.9255] | 3101 [2835, 3110] | 322.3 [320.3, 349.1] | 407.4 [403.8, 443.9] | 是 |
| 0.95 | (16, 200) | faiss-hnsw | 48 | 5 | 0.9581 [0.9540, 0.9583] | 6276 [5750, 6299] | 163.3 [162.7, 178.5] | 193.0 [191.0, 215.4] | 是 |
| 0.95 | (16, 200) | rust-hnsw | 48 | 5 | 0.9538 [0.9537, 0.9548] | 2298 [2103, 2301] | 439.9 [438.4, 476.5] | 531.9 [525.8, 583.3] | 是 |
| 0.99 | (16, 200) | faiss-hnsw | 128 | 5 | 0.9906 [0.9900, 0.9909] | 2690 [2458, 2699] | 380.2 [379.8, 415.2] | 437.0 [435.7, 492.0] | 是 |
| 0.99 | (16, 200) | rust-hnsw | 192 | 5 | 0.9930 [0.9929, 0.9930] | 743 [677, 744] | 1362.5 [1359.9, 1493.6] | 1642.0 [1635.5, 1794.4] | 是 |

### 4.6 insertion order variance（(16, 200)、seed 1）

原始順序加 shuffles 101–105，共 6 個 build 的 Recall@10 `median [min, max]`（省略報告中的 QPS 欄）。FAISS 的實際插入順序不是 row 順序（007 §5）：打亂 row 只改變 id 與 level 的對應，意義與另外兩者不同，不並列比較。

<!-- check:core-order:simplewiki-embeddinggemma2-100k -->
| ef | rust-hnsw Recall@10 | hnswlib Recall@10 | faiss-hnsw Recall@10（不並列比較） |
| --- | --- | --- | --- |
| 24 | 0.8960 [0.8917, 0.8983] | 0.8956 [0.8910, 0.8988] | 0.9002 [0.8985, 0.9029] |
| 32 | 0.9239 [0.9214, 0.9244] | 0.9233 [0.9215, 0.9251] | 0.9277 [0.9263, 0.9287] |
| 48 | 0.9534 [0.9503, 0.9548] | 0.9527 [0.9502, 0.9543] | 0.9568 [0.9549, 0.9577] |
| 64 | 0.9668 [0.9643, 0.9683] | 0.9673 [0.9647, 0.9676] | 0.9701 [0.9694, 0.9710] |
| 96 | 0.9817 [0.9809, 0.9827] | 0.9816 [0.9810, 0.9821] | 0.9839 [0.9834, 0.9842] |
| 128 | 0.9875 [0.9871, 0.9883] | 0.9872 [0.9866, 0.9882] | 0.9902 [0.9892, 0.9906] |
| 192 | 0.9934 [0.9930, 0.9941] | 0.9935 [0.9930, 0.9940] | 0.9954 [0.9941, 0.9956] |
| 256 | 0.9959 [0.9955, 0.9964] | 0.9959 [0.9957, 0.9964] | 0.9970 [0.9967, 0.9976] |
| 384 | 0.9981 [0.9979, 0.9982] | 0.9981 [0.9978, 0.9983] | 0.9987 [0.9983, 0.9990] |
| 512 | 0.9989 [0.9988, 0.9991] | 0.9989 [0.9987, 0.9991] | 0.9992 [0.9991, 0.9994] |

### 4.7 recall 隨 efSearch（010 §6）

<!-- check:core-recall-vs-ef:simplewiki-embeddinggemma2-100k -->
報告的表共 24 列（3 個實作 × 8 個 (n, M, efC)），**每一列都是「系統性惡化：否」、最大相鄰下降 0.0000**。(n, M, efC)：(10000, 16, 200)、(30000, 16, 200)、(100000, 8, 200)、(100000, 16, 100)、(100000, 16, 200)、(100000, 16, 400)、(100000, 32, 200)、(100000, 48, 200)。

### 4.8 scaling：(M, efC) = (16, 200)

每個 build 計一次、排除被標記的 build。三者建出的 graph 不等價，同 (M, efC) 的 build time 只作描述，不能直接對照（007）。完整的 scaling 表（每個 (M, efC)）在報告中。

<!-- check:core-scaling:simplewiki-embeddinggemma2-100k -->
| 實作 | n | build (s) | peak RSS (MiB) | index memory，phys_footprint (MiB) | builds | 被標記的 build |
| --- | --- | --- | --- | --- | --- | --- |
| faiss-hnsw | 10000 | 2.227 [2.227, 2.227] | 119.8 [119.8, 119.8] | 31.3 [31.3, 31.3] | 1 | 0 |
| faiss-hnsw | 30000 | 11.962 [11.962, 11.962] | 242.5 [242.5, 242.5] | 94.9 [94.9, 94.9] | 1 | 0 |
| faiss-hnsw | 100000 | 56.598 [56.586, 56.693] | 668.4 [668.0, 669.2] | 312.5 [311.0, 313.5] | 4 | 1 |
| hnswlib | 10000 | 9.015 [9.015, 9.015] | 110.2 [110.2, 110.2] | 32.4 [32.4, 32.4] | 1 | 0 |
| hnswlib | 30000 | 36.301 [36.301, 36.301] | 237.5 [237.5, 237.5] | 101.0 [101.0, 101.0] | 1 | 0 |
| hnswlib | 100000 | 147.692 [147.534, 148.921] | 667.4 [666.9, 667.5] | 323.3 [323.1, 324.1] | 5 | 0 |
| rust-hnsw | 10000 | 9.467 [9.467, 9.467] | 135.9 [135.9, 135.9] | 62.1 [62.1, 62.1] | 1 | 0 |
| rust-hnsw | 30000 | 37.893 [37.893, 37.893] | 317.5 [317.5, 317.5] | 97.5 [97.5, 97.5] | 1 | 0 |
| rust-hnsw | 100000 | 154.660 [154.594, 155.044] | 950.5 [950.3, 950.8] | 318.8 [318.6, 319.2] | 4 | 1 |

### 4.9 被標記的點與其他 n

文字資料集的每個格點都有主要讀值。n = 10,000 與 30,000 只有 seed 1、(16, 200)，frontier 讀值如下（沒有被標記的點）：

<!-- check:core-other-n:simplewiki-embeddinggemma2-100k -->
| 實作 | n | 目標 | 主要讀值 QPS | 排除標記點是否改變讀值 | 包含被標記點的 QPS |
| --- | --- | --- | --- | --- | --- |
| faiss-hnsw | 10000 | 0.90 | 24274 @ M=16, efC=200, ef=24 | 否 | （與主要讀值相同） |
| faiss-hnsw | 10000 | 0.95 | 13779 @ M=16, efC=200, ef=48 | 否 | （與主要讀值相同） |
| faiss-hnsw | 10000 | 0.99 | 7400 @ M=16, efC=200, ef=96 | 否 | （與主要讀值相同） |
| faiss-hnsw | 30000 | 0.90 | 13205 @ M=16, efC=200, ef=24 | 否 | （與主要讀值相同） |
| faiss-hnsw | 30000 | 0.95 | 7586 @ M=16, efC=200, ef=48 | 否 | （與主要讀值相同） |
| faiss-hnsw | 30000 | 0.99 | 4153 @ M=16, efC=200, ef=96 | 否 | （與主要讀值相同） |
| hnswlib | 10000 | 0.90 | 5524 @ M=16, efC=200, ef=24 | 否 | （與主要讀值相同） |
| hnswlib | 10000 | 0.95 | 3352 @ M=16, efC=200, ef=48 | 否 | （與主要讀值相同） |
| hnswlib | 10000 | 0.99 | 1982 @ M=16, efC=200, ef=96 | 否 | （與主要讀值相同） |
| hnswlib | 30000 | 0.90 | 4572 @ M=16, efC=200, ef=24 | 否 | （與主要讀值相同） |
| hnswlib | 30000 | 0.95 | 2769 @ M=16, efC=200, ef=48 | 否 | （與主要讀值相同） |
| hnswlib | 30000 | 0.99 | 1266 @ M=16, efC=200, ef=128 | 否 | （與主要讀值相同） |
| rust-hnsw | 10000 | 0.90 | 5372 @ M=16, efC=200, ef=24 | 否 | （與主要讀值相同） |
| rust-hnsw | 10000 | 0.95 | 3217 @ M=16, efC=200, ef=48 | 否 | （與主要讀值相同） |
| rust-hnsw | 10000 | 0.99 | 1880 @ M=16, efC=200, ef=96 | 否 | （與主要讀值相同） |
| rust-hnsw | 30000 | 0.90 | 4325 @ M=16, efC=200, ef=24 | 否 | （與主要讀值相同） |
| rust-hnsw | 30000 | 0.95 | 2622 @ M=16, efC=200, ef=48 | 否 | （與主要讀值相同） |
| rust-hnsw | 30000 | 0.99 | 1185 @ M=16, efC=200, ef=128 | 否 | （與主要讀值相同） |

被標記的點共 24 個，依 build 整理如下：

<!-- check:core-flagged:simplewiki-embeddinggemma2-100k -->
| 實作 | n | (M, efC) | seed | shuffle | 被標記的 ef 與標記 |
| --- | --- | --- | --- | --- | --- |
| rust-hnsw | 100000 | (16, 200) | 1 | 原始 | ef 24、32、48、64、96、128、192、256、384、512：`background` |
| faiss-hnsw | 100000 | (16, 200) | 1 | 原始 | ef 24、32、48、64、96、128、256、384、512：`background`；ef 192：`background`, `load` |
| hnswlib | 100000 | (16, 200) | 1 | 105 | ef 256：`load` |
| faiss-hnsw | 100000 | (8, 200) | 1 | 原始 | ef 512、768：`load` |
| hnswlib | 100000 | (32, 200) | 1 | 原始 | ef 192：`load` |

### 4.10 被排除的 run

共 7 個 run：4 個 aborted / invalid，3 個 complete 但依 decision 016、017 登記作廢（`report.py` 的 `VOIDED_RUNS`）。原因照報告轉述成中文；第 2 列報告原文另列出兩個背景程序的命令列（含本機路徑），這裡省略。

<!-- check:core-excluded:simplewiki-embeddinggemma2-100k -->
| run | 狀態 | 處理 | 原因 |
| --- | --- | --- | --- |
| `20261010T083544Z-formal` | aborted | 排除 | `g004-rust-hnsw: aborted`；收到 SIGTERM（NEXT.md 記為專案作者暫停全部工作時中止） |
| `20261010T133129Z-formal` | aborted | 排除 | load average 超過 4.0 或有背景 heavy workload，持續 600 s（008 §2、014）；NEXT.md 記為另一個專案的兩個 worker |
| `20261010T202624Z-formal` | invalid | 排除 | formal 複查失敗：run 期間電量從 98% 降到 95%（當時的上限 3 點），run 作廢 |
| `20261011T033341Z-formal` | invalid | 排除 | formal 複查失敗：run 期間電量從 100% 降到 97%（當時的上限 3 點），run 作廢 |
| `20261010T183115Z-formal` | complete | 作廢（voided） | decision 016：text-formal-1 之後，AC 電力模式在本專案之外由高效能（2）被改成自動（0）；文字 workload 維持高效能模式 |
| `20261010T195251Z-formal` | complete | 作廢（voided） | decision 016：同一個外部改動之後，在自動模式（0）下執行 |
| `20261010T112849Z-formal` | complete | 作廢（voided） | decision 017：在協定 008-v2（電量下降規則）下執行；專案作者取消了該規則，文字 workload 以之後的協定（008-v4，decision 018）重跑，一份報告不能合併不同版本 |

## 5. Gate 四條件總表

來源：`NEXT.md` 的「Gate 的四個條件」與 decision 015 §1 的讀法；本表只轉錄，不改任何判定。條件 3 的讀值見 §3.4、§4.4。

<!-- check:gate-summary -->
| SPEC 的條件 | 讀值（015 §1） | 判定 |
| --- | --- | --- |
| 1. correctness 通過 | `./scripts/test.sh` 全部通過；真實資料的 correctness run 通過 | 成立 |
| 2. Recall 隨 efSearch 整體改善 / 飽和，沒有系統性惡化 | 兩個資料集、每個 (n, M, efC) 都沒有系統性惡化，最大相鄰下降 0.0000（§3.7、§4.7） | 成立 |
| 3. 同 recall 下與 hnswlib 有差距 | 主要目標 0.95：GloVe `within_quantization`、文字 `within_quantization`。輔助：GloVe 0.90 `uncertain`、0.99 `within_quantization`；文字 0.90、0.99 都是 `within_quantization`。沒有任何資料集是 `gap` | **不成立** |
| 4. profiler 已顯示時間主要花在哪 | 兩個資料集各 6 次取樣（3 個實作 × search / build），12 次全部通過 `gate_profile check`（NEXT.md 的摘要：rust-hnsw 在 100 維時約 4 成時間在 `HashMap`（SipHash）的 visited set〔程式中是 `std::collections::HashSet<u32>`，內部為 hashbrown 的 `HashMap`，004〕，768 維時距離計算占絕大部分；函式的分類是看過取樣之後的解讀，015 §2） | 成立 |

- **Gate 沒有通過**：依事先登記的 015，條件 3 不成立。「在 ef 梯子的量化範圍內無法區分」不等於「很接近 hnswlib」。
- **進入 Advanced C 是專案作者的決定**：2026-10-16，專案作者依 SPEC 的特例（「若 Core 已很接近 hnswlib：不必硬做 optimization attribution，Advanced 改聚焦 vector quantization frontier」）決定進入 Advanced C。這**不代表 Gate 通過**；Core 的正式結果、Gate 判讀與事先登記的比較規則都沒有修改（NEXT.md）。
- 看到結果之後的觀察（NEXT.md 記錄；**不是**事先登記的判讀）：GloVe 的 variance 格點上，rust-hnsw 與 hnswlib 在同一個 ef 達到目標、recall 相同到小數第 3–4 位，hnswlib 的 seed 範圍都整段高於 rust-hnsw，median 比值 1.22–1.38；文字上 hnswlib 只快 6–8%。015 的 QPS 範圍規則刻意保守，所以沒有判為 `gap`。

## 6. Advanced C：vector quantization（decision 019）

來源：`results/reports/quant-glove/report.md` 與 `results/reports/quant-text/report.md`（協定 008-v5；run 見 §1.1），以及 019 的完成補記（2026-10-19）。

- 範圍：只限 Rust 的依序純量 kernel、搜尋期間的表示、n = 100,000、(M, efC) = (16, 200)、原始插入順序；build time 與 peak RSS 不會下降（SQ8 先建 FP32 的圖再轉換，019 §1）。
- seeds：F、S 為 1–5；P、Q 為 1–3；R 只有 seed 1（錨點）。10 份 formal config 全部 complete，沒有任何 run 被排除。
- C 用了 3 個假設名額中的 2 個（H1、H2），之後封版；之後的實驗必須另寫 decision，不得取代 019 的判定。

### 6.1 完整判定表（兩個資料集共用，019 §6）

列出報告的全部 37 列，包括無法判定的列。「狀態」逐字照報告；「數字」是報告「判定所用的數字」的精簡版，取報告中已經格式化的值（例如 §6.2 表的 4 位小數），沒有另外取位。

<!-- check:quant-judgement -->
| 列 Row | 狀態 Status | 說明 Note | 數字（精簡） Numbers |
| --- | --- | --- | --- |
| H1 前提（反組譯） | 成立 |  | FP32 kernel：累加暫存器 `s0`、累加 21 次、沒有 violation；SQ8 kernel：`s1`、69 次、沒有 violation（加總都是依序的純量相依鏈） |
| H1a 文字 | 成立 |  | e\* = 48，乾淨配對 5 個，ρ_eval 0.9806、1.0924、0.9615、1.0404、0.9613（seeds 1–5）；成立區 [0.90, 1.10] |
| H1b GloVe | 無法判定 | 乾淨配對少於 3 個 | e\* = 304，乾淨配對只有 2 個（seeds 2、3；至少要 3 個），ρ_eval 1.5464、1.2917；成立區 min ≥ 1.05 |
| H1c 維度對比 | 無法判定 | 乾淨配對少於 3 個 | GloVe ρ_eval 1.5464、1.2917（2 個）；文字 0.9806、1.0924、0.9615、1.0404、0.9613；理由同 H1b |
| H1d profiler（seed 1） | 無法判定 | 見各資料集的原因 | GloVe κ = 0.7874、ν = 0.7967，原因：rust-hnsw-sq8: seed 1 在 e\* 有 hygiene 標記 ['background']；文字 κ = 1.0067、ν = 1.1047（沒有原因） |
| H1 整體 | 無法判定 |  | 前提 成立；H1a 成立；H1b 無法判定；H1c 無法判定；H1d 無法判定 |
| H2 GloVe | 成立 |  | 有效 seed [1, 2, 3, 4, 5]，median L_total 0.01379；每個有效 seed 的 L_trav ≤ thr、δ_F 都是 0（逐 seed 見 §6.3） |
| H2 文字 | 成立 |  | 有效 seed [1, 2, 3, 4, 5]，median L_total 0.0046；每個有效 seed 的 L_trav ≤ thr、δ_F 都是 0（逐 seed 見 §6.3） |
| H2 整體 | 成立 |  | GloVe 成立、文字 成立 |
| quantization frontier glove S 對 F 0.90 | uncertain |  | FP32 [4291, 4533]、SQ8 [5472, 6402]；seed 範圍：rust-hnsw-sq8 5359.372621250835 vs rust-hnsw-fp32g 4944.953907963472 → 未分開；包含被標記 build：相同 |
| quantization frontier glove S 對 F 0.95 | uncertain |  | FP32 [1877, 2199]、SQ8 [2299, 2944]；seed 範圍：rust-hnsw-sq8 1831.089906028781 vs rust-hnsw-fp32g 2665.8497614381035 → 未分開；包含被標記 build：相同 |
| quantization frontier glove Q 對 P 0.90 | uncertain |  | FP32 [8005, 10775]、SQ8 [6255, 7321]；seed 範圍：faiss-hnsw 6797.804241252033 vs faiss-hnsw-sq8 7472.282036802109 → 未分開；包含被標記 build：within_quantization（不同） |
| quantization frontier glove Q 對 P 0.95 | uncertain |  | FP32 [3350, 4014]、SQ8 [2445, 3119]；seed 範圍：faiss-hnsw 2952.624875808243 vs faiss-hnsw-sq8 3172.3111797757865 → 未分開；包含被標記 build：within_quantization（不同） |
| quantization frontier text S 對 F 0.90 | within_quantization |  | FP32 [3085, 3744]、SQ8 [2965, 3594]；包含被標記 build：相同 |
| quantization frontier text S 對 F 0.95 | uncertain |  | FP32 [2277, 3085]、SQ8 [1769, 2198]；seed 範圍：rust-hnsw-fp32g 2097.688660281287 vs rust-hnsw-sq8 2485.856770727403 → 未分開；包含被標記 build：相同 |
| quantization frontier text Q 對 P 0.90 | within_quantization |  | FP32 [8603, 10728]、SQ8 [7716, 9427]；包含被標記 build：相同 |
| quantization frontier text Q 對 P 0.95 | within_quantization |  | FP32 [6234, 8603]、SQ8 [5605, 7716]；包含被標記 build：相同 |
| C1 記憶體 glove F − S | 不一致 |  | Δ_theory 29999200 B，容許 2097152 B；seed 5 偏差 4030488 B 超出；其餘 seed 在容許內（逐 seed 見 §6.4） |
| C1 記憶體 glove P − Q | 一致 |  | Δ_theory 29999200 B，容許 2097152 B；3 個 seed 都在容許內（逐 seed 見 §6.4） |
| C2 FP32 重排 glove | 一致 |  | max \|δ_F\| = 0.0，violations 0 |
| C3 天花板 glove | 一致 |  | 天花板 c = 0.98048，最大 Recall@10 = 0.97173，violations 0 |
| C4 路徑失真 glove | 一致 |  | F ÷ R 的 QPS（seed 1、e\*）= 0.9725035618147205；一致區 [0.90, 1.10] |
| C5 Gate 重現 glove | 不一致 |  | R 在 e\* 的占比 traversal 84.4%、visited set 8.0%、heap 7.2%；Gate profiling 的占比 traversal 0.4867079229495337、visited_set 0.4356314826113484、heap 0.07421772858014469；原因在任何真實資料上的量化結果之前已寫明（019 第一則補記第 5 節）：inline 決策改變（visited set 的 `HashMap::insert` 被 inline 進 `search_layer`，那部分時間改算在 traversal），不是 Gate 無法重現 |
| C6 漂移 glove | 只描述 |  | C 中 R 的 QPS ÷ Core 正式 run 中 rust-hnsw seed 1 的 QPS（共同的 ef）：ef 64 0.817；ef 96 0.823；ef 128 0.834；ef 152 0.824；ef 192 0.825；ef 256 0.847；ef 304 0.833；ef 384 0.849 |
| C7a 逐位元 glove | 一致 |  | 逐位元相同 58 個，不相同 0 個 |
| C7b 逐位元 glove | 一致 |  | 逐位元相同 24 個，不相同 0 個 |
| C8 FAISS glove | 一致 | codes 一致比例只描述 | seeds 1, 2, 3 都「一致」：graph 相同、e\* ids 相符、codebook 相同、codes 一致比例 1.0, 1.0, 1.0 |
| C1 記憶體 text F − S | 一致 |  | Δ_theory 230393856 B，容許 11519693 B；5 個 seed 都在容許內（逐 seed 見 §6.4） |
| C1 記憶體 text P − Q | 一致 |  | Δ_theory 230393856 B，容許 11519693 B；3 個 seed 都在容許內（逐 seed 見 §6.4） |
| C2 FP32 重排 text | 一致 |  | max \|δ_F\| = 0.0，violations 0 |
| C3 天花板 text | 一致 |  | 天花板 c = 0.9935，最大 Recall@10 = 0.99265，violations 0 |
| C4 路徑失真 text | 一致 |  | F ÷ R 的 QPS（seed 1、e\*）= 0.9934802403512827；一致區 [0.90, 1.10] |
| C5 Gate 重現 text | 不一致 |  | R 在 e\* 的占比 traversal 97.1%、visited set 1.6%、heap 0.9%；Gate profiling 的占比 traversal 0.9219660722053067、visited_set 0.06450630709003914、heap 0.009525880817746846；原因在任何真實資料上的量化結果之前已寫明（019 第一則補記第 5 節）：inline 決策改變（visited set 的 `HashMap::insert` 被 inline 進 `search_layer`，那部分時間改算在 traversal），不是 Gate 無法重現 |
| C6 漂移 text | 只描述 |  | C 中 R 的 QPS ÷ Core 正式 run 中 rust-hnsw seed 1 的 QPS（共同的 ef）：ef 24 1.094；ef 32 1.089；ef 48 1.090；ef 64 1.089；ef 96 1.089；ef 128 1.133；ef 192 1.097；ef 256 1.097；ef 384 1.099；ef 512 1.106 |
| C7a 逐位元 text | 一致 |  | 逐位元相同 70 個，不相同 0 個 |
| C7b 逐位元 text | 一致 |  | 逐位元相同 30 個，不相同 0 個 |
| C8 FAISS text | 一致 | codes 一致比例只描述 | seeds 1, 2, 3 都「一致」：graph 相同、e\* ids 相符、codebook 相同、codes 一致比例 1.0, 1.0, 1.0 |

要點（照判定表，不另加判定）：

- **H1 整體：無法判定**。H1a（文字）**成立**；H1b（GloVe）**無法判定**：e\* = 304 只剩 seeds 2、3 兩個乾淨配對，少於登記的 3 個；H1c 理由相同；H1d **無法判定**：GloVe seed 1 的 S 在 e\* 有 `background` 標記（這條只看 seed 1，與乾淨配對的數目無關）。
- **H2 整體：成立**（GloVe 成立、文字成立）：SQ8 的 recall 損失來自最終排序，不是走訪。
- **C1**：GloVe 的 F − S **不一致**，只有 seed 5 偏離 4030488 B（換算約 3.84 MiB），超過 max(5% × Δ_theory, 2 MiB) = 2097152 B（2 MiB）；GloVe 的 P − Q、文字的 F − S 與 P − Q 都一致。
- **C5** 兩個資料集都不一致，原因在任何真實資料上的量化結果之前已寫明（019 第一則補記第 5 節）。C6 只描述。其他檢查都一致。
- 專案作者 2026-10-19 決定照實報告、**不補 seed**，C 收尾後封版（019 完成補記 §3）。

### 6.2 H1：每個 seed 的 ρ_eval（e\*）

照抄報告的 H1 表。只有乾淨配對用於判定。

**GloVe（e\* = 304）**

<!-- check:quant-h1:quant-glove -->
| seed | 乾淨 | 標記 | ρ_QPS | D_S / D_F | ρ_eval | 用於判定 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 否 | background | 1.2595 | 1.0000 | 1.2595 | 否 |
| 2 | 是 | — | 1.5464 | 1.0000 | 1.5464 | 是 |
| 3 | 是 | — | 1.2918 | 0.9999 | 1.2917 | 是 |
| 4 | 否 | background | 1.2874 | 1.0000 | 1.2875 | 否 |
| 5 | 否 | background, load | 1.1484 | 1.0000 | 1.1484 | 否 |

- H1d（seed 1）：κ = 0.7874、ν = 0.7967；f_k = {'rust-hnsw-fp32g': 0.29411764705882354, 'rust-hnsw-sq8': 0.29168135354247443}；ν < 0.90：cache spillover（向量變小之後，graph 與 visited set 也跟著變快）。原因：rust-hnsw-sq8: seed 1 在 e\* 有 hygiene 標記 ['background']

**文字（e\* = 48）**

<!-- check:quant-h1:quant-text -->
| seed | 乾淨 | 標記 | ρ_QPS | D_S / D_F | ρ_eval | 用於判定 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 是 | — | 0.9799 | 1.0007 | 0.9806 | 是 |
| 2 | 是 | — | 1.0921 | 1.0003 | 1.0924 | 是 |
| 3 | 是 | — | 0.9617 | 0.9998 | 0.9615 | 是 |
| 4 | 是 | — | 1.0400 | 1.0004 | 1.0404 | 是 |
| 5 | 是 | — | 0.9616 | 0.9996 | 0.9613 | 是 |

- H1d（seed 1）：κ = 1.0067、ν = 1.1047；f_k = {'rust-hnsw-fp32g': 0.8667247766397908, 'rust-hnsw-sq8': 0.8556193958387742}；ν > 1.10：原本藏在長相依鏈後面的非 kernel 工作露出來了。

- GloVe 被標記的三個配對（seeds 1、4、5）的 ρ_eval 為 1.2595、1.2875、1.1484，都大於 1.05；依登記的規則它們不能用於判定，這裡只作**看到結果之後的描述**，不改變判定表的任何一列（019 完成補記 §3）。

### 6.3 H2：每個 seed 的損失分解（e\*）

照抄報告的 H2 表（判定只用 e\*；其他 ef 的分解在報告中，不判定）。成立條件：所有有效 seed 的 L_trav ≤ thr。

**GloVe（e\* = 304）**

<!-- check:quant-h2:quant-glove -->
| seed | 有效 | ids 核對 | L_total | L_trav | L_rank | δ_F | thr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 是 | 通過 | 0.01384 | 0.00005 | 0.01379 | 0.00000 | 0.00446 |
| 2 | 是 | 通過 | 0.01387 | 0.00011 | 0.01376 | 0.00000 | 0.00447 |
| 3 | 是 | 通過 | 0.01376 | 0.00003 | 0.01373 | 0.00000 | 0.00444 |
| 4 | 是 | 通過 | 0.01379 | 0.00004 | 0.01375 | 0.00000 | 0.00445 |
| 5 | 是 | 通過 | 0.01376 | -0.00004 | 0.01380 | 0.00000 | 0.00444 |

**文字（e\* = 48）**

<!-- check:quant-h2:quant-text -->
| seed | 有效 | ids 核對 | L_total | L_trav | L_rank | δ_F | thr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 是 | 通過 | 0.00460 | 0.00035 | 0.00425 | 0.00000 | 0.00215 |
| 2 | 是 | 通過 | 0.00445 | 0.00040 | 0.00405 | 0.00000 | 0.00211 |
| 3 | 是 | 通過 | 0.00460 | 0.00050 | 0.00410 | 0.00000 | 0.00215 |
| 4 | 是 | 通過 | 0.00470 | 0.00060 | 0.00410 | 0.00000 | 0.00217 |
| 5 | 是 | 通過 | 0.00445 | 0.00045 | 0.00400 | 0.00000 | 0.00211 |

### 6.4 C1：記憶體差對照理論值（每個 seed）

ΔM_relief = M_relief(FP32 臂) − M_relief(SQ8 臂)，對照 Δ_theory = 3·n·d − 8·d；每個 seed 的偏差絕對值 ≤ max(5% × Δ_theory, 2 MiB) 為一致。M_relief 只是診斷（019 §6）；ΔM_014 照列、不判定。單位都是 byte，照報告判定表的數字。

<!-- check:quant-c1 -->
| 配對 | seed | ΔM_relief (B) | 偏差 (B) | 容許 (B) | 在容許內 | 標記 | ΔM_014 (B) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glove F − S | 1 | 30605264 | 606064 | 2097152 | 是 | `background` | 30310376 |
| glove F − S | 2 | 29900920 | -98280 | 2097152 | 是 | — | 29835312 |
| glove F − S | 3 | 31309824 | 1310624 | 2097152 | 是 | — | 31064040 |
| glove F − S | 4 | 30834784 | 835584 | 2097152 | 是 | `background` | 30343240 |
| glove F − S | 5 | 34029688 | 4030488 | 2097152 | **否** | `background` | 33800456 |
| glove P − Q | 1 | 29360080 | -639120 | 2097152 | 是 | — | -13090864 |
| glove P − Q | 2 | 29491200 | -508000 | 2097152 | 是 | `load` | -11599872 |
| glove P − Q | 3 | 29114440 | -884760 | 2097152 | 是 | — | -11796432 |
| text F − S | 1 | 229752976 | -640880 | 11519693 | 是 | — | 230539240 |
| text F − S | 2 | 230490232 | 96376 | 11519693 | 是 | — | 230932528 |
| text F − S | 3 | 231227680 | 833824 | 11519693 | 是 | — | 230113208 |
| text F − S | 4 | 229638240 | -755616 | 11519693 | 是 | — | 229801888 |
| text F − S | 5 | 230637736 | 243880 | 11519693 | 是 | — | 230506448 |
| text P − Q | 1 | 229146744 | -1247112 | 11519693 | 是 | — | 216301472 |
| text P − Q | 2 | 230916312 | 522456 | 11519693 | 是 | — | 216203216 |
| text P − Q | 3 | 229818488 | -575368 | 11519693 | 是 | — | 218201992 |

- Δ_theory：GloVe 29999200 B（容許 2097152 B，即 2 MiB），文字 230393856 B（容許 11519693 B）。
- GloVe seed 5 的偏差 4030488 B（換算約 3.84 MiB）是唯一超出容許的 seed，所以「C1 記憶體 glove F − S」判為不一致。
- GloVe 的 P − Q，ΔM_014 是負的（Q 的 M_014 比 P 大）。ΔM_014 照列、不判定；報告另列每個 seed 的 allocator 保留量（M_014 − M_relief）：GloVe 的 Q 為 46972952、47661080、46972952 B，P 為 4522008、6570008、6062080 B（seeds 1–3）。

### 6.5 C2–C8

<!-- check:quant-checks -->
| 檢查 | 狀態 | 數字 |
| --- | --- | --- |
| C2 FP32 重排 glove | 一致 | max \|δ_F\| = 0.0，violations 0 |
| C3 天花板 glove | 一致 | 天花板 c = 0.98048，最大 Recall@10 = 0.97173，violations 0 |
| C4 路徑失真 glove | 一致 | F ÷ R 的 QPS（seed 1、e\*）= 0.9725035618147205；一致區 [0.90, 1.10] |
| C5 Gate 重現 glove | 不一致 | R 在 e\* 的占比 traversal 84.4%、visited set 8.0%、heap 7.2%；Gate profiling 的占比 traversal 0.4867079229495337、visited_set 0.4356314826113484、heap 0.07421772858014469；原因在任何真實資料上的量化結果之前已寫明（019 第一則補記第 5 節）：inline 決策改變（visited set 的 `HashMap::insert` 被 inline 進 `search_layer`，那部分時間改算在 traversal），不是 Gate 無法重現 |
| C6 漂移 glove | 只描述 | C 中 R 的 QPS ÷ Core 正式 run 中 rust-hnsw seed 1 的 QPS（共同的 ef）：ef 64 0.817；ef 96 0.823；ef 128 0.834；ef 152 0.824；ef 192 0.825；ef 256 0.847；ef 304 0.833；ef 384 0.849 |
| C7a 逐位元 glove | 一致 | 逐位元相同 58 個，不相同 0 個 |
| C7b 逐位元 glove | 一致 | 逐位元相同 24 個，不相同 0 個 |
| C8 FAISS glove | 一致 | seeds 1, 2, 3 都「一致」：graph 相同、e\* ids 相符、codebook 相同、codes 一致比例 1.0, 1.0, 1.0 |
| C2 FP32 重排 text | 一致 | max \|δ_F\| = 0.0，violations 0 |
| C3 天花板 text | 一致 | 天花板 c = 0.9935，最大 Recall@10 = 0.99265，violations 0 |
| C4 路徑失真 text | 一致 | F ÷ R 的 QPS（seed 1、e\*）= 0.9934802403512827；一致區 [0.90, 1.10] |
| C5 Gate 重現 text | 不一致 | R 在 e\* 的占比 traversal 97.1%、visited set 1.6%、heap 0.9%；Gate profiling 的占比 traversal 0.9219660722053067、visited_set 0.06450630709003914、heap 0.009525880817746846；原因在任何真實資料上的量化結果之前已寫明（019 第一則補記第 5 節）：inline 決策改變（visited set 的 `HashMap::insert` 被 inline 進 `search_layer`，那部分時間改算在 traversal），不是 Gate 無法重現 |
| C6 漂移 text | 只描述 | C 中 R 的 QPS ÷ Core 正式 run 中 rust-hnsw seed 1 的 QPS（共同的 ef）：ef 24 1.094；ef 32 1.089；ef 48 1.090；ef 64 1.089；ef 96 1.089；ef 128 1.133；ef 192 1.097；ef 256 1.097；ef 384 1.099；ef 512 1.106 |
| C7a 逐位元 text | 一致 | 逐位元相同 70 個，不相同 0 個 |
| C7b 逐位元 text | 一致 | 逐位元相同 30 個，不相同 0 個 |
| C8 FAISS text | 一致 | seeds 1, 2, 3 都「一致」：graph 相同、e\* ids 相符、codebook 相同、codes 一致比例 1.0, 1.0, 1.0 |

C2 FP32 重排、C3 天花板、C4 路徑失真、C5 Gate 重現、C6 漂移（只描述）、C7a / C7b 與 Core 正式 run 的 ids 逐位元比對、C8 FAISS 移植的 graph（019 §2）。C6 的比值同時包含時間漂移與機器碼差異，兩者無法分開；C 的絕對 QPS 不與 Core 的表直接比較。

### 6.6 frontier 讀值（0.90 / 0.95 / 0.99，五個臂）

照抄報告的「讀值 1：frontier」表（排除被標記的 build；M_014 為主要記憶體指標）。目標 0.90、0.95 為主，0.99 只描述（019 §6）。R 只有 seed 1 的一個 build。

**GloVe**

<!-- check:quant-frontier:quant-glove -->
| 實作 | 目標 | QPS | P50 (µs) | P95 (µs) | M_014 (MiB) | 排除標記點是否改變讀值 |
| --- | --- | --- | --- | --- | --- | --- |
| rust-hnsw-fp32g | 0.90 | 4291 [3518, 4291]（3 builds） @ M=16, efC=200, ef=152 | 239.2 [233.8, 287.8]（3 builds） @ M=16, efC=200, ef=152 | 296.4 [288.5, 346.5]（3 builds） @ M=16, efC=200, ef=152 | 105.4 [104.5, 105.6]（3 builds） @ M=16, efC=200, ef=152 | 是 |
| rust-hnsw-fp32g | 0.95 | 1877 [1870, 2281]（3 builds） @ M=16, efC=200, ef=304 | 546.7 [441.5, 548.7]（3 builds） @ M=16, efC=200, ef=304 | 625.1 [510.4, 629.4]（3 builds） @ M=16, efC=200, ef=304 | 105.4 [104.5, 105.6]（3 builds） @ M=16, efC=200, ef=304 | 是 |
| rust-hnsw-fp32g | 0.99 | 未達到 | 未達到 | 未達到 | 未達到 | 否 |
| rust-hnsw-sq8 | 0.90 | 5472 [5359, 5529]（3 builds） @ M=16, efC=200, ef=152 | 183.7 [183.1, 187.5]（3 builds） @ M=16, efC=200, ef=152 | 231.7 [229.0, 234.7]（3 builds） @ M=16, efC=200, ef=152 | 76.1 [73.4, 77.0]（4 builds） @ M=16, efC=200, ef=384 | 是 |
| rust-hnsw-sq8 | 0.95 | 2299 [1831, 2324]（4 builds） @ M=16, efC=200, ef=384 | 433.7 [428.1, 543.0]（4 builds） @ M=16, efC=200, ef=384 | 537.2 [531.0, 658.1]（4 builds） @ M=16, efC=200, ef=384 | 76.1 [73.4, 77.0]（4 builds） @ M=16, efC=200, ef=384 | 是 |
| rust-hnsw-sq8 | 0.99 | 未達到 | 未達到 | 未達到 | 未達到 | 否 |
| faiss-hnsw | 0.90 | 8005 [6798, 9213]（2 builds） @ M=16, efC=200, ef=128 | 131.7 [112.7, 150.7]（2 builds） @ M=16, efC=200, ef=128 | 162.9 [138.8, 186.9]（2 builds） @ M=16, efC=200, ef=128 | 58.1 [57.1, 59.0]（2 builds） @ M=16, efC=200, ef=128 | 是 |
| faiss-hnsw | 0.95 | 3350 [2953, 3748]（2 builds） @ M=16, efC=200, ef=304 | 312.4 [273.9, 351.0]（2 builds） @ M=16, efC=200, ef=304 | 372.9 [324.6, 421.3]（2 builds） @ M=16, efC=200, ef=304 | 58.1 [57.1, 59.0]（2 builds） @ M=16, efC=200, ef=304 | 是 |
| faiss-hnsw | 0.99 | 未達到 | 未達到 | 未達到 | 未達到 | 否 |
| faiss-hnsw-sq8 | 0.90 | 6255 [5063, 6360]（3 builds） @ M=16, efC=200, ef=152 | 168.8 [164.8, 206.4]（3 builds） @ M=16, efC=200, ef=152 | 205.5 [197.8, 246.8]（3 builds） @ M=16, efC=200, ef=152 | 69.8 [69.6, 70.3]（3 builds） @ M=16, efC=200, ef=152 | 否 |
| faiss-hnsw-sq8 | 0.95 | 2445 [1968, 2468]（3 builds） @ M=16, efC=200, ef=384 | 423.3 [418.2, 524.9]（3 builds） @ M=16, efC=200, ef=384 | 494.7 [486.7, 617.2]（3 builds） @ M=16, efC=200, ef=384 | 69.8 [69.6, 70.3]（3 builds） @ M=16, efC=200, ef=384 | 否 |
| faiss-hnsw-sq8 | 0.99 | 未達到 | 未達到 | 未達到 | 未達到 | 否 |
| rust-hnsw（錨點） | 0.90 | 3624 @ M=16, efC=200, ef=152 | 279.5 @ M=16, efC=200, ef=152 | 338.1 @ M=16, efC=200, ef=152 | 104.2 @ M=16, efC=200, ef=152 | 否 |
| rust-hnsw（錨點） | 0.95 | 1923 @ M=16, efC=200, ef=304 | 532.3 @ M=16, efC=200, ef=304 | 611.8 @ M=16, efC=200, ef=304 | 104.2 @ M=16, efC=200, ef=304 | 否 |
| rust-hnsw（錨點） | 0.99 | 未達到 | 未達到 | 未達到 | 未達到 | 否 |

**文字**

<!-- check:quant-frontier:quant-text -->
| 實作 | 目標 | QPS | P50 (µs) | P95 (µs) | M_014 (MiB) | 排除標記點是否改變讀值 |
| --- | --- | --- | --- | --- | --- | --- |
| rust-hnsw-fp32g | 0.90 | 3085 [2830, 3092]（5 builds） @ M=16, efC=200, ef=32 | 323.3 [322.3, 350.9]（5 builds） @ M=16, efC=200, ef=32 | 408.1 [407.5, 449.3]（5 builds） @ M=16, efC=200, ef=32 | 318.8 [318.2, 319.2]（5 builds） @ M=16, efC=200, ef=32 | 否 |
| rust-hnsw-fp32g | 0.95 | 2277 [2098, 2285]（5 builds） @ M=16, efC=200, ef=48 | 441.5 [440.2, 478.4]（5 builds） @ M=16, efC=200, ef=48 | 533.7 [531.7, 584.2]（5 builds） @ M=16, efC=200, ef=48 | 318.8 [318.2, 319.2]（5 builds） @ M=16, efC=200, ef=48 | 否 |
| rust-hnsw-fp32g | 0.99 | 738 [682, 740]（5 builds） @ M=16, efC=200, ef=192 | 1372.0 [1367.7, 1484.5]（5 builds） @ M=16, efC=200, ef=192 | 1655.5 [1643.6, 1792.1]（5 builds） @ M=16, efC=200, ef=192 | 318.8 [318.2, 319.2]（5 builds） @ M=16, efC=200, ef=192 | 否 |
| rust-hnsw-sq8 | 0.90 | 2965 [2937, 3337]（5 builds） @ M=16, efC=200, ef=32 | 336.1 [295.0, 337.2]（5 builds） @ M=16, efC=200, ef=32 | 429.3 [375.6, 436.8]（5 builds） @ M=16, efC=200, ef=32 | 99.0 [98.7, 99.4]（5 builds） @ M=16, efC=200, ef=32 | 否 |
| rust-hnsw-sq8 | 0.95 | 1769 [1759, 1993]（5 builds） @ M=16, efC=200, ef=64 | 569.4 [507.7, 573.8]（5 builds） @ M=16, efC=200, ef=64 | 685.9 [609.5, 693.2]（5 builds） @ M=16, efC=200, ef=64 | 99.0 [98.7, 99.4]（5 builds） @ M=16, efC=200, ef=64 | 否 |
| rust-hnsw-sq8 | 0.99 | 560 [559, 631]（5 builds） @ M=16, efC=200, ef=256 | 1813.3 [1606.4, 1816.8]（5 builds） @ M=16, efC=200, ef=256 | 2163.7 [1906.2, 2172.4]（5 builds） @ M=16, efC=200, ef=256 | 99.0 [98.7, 99.4]（5 builds） @ M=16, efC=200, ef=256 | 否 |
| faiss-hnsw | 0.90 | 8603 [8536, 8619]（3 builds） @ M=16, efC=200, ef=32 | 118.5 [117.8, 121.2]（3 builds） @ M=16, efC=200, ef=32 | 146.0 [145.9, 148.9]（3 builds） @ M=16, efC=200, ef=32 | 313.0 [312.8, 313.4]（3 builds） @ M=16, efC=200, ef=32 | 否 |
| faiss-hnsw | 0.95 | 6234 [6163, 6260]（3 builds） @ M=16, efC=200, ef=48 | 165.8 [163.8, 167.6]（3 builds） @ M=16, efC=200, ef=48 | 196.5 [192.8, 199.2]（3 builds） @ M=16, efC=200, ef=48 | 313.0 [312.8, 313.4]（3 builds） @ M=16, efC=200, ef=48 | 否 |
| faiss-hnsw | 0.99 | 2663 [2650, 2682]（3 builds） @ M=16, efC=200, ef=128 | 384.3 [382.1, 390.1]（3 builds） @ M=16, efC=200, ef=128 | 440.7 [439.2, 451.2]（3 builds） @ M=16, efC=200, ef=128 | 313.0 [312.8, 313.4]（3 builds） @ M=16, efC=200, ef=128 | 否 |
| faiss-hnsw-sq8 | 0.90 | 7716 [7649, 7751]（3 builds） @ M=16, efC=200, ef=32 | 134.3 [134.2, 139.6]（3 builds） @ M=16, efC=200, ef=32 | 165.8 [165.8, 170.5]（3 builds） @ M=16, efC=200, ef=32 | 106.5 [105.3, 106.9]（3 builds） @ M=16, efC=200, ef=32 | 否 |
| faiss-hnsw-sq8 | 0.95 | 5605 [5144, 5663]（3 builds） @ M=16, efC=200, ef=48 | 186.3 [181.4, 200.7]（3 builds） @ M=16, efC=200, ef=48 | 223.5 [214.0, 237.5]（3 builds） @ M=16, efC=200, ef=48 | 106.5 [105.3, 106.9]（3 builds） @ M=16, efC=200, ef=48 | 否 |
| faiss-hnsw-sq8 | 0.99 | 1350 [1224, 1360]（3 builds） @ M=16, efC=200, ef=256 | 762.7 [756.2, 836.8]（3 builds） @ M=16, efC=200, ef=256 | 894.8 [876.8, 973.0]（3 builds） @ M=16, efC=200, ef=256 | 106.5 [105.3, 106.9]（3 builds） @ M=16, efC=200, ef=256 | 否 |
| rust-hnsw（錨點） | 0.90 | 3086 @ M=16, efC=200, ef=32 | 321.5 @ M=16, efC=200, ef=32 | 405.8 @ M=16, efC=200, ef=32 | 318.6 @ M=16, efC=200, ef=32 | 否 |
| rust-hnsw（錨點） | 0.95 | 2292 @ M=16, efC=200, ef=48 | 440.0 @ M=16, efC=200, ef=48 | 535.1 @ M=16, efC=200, ef=48 | 318.6 @ M=16, efC=200, ef=48 | 否 |
| rust-hnsw（錨點） | 0.99 | 743 @ M=16, efC=200, ef=192 | 1362.9 @ M=16, efC=200, ef=192 | 1644.4 @ M=16, efC=200, ef=192 | 318.6 @ M=16, efC=200, ef=192 | 否 |

**GloVe：包含被標記點的讀值**（只列出與主要讀值不同者；文字沒有被標記的點）

<!-- check:quant-frontier-flagged:quant-glove -->
| 實作 | 目標 | QPS | P50 (µs) | P95 (µs) | M_014 (MiB) |
| --- | --- | --- | --- | --- | --- |
| rust-hnsw-fp32g | 0.90 | 4087 [3518, 4291]（5 builds） @ M=16, efC=200, ef=152 | 246.6 [233.8, 287.8]（5 builds） @ M=16, efC=200, ef=152 | 299.2 [288.5, 349.4]（5 builds） @ M=16, efC=200, ef=152 | 105.4 [104.5, 105.6]（5 builds） @ M=16, efC=200, ef=152 |
| rust-hnsw-fp32g | 0.95 | 1877 [1800, 2287]（5 builds） @ M=16, efC=200, ef=304 | 545.2 [441.5, 548.7]（5 builds） @ M=16, efC=200, ef=304 | 625.1 [510.4, 696.9]（5 builds） @ M=16, efC=200, ef=304 | 105.4 [104.5, 105.6]（5 builds） @ M=16, efC=200, ef=304 |
| rust-hnsw-sq8 | 0.90 | 5359 [3903, 5529]（5 builds） @ M=16, efC=200, ef=152 | 187.5 [183.1, 256.3]（5 builds） @ M=16, efC=200, ef=152 | 234.7 [229.0, 323.5]（5 builds） @ M=16, efC=200, ef=152 | 76.0 [73.4, 77.0]（5 builds） @ M=16, efC=200, ef=152 |
| rust-hnsw-sq8 | 0.95 | 2288 [1831, 2324]（5 builds） @ M=16, efC=200, ef=384 | 436.8 [428.1, 543.0]（5 builds） @ M=16, efC=200, ef=384 | 539.0 [531.0, 658.1]（5 builds） @ M=16, efC=200, ef=384 | 76.0 [73.4, 77.0]（5 builds） @ M=16, efC=200, ef=384 |
| faiss-hnsw | 0.90 | 7308 [6798, 9213]（3 builds） @ M=16, efC=200, ef=128 | 143.4 [112.7, 150.7]（3 builds） @ M=16, efC=200, ef=128 | 174.7 [138.8, 186.9]（3 builds） @ M=16, efC=200, ef=128 | 58.8 [57.1, 59.0]（3 builds） @ M=16, efC=200, ef=128 |
| faiss-hnsw | 0.95 | 2969 [2953, 3748]（3 builds） @ M=16, efC=200, ef=304 | 347.1 [273.9, 351.0]（3 builds） @ M=16, efC=200, ef=304 | 411.3 [324.6, 421.3]（3 builds） @ M=16, efC=200, ef=304 | 58.8 [57.1, 59.0]（3 builds） @ M=16, efC=200, ef=304 |

**比值**（SQ8 ÷ FP32，兩者的主要讀值都達到目標才計算）

<!-- check:quant-ratios:quant-glove -->
| 資料集 | 配對 | 目標 | QPS | P50 | P95 | M_014 |
| --- | --- | --- | --- | --- | --- | --- |
| GloVe | S 對 F | 0.90 | 1.275 | 0.768 | 0.782 | 0.722 |
| GloVe | S 對 F | 0.95 | 1.224 | 0.793 | 0.859 | 0.722 |
| GloVe | S 對 F | 0.99 | —（未達到） |  |  |  |
| GloVe | Q 對 P | 0.90 | 0.781 | 1.282 | 1.262 | 1.202 |
| GloVe | Q 對 P | 0.95 | 0.730 | 1.355 | 1.326 | 1.202 |
| GloVe | Q 對 P | 0.99 | —（未達到） |  |  |  |

<!-- check:quant-ratios:quant-text -->
| 資料集 | 配對 | 目標 | QPS | P50 | P95 | M_014 |
| --- | --- | --- | --- | --- | --- | --- |
| 文字 | S 對 F | 0.90 | 0.961 | 1.040 | 1.052 | 0.311 |
| 文字 | S 對 F | 0.95 | 0.777 | 1.290 | 1.285 | 0.311 |
| 文字 | S 對 F | 0.99 | 0.758 | 1.322 | 1.307 | 0.311 |
| 文字 | Q 對 P | 0.90 | 0.897 | 1.133 | 1.136 | 0.340 |
| 文字 | Q 對 P | 0.95 | 0.899 | 1.124 | 1.138 | 0.340 |
| 文字 | Q 對 P | 0.99 | 0.507 | 1.984 | 2.031 | 0.340 |

**e\* 的 recall**（報告「每個 ef 的 recall」表中 e\* 的列；recall 用全部 build）

<!-- check:quant-recall:quant-glove -->
| 資料集 | 實作 | ef | Recall@10（主要） | tie-aware Recall@10 | Recall@1 |
| --- | --- | --- | --- | --- | --- |
| GloVe | faiss-hnsw | 304 | 0.9581 [0.9576, 0.9588]（3） | 0.9581 [0.9576, 0.9588]（3） | 0.9735 [0.9705, 0.9737]（3） |
| GloVe | faiss-hnsw-sq8 | 304 | 0.9439 [0.9438, 0.9442]（3） | 0.9439 [0.9438, 0.9443]（3） | 0.9509 [0.9486, 0.9522]（3） |
| GloVe | rust-hnsw | 304 | 0.9570 [0.9570, 0.9570]（1） | 0.9570 [0.9570, 0.9570]（1） | 0.9744 [0.9744, 0.9744]（1） |
| GloVe | rust-hnsw-fp32g | 304 | 0.9567 [0.9565, 0.9570]（5） | 0.9567 [0.9565, 0.9570]（5） | 0.9746 [0.9741, 0.9753]（5） |
| GloVe | rust-hnsw-sq8 | 304 | 0.9429 [0.9427, 0.9432]（5） | 0.9429 [0.9428, 0.9432]（5） | 0.9524 [0.9519, 0.9534]（5） |

<!-- check:quant-recall:quant-text -->
| 資料集 | 實作 | ef | Recall@10（主要） | tie-aware Recall@10 | Recall@1 |
| --- | --- | --- | --- | --- | --- |
| 文字 | faiss-hnsw | 48 | 0.9571 [0.9540, 0.9583]（3） | 0.9571 [0.9540, 0.9583]（3） | 0.9760 [0.9745, 0.9835]（3） |
| 文字 | faiss-hnsw-sq8 | 48 | 0.9524 [0.9496, 0.9533]（3） | 0.9524 [0.9496, 0.9533]（3） | 0.9710 [0.9685, 0.9785]（3） |
| 文字 | rust-hnsw | 48 | 0.9537 [0.9537, 0.9537]（1） | 0.9537 [0.9537, 0.9537]（1） | 0.9725 [0.9725, 0.9725]（1） |
| 文字 | rust-hnsw-fp32g | 48 | 0.9538 [0.9537, 0.9548]（5） | 0.9538 [0.9537, 0.9548]（5） | 0.9730 [0.9725, 0.9745]（5） |
| 文字 | rust-hnsw-sq8 | 48 | 0.9494 [0.9492, 0.9502]（5） | 0.9494 [0.9492, 0.9502]（5） | 0.9695 [0.9680, 0.9705]（5） |

### 6.7 quantization frontier 的判讀（事先登記的描述性判讀，不是假設）

015 §1 的 QPS 範圍與 seed 分開規則，套用到 S 對 F、Q 對 P；只判 0.90 與 0.95（019 §6）。照抄報告。

**GloVe**

<!-- check:quant-qf:quant-glove -->
| 配對 | 目標 | FP32 QPS 範圍 | SQ8 QPS 範圍 | seed 範圍 | 判讀 | 包含被標記 build 的判讀 |
| --- | --- | --- | --- | --- | --- | --- |
| S 對 F | 0.90 | [4291, 4533] | [5472, 6402] | rust-hnsw-sq8 5359.372621250835 vs rust-hnsw-fp32g 4944.953907963472 → 未分開 | uncertain | 相同 |
| S 對 F | 0.95 | [1877, 2199] | [2299, 2944] | rust-hnsw-sq8 1831.089906028781 vs rust-hnsw-fp32g 2665.8497614381035 → 未分開 | uncertain | 相同 |
| Q 對 P | 0.90 | [8005, 10775] | [6255, 7321] | faiss-hnsw 6797.804241252033 vs faiss-hnsw-sq8 7472.282036802109 → 未分開 | uncertain | within_quantization（不同） |
| Q 對 P | 0.95 | [3350, 4014] | [2445, 3119] | faiss-hnsw 2952.624875808243 vs faiss-hnsw-sq8 3172.3111797757865 → 未分開 | uncertain | within_quantization（不同） |

**文字**

<!-- check:quant-qf:quant-text -->
| 配對 | 目標 | FP32 QPS 範圍 | SQ8 QPS 範圍 | seed 範圍 | 判讀 | 包含被標記 build 的判讀 |
| --- | --- | --- | --- | --- | --- | --- |
| S 對 F | 0.90 | [3085, 3744] | [2965, 3594] | — | within_quantization | 相同 |
| S 對 F | 0.95 | [2277, 3085] | [1769, 2198] | rust-hnsw-fp32g 2097.688660281287 vs rust-hnsw-sq8 2485.856770727403 → 未分開 | uncertain | 相同 |
| Q 對 P | 0.90 | [8603, 10728] | [7716, 9427] | — | within_quantization | 相同 |
| Q 對 P | 0.95 | [6234, 8603] | [5605, 7716] | — | within_quantization | 相同 |

### 6.8 速度與記憶體的取捨（看到結果之後的描述，**不是判定**）

以下只是把 §6.6 的讀值並列；事先登記的判讀只有 §6.7（S 對 F：GloVe 0.90 / 0.95 都是 `uncertain`；文字 0.90 `within_quantization`、0.95 `uncertain`）。

<!-- check:quant-tradeoff -->
| 資料集 | 目標 | F（FP32）QPS | S（SQ8）QPS | F 的 M_014 (MiB) | S 的 M_014 (MiB) | S ÷ F：QPS / M_014 | 事先登記的判讀 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 文字 | 0.95 | 2277 [2098, 2285]（5 builds）（ef 48） | 1769 [1759, 1993]（5 builds）（ef 64） | 318.8 [318.2, 319.2]（5 builds） | 99.0 [98.7, 99.4]（5 builds） | 0.777 / 0.311 | `uncertain` |
| GloVe | 0.95 | 1877 [1870, 2281]（3 builds）（ef 304） | 2299 [1831, 2324]（4 builds）（ef 384） | 105.4 [104.5, 105.6]（3 builds） | 76.1 [73.4, 77.0]（4 builds） | 1.224 / 0.722 | `uncertain` |

- 文字（768 維）：0.95 時 S 1769 對 F 2277 QPS，M_014 從 318.8 降到 99.0 MiB。S 要較大的 ef 才達到相同 recall。
- GloVe（100 維）：0.95 時 S 2299 對 F 1877 QPS，M_014 從 105.4 降到 76.1 MiB；但 S 的讀值有 4 個 build、F 只剩 3 個（被標記的 build 已排除），判讀是 `uncertain`，不能讀成「S 比 F 快」。
- build time 與 peak RSS 不會下降（019 §1）：文字的 peak RSS，S 的 5 個 build 在 1023.2–1023.6 MiB，F 在 950.1–950.9 MiB（§6.10）。
- C 的絕對 QPS 不與 Core 的表直接比較：兩者是不同的 run，C 中 R 的機器碼也與 Core 正式 run 不同（inline 決策改變）；C6 的比值（GloVe 0.817–0.849、文字 1.089–1.133）同時包含時間漂移與機器碼差異。

### 6.9 profiling（seed 1；019 §8，分類規則在取樣之前固定）

8 次取樣都通過 `quant_profile check`（取樣時的 Recall@10 與 formal 的同一點完全相同）。照抄報告；D 是每個 query 的距離評估數。

**GloVe**

<!-- check:quant-profile:quant-glove -->
| 實作 | ef | 角色 | check | QPS | D | kernel | traversal | visited set | heap | 記憶體配置 | 其他 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rust-hnsw | 304 | e_star | 通過 | 1923 | — | 0.0% | 84.4% | 8.0% | 7.2% | 0.3% | 0.0% |
| rust-hnsw-fp32g | 304 | e_star | 通過 | 1870 | 5343.3 | 29.4% | 20.1% | 43.0% | 7.2% | 0.2% | 0.0% |
| rust-hnsw-sq8 | 304 | e_star | 通過 | 2355 | 5343.5 | 29.2% | 22.4% | 39.2% | 8.9% | 0.3% | 0.0% |
| rust-hnsw-sq8 | 384 | same_recall_0.95 | 通過 | 1858 | 6523.2 | 28.6% | 21.3% | 40.4% | 9.0% | 0.4% | 0.4% |

- rust-hnsw-fp32g ef 304 每 query（µs）：kernel 157.28、visited_set 230.13、heap 38.45、traversal 107.44、allocation 1.22、other 0.23；每次評估（ns）：kernel 29.43、visited_set 43.07、heap 7.20、traversal 20.11、allocation 0.23、other 0.04
- rust-hnsw-sq8 ef 304 每 query（µs）：kernel 123.84、visited_set 166.25、heap 37.90、traversal 95.25、allocation 1.20、other 0.13；每次評估（ns）：kernel 23.18、visited_set 31.11、heap 7.09、traversal 17.83、allocation 0.22、other 0.02
- rust-hnsw-sq8 ef 384 每 query（µs）：kernel 153.69、visited_set 217.36、heap 48.39、traversal 114.81、allocation 1.97、other 1.92；每次評估（ns）：kernel 23.56、visited_set 33.32、heap 7.42、traversal 17.60、allocation 0.30、other 0.29

**文字**

<!-- check:quant-profile:quant-text -->
| 實作 | ef | 角色 | check | QPS | D | kernel | traversal | visited set | heap | 記憶體配置 | 其他 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rust-hnsw | 48 | e_star | 通過 | 2292 | — | 0.0% | 97.1% | 1.6% | 0.9% | 0.2% | 0.2% |
| rust-hnsw-fp32g | 48 | e_star | 通過 | 2277 | 1024.3 | 86.7% | 5.8% | 6.1% | 0.9% | 0.3% | 0.2% |
| rust-hnsw-sq8 | 48 | e_star | 通過 | 2231 | 1025.0 | 85.6% | 6.3% | 6.6% | 1.0% | 0.3% | 0.3% |
| rust-hnsw-sq8 | 64 | same_recall_0.95 | 通過 | 1770 | 1273.8 | 85.7% | 6.4% | 6.7% | 0.8% | 0.2% | 0.3% |

- rust-hnsw-fp32g ef 48 每 query（µs）：kernel 380.66、visited_set 26.93、heap 3.94、traversal 25.67、allocation 1.15、other 0.84；每次評估（ns）：kernel 371.64、visited_set 26.29、heap 3.85、traversal 25.06、allocation 1.12、other 0.82
- rust-hnsw-sq8 ef 48 每 query（µs）：kernel 383.47、visited_set 29.42、heap 4.27、traversal 28.33、allocation 1.23、other 1.46；每次評估（ns）：kernel 374.13、visited_set 28.70、heap 4.17、traversal 27.64、allocation 1.20、other 1.43
- rust-hnsw-sq8 ef 64 每 query（µs）：kernel 483.82、visited_set 37.89、heap 4.41、traversal 36.10、allocation 1.10、other 1.52；每次評估（ns）：kernel 379.83、visited_set 29.75、heap 3.46、traversal 28.34、allocation 0.86、other 1.19

### 6.10 記憶體（每個 build 的摘要）

報告的記憶體表依臂整理：各 seed 的值依 seed 順序以「/」分隔。M_014 為主要指標；M_relief、M_struct 只是診斷（M_struct 取自不計時的診斷，不是 formal run 的量測；R 沒有這個診斷）。

<!-- check:quant-memory:quant-glove -->
| 資料集 | 實作 | seeds | M_014 (MiB) | M_relief（診斷） | M_struct（診斷） | peak RSS (MiB) | 標記（子 process 或 build 層級，見 §6.11） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GloVe | rust-hnsw | 1 | 104.2 | 64.8 | — | 187.0 | — |
| GloVe | rust-hnsw-fp32g | 1 / 2 / 3 / 4 / 5 | 104.5 / 105.4 / 105.6 / 105.1 / 105.6 | 64.8 / 65.2 / 66.4 / 66.1 / 65.3 | 52.3 / 52.3 / 52.3 / 52.3 / 52.3 | 188.6 / 188.9 / 188.3 / 187.8 / 189.0 | seed 4 `background`、seed 5 `background` |
| GloVe | rust-hnsw-sq8 | 1 / 2 / 3 / 4 / 5 | 75.5 / 77.0 / 76.0 / 76.2 / 73.4 | 35.6 / 36.6 / 36.6 / 36.7 / 32.8 | 23.7 / 23.7 / 23.7 / 23.7 / 23.7 | 197.7 / 198.7 / 197.1 / 198.2 / 196.8 | seed 1 `background` |
| GloVe | faiss-hnsw | 1 / 2 / 3 | 57.1 / 58.8 / 59.0 | 52.8 / 52.5 / 53.2 | 51.9 / 51.9 / 51.9 | 157.4 / 158.4 / 156.5 | seed 2 `load` |
| GloVe | faiss-hnsw-sq8 | 1 / 2 / 3 | 69.6 / 69.8 / 70.3 | 24.8 / 24.4 / 25.5 | 23.3 / 23.3 / 23.3 | 180.6 / 181.4 / 180.6 | — |

<!-- check:quant-memory:quant-text -->
| 資料集 | 實作 | seeds | M_014 (MiB) | M_relief（診斷） | M_struct（診斷） | peak RSS (MiB) | 標記（子 process 或 build 層級，見 §6.11） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 文字 | rust-hnsw | 1 | 318.6 | 315.5 | — | 951.9 | — |
| 文字 | rust-hnsw-fp32g | 1 / 2 / 3 / 4 / 5 | 318.6 / 319.2 / 318.8 / 318.2 / 319.0 | 315.9 / 317.0 / 317.7 / 315.9 / 317.0 | 307.0 / 307.0 / 307.1 / 307.1 / 307.1 | 950.9 / 950.5 / 950.1 / 950.6 / 950.4 | — |
| 文字 | rust-hnsw-sq8 | 1 / 2 / 3 / 4 / 5 | 98.7 / 99.0 / 99.4 / 99.0 / 99.2 | 96.8 / 97.2 / 97.2 / 96.9 / 97.1 | 87.3 / 87.3 / 87.3 / 87.3 / 87.3 | 1023.4 / 1023.4 / 1023.6 / 1023.2 / 1023.5 | — |
| 文字 | faiss-hnsw | 1 / 2 / 3 | 312.8 / 313.0 / 313.4 | 307.9 / 308.1 / 307.1 | 306.7 / 306.7 / 306.7 | 669.2 / 669.4 / 669.4 | — |
| 文字 | faiss-hnsw-sq8 | 1 / 2 / 3 | 106.5 / 106.9 / 105.3 | 89.4 / 87.9 / 88.0 | 87.0 / 87.0 / 87.0 | 755.9 / 756.7 / 754.3 | — |

配對的記憶體差（ΔM = FP32 − SQ8，MiB；照抄報告「讀值 2」的 ΔM 表）：

<!-- check:quant-delta-m -->
| 資料集 | 配對 | ΔM | 排除 memory / swap 標記 | 全部 build |
| --- | --- | --- | --- | --- |
| GloVe | S 對 F | delta_m014 | 28.94 [28.45, 32.23]（5） | 28.94 [28.45, 32.23]（5） |
| GloVe | S 對 F | delta_relief | 29.41 [28.52, 32.45]（5） | 29.41 [28.52, 32.45]（5） |
| GloVe | S 對 F | delta_struct | 28.61 [28.61, 28.61]（5） | 28.61 [28.61, 28.61]（5） |
| GloVe | Q 對 P | delta_m014 | -11.25 [-12.48, -11.06]（3） | -11.25 [-12.48, -11.06]（3） |
| GloVe | Q 對 P | delta_relief | 28.00 [27.77, 28.12]（3） | 28.00 [27.77, 28.12]（3） |
| GloVe | Q 對 P | delta_struct | 28.61 [28.61, 28.61]（3） | 28.61 [28.61, 28.61]（3） |
| 文字 | S 對 F | delta_m014 | 219.83 [219.16, 220.23]（5） | 219.83 [219.16, 220.23]（5） |
| 文字 | S 對 F | delta_relief | 219.81 [219.00, 220.52]（5） | 219.81 [219.00, 220.52]（5） |
| 文字 | S 對 F | delta_struct | 219.72 [219.72, 219.72]（5） | 219.72 [219.72, 219.72]（5） |
| 文字 | Q 對 P | delta_m014 | 206.28 [206.19, 208.09]（3） | 206.28 [206.19, 208.09]（3） |
| 文字 | Q 對 P | delta_relief | 219.17 [218.53, 220.22]（3） | 219.17 [218.53, 220.22]（3） |
| 文字 | Q 對 P | delta_struct | 219.72 [219.72, 219.72]（3） | 219.72 [219.72, 219.72]（3） |

### 6.11 hygiene 標記清單

以各 run 的 `run.json` 為準（子 process 的 `flags`、build 的 `flags`、每個量測點的 `flags`）；來源欄取自 019 的完成補記。**文字的 5 個 run 沒有任何標記。** GloVe 的標記如下（表中沒有列出的臂與點都沒有標記）：

<!-- check:quant-hygiene -->
| seed | 臂 | 子 process | build | 量測點 | 背景來源（019 完成補記） |
| --- | --- | --- | --- | --- | --- |
| 1 | S（`rust-hnsw-sq8`） | `background` | — | ef 512 `load` | macOS 的 `duetexpertd`（持續 4 個區間） |
| 2 | P（`faiss-hnsw`） | — | `load` | ef 64 `load`、ef 96 `load` | — |
| 3 | F（`rust-hnsw-fp32g`） | — | — | ef 96 `load`、ef 128 `load` | — |
| 4 | F（`rust-hnsw-fp32g`） | `background` | — | — | `ugrep`（持續 6 個區間） |
| 5 | F（`rust-hnsw-fp32g`） | `background` | — | ef 304 `load`、ef 384 `load`、ef 512 `load`、ef 768 `load` | 一個不屬於本次 run 的 `.venv/bin/python`（平均約 61% CPU；紀錄沒有記下它屬於哪個專案） |
| 5 | S（`rust-hnsw-sq8`） | — | — | ef 64 `load`、ef 128 `load`、ef 152 `load`、ef 192 `load`、ef 256 `load`、ef 304 `load` | — |

- 這些負載都是**沒有取 lock 的其他工作**；紀錄只證明它們不屬於本次 run。`machine_quiet` 只決定開始的時機，run 期間出現的負載由 runner 照實標記（019 §5）。
- 影響：e\* = 304 時，seeds 1（S 的子 process `background`）、4（F `background`）、5（F `background` 加 ef 304 `load`，S ef 304 `load`）的配對都不乾淨，只剩 seeds 2、3 兩個乾淨配對 → H1b、H1c 無法判定；seed 1 的 S 被標記 → H1d 無法判定。GloVe seed 5 的 S 是 ef 64、128–304 為 `load`，ef 96 沒有標記。
- 依事先登記的規則，complete 的 run 不得重跑；專案作者決定不補 seed（019 完成補記 §3）。

### 6.12 附帶預期（不作判定，019 §2）

<!-- check:quant-expectations -->
- c_文字 > c_GloVe：觀察 {'glove-100-angular': 0.98048, 'simplewiki-embeddinggemma2-100k': 0.9935} → 符合
- 文字的 S 在 ef ≤ 512 內達到 0.99：觀察 256 → 符合
- GloVe 的 S 在 ef ≤ 768 內達到 0.95：觀察 384 → 符合
- GloVe 的 0.99：所有臂都未達到：觀察 {'rust-hnsw-fp32g': None, 'rust-hnsw-sq8': None, 'faiss-hnsw': None, 'faiss-hnsw-sq8': None, 'rust-hnsw': None} → 符合

## 7. 本文件沒有轉錄的內容

以下內容只在本機報告中（章節名照報告）：

- Core：每個 (n, M, efC, ef) 的完整 seed variance 表、輔助 scaling 讀值（每個 n 只用 (16, 200)）、frontier 的 index RSS 欄、其他 (M, efC) 的 scaling 列、insertion order variance 的 QPS 欄、逐點的被標記清單、圖。
- Advanced C：每個 ef 的 H2 分解、每個 ef 的配對讀值（ρ 與 ΔRecall）、每個 ef 的 recall 表、記憶體表的 rss 增量與 build time、一致性檢查的完整數字（GloVe P、Q 以外的 allocator 保留量）、天花板的 codes SHA-256、圖。
- `run.json` 與 `report.json`（含本機路徑與其他程序的命令列）。
