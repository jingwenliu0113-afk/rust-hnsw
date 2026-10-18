# rust-hnsw

[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
![Rust edition 2024](https://img.shields.io/badge/rust-edition%202024-orange.svg)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
![macOS arm64](https://img.shields.io/badge/platform-macOS%20arm64-lightgrey.svg)

[English](README.md) | **繁體中文**

研究問題：哪些演算法與低階實作決策，真正決定 HNSW 的 recall、延遲與記憶體取捨？
完整規格見 [`SPEC.md`](SPEC.md)；開發過程的工作日誌見 [`NEXT.md`](NEXT.md)。decision logs 與其他詳細文件都是繁體中文（導覽見[文件導覽](#文件導覽)）。

## 60 秒看懂

- **問題**：從零實作 HNSW 時，哪些演算法與低階實作決策決定 recall、延遲與記憶體的取捨？自己的實作和成熟的 hnswlib、FAISS 相比，差在哪裡、差多少？
- **方法**：
  - 以 Rust 從零實作 HNSW（cosine；insert、search、save / load）。`hnsw` crate 沒有任何外部相依；以 PyO3 提供 Python binding。
  - rust-hnsw、hnswlib 0.8.0、FAISS HNSW（faiss-cpu 1.15.1）走同一層 Python batch interface、單執行緒；每個 build 在全新的子 process 中量測（見[公平性與方法](#公平性與方法)）。
  - 資料：標準 ANN 資料集 glove-100-angular（100 維），以及真實文字 embedding（Simple English Wikipedia + `google/embeddinggemma-2`，768 維）。
  - 比較規則、Gate 的讀法與 Advanced C 的假設，都在看到對應的結果之前事先登記（decision 010、015、019；時序見[研究紀律](#研究紀律)）。
- **主要結果**（n = 100,000、目標 Recall@10 ≥ 0.95 的 frontier 讀值；數字的來源報告寫在各節；用語見[詞彙表](#詞彙表)）：
  - QPS：GloVe 上 rust-hnsw 2,537、hnswlib 3,135、faiss-hnsw 5,650；文字上 2,387、2,541、6,811。
  - Gate 條件 3「同 recall 下與 hnswlib 有差距」依事先登記的 015 **不成立**：兩個資料集的主要目標都是 `within_quantization`（在 ef 梯子的解析度內無法區分；與 vector quantization 無關）。
    看到結果之後的觀察（不是判讀）：同 ef、同 recall 時，hnswlib 在 GloVe 上快 1.22–1.38 倍，在文字上只快 6–8%。
  - Gate profiling（觀察，沒有以介入實驗驗證）：100 維時，rust-hnsw 的 search 有 43.6% 的 self time 在 visited set（`std::collections::HashSet`，預設的 SipHash hasher）；768 維時距離計算占絕大部分。
  - **Advanced C（SQ8 vector quantization）**：專案作者依 SPEC 的特例決定進入，**不代表 Gate 通過**。
    H1a 成立、H1b 無法判定、**H1 整體無法判定**；**H2 成立**（SQ8 的 recall 損失主要來自最終排序，不是走訪）。
    主要記憶體指標 M_014：文字 318.8 → 99.0 MiB、GloVe 105.4 → 76.1 MiB；速度的取捨在事先登記的讀法下是 `uncertain` 或 `within_quantization`（見 [Advanced C](#advanced-cvector-quantizationdecision-019)）。
- **驗證範圍與限制**：
  - 一台 Apple M4 Pro（macOS 27.0.1）、單執行緒、主要 n = 100,000。量測用到 macOS 專屬的系統介面，沒有在其他平台執行。
  - GloVe 的 C 有 3 個 seed（1、4、5）被背景負載標記，H1b 因此無法判定；GloVe n = 1,183,514 那一格沒有主要讀值。
  - 沒有做 Advanced A、B 與任何 Optional（SIMD、平行建置、x86 等）；見[限制、沒做的事與 Future work](#限制沒做的事與-future-work)。
  - 正式報告與原始 run 不進版控；關鍵表格逐值轉錄在 [`docs/results.md`](docs/results.md)。
- **狀態（2026-10-19）**：Core 與 Advanced C 都已完成，Advanced 已封版。

## 目前狀態

- **Core 完成**（2026-10-16）：
  - HNSW insert / search、save / load、Python binding、hnswlib / FAISS adapters、benchmark runner、兩個資料集的正式 run 與報告、Gate profiling。
  - 結果見下面各節。
  - Gate 依事先登記的讀法（decision 015），條件 3「同 recall 下與 hnswlib 有差距」**不成立**（見 [Gate 四條件總表](#gate-四條件總表)）。
- **Advanced C：vector quantization 完成並封版**（2026-10-19）：
  - 專案作者依 SPEC 的特例決定進入，**不宣稱原本的 Gate 通過**。
  - 假設、baseline、評測協定與停止條件都在任何量化結果之前事先登記（decision 019）。
  - 結果見「Advanced C」一節。
  - 專案作者決定 C 做完就封版：Advanced 不再做其他實驗，也沒有做任何 Optional。
- **真實資料上的 correctness**（SPEC Core 3）：通過，見[下方一節](#真實資料上的-correctness)。

## Gate 四條件總表

SPEC 規定進入 Advanced 的四個條件。判讀依事先登記的 decision 015 §1；來源是 NEXT.md 的「Gate 的四個條件」與兩份 Core 報告的「Gate 讀值」一節
（`results/reports/glove-100-angular/report.md`、`results/reports/simplewiki-embeddinggemma2-100k/report.md`；本機，不進版控，轉錄見 [`docs/results.md`](docs/results.md) §5）。

| 條件（SPEC） | 判讀 | 依據 |
| --- | --- | --- |
| 1. correctness 通過 | 成立 | `./scripts/test.sh` 全部通過（含小型人工可驗證案例，decision 004）；真實資料的 correctness 通過 |
| 2. Recall 隨 efSearch 整體改善或飽和，沒有系統性惡化 | 成立 | 兩個資料集、每個 (n, M, efC) 都沒有系統性惡化（報告的 010 §6 檢查，最大相鄰下降 0） |
| 3. 同 recall 下與 hnswlib 有差距 | **不成立** | GloVe 主要目標 0.95 為 `within_quantization`（0.90 為 `uncertain`、0.99 為 `within_quantization`）；文字三個目標都是 `within_quantization`。依 015，沒有任何資料集的主要目標是 `gap`，所以條件不成立 |
| 4. profiler 已顯示時間主要花在哪 | 成立 | 12 次取樣都通過 `gate_profile check`（見 [Gate profiling](#gate-profiling時間花在哪decision-009015-2)） |

- `within_quantization`、`uncertain` 都**不**代表「很接近 hnswlib」（015 §1）。詞彙見[詞彙表](#詞彙表)。
- SPEC 的特例是：「若 Core 已很接近 hnswlib：不必硬做 optimization attribution，Advanced 改聚焦 vector quantization frontier」。
  依 015，是否適用這個特例由專案作者判斷。
- **進入 Advanced C 是專案作者 2026-10-16 依這個特例做的決定**，不是 015 的讀值，**不代表 Gate 通過**。
  Core 的正式結果、Gate 的判讀與事先登記的比較規則都沒有因此修改。

## 公平性與方法

以下的量測與比較規則都在第一個 formal run 之前、也就是看到任何正式結果之前登記（decision 002、007、008、010）。電力條件之後有演變（014 在第一個 formal run 之前，016–018 在 GloVe 的正式結果之後）；唯一一項 Advanced C 的規則（019 §5）在出現處另外標明。完整說明見 [`docs/architecture.md`](docs/architecture.md) 第 3 節。

- **同一層 interface**（002）：三個實作只提供 `build(data)` 與 `search(queries, k)`。每次 `search` 是一次 FFI 呼叫、處理整批 query，adapter 內沒有 per-query 的 Python 迴圈。
  三者吃同一份 float32、C-contiguous、L2 正規化的 unit vectors（在計時之外準備一次）。P50 / P95 以 batch size 1 逐筆呼叫**同一個介面**量測，所以三者付出相同的 Python / FFI 成本。
- **單執行緒**（007 §6）：hnswlib 每次呼叫都傳 `num_threads=1`；FAISS 的 OpenMP 設定是 per-thread，在 build 與每次 search 開頭都設定。測試以「其他 thread 消耗的 CPU time」驗證設定真的生效。
- **子 process 隔離**（008）：每個（build、實作）在全新的 interpreter 中量測，只 import 受測的 library；recall 由父 process 在子 process 結束後計算。
- **交錯**（008）：外層迴圈為 build、內層為實作，第 g 個 build 把實作的順序輪換 g 位，讓時間漂移不與「實作」重疊。
  單一 build 的 config（GloVe 的 M8 與 1,183,514，以及 Advanced C 的 10 份）只有一種順序，每個單一 seed 的格點也是（013 §5 照實註明）；Advanced C 以 config 之間的手動位移平衡成對臂的先後（019 §5，在 Core 的正式結果之後、任何量化結果之前登記）。
- **warm-up 與重跑**（`benchmarks/protocol.py`）：計時 build 之前先做 library warm-up；每個 ef 先整批 warm-up 1 次，QPS 重跑 5 次；latency 先以前 100 個 query warm-up，再重跑 3 次。
- **只在相近的 Recall@10 下比較**（010；SPEC §4）：frontier 讀值不內插、不外推，沒有達標就記「未達到」；比值只在兩者都達標時計算。
- **被標記的點不進主要讀值**（010 §3）：有 hygiene 標記的點照常報告、不會被悄悄重跑；排除與否會改變讀值時，兩種讀值都列。
- **版本 pin**（007 §9）：hnswlib 0.8.0、faiss-cpu 1.15.1、numpy 2.5.3、maturin 1.15.0（`requirements-dev.txt`）；兩個 library 的實測語意由 `tests/test_third_party_semantics.py` 釘住。
- **SIMD 與編譯旗標**（decision 007 §8，規定要與結果一起報告；runner 也寫進每份 run.json）：

  | 實作 | 編譯 | distance kernel |
  | --- | --- | --- |
  | hnswlib 0.8.0 | PyPI 只有 sdist，在本機編譯；依其 `setup.py` 為 `-O3 -march=native`（推論，不是觀察到的） | arm64 上**沒有手寫 SIMD**（`USE_SSE` / `USE_AVX` 只在 x86 定義）。反組譯可見乘法被自動向量化（`fmul.4s`），加總仍是依序的純量 `fadd` |
  | faiss-cpu 1.15.1 | 官方 NEON wheel（`get_compile_options()` 為 `OPTIMIZE DD ARM_NEON MAC_METAL`） | NEON |
  | rust-hnsw | `maturin develop --release` 的預設（opt-level 3，**沒有** `target-cpu=native`，沒有 LTO；008 §11） | 純量 f32 迴圈（SIMD 屬 Optional，沒有做） |

- **已知的不對稱**（照實保留）：
  - 三者建出的 graph 不等價：FAISS 的新點在 layer 0 最多選 2M 個鄰居、插入順序也不是 row 順序（007 §4）。所以同 (M, efC, ef) 的 recall 曲線與 build time 不能直接對照，build time 只作描述。
  - rust-hnsw 的 insert 與 query 內部會再正規化一次，對自己略為不利（002）。
  - hnswlib 結果不足 k 時整批 raise，adapter 不改成逐筆查詢；FAISS 在 `efSearch < k` 時語意不同，所以 runner 強制 `ef ≥ k`（007）。
- **機器與條件**：所有正式量測都在同一台 Apple M4 Pro（8P + 4E）、48 GB、macOS 27.0.1 上，在 Mac heavy lock 下、AC 電源、固定的電力模式（decision 014、016–018）。

**SPEC 的指標在哪裡**：

| SPEC 指標 | 位置 |
| --- | --- |
| Recall@10、QPS、P50、P95 | 本頁各 frontier 表；完整讀值見 [`docs/results.md`](docs/results.md) §3.1、§4.1 |
| Recall@1 | [`docs/results.md`](docs/results.md) §3.2、§4.2（frontier 點的 Recall@1；另以每個 ef 一次 k = 1 的 search 取得，008 §7） |
| build time、peak RSS | 本頁各資料集的「build time 與記憶體」 |
| index size | index memory = build 前後 phys_footprint 的增量（decision 014）；原生檔案大小不用於跨實作的記憶體結論（008） |
| scaling with N | 本頁的 scaling 圖；[`docs/results.md`](docs/results.md) §3.8、§4.8 |
| seed / insertion order variance | [`docs/results.md`](docs/results.md) §3.5、§3.6、§4.5、§4.6 |

## 詞彙表

| 詞 | 意思 | 出處 |
| --- | --- | --- |
| Core | SPEC 的必做階段：以 Rust 從零實作 HNSW（insert、search、save / load；cosine）、Python binding、以 hnswlib / FAISS HNSW 為 baseline，以及在一個標準 ANN 資料集與一個真實文字 embedding 資料集上的正式 benchmark | SPEC「Core」 |
| Gate | SPEC 從 Core 進入 Advanced 的四個條件：(1) correctness 通過；(2) recall 隨 efSearch 整體改善或飽和，沒有系統性惡化；(3) 同 recall 下與 hnswlib 有差距；(4) profiler 已顯示時間主要花在哪。依 015 判讀；條件 3 不成立（見 [Gate 四條件總表](#gate-四條件總表)） | SPEC「Gate」、015 §1 |
| SPEC 的特例 | 「若 Core 已很接近 hnswlib：不必硬做 optimization attribution，Advanced 改聚焦 vector quantization frontier」。依它進入 Advanced C 是專案作者的決定，不代表 Gate 通過 | SPEC「Gate」的特例 |
| Advanced C | SPEC 的 Advanced 實驗 C，vector quantization：FP32 對 8-bit scalar quantization 的表示（SQ8），看 memory、recall 與 latency。是唯一做了的 Advanced 實驗 | SPEC「Advanced」、019 |
| frontier 讀值 | 對每個實作與目標 Recall@10（0.90 / 0.95 / 0.99）：在所有量測的 (M, efC, ef) 點中，跨 seed median Recall@10 ≥ 目標的點裡 QPS 最高者（P50、P95 與記憶體各自取最佳的點）。不內插、不外推；沒有點達標時記「未達到」 | 010 §2 |
| R / F / S / P / Q | Advanced C 的比較臂。R = `rust-hnsw`（Core，錨點）、F = `rust-hnsw-fp32g`（FP32，複製的搜尋路徑）、S = `rust-hnsw-sq8`（SQ8，與 F 同一張圖、同一條路徑）、P = `faiss-hnsw`（FAISS FP32）、Q = `faiss-hnsw-sq8`（P 的圖移植進 `IndexHNSWSQ`）；表見 [Advanced C](#advanced-cvector-quantizationdecision-019) | 019 §2、§3 |
| H1 | Advanced C 的 latency 假設。rust-hnsw 的距離 kernel 沿著長度為 d 的依序 f32 相依鏈加總；SQ8 把每個分量從 4 bytes 減為 1 byte，但不縮短這條鏈。所以 768 維（文字）時每次距離評估的成本應該不變（**H1a**：每個乾淨配對的 ρ_eval 都在 [0.90, 1.10]），100 維（GloVe）時鏈短，成本應該下降（**H1b**：每個乾淨配對的 ρ_eval ≥ 1.05）。**H1c** 對比兩個維度，**H1d** 以 profiler 檢查 kernel 時間（κ，seed 1）；H1a–H1c 各自至少要 3 個乾淨配對。前提（兩個 kernel 的加總都是一條依序的純量鏈）先以反組譯檢查 | 019 §2 |
| H2 | Advanced C 的 recall 假設：在同一張圖上，SQ8 的 Recall@10 損失主要來自最終排序（表示本身的距離誤差把候選中真正的近鄰擠出前 10 名），不是來自走訪（SQ8 距離改變了哪些點被找進 ef 個候選） | 019 §2 |
| `within_quantization` | 兩者的 QPS 範圍重疊，在 **ef 梯子的量化**之內無法區分。這裡的 quantization 指 ef 只量了離散的幾個值，**與 Advanced C 的 vector quantization（SQ8）無關**，也不代表「很接近 hnswlib」 | 015 §1 |
| `uncertain` | QPS 範圍分開，但某個事先附上的 variance 格點的 seed 範圍沒有分開（或缺資料、上界未知、未標記的 build 不足） | 015 §1 |
| `gap` / `rust_faster` | QPS 範圍分開、方向為 hnswlib 較快（`rust_faster` 則相反），而且每個附上的 variance 格點的 seed 範圍都分開。Advanced C 的對應代碼為 `sq8_faster` / `fp32_faster` | 015 §1、019 §6 |
| `undetermined` | 任一方沒有達到目標 | 015 §1 |
| QPS 範圍 | 下界是 frontier 讀值；上界是各 (M, efC) 格點第一次達標的 ef 的**前一個** ef 的 QPS（取最大）。用來處理「真正的交點落在兩個量測的 ef 之間」 | 015 §1 |
| hygiene 標記 | 被標記的點照常報告，但不進主要讀值。`load`：該點或 build 結束時 1 分鐘 load 超過 4.0。`background`：子 process 期間，其他 process 同一 pid 整段平均 ≥ 50% CPU、同一 command 合計 ≥ 100%，或同一 pid 連續 4 個區間 ≥ 50%；只說明有不屬於本次 run 的負載，不說明那是什麼工作。另有 `thermal`、`memory`（build 期間被壓縮的量過大）、`swap`、`power`（不是 AC）、`multithread`、`nondeterministic` | 008 §9、014 與補記 |
| e\* | Advanced C 判讀用的固定 ef：Core 正式報告中 (100k, 16, 200) 的 rust-hnsw，跨 seed median Recall@10 第一次 ≥ 0.95 的 ef。GloVe 304、文字 48 | 019 §2 |
| ρ_eval | ρ_QPS × D_S / D_F，即 S 與 F 每秒完成的距離評估數之比。ρ_QPS = QPS_S / QPS_F；D 是每個 query 平均呼叫距離 kernel 的次數（由不計時的診斷取得） | 019 §2 |
| κ | kernel 每次評估的時間比 t_k(S) / t_k(F)，t_k(X) = f_k(X) / (QPS_X · D_X)，f_k 是 kernel 類別的 self-time 占比。只取 seed 1 的 profiling，供 H1d 使用 | 019 §2、§8 |
| ν | 非 kernel 工作每次評估的時間比。只是診斷，不作判定 | 019 §2 |
| L_total / L_trav / L_rank | L_total = R_F − R_S（總損失）、L_trav = T_F − T_S（走訪造成）、L_rank = T_S − R_S（排序造成）。R 是 formal 的 Recall@10；T 是把 ef 個候選以精確距離重排後的 Recall@10。H2 判定表中「成立」的條件是所有有效 seed 的 L_trav ≤ 0.25 × L_total + 0.001（先排除有效 seed 不足等情形） | 019 §2 |
| M_014 / M_relief | M_014：index memory 的主要指標，build 前後 phys_footprint 的增量（decision 014）。M_relief：子 process 最後呼叫 `malloc_zone_pressure_relief` 之後的增量，只是診斷 | 014、019 §6 |
| 乾淨配對 | 同一個 seed、同一個 ef 的 F 與 S，兩者的子 process、build 與該點都沒有 hygiene 標記。H1 的時間類讀值只用乾淨配對，H1a–H1c 每個資料集至少要 3 個 | 019 §2 |
| 協定 008-v1–v5 | v1：最初的 runner（只有 smoke run）。v2（014）：formal 的電力條件、電量下降的作廢規則、背景 workload 的 gate 與 `background` 標記、index memory 改以 phys_footprint 為主；GloVe 的 Core formal。v3（017）：取消電量下降的作廢規則；沒有任何 run 在 v3 下執行（018）。v4（018）：允許 Apple 140W 變壓器（同一份報告不混用）；文字的 Core formal。v5（019）：只用於 Advanced C，新增實作名稱與子 process 最後的 relief 診斷，計時常數、gate、標記與電力條件與 v4 相同。報告拒絕合併不同協定版本的 run | 008 補記、014、017、018、019 |

## 真實資料上的 correctness

SPEC Core 3。以 smoke 模式在真實資料上確認：三個實作的 Recall@10 都隨 efSearch 單調上升，沒有系統性惡化，也沒有結果不足 k。smoke 的計時不作任何結論。
來源：本機 `results/runs/` 中的 smoke run 與其報告（不進版控）。NEXT.md 記錄 GloVe 100k 與文字 pilot 的 run ID；GloVe 1,183,514 的 smoke run（`20261008T154054Z-smoke`）記在 decision 013 的表，
文字 100k 的 smoke run（`text-correctness.toml`、`20261008T183825Z-smoke`）記在 013 的補記（run ID）與 012 的補記（recall）。

| 資料集 | n | M / efC | Recall@10（ef = 10 → 64 → 256；三者相近） |
| --- | --- | --- | --- |
| glove-100-angular | 100,000 | 16 / 200 | 約 0.55 → 0.83 → 0.95 |
| glove-100-angular | 1,183,514 | 16 / 200 | ef = 64 → 256 → 1024：約 0.76 → 0.88 → 0.955 |
| 文字 pilot（768 維） | 10,000 | 16 / 200 | 約 0.79 → 0.98 → 0.999 |
| 文字（768 維，正式資料集） | 100,000 | 16 / 200 | 約 0.77 → 0.97 → 0.996 |

完整 GloVe 上，ann-benchmarks 自帶的 neighbors 與本專案以 exact search 產生的 ground truth 完全一致。

下面兩張圖來自**正式 run**（不是上表的 smoke）：(n, M, efC) = (100,000, 16, 200) 的 Recall@10 隨 efSearch，每個 ef 是跨 seed 的 median，誤差棒是 min–max。
由 `benchmarks/report.py` 從本機的正式 run 產生（出處見 [`docs/figures/README.md`](docs/figures/README.md)）。數字以報告的表格為準，圖只呈現形狀。

![GloVe n=100k、(16, 200) 的 Recall@10 隨 efSearch（正式 run）](docs/figures/glove-100-angular/recall_vs_ef_n100000_m16_efc200.png)
![文字 n=100k、(16, 200) 的 Recall@10 隨 efSearch（正式 run）](docs/figures/simplewiki-embeddinggemma2-100k/recall_vs_ef_n100000_m16_efc200.png)

## 正式結果：glove-100-angular（標準 ANN 資料集）

條件：Apple M4 Pro（8P + 4E）、48 GB、macOS 27.0.1；協定 008-v2、高效能電力模式、Apple 70W、Mac heavy lock；三個實作都透過同一層
Python batch interface、單執行緒（decision 002、008）。8 份 formal config（decision 013）、21 個 build × 3 個實作，完整報告由
`benchmarks/report.py` 產生（`results/` 不進版控）。所有讀值依事先登記的 decision 010：id-based Recall@10、跨 seed median、
frontier 取點不內插、有 hygiene 標記的點不進入主要讀值。

數字來源：本機的 `results/reports/glove-100-angular/report.md`（不進版控）；SHA-256 與關鍵表格的轉錄見 [`docs/results.md`](docs/results.md) §1、§3。

![GloVe n=100k 的 recall frontier](docs/figures/glove-100-angular/frontier_n100000.png)

**n = 100,000 的 frontier 讀值**（每格附上該點的 (M, efC, ef)；QPS 有多個 seed 時附 [min, max]）：

| 目標 Recall@10 | 實作 | QPS | P50 (µs) | P95 (µs) | index memory，phys_footprint (MiB) |
| --- | --- | --- | --- | --- | --- |
| 0.90 | rust-hnsw | 4,656 [4,011, 4,972]（48 / 200 / 64） | 219（48 / 200 / 64） | 266（16 / 400 / 128） | 102.8（16 / 100） |
| 0.90 | hnswlib | 6,168（16 / 400 / 128） | 172（16 / 400 / 128） | 221（16 / 400 / 128） | 66.9（16 / 100） |
| 0.90 | faiss-hnsw | 9,423 [8,787, 9,479]（48 / 200 / 64） | 114（48 / 200 / 64） | 143（16 / 200 / 128） | 50.9（8 / 200） |
| 0.95 | rust-hnsw | 2,537 [2,334, 2,706]（48 / 200 / 128） | 414（48 / 200 / 128） | 491（16 / 400 / 256） | 102.8（16 / 100） |
| 0.95 | hnswlib | 3,135 [3,032, 3,153]（48 / 200 / 128） | 318（16 / 400 / 256） | 394（16 / 400 / 256） | 66.9（16 / 100） |
| 0.95 | faiss-hnsw | 5,650（32 / 200 / 128） | 184（32 / 200 / 128） | 223（32 / 200 / 128） | 55.8（16 / 400） |
| 0.99 | rust-hnsw | 1,008（32 / 200 / 384） | 991（32 / 200 / 384） | 1,213（32 / 200 / 384） | 115.6（32 / 200） |
| 0.99 | hnswlib | 1,363（32 / 200 / 384） | 797（32 / 200 / 384） | 1,064（32 / 200 / 384） | 80.3（32 / 200） |
| 0.99 | faiss-hnsw | 1,654（32 / 200 / 384） | 561（32 / 200 / 384） | 701（32 / 200 / 384） | 69.6（32 / 200） |

- **QPS 比值**（frontier 讀值）：hnswlib ÷ rust-hnsw = 1.32 / 1.24 / 1.35，faiss-hnsw ÷ rust-hnsw = 2.02 / 2.23 / 1.64（0.90 / 0.95 / 0.99）。
  解讀 FAISS 的倍數時，請一併看[公平性與方法](#公平性與方法)的 SIMD 與編譯旗標：FAISS 是 NEON wheel，rust-hnsw 是純量迴圈。
- **Gate 的判讀（decision 015，事先登記）**：主要目標 0.95 為 `within_quantization`，即「在 ef 梯子的量化範圍內，無法區分」——rust-hnsw 的 QPS 範圍 [2,537, 3,234]、
  hnswlib [3,135, 4,192] 重疊；0.90 為 `uncertain`（QPS 範圍分開，但交點前後的 seed 範圍沒有分開），0.99 也是 `within_quantization`。
  依 015，GloVe 上「同 recall 下與 hnswlib 有差距」**不成立**。
- **看到結果之後的觀察（不是事先登記的判讀）**：在 variance 格點上，兩者在**同一個 ef** 達到目標、Recall@10 相同到小數第 3–4 位時，
  5 組比較的 hnswlib seed 範圍都整段高於 rust-hnsw，median 比值 1.22–1.38。015 的規則刻意保守（容許 rust-hnsw 的交點落在更小的 ef），所以沒有判為有差距。
- **recall 隨 efSearch**：三個實作在每個 (n, M, efC) 都沒有系統性惡化；rust-hnsw 與 hnswlib 在相同 (M, efC, ef) 的 recall 幾乎相同。
- **build time 與記憶體**（(16, 200)，5 個 seed 的 median）：build 31.4 / 22.9 / 14.4 s，peak RSS 189.6 / 157.7 / 157.7 MiB，
  index memory 104.9 / 68.1 / 58.2 MiB（rust-hnsw / hnswlib / faiss-hnsw）。
  - 三者建出的 graph 不等價（decision 007），build time **只作描述**，不能直接對照。
  - rust-hnsw 的 peak RSS 包含 `add` 期間的一份正規化暫時副本（[decision 014](docs/decisions/014-formal-run-conditions-and-index-memory.md) 的 B1）。

![GloVe 的 scaling](docs/figures/glove-100-angular/scaling.png)

**限制（照實列出）**：

- n = 1,183,514 的三個 build 都因沒有取 lock 的其他工作（不屬於本次 run）被標記 `background`，依 010 §3 沒有主要讀值；圖中的 scaling 只有 100k 與 300k。
  complete 的 run 不得重跑（010 §3），專案作者也決定不補跑，所以這一格只有「包含被標記點的讀值」（[`docs/results.md`](docs/results.md) §3.9）。
- 300k 的 rust-hnsw 在 ef 512 / 768 有 `load` 標記，0.95 在主要讀值中「未達到」（包含標記點時為 915 QPS @ ef 512）。
- M = 8 的 rust-hnsw build（`background`）與 hnswlib 的全部點（`load`）被標記，兩者記憶體讀值的最小點因此落在 (16, 100)。
  包含被標記點時，0.90 與 0.95 的最小 index memory 是 94.0 MiB（rust-hnsw）與 63.2 MiB（hnswlib），都在 (8, 200)；frontier 表中只有這一欄改變（[`docs/results.md`](docs/results.md) §3.1）。
- 單一 seed 的格點（300k、1,183,514、M8、M32、efC 100 / 400）沒有自己的 seed 範圍；variance 只在 (16, 200) 與 (48, 200)。

## 正式結果：simplewiki-embeddinggemma2-100k（真實文字、768 維）

條件：同一台 Mac；協定 008-v4（decision 016–018：高效能電力模式、Apple 140W、電量只記錄）、Mac heavy lock、單執行緒 batch。
4 份 formal config、17 個 build × 3 個實作；先前因暫停、其他工作的負載、當時的電量下降規則、電力模式在本專案之外被改變與協定修訂而失敗或作廢的 7 個 run 都以 `--excluded` 列出（decision 016、017；[`docs/results.md`](docs/results.md) §4.10）。
資料：Simple English Wikipedia 的 passage，以 `google/embeddinggemma-2` 產生 document embedding（decision 011、012），base 100,000、query 2,000。

數字來源：本機的 `results/reports/simplewiki-embeddinggemma2-100k/report.md`（不進版控）；轉錄見 [`docs/results.md`](docs/results.md) §4。

![文字 n=100k 的 recall frontier](docs/figures/simplewiki-embeddinggemma2-100k/frontier_n100000.png)

**n = 100,000 的 frontier 讀值**（每格附上該點的 (M, efC, ef)；index memory 附 (M, efC)）：

| 目標 Recall@10 | 實作 | QPS | P50 (µs) | P95 (µs) | index memory，phys_footprint (MiB) |
| --- | --- | --- | --- | --- | --- |
| 0.90 | rust-hnsw | 3,203（16 / 100 / 32） | 311（16 / 100 / 32） | 397（8 / 200 / 64） | 310.4（8 / 200） |
| 0.90 | hnswlib | 3,509（8 / 200 / 64） | 285（8 / 200 / 64） | 362（8 / 200 / 64） | 316.4（8 / 200） |
| 0.90 | faiss-hnsw | 10,290（8 / 200 / 48） | 99（8 / 200 / 48） | 129（8 / 200 / 48） | 306.4（8 / 200） |
| 0.95 | rust-hnsw | 2,387（16 / 100 / 48） | 423（16 / 100 / 48） | 520（16 / 100 / 48） | 310.4（8 / 200） |
| 0.95 | hnswlib | 2,541（16 / 100 / 48） | 400（16 / 100 / 48） | 490（16 / 100 / 48） | 316.4（8 / 200） |
| 0.95 | faiss-hnsw | 6,811（16 / 100 / 48） | 152（16 / 100 / 48） | 182（16 / 100 / 48） | 306.4（8 / 200） |
| 0.99 | rust-hnsw | 934（48 / 200 / 64） | 1,083（48 / 200 / 64） | 1,358（48 / 200 / 64） | 310.4（8 / 200） |
| 0.99 | hnswlib | 844（32 / 200 / 96） | 1,203（32 / 200 / 96） | 1,468（32 / 200 / 96） | 316.4（8 / 200） |
| 0.99 | faiss-hnsw | 2,692 [2,689, 2,699]（16 / 200 / 128） | 380（16 / 200 / 128） | 437（16 / 200 / 128） | 306.4（8 / 200） |

- **QPS 比值**（frontier 讀值，n = 100k）：hnswlib ÷ rust-hnsw = 1.10 / 1.06 / 0.90，faiss-hnsw ÷ rust-hnsw = 3.21 / 2.85 / 2.88（0.90 / 0.95 / 0.99）。
  n = 10k、30k 時 hnswlib ÷ rust-hnsw 為 1.03–1.07，faiss-hnsw ÷ rust-hnsw 為 2.89–4.52。
- **variance 格點 (16, 200)**（hnswlib 5 個 build；rust-hnsw 4 個：seed 1 的 build 被標記 `background`，不進主要讀值；同一個 ef）：hnswlib 對 rust-hnsw 的 QPS median 為 3,276 對 3,103（ef 32）、2,439 對 2,299（ef 48）、799 對 743（ef 192），
  median 相差 6–8%；seed 之間的範圍多半 < 1%（hnswlib 在 ef 48、192 為 6–7%）。
- **被標記的 build**：(16, 200) 中，rust-hnsw 與 faiss-hnsw 的 seed 1（原始順序）build 被標記 `background`（[`docs/results.md`](docs/results.md) §4.9），所以這兩個實作在該格點的 median 來自 4 個 build。
  標記只說明有不屬於本次 run 的負載。包含被標記 build 時，Gate 的判讀相同（§4.4）；frontier 讀值只有 faiss-hnsw 的 0.99 改變：2,690 [2,458, 2,699]（5 builds），主要讀值是 2,692 [2,689, 2,699]（4 builds）（§4.1）。
- **Gate 的判讀（decision 015，事先登記）**：0.90 / 0.95 / 0.99 都是 `within_quantization`，即「在 ef 梯子的量化範圍內，無法區分」（主要目標 0.95：rust-hnsw [2,387, 3,203]、hnswlib [2,541, 3,401]）。
- **recall 隨 efSearch**：三個實作在每個 (n, M, efC) 都沒有系統性惡化，最大相鄰下降 0。
- **build time 與記憶體**（n = 100k，(16, 200) 的 median；rust-hnsw 與 faiss-hnsw 4 個 build、hnswlib 5 個）：build 154.7 / 147.7 / 56.6 s，peak RSS 950.5 / 667.4 / 668.4 MiB，
  index memory 318.8 / 323.3 / 312.5 MiB（rust-hnsw / hnswlib / faiss-hnsw；向量本身約 293 MiB）。
  - 三者建出的 graph 不等價（decision 007），build time **只作描述**，不能直接對照。
  - rust-hnsw 的 peak RSS 多出約一份資料，是 `add` 期間的正規化暫時副本（[decision 014](docs/decisions/014-formal-run-conditions-and-index-memory.md) 的 B1）。

![文字的 scaling](docs/figures/simplewiki-embeddinggemma2-100k/scaling.png)

**兩個資料集合起來（依 decision 015）**：兩個資料集的主要目標都不是 `gap`（有差距），所以 SPEC Gate 的「同 recall 下與 hnswlib 有差距」**不成立**；
「無法區分」不等於「很接近 hnswlib」，是否適用 SPEC 的特例（Advanced 改聚焦 vector quantization）由專案作者判斷（見 [Gate 四條件總表](#gate-四條件總表)）。
看到結果之後的描述（不是判讀）：100 維的 GloVe 上 hnswlib 在同 ef、同 recall 時快 1.22–1.38 倍，768 維的文字上只快 6–8%；FAISS 在兩者都最快（n = 100,000 的 frontier 讀值比值 faiss-hnsw ÷ rust-hnsw：GloVe 1.64–2.23 倍、文字 2.85–3.21 倍）。
時間花在哪裡見下一節的 Gate profiling；那是觀察性的歸因，沒有以介入實驗驗證這些差異的原因。

## Gate profiling：時間花在哪（decision 009、015 §2）

條件：兩個資料集各取 (n, M, efC) = (100,000, 16, 200)、seed 1，各實作在正式 run 中 Recall@10 ≥ 0.95 的最小 ef
（GloVe：rust-hnsw 304、hnswlib 304、faiss-hnsw 256；文字：三者都是 48）；macOS `sample` 每 1 ms 取樣，search 30 秒、build 300 秒；
heavy lock、formal 的電力與背景條件。12 次取樣都通過 `gate_profile check`：取樣時的 Recall@10 與正式 run **完全相同**，git 乾淨、被取樣的程式與正式 run 相同。
下表是 top-of-stack（self time）的占比；分類是看過取樣之後的解讀，原始排行在本機的 `results/profiles/`（`gate-*.top.txt`，不進版控）。

這是取樣 profiler 的**觀察**：它顯示時間分布，沒有以介入實驗（例如換掉 visited set 的實作）驗證因果；Advanced A、B 都沒有做。

| 資料集 / 階段 | rust-hnsw | hnswlib | faiss-hnsw（近似，見下） |
| --- | --- | --- | --- |
| GloVe search | `search_layer`（含被 inline 的距離計算）47.9%；visited set（`HashMap` insert + rehash，SipHash）**43.6%**；heap 7.4% | 距離計算 `InnerProductDistance` 55.5%；`searchBaseLayerST`（含陣列式 visited list）42.4% | ≈ `HNSW::search` 87.6%；≈ 距離 11.9% |
| GloVe build | `search_layer` 38.9%；visited set **38.4%**；`add` 12.4%；heap 7.4% | 距離 51.3%；`searchBaseLayer` 38.7%；鄰居選擇 5.2% | ≈ `add_links_starting_from` 68.3%；≈ 距離 20.1% |
| 文字 search | `search_layer` 84.8%；`greedy_search` 7.3%；visited set 6.4% | 距離 92.0%；`searchBaseLayerST` 7.5% | ≈ 距離 81.3%；≈ `HNSW::search` 18.2% |
| 文字 build | `search_layer` 69.3%；`add` 19.5%；visited set 6.8% | 距離 90.9%；`searchBaseLayer` 7.2% | ≈ 距離 78.0%；≈ `add_links_starting_from` 18.5% |

- rust-hnsw 的距離計算被 inline 進 `search_layer`，無法單獨拆出（009 已知限制）；拆開需要另寫 decision（例如 `xctrace` 搭配 debug line info）。
- faiss-cpu 的 `libfaiss.dylib` 沒有符號表，FAISS 的分類以「之前最近的 text 符號」近似（009 補記），只看大類。
- 看到取樣之後的描述（不是事先登記的判讀）：100 維時，rust-hnsw 約四成的時間花在 visited set：`std::collections::HashSet<u32>`（內部是 hashbrown 的 `HashMap`、預設的 SipHash hasher，所以 profiler 顯示為 `HashMap` 的 insert 與 rehash），
  hnswlib 用陣列式的 visited list，這部分不在它的排行前段；768 維時距離計算占絕大部分，兩者的 visited set 開銷都變小，與文字資料上兩者 QPS 相近一致。
  「一致」只是描述；visited set 是否就是 GloVe 上差異的原因，需要介入實驗才能回答，本專案沒有做。

## Advanced C：vector quantization（decision 019）

**為什麼做 C**：Gate 條件 3 依事先登記的 015 不成立；專案作者 2026-10-16 依 SPEC 的特例決定進入 Advanced C。這是專案作者的判斷，**不代表 Gate 通過**（見 [Gate 四條件總表](#gate-四條件總表)）。

**研究問題**：搜尋期間把向量從 FP32 換成 8-bit scalar quantization（SQ8），Recall、Latency 與 Memory 之間的取捨是什麼？
兩個假設、5 個比較臂、判定規則與停止條件，都在任何量化結果之前事先登記（[019](docs/decisions/019-advanced-c-vector-quantization.md)，commit 4c5cbc2）。

數字來源：本機的 `results/reports/quant-glove/report.md` 與 `results/reports/quant-text/report.md`（不進版控）、019 的第三則補記；轉錄見 [`docs/results.md`](docs/results.md) §6。

- **表示**：SQ8 用 per-dim min–max 編碼（與 FAISS 的 `QT_8bit` + `RS_minmax` 對齊），採非對稱距離：query 保持 FP32，資料是 u8 code。
  - 先以 Core 建好 FP32 的圖，再轉換表示，所以 SQ8 與 FP32 走的是**同一張圖**。
  - kernel 與 Core 一樣，是依序加總的純量 f32 迴圈，沒有手寫 SIMD。
- **比較臂**：

  | 代號 | 實作 | 意義 |
  | --- | --- | --- |
  | R | `rust-hnsw` | Core 本身（錨點，只跑 seed 1） |
  | F | `rust-hnsw-fp32g` | FP32，走新的搜尋路徑（kernel 以 `#[inline(never)]` 拆出，供 profiler 歸屬） |
  | S | `rust-hnsw-sq8` | SQ8，與 F 同一張圖、同一條搜尋路徑 |
  | P | `faiss-hnsw` | FAISS FP32 |
  | Q | `faiss-hnsw-sq8` | P 的圖移植進 `IndexHNSWSQ`（FAISS 的 SQ kernel 是 NEON SIMD，只作描述） |

- **條件**：
  - 資料與參數：兩個資料集各取 n = 100,000、(M, efC) = (16, 200)、原始順序；F、S 跑 seeds 1–5，P、Q 跑 seeds 1–3。
  - 量測：協定 008-v5、高效能電力模式、Apple 140W、Mac heavy lock。批次開始前，`machine_quiet` 連續 3 次成立（heavy lock 沒人持有、沒有 ≥ 50% CPU 的背景 process、1 分鐘 load < 4.0、GPU ≤ 20%、記憶體 free ≥ 30%）；它只決定開始的時機，run 期間的負載由 runner 照實標記（019 §5）。
  - run：10 份 formal config 全部 complete（2026-10-16 20:38 – 10-17 00:08 UTC）；報告由 `benchmarks/quant_report.py` 產生。
  - 正式 run 之前，真實資料的 correctness smoke 已確認：F、R、P 的 ids 與 Core 的正式結果逐位元相同。

**判定表**（事先登記的規則；完整的表與每一列的數字轉錄在 [`docs/results.md`](docs/results.md) §6.1–§6.5）：

| 列 | 判定 | 判定所用的數字 |
| --- | --- | --- |
| H1 前提：兩個 kernel 的加總都是依序的純量相依鏈（反組譯） | 成立 | FP32 累加 21 次、SQ8 69 次，各自只用一個累加暫存器 |
| H1a 文字：ρ_eval（S ÷ F 每秒完成的距離評估數）在 [0.90, 1.10] | **成立** | 5 個乾淨配對，[0.961, 1.092] |
| H1b GloVe：ρ_eval ≥ 1.05 | **無法判定** | 乾淨配對只有 2 個（規定至少 3 個）：1.29、1.55 |
| H1c 維度對比 | 無法判定 | 同上 |
| H1d profiler：kernel 每次評估時間的比值 κ | 無法判定 | GloVe seed 1 的 S 在 e\* 有 `background` 標記 |
| **H1 整體** | **無法判定** | |
| H2 GloVe：recall 損失來自排序，不是走訪 | **成立** | 5 個 seed：L_total ≈ 0.0138，L_trav ≤ 0.00011，門檻 ≈ 0.0044 |
| H2 文字 | **成立** | 5 個 seed：L_total ≈ 0.0046，L_trav ≤ 0.0006，門檻 ≈ 0.0021 |
| **H2 整體** | **成立** | |
| quantization frontier，S 對 F（GloVe 0.90 / 0.95） | `uncertain` / `uncertain` | |
| quantization frontier，S 對 F（文字 0.90 / 0.95） | `within_quantization` / `uncertain` | |
| quantization frontier，Q 對 P（GloVe 0.90 / 0.95；文字 0.90 / 0.95） | `uncertain` / `uncertain`；`within_quantization` / `within_quantization` | 包含被標記 build 時，GloVe 0.90 / 0.95 為 `within_quantization` / `within_quantization`（報告標為「不同」）；主要判讀中，P 在附上的 seed 範圍格點只有 2 個未標記的 build，少於每側要求的 3 個。兩種判讀並列（015 §1） |

quantization frontier 的判讀沿用 015 的 QPS 範圍與 seed 範圍規則（019 §6），是事先登記的描述性判讀，不是假設；這裡的 `within_quantization` 同樣指 ef 梯子的解析度，不是指 SQ8。

- **H1 無法判定的原因**：GloVe 的 seed 1 的 S、seeds 4、5 的 F 在子 process 層級被標記為 `background`（seed 5 另有 `load`），e\* = 304 只剩 seeds 2、3 兩個乾淨配對。
  - 來源是沒有取 lock 的其他工作：macOS 的 `duetexpertd`、`ugrep`，以及一個不屬於本次 run 的 python（紀錄沒有記下它屬於哪個專案）。紀錄只證明它們不屬於本次 run。
  - H1d 只看 seed 1：GloVe seed 1 的 S 在 e\* 被標記，所以無法判定，與乾淨配對的數目無關。
  - 依事先登記的規則，complete 的 run 不得重跑。專案作者 2026-10-19 決定照實報告，不補 seed。
- **一致性檢查**：
  - C2（FP32 重排）、C3（天花板）、C7a / C7b（與 Core 逐位元相同）、C8（FAISS 移植的 graph）都一致。
  - C4（新路徑對 Core 路徑）一致：F ÷ R 的 QPS 為 GloVe 0.97、文字 0.99。
  - C1（診斷用的 ΔM_relief，即 FP32 臂與 SQ8 臂的 M_relief 之差，對照理論值 3·n·d − 8·d；M_relief 只是診斷，主要記憶體指標 M_014 不由 C1 判定）：GloVe 的 F − S **不一致**：只有 seed 5 偏離 4,030,488 B（換算約 3.84 MiB），超過容許值 max(5% × Δ_theory, 2 MiB) = 2 MiB（2,097,152 B），其餘 seed 都在容許值內。GloVe 的 P − Q、文字的 F − S 與 P − Q 都一致。
  - C5（Gate 重現）兩個資料集都**不一致**，原因在 formal 之前就已登記：
    - 加入量化模組之後，Core 的原始碼沒變，但編譯器把 visited set 的 `HashMap::insert` inline 進了 `search_layer`；
    - 所以那部分時間改算在 traversal（019 補記第 5 節）。
  - C6 只描述：C 中 R 的 QPS ÷ Core 正式 run，GloVe 0.82–0.85、文字 1.09–1.13。這同時包含時間漂移與機器碼差異，兩者無法分開。

**frontier 讀值，Recall@10 ≥ 0.95**（跨 seed median，排除有 hygiene 標記的 build；M_014 = phys_footprint 增量，是主要記憶體指標）：

| 資料集 | 實作 | ef | QPS | P50 (µs) | M_014 (MiB) |
| --- | --- | --- | --- | --- | --- |
| GloVe | F（FP32） | 304 | 1,877（3 builds） | 547 | 105.4 |
| GloVe | S（SQ8） | 384 | 2,299（4 builds） | 434 | 76.1 |
| GloVe | P（FAISS FP32） | 304 | 3,350（2 builds） | 312 | 58.1 |
| GloVe | Q（FAISS SQ8） | 384 | 2,445（3 builds） | 423 | 69.8 |
| GloVe | R（Core，錨點） | 304 | 1,923 | 532 | 104.2 |
| 文字 | F | 48 | 2,277（5 builds） | 442 | 318.8 |
| 文字 | S | 64 | 1,769（5 builds） | 569 | 99.0 |
| 文字 | P | 48 | 6,234（3 builds） | 166 | 313.0 |
| 文字 | Q | 48 | 5,605（3 builds） | 186 | 106.5 |
| 文字 | R（錨點） | 48 | 2,292 | 440 | 318.6 |

- **事先登記的判讀**（判定表）：0.95 時 S 對 F 在 GloVe 與文字都是 `uncertain`；Q 對 P 在 GloVe 是 `uncertain`、在文字是 `within_quantization`。
  包含被標記 build 時，GloVe 的 Q 對 P 判讀為 `within_quantization`（不同；主要判讀中 P 只有 2 個未標記的 build，少於每側 3 個）。表中 S 與 F 的 QPS 高低是讀值，不是判定。
- **包含被標記 build 的讀值**在 GloVe 的 F、S、P 不同（報告標為「排除標記點是否改變讀值：是」）：例如 P 在 0.95 為 2,969（3 builds），主要讀值是 3,350（2 builds）；F 在 0.90 為 4,087（5 builds），主要讀值是 4,291（3 builds）；
  S 在 0.95 為 2,288（5 builds），主要讀值是 2,299（4 builds）。兩種讀值見 [`docs/results.md`](docs/results.md) §6.6，兩種判讀見 §6.7。文字沒有被標記的點。
- **C 的絕對 QPS 不與 Core 的表比較**：格點與時段不同，而且 C6 顯示同一個實作（R）在 C 中的 QPS 是 Core 正式 run 的 0.82–0.85 倍（GloVe）、1.09–1.13 倍（文字），時間漂移與機器碼差異無法分開。

![GloVe 的 recall–QPS（Advanced C）](docs/figures/advanced-c/glove/recall_qps.png)
![文字的 recall–QPS（Advanced C）](docs/figures/advanced-c/text/recall_qps.png)

*兩張 recall–QPS 圖排除被標記的點（與主要讀值相同），數字以上表為準；絕對 QPS 不與 Core 的圖比較（C6）。*

![GloVe 的 quantization frontier @ 0.95](docs/figures/advanced-c/glove/quant_frontier_0.95.png)

*GloVe：事先登記的判讀為 S 對 F 0.95 `uncertain`（Q 對 P 也是 `uncertain`）；H1b 無法判定。被標記的 build 已排除（F 3 builds、S 4 builds），所以圖上 S 比 F 快又小只是讀值，不能寫成「SQ8 在 100 維上更快」。*

![文字的 quantization frontier @ 0.95](docs/figures/advanced-c/text/quant_frontier_0.95.png)

*文字：事先登記的判讀為 S 對 F 0.95 `uncertain`（0.90 為 `within_quantization`；Q 對 P 都是 `within_quantization`）。F、S 各 5 builds，沒有被標記的 build。*

**profiling**（seed 1、e\*；分類規則在取樣之前固定；8 次取樣都通過 check，取樣時的 Recall@10 與正式 run 完全相同）：

| 資料集 | 實作 | kernel | visited set | traversal | heap | 每次評估的 kernel 時間 |
| --- | --- | --- | --- | --- | --- | --- |
| GloVe | F | 29.4% | 43.0% | 20.1% | 7.2% | 29.4 ns |
| GloVe | S | 29.2% | 39.2% | 22.4% | 8.9% | 23.2 ns |
| 文字 | F | 86.7% | 6.1% | 5.8% | 0.9% | 371.6 ns |
| 文字 | S | 85.6% | 6.6% | 6.3% | 1.0% | 374.1 ns |

- GloVe seed 1 的 S 在 e\* 有 `background` 標記：GloVe 的 S 那一列（以及由它算出的 κ、ν）只作描述，這也是 H1d 無法判定的原因。文字的 C run 沒有任何標記。
- 時間分解是取樣 profiler 依 019 §8 事先固定的分類規則所做的歸屬，是觀察，不是介入實驗。

![GloVe 的每 query 時間分解，F 對 S](docs/figures/advanced-c/glove/profile_breakdown.png)
![文字的每 query 時間分解，F 對 S](docs/figures/advanced-c/text/profile_breakdown.png)

**看到結果之後的描述**（不是事先登記的判讀，也不改變上表的任何一列）：

- **768 維（文字）**：SQ8 沒有讓每次距離評估變便宜：ρ_eval ≈ 1，κ = 1.01，與 H1 的相依鏈模型一致。
  - 要達到相同 recall，S 需要較大的 ef，frontier 讀值反而比 F 慢（0.95 時 QPS 1,769 對 2,277）；事先登記的判讀是 0.95 `uncertain`、0.90 `within_quantization`，所以這只是讀值，不是判定。
  - 換來的是 index memory：M_014 從 318.8 MiB 降到 99.0 MiB。
- **100 維（GloVe）**：
  - 所有 seed 的 ρ_eval 都 > 1.05，包括被標記的配對（1.15、1.26、1.29）。
  - seed 1 的 κ = 0.79，ν = 0.80：kernel 與 kernel 以外的工作都變快了。但 seed 1 的 S 在 e\* 有 `background` 標記（H1d 因此無法判定），這兩個值只作描述。
  - 依登記的規則，這些都不能判為成立。
- **recall 的代價**：在 e\*，SQ8 的 Recall@10 比 FP32 低約 0.014（GloVe）與 0.005（文字），幾乎全部來自最終排序（H2）。
  - 天花板（以 SQ8 距離做 exact search 能達到的 Recall@10）：GloVe 0.98048、文字 0.9935（C3）。
- **FAISS SQ8 在 GloVe 上的 M_014 反而比 FP32 大**：69.8 對 58.1 MiB。
  - 差別在 allocator 保留的 page：報告列出的 allocator 保留量（M_014 − M_relief，seeds 1–3），Q 為 46,972,952 / 47,661,080 / 46,972,952 B，P 為 4,522,008 / 6,570,008 / 6,062,080 B（[`docs/results.md`](docs/results.md) §6.4）。
  - relief 之後的診斷值 M_relief，Q 為 24.8 / 24.4 / 25.5 MiB，P 為 52.8 / 52.5 / 53.2 MiB（seeds 1–3，§6.10）；C1 的 P − Q 差值與理論值一致。
  - 文字上 Q 的 M_014 比 P 小，median 為 206.28 MiB（3 個 seed 的 ΔM_014，§6.10）。

**限制**：

- 結論只適用於 Rust 的依序純量 kernel、搜尋期間的表示、n = 100,000、(16, 200)。
- build time 與 peak RSS 不會下降：SQ8 先建 FP32 的圖再轉換。文字的 peak RSS，S 的 5 個 build 在 1,023.2–1,023.6 MiB，F 在 950.1–950.9 MiB（[`docs/results.md`](docs/results.md) §6.8）。
- C 沒有做 insertion order 的 variance（Core 已報告 (16, 200) 的 shuffles 101–105，轉錄見 [`docs/results.md`](docs/results.md) §3.6、§4.6）。
- C 中的 R 與 Core 正式 run 的原始碼相同、機器碼不同（inline 決策改變；`scripts/probe_core_codegen.py`）。
- FAISS 的 Q 對 P 只作描述：FAISS 的 SQ kernel 是 NEON SIMD，不屬於 H1 的機制。
- C 用了 Advanced 三個假設名額中的 2 個；第 3 個名額不保留（019 第三則補記）。

## 限制、沒做的事與 Future work

**整體的限制**

- **單一機器、單執行緒**：所有正式量測都在一台 Apple M4 Pro（8P + 4E）、48 GB、macOS 27.0.1 上，build 與 search 都是單執行緒。
  runner 用到 macOS 專屬的介面（phys_footprint、`sample`、`pmset`、`otool` 等），沒有在其他平台執行過；formal 模式只接受 Apple 原廠 70W 或 140W 變壓器。
- **規模**：主要結論在 n = 100,000。scaling 只有 (16, 200)；GloVe n = 1,183,514 那一格**沒有主要讀值**（三個 build 都被標記 `background`），300k 的 rust-hnsw 在 0.95 的主要讀值為「未達到」。
- **Gate**：條件 3 依 015 不成立；進入 Advanced C 是專案作者依 SPEC 特例的決定，不代表 Gate 通過。
- **profiling 只是觀察**：「100 維時 visited set 約占四成」來自取樣 profiler 的歸屬，沒有以介入實驗驗證。rust-hnsw 的距離計算被 inline，無法單獨拆出；FAISS 沒有符號表，只能近似。
- **Advanced C**：H1 整體無法判定；GloVe 的 seeds 1、4、5 被背景負載標記；C1 在 GloVe 的 F − S 不一致（seed 5）；結論的適用範圍見上一節。
- **無法從公開 repo 重產正式數字**：原始 run 與報告不進版控，而且數字與機器有關（[`docs/reproduce.md`](docs/reproduce.md)）。

**沒做的事**

| 項目 | 狀態 | 對現有結論的意義 |
| --- | --- | --- |
| Advanced A：memory representation（per-node allocation、arena / contiguous layout 等） | 未做 | rust-hnsw 的表示法刻意採最直接的寫法（004：每個節點各自的 `Vec`、每次搜尋新建的 visited set），它的影響沒有量 |
| Advanced B：optimization ablation | 未做 | 沒有逐項加入或拿掉 optimization 的介入證據，所以 profiling 的歸屬不能當成因果 |
| Optional：SIMD distance kernel | 未做 | rust-hnsw 是純量 f32 迴圈；與 FAISS（NEON）、hnswlib（乘法自動向量化）的差異是已知的不對稱（007 §8） |
| Optional：parallel index construction、concurrent queries | 未做 | 所有結果都是單執行緒 |
| Optional：ARM M4 Pro vs x86 Ryzen | 未做 | 只有一台 Apple Silicon；沒有使用遠端桌機 |
| Optional：更完整的 Python API | 未做 | binding 只提供 benchmark 需要的 batch API（add / search / save / load）與 C 的診斷方法 |
| C 的 insertion order variance | 未做 | Core 已報告 (16, 200) 的 shuffles 101–105 |
| GloVe 1,183,514 補跑、C 補 seed | 不做（專案作者決定） | complete 的 run 不得重跑；照實報告 |

**Future work**

專案作者已決定 C 做完就封版，以下都**沒有啟動**。任何一項若要做，都必須另寫 decision，事先登記假設、baseline、協定與停止條件，
並標明是在看到 C 的結果之後才提出，而且不得取代 019 的判定（NEXT.md、019 第三則補記）。

- 以介入實驗（Advanced A / B 的形式）檢驗 Gate profiling 觀察到的 visited set 開銷，是否就是 100 維上與 hnswlib 的差異來源。
- SIMD kernel、平行建置與 concurrent queries、x86 對 ARM 的比較（SPEC 的 Optional）。

## 研究紀律

這個專案最有說服力的地方是過程，不是速度：

- **先登記、再看結果**（commit 時間為台灣時間，UTC+8）：

  | 事先登記的內容 | commit | 時點 |
  | --- | --- | --- |
  | 010 比較規則與 variance 彙整 | 2b783bd（2026-10-08 18:55） | 第一個 formal run `20261009T080239Z-formal`（ID 中的時間是 UTC）之前 |
  | 013 正式格點 | 746c1d0（2026-10-09 01:41） | 第一個 formal run 之前 |
  | 014 電力條件、背景 workload 與 index memory 指標 | b73300d（2026-10-09 09:37） | 第一個 formal run 之前；index memory 的主要指標在看到任何正式結果之前決定 |
  | 015 Gate 證據的讀法與 Gate profiling 的參數 | 7a8c86a（2026-10-09 23:39）、審查修正 1fa384b（2026-10-10 00:00）；分支 `prereg-gate` | **GloVe 的 formal 1–4 完成之後**、看任何 formal 的效能數字或報告之前（015 照實記錄；GloVe 報告產生於 2026-10-10 17:58） |
  | 019 Advanced C 的假設、表示與評測協定 | 4c5cbc2（2026-10-16 21:31）；分支 `prereg-quant` | 任何 C 的程式之前（golden test 9cfc35f 21:32、實作 dca22c9 10-17 01:13）、任何真實資料上的量化結果之前（smoke 10-17 03:55 起）、C formal（10-17 04:38–08:08）之前 |

  - 015 不是「在任何 formal run 之前」寫的：撰寫時 GloVe 的 formal 1–4 已完成，撰寫者只看過這些 run 的狀態、hygiene 標記與背景工作的診斷紀錄，沒有看過 recall、QPS、latency、build time 或記憶體。
  - 019 用到 Core 的正式結果（e\* 與 ef 梯子）；這對 Core 是看過結果之後、對 C 是事前資訊，019 照實記錄。019 也揭露 M_relief 的提議參考了 Core 看到結果之後的觀察；主要記憶體指標在 019 commit 之前就決定維持 M_014。
  - 015 在 7a8c86a 登記之後經獨立審查，10 個 findings 全部採用（1fa384b，仍在看任何正式效能結果之前）；019 的草稿經獨立審查，17 個 findings 全部採用之後才登記。
- **作廢與排除的 run 都有紀錄**：
  - GloVe：1 個 aborted（macOS 的背景工作觸發 load gate），整份 config 重跑，原本的 run 以 `--excluded` 列出。
  - 文字：7 個——2 個 aborted、2 個 invalid（當時的電量規則），以及 3 個 complete 但依 decision 016、017 作廢（登記在 `report.py` 的 `VOIDED_RUNS`）。
  - 報告會拒絕：缺少或重複的 formal config、同一份 config 另有沒交出的 run、沒以 `--excluded` 列出的失敗 run、不同協定版本或電力模式的合併（013 補記、014）。所以不能只挑比較有利的 run。
- **complete 的 run 不重跑**：被標記的點照常報告、不悄悄重跑。GloVe 1,183,514 沒有主要讀值、C 的乾淨配對不足，都照實報告，沒有補跑（專案作者決定）。
- **失敗與無法判定照實寫**：Gate 條件 3 不成立、H1 整體無法判定、C1 與 C5 不一致，都寫在本頁與 decision log。
- **事後觀察標明**：看到結果之後的描述（例如 1.22–1.38 倍、SQ8 的快慢）一律標成「不是判讀」或「不是判定」，不改變任何事先登記的判讀。
- **證據核對**：profiling 取樣之後核對參數、lock、git、被取樣的程式與 Recall@10，與正式 run 完全相同才當作證據；H1 的前提以反組譯機械檢查。

## 工程亮點

| 項目 | 內容 | 在哪裡看 |
| --- | --- | --- |
| 從零實作、零相依 | `hnsw` crate 的 `[dependencies]` 是空的。library crate（`crates/hnsw/src`）與 binding（`python/src`）的原始碼都沒有 `unsafe`；`unsafe` 只出現在測試 `crates/hnsw/tests/load_alloc.rs` 的計數用 allocator | `crates/hnsw/Cargo.toml`、`crates/hnsw/src/` |
| 決定性 | level 亂數用自帶的 SplitMix64，避免外部 crate 升級改變序列；同 seed、同插入順序得到相同的 graph（測試） | `crates/hnsw/src/rng.rs`、`crates/hnsw/tests/hnsw.rs` |
| 版本化檔案格式 | 76-byte header（含 header CRC-32）→ vectors → graph → file CRC-32；load 串流讀取，配置量只隨實際讀到的資料成長；損壞與截斷都有測試 | decision 005、`crates/hnsw/src/hnsw/persist.rs`、`crates/hnsw/tests/persist.rs`、`load_alloc.rs` |
| batch binding | PyO3；只收 float32、C-contiguous 的 2-D array，不隱性轉型；每次 search 處理整批 query，計算期間釋放 GIL；`BUILD_INFO` 記錄編譯設定，formal runner 以它確認被量測的是 release build | `python/src/lib.rs`、`python/build.rs`、decision 006、008 |
| 量測隔離 | 每個（build、實作）一個全新的 interpreter，只 import 受測的 library；交錯輪換；recall 由父 process 計算 | `benchmarks/run.py`、`run_child.py`、decision 008 |
| hygiene 與 lock | 8 種 hygiene 標記；跨專案的 Mac heavy lock：formal run 拿不到 lock 就不跑，profiling 只有在 lock 下取樣才算 Gate 證據，都不搶占；formal 期間 repo 有任何變動，run 就記為 invalid | `benchmarks/environment.py`、`protocol.py`、`machine_quiet.py`、`tools/with-mac-lock` |
| 第三方語意釘住 | hnswlib / FAISS 的實測語意（回傳的量、tie、`efC < M`、per-thread 的 OpenMP 設定等）以測試釘住，library 升級時測試會紅燈 | `tests/test_third_party_semantics.py`、decision 007 |
| golden test | 在任何 C 的程式之前，釘住 Core 的 search 輸出與存檔 bytes 的 SHA-256 | `tests/test_core_golden.py`（9cfc35f） |
| 反組譯檢查 | H1 的前提（kernel 的加總是依序的純量相依鏈）以 `otool` 反組譯機械判定，在 formal 之前完成 | `benchmarks/quant_profile.py`（`premise`）、019 §2 |
| 文件與報告的核對 | `docs/results.md` 中加了核對標記的表，在本機有報告時與報告逐項比對。兩份 README 都以 `docs/results.md` 核對，也互相核對。公開文件（兩份 README、`docs/results.md`、`docs/architecture.md`、`docs/reproduce.md`、`docs/figures/README.md`、`docs/decisions/README.md`、`THIRD_PARTY_NOTICES.md`、`LICENSE`）不得含本機路徑或 email，相對連結與錨點都必須存在；decision log 本身不在核對範圍內 | `tests/test_results_doc.py`、`tests/test_readme_claims.py` |

## 快速上手（Python）

**前置**：macOS（Apple Silicon）、Xcode Command Line Tools（`xcode-select --install`）、[rustup](https://rustup.rs/)（含 rustfmt 與 clippy）、[uv](https://docs.astral.sh/uv/)。

- Rust：edition 2024 至少需要 Rust 1.85；實測版本是 1.99.0。程式也用到較新的標準函式庫 API，真正的最低版本沒有測過。
- Python 3.13（`python/pyproject.toml` 要求 ≥ 3.13；uv 可以自行下載）。

**最小的 Demo 路徑**（只需要 numpy 與 maturin，不需要 hnswlib、FAISS 或任何資料集；與 [`docs/reproduce.md`](docs/reproduce.md) §A 相同），在 repo 根目錄執行：

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python maturin==1.15.0 numpy==2.5.3
VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --uv -m python/Cargo.toml
.venv/bin/python examples/quickstart.py
```

- 根目錄的 `Cargo.toml` 是 virtual workspace，`pyproject.toml` 在 `python/` 底下，所以要以 `-m python/Cargo.toml` 指定；`VIRTUAL_ENV` 決定 binding 裝到哪個 venv。這與 `scripts/test.sh` 用的指令相同。
- 要跑全部的 correctness 檢查，改裝完整的 `requirements-dev.txt` 再跑 `./scripts/test.sh`（見[測試](#測試)）。

**範例輸出的樣式**（依 `examples/quickstart.py` 的 print 格式寫出；**不是實測輸出**，「…」處是每次執行才知道的數值，這裡刻意不填。資料是 synthetic，數字**不是研究結果**）：

```text
built 20000 vectors (dim 64, max level …) in … s
ef_search=  16  Recall@10=…  nearest distance of query 0=…
ef_search=  64  Recall@10=…  nearest distance of query 0=…
ef_search= 256  Recall@10=…  nearest distance of query 0=…
saved … bytes; reloaded index gives identical results: True
（synthetic 資料的 Demo，數字不是研究結果。）
```

20000 與 64 是 `--n`、`--dim` 的預設值；最後的 `True` 是 `tests/test_examples.py` 斷言的結果（讀回的 index 給出完全相同的結果）。

**最小用法**：

```python
import numpy as np
import rust_hnsw

data = np.random.default_rng(0).standard_normal((10_000, 64)).astype(np.float32)
data /= np.linalg.norm(data, axis=1, keepdims=True)        # cosine 距離：輸入必須是 unit vectors（decision 001）

index = rust_hnsw.HnswIndex(64, m=16, ef_construction=200, seed=1)
index.add(data)                                             # 依 row 順序插入；id 就是 row 編號
ids, distances = index.search(data[:5], 10, 64)             # (queries, k, ef_search) → int64 / float32，shape (5, 10)
index.save("demo.hnsw")
same = rust_hnsw.HnswIndex.load("demo.hnsw")                # 讀回的 index 給出完全相同的結果
```

- `search` 是一次批次呼叫；結果不足 k 時補 `-1` / `+inf`。距離是 cosine distance（`1 − 內積`，範圍 [0, 2]）。
- `rust_hnsw.ExactIndex(dim)` 是 brute-force 的 exact search，只用來產生 ground truth 與 correctness 檢查。
- 完整的 Demo（建 index、不同 `ef_search` 的 Recall@10、存檔與讀回）就是上面的 `examples/quickstart.py`：

```bash
.venv/bin/python examples/quickstart.py
```

Demo 用隨機產生的 synthetic 向量，只示範 API，印出的數字**不是研究結果**。

## 測試

```bash
uv venv --python 3.13 .venv          # 已建立時可以略過
uv pip install --python .venv/bin/python -r requirements-dev.txt
./scripts/test.sh
```

依序跑 rustfmt、clippy（`-D warnings`）、`cargo test`、`maturin develop --release`、pytest；有 `.venv-embed` 時另跑 EmbeddingGemma 2 的模型測試。
這些是 correctness 檢查，不需要 Mac heavy lock；正式 benchmark 才需要（見 `MACHINE_POLICY.md`）。

**測什麼、在哪、數量**（數量是測試的定義數，以 grep 計數：Rust 為 `grep -rE '^\s*#\[test\]' crates/hnsw` 的行數，Python 為 `grep -hE '^\s*def test_' tests/test_*.py` 的行數；2026-10-19 的工作目錄）：

| 測什麼 | 在哪 | 數量 |
| --- | --- | --- |
| cosine distance、正規化、exact brute force 與 tie 順序 | `crates/hnsw/tests/distance.rs`、`exact.rs` | Rust 34 |
| HNSW insert / search：小型人工可驗證案例（例如單位圓上與 exact 相同）、graph invariants、level 分佈、同 seed 同順序得到相同的 graph、recall 隨 ef 改善或飽和、大量重複向量 | `crates/hnsw/tests/hnsw.rs`、`crates/hnsw/src/hnsw.rs`、`src/rng.rs` | Rust 26 |
| save / load：檔案格式、損壞與截斷、CRC、load 的配置量 | `crates/hnsw/tests/persist.rs`、`load_alloc.rs`、`src/hnsw/persist.rs`、`src/crc32.rs` | Rust 40 |
| Advanced C 的 SQ8 表示與 `into_quant` | `crates/hnsw/tests/quant.rs`、`src/hnsw/quant.rs` | Rust 30 |
| Python binding（`ExactIndex`、`HnswIndex`、`QuantHnswIndex`） | `tests/test_exact_binding.py`、`test_hnsw_binding.py`、`test_quant_binding.py` | Python 57 |
| Core 輸出與存檔 bytes 的 golden test（在任何 C 的程式之前登記，3 組參數） | `tests/test_core_golden.py` | Python 1 |
| batch interface 與 Recall 指標 | `tests/test_interface.py`、`test_metrics.py` | Python 24 |
| adapters 與第三方 library 的實測語意 | `tests/test_adapters.py`、`test_third_party_semantics.py`、`test_quant_adapters.py` | Python 96 |
| runner、前置檢查與 hygiene（stub adapter 的整合測試、crash / timeout / SIGTERM）、閒置 load、`machine_quiet`、formal 批次腳本（以假的 python 與 pmset 驗證流程） | `tests/test_runner.py`、`test_run_config.py`、`test_run_environment.py`、`test_run_evaluate.py`、`test_quant_runner.py`、`test_idle_load.py`、`test_machine_quiet.py`、`test_formal_scripts.py` | Python 178 |
| 報告與判定規則（以手寫的 run 紀錄逐條驗證 010、015 與 019 的規則）；C 的 config 與 019 附錄逐字相同 | `tests/test_report.py`、`test_quant_report.py`、`test_quant_configs.py` | Python 150 |
| profiling：取樣 driver、排行解析、證據核對 | `tests/test_profile_driver.py`、`test_profile_report.py`、`test_gate_profile.py`、`test_quant_profile.py` | Python 67 |
| Advanced C 的不計時診斷與 smoke 核對 | `tests/test_quant_diag.py`、`test_quant_smoke.py` | Python 34 |
| 資料準備（manifest、checksum、frozen ground truth、文字 corpus 的切分、embedding） | `tests/test_dataprep.py`、`test_prepare_text.py`、`test_text_corpus.py`、`test_embed_text.py` | Python 38 |
| Demo 必須能跑 | `tests/test_examples.py` | Python 2 |
| `docs/results.md` 與本機報告逐項一致，而且不含本機路徑 | `tests/test_results_doc.py` | Python 35 |
| 兩份 README 與 `docs/results.md` 一致、彼此一致；公開文件的相對連結與錨點都存在，而且不含本機路徑或 email | `tests/test_readme_claims.py` | Python 29 |
| **合計** | | **Rust 130、Python 711** |

- pytest 實際收集的項目比定義數多（`parametrize`）。最後一次記錄的 `./scripts/test.sh`（019 第一則補記，commit dca22c9）是 pytest 1024 passed、3 skipped，`.venv-embed` 16 passed；之後新增與調整過測試，數量會不同。
- **會 skip 的情況**：
  - 主 `.venv` 沒有 torch 與 sentence-transformers：`test_embed_text.py` 的 3 個模型測試 skip；`.venv-embed` 的本機快取沒有該 model revision 時也會 skip。
  - 沒有 macOS 的 `sample`：`test_profile_driver.py` 的 2 個測試 skip。
  - 設定了 `OMP_NUM_THREADS`：`test_third_party_semantics.py` 的 2 個測試 skip。
  - 沒有本機的 `results/reports/`（乾淨的 clone）：`test_results_doc.py` 中需要報告的測試 skip，並說明原因。`test_readme_claims.py` 不需要本機報告，乾淨的 clone 也會執行。
- **平台**：只在 macOS arm64（Apple M4 Pro）上執行過；沒有設定 CI。執行時間沒有正式記錄（019 §12 把 2–3 次 `./scripts/test.sh` 估為中央 0.3 h、悲觀 0.5 h）。
- 新版 toolchain 可能加入新的 clippy lint，讓 `-D warnings` 失敗；要用實測的版本，以 `RUSTUP_TOOLCHAIN=1.99.0 ./scripts/test.sh` 執行（[`docs/reproduce.md`](docs/reproduce.md) §A）。

## Benchmark runner（decision 008）

完整、分層的重現步驟（時間、資源、平台限制，以及 Advanced C 的 smoke、formal、診斷、profiling 與報告的指令）見 [`docs/reproduce.md`](docs/reproduce.md)。

```bash
# smoke：只驗證 runner 本身（synthetic 資料，數字不得用於報告），不需要 lock
.venv/bin/python -m benchmarks.run benchmarks/configs/smoke.toml --mode smoke

# formal：先在乾淨的 HEAD 上跑 ./scripts/test.sh，再在 lock 下執行
./tools/with-mac-lock .venv/bin/python -m benchmarks.run <formal config> --mode formal
```

輸出在 `results/runs/<UTC 時間>-<mode>/`（不進版控）。formal 只接受已登記的真實資料集。

第一次 formal run 之前，先在 lock 下量閒置 load（008 §2；其他 process 有重工作時判為 `not_idle`）：

```bash
./tools/with-mac-lock .venv/bin/python -m benchmarks.idle_load
```

真實資料上的 correctness run（smoke，只看 recall 隨 efSearch 的變化）：

```bash
.venv/bin/python -m benchmarks.run benchmarks/configs/glove-correctness.toml --mode smoke
.venv/bin/python -m benchmarks.run benchmarks/configs/text-pilot-correctness.toml --mode smoke
.venv/bin/python -m benchmarks.run benchmarks/configs/text-correctness.toml --mode smoke
```

`scripts/profile.sh` 的用法範例（009；**不是** Gate 證據：Gate 的取樣用下方「Gate profiling」所述的 ef、取樣時間與計畫；取樣前寫出參數、git、編譯資訊與 Recall@10）：

```bash
./tools/with-mac-lock ./scripts/profile.sh <名稱> 20 -- --impl rust-hnsw --phase search \
    --dataset glove-100-angular --n 100000 --m 16 --ef-construction 200 --seed 1 --ef 128 --k 10
```

## 資料集（decision 011）

資料檔放在 `datasets/`（不進 Git）；每個資料集的來源、checksum、切分與 frozen ground truth 記在 `benchmarks/manifests/<名稱>.json`。
資料集與 embedding 模型的授權見 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

```bash
# 標準 ANN：glove-100-angular（ann-benchmarks，463 MB）
.venv/bin/python -m benchmarks.prepare_glove --download

# 真實文字：Simple English Wikipedia + google/embeddinggemma-2（768 維、fp32、CPU）
uv venv --python 3.13 .venv-embed
uv pip install --python .venv-embed/bin/python -r requirements-embed.txt
.venv/bin/python -m benchmarks.prepare_text --name simplewiki-embeddinggemma2-pilot --n-base 10000 --n-queries 1000
# 正式文字 workload（decision 012）
.venv/bin/python -m benchmarks.prepare_text --name simplewiki-embeddinggemma2-100k --n-base 100000 --n-queries 2000 \
    --gt-n 10000 --gt-n 30000 --gt-n 100000
```

- Wikipedia 的 parquet 要事先手動放到 `datasets/simplewiki/raw/`（repo 沒有下載它的程式；來源 revision 與 SHA-256 見 [`docs/reproduce.md`](docs/reproduce.md) §B.2）。`prepare_text` 開始時比對 SHA-256，不符就拒絕。
- embedding 模型以匿名方式從 Hugging Face 下載（不使用任何 token）。

報告（依 decision 010 的比較規則：frontier 讀值、variance、scaling、recall vs efSearch 圖）：

```bash
.venv/bin/python -m benchmarks.report results/runs/<run 目錄> [<run 目錄> ...]
```

正式 run 的批次（decision 013 §5、014；等機器可用時在 heavy lock 下依序執行，任一 run 不是 complete 就停）：

```bash
mkdir -p results/batches
./scripts/formal_launcher.sh results/batches/<名稱>.log <最長等待秒數> --powermode 2 [--skip-idle] benchmarks/configs/<formal config>.toml ...
```

Gate profiling（decision 015：依正式 run 選各實作 Recall@10 ≥ 0.95 的最小 ef，在 heavy lock 下取樣，取樣後核對證據條件；
計畫寫在 `results/` 下，不讓 repo 變 dirty）：

```bash
mkdir -p results/profiles
.venv/bin/python -m benchmarks.gate_profile plan <該資料集全部 formal run 目錄> > results/profiles/gate-<資料集>-plan.sh
./tools/with-mac-lock bash results/profiles/gate-<資料集>-plan.sh
.venv/bin/python -m benchmarks.gate_profile check <同一組 run 目錄>
```

## 文件導覽

| 想看什麼 | 文件 |
| --- | --- |
| 研究定義：研究問題、Core / Advanced / Optional、Gate | [`SPEC.md`](SPEC.md) |
| 實驗規範：hygiene、公平性、報告要求 | [`docs/EXPERIMENT_RULES.md`](docs/EXPERIMENT_RULES.md) |
| 正式結果的完整轉錄：四份報告的關鍵表格、完整判定表、每個 seed 的數字、報告的 SHA-256 | [`docs/results.md`](docs/results.md) |
| 架構：資料流、量測隔離、hygiene 與 lock、模組對照、Advanced C 的臂如何共用同一張圖 | [`docs/architecture.md`](docs/architecture.md) |
| 重現：分層的步驟、時間、平台限制、環境版本 | [`docs/reproduce.md`](docs/reproduce.md) |
| 圖的出處：每張圖的 SHA-256、來源報告與 commit | [`docs/figures/README.md`](docs/figures/README.md) |
| 每一項設計決定（001–019，含補記） | [`docs/decisions/`](docs/decisions/)（索引：[`docs/decisions/README.md`](docs/decisions/README.md)） |
| 第三方內容與授權 | [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) |
| 工作日誌與執行手冊（開發過程的紀錄） | [`NEXT.md`](NEXT.md) |
| AI 協作規則：AI 助理（Claude Code）在本專案的工作規則（開發過程的紀錄） | [`CLAUDE.md`](CLAUDE.md) |
| 機器政策：共用 Mac 的 heavy lock 與資源規則（開發過程的紀錄） | [`MACHINE_POLICY.md`](MACHINE_POLICY.md) |

NEXT.md、CLAUDE.md、MACHINE_POLICY.md 是開發過程的紀錄，保留原樣；這三份與部分 decision log 對專案作者的稱呼，沿用開發時與 AI 助理協作的用語。

## 目錄

```text
crates/hnsw/               Rust 實作（distance、exact search、HNSW、save / load、Advanced C 的 SQ8 表示）；沒有外部相依
crates/hnsw/tests/         Rust 整合測試（distance、exact、hnsw、persist 的損壞與截斷、load 的配置量、quant）
python/                    PyO3 binding，Python module 名稱 `rust_hnsw`（build.rs 記錄 BUILD_INFO）
benchmarks/                統一 batch interface、adapters、Recall 指標、benchmark runner（run.py、run_child.py）、Core 報告（report.py）、
                           資料準備（prepare_glove.py、text_corpus.py、embed_text.py、prepare_text.py）、
                           hygiene（environment.py、protocol.py、machine_quiet.py、idle_load.py）、profiling（profile.py、profile_report.py、gate_profile.py）、
                           Advanced C 的報告、診斷、profiling 與 smoke 核對（quant_report.py、quant_diag.py、quant_profile.py、quant_smoke.py）
benchmarks/configs/        runner 的 TOML config：synthetic smoke、真實資料的 correctness、成本探測、12 份 Core formal、10 份 C formal、C 的 smoke
benchmarks/manifests/      資料集的 manifest（來源、checksum、切分、frozen ground truth）
tests/                     Python 測試（binding、adapters、第三方原生語意、interface、metrics、runner、報告、profiling、資料準備、
                           Advanced C、golden test、Demo、results.md 與兩份 README 的核對）
scripts/test.sh            一鍵跑全部 correctness 檢查
scripts/profile.sh         以 macOS `sample` 對 build / search 取樣（decision 009）
scripts/formal_launcher.sh 等機器閒置後在 heavy lock 下啟動 formal / smoke 批次
scripts/formal_batch.sh    在同一次 lock 持有中依序執行批次，任一 run 失敗就停
scripts/probe_reusable_memory.py  重現 macOS 上「已釋放的大型配置仍算在 rss」的量測問題（decision 014）
scripts/probe_core_codegen.py     比對兩個 commit 中 Core 函式的機器碼（019 補記，只描述）
tools/with-mac-lock        正式 benchmark 用的 Mac heavy lock
examples/                  API 的 Demo（synthetic 資料，數字不是研究結果）
docs/decisions/            decision logs 001–019（繁體中文）
docs/figures/              README 用的圖（出處見 docs/figures/README.md）
docs/results.md            正式報告關鍵表格的轉錄
docs/architecture.md       架構與資料流
docs/reproduce.md          重現指南
requirements-dev.txt       主 .venv 的版本 pin（含 hnswlib、faiss-cpu）
requirements-embed.txt     產生 embedding 用的 .venv-embed 的版本 pin
LICENSE                    BSD-3-Clause（原創程式碼）
THIRD_PARTY_NOTICES.md     第三方內容的來源與授權（不在 BSD-3-Clause 範圍內）
```

## 主要決策

完整索引（含各補記的日期）見 [`docs/decisions/README.md`](docs/decisions/README.md)。

- [001 cosine 距離定義、正規化與 tie-break](docs/decisions/001-cosine-distance-and-normalization.md)
- [002 統一 Python batch interface 與計時協定](docs/decisions/002-python-batch-interface.md)
- [003 Recall@K 定義與 tie-aware 診斷](docs/decisions/003-recall-definition.md)
- [004 HNSW Core 演算法細節與 baseline 表示法](docs/decisions/004-hnsw-core-algorithm.md)
- [005 HNSW index 檔案格式（save / load）](docs/decisions/005-index-file-format.md)
- [006 HNSW 的 Python binding 與 efSearch 介面](docs/decisions/006-hnsw-python-binding.md)
- [007 hnswlib / FAISS adapters 與實測語意](docs/decisions/007-third-party-adapters.md)
- [008 benchmark runner 與量測協定的落實](docs/decisions/008-benchmark-runner.md)
- [009 profiling 方法（Gate 證據的產生方式）](docs/decisions/009-profiling-method.md)
- [010 比較規則與 variance 彙整（事先登記）](docs/decisions/010-comparison-rules.md)
- [011 資料集：glove-100-angular 與 EmbeddingGemma 2 文字 workload](docs/decisions/011-datasets.md)
- [012 正式文字 workload 的規模](docs/decisions/012-text-workload-scale.md)
- [013 正式 run 的格點（formal config）](docs/decisions/013-formal-grid.md)
- [014 正式 run 的電力條件、背景 workload 與 index memory 指標](docs/decisions/014-formal-run-conditions-and-index-memory.md)
- [015 Gate 證據的讀法與 Gate profiling 的參數（事先登記）](docs/decisions/015-gate-evidence.md)
- [016 文字 workload 的電力模式不一致：作廢 text-formal-2、3，全部以高效能模式重跑](docs/decisions/016-text-power-mode-voided-runs.md)
- [017 取消電量下降的作廢規則（協定 008-v3）](docs/decisions/017-remove-battery-drop-rule.md)
- [018 允許 Apple 原廠 140W 變壓器（協定 008-v4）](docs/decisions/018-allow-140w-adapter.md)
- [019 Advanced C：vector quantization 的假設、表示與評測協定（事先登記）](docs/decisions/019-advanced-c-vector-quantization.md)

## 授權

- 本專案作者有權授權的**原創程式碼**以 **BSD-3-Clause** 授權（[LICENSE](LICENSE)）；`Cargo.toml` 與 `python/pyproject.toml` 宣告相同的授權。
  較早的 commit 中，`Cargo.toml` 的 `license` 欄位是 `MIT`，repo 內沒有 LICENSE 檔；LICENSE 所列的版權人，也就是專案作者，已改為 BSD-3-Clause。
- repo 不含第三方程式碼、資料集內容、模型權重或 embedding。資料集、embedding 模型與比較用的 baseline 都在執行時下載，**不在**此授權範圍內，保留各自的授權：
  GloVe 向量（PDDL 1.0）經 ann-benchmarks（MIT）、Simple English Wikipedia（CC BY-SA 與 GFDL）、`google/embeddinggemma-2`（Apache-2.0，使用時須遵守 Gemma Prohibited Use Policy）、
  hnswlib（Apache-2.0）、faiss-cpu（MIT）。來源與標示：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
