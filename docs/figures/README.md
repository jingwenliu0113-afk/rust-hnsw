# 圖的來源（Figure provenance）

This directory holds 12 PNG figures, all produced by this project's report tools from local formal runs.
The runs and reports are not under version control (`results/` is git-ignored), so this file records, for each
figure, its SHA-256, the matching local report output, the tool that drew it and the commit that added it.
The numbers live in the report tables; a figure only shows their shape.

- 這個目錄的 12 張 PNG 都由本專案的報告工具，從本機的正式 run（formal 模式）產生。PNG 的 metadata 只有一項 `Software`：`Matplotlib version3.11.2, https://matplotlib.org/`。
- 正式 run 與報告都不進版控（`results/` 在 `.gitignore` 中）。所以這裡記錄每張圖的 SHA-256、對應的本機報告輸出、產生它的工具與加入的 commit，讓手上有本機報告的人可以逐位元核對。
- 數字以報告的表格為準，圖只呈現形狀。判定以事先登記的規則（decision 010、015、019）為準，不是從圖上讀出來的。
- 核對方式（2026-10-19）：以 SHA-256 比對 `results/reports/` 中的同名檔。本階段沒有重新產生或覆蓋任何圖。

## 來源報告（本機，不進版控）

| 報告目錄 | 產生工具 | 合併的 run | 檔案時間（台灣時間） | 當時該工具最後一次修改的 commit |
| --- | --- | --- | --- | --- |
| `results/reports/glove-100-angular/` | `benchmarks/report.py` | 8 個 GloVe formal run；另以 `--excluded` 列出 1 個 aborted run | 2026-10-10 17:58:29 | 1fa384b |
| `results/reports/simplewiki-embeddinggemma2-100k/` | `benchmarks/report.py` | 4 個文字 formal run；另以 `--excluded` 列出 7 個 run | 2026-10-16 07:55:27 | 76a352b |
| `results/reports/quant-glove/` | `benchmarks/quant_report.py` | 5 個 GloVe 的 C formal run；Core 對照 run 4 個 | 2026-10-19 00:39:42 | 888f3f7 |
| `results/reports/quant-text/` | `benchmarks/quant_report.py` | 5 個文字的 C formal run；Core 對照 run 4 個 | 2026-10-19 00:39:43 | 888f3f7 |

- 「最後一次修改的 commit」是依檔案時間與 `git log -- <工具>` 推得的：它只說明當時已 commit 的最新版本，無法證明產生時的工作目錄與該 commit 完全相同。
- `quant_report.py` 只有 dca22c9 與 888f3f7 兩次修改。888f3f7 只改繪圖（[decision 019](../decisions/019-advanced-c-vector-quantization.md) 的第三則補記）。
  repo 外的本機暫存目錄留有修正前（2026-10-17 09:31）的輸出：它的 report.md、report.json 與上表的 SHA-256 相同，只有 PNG 不同。
- 合併的 run（各報告的 `report.json`）：
  - GloVe：`20261009T080239Z-formal`、`20261009T091611Z-formal`、`20261009T101347Z-formal`、`20261009T105616Z-formal`、`20261010T021933Z-formal`、`20261010T025425Z-formal`、`20261010T034635Z-formal`、`20261010T040529Z-formal`；excluded：`20261009T113943Z-formal`。
  - 文字：`20261015T191333Z-formal`、`20261015T210125Z-formal`、`20261015T222221Z-formal`、`20261015T225545Z-formal`；excluded：`20261010T083544Z-formal`、`20261010T133129Z-formal`、`20261010T202624Z-formal`、`20261011T033341Z-formal`、`20261010T183115Z-formal`、`20261010T195251Z-formal`、`20261010T112849Z-formal`。
  - C 的 GloVe：`20261016T203802Z-formal`、`20261016T211434Z-formal`、`20261016T214152Z-formal`、`20261016T220514Z-formal`、`20261016T221833Z-formal`。
  - C 的文字：`20261016T223724Z-formal`、`20261016T230246Z-formal`、`20261016T232100Z-formal`、`20261016T234053Z-formal`、`20261016T235454Z-formal`。
  - C 的 Core 對照（兩份 C 報告相同）：`20261009T080239Z-formal`、`20261009T091611Z-formal`、`20261015T191333Z-formal`、`20261015T210125Z-formal`。

報告本身的 SHA-256（供核對其他文件引用的數字出自哪一版報告）：

| 檔案 | SHA-256 |
| --- | --- |
| `glove-100-angular/report.md` | `83036d815674cb52af2f12964f4d4b3c65fef9313411709b7b13f8212093c58f` |
| `glove-100-angular/report.json` | `6a7911d993071f842e6b4f0bb7383688caa3f2f0ec5215b16860f5e33ac81efc` |
| `simplewiki-embeddinggemma2-100k/report.md` | `bcb93ff37faf28961f76a265281a7d683638e779a85d17b1a7d336d6dde788f2` |
| `simplewiki-embeddinggemma2-100k/report.json` | `2c47be75f7a54f4b8b3a77875227ab808de1805bb6c6c1bbf0f124606bee7c7e` |
| `quant-glove/report.md` | `d77058ce700aabbaf94954beed1f217bb6b33faff36450aa57f28d6c9931b15f` |
| `quant-glove/report.json` | `0f1030630ae99f58ce8a49fd902f6bdaa29bc785144ce4dbc73758c2d17fe936` |
| `quant-text/report.md` | `3675cf6131fad662e670fb27ad758431258bb7a1052c4077a1e025155bbdfccb` |
| `quant-text/report.json` | `fe751acbb0d14e10c2d2f260e30e8a0ccaeb1150ac6172170616470ff76c6f92` |

## 每張圖

「本機報告輸出」一欄是 `results/reports/` 中的同名檔；「相同」表示兩者的 SHA-256 相同。

### Core：`glove-100-angular/`

| 圖 | SHA-256 | 本機報告輸出 | 工具 | 加入的 commit |
| --- | --- | --- | --- | --- |
| `frontier_n100000.png` | `4269e8f9961fd0ce65e61e413733e39153f3cb8f8a89b9593d24b98536aed161` | 本機 GloVe 報告目錄沒有這個檔（見下一節） | `report.py`（76a352b 起才畫這種圖） | f219d39（2026-10-15 22:40 +08:00） |
| `recall_vs_ef_n100000_m16_efc200.png` | `d7b41c8424031f7c2fe08be3a5e080acbe7d13809c7d112ef20eaa3307af54ce` | `glove-100-angular/recall_vs_ef_n100000_m16_efc200.png`：相同 | `report.py` | f219d39（2026-10-15 22:40 +08:00） |
| `scaling.png` | `1b723ca7c1445d1d3486b7d898efea589445c85ea02285fb798e4549c232950d` | 本機 GloVe 報告目錄沒有這個檔（見下一節） | `report.py`（76a352b 起才畫這種圖） | f219d39（2026-10-15 22:40 +08:00） |

### Core：`simplewiki-embeddinggemma2-100k/`

| 圖 | SHA-256 | 本機報告輸出 | 工具 | 加入的 commit |
| --- | --- | --- | --- | --- |
| `frontier_n100000.png` | `1d6f7b77d761405b7d3679396682e2dc6b3b6289431d49935cb48eae5182ddf5` | `simplewiki-embeddinggemma2-100k/frontier_n100000.png`：相同 | `report.py` | a87eed9（2026-10-16 07:57 +08:00） |
| `recall_vs_ef_n100000_m16_efc200.png` | `249f4f2550a22b7e09f10afa10f6aa058a3b4de4fe79f7437849c90d17aab580` | `simplewiki-embeddinggemma2-100k/recall_vs_ef_n100000_m16_efc200.png`：相同 | `report.py` | a87eed9（2026-10-16 07:57 +08:00） |
| `scaling.png` | `def6695bee0060e4f9ede73549d124e0e74d6bcf3e8237b82e37ee1942f94443` | `simplewiki-embeddinggemma2-100k/scaling.png`：相同 | `report.py` | a87eed9（2026-10-16 07:57 +08:00） |

### Advanced C：`advanced-c/`

| 圖 | SHA-256 | 本機報告輸出 | 工具 | 加入的 commit |
| --- | --- | --- | --- | --- |
| `glove/recall_qps.png` | `52bf4702267695a94e244cbc0ef537207da07556f3e18bf70439759ea48fa631` | `quant-glove/recall_qps.png`：相同 | `quant_report.py` | c9b2616（2026-10-19 00:49 +08:00） |
| `glove/quant_frontier_0.95.png` | `af2f8635ccdba67d932e2e8a1da140db59abc33c0713116ec5d653ea10cede24` | `quant-glove/quant_frontier_0.95.png`：相同 | `quant_report.py` | c9b2616（2026-10-19 00:49 +08:00） |
| `glove/profile_breakdown.png` | `7e7755c01a3a5b67a0560c5479617866c416d543f78f06930ddadd06edfb36be` | `quant-glove/profile_breakdown.png`：相同 | `quant_report.py` | c9b2616（2026-10-19 00:49 +08:00） |
| `text/recall_qps.png` | `4fec8346f33156c31a613059d792fa3f1f78d4d83cd5495f4a200e6ed0b8e403` | `quant-text/recall_qps.png`：相同 | `quant_report.py` | c9b2616（2026-10-19 00:49 +08:00） |
| `text/quant_frontier_0.95.png` | `186468a6fd91998aa5b0fe3efe6d69e49a9b715d30770cae627709972194d40d` | `quant-text/quant_frontier_0.95.png`：相同 | `quant_report.py` | c9b2616（2026-10-19 00:49 +08:00） |
| `text/profile_breakdown.png` | `c3d5ce809ad4d555762c5c52ce8002e81116faee8d5f4f54f2409030546df5cc` | `quant-text/profile_breakdown.png`：相同 | `quant_report.py` | c9b2616（2026-10-19 00:49 +08:00） |

## GloVe 的 `frontier_n100000.png` 與 `scaling.png`：能確認與不能確認的部分

本機的 GloVe 報告目錄裡沒有這兩張圖。以下是從 Git 歷史與本機檔案查到的事實，以及查不到的部分。

### 能確認

1. 兩張圖與 `recall_vs_ef_n100000_m16_efc200.png` 一起由 f219d39（2026-10-15 22:40:08 +08:00）加入。
2. 畫這兩種圖的程式（`report.py` 輸出 `frontier_n<n>.png` 與 `scaling.png`）由 76a352b（2026-10-15 22:38:44 +08:00）加入；它的父 commit 4acf799 的 `report.py` 不會產生這兩種圖。
3. 本機 GloVe 報告目錄的檔案時間都是 2026-10-10 17:58，早於 76a352b，所以裡面沒有這兩張圖。
4. repo 外的本機暫存目錄（Claude Code session 的 scratchpad，不進版控）留有兩組逐位元相同的輸出：
   - 一個只有 `frontier_n100000.png`、`frontier_n300000.png`、`scaling.png` 三個檔的目錄，檔案時間 2026-10-15 22:37:57–58 +08:00：晚於 4acf799、早於 76a352b 的 commit 約 47 秒。其中 `frontier_n100000.png` 與 `scaling.png` 的 SHA-256 和本目錄的圖相同。
   - 2026-10-17 00:10 +08:00 的一次重新產生核對（Advanced C 實作期間，比對 `report.py` 修改前後的輸出）：
     - 使用的 `report.py` 與 `git show 76a352b:benchmarks/report.py` 逐位元相同；
     - 輸入是本機 GloVe 報告 `report.json` 記錄的 8 個 run 與 1 個 excluded run（上表）；
     - 輸出的 `frontier_n100000.png` 與 `scaling.png` 和本目錄的圖逐位元相同。同一次核對中，當時工作目錄裡尚未 commit 的 `report.py`（dca22c9 之前）也產生相同的兩張圖。
5. 同一次核對產生的 `report.md`，與本機 10-10 版的 GloVe `report.md` 相比，只多了一節「圖」，並改了「被排除的 run」一節的標題；所有的表與數字都相同。所以這兩張圖背後的數字，就是 10-10 版 GloVe 報告的數字。
6. 綜合第 4、5 點：76a352b 的 `report.py` 對上表的 8 個 GloVe formal run（`--excluded results/runs/20261009T113943Z-formal`）的輸出，與本目錄的兩張圖逐位元相同。

### 不能確認

1. 2026-10-15 22:37 那一次實際執行的指令與 `report.py` 的內容。當時 76a352b 還沒有 commit；暫存目錄只有 3 張 PNG，沒有 report.md、report.json，也沒有留下指令紀錄。所以無法從當時的紀錄證明輸入是哪些 run、用的是哪一版工作目錄，只能由第 4、6 點的重現結果推斷是同一組輸入。
2. 這兩張圖沒有對應的正式報告目錄。GloVe 的數字請以本機 10-10 版的 `results/reports/glove-100-angular/report.md` 為準（第 5 點）。
3. 暫存目錄不是持久的紀錄，之後可能被清除。上面記的是 2026-10-19 觀察到的狀態。
4. HEAD 的 `report.py`（dca22c9 之後）是否仍會逐位元產生相同的圖，本階段沒有驗證（不得重新產生）。
   需要時的產生方式是 `.venv/bin/python -m benchmarks.report <上表 8 個 GloVe run 目錄> --excluded results/runs/20261009T113943Z-formal --out <新的目錄>`，而且需要本機的 `results/runs/`（不公開）。不得用新的輸出覆蓋本目錄的圖。

## 讀圖時的限定

判定以報告與 decision 為準。

- **所有圖都排除被標記的點。**
  - 76a352b 之後的 Core 報告（本機的文字報告）寫明「圖只呈現形狀，數字以上面的表為準（010 §2：不內插）」。本機 10-10 版的 GloVe 報告早於 76a352b，沒有這一節。
  - 兩份 C 報告寫明「排除被標記的點（與主要讀值相同）；數字以上面的表為準」。
- **Core 的 frontier 圖**：Gate 條件 3「同 recall 下與 hnswlib 有差距」依事先登記的 [015](../decisions/015-gate-evidence.md) 讀法**不成立**。來源是兩份 Core 報告的「Gate 讀值」一節：
  - GloVe：主要目標 0.95 為「在 ef 梯子的量化範圍內，無法區分」（`within_quantization`）；0.90 為「不確定」（`uncertain`）；0.99 為 `within_quantization`。
  - 文字：三個目標都是 `within_quantization`。
  - 這裡的 quantization 指 ef 梯子的解析度，與 Advanced C 的 vector quantization 無關。
  - 進入 Advanced C 是專案作者依 SPEC 特例的決定，不代表 Gate 通過（NEXT.md 的 Gate 四條件）。
- **GloVe 的 `scaling.png` 只有 n = 100k 與 300k**：
  - 1,183,514 的三個實作在報告的「scaling」一節都沒有未標記的 build，所以沒有主要讀值。
  - rust-hnsw 在 300k 的 0.95 主要讀值為「未達到」，所以圖上沒有這一點。報告另列的「包含被標記點的讀值」是 915 QPS @ ef 512。
- **Advanced C 的 `quant_frontier_0.95.png`**：
  - 事先登記的 quantization frontier 判定（[019](../decisions/019-advanced-c-vector-quantization.md) §6；兩份 C 報告的完整判定表）：
    - S 對 F：GloVe 0.90、0.95 都是 `uncertain`；文字 0.90 為 `within_quantization`、0.95 為 `uncertain`。
    - Q 對 P：GloVe 都是 `uncertain`；文字都是 `within_quantization`。
    - 包含被標記 build 的判讀：GloVe 的 Q 對 P 在 0.90 與 0.95 都是 `within_quantization`（報告標為「不同」；P 排除被標記的 build 後只剩 2 個，少於每側 3 個）；其餘相同。兩者並列（015 §1），見 [docs/results.md](../results.md) §6.7。
    - 圖排除被標記的 build。GloVe 的 F、S、P 包含被標記 build 時讀值不同（例如 P 在 0.95 為 2,969（3 builds），主要讀值是 3,350（2 builds）），見 [docs/results.md](../results.md) §6.6。
  - 看到結果之後的描述（**不是判定**），數字取自兩份 C 報告的「讀值 1：frontier」：
    - GloVe 圖上，S（2,299 QPS、M_014 76.1 MiB，4 builds）比 F（1,877 QPS、105.4 MiB，3 builds）快，也比較小。
    - 文字圖上，S 比 F 慢（1,769 對 2,277 QPS），M_014 從 318.8 降到 99.0 MiB（各 5 builds）。
  - C1（F − S 的 M_relief 差是否與理論值 Δ_theory 一致；M_relief 只是診斷，主要指標是 M_014）：GloVe 只有 seed 5 偏離 4,030,488 B（約 3.84 MiB），超過容許的 max(5% × Δ_theory, 2 MiB) = 2 MiB，判為不一致；其餘都一致。
- **Advanced C 的 `recall_qps.png`**：絕對 QPS 不與 Core 的圖比較。C6（只描述）是 C 中 R 的 QPS 除以 Core rust-hnsw seed 1 的 QPS：GloVe 0.82–0.85、文字 1.09–1.13。019 的補記把它讀為「時間漂移加上 codegen 差異」，兩者無法分開。
- **Advanced C 的 `profile_breakdown.png`**：seed 1 在 e\* 的每 query 時間分解（F 對 S）。
  - 時間分解是取樣 profiler 依 019 §8 事先固定的分類規則所做的歸屬。
  - H1d（profiler，seed 1）**無法判定**：GloVe seed 1 的 S 在 e\* 有 `background` 標記。所以 GloVe 這張圖只作描述。
- **Advanced C 的整體判定**（完整判定表）：H1a 成立；H1b 無法判定；H1 整體無法判定；H2 成立。
  - H1b 無法判定，是因為 GloVe 的 seeds 1、4、5 有 `background` 標記，e\* 只剩 seeds 2、3 兩個乾淨配對，少於 3 個。
  - 標記的來源是沒有取 lock 的其他工作。紀錄只能證明它們不屬於本次 run（019 的第三則補記）。
