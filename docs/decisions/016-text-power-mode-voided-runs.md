# 016：文字 workload 的電力模式不一致——作廢 text-formal-2、3，全部以高效能模式重跑（使用者決定）

日期：2026-10-11
狀態：採用（使用者決定；在看任何文字 workload 的正式效能結果之前）

## 背景

- 014 補記：formal run 允許自動（powermode 0）或高效能（2），但同一個 run 內不得改變，**一份報告只合併同一個電力模式的 run**。
- 010 §3、013 §5：complete 的 run 不得被取代或捨棄（避免事後挑選）；`report.py` 會拒絕同一份 formal config 另外有沒交出的 complete run。
- 文字 workload 的 formal run（台灣時間，UTC+8）：
  - `text-formal-1` complete `20261010T112849Z-formal`：**powermode 2**。
  - 之後 AC 電源的電力模式設定被改成自動（`pmset -g custom` 的 AC Power 為 0），**不是本專案改的**。期間電池一度降到 56%（可能沒有接電源）。
  - `text-formal-2` complete `20261010T183115Z-formal`、`text-formal-3` complete `20261010T195251Z-formal`：**powermode 0**。
  - `text-formal-4` invalid `20261010T202624Z-formal`：powermode 0，接 AC 時電池從 98% 降到 95%（014）。
- 結果：text-1 與 text-2、3 不能合併進同一份報告，又都不能捨棄，文字報告無法產生。
- 撰寫者只看過這些 run 的狀態、電力紀錄與 hygiene 標記，**沒有看過任何文字 run 的 recall、QPS、latency、build time 或記憶體**。

## 決策

使用者在 2026-10-11 04:55（台灣時間）前後選擇「**全部用高效能**」：

1. **作廢** `20261010T183115Z-formal`（text-formal-2）與 `20261010T195251Z-formal`（text-formal-3）。理由只有一個：
   兩者在 run 之間被外部改變的電力模式（0）下執行，與資料集的第一個 complete run（text-1，mode 2）不同。與結果無關。
   - 登記在 `report.py` 的 `VOIDED_RUNS`。登記的 run 與失敗的 run 一樣，只能以 `--excluded` 列出（附上作廢理由），資料不合併；
     沒有登記的 complete run 仍然不得排除。
2. 使用者把 AC 的電力模式切回高效能（2026-10-11 04:57 台灣時間確認 `pmset -g` 為 2），以高效能模式重跑 text-formal-2、3、4。
   GloVe 的 8 份與 text-1 都在高效能模式，兩個資料集一致。
3. **防止再發生**：`scripts/formal_launcher.sh` 與 `scripts/formal_batch.sh` 新增 `--powermode N`：
   等待腳本在電力模式不是 N 時繼續等；批次在每份 config（與閒置 load 量測）開始前確認，不是 N 就停下來，不開始下一份。
   runner 本身不改（`MEASUREMENT_PATHS` 的檔案改了，改動前後的 run 就不能合併，013 §5）。
4. **之後的通則**：同一個資料集的 formal run 一律在同一個電力模式下執行。若因外部變化而在不同模式下完成了 complete run，
   以新的 decision 登記作廢（只能依據電力紀錄等環境資料，不得依據結果），再在原本的模式下重跑。

## 為什麼

- 電力模式改變 CPU 的效能狀態，跨模式的 run 合併後，seed variance（text-1 的 seed 1 與 text-2 的 seeds 2–5）會混入模式差異。
- 作廢的理由只來自電力紀錄，而且在看任何文字結果之前決定，不是事後挑選。
- 選高效能：與 GloVe 的 8 份、text-1 一致，使用者也允許高效能（014 補記）。

## 放棄的替代方案

- **全部用自動**（作廢 text-1，在自動模式重跑 text-1、4）：成本相近，但 GloVe 與文字會在不同模式；使用者選了高效能。
- **允許文字報告合併兩種模式並逐 run 標示**：要改 014 補記，而且 variance 格點的 seed 會跨模式。
- **在 runner 裡要求固定的電力模式**：要改 `MEASUREMENT_PATHS` 的檔案，text-1 就不能與重跑的 run 合併。

## 對實驗的影響

- 文字報告的 `--excluded`：`20261010T083544Z-formal`（text-1 第一次，使用者暫停而中止）、`20261010T133129Z-formal`（text-2 第一次，
  其他專案的重工作讓 load gate 中止）、`20261010T202624Z-formal`（text-4，電池下降而 invalid）、`20261010T183115Z-formal` 與
  `20261010T195251Z-formal`（本文件作廢）。
- 重跑約 3 小時（text-2 約 1.4 h、text-3 約 0.6 h、text-4 約 1 h）。之後依 015 §2 做 Gate profiling。
- 電池在接 AC 時下降的原因仍未確認（使用者沒有指定）。再發生時照 014 作廢並停止，記到 NEXT。
