# Claude Code 工作規則：Rust HNSW

你目前只負責 **rust-hnsw**。

## 開始任何工作前

1. 完整閱讀 `SPEC.md`。
2. 閱讀 `NEXT.md`。
3. 閱讀 `MACHINE_POLICY.md`。
4. 閱讀 `docs/EXPERIMENT_RULES.md`。
5. 檢查 `git status`。
6. 確認目前正在推進哪個 Core milestone。
7. 若需要遠端 RTX 5070 Ti，先依 `MACHINE_POLICY.md` 判斷是否真的必要。
8. 遠端不可用時，只把該計算步驟標記 blocked，繼續 Mac 上可完成的工作。
9. 除非 Gate 已明確通過，否則不得進 Advanced。
10. 除非使用者明確要求，否則不得實作 Optional。

## 公開與授權（2026-10-19）

- GitHub remote `origin` 是 **Public**（專案作者 2026-10-19 核准），https://github.com/jingwenliu0113-afk/rust-hnsw。
- **授權：BSD-3-Clause**（原創程式碼，`LICENSE`）；第三方內容保留原授權（`THIRD_PARTY_NOTICES.md`）。不得加入 GPL / AGPL 依賴，不得 commit 資料集、embedding 或模型權重。
- 本 repo 的 commit email 是 GitHub noreply（repo-local git config）；不得改寫 Git 歷史、不得 force push。
- Core 與 Advanced C 已封版：不得修改封存結果、Gate 或正式實驗；兩份 README 的數字必須與 `docs/results.md` 一致（`tests/test_readme_claims.py`）。

## 專案邊界

- 不做完整向量資料庫。
- 不做 distributed system。
- 不提前做 sharding / replication / metadata engine。
- 不拿隨機高斯向量當主要 benchmark 結論。
- brute force 只作 ground truth 與 correctness baseline。


## 機器與資源衝突

- 必須遵守 `MACHINE_POLICY.md`。
- 正式 Heavy workload / benchmark 所需 lock 拿不到時，不得搶占。
- 不得刪除別的 session 建立的 lock。
- 遠端桌機不可用時，只把該步驟標記 blocked，繼續 Mac 可做的工作。
- 不得自行要求或尋找任何 credential。

## 共通執行邊界

- 不得自行擴張研究問題。
- 不得因為「未來可能整合」而預先建立通用 framework。
- 不得跳過 Core 直接實作 Advanced 或 Optional。
- 不得只回報成功案例。
- 不得看到結果後偷偷修改指標、測試題、資料切分或 benchmark protocol。
- 任何會改變實驗設計的決策都必須留下 decision log。
- 每次 session 結束前更新 `NEXT.md`。
- 如果發現 SPEC 與程式現況不一致，先記錄差異，再處理，不要默默改規格。
