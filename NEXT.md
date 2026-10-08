# NEXT

## 目前位置

**Core / Week 1 進行中**：步驟 1「HNSW insert / search」已完成。
尚未進 Advanced（Gate 條件都還沒有資料可判斷），沒有做任何 Optional。

## 上次完成（2026-10-08）

### Git

- `rust-hnsw/` 初始化為獨立 git repo（branch `main`），`.gitignore` 補上 `.DS_Store`。
- 重跑全部測試與 `./scripts/test.sh` 皆通過後，建立 baseline commit：
  `ddcb27d chore: establish core baseline`（Day 1 的全部內容）。
- Week 1 步驟 1（HNSW insert / search、decision 004、correctness tests）已依使用者要求做成獨立 commit。

### Week 1 步驟 1：HNSW insert / search

- 先寫 **decision 004**：
  - 演算法細節對齊 hnswlib：heuristic 鄰居選擇、`M0 = 2M`、`ef = max(efConstruction, M)`、
    上層 greedy、下一層 entry 為最近點、候選數 < M 時全收。
  - baseline 表示法刻意不優化：per-node `Vec`、`HashSet` visited、`BinaryHeap`、純量距離。
- 新增 `crates/hnsw/src/hnsw.rs`（`HnswIndex`、`HnswParams`）與 `rng.rs`（自帶 SplitMix64）。
- exact 與 HNSW 共用的輸入驗證抽到 `batch.rs`（`PADDING_ID`、`check_k`、`unit_query`、`unit_rows`）。
- 測試先行，確認紅燈後才實作：HNSW 整合測試 19 個、內部單元測試 5 個（heuristic、padding、SplitMix64）。重點：
  - n ≤ M0 + 1 且 ef ≥ n 時，結果與 exact 完全一致（5 個 seed）。
  - graph 不變量：鄰居數上限、無自環 / 重複、鄰居 level 足夠、entry 在最高層、layer 0 的 2M 容量有被用到。
  - level 分佈符合 `P(level ≥ l) = 1/M^l`。
  - 決定性：相同 seed 與順序產生相同 graph；分批 add 等於一次 add；失敗的 add 不消耗亂數。
  - Recall 隨 efSearch 整體改善或飽和。
- synthetic 資料（n = 3000、d = 16、M = 12、efC = 100、200 queries）的 Recall@10：

  | efSearch  | 10     | 20    | 40    | 80  | 160 | 320 |
  | --------- | ------ | ----- | ----- | --- | --- | --- |
  | Recall@10 | 0.8775 | 0.965 | 0.997 | 1.0 | 1.0 | 1.0 |

  只作 correctness，不作 benchmark 結論。
- mutation 檢查：以下錯誤都會被測試抓到。
  - simple selection
  - `M0 = M`（原本抓不到，已補斷言）
  - ef 不取 `max(ef, k)`
  - mL 錯誤
  - level 不用亂數

### 驗證

`./scripts/test.sh` 全部通過：rustfmt、clippy `-D warnings`、Rust 58 個、maturin build、pytest 50 個。

## 本次發現（如實記錄）

- **大量完全重複的向量會讓 layer 0 斷裂**（300 點、5 方向各 60 份、M = 4：ef = 300 只到得了 9 個點）。
- 原測試假設「ef ≥ n 就等於 exact」，在 n > M0 + 1 時沒有保證，所以改為斷言 graph 不變量與補齊格式。
  機制與影響詳見 decision 004「實作後的觀察」。
- hnswlib 是否有相同行為**尚未實測**。

## 下一步（Week 1，依序）

1. **save / load**：
   - 先寫 decision log，定檔案格式：magic / version、dim、params、vectors、graph、entry point，以及是否存 RNG 狀態。
   - 先寫 round-trip 測試：load 後 graph 與 search 結果逐位元相同；load 後繼續 add 的結果與未存檔時相同（需要存 RNG 狀態）。
2. **binding 加 `HnswIndex`**；`AnnIndex` 加 efSearch 介面，需另寫 decision log。
   Python 端把 `PADDING_ID` 轉成 `-1`（002）。
3. **安裝 hnswlib、faiss-cpu 並寫 adapters**：依 002，單執行緒、同一份 unit vectors、輸出轉成統一格式。
4. **第一個小型真實資料集**：選擇與切分需先寫 decision log。
   同時報告重複率與結果不足 k（padding）的 query 比例（004 的觀察）。
5. **第一張 recall vs efSearch 圖。**

## Blocked

沒有 hard blocker。

## 需要使用者決定或留意

- 本次沒有跑任何正式 benchmark，也沒有取用 Mac heavy lock 或遠端機器（都不需要）。
- session 結束時，Mac heavy lock 由 incident-investigator 的 M1 smoke test 持有（process 仍在執行，不是 stale lock）。
  本專案沒有碰這個 lock；之後要跑正式 benchmark 時再檢查。

## SPEC 與現況差異

目前沒有發現差異。
