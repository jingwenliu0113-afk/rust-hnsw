# NEXT

## 目前位置

**Core / Week 1 進行中**：

- 步驟 1「HNSW insert / search」已完成。
- 步驟 2「save / load」已完成。
- 步驟 3「HNSW 的 Python binding 與 efSearch 介面」已完成。
- 下一步是步驟 4「hnswlib / FAISS adapters」。

尚未進 Advanced（Gate 條件都還沒有資料可判斷），沒有做任何 Optional。

## Commits

- `ddcb27d chore: establish core baseline`
- `7961673 feat(hnsw): add HNSW insert/search with correctness tests`
- `e12fe07 docs: define Docker resource policy`
- `50f469a feat(hnsw): add versioned save/load`
- `feat(python): add HNSW binding and efSearch interface`（含 README / NEXT 的更新）

依使用者要求拆成三個 commit。每個 commit 前，都在「工作目錄只剩該 commit 內容」的狀態下跑 `./scripts/test.sh` 並通過。

## 上次完成（2026-10-08）

### MACHINE_POLICY.md：Docker 規則（使用者要求）

- 新增「Docker」一節：Core 階段不使用 Mac Docker Desktop。
- 禁止 restart、全域 prune、操作其他 Project 的 container / volume / network。
- 真的需要時先停下來提出理由，不得自行導入。

### save / load（decision 005）

**流程：**

1. 寫格式草稿。
2. 審查：三個角度各配一個反駁者，共 37 個 findings，採納 32 個。
3. 定稿後先寫測試，確認紅燈。
4. 實作。
5. mutation 檢查。
6. 實作審查（三個角度 + 反駁者），加上隔離的 fuzz。
7. 依結果補強。

**格式（v1）：**

- 固定 12-byte 前綴（magic + version），76-byte header 含 header CRC。
- 接著是 vectors（f32 bit pattern）、graph（level、count、ids，保留鄰居順序），最後是 file CRC。
- 存放 RNG 狀態，並在 load 時以 `seed + len·γ` 交叉檢查。
- entry point 必須是最高層中 id 最小者。
- load 以串流方式讀取，配置量只隨實際讀到的資料成長。

**API：**`save` / `load` / `save_to_path`（暫存檔 + `sync_all` + rename）/ `load_from_path`，錯誤型別為 `PersistError`。

**實作中由測試抓到的設計缺陷：**

- 「互換 payload」測試的拼接檔被 load 接受。
- 原因：file CRC 涵蓋 header CRC 欄位時，兩個合法 header 的差異是 CRC codeword。
- 修正：file CRC 排除 header CRC 欄位。詳見 005「實作後的修正」。

**補上 004 的缺口：**`HnswIndex::new` 原本沒有 m 上限，`2 * m` 可能溢位；新增 `MAX_M = 2^31 − 1`。

**測試：**

- Rust 共 100 個：crate 內 9、distance 14、exact 20、hnsw 20、load_alloc 1、persist 36。
- 包含：
  - 每個長度截斷、每個 bit 翻轉。
  - Python `zlib` 獨立產生的 golden fixture，以及依規格獨立撰寫的 encoder / decoder。
  - 計數 allocator 驗證記憶體上限。
  - 寫入失敗、讀取失敗、rename 失敗。

**fuzz：**約 2,800 萬次 load，0 個 panic、0 個 silent accept，峰值配置固定 64 KiB。

**mutation：**decision 005 列出的必抓 mutation，以及第二輪審查找到的 11 個測試缺口，全部會被測試抓到。

### HNSW 的 Python binding 與 efSearch 介面（decision 006）

- **介面**：`HnswAnnIndex` protocol 新增 `set_ef_search(ef)`。
  - ef 沒有預設值，必須在 search 前明確設定。
  - 先檢查再存下：非整數回 `TypeError`，`ef < 1` 回 `ValueError`，失敗時保留前一個值。
- **binding**：`rust_hnsw.HnswIndex(dim, m, ef_construction, seed)`。
  - 方法：`add` / `search(queries, k, ef_search)` / `save` / `load`。
  - 屬性：dim、m、ef_construction、seed、max_level。
  - `PADDING_ID` 轉成 `-1`；計算期間釋放 GIL。
  - 錯誤對應：`HnswError` 與格式錯誤 → `ValueError`；I/O 錯誤 → `OSError` 子類。
- **adapter**：`RustHnsw(m, ef_construction, seed)`，參數必須是整數，不默默套用預設值。
- **審查**：兩個角度各配一個反駁者，共 15 個 findings。
  - 沒有實作錯誤；修正集中在測試鑑別力：GIL 釋放、ef 確實傳進核心、sweep 語意、max_level。
  - 審查找到的 binding / adapter mutant，加上我自己的 5 個，全部會被測試抓到。

### 驗證

`./scripts/test.sh` 全部通過：rustfmt、clippy `-D warnings`、Rust 100 個、maturin build、pytest 76 個。

## 下一步（Week 1，依序）

1. **安裝 hnswlib、faiss-cpu 並寫 adapters**：依 002，單執行緒、同一份 unit vectors、輸出轉成統一格式。
   寫 adapter 前，先確認 006「寫 hnswlib / FAISS adapter 前必須確認」的事項，並寫 decision log。目前都未確認：
   - 結果不足 k 時，hnswlib 是否整批 raise；FAISS 補齊時的 id 與距離值。
     用 `duplicate_data(12, 300, 6)`、k = 80 實測。若 hnswlib 會 raise，不得改成 per-query 迴圈。
   - `ef < k` 時的語意，以及預設的 ef、M、efC。
   - hnswlib ip space 回傳的距離定義。
   - 006 已記錄：runner 必須對每個 adapter 分別量測空呼叫成本（包含後處理）。
   - 要在 `requirements-dev.txt` 固定版本。
2. **benchmark runner**：
   - 在計時前檢查每組 (ef, k) 都滿足 `ef ≥ k`。
   - 依 002 / 006 的計時協定，記錄 machine / CPU / git commit 等 metadata。
   - 正式計時需透過 `./tools/with-mac-lock`。
3. **第一個小型真實資料集**：選擇與切分需先寫 decision log。
   同時報告重複率與結果不足 k（padding）的 query 比例（004 的觀察）。
   下載大型資料集前會先向使用者確認。
4. **第一張 recall vs efSearch 圖。**

## Blocked

沒有 hard blocker。

## 需要使用者決定或留意

- 本 session 沒有跑任何正式 benchmark，也沒有取用 Mac heavy lock 或遠端機器（都不需要）。
- fuzz 屬於 correctness 工作，不是 benchmark。執行時 incident-investigator 正持有 heavy lock；
  fuzz agent 等鎖釋放後才開始跑，沒有搶資源，也沒有碰這個 lock。
- session 結束時，heavy lock 又由 incident-investigator 持有（process 仍在執行，不是 stale lock）。本專案沒有碰它。
- 本 Project 沒有使用 Docker。

## SPEC / decision 與現況差異

- 004 只規定 M ≥ 2，`HnswIndex::new` 原本沒有上限。005 補上 `MAX_M = 2^31 − 1`，不影響任何實驗參數。
- 其他沒有發現差異。
