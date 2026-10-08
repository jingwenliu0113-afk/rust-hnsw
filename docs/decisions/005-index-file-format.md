# 005：HNSW index 檔案格式（save / load）

日期：2026-10-08
狀態：採用。草稿經三個角度的審查與反駁後定稿；審查摘要見文末。

## 背景

SPEC Core 1 要求 HNSW 支援 save / load。需要先固定：

- 檔案格式與版本欄位。
- 必要的 metadata。
- 向後相容策略。
- 損壞與截斷檔案的處理。
- round-trip correctness 的驗證方式。

## 決策

### 範圍

- 只有 `HnswIndex` 有 save / load。`ExactIndex` 只作 ground truth，每次重建，不存檔。
- 這是本專案自訂的格式，不與 hnswlib / FAISS 的檔案相容，也不嘗試讀它們。
- 與 hnswlib / FAISS 的公平比較一律在記憶體中建 index，不經過檔案。
- insertion order 實驗的 id 映射不存在檔案內（見 004）。

### 編碼

- 所有整數與浮點數都是 little-endian，與執行機器無關。
- f32 以 IEEE 754 bit pattern 原樣寫入，round-trip 逐位元相同（包括 `-0.0`）。
- id 一律為 u32（`NodeId`）。

### 所有版本共同的前綴（永久固定）

所有版本的前 12 bytes 永遠是：

- `magic`：8 bytes，ASCII `RHNSWIDX`（`52 48 4E 53 57 49 44 58`）。
- `format_version`：u32。

之後的佈局（包括 header 長度、CRC 的位置與涵蓋範圍）都由 `format_version` 決定。

### v1 佈局

#### Header（76 bytes）

| offset | 欄位              | 型別      | 說明                                                                     |
| ------ | ----------------- | --------- | ------------------------------------------------------------------------ |
| 0      | `magic`           | `[u8; 8]` | `RHNSWIDX`                                                               |
| 8      | `format_version`  | u32       | `1`                                                                      |
| 12     | `flags`           | u32       | 保留；v1 必須為 `0`                                                      |
| 16     | `metric`          | u32       | `1` = cosine（見「v1 語意綁定」）                                        |
| 20     | `dim`             | u64       | 向量維度                                                                 |
| 28     | `m`               | u64       | `HnswParams::m`                                                          |
| 36     | `ef_construction` | u64       | `HnswParams::ef_construction`，存使用者設定值（不是 `max(efC, M)`）      |
| 44     | `seed`            | u64       | `HnswParams::seed`                                                       |
| 52     | `rng_state`       | u64       | SplitMix64 的 state 欄位，也就是下一次 `next_u64` 之前的值；空 index 為 `seed` |
| 60     | `len`             | u64       | 節點數 n                                                                 |
| 68     | `entry_point`     | u32       | entry point id；n = 0 時為 `0xFFFF_FFFF`                                 |
| 72     | `header_crc32`    | u32       | bytes `[0, 72)` 的 CRC-32                                                |

#### Payload（從 offset 76 開始）

1. **vectors**：n × dim 個 f32，依 id 順序的 row-major unit vectors。
2. **graph**：依 id 順序，每個節點寫：
   - `level`：u32。
   - 對 `l = 0..=level`：`count`（u32），接著 `count` 個 u32 neighbor id，順序與記憶體中完全相同。

#### Trailer

- `file_crc32`：u32，bytes `[0, 72)` 接上 `[76, file_len − 4)` 的 CRC-32，也就是排除 header CRC 欄位的 header 加上 payload。
  從初始值重新計算，不與 header CRC 串接。
  - **為何排除 header CRC 欄位**：合法的 header 都是「內容 + CRC(內容)」。CRC 是線性的，所以兩個合法 header 的差異本身是 CRC codeword。
  - 若 file CRC 也涵蓋 header CRC 欄位，把 header 換成另一個合法 header 時，file CRC 不會改變。
  - 這個盲點是實作後由「互換 payload」測試抓到的（見文末）。
- trailer 之後不得再有任何 byte。
  放寬這條規則必須 bump `format_version`，並重新檢討單一 bit 偵測的保證（見下方驗證方式第 6 項）。
- n = 0 時 payload 為空，整個檔案恰好 80 bytes，`file_crc32` 等於 `header_crc32`。

#### CRC-32

IEEE 802.3 / zlib 的 CRC-32：reflected polynomial `0xEDB88320`，init 與 xorout 皆為 `0xFFFF_FFFF`。

- 檢查值：`"123456789"` → `0xCBF4_3926`。
- `crc32(b"") = 0`。
- Python 可用 `zlib.crc32` 獨立驗證。

### v1 語意綁定

v1 檔案的意義依賴以下現行定義：

- `metric = 1` 依 001：存 f32 unit vector，距離為 f32 的 `1 − dot`。
- `rng_state` 依 004：
  - SplitMix64 的狀態轉移是每次 `next_u64` 都讓 state 加 `γ = 0x9E37_79B9_7F4A_7C15`。
  - 每次成功 insert 恰好呼叫一次 `next_u64`；失敗的 add 不抽亂數。
  - `U` 的映射與 level 公式依 004。
- 容量：layer 0 為 2m，其他層為 m。
- entry point 規則：新點的 level **嚴格大於**目前最高層時，才成為 entry。

之後的 decision 若修改以上任何一項，都必須同時 bump `format_version`，並在該 decision log 說明舊版檔案如何處理（拒絕或轉換）。

### 參數範圍（補充 004）

- `HnswIndex::new` 與 load 共用同一組規則：dim ≥ 1、`2 ≤ m ≤ 2^31 − 1`（= `NodeId::MAX / 2`）、efC ≥ 1。
- m 的上限是本次新增的。004 只規定 m ≥ 2；審查時發現現行 `new()` 接受 `m = usize::MAX`，之後 `max_neighbors` 的 `2 * m` 會溢位：
  - debug build 會 panic。
  - release build 會 wrap 成 0 並清空鄰居。
- 上限保證 2m 能放進 u32，也能放進任何平台的 usize。
- 實際使用的 M 遠小於此（hnswlib 也會警告過大的 M），所以不影響任何實驗。

### 不存的東西

- 原始（未正規化）向量：index 本來就只存 unit vectors（001）。
- `mL`：load 必須先呼叫 `HnswIndex::new(dim, params)`，由 new 計算 `mL` 並套用參數範圍規則
  （`HnswError` 轉成 `Corrupt`；u64 → usize 轉換失敗也是 `Corrupt`），之後才填入 vectors、graph、entry point 與 RNG 狀態。
- 最高層：由 entry point 的 level 推導（load 時驗證一致）。

### Load 的驗證

**讀取與錯誤判定規則：**

- 依序讀取固定大小的欄位或區塊：magic 8 bytes → version 4 bytes → header 其餘 64 bytes → vectors → 每個節點的 level、count、neighbor ids → trailer 4 bytes。
- 每個欄位或區塊讀滿之後才檢查內容。
  每個 neighbor id 是獨立欄位：讀到就檢查範圍與自環；同一清單的重複，在整份清單讀完後檢查。
- 任何讀取遇到 `io::ErrorKind::UnexpectedEof` 一律回 `Truncated`；其他 io 錯誤才回 `Io`。
  所以短於 8 bytes 的檔案回 `Truncated`；8 bytes 以上而 magic 不符回 `BadMagic`。
- 回傳讀取過程中最先偵測到的錯誤。
- 任何不符都回傳錯誤，絕不 silent accept，也不 panic。

**依序檢查：**

1. magic 不符 → `BadMagic`。
2. version 不在支援清單（目前只有 `1`）→ `UnsupportedVersion`。
   - 只讀前 12 bytes，不依 version 決定任何大小，也不嘗試解析其餘 header。
3. 讀 header 其餘部分，header CRC 不符 → `HeaderChecksumMismatch`。
   - 在相信任何大小欄位之前驗證。CRC 仍涵蓋 `[0, 72)`，包括 magic 與 version。
4. flags ≠ 0 → `UnsupportedFlags`；metric ≠ 1 → `UnsupportedMetric`。
5. header 一致性 → `Corrupt`：
   - 參數範圍：透過 `HnswIndex::new`。
   - `len ≤ NodeId::MAX`。
   - vectors 的 byte 數 `n × dim × 4` 以 u64 checked 計算，溢位即 `Corrupt`。
   - n = 0 時 entry point 必須是 `0xFFFF_FFFF`；n > 0 時必須 `< n`。
   - `rng_state == seed.wrapping_add(len.wrapping_mul(γ))`；n = 0 時即要求 `rng_state == seed`。
6. vectors → `Corrupt`：
   - 所有值 finite。
   - 每個向量以 f64 累加平方和計算 norm（與 001 相同），且 `|‖v‖ − 1| ≤ 1e-3`。
     合法 writer 的誤差約 ≤ 2^-24、與 dim 無關，所以 1e-3 只用來擋明顯錯誤的資料；位元層級的損壞由 CRC 偵測。
     代價是：手工構造、norm 偏差接近 1e-3 的檔案，搜尋距離可能超出 `[0, 2]` 約 2e-3（`Neighbor` 的文件已註明）。
   - **只檢查、不修改**：load 不得重新正規化向量（不得經過 `normalize_into` / `unit_rows` / `add`），
     向量的 bit pattern 必須與檔案完全相同。
7. graph → `Corrupt`：
   - `level ≤ 64`。`U ≥ 2^-53` 時 level 最大為 `floor(53 · ln 2 / ln M) ≤ 53`；64 是與 M 無關的整數安全上限。
   - **在配置該鄰居清單之前**，檢查 `count ≤ min(cap(l), n − 1)`，其中 `cap(0) = 2m`、`cap(l ≥ 1) = m`。
     `n − 1` 由「id < n、不是自己、不重複」蘊含，提前檢查。
   - neighbor id `< n`、不是自己、同一清單內不重複。
   - 讀完所有節點後：
     - 每個 neighbor 的 level ≥ 該層。
     - n > 0 時，entry point 必須等於 `min{ id : level(id) = 所有節點 level 的最大值 }`。
8. file CRC 不符 → `FileChecksumMismatch`。
9. trailer 之後還有資料 → `TrailingBytes`。

payload 損壞時，可能先觸發第 6、7 項的 `Corrupt`，或在最後觸發 `FileChecksumMismatch`；兩者都是明確失敗。

**記憶體上限：**

- load 中任何 `with_capacity`、`reserve`、`vec![_; x]`、`resize` 的大小，都不得直接取自尚未被已讀資料證實的檔案宣稱量（n、dim、n·dim、level、count）。只能用以下三種方式之一：
  1. 隨實際讀入的資料成長：vectors 以固定大小的 chunk 讀入後 extend，節點逐一 push。
  2. 先與固定的小常數取 min。
  3. 使用已被證實的量，例如通過 `count ≤ min(cap, n − 1)` 之後的 count。
     vectors 已先讀完，所以單一清單的配置不超過已讀資料。
- norm 檢查直接在已讀入的資料上計算，不另外配置 dim 大小的 buffer。
- 因此峰值配置 ≤ c × 已讀 bytes + 固定常數；c 是小常數，來自每個節點 `Vec<Vec<NodeId>>` 的開銷。

### 錯誤型別

新增 `PersistError`，只用於 save / load：

- `Io`
- `Truncated`
- `BadMagic`
- `UnsupportedVersion`
- `HeaderChecksumMismatch`
- `UnsupportedFlags`
- `UnsupportedMetric`
- `Corrupt`（附可讀的原因）
- `FileChecksumMismatch`
- `TrailingBytes`

不重用 `HnswError`：`std::io::Error` 無法 `Clone` / `Eq`。

### API

- `HnswIndex::save(&self, writer: impl Write)`：結尾必須呼叫 `writer.flush()?`。
- `HnswIndex::load(reader: impl Read)`：以小欄位讀取，呼叫端應傳入 buffered reader。
- `save_to_path`：
  1. 寫入同目錄的暫存檔（經 `BufWriter`）。
  2. `BufWriter::into_inner()?` 取回 flush 錯誤。
  3. `File::sync_all()?`。
  4. `rename` 成目標檔名。
  任一步失敗都刪除暫存檔、回 `Io`，且絕不 rename。
- `load_from_path`：以 `BufReader` 包裝後呼叫 `load`。

### 向後相容策略

- `format_version` 從 1 開始。任何改變 byte 佈局或語意（含「v1 語意綁定」各項）的變更，都必須 bump version 並寫 decision log。
- `flags` 在 v1 只是保留欄位，必須為 0。
  任何 flag bit 的意義只能在新的 `format_version` 中定義，同一版本內不得新增 flag 語意。
- loader 明確列出支援的版本。遇到未知（較新）的版本，只讀前 12 bytes 就回 `UnsupportedVersion`，絕不猜測。
- 新版程式**必須**保留讀取舊版的能力（依版本分支解析），除非 decision log 明確放棄該版本。
  放棄時，把對應的 golden test 改為斷言 `UnsupportedVersion`。
- save 一律寫目前最新的版本。
- **golden fixture**：
  - 以 hex literal 寫在測試原始碼中，依本文件手工編碼；永不由 `save` 產生或覆寫，也永不重新產生。
  - 至少兩份：
    - (a) n = 0 的完整 80-byte 檔。
    - (b) 小型多層 graph：至少一個節點 level ≥ 1、至少一個空的鄰居清單、至少一個 layer-0 清單長度大於 m（涵蓋 2m 上限）。
  - 內容必須滿足本文件所有 load 驗證規則。
  - 測試逐欄斷言解碼結果等於手寫的預期值：dim、params（ef_construction 為原值）、len、entry point、每個節點的 level、每層鄰居與順序、向量的 bit pattern，以及 rng_state（透過 save 輸出的 bytes 比較）。並對固定 query 斷言搜尋結果。
  - 只要目前版本仍是 1，另外斷言 `save(load(golden)) == golden`。

### Round-trip correctness 的驗證方式

**測試用 index**（各自固定 seed、dim、m、efC 與 insertion order）：

- (a) n = 0。
- (b) n = 1。
- (c) 多層 graph：小 m（例如 M = 4）、n ≥ 1000，使 max level ≥ 2。
- (d) 004 的大量重複向量資料（會出現結果不足 k）。

**驗證項目：**

1. **結構與參數一致**：以 crate 內的 `#[cfg(test)]` unit test 逐欄位比較：
   - dim、params、`mL.to_bits()`。
   - 每個向量值的 `to_bits()`。
   - 每個節點每層的鄰居（含順序）、entry point、RNG 狀態。

   不為公開型別 derive `PartialEq`（`f32 ==` 會把 `-0.0` 與 `0.0` 視為相等）。
   鄰居順序只能由本項與第 4 項驗出，第 2 項的搜尋結果對它不敏感。
   必須抓到的 mutation：load 時反轉每個鄰居清單。
2. **搜尋等價**（使用者可見行為的端到端驗證；graph 的精確比較由第 1、4 項負責）：
   loaded 與 original 對同一批 query 的結果逐位元相同（id，以及 distance 的 `to_bits`）。至少涵蓋：
   - (c)：至少 100 個 query，efSearch 包含尚未飽和（Recall@10 < 1）的值。
   - (d)：用 `search_batch` 比較，包括 `PADDING_ID` / `+inf` 的位置。
   - (b)。
   - (a)：兩邊 search 都回 `EmptyIndex`。
3. **繼續插入**：對 original 與 loaded 各自 add 同一批向量，斷言 `save(loaded') == save(original')`。
   - 以 bytes 比較，涵蓋 rng_state、所有 level、鄰居清單與順序。
   - 使用小 M（例如 M = 4），繼續 add ≥ 100 點，使新點中幾乎必定出現 level ≥ 1。
   - 包含 (a)：空 index save → load → add，結果與 new 出來的 index add 同一批相同。
   - 必須抓到的 mutation：load 時把 RNG 重設為 `SplitMix64::new(seed)`。
   - 另加 crate 內 unit test：任意 add 序列（含失敗的 add）之後，RNG 狀態 == `seed + len·γ`。
     日後若改變每次 insert 抽取亂數的次數，這個測試會紅燈，強制 bump version。
4. **決定性**：同一 index save 兩次 bytes 完全相同；`save(load(save(x))) == save(x)`。
5. **規格一致**：
   - 測試內依本文件獨立實作的 v1 encoder / decoder 與逐 bit CRC-32（不呼叫 crate 的 save / load 內部程式）。
   - `decode(save(x))` 的欄位與公開 API 一致，向量等於輸入資料經 `normalize` 的 bit pattern。
   - `encode(decode(save(x))) == save(x)`。
   - CRC 實作符合上述檢查值。
6. **損壞與截斷**：
   - 小 index 的檔案在每一個長度截斷 → 一律 `Truncated`。
   - 每一個 bit 單獨翻轉 → 一律回傳錯誤。保證分三種情況：
     - 解析長度不變：CRC 涵蓋相同的 bytes，由 CRC-32 對單一 bit 錯誤的保證偵測（途中也可能先觸發 `Corrupt`）。
     - 解析終點超出檔尾：`Truncated`。
     - 解析終點提前：`TrailingBytes` 或 checksum 錯誤。

     「trailer 之後不得有任何 byte」與「EOF 即 `Truncated`」是這個保證的必要條件。
   - 檔尾多出 bytes → `TrailingBytes`。
   - magic 錯誤 → `BadMagic`；不足 8 bytes → `Truncated`。
   - magic 正確、version = 2，後面接 0 bytes 或 64 bytes 任意資料（不重算 CRC）→ `UnsupportedVersion`。
   - header CRC 錯誤 → `HeaderChecksumMismatch`。
   - flags、metric 錯誤（重算 header CRC，讓檢查走到該欄位）→ 對應的錯誤。
   - 只差 efC 的兩個 index 互換 payload → `FileChecksumMismatch`。
   - CRC 正確但語意不合法 → `Corrupt`，不得 panic。第 5～7 項的每一條至少一個案例，包括：
     - m = 1、2^31、2^63、u64::MAX。
     - dim = u64::MAX（n × dim 溢位）。
     - rng_state 或 seed 被改。
     - entry 改成另一個同為最高層的節點。
     - count = u32::MAX（m 很大、n 小）。
   - **記憶體上限**：放在獨立的 integration test 檔（檔內只有這一個測試，避免平行測試干擾計數），
     安裝一個包住 `System`、只記錄 peak bytes 的 `#[global_allocator]`。案例：
     - CRC 正確、n = `NodeId::MAX`、dim = 2^20、沒有 payload → `Truncated`。
     - CRC 正確、m 很大、n 小、某個 count = u32::MAX → `Corrupt`。

     兩者的 peak 配置都必須小於 1 MiB。必須抓到的 mutation：依 header 的 n 或 count 預先 reserve。
   - save 端：用一個寫到第 k byte 就回錯誤的 `Write`，對小 index 的每個 k 斷言 save 回 `Io`。

## 為什麼

- **保證範圍**：
  - 檔案在任何機器上都能讀，load 後的向量與 graph 逐位元相同。
  - 「load 後繼續 add 與從未存檔逐位元相同」只保證在同一平台、同一 build 下成立：level 計算使用 `f64::ln`，其精度依平台與 Rust 版本而異。
  - 本文件不對跨機器的建圖或搜尋結果做保證。
  - index 檔不記錄建圖程式的版本；需要時由 harness 依 EXPERIMENT_RULES 記錄 git commit。
- **保留鄰居順序**：這是 byte 層級決定性的前提，也是「load 後繼續 add 的 bytes 與未存檔時相同」的前提
  （未被 prune 的清單以 push 追加，順序會延續到之後的 bytes）。
  在現行實作中，因為 `(distance, id)` 是全序，鄰居順序不影響 search 結果，也不影響繼續 add 後各清單的鄰居集合。
- **header CRC 與 file CRC 的分工**：header CRC 在信任大小欄位之前先驗證 header；
  file CRC 涵蓋 header 欄位與 payload，把兩者綁在一起，防止拼接不同檔案的 header 與 payload。
  file CRC 刻意不涵蓋 header CRC 欄位，理由見 Trailer 一節。
- **CRC 之外再做語意驗證**：防範 CRC 正確、但由有 bug 的 writer 產生的檔案。
  例如 neighbor 越界、level 不一致，會讓 search 時 index 越界而 panic；
  rng_state 或 entry point 錯誤則會讓結果默默偏離。
- **固定 12-byte 前綴**：v1 loader 遇到新版檔案時，能回 `UnsupportedVersion`，而不是誤報為損壞。
- **little-endian 固定**：檔案可在其他機器讀取。
- **採用 zlib 相同的 CRC-32**：外部工具可獨立驗證。

## 放棄的替代方案

- **serde / bincode 等序列化 crate**：格式由外部 crate 版本決定，難以保證長期相容，也難以逐 byte 規格化。
- **hnswlib 的檔案格式**：綁定其記憶體佈局（level 0 的固定大小 block），與本專案的 baseline 表示法不符。
- **只做結構驗證、不做 checksum**：無法偵測向量 mantissa 的位元翻轉。
- **整份讀入記憶體後才驗證 CRC**：load 時需要約兩倍記憶體。
- **header CRC 在 version 之前驗證**：新版檔案會被誤報為損壞。
- **payload CRC 只涵蓋 payload**：header 與 payload 沒有綁定。
- **file CRC 涵蓋整個檔案（含 header CRC 欄位）**：兩個合法 header 的差異是 CRC codeword，換 header 時偵測不到。
- **不存 RNG 狀態、load 時重設為 seed**：繼續 add 會與未存檔時不同。
- **由 (seed, len) 推導 RNG 狀態**：目前可行，但會把 004「每次 insert 抽一次亂數」的實作細節隱含在 loader 裡。
  因此選擇明確存放，並在 load 時以推導值交叉檢查。
- **用 flags 擴充格式**：與「任何語意變更都 bump version」衝突，會有兩條擴充路徑。

## 對實驗的影響

- 檔案大小不自動等於 SPEC 的「index size」指標。index size 的定義（in-memory bytes 或檔案大小）在 benchmark protocol 中另行決定。
- load / save 時間不是 SPEC 指標，本階段不做 benchmark。
- 之後若有任何改變檔案內容的擴充，一律以新的 `format_version` 與新的 decision log 處理。
- `HnswIndex::new` 新增 m 上限（2^31 − 1），不影響任何實際使用的參數。

## 審查摘要

草稿由三個角度審查：損壞偵測與記憶體安全、規格精確度與向後相容、round-trip 語意與一致性。
每個角度的 findings 再由獨立的反駁者查證，共 37 個 findings：採納 32 個（含修改後採納），駁回 5 個。

主要修正：

- 固定 12-byte 前綴，並改為先檢查 version。
- 新增 m 上限，修補 004 / 現行 `new()` 的溢位缺口。
- 在配置前檢查 `count ≤ min(cap, n − 1)`。
- trailer 改為整個檔案的 CRC。
- 新增 rng_state 與 entry point 的交叉檢查。
- 記憶體上限改為可檢查的規則，並以計數 allocator 驗證。
- golden fixture 改為逐欄斷言解碼值。
- 繼續插入改以 bytes 比較，並用小 M。
- save 端必須回報 flush / sync 錯誤。
- 更正「鄰居順序影響搜尋結果」的錯誤理由。

### 實作後的修正（2026-10-08）

審查時採納的「trailer 改為整個檔案的 CRC」無法達成它要的效果。

- 「互換 payload」測試的拼接檔（index A 的 header + index B 的 payload 與 trailer）被 load **接受了**。
- 原因：兩個合法 header 的差異（含各自的 header CRC 欄位）是 CRC codeword，所以整個檔案的 CRC 完全相同。
- 修正：file CRC 改為排除 header CRC 欄位。拼接檔改為回 `FileChecksumMismatch`。
- 當時 v1 尚未 commit 或發布，所以 golden fixture 的 trailer 依新定義以 Python 重新產生。
  這是格式定稿前的修改，不違反「golden 永不重新產生」的規則；該規則自 v1 commit 起適用。

### 實作後的第二輪審查（2026-10-08）

實作完成後，再做一輪獨立審查：三個角度（規格符合性、惡意輸入、測試鑑別力），各配一個反駁者，另有一個隔離的 fuzz。

**fuzz：**

- 約 2,800 萬次 load，0 個 panic、0 個依規格判定的 silent accept。
- harness 先用 4 份刻意弄壞的實作自我驗證過。
- 所有輸入的峰值配置都是 64 KiB。

**實作修正（與規格一致化，沒有改變格式）：**

- trailer 之後的探測讀取遇到 `UnexpectedEof` 錯誤時，改為回 `Truncated`（原本回 `Io`）。
- `save_to_path` 的暫存檔已存在時，錯誤訊息附上暫存檔路徑；仍然不刪除不屬於這次的檔案。

**測試補強：**每一項都是「拿掉某條檢查後原測試仍全綠」的缺口，由反駁者以 mutation 重現。

- per-layer 容量（2m / m）：改用只違反這一條的案例。原案例在 golden 中剛好 n − 1 = 2m，或先被其他檢查擋下。
- `level ≤ 64`：測 level 64 被接受、65 被拒絕。原案例會先被 entry 檢查擋下。
- header CRC 的驗證順序：header 每個 bit 單獨翻轉，依位置斷言錯誤種類（BadMagic / UnsupportedVersion / HeaderChecksumMismatch）。
- `n × dim` 不溢位、`× 4` 才溢位的情況。
- 每個 header 案例只給 header 時也必須回 `Corrupt`，確認是在 header 階段擋下的。
- load 端的 I/O 錯誤分類，以及 trailer 之後的讀取錯誤。
- `save_to_path` 在暫存檔建立後才失敗（rename 到既有目錄）時，會刪除暫存檔。
- efC < m 的 index 也做 round-trip，驗證存的是原值。
- count 上限：保留本文件寫的 `count = u32::MAX`（由 2m 上限擋下），另加 `count = 2m`（m = MAX_M 時為 2^32 − 2），
  後者通過 2m 上限，專門驗證 `n − 1` 上限在配置前生效。

**駁回：**

- 檔名接近 255 bytes 時的暫存檔名長度。
- 在 004 回頭加交叉引用（上限已記在本文件「參數範圍」）。
- level 64 off-by-one 的獨立 finding（已併入上面的 level 測試）。
- 單一 bit 偵測論證的措辭。
