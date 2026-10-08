# 013：正式 run 的格點（formal config）

日期：2026-10-09
狀態：採用（在任何正式結果之前）

## 背景

- 010 規定：第一次 formal run 之前，formal config 必須列出 ef 格點與完整 variance 的格點，並 commit；每個資料集至少一格跑 ≥ 5 個 seed 與 ≥ 5 個 shuffle；
  主要結論附上 variance 格點的 seed 範圍；看到正式結果之後才增加格點，必須另寫 decision。
- SPEC §6 要求至少掃 M、efConstruction、efSearch；§7 要求 scaling with N 與 seed / insertion order variance。
- 登記的 n（frozen ground truth）：GloVe 100,000、300,000、1,183,514（011）；文字 `simplewiki-embeddinggemma2-100k`（產生中）10,000、30,000、100,000（012）。
- 機器：heavy lock 常被其他專案長時間持有，load 常在 8–60；每個子 process 啟動前都要通過 load ≤ 4.0（008 §2）。單一 run 越短越好，報告可以合併同一資料集的多個 run。
- 本文件取代 2026-10-08 的草稿。

設計前看過的資料（全部是 smoke、不在 lock 下）：

| run | 內容 | 用到的 Recall@10（三個實作的範圍） | 用到的成本（每個 build 三個實作合計） |
| --- | --- | --- | --- |
| `20261008T142627Z-smoke`（`glove-correctness.toml`，load 約 9–15） | GloVe 100k、(16, 200)、ef 10–256 | 64：0.829–0.836；128：0.897–0.903；192：0.929–0.933；256：0.948–0.950 | build 99 s；每點 64→38.8、128→70.0、256→137.4 s |
| `20261008T144333Z-smoke`（`glove-cost-probe.toml`，aborted，100k 部分完整；load 約 6–10） | GloVe 100k、(48, 400)、ef 64–1024 | 64：0.923–0.925；128：0.970–0.972；256：0.9925–0.9936；512：0.9990–0.9992；1024：約 0.9999 | build 344 s；每點 95.8 / 168.6 / 335.0 / 582.5 / 1,085.3 s |
| `20261008T154054Z-smoke`（`glove-cost-probe-full.toml`，load 約 4–8） | GloVe 1,183,514、(16, 200)、ef 64–1024 | 64：0.763–0.764；128：0.831–0.832；256：0.883–0.885；512：0.921–0.926；1024：0.952–0.956 | build 1,824 s；每點 59.3 / 102.4 / 185.6 / 354.7 / 723.1 s |
| `20261008T143953Z-smoke`（`text-pilot-correctness.toml`，load 約 6–8） | 文字 pilot 10k、1,000 queries、(16, 200)、ef 10–256 | 16：0.866–0.872；24：0.919–0.920；32：0.943–0.947；64：0.982–0.987；96：0.992–0.995；128：0.996–0.997；256：≥ 0.999 | build 25.4 s；每點 10→3.1、32→6.7、64→11.1、128→19.2、256→32.8 s |

- 選格點**只用了**兩類資訊：(1) recall 的涵蓋範圍（各目標大約在哪個 ef 達到）；(2) 每個 build 三個實作合計的耗時（成本）。
  另外，「任一子 process 佔該 build 合計的最大比例」（實測 0.40–0.45）只用來設 timeout。
- 這些 run 也有各實作的 QPS、latency、build time 與記憶體；格點**沒有使用**任何實作之間的差異。三個實作用同一組格點。
- 比較過三個方向（見「放棄的替代方案」），並從合規、成本、完整性三個角度審查後定稿。

## 決策

### 1. 格點結構（兩個資料集相同）

- 主要 n：兩個資料集都是 100,000（GloVe 最小的登記 n；文字的完整資料集）。
- 主要 n 以 (M, efC) = (16, 200) 為中心，一次只改一個因素，共 6 格：
  - M ∈ {8, 16, 32, 48}（efC 200）；
  - efC ∈ {100, 200, 400}（M 16）。
- scaling：其他登記的 n 只跑 (16, 200)。
- variance：
  - **完整 variance 格點**（010 §4）：(100,000, 16, 200)，seeds 1–5（原始順序），加上 shuffles 101–105（搭配 seed 1）。兩個資料集相同。
  - **GloVe 另加 seed variance 格點** (100,000, 48, 200)：seeds 1–5，不做 shuffle。理由：(16, 200) 在成本上限內達不到 0.99（第 2、3 點）。
  - 其他格點只有 seed 1、原始順序。
- 所有 n 都是登記的 n，用 frozen ground truth。

| 資料集 | (n, M, efC) | seeds | shuffles | 角色 |
| --- | --- | --- | --- | --- |
| GloVe | (100k, 16, 200) | 1–5 | 101–105 | 中心、完整 variance、scaling |
| GloVe | (100k, 48, 200) | 1–5 | — | M 掃描、0.99 的 seed variance |
| GloVe | (100k, 8, 200)、(100k, 32, 200) | 1 | — | M 掃描 |
| GloVe | (100k, 16, 100)、(100k, 16, 400) | 1 | — | efC 掃描 |
| GloVe | (300k, 16, 200)、(1,183,514, 16, 200) | 1 | — | scaling |
| 文字 | (100k, 16, 200) | 1–5 | 101–105 | 中心、完整 variance、scaling |
| 文字 | (100k, 8, 200)、(100k, 32, 200)、(100k, 48, 200) | 1 | — | M 掃描 |
| 文字 | (100k, 16, 100)、(100k, 16, 400) | 1 | — | efC 掃描 |
| 文字 | (10k, 16, 200)、(30k, 16, 200) | 1 | — | scaling |

合計 GloVe 21 個 build、文字 17 個 build，每個 build 三個實作。

### 2. ef 格點

規則（三個實作共用）：

1. **梯子**：10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048（32 以上 ×1.5 與 ×1.33 交替）。
2. 每格估計各目標的 crossing（median Recall@10 第一次 ≥ 目標的 ef）：中央估計與悲觀上緣（下表）。
3. **下界**：中央估計第一個達到 0.90 的梯子點，往下兩格。每格因此有 0.90 以下的點，也供 010 §6 的檢查。
4. **上界**：要涵蓋的最高目標，取其悲觀上緣之上的第一個梯子點。悲觀估計下仍至少有一個達標點，中央估計下通常有兩個。
   不涵蓋的目標與理由寫在下表；有「成本上限」的格點，悲觀估計下不保證涵蓋。
5. **細化點**：同一格（同資料集、n、M、efC、seed 1）的 smoke 中，任一實作在梯子點 p 的 Recall@10 與某個目標相差 ≤ 0.005 時，加入 round(p × 2^(1/4))。
   符合的只有：GloVe 100k (16, 200) 的 128（0.90）→ 152、256（0.95）→ 304；GloVe 1,183,514 (16, 200) 的 1024（0.95）→ 1218。
6. 同一 config 共用一份 ef 清單（取聯集，只會多出點）。同一格的所有 build，不論在哪個 config，ef 清單完全相同。

| 格點 | 0.90 | 0.95 | 0.99 | 自己的範圍 | 所在 config 的 ef 清單 |
| --- | --- | --- | --- | --- | --- |
| GloVe 100k (16, 200) | 128（實測邊界） | 256–270（實測邊界） | 約 1,300，不涵蓋（10 個 build 多約 6–9 h） | 64–384，加 152、304 | 64, 96, 128, 152, 192, 256, 304, 384 |
| GloVe 100k (16, 400) | 約 96–128 | 約 192–256 | 不涵蓋 | 64–384 | 同上 |
| GloVe 100k (16, 100) | 約 128–192 | 約 256–420 | 不涵蓋 | 96–512 | 32, 48, 64, 96, 128, 192, 256, 384, 512 |
| GloVe 100k (48, 200) | 約 48–64 | 約 96–128 | 約 256–384 | 32–512 | 同上 |
| GloVe 100k (32, 200) | 約 64–96 | 約 128–192 | 約 384–700（不確定） | 48–768 | 48, 64, 96, 128, 192, 256, 384, 512, 768 |
| GloVe 300k (16, 200) | 約 192–256 | 約 384–700 | 約 2,000 以上，不涵蓋 | 96–768 | 同上 |
| GloVe 100k (8, 200) | 約 192–400 | 約 512–1,000 以上 | 不涵蓋 | 128–1024（成本上限） | 128, 192, 256, 384, 512, 768, 1024 |
| GloVe 1,183,514 (16, 200) | 384–512 | 約 1024（實測邊界） | 約 10^4，不涵蓋 | 192–1536，加 1218 | 192, 256, 384, 512, 768, 1024, 1218, 1536 |
| 文字 100k (16, 200) | 約 32–48 | 約 48–96 | 約 128–450（不確定） | 24–512 | 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 |
| 文字 100k (16, 400) | 不高於 (16, 200) | 同左 | 同左 | 24–512 | 同上 |
| 文字 100k (16, 100) | 約 32–64 | 約 64–128 | 約 192–600（不確定） | 24–768 | 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024 |
| 文字 100k (8, 200) | 約 64–128 | 約 128–300 | 約 500–1,000 以上 | 48–1024（成本上限） | 同上 |
| 文字 100k (32, 200) | 約 24–32 | 約 32–64 | 約 96–300 | 16–384 | 10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384 |
| 文字 100k (48, 200) | 約 16–24 | 約 24–48 | 約 64–200 | 10–256 | 同上 |
| 文字 10k (16, 200) | 約 24 | 約 32–48 | 約 96–128 | 10–192 | 同上 |
| 文字 30k (16, 200) | 約 24–32 | 約 48–64 | 約 128–230 | 16–256 | 同上 |

- 估計的依據：GloVe 的 (16, 200) 與 1,183,514 是實測；(48, 200) 由 (48, 400) 的實測放寬（efC 減半）；(32, 200) 在 log M 上內插；300k 在 log n 上內插 miss rate；
  (8, 200)、(16, 100)、(16, 400) 是外推。文字全部由 pilot 外推：n 大 10 倍時，達到同一 recall 所需的 ef 取 1.4–5 倍以上（GloVe 100k→1,183,514 為 2.7–3.8 倍，且 recall 越高越大）。
- 文字 100k 從未量過，表中的文字估計不確定度最大。

### 3. variance 與主要結論附上的 seed 範圍

- 每個主要結論（某目標下的 frontier 讀值、實作間比值）附上：
  1. 讀值所在點本身參與的 build 數與跨 seed min–max（010 §2、§4）；
  2. variance 格點的 seed 範圍：達到該目標的每個 variance 格點，在其最小達標 ef（各實作各自的）上的跨 seed min–max。
- 對應：
  - GloVe：0.90、0.95 附 (16, 200) 與 (48, 200)；0.99 附 (48, 200)。
  - 文字：三個目標都附 (16, 200)。
- fallback（事先登記）：variance 格點在其 ef 清單內沒有達到某目標時，附上它在最大 ef 的 seed 範圍，並明示「與讀值不在同一 recall 區間」。
  預期可能用到的是文字 (16, 200) 的 0.99（ef 512）；GloVe (48, 200) 的 0.99 也照此處理（ef 512）。
- 300k、1,183,514、10k、30k 只有 1 個 seed：讀值附上主要 n 的 seed 範圍作為代理，並註明「不同 n；大 n 的 seed 變異可能不同」。
- order variance 只在 (16, 200)；FAISS 的 order variance 分開呈現（010 §4、007 §5）。GloVe 0.99 沒有 order variance 的範圍，照實註明。
- 拆法：
  - seed 1 原始順序與 5 個 shuffle 在同一個 sweep、同一個 run：order variance 在一個 run 內量完，報告也能找到原始順序的 build。
  - seeds 2–5 放在另一個 config，shuffles 為空，避免跨 config 重複 (100k, 16, 200, seed 1, 原始順序)。
  - (48, 200) 的 5 個 seed 分在三個 config。
  - 所以 seed 範圍包含 run 之間的漂移：QPS、latency、build time 的範圍偏保守；recall 與記憶體不受影響。

### 4. scaling

- 固定 (16, 200) 的 build time、peak RSS、index memory：每個登記的 n 都有（010 §5）。
- 各目標的 QPS / P50 / P95：依 010 §5，每個 n 用該 n 的全部點讀 frontier。這是主要讀值，不變。
  - 混淆：主要 n 有 6 格（(16, 200) 是 5 個 seed，GloVe 的 (48, 200) 也是），其他 n 只有 (16, 200)、1 個 seed。QPS 隨 n 的趨勢混入了格點數與 seed 數的差異，照實註明。
  - 事先登記一個**輔助**讀值：每個 n 只用 (16, 200) 的點讀 frontier 的 QPS / P50 / P95，只用來呈現上述混淆，不取代主要讀值。
- GloVe 不做 0.99 的 scaling：300k 與 1,183,514 預期「未達到」（不加高 M 的大 n 格點，見放棄的替代方案）。
  0.99 的 scaling 只由文字提供（10k / 30k / 100k 的 (16, 200)；100k 不確定）。

### 5. formal config 與執行

| config（`benchmarks/configs/`） | build | ef 清單 | child_timeout_s |
| --- | --- | --- | --- |
| `glove-formal-1-m16-order.toml` | (100k, 16, 200) seed 1 + shuffles 101–105 | 64–384，加 152、304 | 2,400 |
| `glove-formal-2-m16-seeds-efc400.toml` | (100k, 16, 200) seeds 2–5；(100k, 16, 400) seed 1 | 同上 | 3,000 |
| `glove-formal-3-m48-seeds-1-2.toml` | (100k, 48, 200) seeds 1、2 | 32–512 | 7,200 |
| `glove-formal-4-m48-seeds-3-4.toml` | (100k, 48, 200) seeds 3、4 | 32–512 | 7,200 |
| `glove-formal-5-m48-seed-5-efc100.toml` | (100k, 48, 200) seed 5；(100k, 16, 100) seed 1 | 32–512 | 7,200 |
| `glove-formal-6-m32-300k.toml` | (100k, 32, 200)；(300k, 16, 200) | 48–768 | 6,600 |
| `glove-formal-7-m8.toml` | (100k, 8, 200) | 128–1024 | 3,000 |
| `glove-formal-8-full.toml` | (1,183,514, 16, 200) | 192–1536，加 1218 | 17,400 |
| `text-formal-1-m16-order.toml` | (100k, 16, 200) seed 1 + shuffles 101–105 | 24–512 | 3,600 |
| `text-formal-2-m16-seeds-efc400.toml` | (100k, 16, 200) seeds 2–5；(100k, 16, 400) seed 1 | 24–512 | 4,800 |
| `text-formal-3-m8-efc100.toml` | (100k, 8, 200)；(100k, 16, 100) | 24–1024 | 4,800 |
| `text-formal-4-highm-scaling.toml` | (100k, 32, 200)、(100k, 48, 200)；(10k, 16, 200)、(30k, 16, 200) | 10–384 | 7,800 |

（「a–b」是梯子上 a 到 b 的全部點。）

- **commit**：
  - 12 份 config 全部 commit 之後，才開始第一次 formal run（兩個資料集都是）。這是 010 的字面要求。
  - GloVe 的 8 份與本文件同一個 commit。
  - 文字的 4 份要等 `simplewiki-embeddinggemma2-100k` 登記（`test_shipped_configs_are_valid` 會以 registry 解析所有 config），與 manifest 同一個 commit。
    內容與附錄逐字相同，唯一例外是第 6 點的 timeout 上調。
  - 文字資料集若遲遲無法登記、又要先跑 GloVe：等於把 010 解讀成「依資料集分別適用」，記到 NEXT 的 Blocked 由使用者決定，不自行決定。
- **全部跑完**：每份 config 都必須跑完，不得只跑一部分。要縮減只能在 commit 之前、以新的 decision 為之。
- **建議順序**（排程，不是協定）：GloVe 1 → 8、文字 1 → 4；兩個資料集可以交錯。variance 與主要比較的格點排在前面，最長的 1,183,514 排在最後。
- **失敗的 run**：
  - `aborted`、`invalid`、`error`：整份 config 重跑並記錄原因（010 §3）。失敗的 run 不交給報告，報告另列被排除的 run 與原因。
    不得把失敗 run 的部分結果與重跑合併：報告會把相同 (n, M, efC, seed, shuffle) 的 build 當成兩個 build 一起取 median。
  - `incomplete`（例如單一子 process 逾時）：照實列出，不重跑（010 §3）。
- **報告**：每個資料集在它的全部 config 跑完之後才產生正式報告，每份 config 恰好貢獻一個 run。
- **第一次 formal run 之前，`report.py` 補上以下檢查**（落實既有規則，不改 010）：
  1. 拒絕在多個 run 中重複出現的（實作, n, M, efC, seed, shuffle）。
  2. 拒絕 status 為 `aborted`、`invalid`、`error` 的 run。
  3. 版本一致，不一致就拒絕合併：
     - hnswlib、faiss 擴充模組的 SHA-256 與 pin 的套件版本相同；`rust_hnsw.BUILD_INFO` 相同；
     - 各 run 的 git commit 之間，`crates/`、`python/`、`Cargo.toml`、`Cargo.lock`、`requirements-dev.txt`，
       以及量測程式（`benchmarks/` 的 run.py、run_child.py、adapters.py、interface.py、protocol.py、evaluate.py、metrics.py、environment.py、datasets.py、config.py）沒有差異；
     - `rust_hnsw` 的 SHA-256 每次 formal 前重新編譯、不保證逐位元相同，只記錄，不作為拒絕條件。
  4. 第 4 點的輔助 scaling 讀值。
  - formal run 期間如果必須改量測相關的程式，另寫 decision；改動前後的 run 不得合併。
- **單一 seed 的格點**：三個實作的執行順序只有一種（008 §5 依 build 輪換；`glove-formal-7`、`glove-formal-8` 只有一個 build，順序就是清單順序）。
  時間漂移可能與實作重疊，照實註明。多 seed 的格點才有輪換的平衡。

### 6. timeout

- child_timeout_s = 最慢子 process 的負載下上限估計 × 5，進位到 600 的倍數，最少 1,800。
  最慢子 process 的估計 = 該 config 最貴的 build 的三者合計（build + 全部 ef 點）× 0.45 + 10 s。
- 理由：逾時會讓 run 變成 `incomplete`，依 010 §3 不能重跑，該格就永久缺讀值；寧可在真的 hang 時多等。
- 文字的 timeout 是外推的（±50%）。文字 TOML commit 之前，若有文字 100k 的實測（例如登記時的 correctness smoke），用同一個成本模型重算：
  只有新的估計 × 5 超過附錄的值時，才依同一規則**上調** child_timeout_s，並在本文件補記。
  不得下調；不得因此改 ef 清單、格點、seed 或 shuffle。timeout 不影響任何讀值的定義。

## 成本估算

依據（三個實作合計；所有實測都是 smoke、不在 lock 下）：

- GloVe 100k M16 每點：4.7 + 0.524·ef 秒（load 9–15 的實測擬合）。
- **負載下上限的換算**：load 低於 9 時量到的實測（100k (48, 400)、1,183,514、文字 pilot）一律乘 1.25，換算成與 M16 實測相同的負載水準。
- 其他 M：每點乘 M / 16（M ≥ 16）；M8 乘 0.66（0.5^0.6，比與 M 成正比保守）。
  依據：100k (48, 400) 的實測每點是 M16 模型的 2.0–2.5 倍，乘 1.25 後為 2.5–3.1 倍。文字用同一組倍數（768 維時距離計算的佔比更高，不取較小的指數）。
- build：
  - GloVe 100k：99 s × (efC / 200) × (M / 16)^0.7；M8 用 0.5^0.5。(48, 400) 的模型為 430 s，等於實測 344 s × 1.25。
  - GloVe 1,183,514：1,824 × 1.25 = 2,280 s。300k：99 × 3^1.27 = 400 s（1.27 = ln(2,280 / 99) / ln 11.84）。
  - 文字：25.4 × 1.25 × (n / 10k)^1.27（100k 為 591 s），efC 與 M 的倍數同 GloVe。
- n 的每點效應：
  - 1,183,514：實測 × 1.25；沒有實測的 ef 用 1.25 × (15.0 + 0.6915·ef)。
  - 300k：100k 的 1.3 倍（1,183,514 換算後是 100k 的 1.67 倍，以 log n 內插到 300k 為 1.26 倍）。
  - 文字：pilot 上包絡 3 + 0.125·ef 秒 × 1.25（load）× 2（2,000 queries）× n 倍數（10k 1.0、30k 1.25、100k 1.5，參考 GloVe 100k → 1,183,514）。100k 的 M16 每點為 11.25 + 0.469·ef 秒。
- 每個子 process 另加 5 s（1,183,514 加 10 s，載入資料）。
- 模型對實測的核對：
  - GloVe 100k (16, 200)、ef 10–256 整個 run：模型 600 s，實測 602 s。
  - GloVe 100k (48, 400)、ef 64–1024 整個 run：模型 3,619 s；實測 2,611 s × 1.25 = 3,264 s（模型高 11%，偏保守）。
  - GloVe 1,183,514 每點的線性模型，ef 64–1024 合計：模型 1,447 s，實測 1,425 s。
  - 文字 pilot、ef 10–256 整個 run：模型 164 s，實測 159 s。

每個 build 的估計（三者合計，負載下上限）：GloVe 100k (16, 200) 977 s、(16, 400) 1,076 s、(16, 100) 1,004 s、(8, 200) 1,236 s、(32, 200) 2,827 s、(48, 200) 3,048 s；
300k 2,138 s；1,183,514 7,525 s。文字 100k (16, 200) 1,532 s、(16, 400) 2,124 s、(16, 100) 2,099 s、(8, 200) 1,615 s、(32, 200) 2,398 s、(48, 200) 3,427 s；10k 520 s；30k 735 s。

| config | build | 子 process | 估計（h，負載下上限） | 最慢子 process（s） | child_timeout_s（倍數） |
| --- | --- | --- | --- | --- | --- |
| glove-formal-1 | 6 | 18 | 1.63 | 443 | 2,400（5.4） |
| glove-formal-2 | 5 | 15 | 1.39 | 488 | 3,000（6.2） |
| glove-formal-3 | 2 | 6 | 1.69 | 1,375 | 7,200（5.2） |
| glove-formal-4 | 2 | 6 | 1.69 | 1,375 | 7,200（5.2） |
| glove-formal-5 | 2 | 6 | 1.13 | 1,375 | 7,200（5.2） |
| glove-formal-6 | 2 | 6 | 1.38 | 1,275 | 6,600（5.2） |
| glove-formal-7 | 1 | 3 | 0.34 | 559 | 3,000（5.4） |
| glove-formal-8 | 1 | 3 | 2.09 | 3,383 | 17,400（5.1） |
| **GloVe 小計** | 21 | 63 | **11.3** | | |
| text-formal-1 | 6 | 18 | 2.55 | 693 | 3,600（5.2） |
| text-formal-2 | 5 | 15 | 2.29 | 959 | 4,800（5.0） |
| text-formal-3 | 2 | 6 | 1.03 | 948 | 4,800（5.1） |
| text-formal-4 | 4 | 12 | 1.97 | 1,545 | 7,800（5.0） |
| **文字小計** | 17 | 51 | **7.9** | | |
| **合計** | 38 | 114 | **19.2** | | |

量測時數以外的 wall-clock（不在上表）：

- load 等待：每個子 process 最多 10 分鐘，114 個最壞多 19 h；機器閒置時接近 0。等超過 10 分鐘，整個 run 中止。
- 每次 formal run 前的 `./scripts/test.sh`（12 次，含編譯，會推高 load，跑完要等 load 降回門檻）。
- 008 §2 的 10 分鐘閒置 load 量測（第一次 formal run 之前一次）。
- `aborted` / `invalid` / `error` 要整份重跑：單次最大損失約 2.6 h（text-formal-1），GloVe 約 2.1 h（glove-formal-8）。
- 文字的估計全部是外推，不確定度約 ±50%。閒置時（load ≤ 4）預期較快，但沒有量過。

## 為什麼

- **一次只改一個因素**：6 格就能分開歸因 M 與 efC，全因子要 12 格。smoke 的 (48, 400) 同時改了 M 與 efC，不放進正式格點。
- **M8**：M 掃描的低端，讓 0.90 / 0.95 的 recall–memory 讀值不被 M16 這個格點下界決定。
- **(16, 200) 當中心與完整 variance 格點**：它同時是兩個掃描的中心與 scaling 固定的 (M, efC)，seed 範圍可以附在三類結論上；也是主要 n 上每個 build 最便宜的格點。
- **GloVe 加 (48, 200) 的 seed variance**：
  - (16, 200) 要涵蓋 0.99 得把 ef 拉到 1536–2048，10 個 build 多約 6–9 h；
  - (48, 200) 是 0.99 的主要來源，多 4 個 seed 約 3.4 h，0.99 的結論就有同一 recall 區間的 5-seed 範圍，0.99 frontier 的主要候選點也是 5 個 seed 的 median。
- **ef 規則**：
  - 梯子間距 ×1.33–1.5，讀值造成的 QPS 低估上限約 25–33%。
  - 細化點只加在 smoke 已顯示目標恰好落在梯子點的地方（GloVe 100k 的 0.90 / 0.95、1,183,514 的 0.95）；那裡 recall 只要差 0.005，讀值就會跳到下一個梯子點。
    依三者合計的成本模型，GloVe 100k 0.90 / 0.95 的 QPS 讀值很可能來自 M16 的格點，正是細化點所在的位置。
  - 上界取悲觀上緣之上：每個要涵蓋的目標，悲觀估計下仍至少有一個達標點；成本過高的目標不追，事先寫明預期「未達到」。
- **1,183,514 加 768、1218、1536**：0.95 不只靠 ef 1024 一個點（smoke 在 1024 只比 0.95 高 0.002–0.006）。
  這一格只有 1 個 seed；若某實作在 1024 的點被 hygiene 標記，還有 1218、1536 可讀，避免只有一個實作「未達到」的不對稱。
- **文字 M16 的上界 512**：文字 100k 從未量過；pilot 的 0.99 約在 ef 80，以 GloVe 的經驗 n 大 10 倍時所需 ef 可能放大 5 倍以上，256 不夠涵蓋悲觀估計。
- **12 個 config，每個 0.3–2.6 h**：一次持有 lock 就能跑完；被 abort 時只損失一個 run。
- **文字格點現在就逐字登記**：格點在任何正式結果之前固定；文字 100k 的 smoke 結果與 GloVe 的正式結果都不能影響文字格點。
- 選擇時只用 recall 涵蓋範圍與三者合計的成本；三個實作用同一組格點。

## 放棄的替代方案

- **主要 n 的每一格都跑 5 個 seed、crossing 附近用四分之一倍頻程的 ef（約 46.8 h、18 個 run）**：
  同一 n 內 build 數一致、讀值的 QPS 低估上限降到 16%，統計上最嚴謹。
  但總時數是本設計的 2.4 倍，198 個子 process 都要通過 load 門檻，以目前的 lock 與負載很可能長期跑不完；也犧牲了 M8 與文字的 efC 100、M32。
  採用其中兩點：細化點（只限 smoke 已顯示的邊界），以及「同一格拆在多個 config 時 ef 清單逐字相同」。
  它建議「跨 n 的 QPS 以只含 (16, 200) 的讀值為主」會改變 010 §5 的主要讀值，只採用為輔助讀值。
- **密集的 M 格點 8–64、efC 在 M16 與 M48 各三個水準、每個 n 都跑 4 條 scaling 骨幹、variance 格點在 (32, 200)（約 41.4 h、15 個 run）**：
  資訊最廣，GloVe 每個 n 都嘗試 0.99。但單一 run 最長 4.7 h（1,183,514 的 (48, 400) 無法再拆），timeout 只有估計值的約 3 倍，
  主要 n 上 10 格單一 seed 讓 frontier 的選擇偏誤最嚴重。
  採用其中兩點：「全部 config 在第一次 formal run 之前 commit、之後必須全部跑完」，以及成本模型附上實測核對。
- **只有 (16, 200) 一個 variance 格點，0.99 的結論附上它在最大 ef（約 0.965）的範圍作為代理**：代理範圍與 0.99 不在同一 recall 區間。
  改為加 (48, 200) 的 seed variance，代理只作 fallback（第 3 點）。
- **把完整 variance 格點移到 (32, 200)、ef 到 768**：同樣涵蓋 0.99，但 10 個 build 比 (16, 200) 約多 5 h（扣掉不再需要的 (48, 200) seeds，仍比本設計多約 1.7 h）；
  (32, 200) 能否在 768 內達到 0.99 不確定，而且 variance 格點不再是 scaling 固定的 (16, 200)。
- **GloVe 300k 與 1,183,514 加 (48, 200) 做 0.99 的 scaling**：約多 5 h，1,183,514 的單一 build run 超過 3.5 h。0.99 的 scaling 改由文字提供。
- **文字 100k 先跑 smoke recall 探測，再依規則決定文字的 ef 清單**：清單可以更精準，但文字格點要等探測之後才固定，多一個自由度。
  改為現在就以較寬的清單逐字登記（M16 到 512，M8 與 efC 100 到 1024）；之後的實測只能用來上調 timeout。
- **單一 build 的 config 輪換 implementations 的起始順序**：每個單一 seed 的格點仍只有一種順序，無法消除重疊；維持 008 §5 的輪換與預設清單順序，照實註明。
- **M < 8、M64、efC 800**：M < 8 的 recall 太低；M64 每點成本約 M16 的 4 倍；efC 800 的 build 成本是 efC 400 的兩倍。格點邊界的影響照實註明。
- **1,183,514 跑多個 seed**：每個 build 約 2.1 h；大 n 用主要 n 的 seed 範圍作代理，照實註明。

## 對實驗的影響

- 達不到的目標照 010 記「未達到」，不以最接近的點代替。事先預期的「未達到」：
  - GloVe 100k 的 M16 三格與 (8, 200) 在 0.99；GloVe 300k、1,183,514 在 0.99；
  - 兩個資料集的 (8, 200) 可能在 0.95（GloVe）或 0.99（文字）；
  - GloVe (32, 200) 與文字 100k (16, 200)、(16, 100) 在 0.99 不確定。
- 看到正式結果之後才增加 ef、M、efC、seed 或 n，必須另寫 decision，並註明是在看到結果之後（010 §2）。
- 主要 n 上 5-seed 與 1-seed 的點混在同一個 frontier：讀值附上參與的 build 數（010 §2）。
  單一 seed 的點在目標附近「是否達標」受 seed 雜訊影響，取 QPS 最大值也會偏向雜訊大的點，照實註明。
- QPS 的 scaling 混入格點數的差異（第 4 點）；固定 (16, 200) 的 build time、peak RSS、index memory 不受影響。
- recall–memory frontier 受格點下界 M8 與各格 ef 上界限制；例如 GloVe 0.99 的最低記憶體讀值只會來自 M32 或 M48。
- 文字的估計全部由 10k pilot 外推，文字 100k 從未量過，實際涵蓋範圍可能與表中不同，照實報告。
- 第一次 formal run 之前還必須：
  - 008 §2 的閒置 load 量測；
  - `report.py` 的四項檢查（第 5 點）；
  - 文字 4 份 TOML 的 commit。
- NEXT 的 B1（index memory 的定義）要在看到正式結果之前決定，B2（電力條件）要在 formal run 之前決定；兩者都與格點無關。

## 補記（2026-10-09，在任何正式結果之前）

- §5 要求的四項 `report.py` 檢查已實作並測試（`tests/test_report.py`）：
  1. 只接受 `complete` / `incomplete` 的 run；
  2. 同一個（實作, n, M, efC, seed, shuffle）出現在多個 run 時拒絕；
  3. 多個 run 合併時，hnswlib / faiss 擴充模組的 SHA-256、套件版本、`rust_hnsw.BUILD_INFO` 必須相同；commit 不同時，
     以 `git diff` 確認 §5 列出的實作與量測程式沒有差異；
  4. 輔助 scaling 讀值（每個 n 只用 (16, 200) 的點，報告中另列一節）。
- 失敗的 run 以 `--excluded <run 目錄>` 只列出狀態與原因（資料不合併）；只有 aborted / invalid / error 的 run 可以被排除。

## 補記（2026-10-09，gate 工具的審查，在任何正式結果之前）

以程式正確性、decision 忠實度、測試鑑別力三個角度審查上一則補記的工具，並修正：

- **事後挑選的漏洞**：上一則補記說「complete 的 run 不得以 --excluded 排除，避免事後挑選」，這個說法不成立：直接不把某個 run 交給報告，
  效果一樣，而且不留痕跡（同一份 config 跑兩次都 complete，可以只交比較有利的那一個；也可以只交部分 config）。改為 formal 報告另外檢查：
  1. 交給報告的 run 的 config（formal 前置檢查記下的 repo 相對路徑）必須恰好是該資料集在 §5 的全部 formal config，每份一個 run；
     缺少、重複或不在清單中都拒絕（清單登記在 `benchmarks/report.py` 的 `FORMAL_CONFIGS`，與 §5 的表相同）。
  2. 掃描 `results/runs/` 與所給 run 的上層目錄：同一資料集、同一份 formal config、但沒有交給報告的 formal run，
     complete / incomplete 的一律拒絕（010 §3：已完成的 formal run 不得以重跑取代或略過）；aborted / invalid / error 的必須以 `--excluded` 列出。
  3. 報告記錄每個 run 的 config 路徑與 git blob。
  smoke 報告不做這些檢查。
- **資料集身分**：合併的 run 之間，資料集紀錄（name、kind、n_train、n_queries、dim、metric、各檔案的 SHA-256）與共同 n 的 ground truth 來源
  必須完全相同。原本只比對名稱，同名資料集在兩次 formal run 之間重新準備時會被合併。
- **輔助 scaling 讀值**：「排除標記點是否改變讀值」只比較 QPS / P50 / P95 的數值與所在的點（§4 的定義）；改變時另列包含被標記點的讀值（010 §3）。

## 附錄：文字的 formal config（逐字登記）

與 `simplewiki-embeddinggemma2-100k` 的 manifest 同一個 commit 加入 `benchmarks/configs/`，內容必須與以下完全相同（唯一例外：依第 6 點上調的 `child_timeout_s`）。

`text-formal-1-m16-order.toml`：

```toml
# 正式 run（decision 013）：文字 n = 100,000、(M, efC) = (16, 200) 的完整 variance 格點（010 §4），第一部分。
# seed 1 原始順序加 shuffles 101–105（搭配 seed 1）；seeds 2–5 在 text-formal-2-m16-seeds-efc400.toml，ef 清單必須完全相同（013 §2）。
# 本檔與 simplewiki-embeddinggemma2-100k 的 manifest 同一個 commit 加入，內容與 013 附錄逐字相同（child_timeout_s 只能依 013 §6 上調）。
schema_version = 1
dataset = "simplewiki-embeddinggemma2-100k"
implementations = ["rust-hnsw", "hnswlib", "faiss-hnsw"]
ef_search = [24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
child_timeout_s = 3600

[[sweep]]
n = [100000]
m = [16]
ef_construction = [200]
seeds = [1]
shuffles = [101, 102, 103, 104, 105]
```

`text-formal-2-m16-seeds-efc400.toml`：

```toml
# 正式 run（decision 013）：文字 n = 100,000。
# 第一個 sweep：(16, 200) 的 seeds 2–5（原始順序），與 text-formal-1-m16-order.toml 合起來是完整 variance 格點；shuffles 必須為空。
# 第二個 sweep：efC 掃描的 (16, 400)，seed 1。兩者共用 (16, 200) 的 ef 清單（013 §2）。
# 本檔與 simplewiki-embeddinggemma2-100k 的 manifest 同一個 commit 加入，內容與 013 附錄逐字相同（child_timeout_s 只能依 013 §6 上調）。
schema_version = 1
dataset = "simplewiki-embeddinggemma2-100k"
implementations = ["rust-hnsw", "hnswlib", "faiss-hnsw"]
ef_search = [24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
child_timeout_s = 4800

[[sweep]]
n = [100000]
m = [16]
ef_construction = [200]
seeds = [2, 3, 4, 5]
shuffles = []

[[sweep]]
n = [100000]
m = [16]
ef_construction = [400]
seeds = [1]
shuffles = []
```

`text-formal-3-m8-efc100.toml`：

```toml
# 正式 run（decision 013）：文字 n = 100,000。
# 第一個 sweep：M 掃描的低端 (8, 200)，seed 1；ef 上界 1024 是成本上限。
# 第二個 sweep：efC 掃描的 (16, 100)，seed 1。兩者共用一份 ef 清單（013 §2）。
# 本檔與 simplewiki-embeddinggemma2-100k 的 manifest 同一個 commit 加入，內容與 013 附錄逐字相同（child_timeout_s 只能依 013 §6 上調）。
schema_version = 1
dataset = "simplewiki-embeddinggemma2-100k"
implementations = ["rust-hnsw", "hnswlib", "faiss-hnsw"]
ef_search = [24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
child_timeout_s = 4800

[[sweep]]
n = [100000]
m = [8]
ef_construction = [200]
seeds = [1]
shuffles = []

[[sweep]]
n = [100000]
m = [16]
ef_construction = [100]
seeds = [1]
shuffles = []
```

`text-formal-4-highm-scaling.toml`：

```toml
# 正式 run（decision 013）：文字。
# 第一個 sweep：n = 100,000 的 M 掃描 (32, 200)、(48, 200)，seed 1。
# 第二個 sweep：scaling 的 n = 10,000、30,000，固定 (16, 200)，seed 1。全部共用一份 ef 清單（013 §2）。
# 本檔與 simplewiki-embeddinggemma2-100k 的 manifest 同一個 commit 加入，內容與 013 附錄逐字相同（child_timeout_s 只能依 013 §6 上調）。
schema_version = 1
dataset = "simplewiki-embeddinggemma2-100k"
implementations = ["rust-hnsw", "hnswlib", "faiss-hnsw"]
ef_search = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
child_timeout_s = 7800

[[sweep]]
n = [100000]
m = [32, 48]
ef_construction = [200]
seeds = [1]
shuffles = []

[[sweep]]
n = [10000, 30000]
m = [16]
ef_construction = [200]
seeds = [1]
shuffles = []
```
