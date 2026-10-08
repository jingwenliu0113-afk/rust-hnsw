# 011：資料集（glove-100-angular 與 EmbeddingGemma 2 文字 workload）

日期：2026-10-08
狀態：採用（在任何正式結果之前；具體的檔案 SHA-256、筆數與 token 統計記在各 manifest 中）

## 背景

SPEC §5 要求「一組標準 ANN 真實資料集」與「一組真實文本經現代 embedding model 得到的向量」。
使用者在 2026-10-08 決定：

- 標準 ANN dataset：`glove-100-angular`。
- 真實文字 embedding workload：公開文字 corpus + `google/embeddinggemma-2`。
  - 固定用完整 768 維 embedding，L2 normalize；不用 fp16（使用 bf16 或 fp32）。
  - 固定 model revision、runtime version、preprocessing、seed。
  - base / query 必須 deterministic 而且 disjoint，query 不得同時存在 index（避免 self-match leakage）。
  - 以 exact search 重新產生 frozen ground truth。
  - 先做小型 pilot，不直接生成 1M embeddings；Core 暫時不做 128 / 256 / 512 維的 Matryoshka truncation。
  - 原始資料、完整 embeddings、ground truth 不進 Git，只保留程式、manifest、checksum、split metadata 與 decision log。
  - 正式 benchmark 仍需 Mac heavy lock；lock 被占用時先做資料準備、embedding 產生與 correctness 工作。

## 決策

### 1. 共通規則

- **檔案位置**：所有資料檔放在 `datasets/<名稱>/`（`.gitignore` 已排除 `/datasets/`）。
- **manifest**：每個資料集一份 `benchmarks/manifests/<名稱>.json`，進 Git。內容包括：
  - 來源（URL、revision、原始檔 SHA-256、授權）；
  - 處理後每個檔案的 SHA-256、shape、dtype；
  - 子集規則、split metadata、frozen ground truth 的檔案與 SHA-256；
  - 準備環境：git commit 與 `git status` 的逐行清單、Python 版本、套件版本、計算 frozen ground truth 的 `rust_hnsw.BUILD_INFO`
    （文字 workload 另記 Python 的 Unicode 資料庫版本）。
  `benchmarks/datasets.py` 從 manifest 登記資料集；formal run 比對 manifest 中的 SHA-256（008 §1.7）。
- **向量格式**：`train.npy`、`queries.npy` 為 float32、C-contiguous 的 unit vectors（以 f64 計算 norm 再正規化，與 `prepare_vectors` 相同）。
  零向量或非有限值一律讓準備失敗，不默默丟棄；若真的出現，另寫 decision 說明處理方式。
- **子集**：runner 一律取 train 的前 n 列（008 §4），所以 train 的列順序必須本身就是隨機順序（各資料集的說明見下）。
- **frozen ground truth**（修訂 008 §4 的「每次 run 都重新計算」）：
  - 準備資料時，以與 runner 相同的 `ExactIndex`（001 的 f64 與 (distance, id) tie-break）對每個登記的 n 計算 top-100，存成 `gt_n<n>.npy`（int64）。
  - runner 遇到有 frozen ground truth 的 n：`train.npy`、`queries.npy` 與 gt 檔的 SHA-256 都必須等於 manifest（smoke 與 formal 皆然），
    並以 `ExactIndex` 對固定的前 100 個 query 重新計算 top-10，必須完全相同，才使用 frozen 的檔案；不符就拒絕執行。
    gt 是否 exact 取決於 train 與 queries，所以只比對 gt 檔本身不夠（第 100 個 query 之後的改動不會被重新計算發現）。
    沒有登記的 n 照舊在 run 中計算。run.json 記錄每個 n 的 ground truth 來源。
  - top-100 的前 10 欄等於 k = 10 的答案（tie-break 是全序）。

### 2. glove-100-angular

- **來源**：ann-benchmarks 的 `http://ann-benchmarks.com/glove-100-angular.hdf5`（463 MB；repo 為 MIT）。
  向量來自 Stanford GloVe 的 `glove.twitter.27B` 100 維（Public Domain Dedication and License v1.0）。
- **內容**：train 1,183,514 × 100、test（作為 queries）10,000 × 100、`neighbors`（ann-benchmarks 的 top-100，angular）。
- **完整性**：網站沒有公布 checksum，所以第一次下載時記錄 SHA-256 與檔案大小，寫進 manifest；之後的下載必須相符。
- **子集**：ann-benchmarks 以 `sklearn.model_selection.train_test_split(X, test_size=10000, random_state=1)` 切分（已讀原始碼確認），
  預設會打亂順序，所以 train 的原始列順序就是隨機順序，「前 n 列」是巢狀的隨機子集。保留原始順序，不另外重排，
  這樣 ann-benchmarks 自帶的 `neighbors` 可以直接在 n = 全部時比對（008 §4 的 sanity check）。
- **登記的 n**（scaling with N 的格點，frozen ground truth）：100,000、300,000、1,183,514。
- 重複向量的比例由 runner 依 003 / 008 §4 計算並記錄（GloVe 有重複的 token 向量，屆時照實呈現）。

### 3. 真實文字 workload：Simple English Wikipedia + EmbeddingGemma 2

#### 3.1 corpus

- **來源**：Hugging Face `wikimedia/wikipedia`，config `20231101.simple`，dataset revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`，
  檔案 `20231101.simple/train-00000-of-00001.parquet`（156,885,218 bytes，SHA-256 `31bded16768a47c286becd292079122f5d7d4397a17b87d4250a00ccd581e6f0`）。
  授權 CC BY-SA 3.0 與 GFDL；不需要登入。
- **選擇理由**：公開、授權清楚、單一檔案、內容是一般知識的自然語言；之後要擴大規模時可以換成同一 dataset 的英文版（41 個 shard），前處理規則不變。

#### 3.2 前處理（passage）

對每篇文章（欄位 `id`、`title`、`text`）：

1. `text` 以換行切成段落；每段做 Unicode NFC 正規化，連續空白（含 tab）合併成一個空格，去掉頭尾空白。
2. 長於 1,000 個字元的段落，截在 1,000 字元以內的最後一個空白處（不切斷單字）。
3. 截斷之後只保留長度 ≥ 200 個字元的段落（排除標題、清單項目、短句；先截斷再過濾，所以截斷點落在 200 之前的段落也會被丟棄）。
4. passage id 為 `<文章 id>:<該文章中保留段落的序號>`。
5. **完全相同的 passage 文字**只保留一份：保留文章排序（見 3.3）最前面的那一份。

#### 3.3 base / query 切分（deterministic、disjoint）

- 每篇文章的排序鍵為 `sha256("rust-hnsw-text-v1:" + 文章 id)` 的 16 進位字串，依此排序（等同固定 seed 的隨機排列，而且不依賴 numpy 版本）。
- **query**：依排序取前 Q 篇「至少有一個 passage」的文章，每篇只取第一個 passage（第 5 步去重之後的第一個）。
- **base**：其餘文章的所有 passage，依文章排序、文章內依段落序號排列；取前 N 個。
- **以文章為單位切分**：query 文章的任何 passage 都不會進入 base，同一篇文章的相近段落不會同時出現在兩邊；
  第 5 步的去重保證兩邊沒有完全相同的文字。準備時另外斷言 base 與 query 的 passage id、文章 id、文字三者都沒有交集。
- base 的順序來自雜湊排序，所以「前 n 列」是以文章為單位的隨機子集，可以直接作巢狀子集。
- split metadata（規則、鍵的前綴、Q、N、base / query 的 passage id 清單的 SHA-256）寫進 manifest；完整的 id 清單與文字放在 `datasets/`。

#### 3.4 embedding

- **模型**：`google/embeddinggemma-2`，revision `914f7f89142e33e77833254d9c9b90c3cef7303b`，Apache 2.0、不需要登入。
  `model.safetensors` 為 1,488,915,288 bytes（LFS SHA-256 `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`），下載後比對。
  模型是多模態的（文字 backbone 270M，另有 vision / audio encoder），只使用文字輸入。
- **輸入格式**：model card 建議的 document 格式 `title: {文章標題} | text: {passage}`，base 與 query **都用這個格式**。
  這是「以 held-out 文件查詢相近文件」的 workload（與 ann-benchmarks 以 held-out 資料當 query 的做法相同），不是自然語言問句對文件的檢索；
  後者需要另一組 query 來源，不在 Core 範圍內。
- **輸出**：完整 768 維（不做 Matryoshka truncation）。sentence-transformers 的 pipeline 本身含 Normalize；
  另外再以 f64 計算 norm 並正規化後存成 float32（與 `prepare_vectors` 相同）。
- **精度與 runtime**：fp32、CPU（PyTorch CPU backend），`torch.set_num_threads(8)`、`torch.manual_seed(0)`、`torch.use_deterministic_algorithms(True)`，
  **batch size 1**（每段文字單獨 forward、不 padding；理由見文末補記）。
  - 不用 fp16（model card 明示會產生 NaN 或劣化的 embedding）。
  - 選 fp32 而不是 bf16：CPU 上 fp32 的結果最容易重現，pilot 規模下速度可以接受。
  - 選 CPU 而不是 MPS：避免 GPU kernel 的非決定性；之後若需要加速再另寫 decision。
- **套件版本**：獨立的 `.venv-embed`（不進 Git），`requirements-embed.txt` pin 死：`torch==2.14.1`、`transformers==5.19.0`、`sentence-transformers==6.1.0`
  （model 的 config 要求 sentence-transformers ≥ 6.1.0），以及會影響輸出的 `tokenizers==0.23.2`、`safetensors==0.8.0`、`huggingface-hub==1.33.0`。
  `embed_text` 開始前比對已安裝版本與 pin，不符就拒絕。與 benchmark 用的 `.venv` 分開，避免 numpy 等套件被連帶改版而影響 007 的實測語意。
- **重現性檢查**：準備時把每個 job 的前 32 個 passage 單獨重新 encode 一次，結果必須逐位元相同；全部向量必須有限、norm 與 1 的差 ≤ 1e-5。
- 每次產生都記錄：模型 revision 與檔案 SHA-256、所有 pin 的套件版本、dtype、device、thread 數、batch size、padding、最長 token 數、耗時。

#### 3.5 pilot

- **規模**：base N = 10,000、query Q = 1,000。目的：驗證前處理、切分、embedding、frozen ground truth 與 runner 的整條流程。
- **登記的 n**：10,000（frozen ground truth）。
- pilot 的數字只作流程驗證；正式的文字 workload 規模（N 與登記的 n）在 pilot 之後另寫 decision 決定。

## 為什麼

- glove-100-angular 是 angular（cosine）最常用的標準資料集之一，n 夠大，可以做 scaling；保留 ann-benchmarks 的切分，結果可以和公開數字對照。
- frozen ground truth：大 n 的 exact search 很貴（1.18M × 10k），每次 run 都重算浪費時間；以 SHA-256 加上抽樣重算確保它仍是同一個 exact 答案。
- 以文章為單位切分、加上完全相同文字的去重，比單純隨機抽 passage 更能避免 self-match 與近似重複的洩漏。
- 雜湊排序不依賴任何亂數產生器的實作，換 numpy 版本也能重現。

## 放棄的替代方案

- **dbpedia OpenAI embeddings（預先算好的向量）**：使用者改為以開源模型自行產生，可以固定 revision 與前處理。
- **英文 Wikipedia 全量**：pilot 不需要；擴大規模時再評估。
- **自然語言問句當 query（例如 QA 資料集）**：需要另一個資料來源與不同的 prompt，Core 先固定一種 workload。
- **把 GloVe 重新打亂**：ann-benchmarks 的切分本身已是隨機順序；重排會讓自帶的 `neighbors` 無法直接比對。
- **bf16 或 MPS**：可能更快，但 pilot 規模下 fp32 + CPU 的重現性最直接。

## 對實驗的影響

- formal run 只接受 manifest 已登記、SHA-256 相符的資料集。
- 文字 workload 的結論只代表「EmbeddingGemma 2 的 768 維 document embedding、held-out 文件查詢」這個分布。
- Matryoshka truncation 與 query / document 不對稱的 workload 不在 Core。
- 008 §4 的 ground truth 規則改為「有 frozen 檔案就驗證後使用，否則照舊計算」。

## 補記（2026-10-08，資料準備程式的審查，在任何資料集結果之前）

以 data correctness 與 integration 兩個角度審查資料準備程式，共 19 個 findings，每個都由另一位驗證者重現或反駁。採用 8 個（兩個角度有兩組重疊，實際是 6 項修正，多數縮小了修法），駁回 11 個。

**embedding 改用 batch size 1（原為 32）**：
- 原本的重現性檢查在真實資料上必然失敗，第一次文字 pilot 也因此在 encode 完 base 之後中止。
  原因：sentence-transformers 先依長度把整個 job 排序再切批並 padding。同一段文字有無 padding，輸出差約 2e-7（最多 3.3e-7）。
  所以「前 32 筆單獨重新 encode」與「完整 encode 中的前 32 筆」批次組成不同，位元必然不同。
  以完全相同的組成重新 encode 則逐位元相同。
- 改用 batch size 1：每段文字單獨 forward、不 padding，向量只取決於文字與固定的 runtime，與批次組成、job 大小無關。
  原本的檢查不用改就有意義；日後較大 N 的 `train.npy` 前 n 列也會與 pilot 逐位元相同。
  代價是速度：審查者以 96 段實測約慢 1.7 倍，pilot 規模可以接受。
- 放棄的替代方案：自己依長度切批，再以相同組成重新 encode 固定幾批。可以保留 batch 32 的速度，但向量會隨 job 組成改變，
  不同 N 的檔案之間無法逐位元對應。只有到較大 N、速度真的成為問題時，才另寫 decision 改用這個做法。
- 測試：40 段長度不一的文字，前 32 筆單獨 encode、反轉順序 encode，都必須與完整 encode 逐位元相同（batch 32 下會失敗，已確認）。

**其他採用的修正**：
- frozen ground truth：使用前另外比對 `train.npy`、`queries.npy` 的 SHA-256（見 §1）。審查者實測：只改 queries 第 100 列之後，
  smoke run 會沿用過期的 gt，status 仍是 complete，exact search 的 recall@10 只有 0.8367。另補兩個 runner 測試：
  gt 的 SHA 相符但內容不是 exact 答案時必須拒絕；資料集自帶的 neighbors 只在 n = 全部 train 時比對。
- 前處理改為先截斷再過濾（§3.2）：原本前段很長沒有空白的段落，截斷後可能只剩幾個字元。段落 ≤ 1,000 字元時結果不變。
- 套件版本：pin tokenizers、safetensors、huggingface-hub，並在執行時比對（§3.4）。
- manifest 的準備環境：以 `git status` 的逐行清單取代 `-dirty` 字尾，另記 Python 版本、`rust_hnsw.BUILD_INFO`，文字 workload 另記 Unicode 資料庫版本（§1）。

**駁回的 findings**（理由摘要）：
- 去重時移除 Unicode 格式字元（零寬空格等）：§3.2 定義的去重就是「完全相同的文字」，實作照做。移除 ZWJ / ZWNJ 會改變送進模型的文字；
  「query 到最近 base 的距離分位數」是新的資料品質指標，屬於擴張範圍。
- 斷言文章 id 唯一：Wikipedia page id 本來就唯一，輸入是 SHA-256 固定的單一檔案；只在 docstring 註明這個假設。
- 斷言實際參數 dtype、記錄儲存 dtype：實測參數全部是 float32（權重以 bf16 儲存、以 fp32 載入），程式中沒有其他決定 dtype 的路徑。
- max_seq_length 的檢查永遠不會觸發：這個模型沒有設定上限，tokenize 時也不會截斷；實際 token 數已記錄在 meta。
- manifest schema 驗證、atomic 寫入、目錄自動建立、重跑時拒絕覆寫、FileNotFoundError 的訊息格式：都是大聲失敗或暫時狀態，
  不會造成錯誤的 recall 或不可重現。
- config 與 manifest 必須在同一個 commit：作為流程規則照做，不寫成測試。
