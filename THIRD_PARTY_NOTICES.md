# 第三方內容與授權（Third-party notices）

本專案作者有權授權的**原創程式碼**以 [BSD-3-Clause](LICENSE) 授權。下列第三方內容**不在**該授權範圍內，保留其原有授權。

The original code in this repository that its author has the right to license is licensed under
[BSD-3-Clause](LICENSE). The third-party content below is not covered by that license and keeps its own license.
The repository contains no third-party code, dataset content, model weights or embeddings; datasets, the embedding
model and the baseline libraries are downloaded at run time and keep their own licenses (tables below).

查證日期：2026-10-19。

- repo 內的內容：以 tracked 檔與 Git 全歷史查證。
- 套件授權：取自本機安裝的 metadata（Python 套件的 `*.dist-info`；Rust crate 在 crates.io 原始碼中的 `Cargo.toml`）。
- 資料集與模型：以原始來源（下表的連結）為準，並與 `benchmarks/manifests/` 及 [decision 011](docs/decisions/011-datasets.md) 的登記核對。

以下是摘要，不是法律意見；各項以原始來源的授權條款為準。

## repo 內含的第三方內容

無。

- **程式碼**：本 repo 不含第三方程式碼，沒有 vendored 的程式庫，也沒有 submodule。
  - HNSW 依 Malkov & Yashunin 的論文以 Rust 自行實作。演算法細節刻意對齊 hnswlib 的行為（[decision 004](docs/decisions/004-hnsw-core-algorithm.md)），但沒有複製它的程式碼。
  - SQ8 codec 依 FAISS `QT_8bit` + `RS_minmax`（`rangestat_arg = 0`）的行為重新實作（[decision 019](docs/decisions/019-advanced-c-vector-quantization.md) §4）。
  - CRC-32（IEEE）與 SplitMix64 是公開的標準演算法，自行實作（`crates/hnsw/src/crc32.rs`、`crates/hnsw/src/rng.rs`）。
  - hnswlib 與 FAISS 只在執行時透過它們的 Python 套件被呼叫（`benchmarks/adapters.py`），不在 repo 內。
    decision 019 所說的「移植」，是在執行時把 FAISS 的 graph 從 `IndexHNSWFlat` 搬到 `IndexHNSWSQ`，不是移植程式碼。
  - [decision 007](docs/decisions/007-third-party-adapters.md) 為了說明實測語意，引用 hnswlib 原始碼的檔名、行號與短識別字；這些不是程式碼摘錄。
- **資料**：`benchmarks/manifests/*.json` 只記錄來源 URL 或 dataset revision、SHA-256、筆數、切分規則，以及 passage id 清單的 SHA-256。不含向量、文字或標題。
- **圖與數字**：`docs/figures/` 的圖，以及 README 與 decision 中的數字，都是本專案在本機的量測結果（圖的來源見 [docs/figures/README.md](docs/figures/README.md)），不含資料集的向量或文字。
- **測試**：測試用的文字與向量由本專案撰寫，或以固定 seed 產生（synthetic）。

## 執行時下載、不在 repo 內的內容

### 資料集與 embedding 模型

| 內容 | 取得方式 | 授權（以原始來源為準） |
|---|---|---|
| `glove-100-angular`：ann-benchmarks 的 HDF5（`http://ann-benchmarks.com/glove-100-angular.hdf5`，485,413,888 bytes） | `.venv/bin/python -m benchmarks.prepare_glove --download` 下載到 `datasets/`（不進 Git）。網站沒有公布 checksum，第一次下載時把 SHA-256 記進 manifest，之後的下載必須相符 | 向量：Stanford [GloVe](https://nlp.stanford.edu/projects/glove/) `glove.twitter.27B` 100 維，[PDDL 1.0](https://opendatacommons.org/licenses/pddl/1-0/)。train / test 切分與 `neighbors`：[ann-benchmarks](https://github.com/erikbern/ann-benchmarks)，MIT |
| Simple English Wikipedia：Hugging Face [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)，config `20231101.simple`，revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`，檔案 `20231101.simple/train-00000-of-00001.parquet` | 事先把這個 revision 的 parquet 下載到 `datasets/simplewiki/raw/`（不需要登入；repo 沒有下載程式）。`benchmarks.prepare_text` 比對 SHA-256 之後才切出 passage | Wikipedia 的文字以 CC BY-SA 與 GFDL 授權。manifest 依 HF dataset card 記為 CC BY-SA 3.0 與 GFDL；[Wikipedia:Copyrights](https://en.wikipedia.org/wiki/Wikipedia:Copyrights) 目前寫的是 CC BY-SA 4.0 與 GFDL |
| [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)，revision `914f7f89142e33e77833254d9c9b90c3cef7303b`（`model.safetensors` 的 SHA-256 記在 `benchmarks/embed_text.py` 與文字資料集的 manifest） | `benchmarks.embed_text`（在獨立的 `.venv-embed` 執行）經 sentence-transformers / huggingface-hub 下載到本機的 Hugging Face cache（不需要登入），並比對 `model.safetensors` 的 SHA-256 | Apache-2.0（固定 revision 的 model card；decision 011 §3.4）。model card 要求使用時遵守 [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) |

- Wikipedia 的 passage 文字與產生的 embedding 只存在本機的 `datasets/`，從未進入 Git 歷史。本 repo 沒有散布任何 Wikipedia 文字，因此不涉及 CC BY-SA 的署名與相同方式分享。若日後要公開這些檔案，必須另外處理這兩項要求。
- EmbeddingGemma 2 的權重與產生的 embedding 都不在 repo 中，本專案也不散布它們。

### 比較用的 baseline 與其他套件

| 內容 | 取得方式 | 授權（以原始來源為準） |
|---|---|---|
| hnswlib 0.8.0（baseline） | `requirements-dev.txt`（PyPI 只有 sdist，在本機編譯） | Apache-2.0（本機由 sdist 建出的套件 dist-info 內附的 LICENSE） |
| faiss-cpu 1.15.1（FAISS HNSW 與 SQ8 的 baseline） | `requirements-dev.txt`（PyPI） | MIT。wheel 另附第三方授權：macOS wheel 內的 LLVM OpenMP runtime（libomp）為 Apache-2.0 WITH LLVM-exception；Linux / Windows wheel 內的 OpenBLAS 為 BSD-3-Clause（見 wheel 的 THIRD_PARTY_NOTICES） |
| 開發與報告用的 Python 套件 | `requirements-dev.txt` | maturin 1.15.0（MIT OR Apache-2.0）、numpy 2.5.3（BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0）、pytest 9.1.1（MIT）、matplotlib 3.11.2（Matplotlib License，PSF 型）、h5py 3.16.0（BSD-3-Clause）、pyarrow 25.0.1（Apache-2.0） |
| 產生 embedding 用的 Python 套件 | `requirements-embed.txt`（安裝在 `.venv-embed`） | torch 2.14.1（Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT）、torchvision 0.29.1（BSD）、transformers 5.19.0、sentence-transformers 6.1.0、tokenizers 0.23.2、safetensors 0.8.0、huggingface-hub 1.33.0（都是 Apache-2.0）、pillow 12.3.0（MIT-CMU） |
| Rust crates | `Cargo.lock`（crates.io） | `crates/hnsw` 沒有任何相依。binding（`python/`）相依 pyo3 0.29.3 與 numpy（rust-numpy）0.29.0，連同傳遞依賴共 24 個外部 crate：多數為 MIT OR Apache-2.0；rust-numpy 為 BSD-2-Clause；target-lexicon 為 Apache-2.0 WITH LLVM-exception；unicode-ident 為 (MIT OR Apache-2.0) AND Unicode-3.0 |

- 兩份 requirements 的傳遞依賴中，沒有 GPL、AGPL 或限非商用的授權。只有 certifi（MPL-2.0）與 tqdm（MPL-2.0 AND MIT）屬於檔案層級的弱 copyleft；本專案沒有修改或散布它們。
- 本 repo 只散布原始碼：`Cargo.toml` 設定 `publish = false`，沒有 wheel、binary 或 release。
  若日後散布編譯好的 binding（wheel 或 `.so`），需要附上靜態連結的 Rust crate 的授權全文（MIT / Apache-2.0、rust-numpy 的 BSD-2-Clause、unicode-ident 的 Unicode-3.0 等），並確認 wheel 內含本專案的 LICENSE。
  `python/pyproject.toml` 位於子目錄，而 PEP 639 不允許 `license-files` 用 `..` 指向根目錄的 LICENSE；目前沒有驗證 maturin 建出的 wheel 是否會帶上它。

## 參考文獻

學術慣例的引用，不是授權要求。

- Yu. A. Malkov, D. A. Yashunin. *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE TPAMI 42(4), 2020.（[arXiv:1603.09320](https://arxiv.org/abs/1603.09320)）
- J. Pennington, R. Socher, C. D. Manning. *GloVe: Global Vectors for Word Representation.* EMNLP 2014.
- M. Aumüller, E. Bernhardsson, A. Faithfull. *ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms.* Information Systems 87, 2020.
- hnswlib：<https://github.com/nmslib/hnswlib>；FAISS：<https://github.com/facebookresearch/faiss>
