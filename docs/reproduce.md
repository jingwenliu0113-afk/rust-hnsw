# 重現指南

> **公開的 repo 無法重產 README 的正式數字。**
>
> - 正式數字來自 `results/runs/` 的原始 run（約 1.2 GB），它不進版控，也不公開；由它產生的報告（`results/reports/`）同樣不進版控。
>   README、[results.md](results.md) 與 [architecture.md](architecture.md) 引用的正式數字都來自這些報告，而報告只存在專案作者的機器上；results.md 照原值轉錄了報告的關鍵表格。
> - formal 的計時與記憶體數字與機器有關：一台 Apple M4 Pro、特定的電力條件，以及當時的背景負載。在其他機器上照同樣的步驟重跑，得到的是另一組數字，不是 README 的數字。
> - 其他 Apple Silicon 的 Mac 可以用 **smoke 模式**驗證 correctness（recall 隨 efSearch 的變化、輸出格式、ids 的逐位元核對），也可以觀察相對趨勢。
>   smoke 的計時不在 heavy lock 下量，也不排除背景負載，不得寫成任何結論，也不能與 README 的數字對照。

各層由上往下越完整、也越花時間。時間是專案作者在 Apple M4 Pro（48 GB）上的紀錄或事先的估計，來源寫在括號中。

| 層級 | 需要 | 時間 | 能檢查什麼 |
| --- | --- | --- | --- |
| **A. 測試與 Demo** | macOS（Apple Silicon）、Rust、uv、Python 3.13 | 沒有正式記錄；019 §12 把 2–3 次 `./scripts/test.sh` 估為約 0.3 h | correctness 測試、binding、Demo、runner 本身 |
| **B. 資料準備** | 網路（ann-benchmarks、Hugging Face，都是匿名下載）、`.venv-embed` | GloVe 約 466 s；文字 100k 的 embedding 約 10,333 s 加 170 s（NEXT.md、012 補記） | 資料檔與 manifest 的 SHA-256 |
| **C. 真實資料的 correctness smoke** | B 的資料集 | GloVe 100k 約 602 s、文字 pilot 約 159 s（013 的成本模型核對，都是 smoke） | recall 隨 efSearch 沒有系統性惡化、結果格式 |
| **D. Core formal** | 只限 macOS，Apple 原廠 70W 或 140W 變壓器、heavy lock、閒置的機器 | 12 份 config，負載下的上限估計 19.2 h（013） | 正式 run；數字與機器有關 |
| **E. Gate profiling** | D 的 formal run | 每個實作 search 30 s、build 300 s 的取樣（015 §2） | 時間花在哪裡 |
| **F. Advanced C** | D 的 Core run（019 §2 登記的 4 個）、與 D 相同的條件 | lock 合計的中央估計約 5.8 h、悲觀估計約 10 h（019 §12） | C 的判定表 |
| **G. 報告與 README 圖** | `results/runs/`（不公開） | 沒有記錄 | 報告與圖的逐位元重產 |

## 環境版本

只列非個人化的資訊；來源是 `requirements-dev.txt`、`requirements-embed.txt`、`Cargo.lock`，以及正式 run 的 run.json metadata（抽查 GloVe 與文字的 Core run、Advanced C 的 run，三者相同）。

| 項目 | 版本 | 來源 |
| --- | --- | --- |
| OS | macOS 27.0.1（build 26A434），arm64 | run.json |
| 機器 | Apple M4 Pro（8 個 P core、4 個 E core），48 GB RAM | run.json |
| Rust | rustc 1.99.0（b940084d7 2026-09-28）；edition 2024 至少需要 1.85 | `rust_hnsw.BUILD_INFO`（run.json）、`Cargo.toml` |
| PyO3 / rust-numpy | 0.29.3 / 0.29.0 | `Cargo.lock` |
| Python | CPython 3.13.9（`requires-python >= 3.13`）。run.json 記錄的是 Anaconda 打包的版本（`packaged by Anaconda, Inc.`）；以本文件的 `uv venv --python 3.13` 建立的 interpreter 是另一個 build，沒有驗證是否影響計時 | run.json、`python/pyproject.toml` |
| maturin | 1.15.0 | `requirements-dev.txt`、run.json |
| numpy | 2.5.3 | 同上 |
| hnswlib | 0.8.0：PyPI 只有 sdist，由 uv 在本機以 clang 21 編譯；`-O3 -march=native` 依其 `setup.py` 推論，不是觀察到的 | `requirements-dev.txt`、run.json、007 |
| faiss-cpu | 1.15.1：官方 wheel，`get_compile_options()` 為 `OPTIMIZE DD ARM_NEON MAC_METAL` | 同上 |
| 其他（`.venv`） | pytest 9.1.1、matplotlib 3.11.2、h5py 3.16.0、pyarrow 25.0.1 | `requirements-dev.txt` |
| `.venv-embed` | torch 2.14.1、torchvision 0.29.1、transformers 5.19.0、sentence-transformers 6.1.0、tokenizers 0.23.2、safetensors 0.8.0、huggingface-hub 1.33.0、numpy 2.5.3、pyarrow 25.0.1、pillow 12.3.0、pytest 9.1.1 | `requirements-embed.txt` |
| embedding 模型 | `google/embeddinggemma-2`，revision `914f7f89142e33e77833254d9c9b90c3cef7303b` | 011 §3.4 |
| C 編譯器與連結器 | Apple clang 21（Xcode Command Line Tools） | 007、NEXT.md |
| uv | run.json 沒有記錄（撰寫本文件時本機為 0.11.8，只供參考） | — |

## A. 測試與 Demo

### 前置

- macOS（Apple Silicon）。測試與 runner 只在 macOS arm64 上執行過；其他平台沒有驗證。
- Xcode Command Line Tools（`xcode-select --install`）：提供 git、連結器，以及 profiling 用的 `otool`。
- Rust（rustup，含 rustfmt 與 clippy）。edition 2024 至少需要 Rust 1.85；程式也用到較新的標準函式庫 API（例如 `is_multiple_of`），
  真正的最低版本沒有測過。實測版本是 1.99.0。
- [uv](https://docs.astral.sh/uv/)：建立 venv；`scripts/test.sh` 也以 `maturin develop --uv` 安裝 binding。
- Python 3.13（`python/pyproject.toml` 要求 ≥ 3.13；uv 可以自行下載）。

### 最小的 Demo（只需要 numpy 與 maturin）

在 repo 根目錄執行：

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python maturin==1.15.0 numpy==2.5.3
VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --uv -m python/Cargo.toml
.venv/bin/python examples/quickstart.py
```

- 根目錄的 `Cargo.toml` 是 virtual workspace，`pyproject.toml` 在 `python/` 底下，所以要以 `-m python/Cargo.toml` 指定；
  `VIRTUAL_ENV` 決定 binding 安裝到哪個 venv。這個指令與 `scripts/test.sh` 中的相同（同樣設定 `VIRTUAL_ENV` 與 `PYO3_PYTHON`）。
- `examples/quickstart.py` 以隨機產生的 synthetic 向量建 index，印出建置的筆數、`ef_search` 16 / 64 / 256 的 Recall@10，以及存檔再讀回之後結果是否完全相同。
  **這些數字不是研究結果**。`tests/test_examples.py` 檢查最大 ef 的 recall 不低於最小 ef，而且讀回的 index 給出完全相同的結果。
- 只要求 numpy 與 binding；不需要 hnswlib、FAISS 或任何資料集。

### 全部的 correctness 檢查

```bash
uv venv --python 3.13 .venv          # A 的 Demo 已建立時可以略過
uv pip install --python .venv/bin/python -r requirements-dev.txt
./scripts/test.sh
```

- 依序執行：`cargo fmt --all --check`、`cargo clippy --workspace --all-targets -- -D warnings`、`cargo test`（只跑零相依的 hnsw crate）、
  `maturin develop --release --uv -m python/Cargo.toml`、pytest；有 `.venv-embed` 時，另以 `HF_HUB_OFFLINE=1` 跑 EmbeddingGemma 2 的模型測試。
- 這是 correctness 檢查，不需要 heavy lock（[MACHINE_POLICY.md](../MACHINE_POLICY.md)）。
- 最後一次記錄的結果（019 第一則補記，commit dca22c9）：rustfmt、clippy、cargo test 都通過；pytest 1024 passed、3 skipped；`.venv-embed` 16 passed。
  之後的 commit 也調整過測試，數量可能不同。
- 會 skip 的情況：主 `.venv` 沒有 torch 與 sentence-transformers，模型測試會 skip；沒有 macOS 的 `sample` 時，profiling driver 的測試會 skip；
  設定了 `OMP_NUM_THREADS` 時，兩個第三方語意測試會 skip；`.venv-embed` 的模型測試在本機快取沒有該 model revision 時會 skip。
- 新版 toolchain 可能加入新的 clippy lint，讓 `-D warnings` 失敗。要用實測的版本：
  `rustup toolchain install 1.99.0`，再以 `RUSTUP_TOOLCHAIN=1.99.0 ./scripts/test.sh` 執行。

### runner 本身的 smoke（synthetic 資料，不需要資料集）

```bash
.venv/bin/python -m benchmarks.run benchmarks/configs/smoke.toml --mode smoke
.venv/bin/python -m benchmarks.run benchmarks/configs/smoke-duplicates.toml --mode smoke
.venv/bin/python -m benchmarks.report results/runs/<UTC 時間>-smoke
```

- 輸出在 `results/runs/<UTC 時間>-smoke/`；報告預設寫到該 run 目錄下的 `report/`。
- 只驗證 runner 本身：synthetic 資料只用於 correctness 與 stress test（SPEC §5），數字不得用於任何報告。`smoke-duplicates.toml` 驗證大量重複向量時結果不足 k 的紀錄。
- runner 用到 macOS 專屬的系統介面（architecture.md 第 4 節），smoke 模式也一樣。

## B. 資料準備

- 資料準備不需要 heavy lock。資料檔寫到 `datasets/`（不進 Git）。
- **準備程式會改寫 tracked 的 manifest**（`benchmarks/manifests/<名稱>.json`）：`prepared_with` 等欄位記錄當下的 commit 與 git 狀態，文字資料集還記錄 embedding 的耗時。
  準備完先看 `git diff benchmarks/manifests/`：
  - `files` 的 SHA-256 都沒有變：以 `git checkout -- benchmarks/manifests/<名稱>.json` 還原。formal 要求 git 乾淨，而 runner 以 manifest 的 SHA-256 核對資料檔。
  - SHA-256 不同（例如在其他機器或其他浮點環境上產生的 embedding）：那是另一份資料，不是本 repo 登記的資料集。
    runner 在 smoke 與 formal 都會比對 SHA-256，git 中的 manifest 不能用；保留新的 manifest 只能在本機跑 smoke（formal 要求 git 乾淨），數字也不能與本 repo 對照。
- 磁碟（專案作者機器上的 `du -sh`）：`datasets/glove-100-angular` 約 949 MB（含原始 HDF5）、`datasets/simplewiki`（原始 parquet）約 150 MB、
  `simplewiki-embeddinggemma2-100k` 約 390 MB、pilot 約 42 MB；另外 Hugging Face 的模型快取在 repo 外（`model.safetensors` 約 1.5 GB）。

### B.1 glove-100-angular

```bash
.venv/bin/python -m benchmarks.prepare_glove --download
```

- 從 `http://ann-benchmarks.com/glove-100-angular.hdf5` 下載（manifest 記錄 485,413,888 bytes；不需要帳號）。網站沒有公布 checksum，
  所以第一次下載時記下的 SHA-256 已寫在 manifest 中，之後的下載必須相符（011 §2）。
- 產出：train 1,183,514 × 100、queries 10,000 × 100、ann-benchmarks 的 neighbors，以及 n = 100,000、300,000、1,183,514 的 frozen ground truth。
- 專案作者以修正後的程式重跑一次：約 466 秒、peak RSS 約 3.1 GB，6 個檔案的 SHA-256 與第一次完全相同（NEXT.md）。其他機器上是否逐位元相同沒有驗證。

### B.2 simplewiki-embeddinggemma2（真實文字，768 維）

1. **獨立的 embedding 環境**（與 benchmark 用的 `.venv` 分開，避免 numpy 等套件被連帶改版而影響 007 的實測語意）：

   ```bash
   uv venv --python 3.13 .venv-embed
   uv pip install --python .venv-embed/bin/python -r requirements-embed.txt
   ```

2. **corpus**：repo 沒有下載 parquet 的程式，要手動放到 `datasets/simplewiki/raw/train-00000-of-00001.parquet`。
   來源是 Hugging Face `wikimedia/wikipedia`，config `20231101.simple`，revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`，
   檔案 `20231101.simple/train-00000-of-00001.parquet`（156,885,218 bytes，SHA-256 `31bded16768a47c286becd292079122f5d7d4397a17b87d4250a00ccd581e6f0`；011 §3.1）。
   不需要登入。例如：

   ```bash
   mkdir -p datasets/simplewiki/raw
   curl -L -o datasets/simplewiki/raw/train-00000-of-00001.parquet \
     https://huggingface.co/datasets/wikimedia/wikipedia/resolve/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.simple/train-00000-of-00001.parquet
   shasum -a 256 datasets/simplewiki/raw/train-00000-of-00001.parquet
   ```

   `prepare_text` 開始時會比對 SHA-256，不符就拒絕。

3. **模型**：第一次執行 `embed_text` 時，經由 huggingface-hub **匿名**下載 `google/embeddinggemma-2` 的固定 revision（不使用任何 token），
   快取在 `~/.cache/huggingface`。`model.safetensors` 為 1,488,915,288 bytes，下載後比對 SHA-256（011 §3.4）。使用條件見該模型的 model card。

4. **產生資料集**（在 repo 根目錄，以主 `.venv` 執行；embedding 由它以 `.venv-embed` 的子 process 產生）：

   ```bash
   # pilot（decision 011 §3.5）
   .venv/bin/python -m benchmarks.prepare_text --name simplewiki-embeddinggemma2-pilot --n-base 10000 --n-queries 1000
   # 正式文字 workload（decision 012）
   .venv/bin/python -m benchmarks.prepare_text --name simplewiki-embeddinggemma2-100k --n-base 100000 --n-queries 2000 \
       --gt-n 10000 --gt-n 30000 --gt-n 100000
   ```

- embedding 固定 fp32、CPU、8 threads、batch size 1（不 padding）。每個 job 的前 32 筆會單獨重新 encode 一次，必須逐位元相同。
- 時間（專案作者的紀錄，都在有其他負載的機器上）：pilot 的 base 約 1,124 秒、query 約 108 秒（load 約 9–12）；
  100k 的 base 約 10,333 秒、query 約 170 秒（load 約 4–9）（012 與補記）。RAM 用量沒有記錄。

## C. 真實資料的 correctness smoke

```bash
.venv/bin/python -m benchmarks.run benchmarks/configs/glove-correctness.toml --mode smoke
.venv/bin/python -m benchmarks.run benchmarks/configs/text-pilot-correctness.toml --mode smoke
.venv/bin/python -m benchmarks.run benchmarks/configs/text-correctness.toml --mode smoke
.venv/bin/python -m benchmarks.report results/runs/<該 smoke run 的目錄>
```

- 不需要 heavy lock。三份 config 各只有一個 build：GloVe n = 100,000、文字 pilot n = 10,000、文字 n = 100,000，都是 (M, efC) = (16, 200)、seed 1，ef 從 10 開始（SPEC Core 3、Week 1 第 6、7 項）。
- 要看的是報告的「recall 隨 efSearch（010 §6）」：每個 (n, M, efC) 都沒有系統性惡化（最大 ef 的 recall 不低於最小 ef），以及沒有結果不足 k、
  沒有 `nondeterministic` 標記。專案作者的結果在 README 的 correctness 表。
- recall 不受計時影響；在相同版本與相同資料下，預期與 README 的 correctness 表接近。其他機器上是否逐位元相同沒有驗證。
- smoke 報告也有 QPS 等計時欄位，可以看相對趨勢，但不得與 README 的數字對照，也不得寫成結論。
- 完整 GloVe（n = 1,183,514）的 recall 範圍與 ann-benchmarks 自帶 neighbors 的一致率，可用 `benchmarks/configs/glove-cost-probe-full.toml`（smoke 模式）檢查；它是最貴的一份，成本見 013 的成本估算。

## D. Core formal

### 條件

除了「人工確認」一項，下列條件都由程式檢查：開始前不符就拒絕執行；執行中不符時，run 記為 `invalid`，或在 gate 等待逾時後記為 `aborted`。

- **只限 macOS 筆電，接 Apple 原廠 70W 或 140W USB-C 變壓器**（`pmset -g ac` 的製造商為 `Apple Inc.`、名稱為 `70W USB-C Power Adapter` 或 `140W USB-C Power Adapter`；018），
  接 AC 電源，電力模式為自動（0）或高效能（2），而且在 run 內不得改變；同一個資料集的 formal run 用同一個電力模式（016），一份報告只合併同一個變壓器的 run（018）。
  沒有這類變壓器資訊的機器（例如桌上型 Mac）通過不了前置檢查（依程式判斷，沒有實測）。
  專案作者的條件：GloVe 是高效能模式與 70W（協定 008-v2），文字是高效能模式與 140W（008-v4）。
- **heavy lock**：經由 `./tools/with-mac-lock`（`formal_launcher.sh` 會自動取得）。拿不到就不跑，不得刪除別人的 lock（[MACHINE_POLICY.md](../MACHINE_POLICY.md)）。
- **程式與環境**：git 乾淨（含 untracked 檔案）；interpreter 是 repo 的 `.venv`；hnswlib、faiss-cpu、numpy、maturin 的版本等於 `requirements-dev.txt`；
  binding 由乾淨 HEAD 上的 `./scripts/test.sh` 編譯（晚於 HEAD 與每個 tracked 的原始碼）；資料集已登記且 SHA-256 相符（B）；config 已 commit（12 份都已 commit）。
- **人工確認**：家目錄中的 cargo 設定檔（例如 `~/.cargo/config.toml`）中的 `[profile]` 無法從 binary 得知，要確認沒有這類設定（008 §11）。
- **第一次 formal 之前**：閒置 load 量測（`benchmarks.idle_load`，5 分鐘預備期加 10 分鐘，同樣要在 heavy lock 與 formal 的電力條件下）必須是 `ok`。`formal_batch.sh` 在沒有 `--skip-idle` 時會先做這一步。
- **機器閒置**：背景沒有 heavy workload、load 低（architecture.md 第 4 節的 `machine_quiet` 與子 process 前的 gate）。
- **run 進行中**：repo 內不得建立、修改或刪除任何檔案（含 untracked 檔案），不得 commit，不得跑 pytest 或編譯等重工作。

### 指令

```bash
./scripts/test.sh                          # 在乾淨的 HEAD 上；讓 binding 晚於 HEAD
mkdir -p results/batches
# GloVe 8 份（第一個批次不加 --skip-idle，先量閒置 load）
./scripts/formal_launcher.sh results/batches/glove.log 28800 --powermode 2 \
  benchmarks/configs/glove-formal-1-m16-order.toml benchmarks/configs/glove-formal-2-m16-seeds-efc400.toml \
  benchmarks/configs/glove-formal-3-m48-seeds-1-2.toml benchmarks/configs/glove-formal-4-m48-seeds-3-4.toml \
  benchmarks/configs/glove-formal-5-m48-seed-5-efc100.toml benchmarks/configs/glove-formal-6-m32-300k.toml \
  benchmarks/configs/glove-formal-7-m8.toml benchmarks/configs/glove-formal-8-full.toml
# 文字 4 份（閒置 load 已經是 ok）
./scripts/formal_launcher.sh results/batches/text.log 28800 --powermode 2 --skip-idle \
  benchmarks/configs/text-formal-1-m16-order.toml benchmarks/configs/text-formal-2-m16-seeds-efc400.toml \
  benchmarks/configs/text-formal-3-m8-efc100.toml benchmarks/configs/text-formal-4-highm-scaling.toml
```

- 用法是 `formal_launcher.sh <batch log> <最長等待秒數> [--powermode N] [--skip-idle] [--mode formal|smoke] <config>...`。
  它每 60 秒呼叫一次 `machine_quiet`，連續 3 次成立才在 heavy lock 下啟動批次，印出 `LAUNCHED` 後就返回；批次在背景繼續。
  進度看 `<batch log>` 的 `RUN_START` / `RUN_END` / `STOP` / `ALL_DONE`；等待的原因記在 `<batch log>.wait`。
- 也可以一次跑一份：`./tools/with-mac-lock .venv/bin/python -m benchmarks.run <formal config> --mode formal`。
- 輸出在 `results/runs/<UTC 時間>-formal/`。

### 失敗的處理（010 §3、013 §5）

- `aborted`、`invalid`、`error`：整份 config 重跑；原本的 run 之後以報告的 `--excluded` 列出，資料不合併。
- `incomplete`：照實保留，不重跑。complete 的 run 不得被取代；因環境變化而作廢，必須另寫 decision 並登記在 `report.py` 的 `VOIDED_RUNS`（016）。

### 時間

- 013 的估計：12 份 config、38 個 build、負載下的上限約 19.2 h（GloVe 11.3 h、文字 7.9 h），每份 0.3–2.6 h。
- 不含在內的時間：每個子 process 前的 load / 背景 gate（每次最多等 10 分鐘，超過就中止整個 run）、每次 formal 前的 `./scripts/test.sh`、閒置 load 量測，以及失敗後的整份重跑。
- 報告的產生見 G。報告會拒絕：缺少或重複的 formal config、同一份 config 另有沒交出的 complete / incomplete run、沒以 `--excluded` 列出的失敗 run、
  不同的協定版本、電力模式或變壓器，以及版本或量測程式不同的 run（013 補記）。

## E. Gate profiling

需要 D 的 formal run：各實作的 ef 由正式 run 選出。每個資料集：

```bash
mkdir -p results/profiles
.venv/bin/python -m benchmarks.gate_profile plan <該資料集全部 formal run 目錄…> > results/profiles/gate-<資料集>-plan.sh
./tools/with-mac-lock bash results/profiles/gate-<資料集>-plan.sh
.venv/bin/python -m benchmarks.gate_profile check <同一組 run 目錄…>
```

- 參數（015 §2）：(100,000, 16, 200)、seed 1、原始順序；各實作在正式 run 中 Recall@10 ≥ 0.95 的最小 ef（任一實作沒有達到時，三者一起改用 0.90）；
  每個實作取樣 search 30 秒、build 300 秒，`sample` 每 1 ms 一次。每個資料集 6 次取樣。
- `check` 核對：取樣前紀錄、原始報告與排行都存在；參數與計畫相同；由祖先持有 heavy lock；formal 的電力與背景條件；git 乾淨；
  被取樣的程式（`PROFILED_PATHS`）與正式 run 沒有差異；`BUILD_INFO` 與套件版本相同；search 階段的 Recall@10 與正式 run 的同一點完全相同。
- 計畫拒絕覆寫已存在的同名輸出：上一次嘗試的檔案要先移開。
- 專案作者的結果：12 次取樣都通過 `check`。pip 安裝的 faiss-cpu 沒有符號表，FAISS 的歸屬是近似的（009 補記）。
- 沒有 formal run 時，只能驗證工具本身（synthetic 資料，不作證據；009）：

  ```bash
  ./scripts/profile.sh rust-search-smoke 5 -- --impl rust-hnsw --phase search \
      --synthetic 20000,64,2000 --m 16 --ef-construction 100 --seed 1 --ef 100 --k 10
  ```

## F. Advanced C

依 NEXT.md「下一步」第 7 點與 019 §9 整理。判定的結果與背景負載的影響見 [architecture.md 第 6 節](architecture.md#6-advanced-cr--f--s--p--q-如何共用同一張圖)。

### 與專案作者的 run 綁定的部分

- `quant_smoke` 與 `quant_report` 的 `--core-runs` 必須**恰好**列出 019 §2 登記的 4 個 Core formal run（以目錄名稱比對）：

  | 資料集 | seed 1 | seeds 2–5 |
  | --- | --- | --- |
  | GloVe | `20261009T080239Z-formal` | `20261009T091611Z-formal` |
  | 文字 | `20261015T191333Z-formal` | `20261015T210125Z-formal` |

  其他名稱會被拒絕；登記的目錄不存在時記為缺失：`quant_report` 的 C7a、C7b 依 019 §2 記為無法判定，`quant_smoke` 的 `ok` 也不會是 true（依程式判斷，沒有實測）。
  所以在其他機器上，與 Core 的逐位元對照無法改用自己的 Core run，除非修改程式（那要另寫 decision）。
- 判讀用的 e\*（GloVe 304、文字 48）是由專案作者的 Core 報告事先固定的（019 §2），不從新的 run 重算。
- 條件與 D 相同，另外：每一步都先通過 `.venv/bin/python -m benchmarks.machine_quiet --powermode 2`（印出 `yes`），再在 heavy lock 下進行；
  所有 C 的 run 用高效能模式與同一個變壓器；formal、診斷與 profiling 進行中 repo 不得變 dirty（019 §5、§10）。

### 1. 測試與 H1 前提

```bash
./scripts/test.sh
.venv/bin/python -m benchmarks.quant_profile premise      # 寫出 results/quant/premise.json；拒絕覆寫
```

- `premise` 以 `otool -tV` 反組譯 binding，檢查兩個 kernel 的加總是依序的純量相依鏈。
- `quant_report` 要求 premise 的 binding SHA-256 等於 C formal run 記錄的 `rust_hnsw` SHA-256。之後若重新編譯而 bytes 不同，
  先把舊的 premise 改名保留，再重跑 `premise`（019 第一則補記第 3 節）。

### 2. correctness smoke

```bash
mkdir -p results/batches
./scripts/formal_launcher.sh results/batches/quant-smoke.log 86400 --powermode 2 --skip-idle --mode smoke \
  benchmarks/configs/glove-quant-smoke.toml benchmarks/configs/text-quant-smoke.toml
```

### 3. smoke 的核對

```bash
.venv/bin/python -m benchmarks.quant_smoke <GloVe 的 smoke run 目錄> <文字的 smoke run 目錄> \
  --core-runs results/runs/20261009T080239Z-formal results/runs/20261009T091611Z-formal \
              results/runs/20261015T191333Z-formal results/runs/20261015T210125Z-formal
.venv/bin/python -m benchmarks.machine_quiet --powermode 2      # 印出 yes 才繼續
./tools/with-mac-lock .venv/bin/python -m benchmarks.quant_smoke <同一組 smoke run 目錄> --core-runs <同上 4 個目錄> --port
```

- 核對 F、R、P 與 Core 的 ids 逐位元相同、C 內 F 與 R 相同；`--port` 在真實資料上重建 FAISS 的移植並核對 graph。結果以 JSON 印出，不寫檔案；全部通過時 exit code 為 0。
- smoke 之後不得修改假設、門檻、格點、ef、seed 或讀法；只能修 correctness bug（寫補記）或依 019 §5 上調 timeout（019 §9 第 6 步）。
- 專案作者的 smoke：`20261016T195505Z-smoke`、`20261016T201035Z-smoke`，都是 complete、沒有標記，ids 全部逐位元相同（019 第二則補記）。

### 4. formal（10 份 config）

```bash
./scripts/formal_launcher.sh results/batches/quant-formal.log 86400 --powermode 2 --skip-idle \
  benchmarks/configs/quant-glove-seed-1.toml benchmarks/configs/quant-glove-seed-2.toml benchmarks/configs/quant-glove-seed-3.toml \
  benchmarks/configs/quant-glove-seed-4.toml benchmarks/configs/quant-glove-seed-5.toml \
  benchmarks/configs/quant-text-seed-1.toml benchmarks/configs/quant-text-seed-2.toml benchmarks/configs/quant-text-seed-3.toml \
  benchmarks/configs/quant-text-seed-4.toml benchmarks/configs/quant-text-seed-5.toml
```

- `--skip-idle` 只在同樣的條件下已經有 `ok` 的閒置 load 量測時使用；新機器的第一個批次不要加。
- 失敗的處理同 D。乾淨配對不足而「無法判定」時，不得重跑 complete 的 run；是否補 seed 由專案作者決定，並另寫 decision（019 §9）。
- 專案作者的 formal：10 個 run 全部 complete（2026-10-16 20:38 – 10-17 00:08 UTC，commit 2c4e520、Apple 140W、高效能模式）；
  GloVe 的 seeds 1、4、5 有 `background` 標記（019 第三則補記）。

### 5. 不計時的診斷（每個資料集一次，兩次都列出全部 10 個 run 目錄）

```bash
.venv/bin/python -m benchmarks.machine_quiet --powermode 2      # 印出 yes 才開始
./tools/with-mac-lock env VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m benchmarks.quant_diag --dataset glove <10 個 C formal run 目錄>
./tools/with-mac-lock env VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m benchmarks.quant_diag --dataset text <10 個 C formal run 目錄>
```

- 輸出 `results/quant/diag-<glove|text>.json`，拒絕覆寫。診斷以 ids 與 formal 逐位元核對，綁定到 formal（019 §7）。
- 在 zsh 中，存放多個路徑的變數不加引號也不會自動切開，10 個目錄會被當成一個路徑（019 第三則補記記錄過一次這個錯誤）。請直接列出路徑，或使用陣列。

### 6. profiling（每個資料集）

```bash
mkdir -p results/profiles
.venv/bin/python -m benchmarks.quant_profile plan --dataset glove <10 個 C formal run 目錄> > results/profiles/quant-glove-plan.sh
.venv/bin/python -m benchmarks.machine_quiet --powermode 2      # 印出 yes 才開始
./tools/with-mac-lock bash results/profiles/quant-glove-plan.sh
.venv/bin/python -m benchmarks.quant_profile check --dataset glove <同一組 run 目錄>
```

文字以 `--dataset text` 與 `quant-text-plan.sh` 重複一次。取樣 R、F、S 在 e\*，以及 S 在它自己第一次 Recall@10 ≥ 0.95 的 ef；只取樣 search，每次 30 秒，
不取樣 build 與 FAISS（019 §8）。專案作者的 8 次取樣都通過 `check`。

### 7. 報告

```bash
.venv/bin/python -m benchmarks.quant_report <10 個 C formal run 目錄> \
  --core-runs results/runs/20261009T080239Z-formal results/runs/20261009T091611Z-formal \
              results/runs/20261015T191333Z-formal results/runs/20261015T210125Z-formal
```

- 其餘參數的預設值：`--diag results/quant/diag-glove.json results/quant/diag-text.json`、`--premise results/quant/premise.json`、
  `--profiles results/profiles`、`--out results/reports`（寫到 `quant-glove/` 與 `quant-text/`，兩份列出同一張判定表）。
- 重產既有的報告時，`--out` 請指到其他目錄（見 G）。

### 時間

- 019 §12 的估計（中央 / 悲觀）：formal 3.7 h / 6.2 h、`quant_diag` 0.9 h / 1.1 h、profiling 0.35 h / 0.5 h、smoke 0.6 h / 0.8 h；lock 合計約 5.8 h / 約 10 h。
- 專案作者的實際時段：smoke 2026-10-17 03:55–04:33（台灣時間）；formal 2026-10-16 20:38 – 10-17 00:08 UTC（NEXT.md、019 第三則補記）。

## G. 報告與 README 圖的重產

需要專案作者機器上的 `results/runs/`（不公開）。`report.py` 與 `quant_report.py` 都會寫進已存在的輸出目錄，所以重產時請把 `--out` 指到新的目錄，
再以 SHA-256 與既有的報告比對，不要覆蓋 `results/reports/`。比對時注意：

- 019 第一則補記記錄：隔離審查時以新舊 `report.py`（加入 Advanced C 之前與之後）重產兩份 Core 報告，report.json、report.md 與圖逐位元相同。
- 本機的 GloVe 報告是 2026-10-10 以當時的 `report.py` 產生的。之後 76a352b（2026-10-15）新增 frontier 與 scaling 圖，並讓 report.md 列出圖。
  2026-10-17 以與 76a352b 逐位元相同的 `report.py` 重產時，report.md 只多了「圖」一節，並改了「被排除的 run」一節的標題，所有的表與數字都相同
  （[docs/figures/README.md](figures/README.md)「能確認」第 5 點）。HEAD 的 `report.py` 沒有再對本機的 10-10 版實測。
- 文字報告（2026-10-16）與 C 的報告（2026-10-19）產生於 76a352b 之後。

```bash
# GloVe（8 個 run；1 個 aborted 的 run 以 --excluded 列出）
.venv/bin/python -m benchmarks.report \
  results/runs/20261009T080239Z-formal results/runs/20261009T091611Z-formal results/runs/20261009T101347Z-formal \
  results/runs/20261009T105616Z-formal results/runs/20261010T021933Z-formal results/runs/20261010T025425Z-formal \
  results/runs/20261010T034635Z-formal results/runs/20261010T040529Z-formal \
  --out results/regen/glove-100-angular --excluded results/runs/20261009T113943Z-formal

# 文字（4 個 run；4 個失敗與 3 個作廢的 run 以 --excluded 列出，016、017）
.venv/bin/python -m benchmarks.report \
  results/runs/20261015T191333Z-formal results/runs/20261015T210125Z-formal \
  results/runs/20261015T222221Z-formal results/runs/20261015T225545Z-formal \
  --out results/regen/simplewiki-embeddinggemma2-100k \
  --excluded results/runs/20261010T083544Z-formal --excluded results/runs/20261010T133129Z-formal \
  --excluded results/runs/20261010T202624Z-formal --excluded results/runs/20261011T033341Z-formal \
  --excluded results/runs/20261010T183115Z-formal --excluded results/runs/20261010T195251Z-formal \
  --excluded results/runs/20261010T112849Z-formal

# Advanced C（10 個 run；輸出 results/regen/quant-glove/ 與 quant-text/）
.venv/bin/python -m benchmarks.quant_report \
  results/runs/20261016T203802Z-formal results/runs/20261016T211434Z-formal results/runs/20261016T214152Z-formal \
  results/runs/20261016T220514Z-formal results/runs/20261016T221833Z-formal \
  results/runs/20261016T223724Z-formal results/runs/20261016T230246Z-formal results/runs/20261016T232100Z-formal \
  results/runs/20261016T234053Z-formal results/runs/20261016T235454Z-formal \
  --core-runs results/runs/20261009T080239Z-formal results/runs/20261009T091611Z-formal \
              results/runs/20261015T191333Z-formal results/runs/20261015T210125Z-formal \
  --out results/regen
```

run 的清單取自各報告的「合併的 run」與「被排除的 run」兩節（Core）與 019 第三則補記（C）。

### `docs/figures/` 的來源

README 的圖是從報告的輸出目錄手動複製的。下表以 SHA-256 對照本機的報告輸出：

| `docs/figures/` 的檔案 | 來源 | SHA-256 |
| --- | --- | --- |
| `glove-100-angular/frontier_n100000.png` | **沒有記錄**（見下） | `4269e8f9961fd0ce65e61e413733e39153f3cb8f8a89b9593d24b98536aed161` |
| `glove-100-angular/scaling.png` | **沒有記錄**（見下） | `1b723ca7c1445d1d3486b7d898efea589445c85ea02285fb798e4549c232950d` |
| `glove-100-angular/recall_vs_ef_n100000_m16_efc200.png` | `results/reports/glove-100-angular/` 的同名檔 | `d7b41c8424031f7c2fe08be3a5e080acbe7d13809c7d112ef20eaa3307af54ce` |
| `simplewiki-embeddinggemma2-100k/frontier_n100000.png` | `results/reports/simplewiki-embeddinggemma2-100k/` 的同名檔 | `1d6f7b77d761405b7d3679396682e2dc6b3b6289431d49935cb48eae5182ddf5` |
| `simplewiki-embeddinggemma2-100k/scaling.png` | 同上 | `def6695bee0060e4f9ede73549d124e0e74d6bcf3e8237b82e37ee1942f94443` |
| `simplewiki-embeddinggemma2-100k/recall_vs_ef_n100000_m16_efc200.png` | 同上 | `249f4f2550a22b7e09f10afa10f6aa058a3b4de4fe79f7437849c90d17aab580` |
| `advanced-c/glove/recall_qps.png` | `results/reports/quant-glove/` 的同名檔 | `52bf4702267695a94e244cbc0ef537207da07556f3e18bf70439759ea48fa631` |
| `advanced-c/glove/quant_frontier_0.95.png` | 同上 | `af2f8635ccdba67d932e2e8a1da140db59abc33c0713116ec5d653ea10cede24` |
| `advanced-c/glove/profile_breakdown.png` | 同上 | `7e7755c01a3a5b67a0560c5479617866c416d543f78f06930ddadd06edfb36be` |
| `advanced-c/text/recall_qps.png` | `results/reports/quant-text/` 的同名檔 | `4fec8346f33156c31a613059d792fa3f1f78d4d83cd5495f4a200e6ed0b8e403` |
| `advanced-c/text/quant_frontier_0.95.png` | 同上 | `186468a6fd91998aa5b0fe3efe6d69e49a9b715d30770cae627709972194d40d` |
| `advanced-c/text/profile_breakdown.png` | 同上 | `c3d5ce809ad4d555762c5c52ce8002e81116faee8d5f4f54f2409030546df5cc` |

- GloVe 的 `frontier_n100000.png` 與 `scaling.png`：本機的 GloVe 報告目錄是 2026-10-10 產生的，早於讓 `report.py` 畫 frontier 與 scaling 圖的 commit 76a352b（2026-10-15），
  所以目錄裡沒有這兩張圖；兩張圖在 f219d39 加入 `docs/figures/`，當時產生它們的指令沒有留下紀錄。進一步的考證見 [docs/figures/README.md](figures/README.md)。本文件沒有重產它們。
  若要確認，以上面的 GloVe 指令重產，再比對 SHA-256；不同時不要覆蓋已 commit 的圖，先查明原因。
- 其他 10 張的 SHA-256 與報告輸出相同。重產之後比對同名檔的 SHA-256，例如：

  ```bash
  shasum -a 256 results/regen/quant-glove/recall_qps.png docs/figures/advanced-c/glove/recall_qps.png
  ```
