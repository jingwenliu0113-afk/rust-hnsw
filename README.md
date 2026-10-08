# rust-hnsw

研究問題：哪些演算法與低階實作決策，真正決定 HNSW 的 recall、延遲與記憶體取捨？
完整規格見 [`SPEC.md`](SPEC.md)，目前進度見 [`NEXT.md`](NEXT.md)。

## 目前狀態

- Core / Day 1 完成：cosine distance、exact brute-force（ground truth）、
  Python binding skeleton、固定的 Python batch interface。
- Core / Week 1 進行中：HNSW insert / search、save / load、Python binding 與 efSearch 介面、
  hnswlib / FAISS adapters、benchmark runner（smoke 已驗證）已完成。
- 尚未有正式 benchmark 結果：第一組真實資料集還在等決定（見 `NEXT.md`）。

## 目錄

```text
crates/hnsw/        Rust 實作（distance、exact search、HNSW、save / load）
python/             PyO3 binding，Python module 名稱 `rust_hnsw`
benchmarks/         統一 batch interface、adapters、Recall 指標、benchmark runner（run.py）、報告（report.py）
benchmarks/configs/ runner 的 TOML config（目前只有 smoke）
tests/              Python 測試（binding、adapters、第三方原生語意、interface、metrics）
scripts/test.sh     一鍵跑全部 correctness 檢查
scripts/profile.sh  以 macOS `sample` 對 build / search 取樣（decision 009）
docs/decisions/     decision logs
tools/with-mac-lock 正式 benchmark 用的 Mac heavy lock
```

## 開發環境

```bash
# Rust（rustup 安裝時未改 PATH，每個新 shell 需先載入）
source "$HOME/.cargo/env"

# Python 3.13 venv
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

## 測試

```bash
./scripts/test.sh
```

依序跑 rustfmt、clippy、`cargo test`、`maturin develop --release`、pytest。
這些是 correctness 檢查，不需要 Mac heavy lock；正式 benchmark 才需要（見 `MACHINE_POLICY.md`）。

## Benchmark runner（decision 008）

```bash
# smoke：只驗證 runner 本身（synthetic 資料，數字不得用於報告），不需要 lock
.venv/bin/python -m benchmarks.run benchmarks/configs/smoke.toml --mode smoke

# formal：先在乾淨的 HEAD 上跑 ./scripts/test.sh，再在 lock 下執行
./tools/with-mac-lock .venv/bin/python -m benchmarks.run <formal config> --mode formal
```

輸出在 `results/runs/<UTC 時間>-<mode>/`（不進版控）。formal 需要已登記的真實資料集，目前還沒有。

報告（依 decision 010 的比較規則：frontier 讀值、variance、scaling、recall vs efSearch 圖）：

```bash
.venv/bin/python -m benchmarks.report results/runs/<run 目錄> [<run 目錄> ...]
```

## 主要決策

- [001 cosine 距離定義、正規化與 tie-break](docs/decisions/001-cosine-distance-and-normalization.md)
- [002 統一 Python batch interface 與計時協定](docs/decisions/002-python-batch-interface.md)
- [003 Recall@K 定義與 tie-aware 診斷](docs/decisions/003-recall-definition.md)
- [004 HNSW Core 演算法細節與 baseline 表示法](docs/decisions/004-hnsw-core-algorithm.md)
- [005 HNSW index 檔案格式（save / load）](docs/decisions/005-index-file-format.md)
- [006 HNSW 的 Python binding 與 efSearch 介面](docs/decisions/006-hnsw-python-binding.md)
- [007 hnswlib / FAISS adapters 與實測語意](docs/decisions/007-third-party-adapters.md)
- [008 benchmark runner 與量測協定的落實](docs/decisions/008-benchmark-runner.md)
- [009 profiling 方法（Gate 證據的產生方式）](docs/decisions/009-profiling-method.md)
- [010 比較規則與 variance 彙整（事先登記）](docs/decisions/010-comparison-rules.md)
