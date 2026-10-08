#!/usr/bin/env bash
# 跑全部 correctness 檢查：rustfmt、clippy、Rust 測試、build Python binding、pytest。
# 不是 benchmark，不需要 Mac heavy lock。
#
# 前置：已安裝 Rust（rustup）與 uv，並建立 .venv：
#   uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v cargo >/dev/null 2>&1 && [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

export VIRTUAL_ENV="$PWD/.venv"
export PYO3_PYTHON="$VIRTUAL_ENV/bin/python"

echo "==> cargo fmt --check"
cargo fmt --all --check

echo "==> cargo clippy"
cargo clippy --workspace --all-targets -- -D warnings

echo "==> cargo test"
cargo test

echo "==> maturin develop (release)"
"$VIRTUAL_ENV/bin/maturin" develop --release --uv -m python/Cargo.toml

echo "==> pytest"
"$VIRTUAL_ENV/bin/python" -m pytest

# EmbeddingGemma 2 的模型測試需要 .venv-embed（torch、sentence-transformers；decision 011 §3.4）。
# 沒有建立時只跑上面主 venv 中的純函式測試（模型測試在那裡會 skip）。
if [ -x .venv-embed/bin/python ]; then
  echo "==> pytest (.venv-embed: EmbeddingGemma 2)"
  HF_HUB_OFFLINE=1 .venv-embed/bin/python -m pytest tests/test_embed_text.py
fi
