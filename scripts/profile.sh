#!/usr/bin/env bash
# 以 macOS 內建的 `sample` 對 benchmarks/profile.py 的指定階段取樣，輸出 self-time（top of stack）排行。
#
# 用法：
#   ./scripts/profile.sh <名稱> <取樣秒數> -- <benchmarks.profile 的參數…>
# 例如：
#   ./scripts/profile.sh rust-search-smoke 5 -- --impl rust-hnsw --phase search \
#       --synthetic 20000,64,2000 --m 16 --ef-construction 100 --seed 1 --ef 100 --k 10
# Gate 證據（decision 009：正式資料集、heavy lock、相近的 Recall@10）：
#   ./tools/with-mac-lock ./scripts/profile.sh glove-rust-search 20 -- --impl rust-hnsw --phase search \
#       --dataset glove-100-angular --n 100000 --m 16 --ef-construction 200 --seed 1 --ef 128 --k 10
#
# 結果寫到 results/profiles/<名稱>.{log,json,sample.txt,top.txt}（results/ 不進版控）。
# <名稱>.json 是取樣前寫出的紀錄：參數、資料集 checksum、git 狀態、binding 的編譯資訊、是否持有 heavy lock、
# search 階段的 Recall@10（只有 --dataset 才有）。
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -lt 3 ] || [ "$3" != "--" ]; then
  echo "usage: $0 <name> <seconds> -- <benchmarks.profile args...>" >&2
  exit 64
fi
name=$1
seconds=$2
shift 3

out=results/profiles
mkdir -p "$out"
log="$out/$name.log"

.venv/bin/python -m benchmarks.profile "$@" --seconds "$((seconds + 3))" --meta "$out/$name.json" >"$log" 2>&1 &
pid=$!

# 等待要取樣的階段開始。沒有時間上限：完整 GloVe 的 build 單一實作就要約 10 分鐘，固定上限一到就取樣會取到錯誤的階段（009）。
# workload 沒有印出 READY 就結束時，以非 0 結束，絕不取樣。
until grep -q '^READY' "$log"; do
  if ! kill -0 "$pid" 2>/dev/null; then
    grep -q '^READY' "$log" && break
    echo "workload exited before READY:" >&2
    cat "$log" >&2
    exit 1
  fi
  sleep 0.5
done

sample "$pid" "$seconds" -file "$out/$name.sample.txt" >/dev/null
wait "$pid"
.venv/bin/python -m benchmarks.profile_report "$out/$name.sample.txt" --top 20 | tee "$out/$name.top.txt"
