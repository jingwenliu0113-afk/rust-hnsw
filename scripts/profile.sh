#!/usr/bin/env bash
# 以 macOS 內建的 `sample` 對 benchmarks/profile.py 的指定階段取樣，輸出 self-time（top of stack）排行。
#
# 用法：
#   ./scripts/profile.sh <名稱> <取樣秒數> -- <benchmarks.profile 的參數…>
# 例如：
#   ./scripts/profile.sh rust-search-smoke 5 -- --impl rust-hnsw --phase search \
#       --synthetic 20000,64,2000 --m 16 --ef-construction 100 --seed 1 --ef 100 --k 10
#
# 結果寫到 results/profiles/<名稱>.{log,sample.txt,top.txt}（results/ 不進版控）。
# 這不是 latency benchmark；要作為 Gate 證據的 profiling，應以真實資料、在 ./tools/with-mac-lock 下執行。
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

.venv/bin/python -m benchmarks.profile "$@" --seconds "$((seconds + 3))" >"$log" 2>&1 &
pid=$!

# 等待要取樣的階段開始（build 可能需要一段時間）。
for _ in $(seq 1 1200); do
  if grep -q '^READY' "$log"; then
    break
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "workload exited before READY:" >&2
    cat "$log" >&2
    exit 1
  fi
  sleep 0.5
done

sample "$pid" "$seconds" -file "$out/$name.sample.txt" >/dev/null
wait "$pid"
.venv/bin/python -m benchmarks.profile_report "$out/$name.sample.txt" --top 20 | tee "$out/$name.top.txt"
