#!/usr/bin/env bash
# 正式 run 的批次（decision 013 §5、014）：在同一次 Mac heavy lock 持有中，先量閒置 load（008 §2 與補記），
# 結果是 ok 才依序跑 formal config；任一 run 的 exit code 不是 0（不是 complete）就停下來，先檢查原因
# （aborted / invalid / error 的 run 要整份 config 重跑，原本的 run 之後以 report 的 --excluded 列出）。
#
# 用法（必須由 with-mac-lock 啟動，通常經由 scripts/formal_launcher.sh）：
#   ./tools/with-mac-lock ./scripts/formal_batch.sh <log> [--powermode N] [--skip-idle] [--mode formal|smoke] <config>...
# --powermode N：每份 config 開始前確認 `pmset -g` 的 powermode 是 N，不是就停下來（decision 016：同一個資料集的 formal run 用同一個電力模式）。
# --skip-idle：閒置 load 已在先前的批次量過且為 ok 時使用。
# --mode：傳給 benchmarks.run 的模式，預設 formal。smoke 用於 decision 019 §9 第 6 步的 correctness smoke（同樣要在 heavy lock 下、機器閒置時執行）。
# log 的標記：IDLE_START / IDLE_END / RUN_START / RUN_END / STOP / ALL_DONE。
# 正式 run 進行中，repo 內不得建立或修改任何檔案、不得 commit、不得跑重工作（NEXT 的執行手冊）。
set -u
cd "$(dirname "$0")/.."
PYTHON=${FORMAL_PYTHON:-.venv/bin/python}
PMSET=${FORMAL_PMSET:-pmset}
if [ "$#" -lt 2 ]; then
  echo "usage: $0 <log> [--powermode N] [--skip-idle] [--mode formal|smoke] <config>..." >&2
  exit 64
fi
log=$1
shift
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
powermode=""
if [ "$1" = "--powermode" ]; then
  powermode=$2
  shift 2
fi
check_powermode() {
  [ -z "$powermode" ] && return 0
  current=$("$PMSET" -g | awk '$1 == "powermode" {print $2}')
  if [ "$current" != "$powermode" ]; then
    echo "STOP powermode=$current expected=$powermode $(stamp)" >> "$log"
    exit 1
  fi
}
skip_idle=0
if [ "${1:-}" = "--skip-idle" ]; then
  skip_idle=1
  shift
fi
mode=formal
if [ "${1:-}" = "--mode" ]; then
  mode=${2:-}
  case "$mode" in
    formal | smoke) shift 2 ;;
    *) echo "unknown mode: $mode (formal or smoke)" >&2; exit 64 ;;
  esac
fi
if [ "$skip_idle" -eq 0 ]; then
  check_powermode
  echo "IDLE_START $(stamp)" >> "$log"
  out=$("$PYTHON" -m benchmarks.idle_load 2>&1)
  code=$?
  printf '%s\n' "$out" >> "$log"
  # idle_load 最後一行印出紀錄的路徑。
  record=$(printf '%s\n' "$out" | tail -1)
  status=$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["status"])' "$record" 2>/dev/null || echo unknown)
  echo "IDLE_END $(stamp) exit=$code status=$status" >> "$log"
  if [ "$code" -ne 0 ] || [ "$status" != "ok" ]; then
    echo "STOP idle_load status=$status exit=$code" >> "$log"
    exit 1
  fi
fi
for config in "$@"; do
  check_powermode
  echo "RUN_START $(stamp) $config" >> "$log"
  "$PYTHON" -m benchmarks.run "$config" --mode "$mode" >> "$log" 2>&1
  code=$?
  echo "RUN_END $(stamp) $config exit=$code" >> "$log"
  if [ "$code" -ne 0 ]; then
    echo "STOP $config exit=$code" >> "$log"
    exit "$code"
  fi
done
echo "ALL_DONE $(stamp)" >> "$log"
