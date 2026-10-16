#!/usr/bin/env bash
# 等機器可以跑正式 run 時，才在 Mac heavy lock 下啟動 scripts/formal_batch.sh（decision 008 §2、014）。
#
# 每 60 秒檢查一次：heavy lock 沒人持有、背景沒有 heavy workload（014）、1 分鐘 load < 4.0；連續 3 次都成立才啟動。
# 批次以新的 session（setsid）啟動，不受啟動它的終端或 Claude session 結束影響。lock 被別人先拿走時不搶占，回去繼續等。
#
# 用法：./scripts/formal_launcher.sh <batch log> <最長等待秒數> [--powermode N] [--skip-idle] [--mode formal|smoke] <config>...
# --powermode N：另外要求 `pmset -g` 的 powermode 是 N 才算可以開始，並傳給 formal_batch.sh 在每份 config 前再確認（decision 016）。
# --mode 照原樣傳給 formal_batch.sh（預設 formal）。
# 結束：啟動成功印出 LAUNCHED（exit 0）；等超過上限印出 GAVE_UP（exit 2）。批次本身的進度看 <batch log>。
set -u
cd "$(dirname "$0")/.."
PYTHON=${FORMAL_PYTHON:-.venv/bin/python}
PMSET=${FORMAL_PMSET:-pmset}
POLL_S=${FORMAL_POLL_S:-60}
LOCK_DIR=/tmp/ai-portfolio-mac-heavy.lock
if [ "$#" -lt 3 ]; then
  echo "usage: $0 <batch log> <max wait seconds> [--powermode N] [--skip-idle] [--mode formal|smoke] <config>..." >&2
  exit 64
fi
log=$1
max_wait=$2
shift 2
powermode=""
if [ "$1" = "--powermode" ]; then
  powermode=$2
fi
start=$(date +%s)
ok=0
while :; do
  if [ $(( $(date +%s) - start )) -ge "$max_wait" ]; then
    echo "GAVE_UP after ${max_wait}s"
    exit 2
  fi
  # 機器是否閒置：lock、背景 CPU、load、GPU、記憶體（benchmarks/machine_quiet.py；使用者 2026-10-16 要求也看 GPU 與 RAM）。
  if [ -n "$powermode" ]; then
    quiet=$("$PYTHON" -m benchmarks.machine_quiet --powermode "$powermode" 2>&1)
  else
    quiet=$("$PYTHON" -m benchmarks.machine_quiet 2>&1)
  fi
  [ "$quiet" = "yes" ] || echo "$(date -u +%H:%M:%SZ) $quiet" >> "$log.wait"
  if [ "$quiet" = "yes" ]; then ok=$((ok + 1)); else ok=0; fi
  if [ "$ok" -ge 3 ]; then
    nohup perl -e 'use POSIX qw(setsid); setsid(); exec @ARGV' ./tools/with-mac-lock ./scripts/formal_batch.sh "$log" "$@" \
      >> "$log.launcher" 2>&1 < /dev/null &
    sleep 20
    if grep -qE "IDLE_START|RUN_START" "$log" 2>/dev/null; then
      echo "LAUNCHED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      exit 0
    fi
    echo "launch did not start (lock taken?): $(tail -3 "$log.launcher" 2>/dev/null | tr '\n' ' ')" >&2
    ok=0
  fi
  sleep "$POLL_S"
done
