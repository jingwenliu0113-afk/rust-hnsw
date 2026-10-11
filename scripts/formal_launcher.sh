#!/usr/bin/env bash
# 等機器可以跑正式 run 時，才在 Mac heavy lock 下啟動 scripts/formal_batch.sh（decision 008 §2、014）。
#
# 每 60 秒檢查一次：heavy lock 沒人持有、背景沒有 heavy workload（014）、1 分鐘 load < 4.0；連續 3 次都成立才啟動。
# 批次以新的 session（setsid）啟動，不受啟動它的終端或 Claude session 結束影響。lock 被別人先拿走時不搶占，回去繼續等。
#
# 用法：./scripts/formal_launcher.sh <batch log> <最長等待秒數> [--powermode N] [--skip-idle] <config>...
# --powermode N：另外要求 `pmset -g` 的 powermode 是 N 才算可以開始，並傳給 formal_batch.sh 在每份 config 前再確認（decision 016）。
# 結束：啟動成功印出 LAUNCHED（exit 0）；等超過上限印出 GAVE_UP（exit 2）。批次本身的進度看 <batch log>。
set -u
cd "$(dirname "$0")/.."
PYTHON=${FORMAL_PYTHON:-.venv/bin/python}
PMSET=${FORMAL_PMSET:-pmset}
POLL_S=${FORMAL_POLL_S:-60}
LOCK_DIR=/tmp/ai-portfolio-mac-heavy.lock
if [ "$#" -lt 3 ]; then
  echo "usage: $0 <batch log> <max wait seconds> [--powermode N] [--skip-idle] <config>..." >&2
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
  quiet=$("$PYTHON" - "$LOCK_DIR" <<'PY'
import os
import sys

from benchmarks import environment, protocol

lock = os.path.exists(sys.argv[1])
heavy = environment.background_processes(exclude=environment.ancestor_pids(os.getpid()))
load = os.getloadavg()[0]
if not lock and not heavy and load < protocol.MAX_LOAD_1MIN:
    print("yes")
else:
    print(f"no lock={lock} load={load:.1f} heavy={[h['command'].split('/')[-1] for h in heavy][:3]}")
PY
)
  if [ -n "$powermode" ] && [ "$quiet" = "yes" ]; then
    current=$("$PMSET" -g | awk '$1 == "powermode" {print $2}')
    [ "$current" = "$powermode" ] || quiet="no powermode=$current expected=$powermode"
  fi
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
