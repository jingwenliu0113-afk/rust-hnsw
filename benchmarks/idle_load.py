"""閒置 load 的基準量測（decision 008 §2）：第一次 formal run 之前，在 heavy lock 下量 10 分鐘的 1 分鐘 load average。

用法（必須在 heavy lock 下，與 formal run 相同的條件）：
    ./tools/with-mac-lock .venv/bin/python -m benchmarks.idle_load

- 每 15 秒記錄一次 1 分鐘 load average，共 10 分鐘（含起點 41 個樣本），同時記錄每個 process 的 CPU 使用率。
- 結果：P50 / P95 / 最大值，以及「P95 + 1.0」（runner 前一個子 process 殘留的 load）是否超過 protocol 的門檻。
- 008 §2 要求「沒有其他重工作」：量測期間任何其他 process 的平均 CPU 使用率 ≥ 50% 時，結果標為 not_idle，不能用來確認門檻。
- 只記錄、不修改門檻。超過時依 008 §2 先另寫 decision 修訂門檻，才能跑 formal。
- 輸出 `results/idle_load/<UTC 時間>.json`（results/ 不進 Git）。
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks import dataprep, environment, protocol

DURATION_S = 600
POLL_S = 15
# runner 自己的前一個子 process 在 1 分鐘 load 中殘留的量（008 §2）。
RUNNER_RESIDUAL = 1.0
# 「其他重工作」：量測期間平均 CPU 使用率（ps 的 %CPU，100 = 一個 core）達到這個值的其他 process。
HEAVY_PCPU = 50.0
OUTPUT_DIR = dataprep.REPO / "results" / "idle_load"


def parse_ps(text: str) -> dict[int, tuple[float, str]]:
    """`ps -Ao pid=,pcpu=,comm=` 的輸出 → {pid: (%CPU, command)}。"""
    result = {}
    for line in text.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3:
            try:
                result[int(fields[0])] = (float(fields[1]), fields[2])
            except ValueError:
                continue
    return result


def heavy_processes(samples: list[dict[int, tuple[float, str]]], exclude=(), threshold: float = HEAVY_PCPU) -> list[dict]:
    """每個 pid 在所有樣本中的平均 %CPU（沒出現的樣本算 0）；達到門檻的列出來，由高到低。"""
    totals: dict[int, float] = {}
    names: dict[int, str] = {}
    for sample in samples:
        for pid, (pcpu, comm) in sample.items():
            totals[pid] = totals.get(pid, 0.0) + pcpu
            names[pid] = comm
    heavy = [{"pid": pid, "command": names[pid], "mean_pcpu": total / len(samples)}
             for pid, total in totals.items() if pid not in set(exclude) and total / len(samples) >= threshold]
    return sorted(heavy, key=lambda h: -h["mean_pcpu"])


def summarize(loads: list[float]) -> dict:
    x = np.asarray(loads, dtype=float)
    p95 = float(np.percentile(x, 95, method=protocol.PERCENTILE_METHOD))
    return {
        "samples": len(loads),
        "p50": float(np.percentile(x, 50, method=protocol.PERCENTILE_METHOD)),
        "p95": p95,
        "max": float(x.max()),
        "p95_plus_residual": p95 + RUNNER_RESIDUAL,
        "threshold": protocol.MAX_LOAD_1MIN,
    }


def verdict(summary: dict, heavy: list[dict]) -> str:
    """not_idle：有其他重工作，不能用來確認門檻；exceeds：P95 + 1.0 超過門檻（008 §2：先修訂門檻）；ok：門檻成立。"""
    if heavy:
        return "not_idle"
    return "exceeds" if summary["p95_plus_residual"] > summary["threshold"] else "ok"


def measure(read_load, read_ps, sleep, duration_s: float = DURATION_S, poll_s: float = POLL_S):
    """每 `poll_s` 秒取一次樣本，含起點共 duration_s / poll_s + 1 個。回傳 (loads, ps 樣本)。"""
    loads, processes = [], []
    for i in range(int(duration_s // poll_s) + 1):
        if i:
            sleep(poll_s)
        loads.append(read_load())
        processes.append(read_ps())
    return loads, processes


def main(argv=None) -> int:
    try:
        owner = environment.check_lock()
    except environment.PreflightError as err:
        raise SystemExit(f"run the idle-load measurement under ./tools/with-mac-lock: {err}") from None
    start = environment.snapshot()
    top_start = environment.top_processes()
    loads, processes = measure(
        lambda: os.getloadavg()[0],
        lambda: parse_ps(subprocess.run(["ps", "-Ao", "pid=,pcpu=,comm="], check=True, capture_output=True, text=True).stdout),
        time.sleep,
    )
    heavy = heavy_processes(processes, exclude=environment.ancestor_pids(os.getpid()))
    summary = summarize(loads)
    record = {
        "decision": "docs/decisions/008-benchmark-runner.md §2",
        "status": verdict(summary, heavy),
        "summary": summary,
        "loads": loads,
        "poll_s": POLL_S,
        "heavy_pcpu": HEAVY_PCPU,
        "heavy_processes": heavy,
        "lock_owner": owner,
        "start": {**start, "top_processes": top_start},
        "end": {**environment.snapshot(), "top_processes": environment.top_processes()},
        "git": dataprep.git_state(),
        "machine": environment.machine_metadata(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{record['status']}: p50 {summary['p50']:.2f}, p95 {summary['p95']:.2f}, max {summary['max']:.2f}, "
          f"p95 + {RUNNER_RESIDUAL} = {summary['p95_plus_residual']:.2f} (threshold {summary['threshold']}); "
          f"heavy processes: {[h['command'] for h in heavy]}")
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
