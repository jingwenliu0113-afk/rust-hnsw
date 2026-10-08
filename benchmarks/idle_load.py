"""閒置 load 的基準量測（decision 008 §2）：第一次 formal run 之前，在 heavy lock 下量 10 分鐘的 1 分鐘 load average。

用法（必須在 heavy lock 下，與 formal run 相同的條件）：
    ./tools/with-mac-lock .venv/bin/python -m benchmarks.idle_load

- 先等 5 分鐘的預備期（1 分鐘 load 的時間常數約 60 s，殘留 ≤ e^-5），讓量測前的工作（例如 test.sh）衰減掉；預備期照常記錄，不計入 summary。
- 接著每 15 秒記錄一次 1 分鐘 load average，共 10 分鐘（含起點 41 個樣本），同時記錄每個 process 的 CPU 使用率。
- 結果：P50 / P95 / 最大值，以及「P95 + 1.0」（runner 前一個子 process 殘留的 load）是否超過 protocol 的門檻。
- 008 §2 要求「沒有其他重工作」（預備期與量測期都算；以 ps 的 %CPU，100 = 一個 core；祖先 process 除外），以下任一成立就判 not_idle，
  不能用來確認門檻（008 補記）：
  1. 同一個 pid 的平均 %CPU ≥ 50；
  2. 同一個 command（例如不斷換 pid 的 rustc、同一程式的多個 worker）合計的平均 %CPU ≥ 100；
  3. 同一個 pid 連續 4 個樣本（約 1 分鐘）%CPU ≥ 50。
- 每個樣本記錄非祖先 process 的 %CPU 合計與前 5 名，exceeds 必須先以它們確認沒有漏判的工作，才能作為修訂門檻的依據。
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
SETTLE_S = 300
# runner 自己的前一個子 process 在 1 分鐘 load 中殘留的量（008 §2）。
RUNNER_RESIDUAL = 1.0
# 「其他重工作」的判定（008 補記；ps 的 %CPU，100 = 一個 core）。
HEAVY_PCPU = 50.0
HEAVY_COMMAND_PCPU = 100.0
SUSTAINED_SAMPLES = 4
TOP_PER_SAMPLE = 5
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


def heavy_processes(samples: list[dict[int, tuple[float, str]]], exclude=()) -> list[dict]:
    """其他重工作（008 補記）：pid 平均 ≥ HEAVY_PCPU、command 合計平均 ≥ HEAVY_COMMAND_PCPU，
    或同一 pid 連續 SUSTAINED_SAMPLES 個樣本 ≥ HEAVY_PCPU。沒出現的樣本算 0。"""
    exclude = set(exclude)
    count = len(samples)
    by_pid: dict[int, float] = {}
    by_command: dict[str, float] = {}
    names: dict[int, str] = {}
    streak: dict[int, int] = {}
    longest: dict[int, int] = {}
    for sample in samples:
        present = set()
        for pid, (pcpu, comm) in sample.items():
            if pid in exclude:
                continue
            by_pid[pid] = by_pid.get(pid, 0.0) + pcpu
            by_command[comm] = by_command.get(comm, 0.0) + pcpu
            names[pid] = comm
            if pcpu >= HEAVY_PCPU:
                present.add(pid)
                streak[pid] = streak.get(pid, 0) + 1
                longest[pid] = max(longest.get(pid, 0), streak[pid])
        for pid in list(streak):
            if pid not in present:
                streak[pid] = 0
    heavy = [{"rule": "pid_mean", "pid": pid, "command": names[pid], "mean_pcpu": total / count}
             for pid, total in by_pid.items() if total / count >= HEAVY_PCPU]
    heavy += [{"rule": "command_mean", "command": comm, "mean_pcpu": total / count}
              for comm, total in by_command.items() if total / count >= HEAVY_COMMAND_PCPU]
    heavy += [{"rule": "sustained", "pid": pid, "command": names[pid], "samples": run}
              for pid, run in longest.items() if run >= SUSTAINED_SAMPLES]
    return sorted(heavy, key=lambda h: (h["rule"], -h.get("mean_pcpu", h.get("samples", 0))))


def sample_summaries(samples: list[dict[int, tuple[float, str]]], exclude=()) -> list[dict]:
    """每個樣本：非祖先 process 的 %CPU 合計與前 TOP_PER_SAMPLE 名，讓 exceeds 可以事後審查。"""
    exclude = set(exclude)
    summaries = []
    for sample in samples:
        others = sorted(((pcpu, pid, comm) for pid, (pcpu, comm) in sample.items() if pid not in exclude), reverse=True)
        summaries.append({"total_pcpu": sum(pcpu for pcpu, _, _ in others),
                          "top": [{"pid": pid, "command": comm, "pcpu": pcpu} for pcpu, pid, comm in others[:TOP_PER_SAMPLE]]})
    return summaries


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


def measure(read_load, read_ps, sleep, settle_s: float = SETTLE_S, duration_s: float = DURATION_S,
            poll_s: float = POLL_S):
    """預備期每 `poll_s` 秒取樣 settle_s / poll_s 個，量測期含起點取 duration_s / poll_s + 1 個。

    回傳 (預備期 loads, 量測期 loads, 全部 ps 樣本)；summary 只用量測期的 loads，重工作的判定用全部 ps 樣本。
    """
    settle, loads, processes = [], [], []
    for _ in range(int(settle_s // poll_s)):
        settle.append(read_load())
        processes.append(read_ps())
        sleep(poll_s)
    for i in range(int(duration_s // poll_s) + 1):
        if i:
            sleep(poll_s)
        loads.append(read_load())
        processes.append(read_ps())
    return settle, loads, processes


def main(argv=None) -> int:
    try:
        owner = environment.check_lock()
    except environment.PreflightError as err:
        raise SystemExit(f"run the idle-load measurement under ./tools/with-mac-lock: {err}") from None
    start = environment.snapshot()
    top_start = environment.top_processes()
    settle, loads, processes = measure(
        lambda: os.getloadavg()[0],
        lambda: parse_ps(subprocess.run(["ps", "-Ao", "pid=,pcpu=,comm="], check=True, capture_output=True, text=True).stdout),
        time.sleep,
    )
    ancestors = environment.ancestor_pids(os.getpid())
    heavy = heavy_processes(processes, exclude=ancestors)
    summary = summarize(loads)
    record = {
        "decision": "docs/decisions/008-benchmark-runner.md §2",
        "status": verdict(summary, heavy),
        "summary": summary,
        "loads": loads,
        "settle_loads": settle,
        "settle_s": SETTLE_S,
        "poll_s": POLL_S,
        "heavy_rules": {"pid_mean_pcpu": HEAVY_PCPU, "command_mean_pcpu": HEAVY_COMMAND_PCPU,
                        "sustained_samples": SUSTAINED_SAMPLES, "sustained_pcpu": HEAVY_PCPU},
        "heavy_processes": heavy,
        "samples": sample_summaries(processes, exclude=ancestors),
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
          f"heavy: {[(h['rule'], h['command']) for h in heavy]}")
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
