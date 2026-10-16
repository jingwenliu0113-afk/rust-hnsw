"""排程用的「機器是否閒置」判斷：等待腳本（scripts/formal_launcher.sh）在啟動正式 run 前呼叫。

條件（全部成立才印出 `yes`，否則印出 `no` 與原因）：
- Mac heavy lock 沒人持有；
- 背景沒有 heavy workload（decision 014：自己與祖先以外，ps 瞬間 %CPU ≥ 50 的 process）、1 分鐘 load < 4.0（008 §2）；
- GPU 的 Device Utilization ≤ GPU_MAX_PCT（`ioreg -c IOAccelerator`；使用者 2026-10-16 要求開跑前也看 GPU）；
- 系統記憶體 free percentage ≥ MEMORY_FREE_MIN_PCT（`memory_pressure -Q`）。swap 的使用量只列出（`sysctl vm.swapusage`）：
  已換出的 page 可能在壓力解除後很久才回來，以它為條件會無限期等待；run 期間 swap 增加由 runner 的 `swap` 標記處理（008 §9）；
- `--powermode N` 給定時，pmset 的 powermode 是 N（decision 016）。

這是開始的時機，不是 run 是否有效的條件；有效性仍由 runner 的協定判定（gate、hygiene 標記）。

用法：python -m benchmarks.machine_quiet [--powermode N]
"""

import argparse
import os
import re
import subprocess

from benchmarks import environment, protocol

GPU_MAX_PCT = 20
MEMORY_FREE_MIN_PCT = 30
_GPU = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
_MEMORY_FREE = re.compile(r"System-wide memory free percentage:\s*(\d+)%")
_SWAP_USED = re.compile(r"used\s*=\s*([\d.]+)M")


def parse_gpu_utilization(ioreg_text: str) -> int | None:
    """所有 GPU 中最高的 Device Utilization %；讀不到時 None。"""
    values = [int(v) for v in _GPU.findall(ioreg_text)]
    return max(values) if values else None


def parse_memory_free(text: str) -> int | None:
    match = _MEMORY_FREE.search(text)
    return int(match.group(1)) if match else None


def parse_swap_used_mb(text: str) -> float | None:
    match = _SWAP_USED.search(text)
    return float(match.group(1)) if match else None


def _run(*args) -> str:
    try:
        return subprocess.run(list(args), check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def readings(powermode_needed: bool) -> dict:
    return {
        "lock": os.path.exists(protocol.LOCK_DIR),
        "heavy": [h["command"].split("/")[-1] for h in
                  environment.background_processes(exclude=environment.ancestor_pids(os.getpid()))][:3],
        "load": os.getloadavg()[0],
        "gpu_pct": parse_gpu_utilization(_run("ioreg", "-r", "-d", "1", "-c", "IOAccelerator")),
        "memory_free_pct": parse_memory_free(_run("memory_pressure", "-Q")),
        "swap_used_mb": parse_swap_used_mb(_run("sysctl", "vm.swapusage")),
        "powermode": environment.power_state()["powermode"] if powermode_needed else None,
    }


def reasons(r: dict, powermode: int | None = None) -> list[str]:
    """不閒置的原因；空清單代表可以開始。讀不到的 GPU / 記憶體指標視為不閒置（不能確認就不開始）。"""
    found = []
    if r["lock"]:
        found.append("lock held")
    if r["heavy"]:
        found.append(f"heavy={r['heavy']}")
    if r["load"] >= protocol.MAX_LOAD_1MIN:
        found.append(f"load={r['load']:.1f}")
    if r["gpu_pct"] is None or r["gpu_pct"] > GPU_MAX_PCT:
        found.append(f"gpu={r['gpu_pct']}")
    if r["memory_free_pct"] is None or r["memory_free_pct"] < MEMORY_FREE_MIN_PCT:
        found.append(f"memory_free={r['memory_free_pct']}")
    if powermode is not None and r["powermode"] != powermode:
        found.append(f"powermode={r['powermode']} expected={powermode}")
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--powermode", type=int)
    args = parser.parse_args(argv)
    r = readings(args.powermode is not None)
    found = reasons(r, args.powermode)
    print("yes" if not found else "no " + " ".join(found) + f" (swap_used_mb={r['swap_used_mb']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
