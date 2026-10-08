"""重現 decision 008 §8 的 index memory 量測問題：macOS 上釋放的大型配置仍算在 rss，但不算 phys_footprint。

用法（在 repo root；需要已準備好的 glove-100-angular）：
    PYTHONPATH=. .venv/bin/python scripts/probe_reusable_memory.py rust
    PYTHONPATH=. .venv/bin/python scripts/probe_reusable_memory.py numpy

- rust：在 16 筆的小 index 上一次查詢全部 1,183,514 筆；`search_batch` 經過 `unit_rows`，建立再釋放一份與資料同大小（約 451 MiB）的正規化副本。
  `HnswIndex.add` 也經過同一個函式，所以 build 結束時同樣留下一份已釋放的副本。
- numpy：對照組，複製再釋放同大小的陣列。
- 印出操作後、以及呼叫 `malloc_zone_pressure_relief` 之後，相對於操作前的 rss 與 phys_footprint 增量。
2026-10-09 實測：兩者操作後 Δrss 約 +451–470 MiB、Δfootprint 約 0–14 MiB；pressure relief 之後 Δrss 回到 0–19 MiB。
"""

import ctypes
import gc
import sys

import numpy as np

from benchmarks.environment import phys_footprint_bytes, ps_rss_bytes

MiB = 2**20


def memory() -> tuple[int, int]:
    return ps_rss_bytes(), phys_footprint_bytes()


def main(which: str) -> None:
    x = np.load("datasets/glove-100-angular/train.npy")
    gc.collect()
    rss0, fp0 = memory()
    if which == "rust":
        import rust_hnsw

        index = rust_hnsw.HnswIndex(x.shape[1], m=2, ef_construction=2, seed=1)
        index.add(np.ascontiguousarray(x[:16]))
        index.search(x, 1, 1)
    elif which == "numpy":
        copy = x.copy()
        del copy
    else:
        raise SystemExit("usage: probe_reusable_memory.py rust|numpy")
    rss1, fp1 = memory()
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    libc.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.malloc_zone_pressure_relief.restype = ctypes.c_size_t
    libc.malloc_zone_pressure_relief(None, 0)
    rss2, fp2 = memory()
    print(f"{which}: after the operation  Δrss = {(rss1 - rss0) / MiB:7.1f} MiB  Δfootprint = {(fp1 - fp0) / MiB:7.1f} MiB")
    print(f"{which}: after pressure relief Δrss = {(rss2 - rss0) / MiB:7.1f} MiB  Δfootprint = {(fp2 - fp0) / MiB:7.1f} MiB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
