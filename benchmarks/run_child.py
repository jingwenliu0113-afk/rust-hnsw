"""benchmark 的子 process：在全新的 interpreter 中量測一個（build、實作）（decision 008 §5、§6）。

由 `benchmarks.run` 以 `python -m benchmarks.run_child <spec.json>` 啟動，不直接使用。
- 只 import 受測的那一個 library，而且在記錄 baseline 與開始計時之前完成。
- 不載入 ground truth、不計算 recall（由父 process 計算，008 §8）。
- 每完成一個 ef 點就在 JSONL append 一行並 flush；crash 前完成的點因此會保留。
- 最後（所有計時點與 save_native 之後）才做 relief 診斷，記錄 after_relief（008-v5，decision 019 §4）。
"""

import ctypes
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks import protocol
from benchmarks.environment import phys_footprint_bytes, ps_rss_bytes, ru_maxrss_bytes, task_vm_info
from benchmarks.interface import IncompleteResultsError, check_search_result

MEASURED_LIBRARIES = ("rust_hnsw", "hnswlib", "faiss")

perf_counter_ns = time.perf_counter_ns


class _Log:
    def __init__(self, path: Path):
        self._file = open(path, "a", encoding="utf-8")

    def write(self, kind: str, **fields) -> None:
        # flush 就夠了：process crash 後，已寫入 kernel 的內容仍會留在檔案中。
        self._file.write(json.dumps({"type": kind, "time": time.time(), **fields}) + "\n")
        self._file.flush()


def _resolve(path: str):
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def _cpu_split(fn):
    """執行 `fn`，回傳 (結果, 其他 thread 的 CPU 秒數, 整個 process 的 CPU 秒數)（007 §6）。"""
    process_before, thread_before = time.process_time(), time.thread_time()
    result = fn()
    process_used = time.process_time() - process_before
    return result, process_used - (time.thread_time() - thread_before), process_used


def _memory() -> dict:
    # phys_footprint 是 index memory 的主要指標（014）；rss 為輔助。task_vm_info 的 reusable / compressed 說明兩者的差異，
    # compressed_lifetime 的增量用於 memory 標記（014 補記）。三個實作在同樣的時點以同一段程式量測。
    vm = task_vm_info()
    return {"rss_bytes": ps_rss_bytes(), "ru_maxrss_bytes": ru_maxrss_bytes(), "phys_footprint_bytes": phys_footprint_bytes(),
            "reusable_bytes": vm["reusable_bytes"], "compressed_bytes": vm["compressed_bytes"],
            "compressed_lifetime_bytes": vm["compressed_lifetime_bytes"]}


def _pressure_relief() -> None:
    """請 malloc 把所有 zone 中已釋放、可回收的 page 還給系統（019 §4；ctypes 寫法沿用 scripts/probe_reusable_memory.py）。

    只在子 process 最後呼叫一次（所有計時點與 save_native 之後），供 after_relief 的診斷讀值（M_relief）。
    回傳值不記錄：014 的量法看的是之後 `_memory()` 的讀數。
    """
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    libc.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.malloc_zone_pressure_relief.restype = ctypes.c_size_t
    libc.malloc_zone_pressure_relief(None, 0)


def _percentiles(samples) -> dict:
    method = protocol.PERCENTILE_METHOD
    return {
        "p50_ns": float(np.percentile(samples, 50, method=method)),
        "p95_ns": float(np.percentile(samples, 95, method=method)),
    }


def _empty(view, k):
    return None


def python_loop(view) -> tuple[dict, np.ndarray]:
    """與 latency 相同的計時方式呼叫空函式（008 §6.2，只作參考）。"""
    samples = np.empty(protocol.PYTHON_LOOP["samples"], dtype=np.int64)
    for i in range(samples.shape[0]):
        start = perf_counter_ns()
        _empty(view, protocol.K)
        samples[i] = perf_counter_ns() - start
    calls = protocol.PYTHON_LOOP["mean_calls"]
    start = perf_counter_ns()
    for _ in range(calls):
        _empty(view, protocol.K)
    mean_ns = (perf_counter_ns() - start) / calls
    return {"samples": int(samples.shape[0]), "mean_calls": calls, "mean_ns": mean_ns, **_percentiles(samples)}, samples


def null_call(adapter_cls, params: dict, data: np.ndarray, queries: np.ndarray) -> tuple[dict, np.ndarray]:
    """n = 1 的 index，k = 1、ef = 1：只剩 1 次 d 維距離計算，不含 graph traversal（008 §6.4）。"""
    spec = protocol.NULL_CALL
    adapter = adapter_cls(**params)
    adapter.build(np.ascontiguousarray(data[: spec["n"]]))
    adapter.set_ef_search(spec["ef"])
    view = queries[0:1]
    for _ in range(spec["warmup"]):
        adapter.search(view, spec["k"])
    samples = np.empty(spec["samples"], dtype=np.int64)
    last = None
    for i in range(samples.shape[0]):
        start = perf_counter_ns()
        last = adapter.search(view, spec["k"])
        samples[i] = perf_counter_ns() - start
    if (last[0] < 0).any():
        raise RuntimeError("null call returned padding")
    record = {**spec, "dim": int(data.shape[1]), **_percentiles(samples)}
    record["samples"] = int(samples.shape[0])
    return record, samples


def measure_point(adapter, queries: np.ndarray, views: list) -> tuple[dict, dict]:
    """一個 ef 點的 k = 10 量測（008 §6.7）。回傳 (摘要, 原始陣列)。"""
    k, nq = protocol.K, queries.shape[0]
    arrays: dict[str, np.ndarray] = {}
    summary: dict = {"status": "ok"}
    try:
        warm_ids, warm_distances = adapter.search(queries, k)
        check_search_result(warm_ids, warm_distances, nq, k)

        qps_ns = np.empty(protocol.QPS_REPEATS, dtype=np.int64)
        repeat_ids = []
        latency_ns = np.empty((protocol.LATENCY_REPEATS, nq), dtype=np.int64)
        single_ids = np.empty((nq, k), dtype=np.int64)

        def qps_section():
            for r in range(protocol.QPS_REPEATS):
                start = perf_counter_ns()
                ids, _ = adapter.search(queries, k)
                qps_ns[r] = perf_counter_ns() - start
                repeat_ids.append(ids)

        def latency_section():
            for i in range(protocol.LATENCY_WARMUP_QUERIES):
                adapter.search(views[i], k)
            for r in range(protocol.LATENCY_REPEATS):
                for i in range(nq):
                    start = perf_counter_ns()
                    ids, _ = adapter.search(views[i], k)
                    latency_ns[r, i] = perf_counter_ns() - start
                    if r == 0:
                        single_ids[i] = ids[0]

        gc.collect()
        gc.disable()
        try:
            _, qps_other, qps_process = _cpu_split(qps_section)
            _, latency_other, latency_process = _cpu_split(latency_section)
        finally:
            gc.enable()

        first = repeat_ids[0]
        differs = np.zeros(nq, dtype=bool)
        for ids in [warm_ids, *repeat_ids[1:]]:
            differs |= (ids != first).any(axis=1)
        summary.update(
            repeat_mismatch_queries=int(differs.sum()),
            batch_vs_single_mismatch_queries=int((single_ids != first).any(axis=1).sum()),
            other_thread_cpu={
                "qps_other_s": qps_other,
                "qps_process_s": qps_process,
                "latency_other_s": latency_other,
                "latency_process_s": latency_process,
            },
        )
        arrays.update(ids_k10=first, qps_ns=qps_ns, latency_ns=latency_ns)
    except IncompleteResultsError:
        # 整批失敗（hnswlib，007 §3）：不保留任何計時，不重試、不 fallback。
        summary = {"status": "incomplete"}
    return summary, arrays


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    spec = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    log = _Log(Path(spec["jsonl_path"]))
    log.write("start", pid=os.getpid())

    # 1. 只 import 受測的 library，之後才載入資料。
    importlib.import_module(spec["library"])
    adapter_cls = _resolve(spec["adapter"])
    params = spec["params"]
    data = np.load(spec["data_path"])
    queries = np.load(spec["queries_path"])
    for name, array in (("data", data), ("queries", queries)):
        if array.dtype != np.float32 or not array.flags.c_contiguous or array.ndim != 2:
            raise ValueError(f"{name} must be a C-contiguous 2-D float32 array")
    views = [queries[i : i + 1] for i in range(queries.shape[0])]

    # 2. Python 迴圈成本；3. library warm-up；4. 空呼叫成本（都在 baseline 之前）。
    loop_record, loop_samples = python_loop(views[0])
    warm = adapter_cls(**params)
    warm.build(np.ascontiguousarray(data[: protocol.LIBRARY_WARMUP_ROWS]))
    warm.set_ef_search(1)
    warm.search(views[0], 1)
    del warm
    null_record, null_samples = null_call(adapter_cls, params, data, queries)
    np.savez(spec["samples_path"], python_loop_ns=loop_samples, null_call_ns=null_samples)

    # 5. baseline：受測 library 已在 sys.modules，另外兩個不在。
    gc.collect()
    loaded = sorted({spec["library"], *MEASURED_LIBRARIES} & set(sys.modules))
    if loaded != [spec["library"]]:
        raise RuntimeError(f"expected only {spec['library']} to be imported, found {loaded}")
    baseline = _memory()
    log.write("prelude", python_loop=loop_record, null_call=null_record, loaded_libraries=loaded, baseline=baseline)

    # 6. build。
    adapter = adapter_cls(**params)
    build_ns = 0

    def timed_build():
        nonlocal build_ns
        start = perf_counter_ns()
        adapter.build(data)
        build_ns = perf_counter_ns() - start

    loadavg_start = list(os.getloadavg())
    _, build_other, build_process = _cpu_split(timed_build)
    peak_rss = ru_maxrss_bytes()
    after_build = _memory()
    log.write(
        "build",
        build_ns=build_ns,
        other_thread_cpu={"other_s": build_other, "process_s": build_process},
        peak_rss_bytes=peak_rss,
        after_build=after_build,
        loadavg_start=loadavg_start,
        loadavg_end=list(os.getloadavg()),
    )

    # 7. 每個 ef 點。
    raw_dir = Path(spec["raw_dir"])
    for ef in spec["ef_search"]:
        adapter.set_ef_search(ef)
        start_time, loadavg_start = time.time(), list(os.getloadavg())
        k10, arrays = measure_point(adapter, queries, views)
        # Recall@1 的來源：同一個 ef 再做一次不計時的 k = 1 整批 search（008 §7）。
        try:
            ids_k1, distances_k1 = adapter.search(queries, protocol.RECALL_AT_1_K)
            check_search_result(ids_k1, distances_k1, queries.shape[0], protocol.RECALL_AT_1_K)
            k1 = {"status": "ok"}
            arrays["ids_k1"] = ids_k1
        except IncompleteResultsError:
            k1 = {"status": "incomplete"}
        raw_file = raw_dir / f"ef{ef}.npz"
        np.savez(raw_file, **arrays)
        log.write(
            "point",
            ef_search=ef,
            start_time=start_time,
            end_time=time.time(),
            loadavg_start=loadavg_start,
            loadavg_end=list(os.getloadavg()),
            k10=k10,
            k1=k1,
            raw_file=str(raw_file),
        )

    # 8. 搜尋之後的高水位（只作參考）；9. 原生序列化大小（最後才做，避免 I/O 落在計時區間）。
    #    adapter 沒有 save_native（019 的 rust-hnsw-fp32g、rust-hnsw-sq8：不新增檔案格式）時記為 null（019 §4）。
    peak_after_search = ru_maxrss_bytes()
    serialized_bytes = None
    save_native = getattr(adapter, "save_native", None)
    if save_native is not None:
        index_path = Path(spec["scratch_dir"]) / f"{spec['child_id']}.index"
        save_native(str(index_path))
        serialized_bytes = index_path.stat().st_size
        index_path.unlink()
    # 10. relief 診斷（019 §4，008-v5）：所有計時點與 save_native 之後才請 allocator 釋放可回收的 page，再量一次記憶體。
    #     計時與 014 讀值所處的 allocator 狀態因此不變；所有實作都走這同一段程式。index 仍然存活，所以這是 M_relief。
    _pressure_relief()
    after_relief = _memory()
    log.write("end", peak_rss_after_search_bytes=peak_after_search, serialized_bytes=serialized_bytes,
              after_relief=after_relief)
    return 0


if __name__ == "__main__":
    sys.exit(main())
