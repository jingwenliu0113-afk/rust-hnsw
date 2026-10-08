"""benchmark 的協定常數（decision 008）。

這些值在看到任何正式結果之前固定，整份寫入每次 run 的 run.json（002 的要求）。
`tests/test_run_evaluate.py` 斷言它們與 008 相同：改動任何一項都必須另寫 decision log。
"""

from benchmarks.metrics import TIE_EPS

PROTOCOL_VERSION = "008-v1"

# 主要量測的 k；Recall@1 另外以 k = 1 的整批 search 取得（008 §7）。
K = 10
RECALL_AT_1_K = 1

BATCH_WARMUP_CALLS = 1
QPS_REPEATS = 5
LATENCY_WARMUP_QUERIES = 100
LATENCY_REPEATS = 3
# 單筆 warm-up 固定 100 筆，所以 query 數至少 100（不隨資料改變常數）。
MIN_QUERIES = 100

# library warm-up：同參數、前 16 列建 index 並 search 一次，讓 timed build 不含一次性初始化（008 §6.3）。
LIBRARY_WARMUP_ROWS = 16
# 空呼叫：n = 1 的 index，search 只剩 1 次 d 維距離計算（008 §6.4）。
NULL_CALL = {"n": 1, "k": 1, "ef": 1, "warmup": 100, "samples": 1000}
PYTHON_LOOP = {"samples": 1000, "mean_calls": 1_000_000}

PERCENTILE_METHOD = "linear"
GC = "gc.collect() then gc.disable() around the QPS and latency sections"

# 1 分鐘 load average 的門檻（暫定，第一次 formal run 前依閒置基準確認，008 §2）。
MAX_LOAD_1MIN = 4.0
LOAD_WAIT_S = 600
LOAD_POLL_S = 15

# 「其他 thread 的 CPU time」的容許值（007 §6）。
SINGLE_THREAD_RELATIVE = 0.02
SINGLE_THREAD_ABSOLUTE_S = 0.001

LOCK_DIR = "/tmp/ai-portfolio-mac-heavy.lock"


def protocol_record() -> dict:
    return {
        "version": PROTOCOL_VERSION,
        "k": K,
        "recall_at_1_k": RECALL_AT_1_K,
        "batch_warmup_calls": BATCH_WARMUP_CALLS,
        "qps_repeats": QPS_REPEATS,
        "latency_warmup_queries": LATENCY_WARMUP_QUERIES,
        "latency_repeats": LATENCY_REPEATS,
        "min_queries": MIN_QUERIES,
        "library_warmup_rows": LIBRARY_WARMUP_ROWS,
        "null_call": dict(NULL_CALL),
        "python_loop": dict(PYTHON_LOOP),
        "percentile_method": PERCENTILE_METHOD,
        "gc": GC,
        "max_load_1min": MAX_LOAD_1MIN,
        "load_wait_s": LOAD_WAIT_S,
        "load_poll_s": LOAD_POLL_S,
        "single_thread_tolerance": {"relative": SINGLE_THREAD_RELATIVE, "absolute_s": SINGLE_THREAD_ABSOLUTE_S},
        "tie_eps": TIE_EPS,
    }


def exceeds_single_thread_tolerance(other_thread_cpu_s: float, process_cpu_s: float) -> bool:
    return other_thread_cpu_s > SINGLE_THREAD_RELATIVE * process_cpu_s + SINGLE_THREAD_ABSOLUTE_S
