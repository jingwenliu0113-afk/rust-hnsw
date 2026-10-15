"""benchmark 的協定常數（decision 008）。

這些值在看到任何正式結果之前固定，整份寫入每次 run 的 run.json（002 的要求）。
`tests/test_run_evaluate.py` 斷言它們與 008 相同：改動任何一項都必須另寫 decision log。
"""

from benchmarks.metrics import TIE_EPS

# 008-v2：使用者 2026-10-09 的決定（decision 014）：formal 的電力條件、背景 heavy workload 的 gate、index memory 改以 phys_footprint 為主。
# 008-v3：使用者 2026-10-11 的決定（decision 017）：取消電量下降的作廢規則，電量只記錄。
# 008-v4：使用者 2026-10-15 的決定（decision 018）：Apple 原廠 70W 或 140W 變壓器皆可；同一個 run 內不得更換。
PROTOCOL_VERSION = "008-v4"

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

# 「phys_footprint 增量超過 rss 增量」的容許值：超過代表 build 期間有 page 被壓縮，rss 低估了 index 記憶體（008 §8）。
MEMORY_RELATIVE = 0.02
MEMORY_ABSOLUTE_BYTES = 1 << 20

LOCK_DIR = "/tmp/ai-portfolio-mac-heavy.lock"

# 014：formal run 的電力條件（使用者決定）：自動（0）或高效能（2）電力模式、Apple 原廠變壓器、接 AC。
# 同一個 run 內電力模式與變壓器都不得改變；報告不合併不同電力模式或不同變壓器的 run（014 補記、018）。
FORMAL_POWERMODES = (0, 2)
# 018：014 原本只允許 70W；使用者 2026-10-15 改為 70W 或 140W 皆可。
FORMAL_ADAPTERS = (
    {"manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"},
    {"manufacturer": "Apple Inc.", "name": "140W USB-C Power Adapter"},
)
# 017（使用者決定）：取消 014 的電量下降規則（接 AC 時 macOS 充滿後讓電量回落幾點是正常的電池管理，不影響 CPU 速度）；
# 電量照常記錄（power_samples），不再是 run 是否有效的條件。協定版本因此改為 008-v3。
# 014：背景 heavy workload（自己、祖先與受測子 process 以外；%CPU 以 100 = 一個 core）：
# - gate：子 process 啟動前，ps 的瞬間 %CPU ≥ BACKGROUND_PCPU 就等待（只會延後啟動）；
# - 子 process 期間：每 BACKGROUND_POLL_S 秒讀一次累積 CPU 時間，以區間平均判定（014 補記），任一成立就標記 background：
#   同一 pid 整段平均 ≥ BACKGROUND_PCPU、同一 command 合計整段平均 ≥ BACKGROUND_COMMAND_PCPU、
#   同一 pid 連續 BACKGROUND_SUSTAINED_INTERVALS 個區間平均 ≥ BACKGROUND_PCPU。
BACKGROUND_PCPU = 50.0
BACKGROUND_COMMAND_PCPU = 100.0
BACKGROUND_SUSTAINED_INTERVALS = 4
BACKGROUND_POLL_S = 15
# 014：index memory 的主要指標是 phys_footprint 的增量；rss 的增量保留為輔助指標。
INDEX_MEMORY_PRIMARY = "index_footprint_bytes"
INDEX_MEMORY_SUPPLEMENTARY = "index_memory_bytes"


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
        "memory_tolerance": {"relative": MEMORY_RELATIVE, "absolute_bytes": MEMORY_ABSOLUTE_BYTES},
        "formal_powermodes": list(FORMAL_POWERMODES),
        "formal_adapters": [dict(a) for a in FORMAL_ADAPTERS],
        "battery_rule": "recorded only (decision 017)",
        "background_pcpu": BACKGROUND_PCPU,
        "background_command_pcpu": BACKGROUND_COMMAND_PCPU,
        "background_sustained_intervals": BACKGROUND_SUSTAINED_INTERVALS,
        "background_poll_s": BACKGROUND_POLL_S,
        "index_memory_primary": INDEX_MEMORY_PRIMARY,
        "index_memory_supplementary": INDEX_MEMORY_SUPPLEMENTARY,
        "tie_eps": TIE_EPS,
    }


def exceeds_single_thread_tolerance(other_thread_cpu_s: float, process_cpu_s: float) -> bool:
    return other_thread_cpu_s > SINGLE_THREAD_RELATIVE * process_cpu_s + SINGLE_THREAD_ABSOLUTE_S


def exceeds_memory_tolerance(footprint_delta: int, rss_delta: int) -> bool:
    return footprint_delta - rss_delta > MEMORY_RELATIVE * footprint_delta + MEMORY_ABSOLUTE_BYTES


def exceeds_compression_tolerance(compressed_delta: int, footprint_delta: int) -> bool:
    """build 期間這個 process 累計被壓縮的量超過 2% × footprint 增量 + 1 MiB（014 補記的 memory 標記）。"""
    return compressed_delta > MEMORY_RELATIVE * max(footprint_delta, 0) + MEMORY_ABSOLUTE_BYTES
