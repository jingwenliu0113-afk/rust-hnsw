"""排程用的閒置判斷（benchmarks/machine_quiet.py）：解析 ioreg / memory_pressure / sysctl 與判定。"""

from benchmarks import machine_quiet as mq

IOREG = '''
  | {
  |   "PerformanceStatistics" = {"In use system memory"=9446899712,"Device Utilization %"=98,"Renderer Utilization %"=93}
  | }
'''
QUIET = {"lock": False, "heavy": [], "load": 1.2, "gpu_pct": 3, "memory_free_pct": 62, "swap_used_mb": 0.0, "powermode": 2}


def test_parsers_read_the_macos_tools():
    assert mq.parse_gpu_utilization(IOREG) == 98
    assert mq.parse_gpu_utilization(IOREG + IOREG.replace("=98", "=12")) == 98  # 多個 GPU 取最高
    assert mq.parse_gpu_utilization("nothing") is None
    assert mq.parse_memory_free("The system has 51539607552\nSystem-wide memory free percentage: 62%\n") == 62
    assert mq.parse_swap_used_mb("vm.swapusage: total = 2048.00M  used = 1536.25M  free = 511.75M  (encrypted)") == 1536.25


def test_a_quiet_machine_has_no_reasons():
    assert mq.reasons(QUIET, powermode=2) == []


def test_each_condition_blocks_the_start():
    cases = {"lock": True, "heavy": ["ollama"], "load": 4.0, "gpu_pct": 21, "memory_free_pct": 29}
    for key, value in cases.items():
        assert mq.reasons({**QUIET, key: value}, powermode=2), key
    assert mq.reasons({**QUIET, "powermode": 0}, powermode=2) == ["powermode=0 expected=2"]
    assert mq.reasons({**QUIET, "powermode": 0}) == []  # 沒有要求電力模式


def test_unreadable_metrics_are_not_quiet():
    # 不能確認 GPU / 記憶體的狀態就不開始。
    assert mq.reasons({**QUIET, "gpu_pct": None}) == ["gpu=None"]
    assert mq.reasons({**QUIET, "memory_free_pct": None}) == ["memory_free=None"]


def test_swap_alone_does_not_block():
    # 已換出的 page 可能很久才回來；run 期間的 swap 增加由 runner 的標記處理。
    assert mq.reasons({**QUIET, "swap_used_mb": 2048.0}) == []


def test_thresholds_are_inclusive_where_documented():
    assert mq.reasons({**QUIET, "gpu_pct": mq.GPU_MAX_PCT}) == []
    assert mq.reasons({**QUIET, "memory_free_pct": mq.MEMORY_FREE_MIN_PCT}) == []
