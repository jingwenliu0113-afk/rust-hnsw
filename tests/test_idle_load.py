"""閒置 load 的基準量測（decision 008 §2）：解析、彙整與判定；不真的等待 10 分鐘。"""

import pytest

from benchmarks import environment, idle_load, protocol


def test_ps_output_is_parsed_and_malformed_lines_are_skipped():
    text = "  101  12.5 /usr/bin/python3\n  202 0.0 WindowServer -daemon\nbad line\n  x 1.0 y\n"
    assert idle_load.parse_ps(text) == {101: (12.5, "/usr/bin/python3"), 202: (0.0, "WindowServer -daemon")}


def test_heavy_processes_average_over_all_samples_and_exclude_ancestors():
    samples = [
        {1: (100.0, "emulator"), 2: (90.0, "self"), 3: (60.0, "short burst")},
        {1: (80.0, "emulator"), 2: (90.0, "self")},
        {1: (90.0, "emulator"), 2: (90.0, "self")},
    ]
    heavy = idle_load.heavy_processes(samples, exclude=[2])
    # pid 3 只出現一次：(60 + 0 + 0) / 3 = 20，不算重工作；pid 2 是自己的祖先，排除。
    assert heavy == [{"pid": 1, "command": "emulator", "mean_pcpu": 90.0}]


def test_summary_adds_the_runner_residual_to_p95():
    summary = idle_load.summarize([1.0] * 39 + [3.5, 3.6])
    assert summary["samples"] == 41 and summary["max"] == 3.6
    assert summary["p95_plus_residual"] == pytest.approx(summary["p95"] + 1.0)
    assert summary["threshold"] == protocol.MAX_LOAD_1MIN


@pytest.mark.parametrize("p95, heavy, expected", [
    (2.5, [], "ok"),
    (3.0, [], "ok"),        # 3.0 + 1.0 = 4.0，不超過門檻
    (3.2, [], "exceeds"),
    (1.0, [{"pid": 1}], "not_idle"),
])
def test_verdict(p95, heavy, expected):
    summary = {"p95_plus_residual": p95 + idle_load.RUNNER_RESIDUAL, "threshold": 4.0}
    assert idle_load.verdict(summary, heavy) == expected


def test_measure_takes_a_sample_at_the_start_and_every_poll():
    sleeps, loads = [], iter(range(100))
    result_loads, processes = idle_load.measure(lambda: next(loads), lambda: {}, sleeps.append)
    assert len(result_loads) == 41 and len(processes) == 41
    assert sleeps == [15] * 40


def test_refuses_to_run_without_the_heavy_lock(monkeypatch):
    def no_lock():
        raise environment.PreflightError("owner.txt not found")

    monkeypatch.setattr(environment, "check_lock", no_lock)
    with pytest.raises(SystemExit, match="with-mac-lock"):
        idle_load.main([])
