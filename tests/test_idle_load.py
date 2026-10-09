"""閒置 load 的基準量測（decision 008 §2 與補記）：解析、彙整與判定；不真的等待 15 分鐘。"""

import json
import os

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
    assert heavy == [{"rule": "pid_mean", "pid": 1, "command": "emulator", "mean_pcpu": 90.0}]


def test_pid_mean_threshold_is_50_percent_inclusive():
    heavy = idle_load.heavy_processes([{1: (50.0, "a"), 2: (49.9, "b")}])
    assert [h["pid"] for h in heavy if h["rule"] == "pid_mean"] == [1]


def test_a_stream_of_short_lived_processes_is_heavy_by_command():
    # 每 2 個樣本換一個 100% 的 rustc：每個 pid 平均很低，但同一個 command 一直占著一個 core。
    samples = [{1000 + i // 2: (100.0, "rustc")} for i in range(41)]
    heavy = idle_load.heavy_processes(samples)
    assert [h for h in heavy if h["rule"] == "pid_mean"] == []
    assert [(h["rule"], h["command"]) for h in heavy] == [("command_mean", "rustc")]


def test_intermittent_work_is_heavy_when_sustained_for_a_minute():
    # 41 個樣本中只有連續 4 個 100%：平均只有約 10%，但持續約 1 分鐘。
    samples = [{7: (100.0 if 10 <= i < 14 else 0.0, "worker")} for i in range(41)]
    heavy = idle_load.heavy_processes(samples)
    assert heavy == [{"rule": "sustained", "pid": 7, "command": "worker", "samples": 4}]
    shorter = [{7: (100.0 if 10 <= i < 13 else 0.0, "worker")} for i in range(41)]
    assert idle_load.heavy_processes(shorter) == []  # 單次短暫尖峰不算


def test_gui_background_below_the_thresholds_is_idle():
    # 桌面程式各自不到 50%、同名合計不到 100%：屬於 formal run 時也存在的背景，不判為重工作。
    samples = [{1: (40.0, "WindowServer"), 2: (35.0, "Claude Helper"), 3: (30.0, "Claude Helper")} for _ in range(41)]
    assert idle_load.heavy_processes(samples) == []


def test_sample_summaries_keep_the_total_and_the_top_processes():
    samples = [{1: (10.0, "a"), 2: (30.0, "b"), 3: (5.0, "self")}]
    (summary,) = idle_load.sample_summaries(samples, exclude=[3])
    assert summary["total_pcpu"] == 40.0
    assert [t["pid"] for t in summary["top"]] == [2, 1]


def test_summary_adds_the_runner_residual_to_p95():
    summary = idle_load.summarize([1.0] * 39 + [3.5, 3.6])
    assert summary["samples"] == 41 and summary["max"] == 3.6
    assert summary["p95_plus_residual"] == pytest.approx(summary["p95"] + 1.0)
    assert summary["threshold"] == protocol.MAX_LOAD_1MIN


def test_summary_uses_the_95th_percentile():
    summary = idle_load.summarize([float(i) for i in range(41)])
    assert (summary["p50"], summary["p95"], summary["max"]) == (20.0, 38.0, 40.0)


@pytest.mark.parametrize("p95, heavy, expected", [
    (2.5, [], "ok"),
    (3.0, [], "ok"),        # 3.0 + 1.0 = 4.0，不超過門檻
    (3.2, [], "exceeds"),
    (1.0, [{"pid": 1}], "not_idle"),
    (3.5, [{"pid": 1}], "not_idle"),  # 有其他重工作時，不得用超過門檻的量測去修訂門檻
])
def test_verdict(p95, heavy, expected):
    summary = {"p95_plus_residual": p95 + idle_load.RUNNER_RESIDUAL, "threshold": 4.0}
    assert idle_load.verdict(summary, heavy) == expected


def test_measure_settles_first_and_keeps_settling_samples_out_of_the_window():
    sleeps, loads = [], iter(range(1000))
    settle, window, processes = idle_load.measure(lambda: next(loads), lambda: {}, sleeps.append)
    assert len(settle) == 20 and len(window) == 41 and len(processes) == 61
    assert settle == list(range(20)) and window[0] == 20
    assert sleeps == [15] * 60


def test_refuses_to_run_without_the_heavy_lock(monkeypatch):
    def no_lock():
        raise environment.PreflightError("owner.txt not found")

    monkeypatch.setattr(environment, "check_lock", no_lock)
    with pytest.raises(SystemExit, match="with-mac-lock"):
        idle_load.main([])


FORMAL_POWER = {"power_source": "AC Power", "powermode": 0, "adapter_watts": 68,
                "adapter": {"manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"},
                "battery": {"percent": 100, "state": "charged"}}


def test_refuses_to_measure_outside_the_formal_power_conditions(monkeypatch):
    monkeypatch.setattr(environment, "check_lock", lambda: "owner")
    monkeypatch.setattr(environment, "power_state", lambda: {**FORMAL_POWER, "powermode": 2})
    with pytest.raises(SystemExit, match="power conditions"):
        idle_load.main([])


def test_main_uses_the_one_minute_load_and_excludes_only_ancestors(tmp_path, monkeypatch):
    me = os.getpid()
    monkeypatch.setattr(environment, "check_lock", lambda: "owner")
    monkeypatch.setattr(environment, "power_state", lambda: dict(FORMAL_POWER))
    monkeypatch.setattr(environment, "snapshot", lambda: {})
    monkeypatch.setattr(environment, "top_processes", lambda: "")
    monkeypatch.setattr(environment, "machine_metadata", lambda: {})
    monkeypatch.setattr(environment, "ancestor_pids", lambda pid: [pid, 7])
    monkeypatch.setattr(idle_load.dataprep, "git_state", lambda: {})
    monkeypatch.setattr(idle_load.os, "getloadavg", lambda: (1.0, 9.0, 9.0))
    monkeypatch.setattr(idle_load.time, "sleep", lambda s: None)
    monkeypatch.setattr(idle_load, "OUTPUT_DIR", tmp_path)

    class Done:
        stdout = f"{me} 99.0 python\n 7 99.0 with-mac-lock\n 42 80.0 emulator\n"

    monkeypatch.setattr(idle_load.subprocess, "run", lambda *a, **k: Done())
    assert idle_load.main([]) == 0
    record = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert record["loads"] == [1.0] * 41 and record["settle_loads"] == [1.0] * 20
    assert record["status"] == "not_idle"
    assert {h.get("pid") for h in record["heavy_processes"]} == {42}
    assert record["samples"][0]["total_pcpu"] == 80.0 and len(record["samples"]) == 61
    assert record["power"]["powermode"] == 0
