"""測試共用設定。"""

import pytest

# 測試用的電力狀態：符合 decision 014 的條件。
FORMAL_POWER = {"power_source": "AC Power", "powermode": 0, "adapter_watts": 68,
                "adapter": {"manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"},
                "battery": {"percent": 100, "state": "charged"}}


@pytest.fixture(autouse=True)
def isolated_machine_state(monkeypatch):
    """runner 讀取機器狀態的函式（背景 heavy workload、累積 CPU 時間、電力狀態；decision 014）在測試中預設為
    「沒有其他工作、電力條件成立」，避免結果依機器上其他專案的負載或當下的電力設定而變。
    真實的讀取函式另有專門的測試（tests/test_run_environment.py、tests/test_runner.py 的 real_readers 測試）。"""
    from benchmarks import run

    monkeypatch.setattr(run, "read_background", lambda: [])
    monkeypatch.setattr(run, "read_cpu", lambda: {})
    monkeypatch.setattr(run, "read_power", lambda: dict(FORMAL_POWER))
