"""測試共用設定。"""

import pytest


@pytest.fixture(autouse=True)
def no_background_workload(monkeypatch):
    """runner 的背景 heavy workload 檢查（decision 014）在測試中預設為「沒有」，避免結果依機器上其他專案的負載而變。
    專門的測試以 Hooks(background_reader=...) 或直接呼叫 environment.background_processes 覆蓋真實的行為。"""
    from benchmarks import run

    monkeypatch.setattr(run, "read_background", lambda: [])
