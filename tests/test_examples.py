"""examples/ 的 Demo 必須一直能跑（工程改善；Demo 的數字不是研究結果）。"""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quickstart_runs_and_recall_rises_with_ef(capsys):
    result = load("quickstart").main(["--n", "2000", "--dim", "16", "--queries", "50"])
    recalls = [result["recall"][ef] for ef in sorted(result["recall"])]
    assert all(0.0 <= r <= 1.0 for r in recalls) and recalls[-1] >= recalls[0]
    assert result["reload_identical"] is True
    assert "不是研究結果" in capsys.readouterr().out


def test_quickstart_recall_matches_the_benchmark_metric():
    import numpy as np

    from benchmarks import metrics

    quickstart = load("quickstart")
    found = np.array([[1, 2, -1], [4, 5, 6]])
    truth = np.array([[1, 3, 2], [7, 8, 9]])
    assert quickstart.recall_at_k(found, truth, 3) == metrics.recall_at_k(found, truth, 3)
