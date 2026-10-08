"""profiling 工作負載產生器的 smoke 測試（不量任何效能數字）。"""

import subprocess
import sys

import numpy as np
import pytest


def run_driver(*args):
    return subprocess.run(
        [sys.executable, "-m", "benchmarks.profile", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("impl", ["rust-hnsw", "hnswlib", "faiss-hnsw"])
@pytest.mark.parametrize("phase", ["build", "search"])
def test_driver_announces_ready_and_exits(impl, phase):
    result = run_driver(
        "--impl", impl, "--phase", phase, "--seconds", "0.2",
        "--synthetic", "500,16,50", "--m", "8", "--ef-construction", "32",
        "--seed", "1", "--ef", "32", "--k", "10",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("READY "), result.stdout


def test_driver_reads_npy_files(tmp_path):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "data.npy", rng.standard_normal((300, 8)).astype(np.float32))
    np.save(tmp_path / "queries.npy", rng.standard_normal((20, 8)).astype(np.float32))
    result = run_driver(
        "--impl", "rust-hnsw", "--phase", "search", "--seconds", "0.1",
        "--data", str(tmp_path / "data.npy"), "--queries", str(tmp_path / "queries.npy"),
        "--m", "8", "--ef-construction", "32", "--seed", "1", "--ef", "32", "--k", "10",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("READY ")


def test_driver_requires_explicit_params():
    # 與 007 一致：不提供預設的 M / efConstruction / seed。
    result = run_driver("--impl", "rust-hnsw", "--phase", "build", "--seconds", "0.1", "--synthetic", "100,4,10")
    assert result.returncode != 0
    assert "--m" in result.stderr
