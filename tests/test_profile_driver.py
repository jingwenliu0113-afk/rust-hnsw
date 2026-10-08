"""profiling 工作負載產生器的 smoke 測試（不量任何效能數字）。"""

import json
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import datasets
from benchmarks import profile as profile_module
from test_dataprep import make_npy_dataset


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


def test_driver_requires_n_with_dataset():
    result = run_driver("--impl", "rust-hnsw", "--phase", "build", "--seconds", "0.1", "--dataset", "x",
                        "--m", "8", "--ef-construction", "32", "--seed", "1")
    assert result.returncode != 0
    assert "--n" in result.stderr


@pytest.mark.parametrize("n, gt_kind", [(300, "frozen"), (200, "computed")])
def test_driver_profiles_a_registered_dataset_and_records_the_context(tmp_path, monkeypatch, capsys, n, gt_kind):
    # 與 runner 相同的登記向量與前 n 列；取樣前寫出 recall、checksum、git、編譯資訊與 lock 狀態（decision 009）。
    make_npy_dataset(tmp_path, frozen_n=(300,))
    monkeypatch.setattr(datasets, "REGISTRY", datasets.load_registry(tmp_path))
    meta = tmp_path / "meta.json"
    profile_module.main([
        "--impl", "rust-hnsw", "--phase", "search", "--seconds", "0.1", "--dataset", "tiny-npy", "--n", str(n),
        "--m", "8", "--ef-construction", "64", "--seed", "1", "--ef", "64", "--k", "10", "--meta", str(meta),
    ])
    assert capsys.readouterr().out.startswith("READY ")
    record = json.loads(meta.read_text())
    assert record["dataset"]["n"] == n and record["dataset"]["gt_source"]["kind"] == gt_kind
    assert set(record["dataset"]["files_sha256"]) == {"train.npy", "queries.npy", "gt_n300.npy"}
    assert 0.9 <= record["recall_at_10"] <= 1.0
    assert record["args"]["ef"] == 64 and len(record["git"]["commit"]) == 40
    assert {"profile", "rustflags"} <= set(record["build_info"])
    assert isinstance(record["lock"]["held_by_ancestor"], bool)


def test_driver_refuses_a_subset_larger_than_the_dataset(tmp_path, monkeypatch):
    make_npy_dataset(tmp_path, frozen_n=(300,))
    monkeypatch.setattr(datasets, "REGISTRY", datasets.load_registry(tmp_path))
    with pytest.raises(SystemExit, match="--n"):
        profile_module.main(["--impl", "rust-hnsw", "--phase", "build", "--seconds", "0.1", "--dataset", "tiny-npy",
                             "--n", "301", "--m", "8", "--ef-construction", "32", "--seed", "1"])
