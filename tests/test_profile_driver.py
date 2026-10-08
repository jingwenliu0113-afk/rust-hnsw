"""profiling 工作負載產生器的 smoke 測試（不量任何效能數字）。"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks import datasets, environment, metrics
from benchmarks import profile as profile_module
from test_dataprep import make_npy_dataset

REPO = Path(__file__).resolve().parents[1]


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


class Recording:
    """記錄 build 收到的資料、每次 search 的 ef，以及第一次取樣時 metadata 是否已經寫出。"""

    instances = []
    meta_path = None

    def __init__(self, m, ef_construction, seed):
        self.ef, self.searches, self.data, self.meta_seen = None, [], None, None
        Recording.instances.append(self)

    def build(self, data):
        self.data = np.array(data, copy=True)

    def set_ef_search(self, ef):
        self.ef = ef

    def search(self, queries, k):
        self.searches.append(self.ef)
        if len(self.searches) == 2:  # 第一次是不計時的 recall，第二次是取樣迴圈
            self.meta_seen = Path(Recording.meta_path).exists()
        ids = np.tile(np.arange(k), (queries.shape[0], 1))
        return ids, np.zeros(ids.shape, dtype=np.float32)


def run_recorded(tmp_path, monkeypatch, phase, n):
    make_npy_dataset(tmp_path, frozen_n=(200, 300))
    monkeypatch.setattr(datasets, "REGISTRY", datasets.load_registry(tmp_path))
    monkeypatch.setitem(profile_module.ADAPTERS, "rust-hnsw", Recording)
    Recording.instances = []
    Recording.meta_path = tmp_path / "meta.json"
    args = ["--impl", "rust-hnsw", "--phase", phase, "--seconds", "0.05", "--dataset", "tiny-npy", "--n", str(n),
            "--m", "8", "--ef-construction", "64", "--seed", "1", "--meta", str(Recording.meta_path)]
    if phase == "search":
        args += ["--ef", "37", "--k", "10"]
    profile_module.main(args)
    return json.loads(Recording.meta_path.read_text())


def test_recall_is_measured_at_the_sampled_ef_on_the_runner_subset(tmp_path, monkeypatch):
    record = run_recorded(tmp_path, monkeypatch, "search", 200)
    adapter = Recording.instances[0]
    loaded = datasets.load(datasets.REGISTRY["tiny-npy"])
    assert np.array_equal(adapter.data, loaded.train[:200])  # 與 runner 相同：登記的向量、前 n 列、不重新正規化
    assert set(adapter.searches) == {37}
    gt = np.load(Path(datasets.REGISTRY["tiny-npy"].directory) / "gt_n200.npy")[:, :10]
    expected = metrics.recall_at_k(np.tile(np.arange(10), (gt.shape[0], 1)), gt, 10)
    assert expected < 0.5 and record["recall_at_10"] == pytest.approx(expected)
    assert adapter.meta_seen is True  # 取樣前就寫出（009 補記）


def test_build_phase_writes_the_record_without_recall(tmp_path, monkeypatch):
    record = run_recorded(tmp_path, monkeypatch, "build", 300)
    assert record["dataset"]["n"] == 300 and record["recall_at_10"] is None


@pytest.mark.parametrize("held", [True, False])
def test_lock_state_reports_whether_an_ancestor_holds_the_lock(monkeypatch, held):
    def check():
        if not held:
            raise environment.PreflightError("no lock")
        return "owner text"

    monkeypatch.setattr(environment, "check_lock", check)
    state = profile_module.lock_state()
    assert state["held_by_ancestor"] is held
    assert state.get("owner") == ("owner text" if held else None)


needs_sample = pytest.mark.skipif(shutil.which("sample") is None, reason="macOS `sample` is not available")


def run_script(name, *args):
    return subprocess.run(["./scripts/profile.sh", name, "1", "--", *args], cwd=REPO, capture_output=True,
                          text=True, timeout=300)


@needs_sample
def test_profile_script_samples_after_ready_and_writes_the_record():
    name = f"pytest-profile-{os.getpid()}"
    try:
        result = run_script(name, "--impl", "rust-hnsw", "--phase", "search", "--synthetic", "500,8,50",
                            "--m", "8", "--ef-construction", "32", "--seed", "1", "--ef", "32", "--k", "10")
        assert result.returncode == 0, result.stderr
        record = json.loads((REPO / "results" / "profiles" / f"{name}.json").read_text())
        assert record["args"]["ef"] == 32
        assert (REPO / "results" / "profiles" / f"{name}.top.txt").exists()
    finally:
        for path in (REPO / "results" / "profiles").glob(f"{name}.*"):
            path.unlink()


@needs_sample
def test_profile_script_never_samples_a_workload_that_did_not_reach_ready():
    name = f"pytest-profile-fail-{os.getpid()}"
    try:
        result = run_script(name, "--impl", "rust-hnsw", "--phase", "build", "--dataset", "glove-100-angular",
                            "--m", "8", "--ef-construction", "32", "--seed", "1")  # 沒有 --n：workload 直接失敗
        assert result.returncode != 0
        assert "exited before READY" in result.stderr
        assert not (REPO / "results" / "profiles" / f"{name}.sample.txt").exists()
    finally:
        for path in (REPO / "results" / "profiles").glob(f"{name}.*"):
            path.unlink()
