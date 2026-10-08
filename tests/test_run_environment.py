"""formal 模式的前置檢查與 hygiene 紀錄（decision 008 §1、§2、§8、§9）。

這些檢查在測試中改用注入的路徑與數值，所以「通過」與「拒絕」兩條路徑都測得到；
真正的 lock、git 狀態與機器負載只在 formal run 時讀取。
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks import datasets, environment
from benchmarks.environment import PreflightError

REPO = Path(__file__).resolve().parents[1]


# ---- lock ----


def write_owner(lock_dir: Path, text: str) -> Path:
    lock_dir.mkdir()
    (lock_dir / "owner.txt").write_text(text, encoding="utf-8")
    return lock_dir


def owner_text(pid) -> str:
    return f"pid={pid}\nstarted_at=2026-10-08T00:00:00+08:00\ncwd=/tmp\ncommand=python -m benchmarks.run\n"


def test_lock_owned_by_self_is_accepted(tmp_path):
    lock = write_owner(tmp_path / "lock", owner_text(os.getpid()))
    assert environment.check_lock(lock).startswith("pid=")


def test_lock_owned_by_an_ancestor_is_accepted(tmp_path):
    lock = write_owner(tmp_path / "lock", owner_text(os.getppid()))
    environment.check_lock(lock)


def test_lock_owned_by_a_live_non_ancestor_is_rejected(tmp_path):
    other = subprocess.Popen(["sleep", "30"])
    try:
        lock = write_owner(tmp_path / "lock", owner_text(other.pid))
        with pytest.raises(PreflightError, match="not an ancestor"):
            environment.check_lock(lock)
    finally:
        other.kill()
        other.wait()


@pytest.mark.parametrize("text", ["started_at=x\n", "pid=abc\n", "pid=\n", "pid=12 13\n"])
def test_malformed_owner_file_is_rejected(tmp_path, text):
    lock = write_owner(tmp_path / "lock", text)
    with pytest.raises(PreflightError, match="pid"):
        environment.check_lock(lock)


def test_missing_lock_is_rejected(tmp_path):
    with pytest.raises(PreflightError, match="with-mac-lock"):
        environment.check_lock(tmp_path / "absent")


def test_lock_path_matches_with_mac_lock():
    script = (REPO / "tools" / "with-mac-lock").read_text(encoding="utf-8")
    match = re.search(r'^LOCK_DIR="([^"]+)"$', script, flags=re.MULTILINE)
    assert match is not None
    assert str(environment.LOCK_DIR) == match.group(1)


# ---- git ----


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")
    (root / ".gitignore").write_text("/results/\n", encoding="utf-8")
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    return root


def test_clean_repo_passes(repo):
    environment.check_git_clean(repo)
    sha, commit_time = environment.git_head(repo)
    assert len(sha) == 40
    assert commit_time > 0


def test_ignored_files_do_not_count_as_dirty(repo):
    (repo / "results").mkdir()
    (repo / "results" / "run.json").write_text("{}", encoding="utf-8")
    environment.check_git_clean(repo)


def test_untracked_files_count_as_dirty(repo):
    (repo / "new_module.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(PreflightError, match="new_module.py"):
        environment.check_git_clean(repo)


def test_modified_files_count_as_dirty(repo):
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(PreflightError, match="a.txt"):
        environment.check_git_clean(repo)


# ---- interpreter、版本、binding ----


def test_interpreter_must_be_the_repo_venv(tmp_path):
    environment.check_venv(str(tmp_path / ".venv"), tmp_path)
    with pytest.raises(PreflightError, match="venv"):
        environment.check_venv("/usr", tmp_path)


def test_installed_versions_must_match_the_pins():
    pins = environment.parse_pins("# comment\nmaturin==1.15.0\nnumpy==2.5.3\n\nhnswlib==0.8.0\n")
    assert pins == {"maturin": "1.15.0", "numpy": "2.5.3", "hnswlib": "0.8.0"}
    installed = {"maturin": "1.15.0", "numpy": "2.5.3", "hnswlib": "0.8.0"}
    environment.check_versions(pins, installed.__getitem__)
    installed["numpy"] = "2.5.4"
    with pytest.raises(PreflightError, match="numpy"):
        environment.check_versions(pins, installed.__getitem__)


def test_repo_pins_cover_the_measured_libraries():
    pins = environment.parse_pins((REPO / "requirements-dev.txt").read_text(encoding="utf-8"))
    assert {"hnswlib", "faiss-cpu", "numpy", "maturin"} <= set(pins)


GOOD_BUILD = {
    "profile": "release",
    "opt_level": "3",
    "debug": "false",
    "debug_assertions": False,
    # maturin 在 macOS 為擴充模組加上的連結旗標；不影響 codegen。
    "rustflags": "-C link-arg=-undefined -C link-arg=dynamic_lookup",
    "rustc": "rustc 1.99.0",
}


def test_build_info_must_be_an_optimized_default_build():
    environment.check_build_info(GOOD_BUILD)
    environment.check_build_info({**GOOD_BUILD, "rustflags": ""})
    for key, value in [
        ("opt_level", "0"),
        ("debug_assertions", True),
        ("rustflags", "-C link-arg=-undefined -C target-cpu=native"),
        ("rustflags", "-Ctarget-cpu=native"),
        ("rustflags", "-C lto=fat"),
    ]:
        with pytest.raises(PreflightError, match=key):
            environment.check_build_info({**GOOD_BUILD, key: value})


def test_installed_binding_passes_the_build_check_after_test_sh():
    """scripts/test.sh 以 `maturin develop --release` 編譯；這也確認 maturin 的預設旗標被接受。"""
    import rust_hnsw

    environment.check_build_info(rust_hnsw.BUILD_INFO)


def test_binding_must_be_newer_than_head(tmp_path):
    so = tmp_path / "rust_hnsw.so"
    so.write_bytes(b"x")
    os.utime(so, (1000, 1000))
    environment.check_binding_fresh(so, head_commit_time=999)
    with pytest.raises(PreflightError, match="maturin develop --release"):
        environment.check_binding_fresh(so, head_commit_time=1001)


# ---- 資料集 ----


def test_formal_rejects_synthetic_datasets():
    with pytest.raises(PreflightError, match="synthetic"):
        environment.check_dataset_registered(datasets.REGISTRY["smoke-gaussian"])


def test_formal_rejects_checksum_mismatch(tmp_path):
    np.save(tmp_path / "train.npy", np.ones((20, 4), dtype=np.float32))
    np.save(tmp_path / "queries.npy", np.ones((3, 4), dtype=np.float32))
    good = {name: environment.sha256_file(tmp_path / name) for name in ("train.npy", "queries.npy")}
    dataset = datasets.NpyDataset(
        name="tiny", directory=tmp_path, train="train.npy", queries="queries.npy", sha256=good, metric="angular"
    )
    environment.check_dataset_registered(dataset)
    bad = datasets.NpyDataset(
        name="tiny",
        directory=tmp_path,
        train="train.npy",
        queries="queries.npy",
        sha256={**good, "train.npy": "0" * 64},
        metric="angular",
    )
    with pytest.raises(PreflightError, match="train.npy"):
        environment.check_dataset_registered(bad)


# ---- 記憶體與 hygiene 的解析 ----


def test_ps_rss_is_converted_from_kib_to_bytes():
    assert environment.parse_ps_rss("  12912\n") == 12912 * 1024
    assert environment.ps_rss_bytes(os.getpid()) > 10 * 1024 * 1024


def test_ru_maxrss_is_in_bytes_on_macos():
    import resource

    assert sys.platform == "darwin"
    assert environment.ru_maxrss_bytes() == resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert environment.ru_maxrss_bytes() > 10 * 1024 * 1024


NO_WARNING = (
    "Note: No thermal warning level has been recorded\n"
    "Note: No performance warning level has been recorded\n"
    "Note: No CPU power status has been recorded\n"
)


def test_thermal_output_without_warnings():
    assert environment.therm_warnings(NO_WARNING) == []


@pytest.mark.parametrize(
    "text",
    [
        "Thermal warning level set to 1.\n",
        "Performance warning level set to 2.\n",
        "2026-10-08 CPU Power notify\n\tCPU_Scheduler_Limit \t= 100\n\tCPU_Available_CPUs \t= 12\n\tCPU_Speed_Limit \t= 80\n",
        "\tCPU_Scheduler_Limit = 90\n",
    ],
)
def test_thermal_warnings_are_detected(text):
    assert environment.therm_warnings(text) != []


def test_full_speed_power_status_is_not_a_warning():
    text = "\tCPU_Scheduler_Limit \t= 100\n\tCPU_Available_CPUs \t= 12\n\tCPU_Speed_Limit \t= 100\n"
    assert environment.therm_warnings(text) == []


def test_swap_usage_is_parsed_in_bytes():
    text = "vm.swapusage: total = 2048.00M  used = 512.25M  free = 1535.75M  (encrypted)\n"
    assert environment.parse_swap_used(text) == int(512.25 * 1024 * 1024)
    assert environment.parse_swap_used("vm.swapusage: total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)") == 0


def test_compressor_pages_are_parsed():
    text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages occupied by compressor:   816108.\n"
    assert environment.parse_compressor_pages(text) == 816108


def test_snapshot_records_the_hygiene_fields():
    snapshot = environment.snapshot()
    assert set(snapshot) >= {"time", "loadavg", "therm", "therm_warnings", "swap_used_bytes", "compressor_pages"}
    assert len(snapshot["loadavg"]) == 3


# ---- load 門檻 ----


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_load_gate_passes_immediately_when_below_threshold():
    clock = FakeClock()
    ok, samples = environment.wait_for_load(lambda: 2.0, 4.0, 600, 15, clock.sleep, clock.monotonic)
    assert ok
    assert samples == [(0.0, 2.0)]


def test_load_gate_waits_until_load_drops():
    clock = FakeClock()
    loads = iter([6.0, 5.0, 3.5])
    ok, samples = environment.wait_for_load(lambda: next(loads), 4.0, 600, 15, clock.sleep, clock.monotonic)
    assert ok
    assert [load for _, load in samples] == [6.0, 5.0, 3.5]
    assert clock.now == 30


def test_load_gate_gives_up_after_the_wait_limit():
    clock = FakeClock()
    ok, samples = environment.wait_for_load(lambda: 9.0, 4.0, 600, 15, clock.sleep, clock.monotonic)
    assert not ok
    assert clock.now >= 600
    assert samples[-1][0] >= 600


def test_machine_metadata_has_core_counts_and_power_state():
    meta = environment.machine_metadata()
    assert meta["performance_cores"] + meta["efficiency_cores"] >= 1
    assert "System-wide power settings" in meta["pmset"]
    assert meta["cpu_brand"]
    assert meta["memory_bytes"] > 0
