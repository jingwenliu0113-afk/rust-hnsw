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
    "profile_env": "",
    "rustc": "rustc 1.99.0",
}


def test_build_info_must_be_an_optimized_default_build():
    environment.check_build_info(GOOD_BUILD)
    environment.check_build_info({**GOOD_BUILD, "profile_env": "CARGO_PROFILE_DEV_OPT_LEVEL=1"})
    environment.check_build_info({**GOOD_BUILD, "rustflags": ""})
    for key, value in [
        ("opt_level", "0"),
        ("debug_assertions", True),
        ("rustflags", "-C link-arg=-undefined -C target-cpu=native"),
        ("rustflags", "-Ctarget-cpu=native"),
        ("rustflags", "-C lto=fat"),
        ("profile_env", "CARGO_PROFILE_RELEASE_LTO=fat"),
        ("profile_env", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=true"),
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


def test_binding_must_be_newer_than_every_tracked_source(tmp_path):
    so = tmp_path / "rust_hnsw.so"
    so.write_bytes(b"x")
    sources = [tmp_path / "lib.rs", tmp_path / "Cargo.toml"]
    for path in sources:
        path.write_text("x", encoding="utf-8")
        os.utime(path, (900, 900))
    os.utime(so, (1000, 1000))
    environment.check_binding_fresh(so, 999, sources)
    # 例如：改了原始碼、編譯、再 git checkout 還原 → 原始碼的 mtime 比 .so 新。
    os.utime(sources[0], (1100, 1100))
    with pytest.raises(PreflightError, match="lib.rs"):
        environment.check_binding_fresh(so, 999, sources)


def test_checkout_after_building_is_detected_in_a_real_repo(repo):
    src = repo / "src.rs"
    src.write_text("fn a() {}\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "src")
    _, head_time = environment.git_head(repo)
    src.write_text("fn b() {}\n", encoding="utf-8")
    so = repo.parent / "rust_hnsw.so"
    so.write_bytes(b"built from uncommitted code")
    os.utime(so, (head_time + 10, head_time + 10))
    os.utime(src, (head_time + 5, head_time + 5))
    git(repo, "checkout", "--", "src.rs")
    os.utime(repo / "src.rs", (head_time + 20, head_time + 20))  # checkout 會寫檔，mtime 晚於 .so
    with pytest.raises(PreflightError):
        environment.check_binding_fresh(so, head_time, environment.tracked_files(repo, ["src.rs"]))
    os.utime(so, (head_time + 30, head_time + 30))  # 在乾淨的 HEAD 上重新編譯
    environment.check_binding_fresh(so, head_time, environment.tracked_files(repo, ["src.rs"]))


def test_repo_binding_sources_are_tracked():
    files = environment.tracked_files(REPO)
    names = {str(p.relative_to(REPO)) for p in files}
    assert {"crates/hnsw/src/hnsw.rs", "python/src/lib.rs", "python/build.rs", "Cargo.toml", "Cargo.lock"} <= names


# ---- formal config 必須已 commit（008 §1.8） ----


def committed_config(repo, text="schema_version = 1\n"):
    path = repo / "benchmarks" / "configs" / "f.toml"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "config")
    return path, text


def test_committed_config_passes(repo):
    path, text = committed_config(repo)
    record = environment.check_config_committed(path, text, repo)
    assert record["path"] == "benchmarks/configs/f.toml"
    assert len(record["blob"]) == 40


def test_config_must_be_committed_in_benchmarks_configs(repo, tmp_path):
    path, text = committed_config(repo)
    with pytest.raises(PreflightError, match="differs"):
        environment.check_config_committed(path, text + "x = 1\n", repo)
    untracked = repo / "benchmarks" / "configs" / "new.toml"
    untracked.write_text(text, encoding="utf-8")
    with pytest.raises(PreflightError, match="not committed"):
        environment.check_config_committed(untracked, text, repo)
    ignored = repo / "results" / "f.toml"
    ignored.parent.mkdir()
    ignored.write_text(text, encoding="utf-8")
    with pytest.raises(PreflightError, match="benchmarks/configs"):
        environment.check_config_committed(ignored, text, repo)
    outside = tmp_path / "outside.toml"
    outside.write_text(text, encoding="utf-8")
    with pytest.raises(PreflightError, match="outside"):
        environment.check_config_committed(outside, text, repo)
    with pytest.raises(PreflightError, match="path"):
        environment.check_config_committed(None, text, repo)


def test_shipped_smoke_config_is_committed():
    path = REPO / "benchmarks" / "configs" / "smoke.toml"
    environment.check_config_committed(path, path.read_text(encoding="utf-8"), REPO)


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


# 格式取自本機 /usr/bin/pmset 的字串："Thermal Warning Level = %d"、"Performance Warning Level = %d"。
@pytest.mark.parametrize(
    "text",
    [
        "Thermal Warning Level = 5\n",
        "Performance Warning Level = 100\n",
        "Error:Failed to get thermal warning level with error code 0x00000001\n",
        "Thermal warning level set to 1.\n",
        "2026-10-08 CPU Power notify\n\tCPU_Scheduler_Limit \t= 100\n\tCPU_Available_CPUs \t= 12\n\tCPU_Speed_Limit \t= 80\n",
        "\tCPU_Scheduler_Limit = 90\n",
    ],
)
def test_thermal_warnings_are_detected(text):
    assert environment.therm_warnings(text) != []


def test_warning_level_zero_means_normal():
    assert environment.therm_warnings("Thermal Warning Level = 0\nPerformance Warning Level = 0\n") == []


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


def test_power_source_is_parsed_and_formal_needs_ac_power():
    batt = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t80%; discharging\n"
    assert environment.parse_power_source(batt) == "Battery Power"
    assert environment.parse_power_source("Now drawing from 'AC Power'\n") == "AC Power"
    environment.check_ac_power("AC Power")
    with pytest.raises(PreflightError, match="AC power"):
        environment.check_ac_power("Battery Power")


def test_power_mode_adapter_and_battery_are_parsed():
    pmset = "System-wide power settings:\nCurrently in use:\n standby              1\n powermode            2\n womp 1\n"
    assert environment.parse_powermode(pmset) == 2
    assert environment.parse_powermode("Currently in use:\n sleep 0\n") is None
    ac = " Wattage = 68W\n Current = 3390mA\n Adapter Name = 70W USB-C Power Adapter \n"
    assert environment.parse_adapter_watts(ac) == 68
    assert environment.parse_adapter_watts("No adapter attached.\n") is None
    batt = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t56%; charging; 2:47 remaining present: true\n"
    assert environment.parse_battery(batt) == {"percent": 56, "state": "charging"}
    assert environment.parse_battery("Now drawing from 'AC Power'\n") is None


def test_power_state_parses_each_field_from_its_own_command(monkeypatch):
    # 每個 child 的 snapshot 不保存原始 pmset 文字，接線錯了事後無法還原（NEXT.md B2 依賴這些紀錄）。
    outputs = {
        ("pmset", "-g"): "Currently in use:\n lowpowermode         1\n powermode            2\n",
        ("pmset", "-g", "batt"): "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t93%; charging; 0:30 remaining\n",
        ("pmset", "-g", "ac"): " Wattage = 68W\n Current = 3390mA\n",
    }
    monkeypatch.setattr(environment, "_run", lambda *args, **kwargs: outputs[args])
    state = environment.power_state()
    assert (state["power_source"], state["powermode"], state["adapter_watts"], state["battery"]) == \
        ("AC Power", 2, 68, {"percent": 93, "state": "charging"})


def test_phys_footprint_tracks_new_allocations():
    before = environment.phys_footprint_bytes()
    block = np.ones(32 * 2**20 // 8)
    after = environment.phys_footprint_bytes()
    assert after - before > 16 * 2**20
    del block


def test_snapshot_records_the_hygiene_fields():
    snapshot = environment.snapshot()
    assert set(snapshot) >= {"time", "loadavg", "therm", "therm_warnings", "swap_used_bytes", "compressor_pages",
                             "power_source", "powermode", "adapter_watts", "battery"}
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
    ok, samples, background = environment.wait_for_load(lambda: 2.0, 4.0, 600, 15, clock.sleep, clock.monotonic)
    assert ok
    assert samples == [(0.0, 2.0)] and background == []


def test_load_gate_waits_until_load_drops():
    clock = FakeClock()
    loads = iter([6.0, 5.0, 3.5])
    ok, samples, _ = environment.wait_for_load(lambda: next(loads), 4.0, 600, 15, clock.sleep, clock.monotonic)
    assert ok
    assert [load for _, load in samples] == [6.0, 5.0, 3.5]
    assert clock.now == 30


def test_load_gate_gives_up_after_the_wait_limit():
    clock = FakeClock()
    ok, samples, _ = environment.wait_for_load(lambda: 9.0, 4.0, 600, 15, clock.sleep, clock.monotonic)
    assert not ok
    assert clock.now >= 600
    assert samples[-1][0] >= 600


def test_machine_metadata_has_core_counts_and_power_state():
    meta = environment.machine_metadata()
    assert meta["performance_cores"] + meta["efficiency_cores"] >= 1
    assert "System-wide power settings" in meta["pmset"]
    assert {"pmset_ac", "powermode", "adapter_watts", "battery"} <= set(meta)
    assert meta["cpu_brand"]
    assert meta["memory_bytes"] > 0


def test_load_gate_also_waits_for_background_workload():
    clock = FakeClock()
    seen = iter([[{"pid": 1, "pcpu": 99.0, "command": "x"}], []])
    ok, samples, background = environment.wait_for_load(lambda: 1.0, 4.0, 600, 15, clock.sleep, clock.monotonic,
                                                         read_background=lambda: next(seen))
    assert ok and len(samples) == 2 and background == [] and clock.now == 15


# ---- decision 014：formal 的電力條件與背景 heavy workload ----

FORMAL_POWER = {"power_source": "AC Power", "powermode": 0, "adapter_watts": 68,
                "adapter": {"manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"},
                "battery": {"percent": 100, "state": "charged"}}


def test_adapter_is_parsed_from_pmset_ac():
    ac = " Wattage = 68W\n Manufacturer = Apple Inc.\n Adapter Name = 70W USB-C Power Adapter \n"
    assert environment.parse_adapter(ac) == {"manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"}
    assert environment.parse_adapter("No adapter attached.\n") is None


def test_formal_power_accepts_the_registered_conditions():
    environment.check_formal_power(FORMAL_POWER)
    environment.check_formal_power(FORMAL_POWER, powermode=0)
    environment.check_formal_power({**FORMAL_POWER, "battery": None})  # 沒有電池的機器


@pytest.mark.parametrize("battery", [{"percent": 96, "state": "AC attached"}, {"percent": 40, "state": "charging"},
                                     {"percent": 90, "state": "discharging"}])
def test_battery_level_is_not_a_formal_condition(battery):
    # decision 017：接 AC 時電量回落（例如充滿後 macOS 停止充電）不影響 CPU 速度，電量只記錄、不作為條件。
    environment.check_formal_power({**FORMAL_POWER, "battery": battery}, powermode=0)


@pytest.mark.parametrize("change, message", [
    ({"power_source": "Battery Power"}, "AC power"),
    ({"powermode": 1}, "powermode"),   # 低耗電模式
    ({"powermode": None}, "powermode"),  # pmset 沒有 powermode 這一行
    ({"adapter": {"manufacturer": "Apple Inc.", "name": "140W USB-C Power Adapter"}}, "70W"),
    ({"adapter": None}, "70W"),
    ({"adapter": {"manufacturer": "Other Inc.", "name": "70W USB-C Power Adapter"}}, "70W"),
])
def test_formal_power_refuses_each_violation(change, message):
    with pytest.raises(PreflightError, match=message):
        environment.check_formal_power({**FORMAL_POWER, **change})


def test_high_performance_is_allowed_but_must_not_change_during_a_run():
    environment.check_formal_power({**FORMAL_POWER, "powermode": 2})
    environment.check_formal_power({**FORMAL_POWER, "powermode": 2}, powermode=2)
    with pytest.raises(PreflightError, match="changed during the run"):
        environment.check_formal_power({**FORMAL_POWER, "powermode": 2}, powermode=0)


def test_background_processes_exclude_ancestors_and_use_the_threshold(monkeypatch):
    ps = " 1 99.0 launchd\n 2 50.0 agent-gym python\n 3 49.9 WindowServer\n 4 80.0 self\n bad\n"
    monkeypatch.setattr(environment, "_run", lambda *args, **kwargs: ps)
    heavy = environment.background_processes(exclude=[1, 4])
    assert heavy == [{"pid": 2, "pcpu": 50.0, "command": "agent-gym python"}]


# ---- 014 補記：子 process 期間以累積 CPU 時間判定背景 heavy workload ----


@pytest.mark.parametrize("text, seconds", [("0:00.29", 0.29), ("493:17.83", 29597.83), ("1:02:03.5", 3723.5),
                                           ("2-01:00:00.0", 176400.0)])
def test_cputime_is_parsed(text, seconds):
    assert environment.parse_cputime(text) == pytest.approx(seconds)


def test_cpu_table_and_task_vm_info_read_this_machine():
    table = environment.cpu_table()
    assert os.getpid() in table and table[os.getpid()][0] == os.getppid()
    assert os.getpid() in environment.descendants(table, os.getppid())
    vm = environment.task_vm_info()
    assert vm["phys_footprint_bytes"] == pytest.approx(environment.phys_footprint_bytes(), rel=0.05)
    assert {"reusable_bytes", "compressed_bytes", "compressed_lifetime_bytes"} <= set(vm)


def cpu_samples(rates_per_interval, poll=15.0):
    """[(t, table)]：rates_per_interval[i] = {pid: (command, 該區間的 core 數)}；累積 CPU 時間依此遞增。"""
    t, totals, samples = 0.0, {}, [(0.0, {})]
    for rates in rates_per_interval:
        t += poll
        table = {}
        for pid, (command, rate) in rates.items():
            totals[pid] = totals.get(pid, 0.0) + rate * poll
            table[pid] = (1, totals[pid], command)
        samples.append((t, table))
    return samples


def test_a_single_spike_does_not_flag_the_child():
    # 10 個區間中只有 1 個 150%：整段平均 15%，不連續，不算背景 heavy workload（瞬間尖峰的誤標，審查 F1）。
    rates = [{7: ("Claude Helper (Renderer)", 1.5 if i == 5 else 0.1)} for i in range(10)]
    assert environment.background_from_cpu_samples(cpu_samples(rates))["heavy"] == []


def test_short_children_are_not_flagged_by_one_spike():
    rates = [{7: ("Codex (Renderer)", 1.5)}, {7: ("Codex (Renderer)", 0.0)}, {7: ("Codex (Renderer)", 0.0)}]
    assert environment.background_from_cpu_samples(cpu_samples(rates))["heavy"] == []


def test_sustained_mid_child_work_is_flagged_even_if_it_stops_before_the_end():
    rates = [{9: ("worker", 1.0 if 2 <= i < 6 else 0.0)} for i in range(20)]
    heavy = environment.background_from_cpu_samples(cpu_samples(rates))["heavy"]
    assert heavy == [{"rule": "sustained", "pid": 9, "command": "worker", "intervals": 4}]


def test_a_stream_of_short_processes_is_flagged_by_command():
    rates = [{100 + i: ("rustc", 1.0)} for i in range(10)]  # 每個區間換一個 pid
    heavy = environment.background_from_cpu_samples(cpu_samples(rates))["heavy"]
    assert [(h["rule"], h["command"]) for h in heavy] == [("command_mean", "rustc")]


def test_the_runner_its_ancestors_and_the_measured_child_are_excluded():
    samples = cpu_samples([{1: ("launchd", 1.0), 50: ("runner", 1.0), 60: ("child", 1.0)}] * 4)
    for t, table in samples[1:]:
        table[60] = (50, table[60][1], table[60][2])  # 受測子 process 是 runner（pid 50）的子 process
    result = environment.background_from_cpu_samples(samples, exclude={1, 50}, root=50)
    assert result["heavy"] == []


def test_a_reused_pid_is_treated_as_a_new_process():
    samples = [(0.0, {5: (1, 1000.0, "old")}), (15.0, {5: (1, 1.5, "new")})]  # 舊的 process 結束，pid 被重用
    result = environment.background_from_cpu_samples(samples)
    assert result["heavy"] == [] and result["intervals"][0]["top"][0]["pcpu"] == pytest.approx(10.0)
