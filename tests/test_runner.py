"""benchmark runner 的整合測試（decision 008）：smoke 模式、stub adapter 與注入的失敗。

smoke 只驗證 runner 本身，數字不作任何結論。子 process 都是真的 subprocess。
"""

import hashlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from benchmarks import datasets, environment, evaluate, protocol
from benchmarks import run as run_module
from benchmarks.config import IMPLEMENTATIONS, ConfigError
from benchmarks.environment import PreflightError
from benchmarks.environment import check_formal_power as REAL_CHECK_FORMAL_POWER
from benchmarks.interface import prepare_vectors
from benchmarks.metrics import recall_at_k, tie_aware_recall_at_k
# conftest 在每個測試替換這些讀取函式；import 時取得的是真正的函式，供專門的測試使用。
from benchmarks.run import read_background as REAL_READ_BACKGROUND, read_cpu as REAL_READ_CPU, read_power as REAL_READ_POWER
from benchmarks.run import Hooks, main, run

REPO = Path(__file__).resolve().parents[1]
SMOKE = (REPO / "benchmarks" / "configs" / "smoke.toml").read_text(encoding="utf-8")
SMOKE_DUPLICATES = (REPO / "benchmarks" / "configs" / "smoke-duplicates.toml").read_text(encoding="utf-8")

STUBS = {
    **IMPLEMENTATIONS,
    "exact-stub": ("tests.stub_adapters:ExactStub", "numpy"),
    "tie-stub": ("tests.stub_adapters:TieStub", "numpy"),
    "incomplete-stub": ("tests.stub_adapters:IncompleteStub", "numpy"),
    "padding-stub": ("tests.stub_adapters:PaddingStub", "numpy"),
    "crash-stub": ("tests.stub_adapters:CrashStub", "numpy"),
    "exit-stub": ("tests.stub_adapters:ExitStub", "numpy"),
    "slow-stub": ("tests.stub_adapters:SlowStub", "numpy"),
    "tracing-stub": ("tests.stub_adapters:TracingStub", "numpy"),
    "memory-stub": ("tests.stub_adapters:MemoryStub", "numpy"),
    "nondeterministic-stub": ("tests.stub_adapters:NondeterministicStub", "numpy"),
    "multithread-stub": ("tests.stub_adapters:MultithreadStub", "numpy"),
}


def stub_config(implementations, ef_search=(10,), n=(300,), seeds=(1,), shuffles=(), timeout=120):
    def toml_list(values):
        return "[" + ", ".join(json.dumps(v) for v in values) + "]"

    return f"""
schema_version = 1
dataset = "smoke-gaussian"
implementations = {toml_list(implementations)}
ef_search = {toml_list(ef_search)}
child_timeout_s = {timeout}

[[sweep]]
n = {toml_list(n)}
m = [8]
ef_construction = [32]
seeds = {toml_list(seeds)}
shuffles = {toml_list(shuffles)}
"""


def all_points(record):
    return [point for child in record["children"] for point in child["points"]]


def by_implementation(record, name):
    return [child for child in record["children"] if child["implementation"] == name]


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    result = run(SMOKE, "smoke", tmp_path_factory.mktemp("smoke"))
    return result


# ---- smoke（三個真實 adapter） ----


def test_smoke_run_completes(smoke):
    assert smoke.exit_code == 0
    record = json.loads((smoke.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record == smoke.record
    assert record["status"] == "complete"
    assert record["mode"] == "smoke"
    assert record["protocol"] == protocol.protocol_record()
    assert record["config"]["text"] == SMOKE
    assert smoke.run_dir.name.endswith("-smoke")
    children = record["children"]
    assert len(children) == 18  # 2 個 n × (2 seed + 1 shuffle) × 3 實作
    for child in children:
        assert child["status"] == "ok", child
        assert not child["partial"]
        assert [point["ef_search"] for point in child["points"]] == [10, 32]


def test_execution_order_interleaves_and_rotates_implementations(smoke):
    order = smoke.record["execution_order"]
    names = ("rust-hnsw", "hnswlib", "faiss-hnsw")
    for group, entry in enumerate(order):
        assert entry["group"] == group
        assert tuple(entry["implementations"]) == names[group % 3 :] + names[: group % 3]
    children = smoke.record["children"]
    assert [c["implementation"] for c in children[:6]] == [*order[0]["implementations"], *order[1]["implementations"]]


def test_child_imports_only_the_measured_library(smoke):
    expected = {"rust-hnsw": "rust_hnsw", "hnswlib": "hnswlib", "faiss-hnsw": "faiss"}
    for child in smoke.record["children"]:
        assert child["loaded_libraries"] == [expected[child["implementation"]]]


def test_recall_is_high_at_the_larger_ef(smoke):
    for point in all_points(smoke.record):
        assert point["k10"]["status"] == "ok"
        if point["ef_search"] == 32:
            assert point["k10"]["recall"] >= 0.9, point["point_id"]
            assert point["k1"]["recall"] >= 0.9, point["point_id"]


def test_smoke_points_are_single_threaded_and_deterministic(smoke):
    for child in smoke.record["children"]:
        assert "multithread" not in child["flags"], child["child_id"]
        for point in child["points"]:
            assert point["k10"]["repeat_mismatch_queries"] == 0
            assert point["k10"]["batch_vs_single_mismatch_queries"] == 0
            assert "nondeterministic" not in point["flags"]
            assert "multithread" not in point["flags"]


def test_memory_fields_are_bytes(smoke):
    for child in smoke.record["children"]:
        build = child["build"]
        assert build["build_ns"] > 0
        assert build["peak_rss_bytes"] >= build["baseline"]["ru_maxrss_bytes"] > 10 * 1024 * 1024
        assert build["peak_rss_minus_baseline_bytes"] == build["peak_rss_bytes"] - build["baseline"]["ru_maxrss_bytes"]
        assert build["index_memory_bytes"] == build["after_build"]["rss_bytes"] - build["baseline"]["rss_bytes"]
        assert build["index_footprint_bytes"] == (
            build["after_build"]["phys_footprint_bytes"] - build["baseline"]["phys_footprint_bytes"]
        )
        assert build["peak_rss_after_search_bytes"] >= build["peak_rss_bytes"]
        assert build["serialized_bytes"] > 0


def test_null_call_and_python_loop_are_recorded(smoke):
    for child in smoke.record["children"]:
        null_call = child["null_call"]
        assert (null_call["n"], null_call["k"], null_call["ef"]) == (1, 1, 1)
        assert null_call["samples"] == 1000
        assert 0 < null_call["p50_ns"] <= null_call["p95_ns"]
        loop = child["python_loop"]
        assert loop["mean_ns"] > 1.0
        assert 0 < loop["p50_ns"] <= loop["p95_ns"]
        samples = np.load(smoke.run_dir / child["samples_file"])
        assert samples["python_loop_ns"].shape == (1000,)
        assert samples["null_call_ns"].shape == (1000,)


def test_subset_checks_are_recorded(smoke):
    subsets = smoke.record["subsets"]
    assert set(subsets) == {"300", "600"}
    for subset in subsets.values():
        assert subset["duplicate_row_fraction"] == 0.0
        assert subset["rank1_tie_queries"] == 0
        assert subset["tie_eps"] == protocol.TIE_EPS
        assert subset["provided_gt_agreement"]["status"] == "not_applicable"


def test_metadata_is_recorded(smoke):
    meta = smoke.record["metadata"]
    assert meta["git"]["commit"]
    assert "status" in meta["git"]
    assert meta["build_info"]["profile"] in {"debug", "release"}
    assert set(meta["versions"]) >= {"numpy", "hnswlib", "faiss-cpu", "maturin"}
    assert meta["clock"]["implementation"]
    assert set(meta["extension_sha256"]) == {"rust_hnsw", "hnswlib", "faiss"}
    assert "faiss_compile_options" in meta
    assert meta["machine"]["performance_cores"] >= 1
    assert meta["start"]["loadavg"] and meta["end"]["loadavg"]
    assert meta["end"]["pmset"] and meta["end"]["pmset_batt"] and meta["end"]["power_source"]
    for key in meta["child_env"]:
        assert key.startswith(run_module.CHILD_ENV_PREFIXES)


def test_metrics_can_be_recomputed_from_the_files(smoke):
    run_dir = smoke.run_dir
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    loaded = datasets.load(datasets.REGISTRY[record["dataset"]["name"]])
    queries = prepare_vectors(loaded.queries)
    train = prepare_vectors(loaded.train)
    assert set(record["files"]) == {
        str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file() and p.name != "run.json"
    }
    assert not any("/raw/" in path for path in record["files"])
    for path, digest in record["files"].items():
        assert environment.sha256_file(run_dir / path) == digest
    for child in record["children"]:
        n = child["build"]["n"]
        gt = np.load(run_dir / record["subsets"][str(n)]["gt_file"])
        for point in child["points"]:
            arrays = np.load(run_dir / point["file"])
            assert arrays["ids_k10"].shape == (queries.shape[0], 10)
            assert recall_at_k(arrays["ids_k10"], gt, 10) == point["k10"]["recall"]
            assert recall_at_k(arrays["ids_k1"], gt, 1) == point["k1"]["recall"]
            assert tie_aware_recall_at_k(arrays["ids_k10"], gt, train[:n], queries, 10) == point["k10"]["tie_aware_recall"]
            assert tie_aware_recall_at_k(arrays["ids_k1"], gt, train[:n], queries, 1) == point["k1"]["tie_aware_recall"]
            assert evaluate.qps_summary(arrays["qps_ns"], queries.shape[0]) == point["k10"]["qps"]
            assert evaluate.latency_summary(arrays["latency_ns"]) == point["k10"]["latency"]
            assert arrays["latency_ns"].shape == (3, queries.shape[0])


def test_run_status_is_reported_by_main(tmp_path):
    config = tmp_path / "bad.toml"
    config.write_text(SMOKE.replace("ef_construction = [32]", "ef_construction = [4]"), encoding="utf-8")
    assert main([str(config), "--mode", "smoke", "--output-root", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


# ---- 重複向量：補齊與結果不足 k ----


def test_duplicates_smoke_marks_short_results(tmp_path):
    result = run(SMOKE_DUPLICATES, "smoke", tmp_path)
    assert result.record["status"] == "complete"
    subset = result.record["subsets"]["300"]
    assert subset["duplicate_row_fraction"] == 1.0
    for child in result.record["children"]:
        for point in child["points"]:
            k10 = point["k10"]
            if child["implementation"] == "hnswlib" and k10["status"] == "incomplete":
                # 整批失敗：不足 k 的 query 數無法取得（007 §3）。
                assert k10["incomplete_queries"] is None
                assert k10["qps"] is None
            else:
                # Rust / FAISS 補齊；hnswlib 成功時每個 query 都有 k 個結果。
                assert k10["status"] == "ok"
                assert k10["incomplete"] == (k10["incomplete_queries"] > 0)
    # Rust 在這份資料上恆為 2M + 1 = 9 個結果（007 的表格）。
    rust = by_implementation(result.record, "rust-hnsw")[0]["points"][0]["k10"]
    assert rust["incomplete_queries"] == 100


# ---- stub：id 映射、Recall@1 的來源、結果不足 k ----


def test_shuffled_builds_are_mapped_back_to_subset_rows(tmp_path):
    result = run(stub_config(["exact-stub"], seeds=(1, 2), shuffles=(7, 8)), "smoke", tmp_path, implementations=STUBS)
    assert result.record["status"] == "complete"
    shuffles = [child["build"]["shuffle"] for child in result.record["children"]]
    assert shuffles == [None, None, 7, 8]
    for point in all_points(result.record):
        assert point["k10"]["recall"] == 1.0
        assert point["k1"]["recall"] == 1.0
    expected = hashlib.sha256(np.random.default_rng(7).permutation(300).astype("<i8").tobytes()).hexdigest()
    assert result.record["children"][2]["build"]["permutation_sha256"] == expected
    # exact-stub 沒有 tie：k = 1 的答案與第一欄在同一個 id 空間中完全相同。
    assert all(p["k1_differs_from_k10_first_column"] == 0 for p in all_points(result.record))


def test_padding_survives_the_mapping(tmp_path):
    result = run(stub_config(["padding-stub"], shuffles=(7,)), "smoke", tmp_path, implementations=STUBS)
    for point in all_points(result.record):
        assert point["k10"]["incomplete_queries"] == 100
        assert point["k10"]["recall"] == pytest.approx(0.9)
        arrays = np.load(result.run_dir / point["file"])
        assert (arrays["ids_k10"][:, -1] == -1).all()


def test_recall_at_1_comes_from_the_k1_search(tmp_path):
    result = run(stub_config(["tie-stub"]), "smoke", tmp_path, implementations=STUBS)
    point = all_points(result.record)[0]
    assert point["k1"]["recall"] == 1.0
    assert point["k10"]["recall"] == 1.0
    assert point["k1_differs_from_k10_first_column"] == 100


def test_incomplete_results_mark_the_point_without_retrying(tmp_path, monkeypatch):
    log = tmp_path / "calls.log"
    monkeypatch.setenv("STUB_CALL_LOG", str(log))
    result = run(stub_config(["incomplete-stub"], ef_search=(10, 20)), "smoke", tmp_path / "out", implementations=STUBS)
    assert result.record["status"] == "complete"
    for point in all_points(result.record):
        k10 = point["k10"]
        assert k10["status"] == "incomplete"
        assert k10["incomplete"] is True
        assert k10["incomplete_queries"] is None
        assert k10["recall"] is None and k10["tie_aware_recall"] is None
        assert k10["qps"] is None and k10["latency"] is None
        assert point["k1"]["status"] == "ok"
        assert point["k1"]["recall"] == 1.0
    calls = log.read_text(encoding="utf-8").split()
    # 只有每個 ef 一次 k = 10 的 warm-up（整批）；沒有逐筆重試。
    assert [c for c in calls if c.startswith("10,")] == ["10,100", "10,100"]


# ---- 子 process 的失敗 ----


def test_crash_keeps_completed_points(tmp_path):
    result = run(stub_config(["crash-stub", "exit-stub", "exact-stub"], ef_search=(10, 20)), "smoke", tmp_path,
                 implementations=STUBS)
    assert result.exit_code != 0
    assert result.record["status"] == "incomplete"
    crash = by_implementation(result.record, "crash-stub")[0]
    assert crash["status"] == "signal"
    assert crash["returncode"] == -signal.SIGABRT
    assert crash["partial"] is True
    assert [point["ef_search"] for point in crash["points"]] == [10]
    assert by_implementation(result.record, "exact-stub")[0]["status"] == "ok"
    exited = by_implementation(result.record, "exit-stub")[0]
    assert (exited["status"], exited["returncode"], exited["partial"]) == ("exit_nonzero", 3, True)
    assert [point["ef_search"] for point in exited["points"]] == [10]
    assert json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "incomplete"
    spec = json.loads((result.run_dir / "children" / crash["child_id"] / "spec.json").read_text(encoding="utf-8"))
    assert not Path(spec["scratch_dir"]).exists()
    assert not any("/raw/" in path for path in result.record["files"])


def test_nonzero_exit_and_timeout_are_classified(tmp_path):
    commands = iter(
        [
            [sys.executable, "-c", "raise SystemExit(3)"],
            [sys.executable, "-c", "import time; time.sleep(30)"],
        ]
    )
    hooks = Hooks(child_command=lambda spec: next(commands))
    result = run(stub_config(["exact-stub"], seeds=(1, 2), timeout=1), "smoke", tmp_path, implementations=STUBS, hooks=hooks)
    statuses = [(c["status"], c["returncode"]) for c in result.record["children"]]
    assert statuses == [("exit_nonzero", 3), ("timeout", None)]
    assert result.record["status"] == "incomplete"
    assert result.exit_code != 0


def test_sigterm_aborts_and_still_writes_run_json(tmp_path):
    calls = []

    def command(spec):
        calls.append(spec)
        if len(calls) == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return [sys.executable, "-m", "benchmarks.run_child", str(spec)]

    before = signal.getsignal(signal.SIGTERM)
    result = run(stub_config(["exact-stub"], seeds=(1, 2, 3)), "smoke", tmp_path, implementations=STUBS, hooks=Hooks(child_command=command))
    assert signal.getsignal(signal.SIGTERM) is before
    assert result.record["status"] == "aborted"
    assert result.exit_code != 0
    assert [c["status"] for c in result.record["children"]] == ["ok"]
    assert (result.run_dir / "run.json").exists()


# ---- formal 模式 ----


def passing_formal_hooks(**overrides):
    defaults = dict(
        formal_preflight=lambda context: {"checks": "passed (test)"},
        formal_recheck=lambda context: None,
        load_reader=lambda: 1.0,
    )
    return Hooks(**{**defaults, **overrides})


def test_formal_mode_refuses_to_run_without_its_checks(tmp_path):
    # 測試環境沒有持有 heavy lock，真正的前置檢查一定拒絕。
    with pytest.raises(PreflightError):
        run(stub_config(["exact-stub"]), "formal", tmp_path, implementations=STUBS)
    assert not any(tmp_path.iterdir())


def test_formal_output_matches_smoke_except_mode_and_checks(tmp_path):
    text = stub_config(["exact-stub"])
    smoke_result = run(text, "smoke", tmp_path / "smoke", implementations=STUBS)
    formal_result = run(text, "formal", tmp_path / "formal", implementations=STUBS, hooks=passing_formal_hooks())
    assert formal_result.record["mode"] == "formal"
    assert formal_result.record["status"] == "complete"
    assert formal_result.record["metadata"]["formal_checks"] == {"checks": "passed (test)"}
    assert set(formal_result.record) == set(smoke_result.record)
    assert set(formal_result.record["children"][0]) == set(smoke_result.record["children"][0])
    assert formal_result.run_dir.name.endswith("-formal")


def test_formal_recheck_failure_marks_the_run_invalid(tmp_path):
    checks = []

    def recheck(context):
        checks.append(1)
        if len(checks) == 2:
            raise PreflightError("owner.txt changed")

    hooks = passing_formal_hooks(formal_recheck=recheck)
    result = run(stub_config(["exact-stub"], seeds=(1, 2, 3)), "formal", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["status"] == "invalid"
    assert "owner.txt changed" in " ".join(result.record["status_reasons"])
    assert len(result.record["children"]) == 1
    assert result.exit_code != 0


def test_formal_load_gate_aborts_after_waiting(tmp_path):
    clock = {"now": 0.0}

    def sleep(seconds):
        clock["now"] += seconds

    hooks = passing_formal_hooks(load_reader=lambda: 9.0, sleep=sleep, monotonic=lambda: clock["now"])
    result = run(stub_config(["exact-stub"]), "formal", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["status"] == "aborted"
    assert "load" in " ".join(result.record["status_reasons"])
    assert result.record["children"] == []
    assert clock["now"] >= protocol.LOAD_WAIT_S


def test_formal_gate_waits_for_background_workload_to_end(tmp_path):
    # 014：背景有其他 heavy workload 時，即使 load 低於門檻也要等。
    clock = {"now": 0.0}
    seen = iter([[{"pid": 42, "pcpu": 99.0, "command": "agent"}], []] + [[]] * 100)
    hooks = passing_formal_hooks(background_reader=lambda: next(seen), sleep=lambda s: clock.__setitem__("now", clock["now"] + s),
                                 monotonic=lambda: clock["now"])
    result = run(stub_config(["exact-stub"]), "formal", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["status"] == "complete"
    child = result.record["children"][0]
    assert len(child["load_gate"]["samples"]) == 2 and child["load_gate"]["background"] == []
    assert clock["now"] == protocol.LOAD_POLL_S


def test_formal_gate_aborts_when_background_workload_persists(tmp_path):
    clock = {"now": 0.0}
    hooks = passing_formal_hooks(background_reader=lambda: [{"pid": 42, "pcpu": 99.0, "command": "agent"}],
                                 sleep=lambda s: clock.__setitem__("now", clock["now"] + s), monotonic=lambda: clock["now"])
    result = run(stub_config(["exact-stub"]), "formal", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["status"] == "aborted"
    assert "background" in " ".join(result.record["status_reasons"]) and "agent" in " ".join(result.record["status_reasons"])
    assert result.record["children"] == []


def busy_cpu_reader(pid=42, command="agent", rate=1.0):
    """假的累積 CPU 時間：`pid` 以 `rate` 個 core 持續占用（014 補記的區間平均判定）。"""
    start = time.monotonic()
    return lambda: {pid: (1, rate * (time.monotonic() - start), command)}


@pytest.fixture
def fast_sampling(monkeypatch):
    """假的子 process 不到 1 秒：把取樣間隔縮短，判定的最短視窗（4 個區間）也跟著縮短。"""
    monkeypatch.setattr(protocol, "BACKGROUND_POLL_S", 0.01)


def test_background_workload_during_a_child_flags_it(tmp_path, fast_sampling):
    # gate 通過之後、子 process 期間持續占用一個 core 的背景工作：標記 background，離開主要讀值（010 §3）。
    hooks = passing_formal_hooks(cpu_reader=busy_cpu_reader())
    result = run(stub_config(["exact-stub"]), "formal", tmp_path, implementations=STUBS, hooks=hooks)
    child = result.record["children"][0]
    assert "background" in child["flags"]
    assert {h["rule"] for h in child["background"]["heavy"]} >= {"pid_mean"}
    assert child["background"]["samples"] >= 2 and child["background"]["intervals"]


def test_an_idle_background_does_not_flag_the_child(tmp_path, fast_sampling):
    hooks = passing_formal_hooks(cpu_reader=busy_cpu_reader(rate=0.1))  # 只用 10% 的 core
    result = run(stub_config(["exact-stub"]), "formal", tmp_path, implementations=STUBS, hooks=hooks)
    assert "background" not in result.record["children"][0]["flags"]


def test_cpu_is_sampled_before_the_child_starts_and_after_it_exits(tmp_path):
    marker = tmp_path / "child-finished"
    calls = []

    def cpu_reader():
        calls.append(marker.exists())
        return {}

    script = f"import pathlib, sys; pathlib.Path({str(marker)!r}).write_text('done'); sys.exit(1)"
    hooks = Hooks(child_command=lambda spec: [sys.executable, "-c", script], cpu_reader=cpu_reader)
    run(stub_config(["exact-stub"]), "smoke", tmp_path / "runs", implementations=STUBS, hooks=hooks)
    assert calls[0] is False and calls[-1] is True


def test_smoke_records_background_workload_without_waiting(tmp_path, fast_sampling):
    hooks = Hooks(background_reader=lambda: [{"pid": 42, "pcpu": 99.0, "command": "agent"}], cpu_reader=busy_cpu_reader(),
                  sleep=lambda s: pytest.fail("smoke must not wait"))
    result = run(stub_config(["exact-stub"]), "smoke", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["status"] == "complete"
    assert all("background" in c["flags"] for c in result.record["children"])


def test_a_power_violation_during_a_child_voids_and_stops_the_run(tmp_path):
    # 014 補記：子 process 期間的電力取樣發現改用電池供電：記錄這個子 process，run 作廢（invalid）並停止。
    states = iter([dict(FORMAL_POWER), {**FORMAL_POWER, "power_source": "Battery Power"}] + [dict(FORMAL_POWER)] * 100)

    def preflight(context):
        context.update(powermode=0)
        return {"checks": "passed (test)"}

    hooks = passing_formal_hooks(formal_preflight=preflight, power_reader=lambda: next(states))
    result = run(stub_config(["exact-stub"], seeds=(1, 2)), "formal", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["status"] == "invalid"
    assert "AC power" in " ".join(result.record["status_reasons"])
    assert len(result.record["children"]) == 1 and "AC power" in result.record["children"][0]["power_violation"]


def test_a_battery_drop_on_ac_is_recorded_but_does_not_void_the_run(tmp_path):
    # decision 017：電量下降只記錄在 power_samples，run 照常完成。
    states = iter([{**FORMAL_POWER, "battery": {"percent": p, "state": "AC attached"}} for p in [100, 96, 92] * 100])

    def preflight(context):
        context.update(powermode=0)
        return {"checks": "passed (test)"}

    hooks = passing_formal_hooks(formal_preflight=preflight, power_reader=lambda: next(states))
    result = run(stub_config(["exact-stub"], seeds=(1, 2)), "formal", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["status"] == "complete"
    samples = [s["battery"]["percent"] for c in result.record["children"] for s in c["power_samples"]]
    assert min(samples) < 100 and all(c["power_violation"] is None for c in result.record["children"])


def test_smoke_records_but_does_not_wait_for_load(tmp_path):
    hooks = Hooks(load_reader=lambda: 9.0, sleep=lambda s: pytest.fail("smoke must not wait"))
    result = run(stub_config(["exact-stub"]), "smoke", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["status"] == "complete"
    assert result.record["children"][0]["load_gate"]["samples"][0][1] == 9.0


# ---- 008 §6 的呼叫順序、計時邊界與記憶體時點（實作審查補上） ----


def test_child_call_sequence_matches_008(tmp_path, monkeypatch):
    log = tmp_path / "calls.log"
    monkeypatch.setenv("STUB_CALL_LOG", str(log))
    result = run(stub_config(["tracing-stub"], ef_search=(10, 20)), "smoke", tmp_path / "out", implementations=STUBS)
    assert result.record["status"] == "complete"
    expected = ["build:16", "ef:1", "search:1,1,1,1", "build:1", "ef:1"] + ["search:1,1,1,1"] * 1100 + ["build:300"]
    for ef in (10, 20):
        expected += [f"ef:{ef}", f"search:10,100,{ef},1"] + [f"search:10,100,{ef},0"] * 5
        expected += [f"search:10,1,{ef},0"] * 400 + [f"search:1,100,{ef},1"]
    expected.append("save")
    assert log.read_text(encoding="utf-8").split() == expected
    for point in all_points(result.record):
        assert np.load(result.run_dir / point["file"])["qps_ns"].shape == (5,)
        assert len(point["k10"]["qps"]["qps"]) == 5


class TickingArray(np.ndarray):
    """每次被切片就讓假時鐘前進：計時區間若包含切片，latency 就會大於 search 本身的時間。"""

    clock = None

    def __getitem__(self, item):
        TickingArray.clock["t"] += 1
        return super().__getitem__(item)


class FixedCostAdapter:
    def __init__(self, clock):
        self._clock = clock

    def search(self, queries, k):
        self._clock["t"] += 1000
        nq = queries.shape[0]
        ids = np.tile(np.arange(k, dtype=np.int64), (nq, 1))
        distances = np.tile(np.linspace(0.0, 1.0, k, dtype=np.float32), (nq, 1))
        return ids, distances


def test_timed_regions_contain_only_the_search_call(monkeypatch):
    from benchmarks import run_child

    clock = {"t": 0}
    monkeypatch.setattr(run_child, "perf_counter_ns", lambda: clock["t"])
    TickingArray.clock = clock
    queries = np.zeros((120, 4), dtype=np.float32).view(TickingArray)
    views = [np.asarray(queries[i : i + 1]) for i in range(120)]
    summary, arrays = run_child.measure_point(FixedCostAdapter(clock), queries, views)
    assert summary["status"] == "ok"
    assert (arrays["qps_ns"] == 1000).all()
    assert (arrays["latency_ns"] == 1000).all()


def test_memory_is_sampled_at_the_right_moments(tmp_path):
    result = run(stub_config(["memory-stub"]), "smoke", tmp_path, implementations=STUBS)
    build = result.record["children"][0]["build"]
    # library warm-up 的 64 MiB 在 baseline 之前；第一次 k = 10 search 的 128 MiB 在 peak RSS 之後。
    assert build["index_memory_bytes"] < 32 * 2**20
    assert build["peak_rss_minus_baseline_bytes"] < 32 * 2**20
    assert build["peak_rss_after_search_bytes"] - build["peak_rss_bytes"] > 96 * 2**20


def test_run_checks_that_the_dataset_is_large_enough(tmp_path):
    with pytest.raises(ConfigError, match="n = 1000"):
        run(stub_config(["exact-stub"], n=(1000,)), "smoke", tmp_path, implementations=STUBS)
    assert not any(tmp_path.iterdir())


# ---- hygiene 標記（008 §9） ----


def fake_snapshot(swap=0, warnings=(), power="AC Power"):
    return {"time": 0.0, "loadavg": [1.0, 1.0, 1.0], "therm": "", "therm_warnings": list(warnings),
            "swap_used_bytes": swap, "compressor_pages": 0, "power_source": power}


def test_child_flags_come_from_the_snapshots_around_it(tmp_path):
    base = fake_snapshot()
    pairs = [
        (fake_snapshot(swap=0), fake_snapshot(swap=1 << 20)),
        (fake_snapshot(swap=1 << 20), fake_snapshot(swap=0)),
        (fake_snapshot(), fake_snapshot(warnings=["Thermal Warning Level = 5"])),
        (fake_snapshot(warnings=["Thermal Warning Level = 5"]), fake_snapshot()),
        (fake_snapshot(), fake_snapshot(power="Battery Power")),
    ]
    snaps = iter([base, *[snap for pair in pairs for snap in pair], base])
    hooks = Hooks(snapshot=lambda: next(snaps))
    result = run(stub_config(["exact-stub"], seeds=(1, 2, 3, 4, 5)), "smoke", tmp_path, implementations=STUBS, hooks=hooks)
    flags = [set(c["flags"]) & {"swap", "thermal", "power"} for c in result.record["children"]]
    assert flags == [{"swap"}, set(), {"thermal"}, {"thermal"}, {"power"}]


def test_nondeterministic_results_are_counted_and_flagged(tmp_path):
    result = run(stub_config(["nondeterministic-stub"]), "smoke", tmp_path, implementations=STUBS)
    point = all_points(result.record)[0]
    assert point["k10"]["repeat_mismatch_queries"] == 100
    assert point["k10"]["batch_vs_single_mismatch_queries"] == 100
    assert "nondeterministic" in point["flags"]


def test_work_on_other_threads_is_flagged(tmp_path):
    result = run(stub_config(["multithread-stub"]), "smoke", tmp_path, implementations=STUBS)
    child = result.record["children"][0]
    assert "multithread" in child["flags"]
    assert all("multithread" in point["flags"] for point in child["points"])


@pytest.mark.parametrize(("other", "process", "flagged"), [(0.0205, 1.0, False), (0.0215, 1.0, True),
                                                           (0.0009, 0.0, False), (0.0011, 0.0, True)])
def test_single_thread_tolerance_boundaries(other, process, flagged):
    assert protocol.exceeds_single_thread_tolerance(other, process) is flagged


@pytest.mark.parametrize(("footprint", "rss", "flagged"), [(100 << 20, 100 << 20, False), (100 << 20, 97 << 20, False),
                                                           (100 << 20, 50 << 20, True), (0, -(2 << 20), True)])
def test_memory_tolerance_boundaries(footprint, rss, flagged):
    assert protocol.exceeds_memory_tolerance(footprint, rss) is flagged


def evaluate_with_load(tmp_path, load_end):
    raw = tmp_path / f"raw-{load_end}.npz"
    ids = np.tile(np.arange(10, dtype=np.int64), (100, 1))
    np.savez(raw, ids_k10=ids, qps_ns=np.full(5, 1000), latency_ns=np.full((3, 100), 1000), ids_k1=ids[:, :1])
    (tmp_path / "points").mkdir(exist_ok=True)
    line = {"ef_search": 10, "raw_file": str(raw), "start_time": 0.0, "end_time": 1.0, "loadavg_start": [9.0, 0, 0],
            "loadavg_end": [load_end, 0, 0], "k1": {"status": "ok"},
            "k10": {"status": "ok", "repeat_mismatch_queries": 0, "batch_vs_single_mismatch_queries": 0,
                    "other_thread_cpu": {"qps_other_s": 0.0, "qps_process_s": 1.0, "latency_other_s": 0.0,
                                         "latency_process_s": 1.0}}}
    data = prepare_vectors(np.random.default_rng(0).standard_normal((20, 4)))
    queries = prepare_vectors(np.random.default_rng(1).standard_normal((100, 4)))
    gt = np.tile(np.arange(10), (100, 1))
    return run_module._evaluate_point(line, f"c{load_end}", None, gt, data, queries, tmp_path)


def test_point_load_flag_uses_the_load_at_the_end_of_the_point(tmp_path):
    assert "load" not in evaluate_with_load(tmp_path, protocol.MAX_LOAD_1MIN)["flags"]
    assert "load" in evaluate_with_load(tmp_path, protocol.MAX_LOAD_1MIN + 0.01)["flags"]


def child_lines(load_end, footprint_delta, rss_delta):
    baseline = {"rss_bytes": 100 << 20, "ru_maxrss_bytes": 100 << 20, "phys_footprint_bytes": 100 << 20}
    after = {"rss_bytes": baseline["rss_bytes"] + rss_delta, "ru_maxrss_bytes": 200 << 20,
             "phys_footprint_bytes": baseline["phys_footprint_bytes"] + footprint_delta}
    return [{"type": "prelude", "baseline": baseline},
            {"type": "build", "build_ns": 1, "other_thread_cpu": {"other_s": 0.0, "process_s": 1.0},
             "peak_rss_bytes": 200 << 20, "after_build": after, "loadavg_start": [1.0, 0, 0],
             "loadavg_end": [load_end, 0, 0]}]


def test_build_level_flags_for_load_and_compressed_memory():
    build, _ = run_module._child_record(child_lines(9.0, 50 << 20, 50 << 20), {})
    assert build["flags"] == ["load"]
    build, _ = run_module._child_record(child_lines(1.0, 50 << 20, 20 << 20), {})
    assert build["flags"] == ["memory"]
    assert build["index_footprint_bytes"] == 50 << 20
    build, _ = run_module._child_record(child_lines(1.0, 50 << 20, 50 << 20), {})
    assert build["flags"] == []


# ---- 子 process 的環境（008 §5） ----


def test_children_receive_the_parent_environment_unchanged(tmp_path):
    dump = tmp_path / "env.json"
    script = "import json, os, sys; json.dump(dict(os.environ), open(sys.argv[1], 'w'))"
    hooks = Hooks(child_command=lambda spec: [sys.executable, "-c", script, str(dump)])
    result = run(stub_config(["exact-stub"]), "smoke", tmp_path / "out", implementations=STUBS, hooks=hooks)
    child_env = json.loads(dump.read_text(encoding="utf-8"))
    keys = {k for k in {*child_env, *os.environ} if k.startswith(run_module.CHILD_ENV_PREFIXES)}
    assert all(child_env.get(k) == os.environ.get(k) for k in keys)
    assert result.record["children"][0]["status"] == "missing_output"


def test_secret_looking_variables_are_not_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("CARGO_REGISTRY_TOKEN", "cio_secret_value")
    monkeypatch.setenv("PYTHON_TEST_TOKEN", "py_secret_value")
    monkeypatch.setenv("OMP_NUM_THREADS_TEST", "3")
    result = run(stub_config(["exact-stub"]), "smoke", tmp_path, implementations=STUBS)
    recorded = result.record["metadata"]["child_env"]
    assert "CARGO_REGISTRY_TOKEN" not in recorded and "PYTHON_TEST_TOKEN" not in recorded
    assert recorded["OMP_NUM_THREADS_TEST"] == "3"
    text = (result.run_dir / "run.json").read_text(encoding="utf-8")
    assert "cio_secret_value" not in text and "py_secret_value" not in text


# ---- 失敗、中止與收尾（008 §5、§12） ----


def test_missing_output_is_a_failure(tmp_path):
    hooks = Hooks(child_command=lambda spec: [sys.executable, "-c", "pass"])
    result = run(stub_config(["exact-stub"]), "smoke", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["children"][0]["status"] == "missing_output"
    assert result.record["status"] == "incomplete"
    assert result.exit_code != 0


def test_truncated_jsonl_line_is_ignored(tmp_path):
    script = (
        "import json, os, sys; spec = json.load(open(sys.argv[1])); f = open(spec['jsonl_path'], 'w');"
        "f.write(json.dumps({'type': 'start', 'time': 0}) + '\\n' + '{\"type\": \"pre'); f.flush(); os._exit(1)"
    )
    hooks = Hooks(child_command=lambda spec: [sys.executable, "-c", script, str(spec)])
    result = run(stub_config(["exact-stub"]), "smoke", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["children"][0]["status"] == "exit_nonzero"
    assert result.record["status"] == "incomplete"


def test_sigterm_during_a_child_keeps_its_completed_points(tmp_path):
    timer = threading.Timer(4.0, os.kill, args=(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        result = run(stub_config(["slow-stub"], ef_search=(10, 20)), "smoke", tmp_path, implementations=STUBS)
    finally:
        timer.cancel()
    assert result.record["status"] == "aborted"
    (child,) = result.record["children"]
    assert child["status"] == "aborted"
    assert child["partial"] is True
    assert [point["ef_search"] for point in child["points"]] == [10]
    assert not any("/raw/" in path for path in result.record["files"])


@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGINT])
def test_sighup_and_sigint_abort_and_still_write_run_json(tmp_path, signum):
    calls = []

    def command(spec):
        calls.append(spec)
        if len(calls) == 2:
            os.kill(os.getpid(), signum)
        return [sys.executable, "-m", "benchmarks.run_child", str(spec)]

    previous_int = signal.signal(signal.SIGINT, signal.default_int_handler)
    before = signal.getsignal(signal.SIGHUP)
    try:
        result = run(stub_config(["exact-stub"], seeds=(1, 2, 3)), "smoke", tmp_path, implementations=STUBS,
                     hooks=Hooks(child_command=command))
    finally:
        signal.signal(signal.SIGINT, previous_int)
    assert signal.getsignal(signal.SIGHUP) is before
    assert result.record["status"] == "aborted"
    assert (result.run_dir / "run.json").exists()


def failing_snapshot(fail_on):
    calls = []

    def snapshot():
        calls.append(1)
        if len(calls) == fail_on:
            raise ValueError("snapshot failed")
        return fake_snapshot()

    return snapshot


def test_a_failing_end_snapshot_still_writes_run_json(tmp_path):
    result = run(stub_config(["exact-stub"]), "smoke", tmp_path, implementations=STUBS,
                 hooks=Hooks(snapshot=failing_snapshot(4)))
    assert result.record["status"] == "complete"
    assert "snapshot failed" in result.record["metadata"]["end"]["error"]


def test_unexpected_errors_are_recorded_with_a_traceback_and_reraised(tmp_path):
    with pytest.raises(ValueError, match="snapshot failed"):
        run(stub_config(["exact-stub"]), "smoke", tmp_path, implementations=STUBS, hooks=Hooks(snapshot=failing_snapshot(2)))
    (run_json,) = tmp_path.glob("*/run.json")
    record = json.loads(run_json.read_text(encoding="utf-8"))
    assert record["status"] == "error"
    assert "ValueError" in record["traceback"]
    assert not any(tmp_path.glob("*/run.json.tmp"))


# ---- formal：前置檢查、重新確認與 load 門檻的接線 ----


PREFLIGHT_CHECKS = ("check_lock", "check_git_clean", "check_config_committed", "check_venv", "check_versions",
                    "check_build_info", "check_binding_fresh", "check_formal_power", "check_dataset_registered")
FORMAL_POWER = {"power_source": "AC Power", "powermode": 0, "adapter_watts": 68,
                "adapter": {"manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"},
                "battery": {"percent": 100, "state": "charged"}}


@pytest.fixture
def passing_environment(monkeypatch):
    called = []
    results = {"check_lock": "pid=1\n", "check_config_committed": {"path": "benchmarks/configs/x.toml"},
               "check_dataset_registered": {}}
    for name in PREFLIGHT_CHECKS:
        def check(*args, _name=name, **kwargs):
            called.append(_name)
            return results.get(_name)
        monkeypatch.setattr(environment, name, check)
    monkeypatch.setattr(environment, "git_head", lambda repo: ("abc", 0))
    monkeypatch.setattr(environment, "tracked_files", lambda repo: [])
    monkeypatch.setattr(environment, "power_state", lambda: dict(FORMAL_POWER))
    monkeypatch.setattr(environment, "read_lock_owner", lambda: (1, "pid=1\n"))
    files = {"rust_hnsw": "a", "hnswlib": "b", "faiss": "c"}
    digests = {"a": "1", "b": "2", "c": "3"}
    monkeypatch.setattr(run_module, "extension_files", lambda: dict(files))
    monkeypatch.setattr(environment, "sha256_file", lambda path: digests[path])
    return called, digests


def preflight_context():
    return {"dataset": datasets.REGISTRY["smoke-gaussian"], "config_path": None, "config_text": ""}


def test_formal_preflight_runs_every_check(passing_environment):
    called, _ = passing_environment
    context = preflight_context()
    checks = run_module.formal_preflight(context)
    assert sorted(called) == sorted(PREFLIGHT_CHECKS)
    assert checks["extension_sha256"] == {"rust_hnsw": "1", "hnswlib": "2", "faiss": "3"}
    # 014：電力條件寫進 formal_checks，run 開始時的電力模式供之後的重新確認比較（電量只記錄，decision 017）。
    assert checks["power"]["powermode"] == 0 and context["powermode"] == 0 and "battery_peak" not in context


@pytest.mark.parametrize("name", PREFLIGHT_CHECKS)
def test_each_preflight_check_can_refuse(passing_environment, monkeypatch, name):
    def refuse(*args, **kwargs):
        raise PreflightError(f"{name} refused")

    monkeypatch.setattr(environment, name, refuse)
    with pytest.raises(PreflightError, match=name):
        run_module.formal_preflight(preflight_context())


def recheck_context():
    return {"owner": "pid=1\n", "head": "abc", "extension_sha256": {"rust_hnsw": "1", "hnswlib": "2", "faiss": "3"},
            "powermode": 0}


def test_formal_recheck_checks_the_real_power_conditions(passing_environment, monkeypatch):
    # 真正的 check_formal_power：電量下降不作廢（decision 017）；電力模式不得改變、不得改用電池供電。
    monkeypatch.setattr(environment, "check_formal_power", REAL_CHECK_FORMAL_POWER)
    for percent in (100, 96, 80):
        monkeypatch.setattr(environment, "power_state",
                            lambda p=percent: {**FORMAL_POWER, "battery": {"percent": p, "state": "AC attached"}})
        run_module.formal_recheck(recheck_context())
    monkeypatch.setattr(environment, "power_state", lambda: {**FORMAL_POWER, "powermode": 2})
    with pytest.raises(PreflightError, match="changed during the run"):
        run_module.formal_recheck(recheck_context())
    monkeypatch.setattr(environment, "power_state", lambda: {**FORMAL_POWER, "power_source": "Battery Power"})
    with pytest.raises(PreflightError, match="AC power"):
        run_module.formal_recheck(recheck_context())


def test_the_real_readers_exclude_the_runner_and_its_ancestors(monkeypatch):
    seen = {}
    monkeypatch.setattr(environment, "background_processes", lambda exclude: seen.setdefault("exclude", set(exclude)) and [])
    REAL_READ_BACKGROUND()
    assert os.getpid() in seen["exclude"] and os.getppid() in seen["exclude"]
    monkeypatch.setattr(environment, "cpu_table", lambda: {"cpu": True})
    monkeypatch.setattr(environment, "power_state", lambda: {"power": True})
    assert REAL_READ_CPU() == {"cpu": True} and REAL_READ_POWER() == {"power": True}


def test_formal_recheck_passes_when_nothing_changed(passing_environment):
    run_module.formal_recheck(recheck_context())


@pytest.mark.parametrize(
    ("change", "message"),
    [("owner", "owner"), ("head", "HEAD"), ("dirty", "dirty"), ("faiss", "faiss"), ("missing", "FileNotFoundError"),
     ("power", "powermode")],
)
def test_formal_recheck_detects_each_change(passing_environment, monkeypatch, change, message):
    _, digests = passing_environment
    if change == "owner":
        monkeypatch.setattr(environment, "read_lock_owner", lambda: (2, "pid=2\n"))
    elif change == "head":
        monkeypatch.setattr(environment, "git_head", lambda repo: ("def", 0))
    elif change == "dirty":
        def dirty(repo):
            raise PreflightError("working tree is dirty")
        monkeypatch.setattr(environment, "check_git_clean", dirty)
    elif change == "faiss":
        digests["c"] = "changed"
    elif change == "power":
        def changed(state, powermode=None):
            raise PreflightError(f"pmset powermode changed during the run: {powermode} -> 2")
        monkeypatch.setattr(environment, "check_formal_power", changed)
    else:
        def missing(path):
            raise FileNotFoundError(path)
        monkeypatch.setattr(environment, "sha256_file", missing)
    with pytest.raises(PreflightError, match=message):
        run_module.formal_recheck(recheck_context())


def test_formal_load_gate_is_checked_before_every_child(tmp_path):
    loads = iter([1.0] + [9.0] * 1000)
    clock = {"now": 0.0}
    hooks = passing_formal_hooks(load_reader=lambda: next(loads), sleep=lambda s: clock.__setitem__("now", clock["now"] + s),
                                 monotonic=lambda: clock["now"])
    result = run(stub_config(["exact-stub"], seeds=(1, 2)), "formal", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["status"] == "aborted"
    assert [c["status"] for c in result.record["children"]] == ["ok"]


def test_formal_recheck_happens_after_waiting_for_load(tmp_path):
    state = {"changed": False, "first": True}

    def load():
        if state["first"]:
            state["first"] = False
            state["changed"] = True  # 例如等待期間另一個 session 重新編譯了 binding
            return 9.0
        return 1.0

    def recheck(context):
        if state["changed"]:
            raise PreflightError("rust_hnsw extension module changed during the run")

    hooks = passing_formal_hooks(load_reader=load, formal_recheck=recheck, sleep=lambda s: None)
    result = run(stub_config(["exact-stub"]), "formal", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["status"] == "invalid"
    assert result.record["children"] == []


def test_formal_recheck_runs_once_more_after_the_last_child(tmp_path):
    calls = []

    def recheck(context):
        calls.append(1)
        if len(calls) == 3:
            raise PreflightError("HEAD changed during the run")

    hooks = passing_formal_hooks(formal_recheck=recheck)
    result = run(stub_config(["exact-stub"], seeds=(1, 2)), "formal", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["status"] == "invalid"
    assert len(result.record["children"]) == 2
