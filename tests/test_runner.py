"""benchmark runner 的整合測試（decision 008）：smoke 模式、stub adapter 與注入的失敗。

smoke 只驗證 runner 本身，數字不作任何結論。子 process 都是真的 subprocess。
"""

import json
import os
import signal
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks import datasets, environment, evaluate, protocol
from benchmarks.config import IMPLEMENTATIONS
from benchmarks.environment import PreflightError
from benchmarks.interface import prepare_vectors
from benchmarks.metrics import recall_at_k, tie_aware_recall_at_k
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
        assert build["peak_rss_after_search_bytes"] >= build["peak_rss_bytes"]
        assert build["serialized_bytes"] > 0


def test_null_call_and_python_loop_are_recorded(smoke):
    for child in smoke.record["children"]:
        null_call = child["null_call"]
        assert (null_call["n"], null_call["k"], null_call["ef"]) == (1, 1, 1)
        assert null_call["samples"] == 1000
        assert 0 < null_call["p50_ns"] <= null_call["p95_ns"]
        assert child["python_loop"]["mean_ns"] > 0


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
    for key in meta["child_env"]:
        assert key.startswith(("OMP_", "KMP_", "RUSTFLAGS", "CARGO_", "PYTHON", "VIRTUAL_ENV"))


def test_metrics_can_be_recomputed_from_the_files(smoke):
    run_dir = smoke.run_dir
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    loaded = datasets.load(datasets.REGISTRY[record["dataset"]["name"]])
    queries = prepare_vectors(loaded.queries)
    train = prepare_vectors(loaded.train)
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
    assert result.record["children"][2]["build"]["permutation_sha256"]


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
    result = run(stub_config(["crash-stub", "exact-stub"], ef_search=(10, 20)), "smoke", tmp_path, implementations=STUBS)
    assert result.exit_code != 0
    assert result.record["status"] == "incomplete"
    crash = by_implementation(result.record, "crash-stub")[0]
    assert crash["status"] == "signal"
    assert crash["returncode"] == -signal.SIGABRT
    assert crash["partial"] is True
    assert [point["ef_search"] for point in crash["points"]] == [10]
    assert by_implementation(result.record, "exact-stub")[0]["status"] == "ok"
    assert json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "incomplete"


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


def test_smoke_records_but_does_not_wait_for_load(tmp_path):
    hooks = Hooks(load_reader=lambda: 9.0, sleep=lambda s: pytest.fail("smoke must not wait"))
    result = run(stub_config(["exact-stub"]), "smoke", tmp_path, implementations=STUBS, hooks=hooks)
    assert result.record["status"] == "complete"
    assert result.record["children"][0]["load_gate"]["samples"][0][1] == 9.0
