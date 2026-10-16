"""Advanced C 的真實資料 correctness smoke 的核對（decision 019 §9 第 6 步）：run 的檢查、與 Core 的 ids 核對、Q 的移植。

全部是 synthetic 資料（隨機 unit vectors）；smoke run 與 Core run 由同一組 adapter 在 synthetic 資料上產生，只用來驗證核對邏輯。
"""

import json

import faiss
import numpy as np
import pytest

from benchmarks import quant_report, quant_smoke
from benchmarks.quant_report import CORE_RUNS, GLOVE, F, P, Q, R, S
from helpers import random_unit
from test_quant_diag import formal_child

SMALL_N = 600
EFS = quant_smoke.SMOKE_EFS[GLOVE]


@pytest.fixture(autouse=True)
def synthetic_cell(monkeypatch):
    """synthetic 資料只有 600 列：把 C 的格點換成 n = 600（M、efC 仍是 16、200）。"""
    monkeypatch.setattr(quant_report, "N", SMALL_N)
    before = faiss.omp_get_max_threads()
    yield
    faiss.omp_set_num_threads(before)


@pytest.fixture(scope="module")
def vectors():
    data = random_unit(61, SMALL_N, 16)
    queries = random_unit(62, 30, 16)
    exact = __import__("rust_hnsw").ExactIndex(16)
    exact.add(data)
    gt, _ = exact.search(queries, 10)
    return data, queries, gt


def write_run(directory, children, mode="smoke", config_path=quant_smoke.SMOKE_CONFIGS[GLOVE], efs=EFS, status="complete"):
    record = {"mode": mode, "status": status, "status_reasons": [], "protocol": {"version": "008-v5"},
              "dataset": {"name": GLOVE, "n_queries": 30, "dim": 16},
              "config": {"path": config_path, "parsed": {"ef_search": list(efs)}}, "metadata": {}, "children": children}
    (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return str(directory), record


@pytest.fixture(scope="module")
def runs(vectors, tmp_path_factory):
    """一個 GloVe 的 smoke run（五個臂，seed 1）與 019 §2 登記的兩個 GloVe Core run（R seeds 1–5 中用到 seed 1、P seeds 1–3）。"""
    data, queries, gt = vectors
    root = tmp_path_factory.mktemp("smoke")
    smoke = root / "smoke"
    (smoke / "points").mkdir(parents=True)
    children = [formal_child(smoke, impl, 1, EFS, data, queries, gt, group=i) for i, impl in enumerate(quant_smoke.SMOKE_ARMS)]
    labelled = [write_run(smoke, children)]
    core_dirs = []
    for name in sorted(set(CORE_RUNS[GLOVE].values())):
        directory = root / "core" / name
        (directory / "points").mkdir(parents=True)
        # smoke 只比對 seed 1：seeds 2–5 的那一個 Core run 不需要任何 build。
        core_children = ([formal_child(directory, impl, 1, (64, 304), data, queries, gt, group=i) for i, impl in enumerate((R, P))]
                         if CORE_RUNS[GLOVE][1] == name else [])
        (directory / "run.json").write_text(json.dumps({"dataset": {"name": GLOVE}, "children": core_children}))
        core_dirs.append(str(directory))
    return labelled, core_dirs


def test_a_good_smoke_run_passes(vectors, runs):
    data, queries, _ = vectors
    labelled, core_dirs = runs
    result = quant_smoke.check(labelled, core_dirs, port=True, load=lambda dataset, n: (data, queries, None, None))
    assert result["ok"], result
    ids = result["ids"][GLOVE]
    # Core 的 ef 是 64、304：F、R、P 各 2 個共同的 ef；C 內 F 與 R 3 個 ef。768 不在 Core 的清單，只在 C 內比對。
    assert ids["status"] == "一致" and ids["identical"] == 3 * 2 + 3 and ids["not_identical"] == []
    assert result["port"][GLOVE] == {"graph_identical": True, "graph_differences": [], "e_star": 304,
                                     "ids_match_at_e_star": True}
    run = result["runs"][0]
    assert run["problems"] == [] and sorted(c["implementation"] for c in run["children"]) == sorted(quant_smoke.SMOKE_ARMS)


def test_ids_that_differ_from_core_fail_the_smoke(vectors, runs, tmp_path):
    labelled, core_dirs = runs
    label, record = labelled[0]
    point = next(p for c in record["children"] if c["implementation"] == P for p in c["points"] if p["ef_search"] == 304)
    with np.load(f"{label}/{point['file']}") as arrays:
        saved = {name: arrays[name] for name in arrays.files}
    try:
        ids = saved["ids_k10"].copy()
        ids[0, [0, 1]] = ids[0, [1, 0]]
        np.savez(f"{label}/{point['file']}", **{**saved, "ids_k10": ids})
        result = quant_smoke.check(labelled, core_dirs)
        assert not result["ok"] and result["ids"][GLOVE]["status"] == "不一致"
        assert result["ids"][GLOVE]["not_identical"] == [
            {"against": "core", "implementation": P, "seed": 1, "ef_search": 304, "result": "different"}]
    finally:
        np.savez(f"{label}/{point['file']}", **saved)


def test_the_port_check_detects_a_graph_that_is_not_the_flat_graph(vectors, monkeypatch):
    # C8 的比對在 del flat 之前：移植出來的 graph 若不是 Flat 的 graph（這裡換成另一個 seed 的 Flat），必須照實列出差異。
    from benchmarks import adapters

    data, queries, _ = vectors
    real = adapters.faiss_sq8_from_flat

    def other_graph(flat, data, m):
        return real(adapters.faiss_hnsw_flat(data, m, 200, 2), data, m)

    monkeypatch.setattr(adapters, "faiss_sq8_from_flat", other_graph)
    result = quant_smoke.port_check(data, queries, 304, None)
    assert result["graph_identical"] is False and "neighbors" in result["graph_differences"]


def test_the_port_check_detects_a_q_that_is_not_the_smoke_q(vectors):
    data, queries, _ = vectors
    wrong = np.zeros((len(queries), 10), dtype=np.int64)
    result = quant_smoke.port_check(data, queries, 304, wrong)
    assert result["graph_identical"] and result["ids_match_at_e_star"] is False
    assert quant_smoke.port_check(data, queries, 304, None)["ids_match_at_e_star"] is None


@pytest.mark.parametrize("change, message", [
    ("formal", "smoke mode"), ("config", "is not benchmarks/configs/glove-quant-smoke.toml"), ("ef", "registered"),
    ("status", "run status"), ("child", "child status"), ("arm", "one seed-1 build of each arm"),
])
def test_the_run_itself_must_be_the_registered_smoke(runs, change, message):
    labelled, _ = runs
    label, record = labelled[0]
    record = json.loads(json.dumps(record))
    if change == "formal":
        record["mode"] = "formal"
    elif change == "config":
        record["config"]["path"] = "benchmarks/configs/quant-glove-seed-1.toml"
    elif change == "ef":
        record["config"]["parsed"]["ef_search"] = [64, 304]
    elif change == "status":
        record["status"] = "incomplete"
    elif change == "child":
        record["children"][0]["status"] = "timeout"
    else:
        record["children"] = [c for c in record["children"] if c["implementation"] != S]
    problems = quant_smoke.check_run(label, record)["problems"]
    assert any(message in p for p in problems), problems


def test_one_smoke_run_per_dataset(runs):
    labelled, core_dirs = runs
    with pytest.raises(SystemExit, match="at most one"):
        quant_smoke.check(labelled * 2, core_dirs)


def test_the_port_option_needs_the_heavy_lock(runs, monkeypatch):
    from benchmarks import environment

    labelled, core_dirs = runs

    def no_lock():
        raise environment.PreflightError("lock owner pid 1 is not an ancestor")

    monkeypatch.setattr(environment, "check_lock", no_lock)
    with pytest.raises(SystemExit, match="heavy lock"):
        quant_smoke.main([labelled[0][0], "--core-runs", *core_dirs, "--port"])


def test_main_prints_the_result_and_sets_the_exit_code(runs, capsys):
    labelled, core_dirs = runs
    assert quant_smoke.main([labelled[0][0], "--core-runs", *core_dirs]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True and printed["port"] == {}
    with pytest.raises(SystemExit, match="must list every Core run"):
        quant_smoke.main([labelled[0][0], "--core-runs", core_dirs[0]])


def test_f_and_r_are_compared_inside_c_on_every_smoke_ef(runs):
    labelled, core_dirs = runs
    label, record = labelled[0]
    record = json.loads(json.dumps(record))
    # F 的 768 換成 S 的檔案（不同的 graph 表示，ids 不同）：768 不在 Core 的清單，只有 C 內的比對抓得到。
    files = {c["implementation"]: {p["ef_search"]: p["file"] for p in c["points"]} for c in record["children"]}
    f_child = next(c for c in record["children"] if c["implementation"] == F)
    next(p for p in f_child["points"] if p["ef_search"] == 768)["file"] = files[Q][768]
    core = quant_report.load_core(core_dirs, [GLOVE])
    result = quant_smoke.ids_against_core([(label, record)], core[GLOVE], GLOVE)
    assert result["status"] == "不一致"
    assert result["not_identical"] == [{"against": "C seed 1 rust-hnsw", "implementation": F, "seed": 1, "ef_search": 768,
                                        "result": "different"}]
