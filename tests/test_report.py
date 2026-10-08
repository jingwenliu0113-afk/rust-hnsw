"""分析 / 報告（decision 010）：frontier 讀值、variance 彙整、scaling、被標記與結果不足 k 的點。

大部分測試用手寫的 run 紀錄，讓每條規則都能被精確斷言；最後一個測試對真正的 runner 輸出跑一次。
"""

import json

import pytest

from benchmarks.report import ReportError, build_report, write_report
from benchmarks.run import run
from test_runner import STUBS, stub_config


def make_point(ef, recall10, qps, p50=1000.0, p95=2000.0, recall1=None, flags=(), status="ok", incomplete=False):
    if status == "incomplete":
        k10 = {"status": "incomplete", "incomplete": True, "incomplete_queries": None, "recall": None,
               "tie_aware_recall": None, "qps": None, "latency": None}
    else:
        k10 = {"status": "ok", "incomplete": incomplete, "incomplete_queries": 3 if incomplete else 0,
               "recall": recall10, "tie_aware_recall": recall10,
               "qps": {"qps": [qps] * 5, "qps_median": qps}, "latency": {"p50_ns": p50, "p95_ns": p95}}
    r1 = recall10 if recall1 is None else recall1
    return {"ef_search": ef, "k10": k10, "k1": {"status": "ok", "recall": r1, "tie_aware_recall": r1},
            "flags": list(flags)}


def make_child(impl, points, n=1000, m=16, efc=100, seed=1, shuffle=None, flags=(), status="ok",
               build_ns=1e9, peak=2e9, index_memory=1e8):
    return {"child_id": f"{impl}-{n}-{m}-{efc}-{seed}-{shuffle}", "implementation": impl, "status": status,
            "partial": status != "ok" and bool(points), "flags": list(flags),
            "build": {"n": n, "m": m, "ef_construction": efc, "seed": seed, "shuffle": shuffle, "build_ns": build_ns,
                      "peak_rss_bytes": peak, "index_memory_bytes": index_memory},
            "points": list(points)}


def make_record(children, mode="formal", dataset="d"):
    return {"mode": mode, "status": "complete", "dataset": {"name": dataset}, "protocol": {"version": "008-v1"},
            "children": children}


RUST_CURVE = [(10, 0.85, 1000.0), (20, 0.92, 800.0), (40, 0.96, 500.0), (80, 0.985, 300.0)]


def curve(impl, values, **kwargs):
    return make_child(impl, [make_point(ef, r, q) for ef, r, q in values], **kwargs)


# ---- 010 §2：frontier 讀值 ----


def test_frontier_reads_the_best_point_at_or_above_each_target_without_interpolation():
    report = build_report([make_record([curve("rust-hnsw", RUST_CURVE)])])
    frontier = report["frontier"]["rust-hnsw"]["1000"]
    assert frontier["0.90"]["main"]["qps"] == {"value": 800.0, "m": 16, "ef_construction": 100, "ef_search": 20,
                                                "incomplete": False, "builds": 1}
    assert frontier["0.95"]["main"]["qps"]["value"] == 500.0
    # 0.985 < 0.99：沒有任何點達到，不得以最接近的點代替。
    assert frontier["0.99"]["main"] == {"reached": False}
    assert frontier["0.90"]["main"]["reached"] is True


def test_frontier_spans_all_m_and_ef_construction():
    other = curve("rust-hnsw", [(10, 0.90, 400.0), (40, 0.951, 600.0)], m=8)
    report = build_report([make_record([curve("rust-hnsw", RUST_CURVE), other])])
    reading = report["frontier"]["rust-hnsw"]["1000"]["0.95"]["main"]
    assert reading["qps"] == {"value": 600.0, "m": 8, "ef_construction": 100, "ef_search": 40, "incomplete": False,
                              "builds": 1}


def test_latency_and_memory_readings_take_the_minimum():
    a = make_child("rust-hnsw", [make_point(10, 0.95, 500.0, p50=900.0, p95=5000.0)], index_memory=3e8)
    b = make_child("rust-hnsw", [make_point(10, 0.96, 400.0, p50=1200.0, p95=1500.0)], m=8, index_memory=2e8)
    reading = build_report([make_record([a, b])])["frontier"]["rust-hnsw"]["1000"]["0.95"]["main"]
    assert (reading["p50_ns"]["value"], reading["p50_ns"]["m"]) == (900.0, 16)
    assert (reading["p95_ns"]["value"], reading["p95_ns"]["m"]) == (1500.0, 8)
    assert (reading["index_memory_bytes"]["value"], reading["index_memory_bytes"]["m"]) == (2e8, 8)


def test_each_point_uses_the_median_over_seeds_of_the_original_order():
    children = [
        curve("rust-hnsw", [(10, r, q)], seed=s)
        for s, r, q in [(1, 0.89, 100.0), (2, 0.91, 300.0), (3, 0.95, 200.0)]
    ]
    # shuffle 的 build 不進入主要 frontier（010 §2）。
    children.append(curve("rust-hnsw", [(10, 0.99, 9999.0)], seed=1, shuffle=7))
    report = build_report([make_record(children)])
    reading = report["frontier"]["rust-hnsw"]["1000"]["0.90"]["main"]
    assert reading["qps"]["value"] == 200.0
    aggregate = next(a for a in report["seed_variance"] if a["ef_search"] == 10)
    assert aggregate["builds"] == 3
    assert aggregate["recall10"] == {"median": 0.91, "min": 0.89, "max": 0.95, "values": [0.89, 0.91, 0.95]}
    assert aggregate["qps"]["values"] == [100.0, 300.0, 200.0]


def test_ratios_need_both_implementations_to_reach_the_target():
    hnswlib = curve("hnswlib", [(10, 0.93, 2000.0), (40, 0.995, 900.0)])
    report = build_report([make_record([curve("rust-hnsw", RUST_CURVE), hnswlib])])
    ratios = report["ratios"]["1000"]
    assert ratios["0.90"]["hnswlib"] == pytest.approx(2000.0 / 800.0)
    assert ratios["0.99"]["hnswlib"] is None


# ---- 010 §3：被標記的點與結果不足 k ----


def test_flagged_points_leave_the_main_reading_and_both_readings_are_kept_when_they_differ():
    flagged = make_child("rust-hnsw", [make_point(10, 0.93, 5000.0, flags=["load"]), make_point(20, 0.94, 700.0)])
    report = build_report([make_record([flagged])])
    entry = report["frontier"]["rust-hnsw"]["1000"]["0.90"]
    assert entry["main"]["qps"]["value"] == 700.0
    assert entry["all_points"]["qps"]["value"] == 5000.0
    assert entry["differs"] is True
    assert [p["ef_search"] for p in report["flagged_points"]] == [10]


def test_main_reading_reports_how_many_builds_remain_after_excluding_flagged_ones():
    """010 §2 補記：排除被標記的 build 之後，用剩下的 seed 計算 median，並附上參與的 build 數。"""
    children = [
        make_child("rust-hnsw", [make_point(10, 0.95, q, flags=f)], seed=s)
        for s, q, f in [(1, 100.0, ()), (2, 200.0, ("load",)), (3, 300.0, ())]
    ]
    entry = build_report([make_record(children)])["frontier"]["rust-hnsw"]["1000"]["0.90"]
    assert (entry["main"]["qps"]["value"], entry["main"]["qps"]["builds"]) == (200.0, 2)
    assert (entry["all_points"]["qps"]["value"], entry["all_points"]["qps"]["builds"]) == (200.0, 3)
    assert entry["differs"] is True


@pytest.mark.parametrize("flag", ["power", "memory"])
def test_power_and_memory_flags_are_hygiene_flags(flag):
    child = curve("rust-hnsw", [(10, 0.93, 500.0)], flags=[flag])
    report = build_report([make_record([child])])
    assert report["frontier"]["rust-hnsw"]["1000"]["0.90"]["main"] == {"reached": False}


def test_build_level_flags_exclude_the_build_from_the_main_readings():
    child = curve("rust-hnsw", [(10, 0.93, 500.0)])
    child["build"]["flags"] = ["load"]
    report = build_report([make_record([child])])
    assert report["frontier"]["rust-hnsw"]["1000"]["0.90"]["main"] == {"reached": False}
    (row,) = report["scaling"]
    assert row["builds"] == 0 and row["build_ns"] is None
    assert row["all_builds"]["build_ns"]["values"] == [1e9]
    assert row["differs"] is True


def test_child_level_hygiene_flags_apply_to_every_point_of_the_build():
    child = curve("rust-hnsw", [(10, 0.93, 500.0)], flags=["thermal"])
    report = build_report([make_record([child])])
    assert report["frontier"]["rust-hnsw"]["1000"]["0.90"]["main"] == {"reached": False}
    assert report["flagged_points"][0]["flags"] == ["thermal"]


def test_identical_readings_are_not_reported_as_different():
    report = build_report([make_record([curve("rust-hnsw", RUST_CURVE)])])
    assert report["frontier"]["rust-hnsw"]["1000"]["0.90"]["differs"] is False


def test_incomplete_points_are_listed_and_only_padded_ones_enter_the_frontier():
    hnswlib = make_child("hnswlib", [make_point(10, None, None, status="incomplete"), make_point(20, 0.97, 900.0)])
    rust = make_child("rust-hnsw", [make_point(10, 0.92, 950.0, incomplete=True)])
    report = build_report([make_record([hnswlib, rust])])
    assert report["frontier"]["hnswlib"]["1000"]["0.90"]["main"]["qps"]["ef_search"] == 20
    padded = report["frontier"]["rust-hnsw"]["1000"]["0.90"]["main"]["qps"]
    assert padded["incomplete"] is True
    listed = {(p["implementation"], p["ef_search"], p["status"]) for p in report["incomplete_points"]}
    assert listed == {("hnswlib", 10, "incomplete"), ("rust-hnsw", 10, "ok")}


def test_failed_children_are_listed():
    crashed = make_child("hnswlib", [make_point(10, 0.93, 500.0)], status="signal")
    report = build_report([make_record([crashed])])
    assert report["failed_children"] == [
        {"implementation": "hnswlib", "n": 1000, "m": 16, "ef_construction": 100, "seed": 1, "shuffle": None,
         "status": "signal", "partial": True}
    ]


# ---- 010 §4：variance ----


def test_order_variance_uses_the_first_seed_with_its_shuffles_and_is_separate():
    children = [curve("faiss-hnsw", [(10, r, q)], seed=s) for s, r, q in [(1, 0.90, 100.0), (2, 0.92, 120.0)]]
    children += [curve("faiss-hnsw", [(10, r, q)], seed=1, shuffle=sh) for sh, r, q in [(7, 0.80, 90.0), (8, 0.85, 95.0)]]
    report = build_report([make_record(children)])
    (order,) = report["order_variance"]
    assert order["implementation"] == "faiss-hnsw"
    assert order["shuffles"] == [None, 7, 8]
    assert order["recall10"]["values"] == [0.90, 0.80, 0.85]
    assert order["note"]  # FAISS 的 order variance 意義不同（007 §5）
    (seed,) = report["seed_variance"]
    assert seed["recall10"]["values"] == [0.90, 0.92]


# ---- 010 §5：scaling ----


def test_scaling_reports_build_metrics_per_n_at_fixed_parameters():
    children = [
        curve("rust-hnsw", [(10, 0.95, 100.0)], n=n, seed=s, build_ns=b)
        for n, s, b in [(1000, 1, 1e9), (1000, 2, 3e9), (2000, 1, 5e9)]
    ]
    report = build_report([make_record(children)])
    rows = {(r["n"], r["implementation"]): r for r in report["scaling"]}
    assert rows[(1000, "rust-hnsw")]["build_ns"]["median"] == 2e9
    assert rows[(1000, "rust-hnsw")]["builds"] == 2 and rows[(1000, "rust-hnsw")]["differs"] is False
    assert rows[(2000, "rust-hnsw")]["build_ns"]["values"] == [5e9]
    assert set(report["frontier"]["rust-hnsw"]) == {"1000", "2000"}


# ---- 010 §6：recall 隨 efSearch ----


def test_recall_vs_ef_detects_systematic_deterioration_and_reports_the_largest_drop():
    good = curve("rust-hnsw", [(10, 0.90, 1.0), (20, 0.95, 1.0), (40, 0.94, 1.0), (80, 0.97, 1.0)])
    bad = curve("hnswlib", [(10, 0.90, 1.0), (20, 0.95, 1.0), (40, 0.85, 1.0)])
    checks = {c["implementation"]: c for c in build_report([make_record([good, bad])])["recall_vs_ef"]}
    assert checks["rust-hnsw"]["systematic_deterioration"] is False
    assert checks["rust-hnsw"]["max_adjacent_drop"] == pytest.approx(0.01)
    assert checks["hnswlib"]["systematic_deterioration"] is True
    assert checks["hnswlib"]["ef_search"] == [10, 20, 40]


# ---- 輸入檢查與輸出 ----


def test_runs_with_different_modes_or_datasets_are_not_mixed():
    with pytest.raises(ReportError, match="mode"):
        build_report([make_record([], mode="formal"), make_record([], mode="smoke")])
    with pytest.raises(ReportError, match="dataset"):
        build_report([make_record([], dataset="a"), make_record([], dataset="b")])
    with pytest.raises(ReportError, match="no runs"):
        build_report([])


def test_report_on_a_real_runner_output(tmp_path):
    result = run(stub_config(["exact-stub", "tie-stub"], ef_search=(10, 20), seeds=(1, 2), shuffles=(7,)),
                 "smoke", tmp_path / "runs", implementations=STUBS)
    out = write_report([result.run_dir], tmp_path / "report")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "smoke"
    # 機器忙碌時 smoke 的點會被標記 load 而離開主要讀值（010 §3），所以檢查包含所有點的讀值。
    assert report["frontier"]["exact-stub"]["300"]["0.99"]["all_points"]["reached"] is True
    assert report["order_variance"] and report["seed_variance"]
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "smoke" in markdown and "不得用於報告" in markdown
    if report["frontier"]["exact-stub"]["300"]["0.99"]["differs"]:
        assert "包含被標記點的讀值" in markdown
    assert (out / "recall_vs_ef_n300_m8_efc32.png").stat().st_size > 0
    assert (out / "recall_qps_n300.png").stat().st_size > 0
