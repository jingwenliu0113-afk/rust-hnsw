"""Gate profiling 的計畫（decision 015 §2）：ef 的選法、產生的指令與取樣後的 recall 核對。"""

import json
import shlex

import pytest

from benchmarks import gate_profile
from benchmarks.gate_profile import PlanError, check, plan_commands, plan_script, select_ef
from test_report import make_child, make_point, make_record

GLOVE = "glove-100-angular"


def cell(impl, curve, n=100_000, m=16, efc=200, seed=1, shuffle=None):
    return make_child(impl, [make_point(ef, r, 1.0) for ef, r in curve], n=n, m=m, efc=efc, seed=seed, shuffle=shuffle)


def formal_records(curves=None, extra=()):
    curves = curves or {
        "rust-hnsw": [(128, 0.93), (152, 0.949), (192, 0.951), (256, 0.97)],
        "hnswlib": [(128, 0.95), (152, 0.96)],
        "faiss-hnsw": [(128, 0.94), (256, 0.96)],
    }
    children = [cell(impl, curve) for impl, curve in curves.items()]
    return [make_record(children + list(extra), dataset=GLOVE)]


def test_each_implementation_gets_its_own_smallest_ef_reaching_095():
    selected = select_ef(formal_records())
    assert {impl: s["ef"] for impl, s in selected.items()} == {"rust-hnsw": 192, "hnswlib": 128, "faiss-hnsw": 256}
    assert selected["rust-hnsw"]["recall10"] == 0.951 and selected["hnswlib"]["target"] == 0.95


def test_only_the_seed_1_original_order_build_of_the_cell_is_used():
    # 其他 seed、shuffle、(M, efC) 與 n 的 build 即使 recall 更高也不影響選擇。
    distractors = [cell("rust-hnsw", [(128, 0.99)], seed=2), cell("rust-hnsw", [(128, 0.99)], shuffle=101),
                   cell("rust-hnsw", [(128, 0.99)], m=48), cell("rust-hnsw", [(128, 0.99)], n=300_000)]
    assert select_ef(formal_records(extra=distractors))["rust-hnsw"]["ef"] == 192


def test_falls_back_to_090_when_the_list_never_reaches_095():
    curves = {"rust-hnsw": [(64, 0.89), (96, 0.91), (384, 0.94)], "hnswlib": [(64, 0.95)], "faiss-hnsw": [(64, 0.95)]}
    selected = select_ef(formal_records(curves))
    assert (selected["rust-hnsw"]["ef"], selected["rust-hnsw"]["target"]) == (96, 0.90)
    assert "改用 0.9" in plan_script(formal_records(curves))


def test_refuses_when_neither_target_is_reached_or_the_build_is_missing():
    with pytest.raises(PlanError, match="reaches neither"):
        select_ef(formal_records({"rust-hnsw": [(64, 0.80)], "hnswlib": [(64, 0.95)], "faiss-hnsw": [(64, 0.95)]}))
    with pytest.raises(PlanError, match="found 0"):
        select_ef(formal_records({"rust-hnsw": [(64, 0.95)], "hnswlib": [(64, 0.95)]}))


def test_refuses_smoke_runs_and_mixed_datasets():
    smoke = formal_records()
    smoke[0]["mode"] = "smoke"
    with pytest.raises(PlanError, match="formal"):
        select_ef(smoke)
    other = make_record([], dataset="other")
    with pytest.raises(PlanError, match="mix"):
        select_ef(formal_records() + [other])


def test_failed_points_are_not_selected():
    records = formal_records()
    rust = records[0]["children"][0]
    rust["points"][2]["k10"]["status"] = "incomplete"  # ef 192
    assert select_ef(records)["rust-hnsw"]["ef"] == 256


def test_the_plan_samples_search_then_build_for_every_implementation_with_explicit_parameters():
    commands = plan_commands(GLOVE, select_ef(formal_records()))
    assert [(c[1], c[2]) for c in commands] == [
        ("gate-glove-rust-hnsw-search", "30"), ("gate-glove-hnswlib-search", "30"), ("gate-glove-faiss-hnsw-search", "30"),
        ("gate-glove-rust-hnsw-build", "300"), ("gate-glove-hnswlib-build", "300"), ("gate-glove-faiss-hnsw-build", "300")]
    search = commands[0]
    assert search[:4] == ["./scripts/profile.sh", "gate-glove-rust-hnsw-search", "30", "--"]
    assert " ".join(search[4:]) == ("--impl rust-hnsw --phase search --dataset glove-100-angular --n 100000 --m 16 "
                                    "--ef-construction 200 --seed 1 --ef 192 --k 10")
    assert "--ef" not in commands[3]
    script = plan_script(formal_records())
    assert script.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in script
    assert shlex.join(search) in script


def write_profile(tmp_path, impl, recall, ef, held=True, conditions=True):
    meta = {"recall_at_10": recall, "args": {"ef": ef}, "lock": {"held_by_ancestor": held},
            "evidence_conditions": {"power": {}} if conditions else None}
    (tmp_path / f"gate-glove-{impl}-search.json").write_text(json.dumps(meta))


def test_check_requires_the_formal_recall_lock_and_evidence_conditions(tmp_path):
    records = formal_records()
    write_profile(tmp_path, "rust-hnsw", 0.951, 192)
    write_profile(tmp_path, "hnswlib", 0.95, 128, held=False)
    write_profile(tmp_path, "faiss-hnsw", 0.9601, 256)
    results = {r["implementation"]: r for r in check(records, profiles=tmp_path)}
    assert results["rust-hnsw"]["ok"] is True
    assert results["hnswlib"]["ok"] is False      # 沒有 heavy lock
    assert results["faiss-hnsw"]["ok"] is False   # recall 與正式 run 不同
    write_profile(tmp_path, "rust-hnsw", 0.951, 192, conditions=False)
    write_profile(tmp_path, "faiss-hnsw", 0.96, 152)
    results = {r["implementation"]: r for r in check(records, profiles=tmp_path)}
    assert results["rust-hnsw"]["ok"] is False    # 沒有經過 formal 條件的檢查
    assert results["faiss-hnsw"]["ok"] is False   # ef 不是選定的 ef


def test_check_reports_missing_samples(tmp_path):
    results = check(formal_records(), profiles=tmp_path)
    assert [r["ok"] for r in results] == [False, False, False] and "not found" in results[0]["reason"]


def test_cli_plan_and_check(tmp_path, capsys, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps(formal_records()[0]))
    assert gate_profile.main(["plan", str(run_dir)]) == 0
    assert "gate-glove-hnswlib-build" in capsys.readouterr().out
    monkeypatch.setattr(gate_profile, "PROFILES", tmp_path)
    for impl, recall, ef in (("rust-hnsw", 0.951, 192), ("hnswlib", 0.95, 128), ("faiss-hnsw", 0.96, 256)):
        write_profile(tmp_path, impl, recall, ef)
    assert gate_profile.main(["check", str(run_dir)]) == 0
    write_profile(tmp_path, "hnswlib", 0.95, 128, held=False)
    assert gate_profile.main(["check", str(run_dir)]) == 1
