"""Gate profiling 的計畫（decision 015 §2）：ef 的選法、產生的指令與取樣後的證據核對。"""

import json
import os
import shlex
import subprocess

import pytest

from benchmarks import gate_profile
from benchmarks.gate_profile import PlanError, check, plan_commands, plan_script, select_ef
from test_report import make_child, make_metadata, make_point, make_record

GLOVE = "glove-100-angular"
FORMAL_COMMIT = "f" * 40


def cell(impl, curve, n=100_000, m=16, efc=200, seed=1, shuffle=None):
    return make_child(impl, [make_point(ef, r, 1.0) for ef, r in curve], n=n, m=m, efc=efc, seed=seed, shuffle=shuffle)


def formal_records(curves=None, extra=()):
    curves = curves or {
        "rust-hnsw": [(128, 0.93), (152, 0.949), (192, 0.951), (256, 0.97)],
        "hnswlib": [(128, 0.95), (152, 0.96)],
        "faiss-hnsw": [(128, 0.94), (256, 0.96)],
    }
    children = [cell(impl, curve) for impl, curve in curves.items()]
    metadata = make_metadata(commit=FORMAL_COMMIT, versions={"hnswlib": "0.8.0", "faiss-cpu": "1.15.1", "numpy": "2.5.3"})
    return [make_record(children + list(extra), dataset=GLOVE, metadata=metadata)]


# ---- ef 的選法 ----


def test_each_implementation_gets_its_own_smallest_ef_reaching_095():
    selected = select_ef(formal_records())
    assert {impl: s["ef"] for impl, s in selected.items()} == {"rust-hnsw": 192, "hnswlib": 128, "faiss-hnsw": 256}
    assert selected["rust-hnsw"]["recall10"] == 0.951 and {s["target"] for s in selected.values()} == {0.95}


def test_only_the_seed_1_original_order_build_of_the_cell_is_used():
    # 其他 seed、shuffle、(M, efC) 與 n 的 build 即使 recall 更高也不影響選擇。
    distractors = [cell("rust-hnsw", [(128, 0.99)], seed=2), cell("rust-hnsw", [(128, 0.99)], shuffle=101),
                   cell("rust-hnsw", [(128, 0.99)], m=48), cell("rust-hnsw", [(128, 0.99)], efc=400),
                   cell("rust-hnsw", [(128, 0.99)], n=300_000)]
    assert select_ef(formal_records(extra=distractors))["rust-hnsw"]["ef"] == 192


def test_all_implementations_fall_back_to_090_together():
    # rust 在清單內沒有達到 0.95：三者一起改用 0.90，取樣仍在相近的 recall（009）。
    curves = {"rust-hnsw": [(64, 0.89), (96, 0.91), (384, 0.94)], "hnswlib": [(48, 0.92), (64, 0.95)],
              "faiss-hnsw": [(48, 0.905), (64, 0.95)]}
    selected = select_ef(formal_records(curves))
    assert {impl: (s["ef"], s["target"]) for impl, s in selected.items()} == {
        "rust-hnsw": (96, 0.90), "hnswlib": (48, 0.90), "faiss-hnsw": (48, 0.90)}
    assert "三者一起改用 0.9" in plan_script(formal_records(curves))


def test_refuses_when_neither_target_is_reached_or_the_build_is_missing():
    with pytest.raises(PlanError, match="reach neither"):
        select_ef(formal_records({"rust-hnsw": [(64, 0.80)], "hnswlib": [(64, 0.95)], "faiss-hnsw": [(64, 0.95)]}))
    with pytest.raises(PlanError, match="found 0"):
        select_ef(formal_records({"rust-hnsw": [(64, 0.95)], "hnswlib": [(64, 0.95)]}))


def test_refuses_smoke_runs_and_mixed_datasets():
    smoke = formal_records()
    smoke[0]["mode"] = "smoke"
    with pytest.raises(PlanError, match="formal"):
        select_ef(smoke)
    with pytest.raises(PlanError, match="mix"):
        select_ef(formal_records() + [make_record([], dataset="other")])
    with pytest.raises(PlanError, match="no runs"):
        select_ef([])


def test_failed_points_are_not_selected():
    records = formal_records()
    records[0]["children"][0]["points"][2]["k10"]["status"] = "incomplete"  # rust 的 ef 192
    assert select_ef(records)["rust-hnsw"]["ef"] == 256


# ---- 計畫 ----


def test_the_plan_samples_search_then_build_with_every_parameter_explicit():
    commands = plan_commands(GLOVE, select_ef(formal_records()))
    assert [(c[1], c[2]) for c in commands] == [
        ("gate-glove-rust-hnsw-search", "30"), ("gate-glove-hnswlib-search", "30"), ("gate-glove-faiss-hnsw-search", "30"),
        ("gate-glove-rust-hnsw-build", "300"), ("gate-glove-hnswlib-build", "300"), ("gate-glove-faiss-hnsw-build", "300")]
    assert commands[0][:4] == ["./scripts/profile.sh", "gate-glove-rust-hnsw-search", "30", "--"]
    assert " ".join(commands[0][4:]) == ("--impl rust-hnsw --phase search --dataset glove-100-angular --n 100000 --m 16 "
                                         "--ef-construction 200 --seed 1 --ef 192 --k 10")
    assert " ".join(commands[4][4:]) == ("--impl hnswlib --phase build --dataset glove-100-angular --n 100000 --m 16 "
                                         "--ef-construction 200 --seed 1")


def test_the_plan_refuses_to_overwrite_an_earlier_attempt(tmp_path):
    script = plan_script(formal_records())
    assert script.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in script
    assert shlex.join(plan_commands(GLOVE, select_ef(formal_records()))[0]) in script
    guard = next(line for line in script.splitlines() if line.startswith("for f in"))
    for name in ("gate-glove-rust-hnsw-search.json", "gate-glove-faiss-hnsw-build.sample.txt"):
        assert f"results/profiles/{name}" in guard
    # 守衛真的會擋下：在暫存目錄裡放一個舊檔，執行守衛那一行。
    (tmp_path / "results" / "profiles").mkdir(parents=True)
    (tmp_path / "results" / "profiles" / "gate-glove-hnswlib-build.top.txt").write_text("old")
    blocked = subprocess.run(["bash", "-c", f"set -e; cd {shlex.quote(str(tmp_path))}; {guard}"], capture_output=True,
                             text=True)
    assert blocked.returncode == 1 and "exists" in blocked.stderr
    (tmp_path / "results" / "profiles" / "gate-glove-hnswlib-build.top.txt").unlink()
    assert subprocess.run(["bash", "-c", f"set -e; cd {shlex.quote(str(tmp_path))}; {guard}"]).returncode == 0


# ---- 取樣後的核對 ----


def good_meta(impl, phase, ef=None, recall=None):
    args = {"impl": impl, "phase": phase, "dataset": GLOVE, "n": 100_000, "m": 16, "ef_construction": 200, "seed": 1,
            "ef": ef, "k": 10 if phase == "search" else None, "seconds": 33.0, "meta": "x.json"}
    return {"args": args, "recall_at_10": recall, "lock": {"held_by_ancestor": True}, "evidence_conditions": {"power": {}},
            "git": {"commit": "a" * 40, "status": []}, "build_info": make_metadata()["build_info"],
            "versions": {"hnswlib": "0.8.0", "faiss-cpu": "1.15.1", "numpy": "2.5.3"}}


SELECTED = {"rust-hnsw": (192, 0.951), "hnswlib": (128, 0.95), "faiss-hnsw": (256, 0.96)}


def write_samples(directory, overrides=None):
    """六個取樣的完整輸出；overrides[(impl, phase)] 改寫紀錄的欄位。"""
    for impl, (ef, recall) in SELECTED.items():
        for phase in ("search", "build"):
            meta = good_meta(impl, phase, ef if phase == "search" else None, recall if phase == "search" else None)
            for key, value in (overrides or {}).get((impl, phase), {}).items():
                meta[key] = value
            stem = directory / f"gate-glove-{impl}-{phase}"
            stem.with_name(stem.name + ".json").write_text(json.dumps(meta))
            for suffix in (".sample.txt", ".top.txt"):
                stem.with_name(stem.name + suffix).write_text("sample")


def problems(tmp_path, overrides=None, changed=()):
    write_samples(tmp_path, overrides)
    results = check(formal_records(), profiles=tmp_path, changed_files=lambda a, b: list(changed))
    return {(r["implementation"], r["phase"]): r["problems"] for r in results}


def test_a_complete_set_of_samples_passes(tmp_path):
    found = problems(tmp_path)
    assert len(found) == 6 and all(p == [] for p in found.values())


@pytest.mark.parametrize("key, value, message", [
    ("lock", {"held_by_ancestor": False}, "heavy lock"),
    ("evidence_conditions", None, "power / background"),
    ("git", {"commit": "a" * 40, "status": [" M crates/hnsw/src/lib.rs"]}, "git status"),
    ("git", {"status": []}, "no git commit"),
    ("build_info", {"profile": "debug"}, "BUILD_INFO"),
    ("versions", {"hnswlib": "0.8.1", "faiss-cpu": "1.15.1", "numpy": "2.5.3"}, "package versions"),
])
def test_each_evidence_condition_is_checked_in_both_phases(tmp_path, key, value, message):
    for phase in ("search", "build"):
        found = problems(tmp_path, {("hnswlib", phase): {key: value}})
        assert any(message in p for p in found[("hnswlib", phase)]), found[("hnswlib", phase)]
        assert found[("rust-hnsw", phase)] == []


@pytest.mark.parametrize("field, value", [("impl", "rust-hnsw"), ("n", 10_000), ("m", 32), ("ef_construction", 400),
                                          ("seed", 2), ("dataset", "other"), ("ef", 64), ("k", 1)])
def test_every_parameter_must_match_the_plan(tmp_path, field, value):
    args = good_meta("faiss-hnsw", "search", 256, 0.96)["args"]
    found = problems(tmp_path, {("faiss-hnsw", "search"): {"args": {**args, field: value}}})
    assert any(f"args.{field}" in p for p in found[("faiss-hnsw", "search")])


def test_the_build_phase_must_not_carry_search_parameters(tmp_path):
    args = good_meta("rust-hnsw", "build")["args"]
    found = problems(tmp_path, {("rust-hnsw", "build"): {"args": {**args, "seed": 2}}})
    assert any("args.seed" in p for p in found[("rust-hnsw", "build")])


def test_the_sampled_recall_must_equal_the_formal_run(tmp_path):
    found = problems(tmp_path, {("faiss-hnsw", "search"): {"recall_at_10": 0.9601}})
    assert any("Recall@10" in p for p in found[("faiss-hnsw", "search")])


def test_measurement_code_must_be_unchanged_since_the_formal_runs(tmp_path):
    write_samples(tmp_path)
    calls = []

    def changed(a, b):
        calls.append((a, b))
        return ["crates/hnsw/src/search.rs"]

    results = check(formal_records(), profiles=tmp_path, changed_files=changed)
    assert all(any("profiled code differs" in p for p in r["problems"]) for r in results)
    assert calls[0] == (FORMAL_COMMIT, "a" * 40)


def test_missing_or_stale_outputs_are_reported(tmp_path):
    write_samples(tmp_path)
    (tmp_path / "gate-glove-hnswlib-build.sample.txt").unlink()
    stale = tmp_path / "gate-glove-rust-hnsw-search.top.txt"
    meta = tmp_path / "gate-glove-rust-hnsw-search.json"
    os.utime(stale, (meta.stat().st_mtime - 60, meta.stat().st_mtime - 60))
    (tmp_path / "gate-glove-faiss-hnsw-build.json").unlink()
    results = {(r["implementation"], r["phase"]): r for r in
               check(formal_records(), profiles=tmp_path, changed_files=lambda a, b: [])}
    assert any("not found" in p for p in results[("hnswlib", "build")]["problems"])
    assert any("older than" in p for p in results[("rust-hnsw", "search")]["problems"])
    assert results[("faiss-hnsw", "build")]["ok"] is False


def test_cli_plan_and_check(tmp_path, capsys, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps(formal_records()[0]))
    assert gate_profile.main(["plan", str(run_dir)]) == 0
    assert "gate-glove-hnswlib-build" in capsys.readouterr().out
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    monkeypatch.setattr(gate_profile, "PROFILES", profiles)
    monkeypatch.setattr(gate_profile, "changed_profiled_files", lambda a, b: [])
    write_samples(profiles)
    assert gate_profile.main(["check", str(run_dir)]) == 0
    write_samples(profiles, {("hnswlib", "build"): {"lock": {"held_by_ancestor": False}}})
    assert gate_profile.main(["check", str(run_dir)]) == 1


def test_only_the_profiled_code_is_compared_with_the_formal_commits():
    # 017 改了 runner 的有效性規則（run.py、protocol.py、environment.py），但被取樣的程式沒有變（015 補記）。
    assert gate_profile.changed_profiled_files("f077de9", "41e4ea0") == []
    for path in ("benchmarks/run.py", "benchmarks/protocol.py", "benchmarks/environment.py", "benchmarks/profile.py"):
        assert path not in gate_profile.PROFILED_PATHS
    # ddcb27d → 7961673 新增了 HNSW（crates/）：真的有差異時要列出。
    assert any(name.startswith("crates/") for name in gate_profile.changed_profiled_files("ddcb27d", "7961673"))
    with pytest.raises(PlanError, match="cannot compare"):
        gate_profile.changed_profiled_files("0" * 40, "41e4ea0")
