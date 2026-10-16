"""Advanced C 的 profiling（decision 019 §8）：分類規則、取樣計畫、取樣後的核對與 H1 前提的反組譯檢查。"""

import json
import shlex
import subprocess

import pytest

from benchmarks import gate_profile, quant_profile
from benchmarks.gate_profile import PlanError
from benchmarks.profile_report import Row, top_of_stack
from benchmarks.quant_profile import categorize, check, check_kernel, classify, kernel_bodies, plan_script, select_samples
from benchmarks.quant_report import GLOVE, TEXT, F, R, S
from test_report import make_child, make_metadata, make_point, make_record

ADAPTER = {"manufacturer": "Apple Inc.", "name": "140W USB-C Power Adapter"}


def row(name, image="rust_hnsw.cpython-313-darwin.so", symbol=None, count=1):
    return Row(name, symbol or name, image, count, 0.0)


# ---- 分類規則（取樣之前固定，依序取第一個符合的類別） ----


@pytest.mark.parametrize("item, category", [
    # 1. kernel：Row.name 或 Row.symbol 含 rhnsw_kernel_
    (row("hnsw::hnsw::quant::rhnsw_kernel_sq8"), "kernel"),
    (row("hnsw::hnsw::quant::rhnsw_kernel_fp32"), "kernel"),
    (row("pretty::name", symbol="__RNvNtNtCs5kyc4sahh7U_4hnsw4hnsw5quant16rhnsw_kernel_sq8"), "kernel"),
    # 依序取第一個符合的類別：kernel 優先於其他類別、heap 優先於 traversal
    (row("hnsw::hnsw::quant::search_layer_q::rhnsw_kernel_fp32"), "kernel"),
    (row("hashbrown::map::rhnsw_kernel_sq8"), "kernel"),
    (row("alloc::collections::binary_heap::search_layer"), "heap"),
    (row("hashbrown::search_layer"), "visited_set"),
    # 2. visited set：以 hashbrown:: 開頭，或含 core::hash::sip（Gate profiling 的實際名稱）
    (row("hashbrown::map::HashMap::std::hash::random::RandomState::insert::hnsw"), "visited_set"),
    (row("hashbrown::raw::RawTable::reserve_rehash::map::make_hasher::std::hash::random::RandomState::hnsw"), "visited_set"),
    (row("core::hash::sip::Hasher::write"), "visited_set"),
    (row("std::collections::core::hash::sip::x"), "visited_set"),
    (row("std::hashbrown::map::insert"), "other"),  # 不是以 hashbrown:: 開頭
    # 3. heap：以 alloc::collections::binary_heap:: 開頭
    (row("alloc::collections::binary_heap::hnsw::Candidate::pop"), "heap"),
    (row("alloc::collections::binary_heap::cmp::Reverse::hnsw::hnsw::Candidate::pop"), "heap"),
    (row("x::alloc::collections::binary_heap::pop"), "other"),
    # 4. traversal：含 search_layer、greedy_search 或 search_unit
    (row("hnsw::hnsw::HnswIndex::search_layer"), "traversal"),
    (row("hnsw::hnsw::HnswIndex::greedy_search"), "traversal"),
    (row("hnsw::hnsw::HnswIndex::search_unit"), "traversal"),
    (row("hnsw::hnsw::quant::search_layer_q"), "traversal"),
    # F、S 的走訪是 #[inline(never)] 的泛型函式：binding 裡的名稱帶著單態化的型別參數
    (row("hnsw::hnsw::quant::QuantIndex::search_layer_q::Fp32Store::NoCount"), "traversal"),
    (row("hnsw::hnsw::quant::QuantIndex::greedy_search_q::Sq8Store::NoCount"), "traversal"),
    (row("hnsw::hnsw::quant::QuantIndex::search_unit_q::Sq8Store::NoCount"), "traversal"),
    # search_batch 本身（query 迴圈、prepare、push_padded_row）不含這三個名稱：歸入其他（與 Core 的 search_batch 相同）
    (row("hnsw::hnsw::quant::QuantIndex::search_batch"), "other"),
    (row("hnsw::hnsw::HnswIndex::search_batch"), "other"),
    # 5. 記憶體配置：映像檔是 libsystem_malloc.dylib，或符號以 _platform_mem 開頭
    (row("_szone_free", image="libsystem_malloc.dylib"), "allocation"),
    (row("??? +0x1234", image="libsystem_malloc.dylib", symbol="??? +0x1234"), "allocation"),
    (row("_platform_memmove", image="libsystem_platform.dylib"), "allocation"),
    (row("_platform_memset", image="libsystem_platform.dylib"), "allocation"),
    (row("x_platform_memmove", image="libsystem_platform.dylib"), "other"),
    # 6. 其他：Python / FFI、沒有符號的列與 hnsw::batch::unit_rows
    (row("hnsw::batch::unit_rows"), "other"),
    (row("_PyEval_EvalFrameDefault", image="Python"), "other"),
    (row("??? +0x10", symbol="??? +0x10"), "other"),
    (row("??? [0x1]", image="<unknown binary>", symbol="??? [0x1]"), "other"),
])
def test_classification_follows_the_registered_rules(item, category):
    assert classify(item) == category


def test_categorize_counts_every_row_and_lists_other_only_above_5_percent():
    rows = [row("hnsw::hnsw::quant::rhnsw_kernel_sq8", count=60), row("hnsw::hnsw::HnswIndex::search_layer", count=30),
            row("hashbrown::map::x", count=5), row("py::a", image="Python", count=3), row("py::b", image="Python", count=2)]
    result = categorize(rows)
    assert result["total"] == 100
    assert result["counts"] == {"kernel": 60, "visited_set": 5, "heap": 0, "traversal": 30, "allocation": 0, "other": 5}
    assert result["other_top"] == []  # 恰好 5%：「超過 5%」才列出
    rows.append(row("py::c", image="Python", count=1))
    top = categorize(rows)["other_top"]
    assert [t["name"] for t in top] == ["py::a", "py::b", "py::c"]
    many = [row(f"py::{i}", image="Python", count=20 - i) for i in range(12)] + [row("k::rhnsw_kernel_fp32", count=10)]
    assert len(categorize(many)["other_top"]) == 10
    assert categorize(many)["other_top"][0]["name"] == "py::0"
    # 輸入不依樣本數排序時，前 10 名仍依樣本數遞減。
    shuffled = [row(f"py::{i}", image="Python", count=c) for i, c in enumerate((3, 9, 1, 12, 5, 7, 2, 11, 4, 8, 6, 10))]
    top = categorize(shuffled)["other_top"]
    assert [t["count"] for t in top] == [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    assert top[0]["share"] == pytest.approx(12 / 78)


SAMPLE = """Analysis of sampling python (pid 1) every 1 millisecond
Sort by top of stack, same collapsed (when >= 5):
        __RNvNtNtCs5kyc4sahh7U_4hnsw4hnsw5quant16rhnsw_kernel_sq8  (in rust_hnsw.cpython-313-darwin.so)        700
        __RNvMs_NtCs5kyc4sahh7U_4hnsw4hnswNtB4_9HnswIndex12search_layer  (in rust_hnsw.cpython-313-darwin.so)        200
        _szone_free  (in libsystem_malloc.dylib)        40
        ???  (in rust_hnsw.cpython-313-darwin.so)  load address 0x100000000 + 0x2a3c  [0x100002a3c]        60

Binary Images:
"""


def test_the_full_ranking_is_read_from_the_sample_report(tmp_path):
    path = tmp_path / "x.sample.txt"
    path.write_text(SAMPLE)
    rows = quant_profile.read_rows(path)
    assert [classify(r) for r in rows] == ["kernel", "traversal", "allocation", "other"]
    assert categorize(rows)["counts"]["other"] == 60 and categorize(rows)["total"] == 1000
    assert rows == top_of_stack(SAMPLE)


# ---- 取樣計畫 ----


def cell(impl, curve, seed=1, flags=()):
    return make_child(impl, [make_point(ef, r, 1000.0, flags=flags) for ef, r in curve], n=100_000, m=16, efc=200, seed=seed)


def c_records(s_curve=None, dataset=GLOVE, extra=(), e=304):
    s_curve = s_curve or [(256, 0.93), (e, 0.94), (384, 0.949), (512, 0.951), (768, 0.97)]
    children = [cell(R, [(256, 0.95), (e, 0.956)]), cell(F, [(256, 0.95), (e, 0.956)]), cell(S, s_curve), *extra]
    metadata = make_metadata(commit="f" * 40, versions={"hnswlib": "0.8.0", "faiss-cpu": "1.15.1", "numpy": "2.5.3"})
    metadata["formal_checks"] = {"power": {"powermode": 2, "adapter": ADAPTER}}
    return [make_record(children, dataset=dataset, metadata=metadata)]


def test_samples_are_r_f_s_at_e_star_and_s_at_its_own_095_ef():
    samples, notes = select_samples(c_records())
    assert [(s["implementation"], s["ef"], s["role"]) for s in samples] == [
        (R, 304, "e_star"), (F, 304, "e_star"), (S, 304, "e_star"), (S, 512, "same_recall_0.95")]
    assert samples[2]["recall10"] == 0.94 and samples[3]["recall10"] == 0.951 and notes == []


def test_the_same_recall_sample_falls_back_to_090_and_is_skipped_when_equal_or_unreached():
    samples, _ = select_samples(c_records([(256, 0.85), (304, 0.89), (384, 0.90), (768, 0.94)]))
    assert (samples[-1]["ef"], samples[-1]["role"]) == (384, "same_recall_0.90")
    samples, notes = select_samples(c_records([(256, 0.94), (304, 0.95), (768, 0.97)]))
    assert len(samples) == 3 and "略過" in notes[0]
    samples, notes = select_samples(c_records([(256, 0.80), (304, 0.85), (768, 0.89)]))
    assert len(samples) == 3 and "沒有達到" in notes[0]


def test_only_the_seed_1_build_selects_the_samples():
    other_seed = cell(S, [(256, 0.99), (304, 0.99)], seed=2)
    samples, _ = select_samples(c_records(extra=[other_seed]))
    assert samples[-1]["ef"] == 512


def test_the_plan_refuses_without_an_e_star_point_and_for_other_datasets():
    with pytest.raises(PlanError, match="e\\*"):
        select_samples(c_records(e=300))
    with pytest.raises(PlanError, match="not a dataset of Advanced C"):
        select_samples(c_records(dataset="smoke-gaussian"))
    records = c_records(dataset=TEXT, e=48, s_curve=[(32, 0.90), (48, 0.93), (64, 0.96)])
    assert [s["ef"] for s in select_samples(records)[0]] == [48, 48, 48, 64]


def test_the_plan_samples_search_only_with_every_parameter_explicit(tmp_path):
    script = plan_script(c_records())
    commands = [line for line in script.splitlines() if line.startswith("./scripts/profile.sh")]
    assert [shlex.split(c)[1:3] for c in commands] == [
        ["quant-glove-rust-hnsw-ef304", "30"], ["quant-glove-rust-hnsw-fp32g-ef304", "30"],
        ["quant-glove-rust-hnsw-sq8-ef304", "30"], ["quant-glove-rust-hnsw-sq8-ef512", "30"]]
    assert " ".join(shlex.split(commands[3])[4:]) == (
        "--impl rust-hnsw-sq8 --phase search --dataset glove-100-angular --n 100000 --m 16 --ef-construction 200 "
        "--seed 1 --ef 512 --k 10")
    assert "machine_quiet --powermode 2" in script and "faiss" not in "".join(commands)
    guard = next(line for line in script.splitlines() if line.startswith("for f in"))
    (tmp_path / "results" / "profiles").mkdir(parents=True)
    (tmp_path / "results" / "profiles" / "quant-glove-rust-hnsw-sq8-ef512.sample.txt").write_text("old")
    blocked = subprocess.run(["bash", "-c", f"set -e; cd {shlex.quote(str(tmp_path))}; {guard}"], capture_output=True, text=True)
    assert blocked.returncode == 1 and "exists" in blocked.stderr


def test_plan_and_check_need_the_runs_of_all_ten_c_formal_configs(tmp_path, capsys):
    # 019 §9 第 8 步：profiling 在全部 formal（與診斷）之後。GloVe 的 5 份完整、文字還沒有跑時，plan 與 check 都拒絕。
    from test_quant_report import Scenario, materialize

    runs = [str(d) for d in materialize(Scenario(), tmp_path)["run_dirs"]]
    glove = [d for d in runs if "glove" in d]
    for command in ("plan", "check"):
        with pytest.raises(SystemExit, match="both datasets"):
            quant_profile.main([command, "--dataset", "glove", *glove])
    with pytest.raises(SystemExit):
        quant_profile.main(["plan", *runs])  # 沒有 --dataset
    assert quant_profile.main(["plan", "--dataset", "glove", *runs]) == 0
    script = capsys.readouterr().out
    assert "quant-glove-rust-hnsw-sq8-ef304" in script and "quant-text" not in script


# ---- 取樣後的核對 ----


def good_meta(impl, ef, recall):
    args = {"impl": impl, "phase": "search", "dataset": GLOVE, "n": 100_000, "m": 16, "ef_construction": 200, "seed": 1,
            "ef": ef, "k": 10, "seconds": 33.0, "meta": "x.json"}
    return {"args": args, "recall_at_10": recall, "lock": {"held_by_ancestor": True},
            "evidence_conditions": {"power": {"powermode": 2, "adapter": ADAPTER}},
            "git": {"commit": "a" * 40, "status": []}, "build_info": make_metadata()["build_info"],
            "versions": {"hnswlib": "0.8.0", "faiss-cpu": "1.15.1", "numpy": "2.5.3"}}


SAMPLES = {(R, 304): 0.956, (F, 304): 0.956, (S, 304): 0.94, (S, 512): 0.951}


def write_samples(directory, overrides=None, sample_text=SAMPLE):
    for (impl, ef), recall in SAMPLES.items():
        meta = good_meta(impl, ef, recall)
        for key, value in (overrides or {}).get((impl, ef), {}).items():
            meta[key] = value
        stem = directory / f"quant-glove-{impl}-ef{ef}"
        stem.with_name(stem.name + ".json").write_text(json.dumps(meta))
        stem.with_name(stem.name + ".sample.txt").write_text(sample_text)
        stem.with_name(stem.name + ".top.txt").write_text("ranking")


def problems(tmp_path, overrides=None, sample_text=SAMPLE, records=None):
    write_samples(tmp_path, overrides, sample_text)
    results = check(records or c_records(), profiles=tmp_path, changed_files=lambda a, b: [])
    return {(r["implementation"], r["ef"]): r["problems"] for r in results}


def test_a_complete_set_of_samples_passes(tmp_path):
    found = problems(tmp_path)
    assert set(found) == set(SAMPLES) and all(p == [] for p in found.values())


@pytest.mark.parametrize("key, value, message", [
    ("lock", {"held_by_ancestor": False}, "heavy lock"),
    ("evidence_conditions", None, "power / background"),
    ("git", {"commit": "a" * 40, "status": [" M x"]}, "git status"),
    ("build_info", {"profile": "debug"}, "BUILD_INFO"),
    ("versions", {"hnswlib": "0.8.0", "faiss-cpu": "1.15.2", "numpy": "2.5.3"}, "package versions"),
    ("recall_at_10", 0.9511, "Recall@10"),
    ("evidence_conditions", {"power": {"powermode": 0, "adapter": ADAPTER}}, "high-performance"),
    ("evidence_conditions", {"power": {"powermode": 2, "adapter": {"manufacturer": "Apple Inc.",
                                                                   "name": "70W USB-C Power Adapter"}}}, "power adapter"),
])
def test_each_evidence_condition_is_checked(tmp_path, key, value, message):
    found = problems(tmp_path, {(S, 512): {key: value}})
    assert any(message in p for p in found[(S, 512)]), found[(S, 512)]
    assert found[(S, 304)] == []


@pytest.mark.parametrize("field, value", [("impl", F), ("ef", 304), ("seed", 2), ("k", 1), ("phase", "build"), ("n", 10)])
def test_every_parameter_must_match_the_plan(tmp_path, field, value):
    args = good_meta(S, 512, 0.951)["args"]
    found = problems(tmp_path, {(S, 512): {"args": {**args, field: value}}})
    assert any(f"args.{field}" in p for p in found[(S, 512)])


def test_profiled_code_must_match_the_formal_commits(tmp_path):
    write_samples(tmp_path)
    results = check(c_records(), profiles=tmp_path, changed_files=lambda a, b: ["crates/hnsw/src/hnsw/quant.rs"])
    assert all(any("profiled code differs" in p for p in r["problems"]) for r in results)


def test_an_unparseable_or_empty_ranking_is_not_evidence(tmp_path):
    found = problems(tmp_path, sample_text="Sort by top of stack\n   garbage line\n\n")
    assert all(any("unparsed" in p for p in v) for v in found.values())
    found = problems(tmp_path, sample_text="Sort by top of stack\n\n")
    assert all(any("empty top-of-stack" in p for p in v) for v in found.values())


def test_missing_samples_are_reported(tmp_path):
    write_samples(tmp_path)
    (tmp_path / "quant-glove-rust-hnsw-ef304.json").unlink()
    results = {(r["implementation"], r["ef"]): r for r in check(c_records(), profiles=tmp_path, changed_files=lambda a, b: [])}
    assert results[(R, 304)]["ok"] is False and "not found" in results[(R, 304)]["problems"][0]
    assert results[(F, 304)]["ok"] is True


def test_mixed_power_adapters_in_the_formal_runs_fail_the_check(tmp_path):
    records = c_records() + c_records()
    records[1]["metadata"]["formal_checks"]["power"]["adapter"] = {"manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"}
    records[1]["children"] = []
    found = problems(tmp_path, records=records)
    assert all(any("power adapter" in p for p in v) for v in found.values())


def test_gate_profile_default_behaviour_is_unchanged():
    # 019 §4：gate_profile 只抽出 check_evidence；名稱、實作與階段都不變。
    assert gate_profile.IMPLEMENTATIONS == ("rust-hnsw", "hnswlib", "faiss-hnsw")
    assert gate_profile.profile_name(GLOVE, "rust-hnsw", "search") == "gate-glove-rust-hnsw-search"
    assert gate_profile.PHASES == ("search", "build")


# ---- H1 前提：反組譯（019 §2） ----


def ins(text):
    mnemonic, _, operands = text.partition(" ")
    return mnemonic, operands


FP32_OK = [ins("movi.2d v0, #0"), ins("fmul.4s v1, v1, v2"), ins("mov s2, v1[1]"), ins("fadd s0, s0, s1"),
           ins("fadd s0, s0, s2"), ins("fadd s0, s0, s3"), ins("fmov s1, #1.00000000"), ins("fsub s0, s1, s0"), ins("ret ")]
SQ8_OK = [ins("tbl.16b v3, { v3 }, v4"), ins("ushll.8h v3, v3, #0x0"), ins("ucvtf.4s v3, v3"), ins("fmul.4s v3, v3, v5"),
          ins("fadd s1, s1, s3"), ins("fadd s1, s1, s7"), ins("ucvtf s3, w9"), ins("fadd s1, s1, s3"), ins("fadd s0, s0, s1"),
          ins("fmov s1, #1.00000000"), ins("fsub s0, s1, s0")]


def test_sequential_scalar_accumulation_passes():
    fp32 = check_kernel(FP32_OK, final_adds=0)
    assert fp32["ok"] and fp32["accumulator"] == "s0" and fp32["accumulations"] == 3 and fp32["violations"] == []
    sq8 = check_kernel(SQ8_OK, final_adds=1)
    assert sq8["ok"] and sq8["accumulator"] == "s1" and sq8["final_adds"] == ["fadd s0, s0, s1"]


@pytest.mark.parametrize("bad", ["fadd.4s v0, v0, v1", "faddp.4s v0, v0, v1", "faddp s0, v1.2s", "fadda s0, p0, s0, z1.s",
                                 "fmadd s0, s1, s2, s0", "fmla.4s v0, v1, v2", "fadd d0, d0, d1", "fadd.2d v0, v0, v1"])
def test_vector_pairwise_strict_or_fused_accumulation_fails(bad):
    result = check_kernel([*FP32_OK[:3], ins(bad), *FP32_OK[3:]], final_adds=0)
    assert not result["ok"] and any(bad.split()[0] in v for v in result["violations"])


def test_the_accumulator_must_be_a_single_register():
    two = [ins("fadd s0, s0, s1"), ins("fadd s0, s0, s2"), ins("fadd s3, s3, s4"), ins("fadd s3, s3, s5")]
    assert not check_kernel(two, final_adds=0)["ok"]
    stray = [ins("fadd s0, s0, s1"), ins("fadd s0, s0, s2"), ins("fadd s3, s3, s4")]
    assert any("not an accumulation" in v for v in check_kernel(stray, final_adds=0)["violations"])
    commuted = [ins("fadd s0, s0, s1"), ins("fadd s0, s0, s2"), ins("fadd s0, s2, s0")]
    assert not check_kernel(commuted, final_adds=0)["ok"]
    assert not check_kernel([ins("fmul s0, s0, s1")], final_adds=0)["ok"]  # 沒有累加


def test_a_tie_between_two_accumulators_fails():
    # 只有一次累加時，s1 與 s0 的鏈一樣長：不能確定哪一個是累加暫存器 → 不符。
    assert not check_kernel([ins("fadd s1, s1, s3"), ins("fadd s0, s0, s1")], final_adds=1)["ok"]


def test_a_commuted_add_is_not_an_accumulation_even_when_a_final_add_is_allowed():
    chain = [ins("fadd s1, s1, s2"), ins("fadd s1, s1, s4"), ins("fadd s1, s3, s1")]
    result = check_kernel(chain, final_adds=1)
    assert not result["ok"] and result["final_adds"] == []


def test_the_final_add_count_is_fixed_per_kernel():
    assert not check_kernel(SQ8_OK, final_adds=0)["ok"]  # FP32 不得有 b + Σ
    assert not check_kernel([i for i in SQ8_OK if i != ins("fadd s0, s0, s1")], final_adds=1)["ok"]
    assert not check_kernel([*SQ8_OK, ins("fadd s2, s1, s0")], final_adds=1)["ok"]


OTOOL = """/x/rust_hnsw.cpython-313-darwin.so:
(__TEXT,__text) section
__RNvNtNtCs5kyc4sahh7U_4hnsw4hnsw5quant16rhnsw_kernel_sq8:
0000000000001000\ttbl.16b\tv3, { v3 }, v4
0000000000001004\tucvtf.4s\tv3, v3
0000000000001008\tfadd\ts1, s1, s3
000000000000100c\tfadd\ts1, s1, s4
0000000000001010\tfadd\ts0, s0, s1
0000000000001014\tret
__RNvNtNtCs5kyc4sahh7U_4hnsw4hnsw5quant16sq8_train_encode:
0000000000002000\tfadd.4s\tv0, v0, v1
__RNvNtNtCs5kyc4sahh7U_4hnsw4hnsw5quant17rhnsw_kernel_fp32:
0000000000003000\tfmul.4s\tv1, v1, v2
0000000000003004\tfadd\ts0, s0, s1 ; comment
0000000000003008\tfadd\ts0, s0, s2
000000000000300c\tret
__RNvXs0_other:
0000000000004000\tfaddp.4s\tv0, v0, v1
"""


def test_the_kernels_are_extracted_from_otool_output():
    bodies = kernel_bodies(OTOOL)
    assert sorted(bodies) == ["__RNvNtNtCs5kyc4sahh7U_4hnsw4hnsw5quant16rhnsw_kernel_sq8",
                              "__RNvNtNtCs5kyc4sahh7U_4hnsw4hnsw5quant17rhnsw_kernel_fp32"]
    fp32 = bodies["__RNvNtNtCs5kyc4sahh7U_4hnsw4hnsw5quant17rhnsw_kernel_fp32"]
    assert fp32 == [("fmul.4s", "v1, v1, v2"), ("fadd", "s0, s0, s1"), ("fadd", "s0, s0, s2"), ("ret", "")]
    result = quant_profile.premise(OTOOL)
    assert result["status"] == "成立"
    assert result["kernels"]["rhnsw_kernel_sq8"]["conversions"] == {"tbl.16b": 1, "ucvtf.4s": 1}
    # 其他函式裡的向量加法不影響判定（只看兩個 kernel）。
    broken = OTOOL.replace("fadd\ts0, s0, s2", "fadd.4s\tv0, v0, v2")
    assert quant_profile.premise(broken)["status"] == "前提不成立"


def test_a_missing_or_duplicated_kernel_fails_the_premise():
    missing = OTOOL.replace("17rhnsw_kernel_fp32", "17other_function_xx")
    result = quant_profile.premise(missing)
    assert result["status"] == "前提不成立" and "found 0" in result["kernels"]["rhnsw_kernel_fp32"]["violations"][0]
    duplicated = OTOOL + "__RNvCs1_x17rhnsw_kernel_fp32:\n0000000000005000\tfadd\ts0, s0, s1\n"
    assert quant_profile.premise(duplicated)["status"] == "前提不成立"


def test_the_premise_record_binds_the_binding_and_refuses_to_overwrite(tmp_path):
    import rust_hnsw

    from benchmarks import environment

    out = tmp_path / "premise.json"
    record = quant_profile.run_premise(out)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["binding_sha256"] == environment.sha256_file(rust_hnsw.rust_hnsw.__file__)
    assert saved["build_info"] == dict(rust_hnsw.BUILD_INFO) and saved["status"] == record["status"]
    assert set(saved["kernels"]) == {"rhnsw_kernel_fp32", "rhnsw_kernel_sq8"}
    # 019 §4、§8：已安裝的 binding 裡兩個 kernel 各恰好是一個獨立的函式（#[inline(never)]）。
    # 只斷言符號存在，不斷言 ok（累加的判準是 019 §9 第 4 步的前提檢查，不在測試裡預先判定）。
    for kernel, entry in saved["kernels"].items():
        assert entry.get("symbol", "").endswith(kernel), entry
        assert not any("found" in v for v in entry["violations"]), entry["violations"]
    with pytest.raises(SystemExit, match="exists"):
        quant_profile.run_premise(out)


def test_the_installed_binding_has_separate_kernel_and_traversal_functions():
    # 019 §4、§8：F、S 的 kernel 與三個走訪函式都是獨立的符號，取樣才分得出 kernel 與 traversal 兩類；
    # 計時路徑（NoCount）的每個單態化都要在。Core 的三個走訪函式也是獨立的符號（R 的 traversal）。
    import rust_hnsw

    from benchmarks.profile_report import pretty_symbol

    text = subprocess.run(["nm", "-U", rust_hnsw.rust_hnsw.__file__], check=True, capture_output=True, text=True).stdout
    names = [pretty_symbol(line.split()[-1][1:]) for line in text.splitlines() if " t " in line or " T " in line]
    for kernel in ("rhnsw_kernel_fp32", "rhnsw_kernel_sq8"):
        assert [classify(row(n)) for n in names if n.endswith(f"quant::{kernel}")] == ["kernel"], kernel
    for function in ("search_unit_q", "greedy_search_q", "search_layer_q"):
        for store in ("Fp32Store", "Sq8Store"):
            name = f"hnsw::hnsw::quant::QuantIndex::{function}::{store}::NoCount"
            assert name in names, name
            assert classify(row(name)) == "traversal"
    for function in ("search_unit", "greedy_search", "search_layer"):
        assert f"hnsw::hnsw::HnswIndex::{function}" in names, function
