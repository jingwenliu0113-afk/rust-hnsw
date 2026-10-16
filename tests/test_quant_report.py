"""Advanced C 的報告（decision 019 §2、§6）：以手寫紀錄逐條驗證判定規則、配對讀值、一致性檢查與判定表的狀態。

`Scenario` 產生一組「每個假設都成立、每個檢查都一致」的手寫輸入（10 份 C formal run、Core 對照、診斷、取樣與反組譯），
每個測試只改動要驗證的那一處。數字都是虛構的，只用來驗證規則本身。
"""

import copy
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from benchmarks import quant_report
from benchmarks.quant_report import (
    GLOVE,
    TEXT,
    F,
    P,
    Q,
    R,
    S,
    ReportError,
    build_quant_report,
    h1_overall,
    h2_overall,
    validate_table,
)
from test_report import make_child, make_point, make_record

EFS = {GLOVE: [64, 96, 128, 152, 192, 256, 304, 384, 512, 768], TEXT: [24, 32, 48, 64, 96, 128, 192, 256, 384, 512]}
CORE_EFS = {GLOVE: EFS[GLOVE][:8], TEXT: EFS[TEXT]}
DIMS = {GLOVE: 100, TEXT: 768}
NQ = 1000
TOTAL = NQ * 10
ORDERS = {1: [S, P, F, Q, R], 2: [P, F, Q, S], 3: [F, Q, S, P], 4: [S, F], 5: [F, S]}
RHO = {GLOVE: 1.2, TEXT: 1.0}
MiB = 2**20
ADAPTER = {"manufacturer": "Apple Inc.", "name": "140W USB-C Power Adapter"}
E = quant_report.E_STAR

# 019 §6 的「允許的狀態」，在測試裡獨立寫一份（不 import ALLOWED），用來檢查 quant_report 的輸出。
SECTION_6 = {
    "H1 前提": {"成立", "前提不成立"},
    "H1a": {"成立", "不成立", "不確定", "無法判定"}, "H1b": {"成立", "不成立", "不確定", "無法判定"},
    "H1c": {"成立", "不成立", "不確定", "無法判定"}, "H1d": {"一致", "不一致", "無法判定"},
    "H1 整體": {"成立", "不成立", "計時成立、profile 不一致", "前提不成立", "不確定", "無法判定"},
    "H2 GloVe": {"成立", "不成立", "不確定", "無法判定", "不適用"}, "H2 文字": {"成立", "不成立", "不確定", "無法判定", "不適用"},
    "H2 整體": {"成立", "不成立", "不確定", "無法判定", "不適用"},
    "S 對 F": {"sq8_faster", "fp32_faster", "within_quantization", "uncertain", "undetermined"},
    "Q 對 P": {"sq8_faster", "fp32_faster", "within_quantization", "uncertain", "undetermined", "不適用"},
    "C1": {"一致", "不一致", "無法判定", "不適用"}, "C2": {"一致", "不一致", "無法判定"}, "C3": {"一致", "不一致", "無法判定"},
    "C4": {"一致", "path-distorted", "無法判定"}, "C5": {"一致", "不一致", "無法判定"}, "C6": {"只描述"},
    "C7a": {"一致", "不一致", "無法判定"}, "C7b": {"一致", "不一致", "無法判定"}, "C8": {"一致", "不一致", "無法判定", "不適用"},
}
STOPPED = "不判定（C7a 不一致）"


def section_6_key(label: str) -> str:
    for key in ("H1 前提", "H1 整體", "H2 GloVe", "H2 文字", "H2 整體", "S 對 F", "Q 對 P"):
        if key in label:
            return key
    return label.split()[0]


def recall_of(impl, i):
    return (9000 + 100 * i - (100 if impl in (S, Q) else 0)) / TOTAL


def qps_of(dataset, impl, seed, i):
    f = 4000.0 / (1 + 0.3 * i)
    rho = RHO[dataset] + 0.01 * (seed - 3)
    return {F: f, R: f, S: f * rho, P: 2 * f, Q: 2 * f * 1.1}[impl]


def theory(dataset):
    return 3 * 100_000 * DIMS[dataset] - 8 * DIMS[dataset]


def ids_of(impl, seed, ef):
    family = {F: 0, R: 0, P: 7}[impl]
    return np.full((2, 10), seed * 10_000 + ef + family, dtype=np.int64)


def counts(**values):
    base = {"kernel": 0, "visited_set": 0, "heap": 0, "traversal": 0, "allocation": 0, "other": 0}
    base.update(values)
    return {"ok": True, "problems": [], "role": "e_star", "total": sum(base.values()), "counts": base, "other_top": []}


class Scenario:
    """每個假設都成立、每個檢查都一致的基準；測試改動 records / diag / c_ids / core / profiles / premise 的某一處。"""

    def __init__(self):
        self.records, self.diag, self.c_ids, self.core, self.profiles = {}, {}, {}, {}, {}
        self.premise = {"binding_sha256": "r", "kernels": {
            "rhnsw_kernel_fp32": {"ok": True, "accumulator": "s0", "accumulations": 21, "violations": []},
            "rhnsw_kernel_sq8": {"ok": True, "accumulator": "s1", "accumulations": 69, "violations": [],
                                 "conversions": {"ucvtf.4s": 17}}}}
        for dataset in (GLOVE, TEXT):
            self.records[dataset] = {seed: self._record(dataset, seed) for seed in ORDERS}
            self.diag[dataset] = self._diag(dataset)
            self.c_ids[dataset] = {(impl, seed, ef): ids_of(impl, seed, ef) for seed, impls in ORDERS.items()
                                   for impl in impls if impl in (F, R, P) for ef in EFS[dataset]}
            self.core[dataset] = {
                "runs": {}, "efs": {(impl, seed): list(CORE_EFS[dataset]) for impl, seeds in ((R, range(1, 6)), (P, range(1, 4)))
                                    for seed in seeds},
                "ids": {(impl, seed, ef): ids_of(impl, seed, ef) for impl, seeds in ((R, range(1, 6)), (P, range(1, 4)))
                        for seed in seeds for ef in CORE_EFS[dataset]},
                "qps": {(R, 1, ef): qps_of(dataset, F, 1, i) for i, ef in enumerate(CORE_EFS[dataset])}}
        e_g, e_t = E[GLOVE], E[TEXT]
        self.profiles[GLOVE] = {"samples": {(R, e_g): counts(traversal=480, visited_set=440, heap=70, other=10),
                                            (F, e_g): counts(kernel=500, visited_set=400, heap=70, other=30),
                                            (S, e_g): counts(kernel=400, visited_set=480, heap=80, other=40)},
                                "gate": counts(traversal=480, visited_set=440, heap=70, other=10)}
        self.profiles[TEXT] = {"samples": {(R, e_t): counts(traversal=920, visited_set=60, heap=10, other=10),
                                           (F, e_t): counts(kernel=800, visited_set=60, heap=10, other=130),
                                           (S, e_t): counts(kernel=800, visited_set=60, heap=10, other=130)},
                               "gate": counts(traversal=922, visited_set=64, heap=10, other=4)}

    def _record(self, dataset, seed):
        children = []
        for impl in ORDERS[seed]:
            points = []
            for i, ef in enumerate(EFS[dataset]):
                r, q = recall_of(impl, i), qps_of(dataset, impl, seed, i)
                r1 = round(r * NQ) / NQ
                points.append({"ef_search": ef, "file": f"points/{impl}-ef{ef}.npz", "flags": [],
                               "k10": {"status": "ok", "incomplete": False, "incomplete_queries": 0, "recall": r,
                                       "tie_aware_recall": r, "qps": {"qps": [q] * 5, "qps_median": q},
                                       "latency": {"p50_ns": 1e9 / q, "p95_ns": 2e9 / q}},
                               "k1": {"status": "ok", "recall": r1, "tie_aware_recall": r1}})
            sq8 = impl in (S, Q)
            relieved = 50 * MiB + (0 if sq8 else theory(dataset))
            children.append({
                "child_id": f"g000-{impl}", "implementation": impl, "status": "ok", "partial": False, "flags": [],
                "build": {"n": 100_000, "m": 16, "ef_construction": 200, "seed": seed, "shuffle": None, "build_ns": 3e10,
                          "peak_rss_bytes": 500 * MiB, "index_footprint_bytes": relieved + 10 * MiB,
                          "index_memory_bytes": relieved + 12 * MiB, "index_footprint_relieved_bytes": relieved,
                          "index_memory_relieved_bytes": relieved, "serialized_bytes": None if impl in (F, S) else 40 * MiB,
                          "flags": []},
                "points": points})
        short = quant_report.SHORT_NAMES[dataset]
        return {"mode": "formal", "status": "complete", "status_reasons": [], "protocol": {"version": "008-v5"},
                "dataset": {"name": dataset, "n_queries": NQ, "dim": DIMS[dataset]},
                "config": {"path": f"benchmarks/configs/quant-{short}-seed-{seed}.toml", "parsed": {"ef_search": EFS[dataset]}},
                "metadata": {"extension_sha256": {"rust_hnsw": "r", "hnswlib": "h", "faiss": "f"},
                             "versions": {"hnswlib": "0.8.0", "faiss-cpu": "1.15.1", "numpy": "2.5.3"},
                             "build_info": {"profile": "release", "opt_level": "3"}, "git": {"commit": "c" * 40},
                             "formal_checks": {"config": {"path": f"benchmarks/configs/quant-{short}-seed-{seed}.toml",
                                                          "blob": "b" * 40},
                                               "power": {"powermode": 2, "adapter": ADAPTER}}},
                "children": children}

    def _diag(self, dataset):
        rust = {}
        for seed in quant_report.RUST_SEEDS:
            arms = {}
            for arm in (F, S):
                points = {}
                for i, ef in enumerate(EFS[dataset]):
                    t = recall_of(F, i) - (0.001 if arm == S else 0.0)
                    points[str(ef)] = {"ef_search": ef, "kernel_calls_mean": 1000.0 + i, "visited_inserts_mean": 1500.0,
                                       "formal_ids_match": True, "rerank_recall10": round(t * TOTAL) / TOTAL,
                                       "formal_recall10": recall_of(arm, i)}
                arms[arm] = {"ids_valid": True, "mismatched_ef": [], "missing_formal_ef": [], "points": points}
            rust[str(seed)] = {"arms": arms, "memory_accounting": {F: {"total_bytes": 10 * MiB + theory(dataset)},
                                                                   S: {"total_bytes": 10 * MiB}}}
        faiss = {str(seed): {"graph_identical": True, "ids_match_at_e_star": True, "codes_agreement": 0.999,
                             "codebook_identical": False,
                             "struct_bytes": {P: {"total_bytes": 20 * MiB + theory(dataset)}, Q: {"total_bytes": 20 * MiB}}}
                 for seed in quant_report.FAISS_SEEDS}
        return {"dataset": dataset, "formal_runs": [{"run": self.label(dataset, seed)} for seed in ORDERS],
                "rust": rust, "faiss": faiss, "ceiling": {"recall10": 0.99, "recall1": 0.99},
                "codes": {"duplicate_row_fraction": 0.0}}

    def label(self, dataset, seed):
        return f"runs/quant-{quant_report.SHORT_NAMES[dataset]}-seed-{seed}"

    # ---- 存取與改動 ----

    def child(self, dataset, impl, seed):
        return next(c for c in self.records[dataset][seed]["children"] if c["implementation"] == impl)

    def point(self, dataset, impl, seed, ef):
        return next(p for p in self.child(dataset, impl, seed)["points"] if p["ef_search"] == ef)

    def diag_point(self, dataset, arm, seed, ef):
        return self.diag[dataset]["rust"][str(seed)]["arms"][arm]["points"][str(ef)]

    def set_recall(self, dataset, impl, seed, ef, value):
        """formal 的 Recall@10，同步到診斷記下的 formal 值（診斷必須與 formal 相符）。"""
        self.point(dataset, impl, seed, ef)["k10"]["recall"] = value
        if impl in (F, S):
            self.diag_point(dataset, impl, seed, ef)["formal_recall10"] = value

    def set_qps(self, dataset, impl, seed, ef, value):
        self.point(dataset, impl, seed, ef)["k10"]["qps"]["qps_median"] = value

    def set_rho(self, dataset, seed, rho, ef=None):
        """ρ_QPS 設成精確的十進位值：F（seed 1 時連同 R）為 1000，S 為 1000 × ρ（兩者都是可精確表示的浮點數）。"""
        ef = E[dataset] if ef is None else ef
        self.set_qps(dataset, F, seed, ef, 1000.0)
        if seed == 1:
            self.set_qps(dataset, R, seed, ef, 1000.0)
        self.set_qps(dataset, S, seed, ef, round(1000.0 * rho, 9))

    def flag(self, dataset, impl, seed, ef, flag="load"):
        self.point(dataset, impl, seed, ef)["flags"].append(flag)

    def groups(self):
        return {d: [(self.label(d, s), r) for s, r in sorted(records.items())] for d, records in self.records.items()}

    def report(self):
        return build_quant_report(self.groups(), self.c_ids, self.core, self.diag, self.profiles, self.premise)

    def rows(self):
        return {row["row"]: row for row in self.report()["judgement"]["rows"]}

    def status(self, label):
        return self.rows()[label]["status"]


@pytest.fixture
def sc():
    return Scenario()


def assert_allowed(report):
    for row in report["judgement"]["rows"]:
        allowed = SECTION_6[section_6_key(row["row"])]
        if report["judgement"]["context"]["c7a_inconsistent"] and row["kind"] in quant_report.HYPOTHESIS_KINDS:
            assert row["status"] == STOPPED, row
        else:
            assert row["status"] in allowed, row


# ---- 基準 ----


def test_the_baseline_holds_everywhere(sc):
    report = sc.report()
    assert_allowed(report)
    rows = {row["row"]: row["status"] for row in report["judgement"]["rows"]}
    for label in ("H1 前提（反組譯）", "H1a 文字", "H1b GloVe", "H1c 維度對比", "H1 整體", "H2 GloVe", "H2 文字", "H2 整體"):
        assert rows[label] == "成立", label
    assert rows["H1d profiler（seed 1）"] == "一致"
    for short in ("glove", "text"):
        for key in ("C1 記憶體 {} F − S", "C1 記憶體 {} P − Q", "C2 FP32 重排 {}", "C3 天花板 {}", "C4 路徑失真 {}",
                    "C5 Gate 重現 {}", "C7a 逐位元 {}", "C7b 逐位元 {}", "C8 FAISS {}"):
            assert rows[key.format(short)] == "一致", key.format(short)
        assert rows[f"C6 漂移 {short}"] == "只描述"
    # 判定表的列數：H1 6 列、H2 3 列、quantization frontier 2 × 2 × 2、C1–C8 每個資料集 10 列（C1 兩部分、C7 兩列）。
    assert len(rows) == 6 + 3 + 8 + 20


def test_every_row_carries_its_numbers(sc):
    rows = sc.rows()
    assert rows["H1a 文字"]["numbers"]["clean_pairs"] == 5
    assert rows["H1b GloVe"]["numbers"]["rho_eval"]["min"] == pytest.approx(1.18)
    assert rows["H2 GloVe"]["numbers"]["median_L_total"] == pytest.approx(0.01)
    assert rows["C1 記憶體 glove F − S"]["numbers"]["theory_bytes"] == 29_999_200
    assert rows["C1 記憶體 text F − S"]["numbers"]["theory_bytes"] == 230_393_856


# ---- H1（019 §2） ----


@pytest.mark.parametrize("rhos, expected", [
    ((0.90, 0.95, 1.0, 1.05, 1.10), "成立"),  # 邊界含在成立區內
    ((0.89, 0.95, 1.0, 1.05, 1.10), "不確定"),
    ((1.11, 1.2, 1.3, 1.4, 1.5), "不成立"),
    ((0.5, 0.6, 0.7, 0.8, 0.89), "不成立"),
    ((1.10, 1.2, 1.3, 1.4, 1.5), "不確定"),  # min = 1.10 不是 > 1.10
])
def test_h1a_text_band(sc, rhos, expected):
    for seed, rho in zip(range(1, 6), rhos):
        sc.set_rho(TEXT, seed, rho)
    assert sc.status("H1a 文字") == expected


@pytest.mark.parametrize("rhos, expected", [
    ((1.05, 1.1, 1.2, 1.3, 1.4), "成立"),
    ((1.04, 1.1, 1.2, 1.3, 1.4), "不確定"),
    ((0.9, 1.0, 1.0, 1.04, 1.049), "不成立"),
    ((0.9, 1.0, 1.0, 1.04, 1.05), "不確定"),
])
def test_h1b_glove_minimum_gain(sc, rhos, expected):
    for seed, rho in zip(range(1, 6), rhos):
        sc.set_rho(GLOVE, seed, rho)
    assert sc.status("H1b GloVe") == expected


def test_h1c_compares_the_two_ranges(sc):
    for seed in range(1, 6):
        sc.set_rho(GLOVE, seed, 1.02)
        sc.set_rho(TEXT, seed, 1.02)
    sc.set_rho(GLOVE, 1, 1.03)
    assert sc.status("H1c 維度對比") == "不確定"  # GloVe 的 min 等於文字的 max：沒有分開
    for seed in range(1, 6):
        sc.set_rho(TEXT, seed, 1.03)
    assert sc.status("H1c 維度對比") == "不確定"  # GloVe 的 max 等於文字的 min：仍沒有分開
    for seed in range(1, 6):
        sc.set_rho(TEXT, seed, 1.08)
    assert sc.status("H1c 維度對比") == "不成立"  # GloVe 的 max 1.03 < 文字的 min 1.08
    assert sc.status("H1 整體") == "不成立"


def test_rho_eval_uses_the_distance_evaluation_counts(sc):
    # ρ_QPS 本身在 [0.90, 1.10] 內，但 S 多評估 20% 的點：ρ_eval = 1.0 × 1.2 = 1.2 > 1.10。
    for seed in range(1, 6):
        sc.set_rho(TEXT, seed, 1.0)
        point = sc.diag_point(TEXT, S, seed, E[TEXT])
        point["kernel_calls_mean"] = sc.diag_point(TEXT, F, seed, E[TEXT])["kernel_calls_mean"] * 1.2
    assert sc.status("H1a 文字") == "不成立"
    pairs = sc.report()["datasets"][TEXT]["h1"]["pairs"]
    assert all(p["rho_eval"] == pytest.approx(1.2) and p["rho_qps"] == pytest.approx(1.0) for p in pairs)


def test_h1_needs_three_clean_pairs_with_valid_d(sc):
    sc.flag(TEXT, S, 1, E[TEXT])
    sc.flag(TEXT, F, 2, E[TEXT], "background")
    assert sc.status("H1a 文字") == "成立"  # 還有 3 個乾淨配對
    sc.child(TEXT, S, 3)["flags"].append("thermal")  # 子 process 的標記也讓配對不乾淨
    assert sc.status("H1a 文字") == "無法判定" and sc.status("H1c 維度對比") == "無法判定"
    assert sc.status("H1 整體") == "無法判定"
    sc = Scenario()
    for seed in (1, 2, 3):
        sc.diag[GLOVE]["rust"][str(seed)]["arms"][S]["ids_valid"] = False  # D 無效的 seed 不計入
    assert sc.status("H1b GloVe") == "無法判定"
    assert [p["usable"] for p in sc.report()["datasets"][GLOVE]["h1"]["pairs"]] == [False, False, False, True, True]


def test_a_flag_outside_e_star_does_not_affect_h1(sc):
    sc.flag(GLOVE, S, 1, 512)
    sc.child(GLOVE, S, 1)["build"]["flags"] = []
    assert sc.status("H1b GloVe") == "成立"


def test_h1d_kappa_is_the_kernel_time_per_evaluation_ratio(sc):
    # seed 1 的 ρ_QPS：GloVe 1.18；D 相同。κ = t_k(S) / t_k(F) = (f_k(S) / ρ_QPS) / f_k(F)。
    report = sc.report()
    assert report["datasets"][GLOVE]["h1d"]["kappa"] == pytest.approx((0.4 / 1.18) / 0.5)
    # D 不同時以每次評估計：S 多評估 25% 的點，κ 再除以 1.25。
    sc.diag_point(GLOVE, S, 1, E[GLOVE])["kernel_calls_mean"] *= 1.25
    assert sc.report()["datasets"][GLOVE]["h1d"]["kappa"] == pytest.approx((0.4 / 1.18) / 0.5 / 1.25)


RHO_SEED_1 = {GLOVE: 1.25, TEXT: 1.0}  # 精確的 ρ_QPS；兩個資料集的 H1a–H1c 仍成立


def set_kappa(sc, dataset, kappa):
    """seed 1、e*：f_k(F) = 0.4、f_k(S) = κ × ρ × 0.4（樣本數 100,000，使 κ 恰為登記的十進位值）。"""
    sc.set_rho(dataset, 1, RHO_SEED_1[dataset])
    sc.profiles[dataset]["samples"][(F, E[dataset])] = counts(kernel=40_000, other=60_000)
    s_count = round(kappa * RHO_SEED_1[dataset] * 40_000)
    assert s_count == kappa * RHO_SEED_1[dataset] * 40_000
    sc.profiles[dataset]["samples"][(S, E[dataset])] = counts(kernel=s_count, other=100_000 - s_count)


def test_h1d_boundaries(sc):
    set_kappa(sc, GLOVE, 0.95)
    set_kappa(sc, TEXT, 0.90)
    assert sc.report()["datasets"][GLOVE]["h1d"]["kappa"] == 0.95
    assert sc.status("H1d profiler（seed 1）") == "一致"
    set_kappa(sc, TEXT, 1.10)
    assert sc.status("H1d profiler（seed 1）") == "一致"
    set_kappa(sc, TEXT, 1.1025)
    assert sc.status("H1d profiler（seed 1）") == "不一致"
    set_kappa(sc, TEXT, 0.8975)
    assert sc.status("H1d profiler（seed 1）") == "不一致"
    set_kappa(sc, TEXT, 1.0)
    set_kappa(sc, GLOVE, 0.9525)
    assert sc.status("H1d profiler（seed 1）") == "不一致"
    assert sc.status("H1 整體") == "計時成立、profile 不一致"


@pytest.mark.parametrize("change", ["flag", "child_flag", "d_invalid", "check_failed", "sample_missing"])
def test_h1d_is_undetermined_when_seed_1_is_not_usable(sc, change):
    if change == "flag":
        sc.flag(TEXT, F, 1, E[TEXT], "multithread")
    elif change == "child_flag":
        sc.child(GLOVE, S, 1)["flags"].append("swap")
    elif change == "d_invalid":
        sc.diag_point(GLOVE, F, 1, E[GLOVE])["formal_ids_match"] = False
    elif change == "check_failed":
        sc.profiles[TEXT]["samples"][(S, E[TEXT])]["ok"] = False
    else:
        del sc.profiles[GLOVE]["samples"][(F, E[GLOVE])]
    report = sc.report()
    rows = {r["row"]: r for r in report["judgement"]["rows"]}
    assert rows["H1d profiler（seed 1）"]["status"] == "無法判定"
    assert rows["H1 整體"]["status"] == "無法判定"
    reasons = " ".join(r for d in (GLOVE, TEXT) for r in report["datasets"][d]["h1d"]["reasons"])
    assert {"flag": "hygiene 標記", "child_flag": "hygiene 標記", "d_invalid": "D 無效", "check_failed": "check",
            "sample_missing": "check"}[change] in reasons
    # 標記點的 κ 照算並列出原因（019 §8）。
    if change == "flag":
        assert report["datasets"][TEXT]["h1d"]["kappa"] is not None and report["datasets"][TEXT]["h1d"]["reasons"]


def test_nu_is_reported_with_its_registered_reading(sc):
    report = sc.report()
    glove = report["datasets"][GLOVE]["h1d"]
    # GloVe：(1 − 0.4) / 1.18 ÷ (1 − 0.5) = 1.017 → 在 [0.90, 1.10] 內
    assert glove["nu"] == pytest.approx((0.6 / 1.18) / 0.5)
    assert "在 [0.90, 1.10] 內" in glove["nu_reading"]
    sc.profiles[GLOVE]["samples"][(S, E[GLOVE])] = counts(kernel=700, other=300)
    assert "cache spillover" in sc.report()["datasets"][GLOVE]["h1d"]["nu_reading"]


def test_a_failed_premise_overrides_everything(sc):
    sc.premise["kernels"]["rhnsw_kernel_sq8"]["ok"] = False
    for seed in range(1, 6):
        sc.set_rho(GLOVE, seed, 0.5)
    rows = sc.rows()
    assert rows["H1 前提（反組譯）"]["status"] == "前提不成立"
    assert rows["H1b GloVe"]["status"] == "不成立"
    assert rows["H1 整體"]["status"] == "前提不成立"


@pytest.mark.parametrize("premise, subs, h1d, expected", [
    ("成立", ("成立", "成立", "成立"), "一致", "成立"),
    ("成立", ("成立", "成立", "成立"), "不一致", "計時成立、profile 不一致"),
    ("成立", ("成立", "成立", "成立"), "無法判定", "無法判定"),
    ("成立", ("不成立", "無法判定", "不確定"), "一致", "不成立"),
    ("成立", ("成立", "無法判定", "不確定"), "一致", "無法判定"),
    ("成立", ("成立", "不確定", "成立"), "一致", "不確定"),
    ("成立", ("成立", "不確定", "成立"), "無法判定", "不確定"),
    ("前提不成立", ("成立", "成立", "成立"), "一致", "前提不成立"),
    ("前提不成立", ("不成立", "成立", "成立"), "一致", "前提不成立"),
])
def test_h1_overall_takes_the_first_matching_row(premise, subs, h1d, expected):
    assert h1_overall(premise, dict(zip("abc", subs)), h1d) == expected


# ---- H2（019 §2） ----


def set_loss(sc, dataset, seed, total, trav, delta_f=0.0):
    """在 e* 設定 L_total、L_trav、δ_F（R_F 不變）。"""
    ef = E[dataset]
    i = EFS[dataset].index(ef)
    r_f = recall_of(F, i)
    sc.set_recall(dataset, S, seed, ef, round((r_f - total) * TOTAL) / TOTAL)
    t_f = round((r_f + delta_f) * TOTAL) / TOTAL
    sc.diag_point(dataset, F, seed, ef)["rerank_recall10"] = t_f
    sc.diag_point(dataset, S, seed, ef)["rerank_recall10"] = round((t_f - trav) * TOTAL) / TOTAL


def test_h2_holds_when_traversal_loss_is_within_the_threshold_including_equality(sc):
    for seed in range(1, 6):
        set_loss(sc, GLOVE, seed, 0.02, 0.006)  # thr = 0.25 × 0.02 + 0.001 = 0.006
    assert sc.status("H2 GloVe") == "成立"
    set_loss(sc, GLOVE, 2, 0.02, 0.0061)
    assert sc.status("H2 GloVe") == "不確定"
    for seed in range(1, 6):
        set_loss(sc, GLOVE, seed, 0.02, 0.0061)
    assert sc.status("H2 GloVe") == "不成立"
    assert sc.status("H2 整體") == "不成立"


def test_the_threshold_uses_each_seeds_own_total_loss(sc):
    set_loss(sc, TEXT, 1, 0.04, 0.0105)  # thr(1) = 0.011
    for seed in range(2, 6):
        set_loss(sc, TEXT, seed, 0.008, 0.0029)  # thr = 0.003
    assert sc.status("H2 文字") == "成立"
    set_loss(sc, TEXT, 2, 0.008, 0.0031)
    assert sc.status("H2 文字") == "不確定"


@pytest.mark.parametrize("median, expected", [(0.0039, "不適用"), (0.004, "成立"), (-0.003, "不適用"), (-0.0031, "無法判定")])
def test_h2_negligible_and_better_than_fp32(sc, median, expected):
    for seed in range(1, 6):
        set_loss(sc, GLOVE, seed, median, 0.0)
    status = sc.rows()["H2 GloVe"]
    assert status["status"] == expected
    if expected == "無法判定":
        assert "比較有問題" in status["note"]


def test_h2_arithmetic_is_exact_at_the_boundary(sc):
    # 0.9007 − 0.8967 以浮點計算是 0.0039999999999998925；以命中數計算恰好是 0.004，不是「不適用」。
    i = EFS[GLOVE].index(E[GLOVE])
    for seed in range(1, 6):
        sc.set_recall(GLOVE, F, seed, E[GLOVE], 0.9007)
        sc.set_recall(GLOVE, S, seed, E[GLOVE], 0.8967)
        sc.diag_point(GLOVE, F, seed, E[GLOVE])["rerank_recall10"] = 0.9007
        sc.diag_point(GLOVE, S, seed, E[GLOVE])["rerank_recall10"] = 0.9007
    assert 0.9007 - 0.8967 < 0.004 and i >= 0
    assert sc.status("H2 GloVe") == "成立"


def test_h2_needs_three_valid_seeds(sc):
    sc.diag[TEXT]["rust"]["1"]["arms"][F]["ids_valid"] = False
    sc.diag[TEXT]["rust"]["2"]["arms"][S]["ids_valid"] = False
    assert sc.status("H2 文字") == "成立"
    sc.point(TEXT, S, 3, E[TEXT])["k10"]["incomplete"] = True  # 結果不足 k 的點不算「formal recall 存在」
    row = sc.rows()["H2 文字"]
    assert row["status"] == "無法判定" and row["note"] == "有效 seed 不足"
    assert row["numbers"]["valid_seeds"] == [4, 5]
    assert sc.status("H2 整體") == "無法判定"


def test_h2_is_undetermined_when_c2_is_inconsistent(sc):
    i = EFS[TEXT].index(96)
    sc.diag_point(TEXT, F, 4, 96)["rerank_recall10"] = round((recall_of(F, i) + 0.0021) * TOTAL) / TOTAL
    rows = sc.rows()
    assert rows["C2 FP32 重排 text"]["status"] == "不一致"
    assert rows["H2 文字"]["status"] == "無法判定" and "C2" in rows["H2 文字"]["note"]


@pytest.mark.parametrize("statuses, expected", [
    (("不適用", "不適用"), "不適用"), (("不適用", "成立"), "成立"), (("成立", "成立"), "成立"),
    (("不成立", "無法判定"), "不成立"), (("無法判定", "成立"), "無法判定"), (("不確定", "成立"), "不確定"),
    (("不確定", "無法判定"), "無法判定"), (("不適用", "不確定"), "不確定"),
])
def test_h2_overall_takes_the_first_matching_row(statuses, expected):
    assert h2_overall(dict(zip((GLOVE, TEXT), statuses))) == expected


# ---- 一致性檢查（019 §2） ----


@pytest.mark.parametrize("dataset, deviation, expected", [
    (GLOVE, 2 * MiB, "一致"), (GLOVE, 2 * MiB + 1, "不一致"), (GLOVE, -2 * MiB, "一致"),
    (TEXT, 11_519_692, "一致"), (TEXT, 11_519_693, "不一致"),  # 5% × 230,393,856 = 11,519,692.8
])
def test_c1_tolerance(sc, dataset, deviation, expected):
    child = sc.child(dataset, S, 4)
    child["build"]["index_footprint_relieved_bytes"] -= deviation
    assert sc.status(f"C1 記憶體 {quant_report.SHORT_NAMES[dataset]} F − S") == expected


def test_c1_excludes_memory_and_swap_flagged_builds(sc):
    sc.child(GLOVE, S, 4)["build"]["index_footprint_relieved_bytes"] -= 10 * MiB
    sc.child(GLOVE, S, 4)["build"]["flags"].append("memory")
    assert sc.status("C1 記憶體 glove F − S") == "一致"
    sc.child(GLOVE, P, 2)["flags"].append("swap")
    sc.child(GLOVE, Q, 2)["build"]["index_footprint_relieved_bytes"] += 10 * MiB
    assert sc.status("C1 記憶體 glove P − Q") == "一致"
    for seed in (1, 3):
        sc.child(GLOVE, Q, seed)["build"]["flags"].append("memory")
    assert sc.status("C1 記憶體 glove P − Q") == "無法判定"
    sc.child(GLOVE, F, 1)["build"]["flags"].append("load")  # load 不是記憶體類的排除條件
    sc.child(GLOVE, F, 1)["build"]["index_footprint_relieved_bytes"] += 3 * MiB
    assert sc.status("C1 記憶體 glove F − S") == "不一致"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_c1_faiss_checks_every_faiss_seed(sc, seed):
    sc.child(GLOVE, Q, seed)["build"]["index_footprint_relieved_bytes"] += 2 * MiB + 1
    assert sc.status("C1 記憶體 glove P − Q") == "不一致"
    assert sc.status("C1 記憶體 glove F − S") == "一致"


def test_c1_lists_m014_and_allocator_retention(sc):
    entry = sc.report()["datasets"][GLOVE]["checks"]["C1"]["rust"]["per_seed"][0]
    assert entry["retention"] == {F: 10 * MiB, S: 10 * MiB} and entry["delta_m014"] == theory(GLOVE)


def test_c2_uses_valid_seeds_and_exact_arithmetic(sc):
    # T_F − R_F = 0.902 − 0.900：浮點 0.0020000000000000018，命中數恰好 0.002 → 一致。
    sc.set_recall(TEXT, F, 2, 512, 0.9)
    sc.diag_point(TEXT, F, 2, 512)["rerank_recall10"] = 0.902
    assert 0.902 - 0.9 > 0.002
    assert sc.status("C2 FP32 重排 text") == "一致"
    sc.diag_point(TEXT, F, 2, 512)["rerank_recall10"] = 0.9021
    assert sc.status("C2 FP32 重排 text") == "不一致"
    sc.diag[TEXT]["rust"]["2"]["arms"][S]["ids_valid"] = False  # seed 2 不是有效 seed：不納入 C2
    assert sc.status("C2 FP32 重排 text") == "一致"


def test_c3_ceiling(sc):
    # 天花板的 Recall@1 刻意不同於 Recall@10：C3 比的是 Recall@10（019 §2）。
    sc.diag[GLOVE]["ceiling"] = {"recall10": 0.99, "recall1": 0.995}
    sc.set_recall(GLOVE, S, 5, 768, 0.992)
    assert sc.status("C3 天花板 glove") == "一致"
    sc.set_recall(GLOVE, S, 5, 768, 0.9921)
    assert sc.status("C3 天花板 glove") == "不一致"
    sc.diag[GLOVE]["ceiling"] = {"recall10": 0.9901, "recall1": 0.985}
    assert sc.status("C3 天花板 glove") == "一致"  # 0.9921 ≤ 0.9901 + 0.002
    sc.diag[GLOVE]["ceiling"] = {}
    assert sc.status("C3 天花板 glove") == "無法判定"


@pytest.mark.parametrize("ratio, expected", [(1.10, "一致"), (0.90, "一致"), (1.11, "path-distorted"), (0.899, "path-distorted")])
def test_c4_path_distortion(sc, ratio, expected):
    r = sc.point(TEXT, R, 1, E[TEXT])["k10"]["qps"]["qps_median"]
    sc.set_qps(TEXT, F, 1, E[TEXT], r * ratio)
    assert sc.status("C4 路徑失真 text") == expected


def test_c4_is_undetermined_on_flagged_or_missing_points(sc):
    sc.flag(TEXT, R, 1, E[TEXT])
    row = sc.rows()["C4 路徑失真 text"]
    assert row["status"] == "無法判定" and row["numbers"]["qps_f_over_r"] is not None
    sc = Scenario()
    sc.point(GLOVE, F, 1, E[GLOVE])["k10"] = {"status": "incomplete", "incomplete": True, "incomplete_queries": None,
                                                "recall": None, "tie_aware_recall": None, "qps": None, "latency": None}
    sc.diag_point(GLOVE, F, 1, E[GLOVE])["formal_recall10"] = None
    assert sc.status("C4 路徑失真 glove") == "無法判定"


def test_c5_compares_three_categories_within_three_points(sc):
    sc.profiles[TEXT]["gate"] = counts(traversal=890, visited_set=90, heap=10, other=10)
    assert sc.status("C5 Gate 重現 text") == "一致"  # traversal 0.92 − 0.89 = 0.03
    sc.profiles[TEXT]["gate"] = counts(traversal=889, visited_set=91, heap=10, other=10)
    assert sc.status("C5 Gate 重現 text") == "不一致"
    sc.profiles[TEXT]["gate"] = None
    assert sc.status("C5 Gate 重現 text") == "無法判定"
    sc = Scenario()
    sc.profiles[GLOVE]["samples"][(R, E[GLOVE])]["ok"] = False
    assert sc.status("C5 Gate 重現 glove") == "無法判定"
    # 三類都比：只有 heap 相差超過 3 個百分點（7.0% 對 3.9%）也是不一致。
    sc = Scenario()
    sc.profiles[GLOVE]["gate"] = counts(traversal=480, visited_set=440, heap=39, other=41)
    row = sc.rows()["C5 Gate 重現 glove"]
    assert row["status"] == "不一致" and row["numbers"]["differences"]["heap"] == pytest.approx(0.031)
    sc.profiles[GLOVE]["gate"] = counts(traversal=480, visited_set=440, heap=40, other=40)
    assert sc.status("C5 Gate 重現 glove") == "一致"  # 相差恰好 3 個百分點


def test_c6_only_describes_the_drift(sc):
    core = qps_of(GLOVE, F, 1, 0)
    sc.set_qps(GLOVE, R, 1, 64, 2 * core)
    row = sc.rows()["C6 漂移 glove"]
    assert row["status"] == "只描述" and len(row["numbers"]["ratios"]) == 8  # 共同的 ef：Core 的 8 個
    ratios = {r["ef_search"]: r["ratio"] for r in row["numbers"]["ratios"]}
    assert ratios[64] == 2.0 and ratios[96] == 1.0  # C 中 R 的 QPS ÷ Core 的 QPS


def test_c7a_inconsistency_stops_h1_and_h2(sc):
    sc.c_ids[GLOVE][(F, 3, 128)] = sc.c_ids[GLOVE][(F, 3, 128)] + 1
    report = sc.report()
    assert_allowed(report)
    rows = {r["row"]: r for r in report["judgement"]["rows"]}
    assert rows["C7a 逐位元 glove"]["status"] == "不一致"
    for label in ("H1a 文字", "H1b GloVe", "H1c 維度對比", "H1d profiler（seed 1）", "H1 整體", "H2 GloVe", "H2 文字", "H2 整體"):
        assert rows[label]["status"] == STOPPED
        assert rows[label]["numbers"]["status_before_stop"]
    assert rows["H1 前提（反組譯）"]["status"] == "成立"  # 靜態檢查照列
    assert rows["C7a 逐位元 glove"]["numbers"]["not_identical"] == [
        {"against": "core", "implementation": F, "seed": 3, "ef_search": 128, "result": "different"}]


def test_a_c7a_inconsistency_in_either_dataset_stops_both_hypotheses(sc):
    sc.c_ids[TEXT][(R, 1, 48)] = sc.c_ids[TEXT][(R, 1, 48)] + 1
    rows = sc.rows()
    assert rows["C7a 逐位元 text"]["status"] == "不一致" and rows["C7a 逐位元 glove"]["status"] == "一致"
    assert rows["H1b GloVe"]["status"] == rows["H2 GloVe"]["status"] == STOPPED


def test_c7a_compares_f_and_r_inside_c_on_every_ef(sc):
    # 512、768 不在 Core 的清單：只能在 C 內比對 F 與 R。
    sc.c_ids[GLOVE][(R, 1, 768)] = sc.c_ids[GLOVE][(R, 1, 768)] + 1
    assert sc.status("C7a 逐位元 glove") == "不一致"


def test_missing_core_points_are_undetermined_not_inconsistent(sc):
    sc.core[TEXT]["ids"][(R, 4, 96)] = None
    del sc.core[GLOVE]["efs"][(R, 5)]
    rows = sc.rows()
    assert rows["C7a 逐位元 text"]["status"] == "無法判定" and rows["C7a 逐位元 glove"]["status"] == "無法判定"
    assert rows["H1 整體"]["status"] == "成立"  # 無法判定不停止流程
    # C 內 F 與 R 的比對照做：不一致時仍是不一致。
    sc.c_ids[TEXT][(F, 1, 24)] = sc.c_ids[TEXT][(F, 1, 24)] + 1
    assert sc.status("C7a 逐位元 text") == "不一致"


def test_c7b_inconsistency_closes_only_the_faiss_readings(sc):
    sc.c_ids[TEXT][(P, 2, 64)] = sc.c_ids[TEXT][(P, 2, 64)] + 1
    report = sc.report()
    assert_allowed(report)
    rows = {r["row"]: r["status"] for r in report["judgement"]["rows"]}
    assert rows["C7b 逐位元 text"] == "不一致"
    assert rows["quantization frontier text Q 對 P 0.90"] == rows["quantization frontier text Q 對 P 0.95"] == "不適用"
    assert rows["quantization frontier glove Q 對 P 0.90"] != "不適用"
    text = report["datasets"][TEXT]
    assert text["paired"]["faiss"]["status"] == "不適用"
    assert set(text["frontier_ratios"]["faiss"].values()) == {"不適用"}
    assert rows["H1 整體"] == rows["H2 整體"] == "成立"


def test_c7b_undetermined_closes_paired_readings_but_not_the_frontier_judgement(sc):
    sc.core[GLOVE]["ids"][(P, 3, 96)] = None
    report = sc.report()
    rows = {r["row"]: r["status"] for r in report["judgement"]["rows"]}
    assert rows["C7b 逐位元 glove"] == "無法判定"
    assert rows["quantization frontier glove Q 對 P 0.90"] in quant_report.QF_VERDICTS
    assert report["datasets"][GLOVE]["paired"]["faiss"]["status"] == "不適用"


def test_c8_graph_and_binding(sc):
    sc.diag[GLOVE]["faiss"]["2"]["graph_identical"] = False
    report = sc.report()
    assert {r["row"]: r["status"] for r in report["judgement"]["rows"]}["C8 FAISS glove"] == "不一致"
    faiss = report["datasets"][GLOVE]["paired"]["faiss"]
    assert faiss["closed_seeds"] == [2] and {e["seed"] for e in faiss["per_point"]} == {1, 3}
    sc = Scenario()
    sc.diag[TEXT]["faiss"]["3"]["ids_match_at_e_star"] = False
    sc.diag[TEXT]["faiss"]["3"]["graph_identical"] = False  # 重建的 Q 沒有綁定到 formal：graph 的比對不算數
    row = sc.rows()["C8 FAISS text"]
    assert row["status"] == "無法判定"
    assert [(e["seed"], e["status"], e["reason"]) for e in row["numbers"]["per_seed"]] == [
        (1, "一致", None), (2, "一致", None), (3, "無法判定", "ids_mismatch")]


@pytest.mark.parametrize("change, reason", [("formal_ids_missing", "formal_ids_missing"), ("diag_missing", "diag_missing")])
def test_c8_is_undetermined_when_the_rebuilt_q_cannot_be_bound(sc, change, reason):
    # 019 §6 只寫了 ids 不符；formal 的 Q 在 e* 沒有 ids、或診斷沒有該 seed 時同樣無法綁定，以 reason 標明。
    if change == "formal_ids_missing":
        sc.diag[GLOVE]["faiss"]["2"]["ids_match_at_e_star"] = None
    else:
        del sc.diag[GLOVE]["faiss"]["2"]
    row = sc.rows()["C8 FAISS glove"]
    assert row["status"] == "無法判定"
    assert {e["seed"]: e["reason"] for e in row["numbers"]["per_seed"]} == {1: None, 2: reason, 3: None}


# ---- quantization frontier 的判讀（015 §1 套用到 S 對 F、Q 對 P） ----


def qf_aggregates(curves, flags=None):
    children = [make_child(impl, [make_point(ef, r, q, flags=(flags or {}).get((impl, seed, ef), ())) for ef, r, q in curve],
                           n=100_000, m=16, efc=200, seed=seed) for (impl, seed), curve in curves.items()]
    rows = quant_report.point_rows([make_record(children)])
    return quant_report.aggregate(rows), quant_report.aggregate([r for r in rows if not r["flags"]])


def three_seeds(impl, curve, scale=(1.0, 1.0, 1.0)):
    return {(impl, seed): [(ef, r, q * s) for ef, r, q in curve] for seed, s in zip((1, 2, 3), scale)}


def test_quantization_frontier_categories():
    fp32 = [(64, 0.93, 2000.0), (96, 0.96, 1500.0), (128, 0.97, 1200.0)]
    faster = [(64, 0.93, 3000.0), (96, 0.96, 2800.0), (128, 0.97, 2600.0)]
    _, clean = qf_aggregates({**three_seeds(F, fp32), **three_seeds(S, faster)})
    assert quant_report._qf_target(clean, F, S, "0.95")["verdict"] == "sq8_faster"
    assert quant_report._qf_target(clean, S, F, "0.95")["verdict"] == "fp32_faster"
    close = [(64, 0.93, 2100.0), (96, 0.96, 1600.0), (128, 0.97, 1300.0)]
    _, clean = qf_aggregates({**three_seeds(F, fp32), **three_seeds(S, close)})
    assert quant_report._qf_target(clean, F, S, "0.95")["verdict"] == "within_quantization"
    _, clean = qf_aggregates({**three_seeds(F, fp32), **three_seeds(S, [(64, 0.9, 3000.0), (128, 0.94, 2000.0)])})
    assert quant_report._qf_target(clean, F, S, "0.95")["verdict"] == "undetermined"


def test_quantization_frontier_boundaries_are_not_separations():
    fp32 = [(64, 0.93, 2000.0), (96, 0.96, 1500.0), (128, 0.97, 1200.0)]
    # S 的 QPS 下界恰好等於 F 的上界（ef 64 的 2000）：範圍沒有分開 → within_quantization。
    touching = [(64, 0.93, 2500.0), (96, 0.96, 2000.0), (128, 0.97, 1900.0)]
    _, clean = qf_aggregates({**three_seeds(F, fp32), **three_seeds(S, touching)})
    assert quant_report._qf_target(clean, F, S, "0.95")["verdict"] == "within_quantization"
    # 範圍分開，但 S 在交點最慢的 seed 恰好等於 F 在交點之前最快的 seed：seed 範圍沒有分開 → uncertain。
    faster = [(64, 0.93, 3000.0), (96, 0.96, 2800.0), (128, 0.97, 2600.0)]
    _, clean = qf_aggregates({**three_seeds(F, fp32, scale=(1.0, 1.0, 1.1)),
                              **three_seeds(S, faster, scale=(1.0, 1.0, 2200.0 / 2800.0))})
    result = quant_report._qf_target(clean, F, S, "0.95")
    assert result["direction"] == "sq8_faster" and result["cell"]["faster"]["value"] == result["cell"]["slower"]["value"]
    assert result["verdict"] == "uncertain"


def test_quantization_frontier_needs_separated_seed_ranges_with_three_unflagged_builds():
    fp32 = [(64, 0.93, 2000.0), (96, 0.96, 1500.0), (128, 0.97, 1200.0)]
    faster = [(64, 0.93, 3000.0), (96, 0.96, 2800.0), (128, 0.97, 2600.0)]
    # 某個 seed 的 S 在交點特別慢：seed 範圍沒有分開 → uncertain。
    _, clean = qf_aggregates({**three_seeds(F, fp32), **three_seeds(S, faster, scale=(1.0, 1.0, 0.7))})
    assert quant_report._qf_target(clean, F, S, "0.95")["verdict"] == "uncertain"
    # 一個 seed 被標記：未標記的 build 只剩 2 個 → uncertain；包含標記點的判讀不同時並列。
    everything, clean = qf_aggregates({**three_seeds(F, fp32), **three_seeds(S, faster)}, flags={(S, 2, 96): ("load",)})
    result = quant_report.quant_frontier(clean, everything, (F, S))["0.95"]
    assert result["main"]["verdict"] == "uncertain" and result["all_builds"]["verdict"] == "sq8_faster" and result["differs"]


# ---- 配對讀值（019 §6 讀值 2） ----


def test_paired_time_readings_use_only_clean_pairs_but_recall_uses_all(sc):
    sc.flag(GLOVE, S, 2, 96)
    sc.set_rho(GLOVE, 2, 5.0, ef=96)
    sc.set_recall(GLOVE, S, 2, 96, 0.5)
    by_ef = {e["ef_search"]: e for e in sc.report()["datasets"][GLOVE]["paired"]["rust"]["by_ef"]}
    entry = by_ef[96]
    assert entry["clean_pairs"] == 4 and entry["pairs"] == 5
    assert entry["rho_qps"]["main"]["max"] < 2.0 and entry["rho_qps"]["all"]["max"] == pytest.approx(5.0)
    assert len(entry["delta_recall10"]["values"]) == 5 and entry["delta_recall10"]["max"] == pytest.approx(recall_of(F, 1) - 0.5)
    assert entry["rho_eval"]["main"] is not None


def test_paired_memory_excludes_memory_and_swap_flags(sc):
    sc.child(TEXT, S, 5)["build"]["flags"].append("memory")
    sc.child(TEXT, S, 5)["build"]["index_footprint_bytes"] = 0
    memory = sc.report()["datasets"][TEXT]["paired"]["rust"]["memory_summary"]
    assert len(memory["delta_m014"]["main"]["values"]) == 4 and len(memory["delta_m014"]["all"]["values"]) == 5
    assert memory["delta_struct"]["main"]["median"] == theory(TEXT)


def test_paired_time_ratios_are_sq8_over_fp32_and_rho_eval_uses_d(sc):
    # 非 e* 的 ef 96：S 多評估 25% 的點。ρ_eval = ρ_QPS × D_S / D_F；ρ_P50、ρ_P95 = SQ8 ÷ FP32（Scenario 中 = 1 / ρ_QPS）。
    for seed in range(1, 6):
        sc.diag_point(GLOVE, S, seed, 96)["kernel_calls_mean"] *= 1.25
    report = sc.report()["datasets"][GLOVE]
    per_point = {(e["seed"], e["ef_search"]): e for e in report["paired"]["rust"]["per_point"]}
    for seed in range(1, 6):
        entry = per_point[(seed, 96)]
        rho = qps_of(GLOVE, S, seed, 1) / qps_of(GLOVE, F, seed, 1)
        assert entry["d_ratio"] == pytest.approx(1.25) and entry["rho_qps"] == pytest.approx(rho)
        assert entry["rho_eval"] == pytest.approx(rho * 1.25)
        assert entry["rho_p50"] == pytest.approx(1 / rho) and entry["rho_p95"] == pytest.approx(1 / rho)
        assert per_point[(seed, 64)]["d_ratio"] == 1.0
    # e* 上的配對讀值與 H1 的計算是兩份獨立的程式，數字必須相同。
    h1 = {e["seed"]: e for e in report["h1"]["pairs"]}
    for seed in range(1, 6):
        assert per_point[(seed, E[GLOVE])]["rho_eval"] == pytest.approx(h1[seed]["rho_eval"], rel=1e-12)
    by_ef = {e["ef_search"]: e for e in report["paired"]["rust"]["by_ef"]}
    assert by_ef[96]["rho_eval"]["main"]["median"] == pytest.approx(1.25 * 1.2)
    assert by_ef[96]["rho_p50"]["main"]["median"] == pytest.approx(1 / 1.2)


def test_paired_memory_deltas_are_fp32_minus_sq8_per_kind(sc):
    # allocator 保留量 F 12 MiB、S 10 MiB；FAISS 的結構差刻意與 Rust 不同（Q 多 1 MiB）。
    for dataset in (GLOVE, TEXT):
        for seed in range(1, 6):
            build = sc.child(dataset, F, seed)["build"]
            build["index_footprint_bytes"] = build["index_footprint_relieved_bytes"] + 12 * MiB
        for seed in quant_report.FAISS_SEEDS:
            sc.diag[dataset]["faiss"][str(seed)]["struct_bytes"][Q]["total_bytes"] = 21 * MiB
    report = sc.report()["datasets"][GLOVE]
    rust, faiss = report["paired"]["rust"]["memory_summary"], report["paired"]["faiss"]["memory_summary"]
    assert rust["delta_m014"]["main"]["median"] == theory(GLOVE) + 2 * MiB
    assert rust["delta_relief"]["main"]["median"] == theory(GLOVE)
    assert rust["delta_struct"]["main"]["median"] == theory(GLOVE)
    assert faiss["delta_m014"]["main"]["median"] == theory(GLOVE)
    assert faiss["delta_relief"]["main"]["median"] == theory(GLOVE)
    assert faiss["delta_struct"]["main"]["median"] == theory(GLOVE) - MiB
    # 正負號：SQ8 比 FP32 大時 Δ 為負。
    sc.child(GLOVE, S, 1)["build"]["index_footprint_bytes"] = 10_000 * MiB
    memory = {e["seed"]: e for e in sc.report()["datasets"][GLOVE]["paired"]["rust"]["memory"]}
    assert memory[1]["delta_m014"] < 0


def test_other_ef_directions_are_marked_against_e_star(sc):
    # GloVe 的 ρ_QPS 在每個 ef 都約 1.2；ef 96 改成 SQ8 較慢（ρ = 0.8）：與 e* 分屬 1 的兩側。
    for seed in range(1, 6):
        sc.set_rho(GLOVE, seed, 0.8, ef=96)
    sc.set_rho(GLOVE, 1, 1.0, ef=128)
    for seed in range(2, 6):
        sc.set_rho(GLOVE, seed, 1.0, ef=128)
    report = sc.report()
    by_ef = {e["ef_search"]: e for e in report["datasets"][GLOVE]["paired"]["rust"]["by_ef"]}
    assert by_ef[96]["rho_qps"]["direction_differs_from_e_star"] is True
    assert by_ef[96]["rho_eval"]["direction_differs_from_e_star"] is True
    assert by_ef[128]["rho_qps"]["direction_differs_from_e_star"] is False  # 恰好等於 1：不屬於任何一側
    assert by_ef[64]["rho_qps"]["direction_differs_from_e_star"] is False
    assert by_ef[E[GLOVE]]["rho_qps"]["direction_differs_from_e_star"] is False
    text = quant_report.markdown(report, GLOVE)
    row = next(line for line in text.splitlines() if line.startswith("| 96 | 5/5 |"))
    assert row.endswith("| **rho_qps、rho_eval** |")
    assert next(line for line in text.splitlines() if line.startswith("| 64 | 5/5 |")).endswith("| — |")
    # e* 沒有乾淨的配對時，方向無法比較（None），不寫成一致。
    for seed in range(1, 6):
        sc.flag(GLOVE, S, seed, E[GLOVE])
    by_ef = {e["ef_search"]: e for e in sc.report()["datasets"][GLOVE]["paired"]["rust"]["by_ef"]}
    assert by_ef[96]["rho_qps"]["direction_differs_from_e_star"] is None


def test_the_flagged_build_is_excluded_from_the_frontier_and_its_judgement(sc):
    # S 在每個 ef 都是 F 的 2 倍：排除標記點時 S 對 F 0.95 是 sq8_faster。seed 2 在 S 的交點（ef 304）被標記而且很慢：
    # 包含它的判讀只能是 uncertain（seed 範圍沒有分開）。主要讀值必須排除它（019 §6 讀值 1）。
    for seed in range(1, 6):
        for i, ef in enumerate(EFS[GLOVE]):
            sc.set_qps(GLOVE, S, seed, ef, 2 * qps_of(GLOVE, F, seed, i))
    sc.set_qps(GLOVE, S, 2, 304, 100.0)
    sc.flag(GLOVE, S, 2, 304)
    report = sc.report()
    row = {r["row"]: r for r in report["judgement"]["rows"]}["quantization frontier glove S 對 F 0.95"]
    assert row["status"] == "sq8_faster"
    assert row["numbers"]["all_builds"] == "uncertain" and row["numbers"]["differs"] is True
    part = report["datasets"][GLOVE]
    assert part["frontier"][S]["100000"]["0.95"]["main"]["qps"]["builds"] == 4
    assert part["frontier"][S]["100000"]["0.95"]["all_points"]["qps"]["builds"] == 5
    assert part["frontier"][S]["100000"]["0.95"]["differs"] is True


def test_faiss_paired_readings_have_no_rho_eval(sc):
    by_ef = sc.report()["datasets"][GLOVE]["paired"]["faiss"]["by_ef"]
    assert "rho_eval" not in by_ef[0] and by_ef[0]["rho_qps"]["main"]["median"] == pytest.approx(1.1)


def test_frontier_ratios_need_both_arms_to_reach_the_target(sc):
    ratios = sc.report()["datasets"][GLOVE]["frontier_ratios"]["rust"]
    assert ratios["0.99"] is None  # S 的最高 recall 是 0.98
    assert ratios["0.90"]["index_footprint_bytes"] < 1


# ---- 判定表的狀態（019 §6） ----


def test_the_validator_rejects_states_outside_the_table(sc):
    report = sc.report()
    rows, context = copy.deepcopy(report["judgement"]["rows"]), report["judgement"]["context"]
    for label, bad in (("H1d profiler（seed 1）", "成立"), ("H1 前提（反組譯）", "無法判定"), ("C4 路徑失真 glove", "不一致"),
                       ("C6 漂移 text", "一致"), ("H2 GloVe", "path-distorted"), ("quantization frontier glove S 對 F 0.90", "不適用"),
                       ("H1 整體", "計時成立"), ("C2 FP32 重排 glove", "不適用")):
        broken = copy.deepcopy(rows)
        next(r for r in broken if r["row"] == label)["status"] = bad
        with pytest.raises(ReportError, match="not allowed"):
            validate_table(broken, context)
    validate_table(rows, context)


def test_the_validator_enforces_the_conditional_states(sc):
    report = sc.report()
    rows, context = report["judgement"]["rows"], report["judgement"]["context"]
    for label, bad, message in (("H1a 文字", STOPPED, "C7a"), ("quantization frontier text Q 對 P 0.95", "不適用", "C7b"),
                                ("C8 FAISS glove", "不適用", "fallback"), ("C1 記憶體 text P − Q", "不適用", "fallback")):
        broken = copy.deepcopy(rows)
        next(r for r in broken if r["row"] == label)["status"] = bad
        with pytest.raises(ReportError, match=message):
            validate_table(broken, context)
    with pytest.raises(ReportError, match="C7a"):
        validate_table(rows, {**context, "c7a_inconsistent": True})


def test_the_validator_ties_undetermined_c7_and_c8_to_their_conditions(sc):
    # 019 §6：C7a / C7b 的「無法判定」只在點缺失時；C8 的「無法判定」只在重建的 Q 沒有與 formal 綁定時。
    report = sc.report()
    rows, context = report["judgement"]["rows"], report["judgement"]["context"]
    broken = copy.deepcopy(rows)
    next(r for r in broken if r["row"] == "C7b 逐位元 glove")["status"] = "無法判定"
    with pytest.raises(ReportError, match="points are missing"):
        validate_table(broken, context)
    broken = copy.deepcopy(rows)
    c8 = next(r for r in broken if r["row"] == "C8 FAISS text")
    c8["status"] = "無法判定"
    c8["numbers"]["per_seed"][0]["status"] = "無法判定"  # 沒有 reason（ids 相符）卻記成無法判定
    with pytest.raises(ReportError, match="undetermined only when"):
        validate_table(broken, context)
    broken = copy.deepcopy(rows)
    next(r for r in broken if r["row"] == "C8 FAISS text")["status"] = "無法判定"  # 各 seed 都一致
    with pytest.raises(ReportError, match="does not follow"):
        validate_table(broken, context)
    # 條件成立時照常通過。
    sc.core[TEXT]["ids"][(P, 2, 64)] = None
    sc.diag[GLOVE]["faiss"]["1"]["ids_match_at_e_star"] = False
    report = sc.report()
    statuses = {r["row"]: r["status"] for r in report["judgement"]["rows"]}
    assert statuses["C7b 逐位元 text"] == statuses["C8 FAISS glove"] == "無法判定"


def test_c7_is_undetermined_when_a_c_point_is_missing(sc):
    # C 自己的點缺失（例如子 process 中途失敗）也無法比對，記為無法判定（019 只寫了 Core 的點缺失，記入補記）。
    del sc.c_ids[TEXT][(F, 4, 96)]
    row = sc.rows()["C7a 逐位元 text"]
    assert row["status"] == "無法判定"
    assert {"against": "core", "implementation": F, "seed": 4, "ef_search": 96, "result": "c_missing"} in row["numbers"]["not_identical"]


@pytest.mark.parametrize("mutate", [
    lambda sc: sc.set_rho(TEXT, 2, 1.5),
    lambda sc: [sc.flag(GLOVE, S, seed, E[GLOVE]) for seed in (1, 2, 3)],
    lambda sc: sc.premise["kernels"]["rhnsw_kernel_fp32"].update(ok=False),
    lambda sc: [set_loss(sc, TEXT, seed, 0.001, 0.0) for seed in range(1, 6)],
    lambda sc: [set_loss(sc, GLOVE, seed, 0.001, 0.0) for seed in range(1, 6)] + [
        set_loss(sc, TEXT, seed, 0.001, 0.0) for seed in range(1, 6)],
    lambda sc: sc.diag[GLOVE].update(ceiling={}),
    lambda sc: sc.c_ids[TEXT].update({(P, 1, 24): np.zeros((2, 10), dtype=np.int64)}),
    lambda sc: sc.c_ids[TEXT].update({(F, 1, 24): np.zeros((2, 10), dtype=np.int64)}),
    lambda sc: sc.diag[TEXT]["faiss"]["1"].update(graph_identical=False),
    lambda sc: sc.profiles[GLOVE].update(gate=None),
])
def test_every_scenario_only_produces_registered_states(sc, mutate):
    mutate(sc)
    report = sc.report()
    assert_allowed(report)
    json.dumps(report, default=quant_report._json_default)


def test_both_datasets_are_needed(sc):
    del sc.records[TEXT]
    with pytest.raises(ReportError, match="both datasets"):
        sc.report()


# ---- 輸入的綁定 ----


def test_the_diag_must_belong_to_these_formal_runs(sc):
    sc.diag[GLOVE]["formal_runs"][0]["run"] = "runs/other"
    with pytest.raises(ReportError, match="other formal runs"):
        sc.report()
    sc = Scenario()
    sc.diag_point(TEXT, S, 2, 64)["formal_recall10"] = 0.1
    with pytest.raises(ReportError, match="formal Recall@10"):
        sc.report()
    sc = Scenario()
    sc.diag[GLOVE] = None
    with pytest.raises(ReportError, match="quant_diag"):
        sc.report()


def test_the_premise_must_be_of_the_measured_binding(sc):
    sc.premise["binding_sha256"] = "other"
    with pytest.raises(ReportError, match="disassembled binding"):
        sc.report()
    sc = Scenario()
    sc.premise = None
    with pytest.raises(ReportError, match="premise"):
        sc.report()


def test_recall_must_be_a_multiple_of_the_query_count(sc):
    sc.set_recall(GLOVE, S, 1, E[GLOVE], 0.95005)
    with pytest.raises(ReportError, match="multiple"):
        sc.report()


def test_expectations_are_listed_without_judging(sc):
    expectations = {e["expectation"]: e for e in sc.report()["judgement"]["expectations"]}
    assert expectations["GloVe 的 S 在 ef ≤ 768 內達到 0.95"]["observed"] == 304
    assert expectations["文字的 S 在 ef ≤ 512 內達到 0.99"]["met"] is False
    assert expectations["c_文字 > c_GloVe"]["met"] is False
    # F 與 P 在 ef 768 達到 0.99：「所有臂都未達到」不符合。
    glove_099 = expectations["GloVe 的 0.99：所有臂都未達到"]
    assert glove_099["met"] is False and glove_099["observed"][F] == 768 and glove_099["observed"][S] is None


def test_the_glove_099_expectation_includes_the_anchor_arm(sc):
    # 019 §2 的「所有臂」含 R：只有 R 達到 0.99 時，附帶預期也不符合。
    for impl in (F, P):
        for seed in sc.records[GLOVE]:
            if impl in ORDERS[seed]:
                sc.set_recall(GLOVE, impl, seed, 768, 0.985)
    expectations = {e["expectation"]: e for e in sc.report()["judgement"]["expectations"]}
    glove_099 = expectations["GloVe 的 0.99：所有臂都未達到"]
    assert glove_099["observed"] == {F: None, S: None, P: None, Q: None, R: 768}
    assert glove_099["met"] is False
    sc.set_recall(GLOVE, R, 1, 768, 0.985)
    expectations = {e["expectation"]: e for e in sc.report()["judgement"]["expectations"]}
    assert expectations["GloVe 的 0.99：所有臂都未達到"]["met"] is True


def test_other_ef_decompositions_are_listed_with_their_seed_validity(sc):
    sc.diag[GLOVE]["rust"]["2"]["arms"][S]["ids_valid"] = False
    others = sc.report()["datasets"][GLOVE]["h2"]["other_ef"]
    assert len(others) == 5 * len(EFS[GLOVE])
    for entry in others:
        # Scenario：R_F − R_S = 0.01、T_F = R_F、T_S = T_F − 0.001。
        assert entry["L_total"] == pytest.approx(0.01) and entry["L_trav"] == pytest.approx(0.001)
        assert entry["L_rank"] == pytest.approx(0.009) and entry["delta_F"] == pytest.approx(0.0, abs=1e-12)
        assert entry["valid"] is (entry["seed"] != 2)


def test_profile_readings_divide_the_shares_by_qps_and_d(sc):
    profiles = {(e["implementation"], e["ef_search"]): e for e in sc.report()["datasets"][TEXT]["profiles"]}
    entry = profiles[(F, E[TEXT])]
    i = EFS[TEXT].index(E[TEXT])
    qps, d = qps_of(TEXT, F, 1, i), 1000.0 + i
    assert entry["qps"] == qps and entry["d"] == d
    assert entry["per_query_us"]["kernel"] == pytest.approx(0.8 * 1e6 / qps)
    assert entry["per_eval_ns"]["kernel"] == pytest.approx(0.8 * 1e9 / (qps * d))
    assert entry["per_query_us"]["other"] == pytest.approx(0.13 * 1e6 / qps)
    assert profiles[(R, E[TEXT])]["d"] is None and profiles[(R, E[TEXT])]["per_eval_ns"] is None


# ---- CLI：檔案輸入、合併檢查與輸出 ----

PROFILE_ROWS = {"kernel": "__RNvNtNtCs5kyc4sahh7U_4hnsw4hnsw5quant17rhnsw_kernel_fp32", "visited_set": "hashbrown::map::insert",
                "heap": "alloc::collections::binary_heap::pop", "traversal": "hnsw::hnsw::HnswIndex::search_layer",
                "allocation": "_platform_memmove", "other": "hnsw::hnsw::quant::QuantIndex::search_batch"}


def sample_text(category_counts):
    lines = ["Sort by top of stack, same collapsed (when >= 5):"]
    for category, count in category_counts.items():
        if count:
            lines.append(f"        {PROFILE_ROWS[category]}  (in rust_hnsw.cpython-313-darwin.so)        {count}")
    return "\n".join(lines) + "\n\n"


def materialize(sc, root):
    """Scenario 寫成檔案：10 個 C run、4 個 Core run、診斷、反組譯與取樣。回傳 write_quant_report 的參數。"""
    runs, core_dirs = [], []
    for dataset, records in sc.records.items():
        for seed, record in records.items():
            directory = root / "runs" / Path(sc.label(dataset, seed)).name
            (directory / "points").mkdir(parents=True)
            for child in record["children"]:
                for point in child["points"]:
                    ids = sc.c_ids[dataset].get((child["implementation"], seed, point["ef_search"]), np.zeros((2, 10), np.int64))
                    np.savez(directory / point["file"], ids_k10=ids)
            (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
            runs.append(directory)
        sc.diag[dataset]["formal_runs"] = [{"run": str(root / "runs" / Path(sc.label(dataset, s)).name)} for s in records]
        (root / f"diag-{dataset}.json").write_text(json.dumps(sc.diag[dataset]), encoding="utf-8")
        for name in sorted(set(quant_report.CORE_RUNS[dataset].values())):
            directory = root / "core" / name
            seeds = [s for s, n in quant_report.CORE_RUNS[dataset].items() if n == name]
            (directory / "points").mkdir(parents=True)
            children = []
            for impl, impl_seeds in ((R, range(1, 6)), (P, range(1, 4))):
                for seed in (s for s in seeds if s in impl_seeds):
                    points = []
                    for i, ef in enumerate(CORE_EFS[dataset]):
                        file = f"points/{impl}-{seed}-ef{ef}.npz"
                        np.savez(directory / file, ids_k10=sc.core[dataset]["ids"][(impl, seed, ef)])
                        points.append({"ef_search": ef, "file": file,
                                       "k10": {"status": "ok", "qps": {"qps_median": qps_of(dataset, F, 1, i)}}})
                    children.append({"implementation": impl, "build": {"n": 100_000, "m": 16, "ef_construction": 200,
                                                                       "seed": seed, "shuffle": None}, "points": points})
            (directory / "run.json").write_text(json.dumps({"dataset": {"name": dataset}, "children": children}))
            core_dirs.append(directory)
    (root / "premise.json").write_text(json.dumps(sc.premise))
    profiles = root / "profiles"
    profiles.mkdir()
    from benchmarks import quant_profile

    for dataset in sc.records:
        records = [sc.records[dataset][s] for s in sorted(sc.records[dataset])]
        samples, _ = quant_profile.select_samples(records)
        for sample in samples:
            stem = profiles / quant_profile.profile_name(dataset, sample["implementation"], sample["ef"])
            args = {**quant_profile.expected_args(dataset, sample), "seconds": 33.0}
            meta = {"args": args, "recall_at_10": sample["recall10"], "lock": {"held_by_ancestor": True},
                    "evidence_conditions": {"power": {"powermode": 2, "adapter": ADAPTER}},
                    "git": {"commit": "a" * 40, "status": []}, "build_info": {"profile": "release", "opt_level": "3"},
                    "versions": {"hnswlib": "0.8.0", "faiss-cpu": "1.15.1", "numpy": "2.5.3"}}
            stem.with_name(stem.name + ".json").write_text(json.dumps(meta))
            known = sc.profiles[dataset]["samples"].get((sample["implementation"], sample["ef"]),
                                                       counts(kernel=500, other=500))
            stem.with_name(stem.name + ".sample.txt").write_text(sample_text(known["counts"]))
            stem.with_name(stem.name + ".top.txt").write_text("ranking")
        gate = profiles / f"gate-{quant_report.SHORT_NAMES[dataset]}-rust-hnsw-search.sample.txt"
        gate.write_text(sample_text(sc.profiles[dataset]["gate"]["counts"]))
    return {"run_dirs": runs, "core_dirs": core_dirs, "out_root": root / "reports",
            "diag_paths": [root / f"diag-{d}.json" for d in sc.records], "premise_path": root / "premise.json",
            "profile_dir": profiles, "roots": [root / "runs"], "changed_files": lambda a, b: [],
            "profile_changed_files": lambda a, b: []}


def test_the_cli_writes_one_report_per_dataset(sc, tmp_path):
    args = materialize(sc, tmp_path)
    written = quant_report.write_quant_report(**args)
    assert [p.name for p in written] == ["quant-glove", "quant-text"]
    for out in written:
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        text = (out / "report.md").read_text(encoding="utf-8")
        assert "完整判定表" in text and "| H1 整體 | 成立 |" in text
        assert {"recall_qps.png", "recall_vs_ef.png", "quant_frontier_0.90.png", "quant_frontier_0.90_relief.png",
                "profile_breakdown.png"} <= set(report["figures"])
        assert all((out / name).stat().st_size > 0 for name in report["figures"])
        assert len(report["datasets"]) == 1 and len(report["judgement"]["rows"]) == 37
    text = json.loads((written[1] / "report.json").read_text(encoding="utf-8"))
    profiles = {(e["implementation"], e["ef_search"]): e for e in text["datasets"][TEXT]["profiles"]}
    assert profiles[(S, 192)]["role"] == "same_recall_0.95" and profiles[(S, 48)]["ok"] is True
    assert profiles[(F, 48)]["shares"]["kernel"] == pytest.approx(0.8)
    assert profiles[(F, 48)]["per_query_us"]["kernel"] == pytest.approx(0.8 * 1e6 / qps_of(TEXT, F, 1, 2))
    # 記憶體表的 M_struct（診斷）與 019 §6 的 tie-aware Recall@10 都列在 markdown。
    markdown = (written[0] / "report.md").read_text(encoding="utf-8")
    assert "| M_struct（診斷） |" in markdown and "tie-aware Recall@10" in markdown
    builds = {(b["implementation"], b["seed"]): b for b in json.loads((written[0] / "report.json").read_text())["datasets"][GLOVE]["builds"]}
    assert builds[(F, 2)]["struct"] == 10 * MiB + theory(GLOVE) and builds[(S, 2)]["struct"] == 10 * MiB
    assert builds[(P, 3)]["struct"] == 20 * MiB + theory(GLOVE) and builds[(Q, 3)]["struct"] == 20 * MiB
    assert builds[(R, 1)]["struct"] is None
    row = next(line for line in markdown.splitlines() if line.startswith(f"| {S} | 2 |"))
    assert row.split(" | ")[4] == "10.0"  # 實作、seed、M_014、M_relief 之後是 M_struct
    recall_rows = {(e["implementation"], e["ef_search"]): e
                   for e in json.loads((written[0] / "report.json").read_text())["datasets"][GLOVE]["recall_by_ef"]}
    assert recall_rows[(S, 304)]["tie10"]["median"] == pytest.approx(0.95) and recall_rows[(R, 304)]["tie10"]["values"] == [0.96]


def test_the_cli_reads_the_profiles_through_the_check(sc, tmp_path):
    args = materialize(sc, tmp_path)
    meta = args["profile_dir"] / "quant-text-rust-hnsw-sq8-ef48.json"
    data = json.loads(meta.read_text())
    data["evidence_conditions"]["power"]["powermode"] = 0
    meta.write_text(json.dumps(data))
    written = quant_report.write_quant_report(**args)
    report = json.loads((written[0] / "report.json").read_text(encoding="utf-8"))
    rows = {r["row"]: r["status"] for r in report["judgement"]["rows"]}
    assert rows["H1d profiler（seed 1）"] == "無法判定"


@pytest.mark.parametrize("change, message", [
    ("protocol", "008-v5"), ("powermode", "high-performance"), ("adapter", "power adapter"), ("extra_run", "cannot be replaced"),
    ("missing_core", "must list every Core run"), ("unknown_core", "not the Core runs registered"),
    ("core_dataset", "is a run of"), ("smoke", "formal runs only"), ("missing_run", "exactly one"),
    ("missing_point", "but missing"),
    # 同一個資料集內的單一 run 不同（只改 GloVe 的 seed 3）：由每個資料集的 _check_mergeable 拒絕（019 §5）。
    ("one_adapter", "differs from .* in the power adapter"), ("one_build_info", "differs from .* in rust_hnsw.BUILD_INFO"),
    ("one_status", "status 'aborted'"), ("one_duplicate", "appears in both"),
])
def test_the_cli_refuses_inputs_that_cannot_be_merged(sc, tmp_path, change, message):
    args = materialize(sc, tmp_path)
    runs = args["run_dirs"]

    def rewrite(path, fn):
        record = json.loads((path / "run.json").read_text())
        fn(record)
        (path / "run.json").write_text(json.dumps(record))

    if change == "protocol":
        for path in runs:
            rewrite(path, lambda r: r["protocol"].update(version="008-v4"))
    elif change == "powermode":
        # 同一個資料集內混用電力模式由 _check_mergeable 拒絕；全部都是自動模式（0）則由 C 的條件拒絕（016、019 §5）。
        for path in runs:
            rewrite(path, lambda r: r["metadata"]["formal_checks"]["power"].update(powermode=0))
    elif change == "adapter":
        for path in runs[5:]:
            rewrite(path, lambda r: r["metadata"]["formal_checks"]["power"].update(
                adapter={"manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"}))
    elif change == "extra_run":
        import shutil

        shutil.copytree(runs[0], tmp_path / "runs" / "again")
    elif change == "missing_core":
        args["core_dirs"] = args["core_dirs"][1:]
    elif change == "unknown_core":
        args["core_dirs"] = [*args["core_dirs"], tmp_path / "core" / "20990101T000000Z-formal"]
    elif change == "core_dataset":
        path = args["core_dirs"][0] / "run.json"
        record = json.loads(path.read_text())
        record["dataset"]["name"] = TEXT
        path.write_text(json.dumps(record))
    elif change == "smoke":
        rewrite(runs[0], lambda r: r.update(mode="smoke"))
    elif change == "missing_run":
        args["run_dirs"] = runs[1:]
    elif change == "missing_point":
        (runs[0] / "points" / f"{F}-ef64.npz").unlink()
    elif change == "one_adapter":
        rewrite(runs[2], lambda r: r["metadata"]["formal_checks"]["power"].update(
            adapter={"manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"}))
    elif change == "one_build_info":
        rewrite(runs[2], lambda r: r["metadata"].update(build_info={"profile": "release", "opt_level": "2"}))
    elif change == "one_status":
        rewrite(runs[2], lambda r: r.update(status="aborted"))
    elif change == "one_duplicate":
        first = json.loads((runs[0] / "run.json").read_text())
        rewrite(runs[2], lambda r: r["children"].append(first["children"][0]))
    with pytest.raises(ReportError, match=message):
        quant_report.write_quant_report(**args)


def test_a_missing_core_run_directory_is_undetermined_not_refused(sc, tmp_path):
    args = materialize(sc, tmp_path)
    import shutil

    shutil.rmtree(args["core_dirs"][0])  # GloVe seed 1 的 Core run 不在了：明確列出路徑，記為缺失
    written = quant_report.write_quant_report(**args)
    rows = {r["row"]: r["status"] for r in json.loads((written[0] / "report.json").read_text())["judgement"]["rows"]}
    assert rows["C7a 逐位元 glove"] == "無法判定" and rows["C7b 逐位元 glove"] == "無法判定"
    assert rows["H1 整體"] == "成立"


def test_the_exact_fraction_helper():
    assert quant_report._exact(0.9567, 10_000) == Fraction(9567, 10_000)
    assert quant_report._exact(None, 10) is None
    with pytest.raises(ReportError):
        quant_report._exact(0.12345, 1000)
