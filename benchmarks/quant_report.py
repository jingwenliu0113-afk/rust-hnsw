"""Advanced C 的報告（decision 019 §6）：frontier、同 (seed, ef) 的配對讀值、H1 / H2 的判定與一致性檢查 C1–C8。

用法（10 份 C formal run、`quant_diag` 與 profiling 的 check 都完成之後，019 §9 第 8 步）：
    .venv/bin/python -m benchmarks.quant_report <10 個 C formal run 目錄…> \\
        --core-runs <019 §2 登記的 4 個 Core formal run 目錄…> [--excluded <已整份重跑的失敗 run>…] \\
        [--diag results/quant/diag-glove.json results/quant/diag-text.json] [--premise results/quant/premise.json] \\
        [--profiles results/profiles] [--out results/reports]

- 輸出：每個資料集一個目錄 `<out>/quant-<glove|text>/`（report.json、report.md 與圖）。
  判定表涵蓋兩個資料集（H1c、H1 / H2 整體），兩份報告列出同一張表。
- 合併前的檢查重用 report.py：每個資料集 `_check_compatible`、`_check_mergeable`、`check_formal_set(registry=QUANT_CONFIGS)`；
  兩個資料集之間另外比對版本、BUILD_INFO、量測程式與電力條件（019 §5：C 的 run 是一整組）。
- 判定規則都在看到任何 C 的結果之前固定（019 §2、§6）；每一列只能是 §6 允許的狀態（`validate_table`）。
- 判定都以精確的分數比較：recall 換成命中數 / (query 數 × k)，QPS、D 等量測值以它們的浮點值本身換成分數，
  門檻是十進位的精確值；避免浮點運算的順序把邊界上的值判到另一側。
- 不呼叫 Gate 的讀值（C 沒有 hnswlib，019 §4）。
"""

import argparse
import json
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from benchmarks import gate_profile
from benchmarks.report import (
    HYGIENE_FLAGS,
    IMPL_STYLE,
    REPO,
    TARGETS,
    VARIANCE_N,
    ReportError,
    _cell_side,
    _check_compatible,
    _check_mergeable,
    _impl_line,
    _plain_log_ticks,
    _reading_cell,
    _style_axes,
    _summary,
    _target_key,
    aggregate,
    check_formal_set,
    excluded_runs,
    frontier,
    point_rows,
    qps_bracket,
)

GLOVE, TEXT = "glove-100-angular", "simplewiki-embeddinggemma2-100k"
SHORT_NAMES = gate_profile.SHORT_NAMES
# 019 附錄：C 的 formal config，每個資料集 5 份；C 的報告必須恰好包含每一份各一個 run（013 §5 的規則，019 §4）。
QUANT_CONFIGS = {
    GLOVE: tuple(f"benchmarks/configs/quant-glove-seed-{seed}.toml" for seed in range(1, 6)),
    TEXT: tuple(f"benchmarks/configs/quant-text-seed-{seed}.toml" for seed in range(1, 6)),
}
# 019 §2 的各臂代號。
R, F, S, P, Q = "rust-hnsw", "rust-hnsw-fp32g", "rust-hnsw-sq8", "faiss-hnsw", "faiss-hnsw-sq8"
ARMS = (F, S, P, Q)
PAIRS = {"rust": (F, S), "faiss": (P, Q)}
PAIR_NAMES = {"rust": "S 對 F", "faiss": "Q 對 P"}
RUST_SEEDS = (1, 2, 3, 4, 5)
FAISS_SEEDS = (1, 2, 3)
ANCHOR_SEED = 1
N, M, EF_CONSTRUCTION = 100_000, 16, 200
K, K1 = 10, 1
# 019 §2：判讀用的固定 ef（Core (16, 200) 的 rust-hnsw，跨 seed median Recall@10 第一次 ≥ 0.95 的 ef）。
E_STAR = {GLOVE: 304, TEXT: 48}
PROTOCOL = "008-v5"
POWERMODE = 2  # 016：所有 C 的 run 都用高效能電力模式（019 §5）
# 019 §2：C7 對照的 Core 正式 run（依 seed）。
CORE_RUNS = {
    GLOVE: {1: "20261009T080239Z-formal", **{seed: "20261009T091611Z-formal" for seed in (2, 3, 4, 5)}},
    TEXT: {1: "20261015T191333Z-formal", **{seed: "20261015T210125Z-formal" for seed in (2, 3, 4, 5)}},
}
CORE_IMPL = {F: R, R: R, P: P}  # C 的臂 → Core 中被比對的實作
# 019 §3：FAISS 的移植可不可行，只由 synthetic 測試在 smoke 之前定案；定案為可行，沒有採用原生 IndexHNSWSQ 的備案
# （019 §9 第 4 步的補記）。採用備案時 C1 的 FAISS 部分與 C8 記為不適用。
FAISS_FALLBACK = False
DIAG_DIR = REPO / "results" / "quant"
PREMISE_PATH = DIAG_DIR / "premise.json"
PROFILES_DIR = REPO / "results" / "profiles"
REPORTS_DIR = REPO / "results" / "reports"

# ---- 事先登記的門檻（019 §2） ----
MIN_CLEAN_PAIRS = 3
H1A_RANGE = (Fraction("0.90"), Fraction("1.10"))
H1B_MIN = Fraction("1.05")
H1D_TEXT_RANGE = (Fraction("0.90"), Fraction("1.10"))
H1D_GLOVE_MAX = Fraction("0.95")
NU_RANGE = (Fraction("0.90"), Fraction("1.10"))
H2_MIN_SEEDS = 3
H2_SLOPE, H2_OFFSET = Fraction(1, 4), Fraction(1, 1000)
H2_BETTER = Fraction(-3, 1000)
H2_NEGLIGIBLE = Fraction(4, 1000)
C1_RELATIVE, C1_ABSOLUTE = Fraction(5, 100), 2 * 2**20
C2_MAX = Fraction(2, 1000)
C3_SLACK = Fraction(2, 1000)
C4_RANGE = (Fraction("0.90"), Fraction("1.10"))
C5_MAX = Fraction(3, 100)
C5_CATEGORIES = ("traversal", "visited_set", "heap")
QF_TARGETS = ("0.90", "0.95")
MEMORY_EXCLUDING = ("memory", "swap")  # 019 §6：記憶體類排除有這兩種標記的 build

# ---- 狀態（019 §6 的完整判定表） ----
HOLDS, FAILS, UNCERTAIN, UNDETERMINED, NA = "成立", "不成立", "不確定", "無法判定", "不適用"
PREMISE_FAILS = "前提不成立"
CONSISTENT, INCONSISTENT = "一致", "不一致"
TIMING_ONLY = "計時成立、profile 不一致"
PATH_DISTORTED = "path-distorted"
DESCRIBED = "只描述"
NOT_JUDGED = "不判定（C7a 不一致）"
QF_VERDICTS = ("sq8_faster", "fp32_faster", "within_quantization", "uncertain", "undetermined")
ALLOWED = {
    "h1_premise": {HOLDS, PREMISE_FAILS},
    "h1_sub": {HOLDS, FAILS, UNCERTAIN, UNDETERMINED, NOT_JUDGED},
    "h1d": {CONSISTENT, INCONSISTENT, UNDETERMINED, NOT_JUDGED},
    "h1_overall": {HOLDS, FAILS, TIMING_ONLY, PREMISE_FAILS, UNCERTAIN, UNDETERMINED, NOT_JUDGED},
    "h2_dataset": {HOLDS, FAILS, UNCERTAIN, UNDETERMINED, NA, NOT_JUDGED},
    "h2_overall": {HOLDS, FAILS, UNCERTAIN, UNDETERMINED, NA, NOT_JUDGED},
    "qf_rust": set(QF_VERDICTS),
    "qf_faiss": {*QF_VERDICTS, NA},
    "c1_rust": {CONSISTENT, INCONSISTENT, UNDETERMINED},
    "c1_faiss": {CONSISTENT, INCONSISTENT, UNDETERMINED, NA},
    "c_basic": {CONSISTENT, INCONSISTENT, UNDETERMINED},
    "c4": {CONSISTENT, PATH_DISTORTED, UNDETERMINED},
    "c6": {DESCRIBED},
    "c7": {CONSISTENT, INCONSISTENT, UNDETERMINED},
    "c8": {CONSISTENT, INCONSISTENT, UNDETERMINED, NA},
}
HYPOTHESIS_KINDS = ("h1_sub", "h1d", "h1_overall", "h2_dataset", "h2_overall")
C7_MISSING = {"core_missing", "c_missing"}  # C7a / C7b 的「無法判定」只能來自點的缺失
# C8 的「無法判定」：§7 重建的 Q 沒有與 formal 綁定（ids 不符；formal 的 Q 在 e* 沒有 ids；診斷沒有這個 seed）。
C8_UNBOUND = ("ids_mismatch", "formal_ids_missing", "diag_missing")


def diag_path(dataset: str) -> Path:
    """019 §7 的輸出：`results/quant/diag-<glove|text>.json`。"""
    return DIAG_DIR / f"diag-{SHORT_NAMES[dataset]}.json"


def decompose(r_f, r_s, t_f, t_s) -> dict:
    """019 §2 的損失分解；恆等式 L_total = L_trav + L_rank − δ_F。float 或 Fraction 都可以。"""
    return {"L_total": r_f - r_s, "L_trav": t_f - t_s, "L_rank": t_s - r_s, "delta_F": t_f - r_f}


def _exact(value, total: int):
    """recall 是命中數 / total；換成精確的分數，門檻比較才不受浮點誤差影響。"""
    if value is None:
        return None
    hits = round(value * total)
    if abs(hits / total - value) > 1e-12:
        raise ReportError(f"recall {value!r} is not a multiple of 1/{total}")
    return Fraction(hits, total)


def _float(value):
    return None if value is None else float(value)


def _ratio(numerator, denominator) -> Fraction:
    """兩個量測值（float）之比的精確值。"""
    return Fraction(numerator) / Fraction(denominator)


def _range(values) -> dict | None:
    present = [v for v in values if v is not None]
    return None if not present else {"min": float(min(present)), "max": float(max(present)), "values": [float(v) for v in present]}


# ---- 從 run 紀錄取值 ----


def _in_cell(build) -> bool:
    return (build["n"], build["m"], build["ef_construction"], build["shuffle"]) == (N, M, EF_CONSTRUCTION, None)


def point_table(records: list[dict]) -> dict:
    """(實作, seed, ef) → `report.point_rows` 的列（只有 (100k, 16, 200)、原始順序）。"""
    return {(r["implementation"], r["seed"], r["ef_search"]): r for r in point_rows(records)
            if (r["n"], r["m"], r["ef_construction"], r["shuffle"]) == (N, M, EF_CONSTRUCTION, None)}


def build_table(records: list[dict]) -> dict:
    """(實作, seed) → build 的記憶體讀值與標記（子 process 與 build 兩層，與 report.build_rows 相同）。"""
    out = {}
    for record in records:
        for child in record["children"]:
            b = child["build"]
            if not _in_cell(b):
                continue
            out[(child["implementation"], b["seed"])] = {
                "status": child["status"], "footprint": b.get("index_footprint_bytes"),
                "relieved": b.get("index_footprint_relieved_bytes"), "rss": b.get("index_memory_bytes"),
                "rss_relieved": b.get("index_memory_relieved_bytes"), "serialized": b.get("serialized_bytes"),
                "build_ns": b.get("build_ns"), "peak_rss": b.get("peak_rss_bytes"),
                "flags": sorted({*child["flags"], *b.get("flags", [])} & set(HYGIENE_FLAGS))}
    return out


def _ok(row) -> bool:
    return row is not None and row["status"] == "ok"


def _strict_recall(row, total: int):
    """019 §2 的「formal recall 存在」：點存在、不是 incomplete（含結果不足 k）。"""
    if not _ok(row) or row["incomplete"]:
        return None
    return _exact(row["recall10"], total)


def _diag_arm(diag, arm, seed) -> dict | None:
    return ((diag or {}).get("rust", {}).get(str(seed)) or {}).get("arms", {}).get(arm)


def _ids_valid(diag, arm, seed) -> bool:
    """019 §7 第 3 步：該 (seed, 臂) 的 k = 10 ids 與 formal run 逐位元相同（不符的 (seed, 臂) 在 H1、H2 中無效）。"""
    entry = _diag_arm(diag, arm, seed)
    return bool(entry and entry.get("ids_valid"))


def _diag_point(diag, arm, seed, ef) -> dict | None:
    entry = _diag_arm(diag, arm, seed)
    return None if not entry else entry.get("points", {}).get(str(ef))


def d_value(diag, arm, seed, ef):
    """D_X(s, e)：只有在 §7 的 ids 核對通過時有效（019 §2）；否則 None。"""
    point = _diag_point(diag, arm, seed, ef)
    if not _ids_valid(diag, arm, seed) or point is None or point.get("formal_ids_match") is not True:
        return None
    return point["kernel_calls_mean"]


def _t_value(diag, arm, seed, ef, total):
    point = _diag_point(diag, arm, seed, ef)
    return None if point is None else _exact(point.get("rerank_recall10"), total)


# ---- 讀值 1：frontier（010 §2、§3 與 013 §3，不修改） ----


def frontier_ratios(front: dict, pair, applicable: bool = True) -> dict:
    """S ÷ F、Q ÷ P：兩者的主要讀值都達到目標才計算（019 §6）；Q ÷ P 在 C7b 不一致時不適用。"""
    fp32, sq8 = pair
    out = {}
    for target in TARGETS:
        key = _target_key(target)
        if not applicable:
            out[key] = NA
            continue
        a = front.get(fp32, {}).get(str(N), {}).get(key, {}).get("main", {})
        b = front.get(sq8, {}).get(str(N), {}).get(key, {}).get("main", {})
        if not (a.get("reached") and b.get("reached")):
            out[key] = None
            continue
        out[key] = {metric: b[metric]["value"] / a[metric]["value"]
                    for metric in ("qps", "p50_ns", "p95_ns", "index_footprint_bytes") if a.get(metric) and b.get(metric)}
    return out


def _qf_target(aggregates: list[dict], fp32: str, sq8: str, target: str) -> dict:
    """015 §1 的 qps_bracket 與 seed 分開規則，套用到單一格點 (16, 200) 的 SQ8 對 FP32（019 §6）。"""
    t = float(target)
    a, b = qps_bracket(aggregates, fp32, VARIANCE_N, t), qps_bracket(aggregates, sq8, VARIANCE_N, t)
    entry = {"brackets": {fp32: a, sq8: b}, "direction": None, "cell": None}
    if not (a["reached"] and b["reached"]):
        return {**entry, "verdict": "undetermined"}
    if a["upper"] is not None and b["lower"] > a["upper"]:
        direction, faster, slower = "sq8_faster", sq8, fp32
    elif b["upper"] is not None and a["lower"] > b["upper"]:
        direction, faster, slower = "fp32_faster", fp32, sq8
    else:
        return {**entry, "verdict": "within_quantization"}
    fast = _cell_side(aggregates, faster, M, EF_CONSTRUCTION, t, "lower")
    slow = _cell_side(aggregates, slower, M, EF_CONSTRUCTION, t, "upper")
    separated = fast["status"] == slow["status"] == "ok" and fast["value"] > slow["value"]
    cell = {"faster": {"implementation": faster, **fast}, "slower": {"implementation": slower, **slow},
            "separated": separated}
    return {**entry, "direction": direction, "cell": cell, "verdict": direction if separated else "uncertain"}


def quant_frontier(clean_aggregates, all_aggregates, pair) -> dict:
    """主要判讀排除有 hygiene 標記的 build；另以全部 build 判讀一次，不同時兩者並列（010 §3）。"""
    result = {}
    for target in QF_TARGETS:
        main = _qf_target(clean_aggregates, *pair, target)
        everything = _qf_target(all_aggregates, *pair, target)
        result[target] = {"main": main, "all_builds": everything, "differs": main["verdict"] != everything["verdict"]}
    return result


# ---- 讀值 2：同 (seed, ef) 的配對讀值 ----

TIME_KEYS = ("rho_qps", "rho_eval", "rho_p50", "rho_p95")
DIRECTION_KEYS = ("rho_qps", "rho_eval")  # 019 §2 H1 的其他讀值：方向與 e* 不一致時照實寫明
RECALL_KEYS = ("delta_recall10", "delta_recall1")
MEMORY_KEYS = ("delta_m014", "delta_relief", "delta_struct")


def _struct_bytes(diag, kind, arm, seed):
    entry = (diag or {}).get(kind, {}).get(str(seed)) or {}
    table = entry.get("memory_accounting" if kind == "rust" else "struct_bytes") or {}
    return (table.get(arm) or {}).get("total_bytes")


def _build_struct_bytes(diag, impl, seed):
    """019 §6 記憶體表的 M_struct（診斷）：F、S 取 §7 的 `memory_accounting`，P、Q 取 FAISS 的陣列大小；R 沒有。"""
    for kind, pair in PAIRS.items():
        if impl in pair:
            return _struct_bytes(diag, kind, impl, seed)
    return None


def recall_by_ef(aggregates: list[dict]) -> list[dict]:
    """019 §6 的 recall 指標：每個臂、每個 ef 的 Recall@10（id-based，主要）、tie-aware Recall@10 與 Recall@1（跨 seed，全部 build）。"""
    return [{"implementation": a["implementation"], "ef_search": a["ef_search"], "builds": a["builds"],
             **{key: a[key] for key in ("recall10", "tie10", "recall1")}}
            for a in sorted(aggregates, key=lambda a: (a["implementation"], a["ef_search"]))]


def _sub(a, b):
    return None if a is None or b is None else a - b


def _differs_from_e_star(value, reference):
    """019 §2「其他讀值」：與 e* 的方向不一致 = 兩者分屬 1 的兩側（一個 > 1、另一個 < 1）。任一缺失時為 None。"""
    if value is None or reference is None:
        return None
    return (value - 1) * (reference - 1) < 0


def paired(points, builds, diag, kind, seeds, efs, totals, e_star=None) -> dict:
    """019 §6 讀值 2：每個 (seed, ef) 的 ρ（SQ8 ÷ FP32）與 Δ（FP32 − SQ8）。

    時間類只用乾淨的配對（main），另列包含標記點的版本（all）；recall 用全部 build；記憶體類排除有 memory / swap 標記的 build。
    ρ_eval 只有 Rust（D 由 §7 取得，ids 核對通過才有效）。
    019 §2「其他讀值」：每個 ef 的 ρ_QPS、ρ_eval（乾淨配對的 median）與 e* 的方向不一致時，以 `direction_differs_from_e_star` 標明（不判定）。
    """
    fp32, sq8 = PAIRS[kind]
    per_point = []
    for seed in seeds:
        for ef in efs:
            a, b = points.get((fp32, seed, ef)), points.get((sq8, seed, ef))
            entry = {"seed": seed, "ef_search": ef}
            if a is None or b is None:
                per_point.append({**entry, "missing": True})
                continue
            flags = sorted(set(a["flags"]) | set(b["flags"]))
            ok = _ok(a) and _ok(b)
            entry.update(missing=False, flags=flags, clean=ok and not flags)
            if ok:
                entry.update(rho_qps=b["qps"] / a["qps"], rho_p50=b["p50_ns"] / a["p50_ns"], rho_p95=b["p95_ns"] / a["p95_ns"])
                for key, total in (("recall10", totals[0]), ("recall1", totals[1])):
                    if a[key] is not None and b[key] is not None:
                        entry[f"delta_{key}"] = float(_exact(a[key], total) - _exact(b[key], total))
            if kind == "rust":
                d_f, d_s = d_value(diag, fp32, seed, ef), d_value(diag, sq8, seed, ef)
                if d_f is not None and d_s is not None:
                    entry["d_ratio"] = d_s / d_f
                    if "rho_qps" in entry:
                        entry["rho_eval"] = entry["rho_qps"] * entry["d_ratio"]
            per_point.append(entry)
    by_ef = []
    for ef in efs:
        members = [e for e in per_point if e["ef_search"] == ef and not e["missing"]]
        entry = {"ef_search": ef, "clean_pairs": sum(e["clean"] for e in members), "pairs": len(members)}
        for key in TIME_KEYS:
            if kind == "faiss" and key == "rho_eval":
                continue
            entry[key] = {"main": _summary([e.get(key) for e in members if e["clean"]]),
                          "all": _summary([e.get(key) for e in members])}
        for key in RECALL_KEYS:
            entry[key] = _summary([e.get(key) for e in members])
        by_ef.append(entry)
    reference = next((e for e in by_ef if e["ef_search"] == e_star), None)
    for entry in by_ef:
        for key in DIRECTION_KEYS:
            if key in entry:
                median = (entry[key]["main"] or {}).get("median")
                base = None if reference is None else (reference[key]["main"] or {}).get("median")
                entry[key]["direction_differs_from_e_star"] = _differs_from_e_star(median, base)
    memory = []
    for seed in seeds:
        a, b = builds.get((fp32, seed)), builds.get((sq8, seed))
        if a is None or b is None:
            continue
        flags = sorted(set(a["flags"]) | set(b["flags"]))
        memory.append({"seed": seed, "flags": flags, "excluded": any(f in MEMORY_EXCLUDING for f in flags),
                       "delta_m014": _sub(a["footprint"], b["footprint"]), "delta_relief": _sub(a["relieved"], b["relieved"]),
                       "delta_struct": _sub(_struct_bytes(diag, kind, fp32, seed), _struct_bytes(diag, kind, sq8, seed))})
    memory_summary = {key: {"main": _summary([e[key] for e in memory if not e["excluded"]]),
                            "all": _summary([e[key] for e in memory])} for key in MEMORY_KEYS}
    return {"per_point": per_point, "by_ef": by_ef, "memory": memory, "memory_summary": memory_summary}


# ---- 一致性檢查 C1–C8（019 §2） ----


def c1_memory(builds, kind, seeds, dim) -> dict:
    """ΔM_relief 對照 Δ_theory = 3·n·d − 8·d；|差| ≤ max(5% × Δ_theory, 2 MiB) 為一致（記憶體類排除 memory / swap 標記）。"""
    fp32, sq8 = PAIRS[kind]
    theory = 3 * N * dim - 8 * dim
    tolerance = max(C1_RELATIVE * theory, Fraction(C1_ABSOLUTE))
    per_seed = []
    for seed in seeds:
        a, b = builds.get((fp32, seed)), builds.get((sq8, seed))
        if a is None or b is None:
            per_seed.append({"seed": seed, "missing": True})
            continue
        flags = sorted(set(a["flags"]) | set(b["flags"]))
        delta = _sub(a["relieved"], b["relieved"])
        entry = {"seed": seed, "missing": delta is None, "flags": flags,
                 "excluded": any(f in MEMORY_EXCLUDING for f in flags), "delta_relief": delta,
                 "delta_m014": _sub(a["footprint"], b["footprint"]),
                 "retention": {fp32: _sub(a["footprint"], a["relieved"]), sq8: _sub(b["footprint"], b["relieved"])}}
        if delta is not None:
            entry.update(deviation=delta - theory, within=abs(delta - theory) <= tolerance)
        per_seed.append(entry)
    used = [e for e in per_seed if not e["missing"] and not e["excluded"]]
    if FAISS_FALLBACK and kind == "faiss":
        status = NA
    elif not used:
        status = UNDETERMINED
    else:
        status = CONSISTENT if all(e["within"] for e in used) else INCONSISTENT
    return {"status": status, "theory_bytes": theory, "tolerance_bytes": float(tolerance), "per_seed": per_seed}


def c3_ceiling(points, diag, efs, total) -> dict:
    """R_S(s, e) ≤ c + 0.002：所有 seed、所有 ef（019 §2）。"""
    c = _exact(((diag or {}).get("ceiling") or {}).get("recall10"), total)
    values = []
    for seed in RUST_SEEDS:
        for ef in efs:
            row = points.get((S, seed, ef))
            if _ok(row) and row["recall10"] is not None:
                values.append((seed, ef, _exact(row["recall10"], total)))
    if c is None or not values:
        return {"status": UNDETERMINED, "ceiling": _float(c), "violations": []}
    violations = [{"seed": s, "ef_search": e, "recall10": float(v)} for s, e, v in values if v > c + C3_SLACK]
    return {"status": INCONSISTENT if violations else CONSISTENT, "ceiling": float(c),
            "max_recall10": float(max(v for _, _, v in values)), "violations": violations}


def c4_path(points, e_star) -> dict:
    """QPS_F / QPS_R（seed 1、e*）在 [0.90, 1.10] 內為一致，否則 path-distorted。
    時間類讀值只用乾淨的點（019 §6）：任一點有 hygiene 標記或缺失時記為無法判定，數字照列。"""
    f, r = points.get((F, ANCHOR_SEED, e_star)), points.get((R, ANCHOR_SEED, e_star))
    if not (_ok(f) and _ok(r)):
        return {"status": UNDETERMINED, "ratio": None, "flags": [], "note": "seed 1 在 e* 沒有 F 或 R 的點"}
    ratio = _ratio(f["qps"], r["qps"])
    flags = sorted(set(f["flags"]) | set(r["flags"]))
    if flags:
        return {"status": UNDETERMINED, "ratio": float(ratio), "flags": flags, "note": "有 hygiene 標記"}
    low, high = C4_RANGE
    return {"status": CONSISTENT if low <= ratio <= high else PATH_DISTORTED, "ratio": float(ratio), "flags": []}


def c5_gate(profile, e_star) -> dict:
    """R 在 e* 的取樣占比對照 Gate profiling（traversal、visited set、heap），各類相差 ≤ 3 個百分點為一致（只作描述）。"""
    sample = (profile or {}).get("samples", {}).get((R, e_star))
    gate = (profile or {}).get("gate")
    if sample is None or not sample.get("ok") or not sample.get("total") or not gate or not gate.get("total"):
        return {"status": UNDETERMINED, "differences": None,
                "note": "R 的取樣沒有通過 §8 的 check" if sample is None or not sample.get("ok") else "沒有 Gate profiling 的取樣"}
    shares = {cat: Fraction(sample["counts"].get(cat, 0), sample["total"]) for cat in C5_CATEGORIES}
    gate_shares = {cat: Fraction(gate["counts"].get(cat, 0), gate["total"]) for cat in C5_CATEGORIES}
    diffs = {cat: shares[cat] - gate_shares[cat] for cat in C5_CATEGORIES}
    return {"status": CONSISTENT if all(abs(d) <= C5_MAX for d in diffs.values()) else INCONSISTENT,
            "shares": {cat: float(v) for cat, v in shares.items()}, "gate_shares": {cat: float(v) for cat, v in gate_shares.items()},
            "differences": {cat: float(v) for cat, v in diffs.items()}}


def c6_drift(points, core, efs) -> dict:
    """C 中 R 的 QPS ÷ Core rust-hnsw seed 1 的 QPS（共同的 ef；只描述）。"""
    ratios = []
    for ef in efs:
        row, base = points.get((R, ANCHOR_SEED, ef)), core["qps"].get((R, ANCHOR_SEED, ef))
        if _ok(row) and base:
            ratios.append({"ef_search": ef, "ratio": row["qps"] / base, "flags": row["flags"]})
    return {"status": DESCRIBED, "ratios": ratios}


def _compare(mine, theirs) -> str:
    if mine is None:
        return "c_missing"
    if theirs is None:
        return "core_missing"
    return "identical" if mine.shape == theirs.shape and np.array_equal(mine, theirs) else "different"


def _c7_status(comparisons) -> str:
    results = {c["result"] for c in comparisons}
    if "different" in results:
        return INCONSISTENT
    if not comparisons or results - {"identical"}:
        return UNDETERMINED
    return CONSISTENT


def c7_core(c_ids, core, efs, plan) -> dict:
    """C7a / C7b：`plan` 是 [(C 的臂, seeds)]；每個共同 ef 的 `ids_k10` 對照 Core 正式 run 的 points 檔（019 §2）。"""
    comparisons = []
    for impl, seeds in plan:
        core_impl = CORE_IMPL[impl]
        for seed in seeds:
            core_efs = core["efs"].get((core_impl, seed))
            if core_efs is None:
                comparisons.append({"against": "core", "implementation": impl, "seed": seed, "ef_search": None,
                                    "result": "core_missing"})
                continue
            for ef in efs:
                if ef in core_efs:
                    comparisons.append({"against": "core", "implementation": impl, "seed": seed, "ef_search": ef,
                                        "result": _compare(c_ids.get((impl, seed, ef)), core["ids"].get((core_impl, seed, ef)))})
    return comparisons


def c7a(c_ids, core, efs) -> dict:
    comparisons = c7_core(c_ids, core, efs, [(F, RUST_SEEDS), (R, (ANCHOR_SEED,))])
    # 另比對 C 的 seed 1 中 F 與 R（所有 ef）。
    for ef in efs:
        result = _compare(c_ids.get((F, ANCHOR_SEED, ef)), c_ids.get((R, ANCHOR_SEED, ef)))
        comparisons.append({"against": "C seed 1 rust-hnsw", "implementation": F, "seed": ANCHOR_SEED, "ef_search": ef,
                            "result": "c_missing" if result == "core_missing" else result})
    return {"status": _c7_status(comparisons), "comparisons": _without_identical(comparisons),
            "identical": sum(c["result"] == "identical" for c in comparisons)}


def c7b(c_ids, core, efs) -> dict:
    comparisons = c7_core(c_ids, core, efs, [(P, FAISS_SEEDS)])
    return {"status": _c7_status(comparisons), "comparisons": _without_identical(comparisons),
            "identical": sum(c["result"] == "identical" for c in comparisons)}


def _without_identical(comparisons):
    return [c for c in comparisons if c["result"] != "identical"]


def c8_faiss(diag) -> dict:
    """移植後 Q 的 graph 與 Flat 相同（每個 FAISS seed）；重建的 Q 與 formal 的 ids 不符時該 seed 無法判定（019 §7）。

    診斷無法與 formal 綁定的另外兩種情形也記為無法判定，並以 `reason` 標明（019 只寫了 ids 不符，記入 019 的補記）：
    formal 的 Q 在 e* 沒有 ids（該點缺失），或診斷沒有這個 seed。
    """
    per_seed = []
    for seed in FAISS_SEEDS:
        entry = ((diag or {}).get("faiss") or {}).get(str(seed))
        reason = None
        if FAISS_FALLBACK:
            status = NA
        elif entry is None or entry.get("ids_match_at_e_star") is not True:
            status = UNDETERMINED
            reason = ("diag_missing" if entry is None else
                      "ids_mismatch" if entry.get("ids_match_at_e_star") is False else "formal_ids_missing")
        elif not entry.get("graph_identical"):
            status = INCONSISTENT
        else:
            status = CONSISTENT
        per_seed.append({"seed": seed, "status": status, "reason": reason, "graph_identical": (entry or {}).get("graph_identical"),
                         "ids_match_at_e_star": (entry or {}).get("ids_match_at_e_star"),
                         "codes_agreement": (entry or {}).get("codes_agreement"),
                         "codebook_identical": (entry or {}).get("codebook_identical")})
    statuses = {e["status"] for e in per_seed}
    if FAISS_FALLBACK:
        status = NA
    elif INCONSISTENT in statuses:
        status = INCONSISTENT
    elif UNDETERMINED in statuses:
        status = UNDETERMINED
    else:
        status = CONSISTENT
    return {"status": status, "per_seed": per_seed}


# ---- H1、H2 每個資料集的輸入 ----


def h1_pairs(points, diag, e_star) -> dict:
    """e* 的 ρ_eval；只用乾淨的配對，而且兩臂的 D 都有效（019 §2）。"""
    pairs = []
    for seed in RUST_SEEDS:
        f, s = points.get((F, seed, e_star)), points.get((S, seed, e_star))
        entry = {"seed": seed, "clean": False, "flags": [], "rho_qps": None, "d_ratio": None, "rho_eval": None}
        exact = None
        if _ok(f) and _ok(s):
            entry["flags"] = sorted(set(f["flags"]) | set(s["flags"]))
            entry.update(clean=not entry["flags"], rho_qps=s["qps"] / f["qps"])
        d_f, d_s = d_value(diag, F, seed, e_star), d_value(diag, S, seed, e_star)
        if d_f is not None and d_s is not None:
            entry["d_ratio"] = d_s / d_f
            if entry["rho_qps"] is not None:
                exact = _ratio(s["qps"], f["qps"]) * _ratio(d_s, d_f)
                entry["rho_eval"] = float(exact)
        entry["usable"] = entry["clean"] and exact is not None
        entry["_exact"] = exact
        pairs.append(entry)
    values = [e["_exact"] for e in pairs if e["usable"]]
    for e in pairs:
        del e["_exact"]
    return {"e_star": e_star, "pairs": pairs, "values": values, "range": _range(values)}


def h1d_inputs(points, diag, profile, e_star) -> dict:
    """seed 1、e* 的 κ 與 ν（019 §2、§8）。標記、D 無效或取樣沒有通過 check 時照算並列出原因，H1d 記為無法判定。"""
    reasons, values = [], {}
    for arm in (F, S):
        row = points.get((arm, ANCHOR_SEED, e_star))
        if not _ok(row):
            reasons.append(f"{arm}: seed 1 在 e* 沒有 formal 點")
        elif row["flags"]:
            reasons.append(f"{arm}: seed 1 在 e* 有 hygiene 標記 {row['flags']}")
        d = d_value(diag, arm, ANCHOR_SEED, e_star)
        if d is None:
            reasons.append(f"{arm}: seed 1 的 D 無效（§7 的 ids 核對沒有通過或缺少診斷）")
        sample = (profile or {}).get("samples", {}).get((arm, e_star))
        if sample is None or not sample.get("ok"):
            reasons.append(f"{arm}: e* 的取樣沒有通過 §8 的 check")
        f_k = None
        if sample is not None and sample.get("total"):
            f_k = Fraction(sample["counts"].get("kernel", 0), sample["total"])
        values[arm] = {"qps": row["qps"] if _ok(row) else None, "d": d, "f_k": f_k}
    out = {"reasons": reasons, "f_k": {arm: _float(v["f_k"]) for arm, v in values.items()},
           "qps": {arm: v["qps"] for arm, v in values.items()}, "d": {arm: v["d"] for arm, v in values.items()},
           "kappa": None, "nu": None, "nu_reading": None}
    if all(v["qps"] and v["d"] and v["f_k"] is not None for v in values.values()):
        # t_k(X) = f_k(X) / (QPS_X · D_X)；以精確的分數計算（判定用），輸出 float。
        t_k = {arm: v["f_k"] / (Fraction(v["qps"]) * Fraction(v["d"])) for arm, v in values.items()}
        out["t_k"] = {arm: float(v) for arm, v in t_k.items()}
        if t_k[F] > 0:
            out["_kappa"] = t_k[S] / t_k[F]
            out["kappa"] = float(out["_kappa"])
        else:
            reasons.append("F 的 kernel 沒有任何樣本，κ 無法計算")
        rest = {arm: (1 - v["f_k"]) / (Fraction(v["qps"]) * Fraction(v["d"])) for arm, v in values.items()}
        if rest[F] > 0:
            nu = rest[S] / rest[F]
            out["nu"] = float(nu)
            low, high = NU_RANGE
            out["nu_reading"] = ("ν < 0.90：cache spillover（向量變小之後，graph 與 visited set 也跟著變快）" if nu < low else
                                 "ν > 1.10：原本藏在長相依鏈後面的非 kernel 工作露出來了" if nu > high else "ν 在 [0.90, 1.10] 內")
    out["complete"] = not reasons and out["kappa"] is not None
    return out


def h2_inputs(points, diag, e_star, efs, total) -> dict:
    """有效 seed（019 §2）：F 與 S 在 e* 的 formal recall 都存在，兩臂都通過 §7 的 ids 核對，且有重排的 T。"""
    seeds, valid = [], []
    for seed in RUST_SEEDS:
        r_f, r_s = _strict_recall(points.get((F, seed, e_star)), total), _strict_recall(points.get((S, seed, e_star)), total)
        t_f, t_s = _t_value(diag, F, seed, e_star, total), _t_value(diag, S, seed, e_star, total)
        ids = _ids_valid(diag, F, seed) and _ids_valid(diag, S, seed)
        entry = {"seed": seed, "ids_valid": ids, "formal_recall": r_f is not None and r_s is not None}
        if None not in (r_f, r_s, t_f, t_s) and ids:
            parts = decompose(r_f, r_s, t_f, t_s)
            threshold = H2_SLOPE * parts["L_total"] + H2_OFFSET
            entry.update(valid=True, exact={**parts, "thr": threshold})
            valid.append(entry)
        else:
            entry["valid"] = False
        seeds.append(entry)
    others = []
    for seed in RUST_SEEDS:
        for ef in efs:
            values = (_strict_recall(points.get((F, seed, ef)), total), _strict_recall(points.get((S, seed, ef)), total),
                      _t_value(diag, F, seed, ef, total), _t_value(diag, S, seed, ef, total))
            if None not in values:
                others.append({"seed": seed, "ef_search": ef,
                               "valid": any(e["seed"] == seed and e["valid"] for e in seeds),
                               **{k: float(v) for k, v in decompose(*values).items()}})
    return {"seeds": seeds, "valid_seeds": [e["seed"] for e in valid], "other_ef": others}


def c2_rerank(h2: dict, points, diag, efs, total) -> dict:
    """δ_F 的絕對值 ≤ 0.002：所有有效 seed、所有 ef（019 §2）。"""
    values = []
    for seed in h2["valid_seeds"]:
        for ef in efs:
            r_f, t_f = _strict_recall(points.get((F, seed, ef)), total), _t_value(diag, F, seed, ef, total)
            if r_f is not None and t_f is not None:
                values.append({"seed": seed, "ef_search": ef, "delta_F": t_f - r_f})
    if not values:
        return {"status": UNDETERMINED, "max_abs": None, "violations": []}
    violations = [{"seed": v["seed"], "ef_search": v["ef_search"], "delta_F": float(v["delta_F"])}
                  for v in values if abs(v["delta_F"]) > C2_MAX]
    return {"status": INCONSISTENT if violations else CONSISTENT,
            "max_abs": float(max(abs(v["delta_F"]) for v in values)), "violations": violations}


def h2_status(h2: dict, c2_status: str) -> tuple[str, str, dict]:
    """019 §2 H2 的判定表（各列依序取第一個符合的列）。"""
    valid = [e for e in h2["seeds"] if e["valid"]]
    numbers = {"valid_seeds": [e["seed"] for e in valid],
               "per_seed": [{"seed": e["seed"], **{k: float(v) for k, v in e["exact"].items()}} for e in valid]}
    if len(valid) < H2_MIN_SEEDS:
        return UNDETERMINED, "有效 seed 不足", numbers
    if c2_status == INCONSISTENT:
        return UNDETERMINED, "比較有問題（C2 不一致）：先調查，不判定", numbers
    median = statistics.median(e["exact"]["L_total"] for e in valid)
    numbers["median_L_total"] = float(median)
    if median < H2_BETTER:
        return UNDETERMINED, "比較有問題（SQ8 系統性地比 FP32 好）：先調查，不判定", numbers
    if median < H2_NEGLIGIBLE:
        return NA, "SQ8 幾乎無損，分解低於解析度", numbers
    if all(e["exact"]["L_trav"] <= e["exact"]["thr"] for e in valid):
        return HOLDS, "", numbers
    if all(e["exact"]["L_trav"] > e["exact"]["thr"] for e in valid):
        return FAILS, "SQ8 的距離誤差帶偏了走訪，只重排 ef 個候選救不回損失", numbers
    return UNCERTAIN, "", numbers


# ---- 判定表（019 §6） ----


def _row(kind, label, status, numbers=None, note="", dataset=None) -> dict:
    return {"row": label, "kind": kind, "status": status, "numbers": numbers or {}, "note": note, "dataset": dataset}


def _range_status(values, holds, fails) -> str:
    low, high = min(values), max(values)
    if holds(low, high):
        return HOLDS
    if fails(low, high):
        return FAILS
    return UNCERTAIN


def h1_overall(premise, subs: dict, h1d: str) -> str:
    """019 §2 H1 整體判定（各列依序取第一個符合的列）。"""
    if premise != HOLDS:
        return PREMISE_FAILS
    values = list(subs.values())
    if FAILS in values:
        return FAILS
    if all(v == HOLDS for v in values):
        if h1d == CONSISTENT:
            return HOLDS
        if h1d == INCONSISTENT:
            return TIMING_ONLY
        return UNDETERMINED
    if UNDETERMINED in values:
        return UNDETERMINED
    return UNCERTAIN


def h2_overall(statuses: dict) -> str:
    applicable = [s for s in statuses.values() if s != NA]
    if not applicable:
        return NA
    if FAILS in applicable:
        return FAILS
    if all(s == HOLDS for s in applicable):
        return HOLDS
    if UNDETERMINED in applicable:
        return UNDETERMINED
    return UNCERTAIN


def premise_status(premise: dict) -> tuple[str, dict]:
    kernels = (premise or {}).get("kernels", {})
    ok = all((kernels.get(name) or {}).get("ok") is True for name in ("rhnsw_kernel_fp32", "rhnsw_kernel_sq8"))
    numbers = {name: {k: (kernels.get(name) or {}).get(k) for k in ("ok", "accumulator", "accumulations", "violations")}
               for name in ("rhnsw_kernel_fp32", "rhnsw_kernel_sq8")}
    numbers["rhnsw_kernel_sq8"]["conversions"] = (kernels.get("rhnsw_kernel_sq8") or {}).get("conversions")
    return (HOLDS if ok else PREMISE_FAILS), numbers


def judge(parts: dict, premise: dict) -> dict:
    """完整判定表：H1（a–d、前提、整體）、H2（兩個資料集、整體）、quantization frontier（2 × 2 × 2）與 C1–C8。"""
    glove, text = parts[GLOVE], parts[TEXT]
    c7a_inconsistent = any(p["checks"]["C7a"]["status"] == INCONSISTENT for p in parts.values())
    rows = []

    premise_value, premise_numbers = premise_status(premise)
    rows.append(_row("h1_premise", "H1 前提（反組譯）", premise_value, premise_numbers))

    def sub(label, part, holds, fails):
        values = part["h1"]["values"]
        numbers = {"e_star": part["h1"]["e_star"], "clean_pairs": len(values), "rho_eval": part["h1"]["range"]}
        if len(values) < MIN_CLEAN_PAIRS:
            return _row("h1_sub", label, UNDETERMINED, numbers, "乾淨配對少於 3 個")
        return _row("h1_sub", label, _range_status(values, holds, fails), numbers)

    low_a, high_a = H1A_RANGE
    h1a = sub("H1a 文字", text, lambda lo, hi: low_a <= lo and hi <= high_a, lambda lo, hi: lo > high_a or hi < low_a)
    h1b = sub("H1b GloVe", glove, lambda lo, hi: lo >= H1B_MIN, lambda lo, hi: hi < H1B_MIN)
    g, t = glove["h1"]["values"], text["h1"]["values"]
    c_numbers = {"glove": glove["h1"]["range"], "text": text["h1"]["range"]}
    if len(g) < MIN_CLEAN_PAIRS or len(t) < MIN_CLEAN_PAIRS:
        h1c = _row("h1_sub", "H1c 維度對比", UNDETERMINED, c_numbers, "乾淨配對少於 3 個")
    elif min(g) > max(t):
        h1c = _row("h1_sub", "H1c 維度對比", HOLDS, c_numbers)
    elif max(g) < min(t):
        h1c = _row("h1_sub", "H1c 維度對比", FAILS, c_numbers)
    else:
        h1c = _row("h1_sub", "H1c 維度對比", UNCERTAIN, c_numbers)
    d_numbers = {name: {k: part["h1d"].get(k) for k in ("kappa", "nu", "f_k", "reasons")}
                 for name, part in (("glove", glove), ("text", text))}
    if not (glove["h1d"]["complete"] and text["h1d"]["complete"]):
        h1d = _row("h1d", "H1d profiler（seed 1）", UNDETERMINED, d_numbers, "見各資料集的原因")
    else:
        low_d, high_d = H1D_TEXT_RANGE
        consistent = low_d <= text["h1d"]["_kappa"] <= high_d and glove["h1d"]["_kappa"] <= H1D_GLOVE_MAX
        h1d = _row("h1d", "H1d profiler（seed 1）", CONSISTENT if consistent else INCONSISTENT, d_numbers)
    overall = h1_overall(premise_value, {"a": h1a["status"], "b": h1b["status"], "c": h1c["status"]}, h1d["status"])
    h1 = _row("h1_overall", "H1 整體", overall,
              {"premise": premise_value, "H1a": h1a["status"], "H1b": h1b["status"], "H1c": h1c["status"], "H1d": h1d["status"]})
    rows += [h1a, h1b, h1c, h1d, h1]

    h2_rows = {}
    for dataset, label in ((GLOVE, "H2 GloVe"), (TEXT, "H2 文字")):
        status, note, numbers = h2_status(parts[dataset]["h2"], parts[dataset]["checks"]["C2"]["status"])
        h2_rows[dataset] = _row("h2_dataset", label, status, numbers, note, dataset)
    h2 = _row("h2_overall", "H2 整體", h2_overall({d: r["status"] for d, r in h2_rows.items()}),
              {SHORT_NAMES[d]: r["status"] for d, r in h2_rows.items()})
    rows += [h2_rows[GLOVE], h2_rows[TEXT], h2]

    if c7a_inconsistent:
        # 019 §6：C7a 不一致時，H1a–H1d、H1 整體與 H2 的所有列都只能是「不判定」（H1 前提是靜態檢查，照列）。
        for row in rows:
            if row["kind"] in HYPOTHESIS_KINDS:
                row["numbers"] = {**row["numbers"], "status_before_stop": row["status"]}
                row["status"] = NOT_JUDGED

    for dataset in (GLOVE, TEXT):
        part = parts[dataset]
        short = SHORT_NAMES[dataset]
        for kind, pair_name in (("rust", PAIR_NAMES["rust"]), ("faiss", PAIR_NAMES["faiss"])):
            reading = part["quant_frontier"][kind]
            for target in QF_TARGETS:
                label = f"quantization frontier {short} {pair_name} {target}"
                if reading == NA:
                    rows.append(_row("qf_faiss", label, NA, {}, "C7b 不一致", dataset))
                    continue
                entry = reading[target]
                numbers = {"brackets": entry["main"]["brackets"], "cell": entry["main"]["cell"],
                           "all_builds": entry["all_builds"]["verdict"], "differs": entry["differs"]}
                rows.append(_row(f"qf_{kind}", label, entry["main"]["verdict"], numbers, "", dataset))
    for dataset in (GLOVE, TEXT):
        checks = parts[dataset]["checks"]
        short = SHORT_NAMES[dataset]
        rows.append(_row("c1_rust", f"C1 記憶體 {short} F − S", checks["C1"]["rust"]["status"], _c1_numbers(checks["C1"]["rust"]),
                         dataset=dataset))
        rows.append(_row("c1_faiss", f"C1 記憶體 {short} P − Q", checks["C1"]["faiss"]["status"],
                         _c1_numbers(checks["C1"]["faiss"]), "採用 §3 的備案" if FAISS_FALLBACK else "", dataset))
        rows.append(_row("c_basic", f"C2 FP32 重排 {short}", checks["C2"]["status"],
                         {"max_abs_delta_F": checks["C2"]["max_abs"], "violations": checks["C2"]["violations"]}, dataset=dataset))
        rows.append(_row("c_basic", f"C3 天花板 {short}", checks["C3"]["status"],
                         {k: checks["C3"].get(k) for k in ("ceiling", "max_recall10", "violations")}, dataset=dataset))
        rows.append(_row("c4", f"C4 路徑失真 {short}", checks["C4"]["status"],
                         {"qps_f_over_r": checks["C4"]["ratio"], "flags": checks["C4"]["flags"]}, checks["C4"].get("note", ""),
                         dataset))
        rows.append(_row("c_basic", f"C5 Gate 重現 {short}", checks["C5"]["status"],
                         {k: checks["C5"].get(k) for k in ("shares", "gate_shares", "differences")}, checks["C5"].get("note", ""),
                         dataset))
        rows.append(_row("c6", f"C6 漂移 {short}", DESCRIBED, {"ratios": checks["C6"]["ratios"]}, dataset=dataset))
        for key in ("C7a", "C7b"):
            rows.append(_row("c7", f"{key} 逐位元 {short}", checks[key]["status"],
                             {"identical": checks[key]["identical"], "not_identical": checks[key]["comparisons"]},
                             dataset=dataset))
        rows.append(_row("c8", f"C8 FAISS {short}", checks["C8"]["status"], {"per_seed": checks["C8"]["per_seed"]},
                         "codes 一致比例只描述", dataset))
    context = {"c7a_inconsistent": c7a_inconsistent,
               "c7b_inconsistent": {d: parts[d]["checks"]["C7b"]["status"] == INCONSISTENT for d in parts},
               "fallback": FAISS_FALLBACK}
    validate_table(rows, context)
    return {"rows": rows, "context": context, "expectations": expectations(parts)}


def _c1_numbers(c1: dict) -> dict:
    return {"theory_bytes": c1["theory_bytes"], "tolerance_bytes": c1["tolerance_bytes"],
            "per_seed": [{k: e.get(k) for k in ("seed", "delta_relief", "deviation", "within", "excluded", "flags",
                                                 "delta_m014", "retention")} for e in c1["per_seed"]]}


def validate_table(rows: list[dict], context: dict) -> None:
    """019 §6：每一列只能是該列允許的狀態之一；條件式的狀態只在條件成立時出現。不符就拒絕輸出報告。"""
    for row in rows:
        status, kind = row["status"], row["kind"]
        if kind not in ALLOWED or status not in ALLOWED[kind]:
            raise ReportError(f"{row['row']}: status {status!r} is not allowed for this row (019 §6)")
        if kind in HYPOTHESIS_KINDS and (status == NOT_JUDGED) != context["c7a_inconsistent"]:
            raise ReportError(f"{row['row']}: {NOT_JUDGED!r} must appear exactly when C7a is inconsistent (019 §6)")
        if kind == "qf_faiss" and status == NA and not context["c7b_inconsistent"].get(row["dataset"]):
            raise ReportError(f"{row['row']}: {NA!r} is only allowed when C7b is inconsistent (019 §6)")
        if kind in ("c1_faiss", "c8") and (status == NA) != context["fallback"]:
            raise ReportError(f"{row['row']}: {NA!r} is only allowed when the §3 fallback is used (019 §6)")
        if kind == "c7" and status == UNDETERMINED:
            # 019 §6：只在 Core 的 points 檔缺失時；C 自己的點缺失（c_missing）同樣無法比對（019 沒有寫，記入 019 的補記）。
            results = {c["result"] for c in row["numbers"].get("not_identical", [])}
            if not results & C7_MISSING:
                raise ReportError(f"{row['row']}: {UNDETERMINED!r} is only allowed when points are missing (019 §6)")
        if kind == "c8" and status != NA:
            seeds = row["numbers"].get("per_seed", [])
            for entry in seeds:
                # 019 §6：只在 §7 重建的 Q 與 formal 的 ids 不符時；無法綁定的另外兩種情形見 c8_faiss（記入 019 的補記）。
                if (entry["status"] == UNDETERMINED) != (entry.get("reason") in C8_UNBOUND):
                    raise ReportError(f"{row['row']}: seed {entry['seed']} is {entry['status']!r} with reason "
                                      f"{entry.get('reason')!r} (019 §6: undetermined only when the rebuilt Q is not bound)")
            statuses = {entry["status"] for entry in seeds}
            expected = INCONSISTENT if INCONSISTENT in statuses else UNDETERMINED if UNDETERMINED in statuses else CONSISTENT
            if status != expected:
                raise ReportError(f"{row['row']}: {status!r} does not follow from the per-seed states {sorted(statuses)}")


def expectations(parts: dict) -> list[dict]:
    """019 §2 的附帶預期（不作判定），照實列出。"""

    def first_ef(part, impl, target):
        points = sorted((a for a in part["aggregates"] if a["implementation"] == impl and a["recall10"] is not None),
                        key=lambda a: a["ef_search"])
        return next((a["ef_search"] for a in points if a["recall10"]["median"] >= target), None)

    c = {d: ((parts[d].get("ceiling") or {}).get("recall10")) for d in parts}
    text_099, glove_095 = first_ef(parts[TEXT], S, 0.99), first_ef(parts[GLOVE], S, 0.95)
    # 「所有臂」是 019 §2 臂代號表的五個臂，包括錨點 R。
    glove_099 = {impl: first_ef(parts[GLOVE], impl, 0.99) for impl in (*ARMS, R)}
    return [
        {"expectation": "c_文字 > c_GloVe", "observed": c,
         "met": None if None in c.values() else c[TEXT] > c[GLOVE]},
        {"expectation": "文字的 S 在 ef ≤ 512 內達到 0.99", "observed": text_099,
         "met": text_099 is not None and text_099 <= 512},
        {"expectation": "GloVe 的 S 在 ef ≤ 768 內達到 0.95", "observed": glove_095,
         "met": glove_095 is not None and glove_095 <= 768},
        {"expectation": "GloVe 的 0.99：所有臂都未達到", "observed": glove_099,
         "met": all(v is None for v in glove_099.values())},
    ]


# ---- 每個資料集的讀值 ----


def dataset_part(dataset, labelled: list[tuple[str, dict]], c_ids, core, diag, profile, premise) -> dict:
    labels, records = [label for label, _ in labelled], [record for _, record in labelled]
    _check_inputs(dataset, labels, records, diag, premise)
    nq, dim = records[0]["dataset"]["n_queries"], records[0]["dataset"]["dim"]
    totals = (nq * K, nq * K1)
    e_star = E_STAR[dataset]
    efs = sorted(records[0]["config"]["parsed"]["ef_search"])
    rows = [r for r in point_rows(records) if (r["n"], r["m"], r["ef_construction"], r["shuffle"]) == (N, M, EF_CONSTRUCTION, None)]
    all_aggregates = aggregate(rows)
    clean_aggregates = aggregate([r for r in rows if not r["flags"]])
    front = frontier(all_aggregates, clean_aggregates)
    points, builds = point_table(records), build_table(records)
    _check_diag_recall(dataset, points, diag)

    checks = {"C7a": c7a(c_ids, core, efs), "C7b": c7b(c_ids, core, efs), "C8": c8_faiss(diag)}
    faiss_open = not FAISS_FALLBACK and checks["C7b"]["status"] == CONSISTENT
    faiss_seeds = [e["seed"] for e in checks["C8"]["per_seed"] if e["status"] == CONSISTENT] if faiss_open else []
    h2 = h2_inputs(points, diag, e_star, efs, totals[0])
    checks.update(
        C1={"rust": c1_memory(builds, "rust", RUST_SEEDS, dim), "faiss": c1_memory(builds, "faiss", FAISS_SEEDS, dim)},
        C2=c2_rerank(h2, points, diag, efs, totals[0]), C3=c3_ceiling(points, diag, efs, totals[0]),
        C4=c4_path(points, e_star), C5=c5_gate(profile, e_star), C6=c6_drift(points, core, efs))
    c7b_bad = checks["C7b"]["status"] == INCONSISTENT
    faiss_paired = (paired(points, builds, diag, "faiss", faiss_seeds, efs, totals, e_star) if faiss_open
                    else {"status": NA, "reason": "C7b 不是一致（019 §6：Q 對 P 的配對讀值只對圖逐位元相同、C7b 一致的配對開放）"})
    if faiss_open:
        faiss_paired["closed_seeds"] = [e["seed"] for e in checks["C8"]["per_seed"] if e["status"] != CONSISTENT]
    return {
        "dataset": dataset, "short": SHORT_NAMES[dataset], "runs": labels, "n_queries": nq, "dim": dim, "e_star": e_star,
        "ef_search": efs, "protocol": records[0]["protocol"]["version"],
        "frontier": front,
        "frontier_ratios": {"rust": frontier_ratios(front, PAIRS["rust"]),
                            "faiss": frontier_ratios(front, PAIRS["faiss"], applicable=not c7b_bad)},
        "quant_frontier": {"rust": quant_frontier(clean_aggregates, all_aggregates, PAIRS["rust"]),
                           "faiss": NA if c7b_bad else quant_frontier(clean_aggregates, all_aggregates, PAIRS["faiss"])},
        "paired": {"rust": paired(points, builds, diag, "rust", RUST_SEEDS, efs, totals, e_star), "faiss": faiss_paired},
        "h1": h1_pairs(points, diag, e_star), "h1d": h1d_inputs(points, diag, profile, e_star), "h2": h2,
        "checks": checks, "profiles": profile_readings(points, diag, profile),
        "builds": [{"implementation": impl, "seed": seed, **values, "struct": _build_struct_bytes(diag, impl, seed)}
                   for (impl, seed), values in sorted(builds.items())],
        "recall_by_ef": recall_by_ef(all_aggregates),
        "ceiling": (diag or {}).get("ceiling"), "codes": (diag or {}).get("codes"),
        "aggregates": all_aggregates, "clean_aggregates": clean_aggregates,
    }


def _check_inputs(dataset, labels, records, diag, premise) -> None:
    """診斷與反組譯必須屬於這一組 formal run 與被量測的 binding，否則拒絕產生報告。"""
    if diag is None:
        raise ReportError(f"{dataset}: no quant_diag output (019 §7); run benchmarks.quant_diag first")
    if diag.get("dataset") != dataset:
        raise ReportError(f"diag is for {diag.get('dataset')!r}, not {dataset!r}")
    names = sorted(Path(label).name for label in labels)
    if sorted(Path(r["run"]).name for r in diag.get("formal_runs", [])) != names:
        raise ReportError(f"{dataset}: the diag was bound to other formal runs ({diag.get('formal_runs')}); rerun quant_diag")
    if premise is None:
        raise ReportError("no H1 premise check (019 §2, §8); run benchmarks.quant_profile premise first")
    for label, record in zip(labels, records):
        if record["metadata"]["extension_sha256"].get("rust_hnsw") != premise.get("binding_sha256"):
            raise ReportError(f"{label}: the disassembled binding (premise) is not the binding of the formal run")


def _check_diag_recall(dataset, points, diag) -> None:
    for seed, entry in (diag.get("rust") or {}).items():
        for arm, arm_entry in (entry.get("arms") or {}).items():
            for ef, point in (arm_entry.get("points") or {}).items():
                row = points.get((arm, int(seed), int(ef)))
                formal = None if row is None else row["recall10"]
                if point.get("formal_recall10") != formal:
                    raise ReportError(f"{dataset}: diag {arm} seed {seed} ef {ef} recorded formal Recall@10 "
                                      f"{point.get('formal_recall10')!r}, but the formal run has {formal!r}")


def profile_readings(points, diag, profile) -> list[dict]:
    """019 §8：各取樣的類別占比、每 query 的時間分解（占比 × 1 / QPS）與每次評估的時間（再除以 D）。"""
    out = []
    for (impl, ef), sample in sorted(((profile or {}).get("samples") or {}).items()):
        row = points.get((impl, ANCHOR_SEED, ef))
        qps = row["qps"] if _ok(row) else None
        d = d_value(diag, impl, ANCHOR_SEED, ef) if impl in (F, S) else None
        entry = {"implementation": impl, "ef_search": ef, "role": sample.get("role"), "ok": sample.get("ok"),
                 "problems": sample.get("problems", []), "qps": qps, "d": d, "flags": row["flags"] if row else None,
                 "shares": None, "per_query_us": None, "per_eval_ns": None, "other_top": sample.get("other_top", [])}
        if sample.get("total"):
            shares = {cat: sample["counts"].get(cat, 0) / sample["total"] for cat in sample["counts"]}
            entry["shares"] = shares
            if qps:
                entry["per_query_us"] = {cat: share * 1e6 / qps for cat, share in shares.items()}
                if d:
                    entry["per_eval_ns"] = {cat: share * 1e9 / (qps * d) for cat, share in shares.items()}
        out.append(entry)
    return out


def build_quant_report(groups: dict, c_ids: dict, core: dict, diags: dict, profiles: dict, premise: dict) -> dict:
    """`groups`：資料集 → [(run 標籤, run 紀錄)]；`c_ids` / `core`：C7 用的 ids（見 load_c_ids、load_core）；
    `diags`：資料集 → §7 的輸出；`profiles`：資料集 → §8 的核對與分類（見 collect_profiles）；`premise`：H1 前提。"""
    missing = sorted(set(QUANT_CONFIGS) - set(groups))
    if missing:
        raise ReportError(f"the C report needs the runs of both datasets (019 §6); missing {missing}")
    parts = {dataset: dataset_part(dataset, groups[dataset], c_ids.get(dataset, {}), core[dataset], diags.get(dataset),
                                   profiles.get(dataset), premise) for dataset in QUANT_CONFIGS}
    judgement = judge(parts, premise)
    return {"protocol": PROTOCOL, "judgement": judgement, "premise": premise, "datasets": parts}


# ---- 讀入檔案（CLI） ----


def _load(run_dir) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))


def load_c_ids(labelled: list[tuple[str, dict]]) -> dict:
    """C7 用的 `ids_k10`：F、R、P 的每個點（019 §2）。run.json 列出的 points 檔不存在時拒絕（輸入不完整）。"""
    out = {}
    for label, record in labelled:
        for child in record["children"]:
            build = child["build"]
            if child["implementation"] not in CORE_IMPL or not _in_cell(build):
                continue
            for point in child["points"]:
                path = Path(label) / point["file"]
                if not path.exists():
                    raise ReportError(f"{path} is listed in {label}/run.json but missing")
                with np.load(path) as arrays:
                    ids = arrays["ids_k10"] if "ids_k10" in arrays.files else None
                out[(child["implementation"], build["seed"], point["ef_search"])] = ids
    return out


def load_core(core_dirs, datasets) -> dict:
    """019 §2 登記的 Core 正式 run。每一個都必須以 --core-runs 列出；目錄不存在時記為缺失（C7 無法判定），不得默默略過。"""
    given = {Path(d).name: Path(d) for d in core_dirs}
    registered = {name for dataset in datasets for name in CORE_RUNS[dataset].values()}
    unknown = sorted(set(given) - registered)
    if unknown:
        raise ReportError(f"--core-runs {unknown} are not the Core runs registered in decision 019 §2")
    missing = sorted(registered - set(given))
    if missing:
        raise ReportError(f"--core-runs must list every Core run registered in decision 019 §2; missing {missing}")
    core = {}
    for dataset in datasets:
        entry = {"runs": {}, "efs": {}, "ids": {}, "qps": {}}
        for name in sorted(set(CORE_RUNS[dataset].values())):
            path = given[name]
            seeds = [s for s, n in CORE_RUNS[dataset].items() if n == name]
            if not (path / "run.json").exists():
                entry["runs"][name] = None
                continue
            record = _load(path)
            if record["dataset"]["name"] != dataset:
                raise ReportError(f"{path} is a run of {record['dataset']['name']}, not {dataset}")
            entry["runs"][name] = str(path)
            for child in record["children"]:
                build, impl = child["build"], child["implementation"]
                if impl not in (R, P) or not _in_cell(build) or build["seed"] not in seeds:
                    continue
                entry["efs"][(impl, build["seed"])] = [p["ef_search"] for p in child["points"]]
                for point in child["points"]:
                    key = (impl, build["seed"], point["ef_search"])
                    file = path / point["file"]
                    ids = None
                    if file.exists():
                        with np.load(file) as arrays:
                            ids = arrays["ids_k10"] if "ids_k10" in arrays.files else None
                    entry["ids"][key] = ids
                    if point["k10"]["status"] == "ok":
                        entry["qps"][key] = point["k10"]["qps"]["qps_median"]
        core[dataset] = entry
    return core


def collect_profiles(records: list[dict], profile_dir, changed_files=None) -> dict:
    """§8 的核對（quant_profile.check）與分類（取樣之前固定的規則）；Gate profiling 的 R 取樣供 C5 對照。"""
    from benchmarks import quant_profile

    dataset = records[0]["dataset"]["name"]
    samples = {}
    try:
        results = quant_profile.check(records, profiles=profile_dir, changed_files=changed_files)
    except gate_profile.PlanError as err:
        results, samples_error = [], str(err)
    else:
        samples_error = None
    for result in results:
        entry = {"ok": result["ok"], "problems": result["problems"], "role": result["role"]}
        sample = Path(profile_dir) / f"{quant_profile.profile_name(dataset, result['implementation'], result['ef'])}.sample.txt"
        if sample.exists():
            try:
                entry.update(quant_profile.categorize(quant_profile.read_rows(sample)))
            except ValueError as err:
                entry["problems"] = [*entry["problems"], f"cannot classify {sample}: {err}"]
                entry["ok"] = False
        samples[(result["implementation"], result["ef"])] = entry
    gate = None
    gate_sample = Path(profile_dir) / f"{gate_profile.profile_name(dataset, R, 'search')}.sample.txt"
    if gate_sample.exists():
        gate = quant_profile.categorize(quant_profile.read_rows(gate_sample))
    return {"samples": samples, "gate": gate, "error": samples_error}


def _group(run_dirs) -> dict:
    groups = defaultdict(list)
    for run_dir in run_dirs:
        record = _load(run_dir)
        groups[record["dataset"]["name"]].append((str(run_dir), record))
    return dict(groups)


def check_runs(groups: dict, excluded=(), roots=None, changed_files=None) -> None:
    """合併前的檢查（013 §5、019 §5）。每個資料集重用 report.py；兩個資料集之間比對 metadata 與量測程式。"""
    for dataset, labelled in groups.items():
        if dataset not in QUANT_CONFIGS:
            raise ReportError(f"{dataset!r} has no C formal configs (019 §5)")
        labels, records = [label for label, _ in labelled], [record for _, record in labelled]
        if {r["mode"] for r in records} != {"formal"}:
            raise ReportError("the C report reads formal runs only (019 §6)")
        _check_compatible(records)
        _check_mergeable(records, labels, changed_files)
        check_formal_set(records, labels, excluded, roots=roots, registry=QUANT_CONFIGS)
    for dataset, labelled in groups.items():
        for label, record in labelled:
            if record["protocol"]["version"] != PROTOCOL:
                raise ReportError(f"{label} ran protocol {record['protocol']['version']}, not {PROTOCOL} (019 §5)")
            power = ((record["metadata"].get("formal_checks") or {}).get("power") or {})
            if power.get("powermode") != POWERMODE:
                raise ReportError(f"{label} ran in power mode {power.get('powermode')!r}; every C run uses high-performance "
                                  f"mode {POWERMODE} (019 §5, decision 016)")
    # 兩個資料集之間：去掉資料集與 build，只比 metadata 與量測程式（重用 _check_mergeable；019 §5：整組 C 只能合併同一個變壓器）。
    firsts = [(labelled[0][0], {**labelled[0][1], "children": [], "dataset": {}, "subsets": {}})
              for labelled in groups.values()]
    _check_mergeable([r for _, r in firsts], [label for label, _ in firsts], changed_files)


def load_formal_set(run_dirs, excluded=(), roots=None, changed_files=None) -> dict:
    """10 份 C formal config 的 run（兩個資料集），經過與 C 報告相同的合併前檢查（check_runs），回傳資料集 → [(run 目錄, run 紀錄)]。

    019 §7、§9 第 8 步：診斷、profiling 與報告都在全部 10 份 C formal config 完成之後才執行，所以三者都以這裡確認完整的一組。
    """
    groups = _group(run_dirs)
    check_runs(groups, excluded, roots, changed_files)
    missing = sorted(set(QUANT_CONFIGS) - set(groups))
    if missing:
        raise ReportError(f"the C tools need the runs of both datasets (019 §6, §7, §9 step 8); missing {missing}")
    return groups


def dataset_name(value: str) -> str:
    """CLI 的資料集參數：完整名稱或簡稱（glove、text）。"""
    names = {**{d: d for d in QUANT_CONFIGS}, **{SHORT_NAMES[d]: d for d in QUANT_CONFIGS}}
    if value not in names:
        raise argparse.ArgumentTypeError(f"{value!r} is not a dataset of Advanced C; use one of {sorted(names)}")
    return names[value]


def write_quant_report(run_dirs, core_dirs, out_root=None, excluded=(), diag_paths=None, premise_path=None,
                       profile_dir=None, roots=None, changed_files=None, profile_changed_files=None) -> list[Path]:
    """`changed_files(a, b)`：MEASUREMENT_PATHS 的比較（合併檢查）；`profile_changed_files(a, b)`：PROFILED_PATHS 的比較（§8 的 check）。
    兩者預設以 git 比較（測試可替換）。"""
    groups = load_formal_set(run_dirs, excluded, roots, changed_files)
    diags = {}
    for path in (diag_paths or [diag_path(d) for d in QUANT_CONFIGS]):
        if Path(path).exists():
            diag = json.loads(Path(path).read_text(encoding="utf-8"))
            diags[diag.get("dataset")] = diag
    premise_file = Path(premise_path or PREMISE_PATH)
    premise = json.loads(premise_file.read_text(encoding="utf-8")) if premise_file.exists() else None
    profile_dir = Path(profile_dir or PROFILES_DIR)
    profiles = {d: collect_profiles([r for _, r in groups[d]], profile_dir, profile_changed_files) for d in QUANT_CONFIGS}
    report = build_quant_report(groups, {d: load_c_ids(groups[d]) for d in QUANT_CONFIGS}, load_core(core_dirs, list(QUANT_CONFIGS)),
                                diags, profiles, premise)
    report["excluded_runs"] = excluded_runs(excluded)
    report["core_runs"] = [str(d) for d in core_dirs]
    out_root = Path(out_root or REPORTS_DIR)
    written = []
    for dataset in QUANT_CONFIGS:
        out = out_root / f"quant-{SHORT_NAMES[dataset]}"
        out.mkdir(parents=True, exist_ok=True)
        figures = [path.name for path in plots(report, dataset, out)]
        text = markdown(report, dataset, figures)
        payload = {**report, "datasets": {dataset: report["datasets"][dataset]}, "figures": figures}
        (out / "report.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=_json_default),
                                         encoding="utf-8")
        (out / "report.md").write_text(text, encoding="utf-8")
        written.append(out)
    return written


def _json_default(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value)} is not JSON serializable")


# ---- 輸出 ----


def _num(value, digits=3, scale=1.0) -> str:
    return "—" if value is None else f"{value / scale:.{digits}f}"


def _span(summary, digits=3, scale=1.0) -> str:
    if summary is None:
        return "—"
    return (f"{summary['median'] / scale:.{digits}f} [{summary['min'] / scale:.{digits}f}, {summary['max'] / scale:.{digits}f}]"
            f"（{len(summary['values'])}）")


def _numbers_text(numbers: dict) -> str:
    return json.dumps(numbers, ensure_ascii=False, default=_json_default)


def markdown(report: dict, dataset: str, figures=()) -> str:
    part = report["datasets"][dataset]
    lines = [f"# Advanced C 報告：{dataset}（decision 019）", "",
             f"- 協定：{part['protocol']}；e* = {part['e_star']}；ef = {part['ef_search']}",
             "- 只限 Rust 的依序純量 kernel、搜尋期間的表示、n = 100,000、(16, 200)；build time 與 peak RSS 不會下降（019 §1）。",
             "- 記憶體的主要指標是 M_014（phys_footprint 的增量，014）；M_relief、M_struct 只是診斷（019 §6、§13）。", "",
             "## 完整判定表（兩個資料集共用，019 §6）", "", "| 列 | 狀態 | 說明 | 判定所用的數字 |", "| --- | --- | --- | --- |"]
    for row in report["judgement"]["rows"]:
        lines.append(f"| {row['row']} | {row['status']} | {row['note']} | `{_numbers_text(row['numbers'])}` |")
    lines += ["", "- 不得只報成立的部分：上表列出全部的列，包括無法判定與不適用（019 §6）。"]
    if report["judgement"]["context"]["c7a_inconsistent"]:
        lines.append("- **C7a 不一致：停止，H1、H2 都不判定，先查明原因（019 §9）。**")

    lines += ["", f"## H1：e* = {part['e_star']} 的配對（ρ_eval = ρ_QPS × D_S / D_F）", "",
              "| seed | 乾淨 | 標記 | ρ_QPS | D_S / D_F | ρ_eval | 用於判定 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for e in part["h1"]["pairs"]:
        lines.append(f"| {e['seed']} | {'是' if e['clean'] else '否'} | {', '.join(e['flags']) or '—'} | {_num(e['rho_qps'], 4)} | "
                     f"{_num(e['d_ratio'], 4)} | {_num(e['rho_eval'], 4)} | {'是' if e['usable'] else '否'} |")
    h1d = part["h1d"]
    lines += ["", f"- H1d（seed 1）：κ = {_num(h1d['kappa'], 4)}、ν = {_num(h1d['nu'], 4)}；f_k = {h1d['f_k']}；"
              f"{h1d['nu_reading'] or 'ν 無法計算'}。" + (f"原因：{'；'.join(h1d['reasons'])}" if h1d["reasons"] else "")]

    lines += ["", f"## H2：e* 的損失分解（有效 seed：{part['h2']['valid_seeds']}）", "",
              "| seed | 有效 | ids 核對 | L_total | L_trav | L_rank | δ_F | thr |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in part["h2"]["seeds"]:
        lines.append(f"| {e['seed']} | {'是' if e['valid'] else '否'} | {'通過' if e['ids_valid'] else '不通過'} | "
                     + " | ".join(_num(_float((e.get("exact") or {}).get(k)), 5) for k in ("L_total", "L_trav", "L_rank", "delta_F", "thr"))
                     + " |")
    lines += ["", "其他 ef（不判定）：", "", "| seed | ef | 有效 seed | L_total | L_trav | L_rank | δ_F |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for e in part["h2"]["other_ef"]:
        lines.append(f"| {e['seed']} | {e['ef_search']} | {'是' if e['valid'] else '否'} | "
                     + " | ".join(_num(e[k], 5) for k in ("L_total", "L_trav", "L_rank", "delta_F")) + " |")

    lines += ["", "## 讀值 1：frontier（010 §2、§3、013 §3）", "",
              "R 只有 seed 1 的一個 build，列為錨點。", "",
              "| 實作 | 目標 | QPS | P50 (µs) | P95 (µs) | M_014 (MiB) | 排除標記點是否改變讀值 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for impl in (F, S, P, Q, R):
        for target, entry in part["frontier"].get(impl, {}).get(str(N), {}).items():
            main = entry["main"]
            lines.append(f"| {impl}{'（錨點）' if impl == R else ''} | {target} | {_reading_cell(main, 'qps', 1, 0)} | "
                         f"{_reading_cell(main, 'p50_ns', 1e3, 1)} | {_reading_cell(main, 'p95_ns', 1e3, 1)} | "
                         f"{_reading_cell(main, 'index_footprint_bytes', 2**20, 1)} | {'是' if entry['differs'] else '否'} |")
    differing = [(impl, target, entry["all_points"]) for impl in (F, S, P, Q, R)
                 for target, entry in part["frontier"].get(impl, {}).get(str(N), {}).items() if entry["differs"]]
    if differing:
        lines += ["", "包含被標記點的讀值（只列出與主要讀值不同者）：", "", "| 實作 | 目標 | QPS | P50 (µs) | P95 (µs) | M_014 (MiB) |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for impl, target, reading in differing:
            lines.append(f"| {impl} | {target} | {_reading_cell(reading, 'qps', 1, 0)} | {_reading_cell(reading, 'p50_ns', 1e3, 1)} | "
                         f"{_reading_cell(reading, 'p95_ns', 1e3, 1)} | {_reading_cell(reading, 'index_footprint_bytes', 2**20, 1)} |")
    lines += ["", "每個 ef 的 recall（019 §6 的指標；跨 seed 的 median [min, max]（build 數），recall 用全部 build）：", "",
              "| 實作 | ef | Recall@10（主要） | tie-aware Recall@10 | Recall@1 |", "| --- | --- | --- | --- | --- |"]
    for e in part["recall_by_ef"]:
        lines.append(f"| {e['implementation']} | {e['ef_search']} | {_span(e['recall10'], 4)} | {_span(e['tie10'], 4)} | "
                     f"{_span(e['recall1'], 4)} |")
    lines += ["", "比值（SQ8 ÷ FP32，兩者的主要讀值都達到目標才計算）：", "", "| 配對 | 目標 | QPS | P50 | P95 | M_014 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for kind in ("rust", "faiss"):
        for target, value in part["frontier_ratios"][kind].items():
            if value == NA or value is None:
                lines.append(f"| {PAIR_NAMES[kind]} | {target} | {'不適用（C7b 不一致）' if value == NA else '—（未達到）'} | | | |")
            else:
                lines.append(f"| {PAIR_NAMES[kind]} | {target} | " + " | ".join(
                    _num(value.get(m), 3) for m in ("qps", "p50_ns", "p95_ns", "index_footprint_bytes")) + " |")
    lines += ["", "quantization frontier 的判讀（015 §1 的 QPS 範圍與 seed 分開規則；事先登記的描述性判讀，不是假設）：", "",
              "| 配對 | 目標 | FP32 QPS 範圍 | SQ8 QPS 範圍 | seed 範圍 | 判讀 | 包含被標記 build 的判讀 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for kind in ("rust", "faiss"):
        reading = part["quant_frontier"][kind]
        if reading == NA:
            lines.append(f"| {PAIR_NAMES[kind]} | — | | | | 不適用（C7b 不一致） | |")
            continue
        fp32, sq8 = PAIRS[kind]
        for target, entry in reading.items():
            main = entry["main"]
            cell = main["cell"]
            cell_text = "—" if cell is None else (f"{cell['faster']['implementation']} {cell['faster'].get('value', cell['faster']['status'])} vs "
                                                 f"{cell['slower']['implementation']} {cell['slower'].get('value', cell['slower']['status'])} → "
                                                 f"{'分開' if cell['separated'] else '未分開'}")
            lines.append(f"| {PAIR_NAMES[kind]} | {target} | {_bracket(main['brackets'][fp32])} | {_bracket(main['brackets'][sq8])} | "
                         f"{cell_text} | {main['verdict']} | {entry['all_builds']['verdict'] + '（不同）' if entry['differs'] else '相同'} |")

    lines += ["", "## 讀值 2：同 (seed, ef) 的配對讀值（ρ = SQ8 ÷ FP32，Δ = FP32 − SQ8；median [min, max]（個數））", "",
              "時間類只用乾淨的配對（括號外），另列包含標記點的版本；recall 用全部 build（019 §6）。"]
    for kind in ("rust", "faiss"):
        block = part["paired"][kind]
        lines += ["", f"### {PAIR_NAMES[kind]}", ""]
        if block.get("status") == NA:
            lines.append(f"不適用：{block['reason']}")
            continue
        if block.get("closed_seeds"):
            lines.append(f"- C8 不是一致而不開放的 seed：{block['closed_seeds']}")
        keys = [k for k in TIME_KEYS if not (kind == "faiss" and k == "rho_eval")]
        lines += ["", "| ef | 乾淨配對 | " + " | ".join(keys) + " | " + " | ".join(f"{k}（含標記）" for k in keys)
                  + " | ΔRecall@10 | ΔRecall@1 | 與 e* 方向不一致（乾淨配對的 median） |",
                  "| --- | --- | " + " | ".join("---" for _ in keys * 2) + " | --- | --- | --- |"]
        for e in block["by_ef"]:
            differing = [k for k in DIRECTION_KEYS if k in e and e[k].get("direction_differs_from_e_star")]
            lines.append(f"| {e['ef_search']} | {e['clean_pairs']}/{e['pairs']} | " + " | ".join(_span(e[k]["main"]) for k in keys) + " | "
                         + " | ".join(_span(e[k]["all"]) for k in keys) + f" | {_span(e['delta_recall10'], 4)} | {_span(e['delta_recall1'], 4)} | "
                         + (("**" + "、".join(differing) + "**") if differing else "—") + " |")
        lines.append(f"- 019 §2：其他 ef 的 ρ 全部列出，不判定；ρ 與 e* = {part['e_star']} 分屬 1 的兩側時，在最後一欄照實標明。")
        lines += ["", "| ΔM（MiB） | 排除 memory / swap 標記 | 全部 build |", "| --- | --- | --- |"]
        for key in MEMORY_KEYS:
            lines.append(f"| {key} | {_span(block['memory_summary'][key]['main'], 2, 2**20)} | {_span(block['memory_summary'][key]['all'], 2, 2**20)} |")

    lines += ["", "## 記憶體（每個 build；M_014 為主要指標，其餘為診斷）", "",
              "| 實作 | seed | M_014 (MiB) | M_relief（診斷） | M_struct（診斷） | rss 增量（輔助） | serialized | build (s) | peak RSS (MiB) | 標記 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for b in part["builds"]:
        lines.append(f"| {b['implementation']} | {b['seed']} | {_num(b['footprint'], 1, 2**20)} | {_num(b['relieved'], 1, 2**20)} | "
                     f"{_num(b['struct'], 1, 2**20)} | {_num(b['rss'], 1, 2**20)} | {_num(b['serialized'], 1, 2**20)} | "
                     f"{_num(b['build_ns'], 2, 1e9)} | {_num(b['peak_rss'], 1, 2**20)} | {', '.join(b['flags']) or '—'} |")
    lines.append("- M_relief 在所有計時點之後才量，可能含有搜尋期間的少量殘留，時點也與 M_014 不同（019 §6）。")
    lines.append("- M_struct 取自 §7 的診斷，不是 formal run 的量測：F、S 是 `build_pair` 由 clone 轉換的 index "
                 "（`memory_accounting`；clone 把鄰居清單的 capacity 收緊成長度，絕對值小於 formal 的 index），"
                 "P、Q 是 FAISS 的陣列大小（storage codes、`hnsw.neighbors`、`offsets`、`levels`，Q 另含 codebook）；R 沒有這個診斷。")

    lines += ["", "## profiling（019 §8；分類規則在取樣之前固定）", "",
              "| 實作 | ef | 角色 | check | QPS | D | kernel | traversal | visited set | heap | 記憶體配置 | 其他 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in part["profiles"]:
        shares = e["shares"] or {}
        lines.append(f"| {e['implementation']} | {e['ef_search']} | {e['role']} | {'通過' if e['ok'] else '不通過'} | {_num(e['qps'], 0)} | "
                     f"{_num(e['d'], 1)} | " + " | ".join(_pct(shares.get(c)) for c in
                                                          ("kernel", "traversal", "visited_set", "heap", "allocation", "other")) + " |")
    for e in part["profiles"]:
        if e["per_query_us"]:
            lines.append(f"- {e['implementation']} ef {e['ef_search']} 每 query（µs）：" + "、".join(
                f"{c} {v:.2f}" for c, v in e["per_query_us"].items())
                         + ("；每次評估（ns）：" + "、".join(f"{c} {v:.2f}" for c, v in e["per_eval_ns"].items()) if e["per_eval_ns"] else ""))
        if e["other_top"]:
            lines.append(f"- {e['implementation']} ef {e['ef_search']}：「其他」超過 5%，前 10 名：" + "；".join(
                f"{o['name']} [{o['image']}] {o['share']:.1%}" for o in e["other_top"]))
        if e["problems"]:
            lines.append(f"- {e['implementation']} ef {e['ef_search']} 的 check 問題：{e['problems']}")

    checks = part["checks"]
    lines += ["", "## 一致性檢查的細節", ""]
    for kind in ("rust", "faiss"):
        c1 = checks["C1"][kind]
        lines.append(f"- C1 {PAIR_NAMES[kind]}：Δ_theory = {c1['theory_bytes']} B，容許 {c1['tolerance_bytes']:.0f} B；"
                     + "；".join(f"seed {e['seed']} ΔM_relief {e.get('delta_relief')}、ΔM_014 {e.get('delta_m014')}、"
                                f"allocator 保留 {e.get('retention')}{'（排除）' if e.get('excluded') else ''}" for e in c1["per_seed"]))
    lines.append(f"- C6（只描述）：" + "；".join(f"ef {r['ef_search']} {r['ratio']:.3f}" for r in checks["C6"]["ratios"]))
    for key in ("C7a", "C7b"):
        lines.append(f"- {key}：逐位元相同 {checks[key]['identical']} 個；其他：{checks[key]['comparisons'] or '（無）'}")
    lines.append(f"- C8：" + "；".join(f"seed {e['seed']} {e['status']}{'（' + e['reason'] + '）' if e.get('reason') else ''}"
                                       f"（graph 相同 {e['graph_identical']}、e* ids 相符 "
                                       f"{e['ids_match_at_e_star']}、codes 一致比例 {e['codes_agreement']}、codebook 相同 "
                                       f"{e['codebook_identical']}）" for e in checks["C8"]["per_seed"]))
    lines.append(f"- 天花板（不計時的分析基準）：{part['ceiling']}；codes：{part['codes']}")

    lines += ["", "## 附帶預期（不作判定，019 §2）", ""]
    lines += [f"- {e['expectation']}：觀察 {e['observed']} → {'符合' if e['met'] else '不符合' if e['met'] is False else '無法判斷'}"
              for e in report["judgement"]["expectations"]]
    if figures:
        lines += ["", "## 圖", "", "排除被標記的點（與主要讀值相同）；數字以上面的表為準。", ""] + [f"- `{name}`" for name in figures]
    lines += ["", "## 合併的 run", ""] + [f"- {label}" for label in part["runs"]]
    lines += ["", "## Core 對照的 run（C7、C6）", ""] + [f"- {d}" for d in report.get("core_runs", [])]
    lines += ["", "## 被排除的 run", ""]
    lines += [f"- {json.dumps(item, ensure_ascii=False)}" for item in report.get("excluded_runs", [])] or ["（無）"]
    return "\n".join(lines) + "\n"


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def _bracket(bracket: dict) -> str:
    if not bracket["reached"]:
        return "未達到"
    return f"[{bracket['lower']:.0f}, {'上界未知' if bracket['upper'] is None else format(bracket['upper'], '.0f')}]"


def plots(report: dict, dataset: str, out_dir: Path) -> list[Path]:
    """019 §6、§8 的圖：recall–QPS（4 個臂，圖例附 M_014）、recall vs ef、每個目標的 quantization frontier
    （x = M_014；另一版 x = M_relief，標為診斷），以及 F 對 S 的每 query 時間分解。只呈現形狀，數字以表為準。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    part = report["datasets"][dataset]
    clean = part["clean_aggregates"]
    title = f"{dataset} (Advanced C, decision 019)"
    written = []
    memory = {impl: _summary([b["footprint"] for b in part["builds"] if b["implementation"] == impl and not b["flags"]])
              for impl in (*ARMS, R)}
    relief = {impl: _summary([b["relieved"] for b in part["builds"] if b["implementation"] == impl
                              and not any(f in MEMORY_EXCLUDING for f in b["flags"])]) for impl in (*ARMS, R)}

    def series(impl, metric):
        points = sorted((a for a in clean if a["implementation"] == impl and a[metric] is not None and a["recall10"]),
                        key=lambda a: a["ef_search"])
        return points

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for impl in ARMS:
        points = series(impl, "qps")
        if points:
            mem = memory[impl]
            label = f"{impl} (M_014 {mem['median'] / 2**20:.1f} MiB)" if mem else impl
            _impl_line(ax, impl, [a["recall10"]["median"] for a in points], [a["qps"]["median"] for a in points])
            ax.lines[-1].set_label(label)
    if ax.lines:
        ax.set_yscale("log")
        _plain_log_ticks(ax.yaxis)
        for target in TARGETS:
            ax.axvline(target, color="#b4b2aa", linestyle=":", linewidth=1)
        ax.set_xlabel("Recall@10 (median over seeds)", fontsize=8, color="#52514e")
        ax.set_ylabel("QPS (median over seeds)", fontsize=8, color="#52514e")
        _style_axes(ax)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"{title}: recall–QPS, flagged points excluded", fontsize=9)
        written.append(_save(fig, out_dir / "recall_qps.png", plt))
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for impl in ARMS:
        points = sorted((a for a in part["aggregates"] if a["implementation"] == impl and a["recall10"]), key=lambda a: a["ef_search"])
        if points:
            style = IMPL_STYLE.get(impl, {"color": "#52514e", "marker": "o"})
            med = np.array([a["recall10"]["median"] for a in points])
            ax.errorbar([a["ef_search"] for a in points], med,
                        yerr=[med - np.array([a["recall10"]["min"] for a in points]),
                              np.array([a["recall10"]["max"] for a in points]) - med],
                        color=style["color"], marker=style["marker"], capsize=3, linewidth=2, markersize=5, label=impl)
    if ax.lines:
        ax.set_xscale("log")
        _plain_log_ticks(ax.xaxis)
        ax.set_xlabel("efSearch", fontsize=8, color="#52514e")
        ax.set_ylabel("Recall@10 (median over seeds, min–max)", fontsize=8, color="#52514e")
        _style_axes(ax)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"{title}: recall vs ef", fontsize=9)
        written.append(_save(fig, out_dir / "recall_vs_ef.png", plt))
    else:
        plt.close(fig)

    for target in TARGETS:
        key = _target_key(target)
        for suffix, x_source, x_label in (("", memory, "index memory M_014, phys_footprint (MiB)"),
                                          ("_relief", relief, "M_relief, diagnostic only (MiB)")):
            fig, ax = plt.subplots(figsize=(6, 4.2))
            drawn = []
            for impl in (*ARMS, R):
                reading = part["frontier"].get(impl, {}).get(str(N), {}).get(key, {}).get("main", {})
                if not reading.get("reached") or not reading.get("qps") or not x_source.get(impl):
                    continue
                x, y = x_source[impl]["median"] / 2**20, reading["qps"]["value"]
                style = IMPL_STYLE.get(impl, {"color": "#52514e", "marker": "o"})
                ax.scatter([x], [y], color=style["color"], marker=style["marker"], s=60, edgecolors="#fcfcfb", linewidths=1,
                           label=impl, zorder=3)
                # R 與 F 的讀值依構造幾乎重合：R 的標籤放在點的下方，避免文字疊在一起。
                ax.annotate(impl, (x, y), xytext=(5, -10 if impl == R else 3), textcoords="offset points", fontsize=7,
                            color="#52514e")
                drawn.append((impl, x, y))
            if not drawn:
                plt.close(fig)
                continue
            located = {impl: (x, y) for impl, x, y in drawn}
            for fp32, sq8 in PAIRS.values():
                if fp32 in located and sq8 in located:
                    ax.annotate("", xy=located[sq8], xytext=located[fp32],
                                arrowprops={"arrowstyle": "->", "color": "#b4b2aa", "linewidth": 1})
            ax.set_xlabel(x_label, fontsize=8, color="#52514e")
            ax.set_ylabel(f"QPS at Recall@10 ≥ {key} (frontier reading)", fontsize=8, color="#52514e")
            _style_axes(ax)
            ax.legend(fontsize=7, frameon=False)
            ax.set_title(f"{title}: quantization frontier @ {key}" + (" (diagnostic x)" if suffix else ""), fontsize=9)
            written.append(_save(fig, out_dir / f"quant_frontier_{key}{suffix}.png", plt))

    bars = [e for e in part["profiles"] if e["implementation"] in (F, S) and e["ef_search"] == part["e_star"] and e["per_query_us"]]
    if bars:
        categories = ("kernel", "traversal", "visited_set", "heap", "allocation", "other")
        colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#b4b2aa")
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for i, e in enumerate(bars):
            left = 0.0
            for category, color in zip(categories, colors):
                value = e["per_query_us"].get(category, 0.0)
                ax.barh(i, value, left=left, color=color, edgecolor="#fcfcfb", linewidth=2,
                        label=category if i == 0 else None)
                left += value
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels([e["implementation"] for e in bars], fontsize=8)
        ax.set_xlabel(f"time per query at e* = {part['e_star']} (µs; share × 1 / QPS)", fontsize=8, color="#52514e")
        _style_axes(ax)
        ax.legend(fontsize=7, frameon=False, ncols=6, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        ax.set_title(f"{title}: per-query time, F vs S (seed 1)", fontsize=9)
        written.append(_save(fig, out_dir / "profile_breakdown.png", plt))
    return written


def _save(fig, path: Path, plt) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="10 份 C formal config 的 run 目錄（兩個資料集）")
    parser.add_argument("--core-runs", nargs="+", required=True,
                        help="019 §2 登記的 Core 正式 run 目錄（C7a、C7b、C6）；全部都要列出")
    parser.add_argument("--excluded", action="append", default=[], help="已整份重跑的失敗 run：只列出狀態與原因，資料不合併")
    parser.add_argument("--diag", nargs="+", help="quant_diag 的輸出；預設為 results/quant/diag-<glove|text>.json")
    parser.add_argument("--premise", help="quant_profile premise 的輸出；預設為 results/quant/premise.json")
    parser.add_argument("--profiles", help="profiling 輸出目錄；預設為 results/profiles")
    parser.add_argument("--out", help="輸出的根目錄；預設為 results/reports（寫到 quant-<glove|text>/）")
    args = parser.parse_args(argv)
    for out in write_quant_report(args.runs, args.core_runs, args.out, excluded=args.excluded, diag_paths=args.diag,
                                  premise_path=args.premise, profile_dir=args.profiles):
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
