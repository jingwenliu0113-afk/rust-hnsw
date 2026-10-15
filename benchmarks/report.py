"""從 runner 的 run.json 產生報告（decision 010 的比較規則）。

用法：
    .venv/bin/python -m benchmarks.report <run 目錄> [<run 目錄> ...] [--out <輸出目錄>]

- 主要比較：每個（實作、n）以所有 (M, efC, ef) 點組成 frontier，每個點取跨 seed 的 median（只用原始插入順序）；
  對每個目標 recall，在 median Recall@10 ≥ 目標的點中取 QPS 最高、P50 / P95 最低、index memory 最低。不內插、不外推。
- 有 hygiene 標記的點不進入主要讀值；若排除與否會改變讀值，兩者都保留（010 §3）。
- index memory 的主要指標是 phys_footprint 的增量（run.json 的 `index_footprint_bytes`）；rss 的增量（`index_memory_bytes`）
  列為輔助指標（decision 014：macOS 上已釋放的大型配置仍算在 rss）。三個實作用完全相同的測法。
- smoke 的報告只用來驗證工具本身，不得用於任何結論。
- 合併多個 run 之前（013 §5）：只接受 complete / incomplete 的 run；同一個 build 不得出現在多個 run；
  各 run 的 hnswlib / faiss 擴充模組、套件版本、`rust_hnsw.BUILD_INFO` 必須相同，commit 不同時實作與量測程式不得有差異。
  aborted / invalid / error 的 run 整份重跑，原本的 run 只能以 `--excluded` 列出（不合併資料）。
  因環境變化而作廢的 complete run 必須登記在 `VOIDED_RUNS`（decision 016），同樣只能以 `--excluded` 列出。
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGETS = (0.90, 0.95, 0.99)
# 008 §9 的 hygiene 標記；子 process、build、點三層的標記都合併到點上（010 §3）。
HYGIENE_FLAGS = ("load", "thermal", "swap", "multithread", "nondeterministic", "power", "memory", "background")
BASELINE = "rust-hnsw"
ORDER_NOTE = {
    "faiss-hnsw": "FAISS 的實際插入順序不是 row 順序（007 §5）：打亂 row 只改變 id 與 level 的對應，意義與另外兩者不同，不並列比較。",
}
METRICS = ("recall10", "tie10", "recall1", "tie1", "qps", "p50_ns", "p95_ns", "index_memory_bytes",
           "index_footprint_bytes", "build_ns", "peak_rss_bytes")
BUILD_METRICS = ("build_ns", "peak_rss_bytes", "index_memory_bytes", "index_footprint_bytes")
# 013 §5：可以進入報告的 run 狀態；失敗的 run 整份重跑，只能以 --excluded 列出。
REPORTABLE_STATUS = ("complete", "incomplete")
FAILED_STATUS = ("aborted", "invalid", "error")
# decision 016、017：因環境變化或協定修訂（不是結果）而作廢的 complete run，以 run 目錄名稱登記。
# 與失敗的 run 一樣只能以 --excluded 列出，資料不合併；沒有登記的 complete run 仍然不得排除（010 §3）。
VOIDED_RUNS = {
    "20261010T183115Z-formal": "decision 016: the AC power mode was changed from high performance (2) to automatic (0) outside this "
                               "project between text-formal-1 and this run; the text workload stays in high-performance mode",
    "20261010T195251Z-formal": "decision 016: run in automatic power mode (0) after the same outside change; the text workload stays "
                               "in high-performance mode",
    "20261010T112849Z-formal": "decision 017: run under protocol 008-v2 (battery-drop rule); the user removed that rule, so the text "
                               "workload is rerun under the later protocol (008-v4, decision 018) and a report cannot merge versions",
}
REPO = Path(__file__).resolve().parents[1]
# 實作與量測程式：合併的 run 之間（不同 commit 時）這些路徑不得有差異（013 §5）。
MEASUREMENT_PATHS = (
    "crates", "python", "Cargo.toml", "Cargo.lock", "requirements-dev.txt",
    *(f"benchmarks/{name}" for name in ("run.py", "run_child.py", "adapters.py", "interface.py", "protocol.py",
                                        "evaluate.py", "metrics.py", "environment.py", "datasets.py", "config.py")),
)
# 輔助 scaling 讀值（013 §4）：每個 n 只用這一格 (M, efC) 的點讀 frontier，只用來呈現格點數不同造成的混淆。
AUX_SCALING_CELL = (16, 200)
AUX_SCALING_METRICS = ("qps", "p50_ns", "p95_ns")
# 013 §5：每個資料集的全部 formal config；formal 報告必須恰好包含每一份各一個 run。
FORMAL_CONFIGS = {
    "glove-100-angular": tuple(f"benchmarks/configs/{name}.toml" for name in (
        "glove-formal-1-m16-order", "glove-formal-2-m16-seeds-efc400", "glove-formal-3-m48-seeds-1-2",
        "glove-formal-4-m48-seeds-3-4", "glove-formal-5-m48-seed-5-efc100", "glove-formal-6-m32-300k",
        "glove-formal-7-m8", "glove-formal-8-full")),
    "simplewiki-embeddinggemma2-100k": tuple(f"benchmarks/configs/{name}.toml" for name in (
        "text-formal-1-m16-order", "text-formal-2-m16-seeds-efc400", "text-formal-3-m8-efc100",
        "text-formal-4-highm-scaling")),
}
RUNS_ROOT = REPO / "results" / "runs"
# 013 §3：主要結論附上的 variance 格點（主要 n 上的 (M, efC)），依資料集與目標。
VARIANCE_N = 100_000
VARIANCE_CELLS = {
    "glove-100-angular": {"0.90": ((16, 200), (48, 200)), "0.95": ((16, 200), (48, 200)), "0.99": ((48, 200),)},
    "simplewiki-embeddinggemma2-100k": {"0.90": ((16, 200),), "0.95": ((16, 200),), "0.99": ((16, 200),)},
}
VARIANCE_METRICS = ("recall10", "qps", "p50_ns", "p95_ns")
# 015：Gate 證據的建議判讀（是否進 Advanced 由使用者決定）。
GATE_COMPARED = "hnswlib"
GATE_PRIMARY_TARGET = "0.95"
# 015 §1：作為 seed 範圍至少要有的未標記 build 數（variance 格點有 5 個 seed）。
GATE_MIN_BUILDS = 3
# 合併的 run 之間必須相同的資料集紀錄（同名資料集重新準備時，向量或 ground truth 可能不同）。
DATASET_IDENTITY = ("name", "kind", "n_train", "n_queries", "dim", "metric", "files_sha256")


class ReportError(ValueError):
    pass


def _target_key(target: float) -> str:
    return f"{target:.2f}"


def _summary(values: list) -> dict | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return {"median": float(np.median(present)), "min": min(present), "max": max(present), "values": present}


def point_rows(records: list[dict]) -> list[dict]:
    """每個（build、實作、ef）一列；hygiene 標記合併子 process 與點兩層。"""
    rows = []
    for record in records:
        for child in record["children"]:
            build = child["build"]
            for point in child["points"]:
                k10, k1 = point["k10"], point["k1"]
                ok = k10["status"] == "ok"
                flags = sorted({*child["flags"], *build.get("flags", []), *point["flags"]} & set(HYGIENE_FLAGS))
                rows.append({
                    "implementation": child["implementation"],
                    "n": build["n"],
                    "m": build["m"],
                    "ef_construction": build["ef_construction"],
                    "seed": build["seed"],
                    "shuffle": build["shuffle"],
                    "ef_search": point["ef_search"],
                    "status": k10["status"],
                    "incomplete": bool(k10["incomplete"]),
                    "incomplete_queries": k10.get("incomplete_queries"),
                    "recall10": k10["recall"],
                    "tie10": k10["tie_aware_recall"],
                    "recall1": k1.get("recall"),
                    "tie1": k1.get("tie_aware_recall"),
                    "qps": k10["qps"]["qps_median"] if ok else None,
                    "p50_ns": k10["latency"]["p50_ns"] if ok else None,
                    "p95_ns": k10["latency"]["p95_ns"] if ok else None,
                    "index_memory_bytes": build.get("index_memory_bytes"),
                    "index_footprint_bytes": build.get("index_footprint_bytes"),
                    "build_ns": build.get("build_ns"),
                    "peak_rss_bytes": build.get("peak_rss_bytes"),
                    "flags": flags,
                })
    return rows


def _point_key(row) -> tuple:
    return row["implementation"], row["n"], row["m"], row["ef_construction"], row["ef_search"]


def aggregate(rows: list[dict]) -> list[dict]:
    """跨 build 的 median 與 min–max，保留原始值（010 §4）。hnswlib 整批失敗的點沒有數字，不進入彙整。"""
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            groups[_point_key(row)].append(row)
    aggregates = []
    for key, members in groups.items():
        impl, n, m, efc, ef = key
        entry = {"implementation": impl, "n": n, "m": m, "ef_construction": efc, "ef_search": ef,
                 "builds": len(members), "incomplete": any(r["incomplete"] for r in members),
                 "flagged_builds": sum(bool(r["flags"]) for r in members)}
        for metric in METRICS:
            entry[metric] = _summary([r[metric] for r in members])
        aggregates.append(entry)
    return sorted(aggregates, key=lambda a: (a["implementation"], a["n"], a["m"], a["ef_construction"], a["ef_search"]))


def _reading(aggregates: list[dict], target: float) -> dict:
    eligible = [a for a in aggregates if a["recall10"] is not None and a["recall10"]["median"] >= target]
    if not eligible:
        return {"reached": False}

    def pick(metric, best):
        candidates = [a for a in eligible if a[metric] is not None]
        if not candidates:
            return None
        chosen = best(candidates, key=lambda a: a[metric]["median"])
        # 013 §3 第 1 點：讀值附上該點本身參與的 build 數與跨 seed min–max。
        return {"value": chosen[metric]["median"], "min": chosen[metric]["min"], "max": chosen[metric]["max"],
                "m": chosen["m"], "ef_construction": chosen["ef_construction"],
                "ef_search": chosen["ef_search"], "incomplete": chosen["incomplete"], "builds": chosen["builds"]}

    return {"reached": True, "qps": pick("qps", max), "p50_ns": pick("p50_ns", min), "p95_ns": pick("p95_ns", min),
            "index_memory_bytes": pick("index_memory_bytes", min),
            "index_footprint_bytes": pick("index_footprint_bytes", min)}


def _location(pick: dict | None) -> tuple | None:
    return None if pick is None else (pick["value"], pick["m"], pick["ef_construction"], pick["ef_search"], pick["incomplete"])


def frontier(all_aggregates: list[dict], clean_aggregates: list[dict], compared=None) -> dict:
    """`compared`：判斷「排除標記點是否改變讀值」時比較的指標（預設全部讀值）。"""
    result = defaultdict(dict)
    keys = sorted({(a["implementation"], a["n"]) for a in all_aggregates})
    for impl, n in keys:
        everything = [a for a in all_aggregates if (a["implementation"], a["n"]) == (impl, n)]
        clean = [a for a in clean_aggregates if (a["implementation"], a["n"]) == (impl, n)]
        per_target = {}
        for target in TARGETS:
            main, all_points = _reading(clean, target), _reading(everything, target)
            if compared is None:
                differs = main != all_points
            else:
                # 只比讀值本身（數值與所在的點），不比參與的 build 數。
                differs = main.get("reached") != all_points.get("reached") or any(
                    _location(main.get(metric)) != _location(all_points.get(metric)) for metric in compared)
            per_target[_target_key(target)] = {"main": main, "all_points": all_points, "differs": differs}
        result[impl][str(n)] = per_target
    return dict(result)


def ratios(front: dict) -> dict:
    """各實作的 QPS 是 rust-hnsw 的幾倍；兩者都達到目標才計算（010 §2）。"""
    result = defaultdict(dict)
    ns = sorted({n for per_n in front.values() for n in per_n}, key=int)
    for n in ns:
        for target in TARGETS:
            key = _target_key(target)
            base = front.get(BASELINE, {}).get(n, {}).get(key, {}).get("main", {})
            entry = {}
            for impl, per_n in sorted(front.items()):
                if impl == BASELINE or n not in per_n:
                    continue
                other = per_n[n][key]["main"]
                if base.get("reached") and other.get("reached") and base["qps"] and other["qps"]:
                    entry[impl] = other["qps"]["value"] / base["qps"]["value"]
                else:
                    entry[impl] = None
            result[n][key] = entry
    return dict(result)


def _cell_points(aggregates: list[dict], impl: str, n: int, m: int, efc: int) -> list[dict]:
    return sorted((a for a in aggregates if (a["implementation"], a["n"], a["m"], a["ef_construction"]) == (impl, n, m, efc)),
                  key=lambda a: a["ef_search"])


def _first_reaching(points: list[dict], target: float) -> int | None:
    """依 ef 排序的點中，median Recall@10 第一次 ≥ 目標的位置（010 §2）。"""
    return next((i for i, a in enumerate(points) if a["recall10"] is not None and a["recall10"]["median"] >= target), None)


def variance_ranges(aggregates: list[dict], dataset: str, implementations) -> dict:
    """013 §3 第 2 點：每個目標、每個 variance 格點、每個實作，在該實作自己的最小達標 ef 上的跨 seed min–max。

    格點在 ef 清單內沒有達到目標時，改用該格最大的 ef，並以 `same_region = False` 標明與讀值不在同一 recall 區間（013 §3 的 fallback）。
    某實作在格點上沒有任何（未標記的）build 時，列為 `missing`，不靜默略過。
    """
    result = {}
    for target, cells in VARIANCE_CELLS.get(dataset, {}).items():
        entries = []
        for m, efc in cells:
            for impl in implementations:
                base = {"implementation": impl, "n": VARIANCE_N, "m": m, "ef_construction": efc}
                points = _cell_points(aggregates, impl, VARIANCE_N, m, efc)
                if not points:
                    entries.append({**base, "missing": True})
                    continue
                first = _first_reaching(points, float(target))
                chosen = points[-1] if first is None else points[first]
                entries.append({**base, "missing": False, "ef_search": chosen["ef_search"], "same_region": first is not None,
                                "builds": chosen["builds"], "incomplete": chosen["incomplete"],
                                **{metric: chosen[metric] for metric in VARIANCE_METRICS}})
        result[target] = entries
    return result


def qps_bracket(aggregates: list[dict], impl: str, n: int, target: float) -> dict:
    """015 §1：不內插，在已量測的 (M, efC) 格點中、recall 恰好達到目標時可得的最高 QPS 的範圍 [lower, upper]。

    每個格點裡，recall 第一次達到目標的 ef 與它前一個 ef 夾住真正的交點；QPS 隨 ef 遞減，所以前一個 ef 的 QPS 是該格的上界。
    - lower：所有達標點中最高的 QPS，等於 frontier 的 QPS 讀值（010 §2）。
    - upper：各格上界的最大值。格點的第一個 ef 就已達標時上界未知（upper = None）；
      格點在 ef 清單內沒有達標時，交點在最大 ef 之後，以最大 ef 的 QPS 為上界。
    假設 recall 不會在相鄰兩個量測的 ef 之間先超過目標再掉回去。
    """
    cells = defaultdict(list)
    for a in aggregates:
        if (a["implementation"], a["n"]) == (impl, n) and a["qps"] is not None:
            cells[(a["m"], a["ef_construction"])].append(a)
    lower, uppers, unbounded = None, [], []
    for (m, efc), points in sorted(cells.items()):
        points.sort(key=lambda a: a["ef_search"])
        first = _first_reaching(points, target)
        if first is None:
            uppers.append(points[-1]["qps"]["median"])
            continue
        best = max(a["qps"]["median"] for a in points[first:] if a["recall10"]["median"] >= target)
        lower = best if lower is None else max(lower, best)
        if first == 0:
            unbounded.append([m, efc])
        else:
            uppers.append(points[first - 1]["qps"]["median"])
    if lower is None:
        return {"reached": False}
    return {"reached": True, "lower": lower, "upper": None if unbounded else max([lower, *uppers]),
            "unbounded_cells": unbounded}


def _cell_side(aggregates: list[dict], impl: str, m: int, efc: int, target: float, side: str) -> dict:
    """variance 格點上，一個實作在目標交點附近的 seed 範圍端點（015 §1）。

    - side = "lower"（較快的一方）：第一個達標 ef 上最慢的 seed；
    - side = "upper"（較慢的一方）：前一個 ef 上最快的 seed；格點內沒有達標時用最大 ef；第一個 ef 就達標時上界未知。
    """
    points = _cell_points(aggregates, impl, VARIANCE_N, m, efc)
    if not points:
        return {"status": "missing"}
    first = _first_reaching(points, target)
    if side == "lower":
        if first is None:
            return {"status": "not_reached"}
        point, value = points[first], points[first]["qps"]["min"]
    else:
        if first == 0:
            return {"status": "unbounded"}
        point = points[-1] if first is None else points[first - 1]
        value = point["qps"]["max"]
    entry = {"ef_search": point["ef_search"], "builds": point["builds"], "value": value}
    if point["builds"] < GATE_MIN_BUILDS:
        return {"status": "few_builds", **entry}
    return {"status": "ok", **entry}


def _gate_target(aggregates: list[dict], dataset: str, target: str) -> dict:
    t = float(target)
    rust = qps_bracket(aggregates, BASELINE, VARIANCE_N, t)
    other = qps_bracket(aggregates, GATE_COMPARED, VARIANCE_N, t)
    entry = {"brackets": {BASELINE: rust, GATE_COMPARED: other}, "direction": None, "cells": []}
    if not (rust["reached"] and other["reached"]):
        return {**entry, "verdict": "undetermined"}
    if rust["upper"] is not None and other["lower"] > rust["upper"]:
        direction, faster, slower = "compared_faster", GATE_COMPARED, BASELINE
    elif other["upper"] is not None and rust["lower"] > other["upper"]:
        direction, faster, slower = "rust_faster", BASELINE, GATE_COMPARED
    else:
        return {**entry, "verdict": "within_quantization"}
    cells = []
    for m, efc in VARIANCE_CELLS[dataset][target]:
        fast = _cell_side(aggregates, faster, m, efc, t, "lower")
        slow = _cell_side(aggregates, slower, m, efc, t, "upper")
        cells.append({"m": m, "ef_construction": efc, "faster": {"implementation": faster, **fast},
                      "slower": {"implementation": slower, **slow},
                      "separated": fast["status"] == slow["status"] == "ok" and fast["value"] > slow["value"]})
    if not all(c["separated"] for c in cells):
        verdict = "uncertain"
    else:
        verdict = "gap" if direction == "compared_faster" else "rust_faster"
    return {**entry, "direction": direction, "cells": cells, "verdict": verdict}


def gate_reading(clean_aggregates: list[dict], all_aggregates: list[dict], dataset: str) -> dict:
    """015 §1：主要 n 上 hnswlib 與 rust-hnsw 的差距（建議判讀；是否進 Advanced 由使用者決定）。

    - `gap`：hnswlib 的 QPS 下界 > rust-hnsw 的上界（qps_bracket），而且每個附上的 variance 格點上，
      hnswlib 在交點之後最慢的 seed 仍快於 rust-hnsw 在交點之前最快的 seed（各至少 GATE_MIN_BUILDS 個 build）；
    - `rust_faster`：方向相反、條件相同；
    - `within_quantization`：兩者的 QPS 範圍重疊，ef 梯子的量化之內無法區分；
    - `uncertain`：範圍分開，但某個附上的 variance 格點的 seed 範圍沒有分開、缺資料或 build 不足；
    - `undetermined`：任一方沒有達到目標。
    主要判讀排除有 hygiene 標記的 build（010 §2）；另以全部 build 再判讀一次，不同時兩者並列（010 §3）。
    """
    result = {}
    for target in VARIANCE_CELLS.get(dataset, {}):
        main = _gate_target(clean_aggregates, dataset, target)
        everything = _gate_target(all_aggregates, dataset, target)
        result[target] = {"main": main, "all_builds": everything, "differs": main["verdict"] != everything["verdict"],
                          "primary": target == GATE_PRIMARY_TARGET}
    return result


def pareto_front(aggregates: list[dict], metric: str, higher_is_better: bool) -> list[dict]:
    """recall–指標的 frontier：沒有其他點的 median Recall@10 不低於它、指標又更好（只用來畫圖，讀值仍依 010 §2）。

    回傳依 Recall@10 由低到高排序的點。指標或 recall 缺值的點不參與。
    """
    points = [a for a in aggregates if a["recall10"] is not None and a.get(metric) is not None]
    points.sort(key=lambda a: (-a["recall10"]["median"], -a[metric]["median"] if higher_is_better else a[metric]["median"]))
    front, best = [], None
    for a in points:
        value = a[metric]["median"]
        if best is None or (value > best if higher_is_better else value < best):
            front.append(a)
            best = value
    return front[::-1]


def recall_vs_ef(aggregates: list[dict]) -> list[dict]:
    """010 §6：最大 ef 的 median Recall@10 低於最小 ef 時為系統性惡化；另報最大的相鄰下降。"""
    groups = defaultdict(list)
    for a in aggregates:
        groups[(a["implementation"], a["n"], a["m"], a["ef_construction"])].append(a)
    checks = []
    for (impl, n, m, efc), members in sorted(groups.items()):
        members.sort(key=lambda a: a["ef_search"])
        recalls = [a["recall10"]["median"] for a in members]
        drops = [max(0.0, earlier - later) for earlier, later in zip(recalls, recalls[1:])]
        checks.append({"implementation": impl, "n": n, "m": m, "ef_construction": efc,
                       "ef_search": [a["ef_search"] for a in members], "recall10_median": recalls,
                       "systematic_deterioration": recalls[-1] < recalls[0],
                       "max_adjacent_drop": max(drops, default=0.0)})
    return checks


def order_variance(rows: list[dict]) -> list[dict]:
    """order variance：shuffle 所用的 seed 搭配原始順序與每個 shuffle（010 §4）。"""
    shuffled = [r for r in rows if r["shuffle"] is not None and r["status"] == "ok"]
    groups = defaultdict(list)
    for row in shuffled:
        groups[(*_point_key(row), row["seed"])].append(row)
    entries = []
    for (impl, n, m, efc, ef, seed), members in sorted(groups.items(), key=lambda item: item[0]):
        original = [r for r in rows if r["status"] == "ok" and r["shuffle"] is None and r["seed"] == seed
                    and _point_key(r) == (impl, n, m, efc, ef)]
        members = original + sorted(members, key=lambda r: r["shuffle"])
        entries.append({"implementation": impl, "n": n, "m": m, "ef_construction": efc, "ef_search": ef, "seed": seed,
                        "shuffles": [r["shuffle"] for r in members],
                        "recall10": _summary([r["recall10"] for r in members]),
                        "qps": _summary([r["qps"] for r in members]),
                        "note": ORDER_NOTE.get(impl, "")})
    return entries


def build_rows(records: list[dict]) -> list[dict]:
    """每個（build、實作）一列，原始插入順序；標記取子 process 與 build 兩層（點層級的標記與 build 量測無關）。"""
    rows = []
    for record in records:
        for child in record["children"]:
            build = child["build"]
            if build["shuffle"] is not None or build.get("build_ns") is None:
                continue
            rows.append({**{k: build[k] for k in ("n", "m", "ef_construction", "seed")},
                         "implementation": child["implementation"],
                         **{metric: build.get(metric) for metric in BUILD_METRICS},
                         "flags": sorted({*child["flags"], *build.get("flags", [])} & set(HYGIENE_FLAGS))})
    return rows


def scaling(builds: list[dict]) -> list[dict]:
    """固定 (M, efC) 下，每個 n 的 build time、peak RSS、index memory（010 §5）。

    main 排除有 hygiene 標記的 build（010 §3）；all_builds 包含全部；兩者不同時 differs 為 True。
    """
    groups = defaultdict(list)
    for row in builds:
        groups[(row["implementation"], row["n"], row["m"], row["ef_construction"])].append(row)
    entries = []
    for (impl, n, m, efc), members in sorted(groups.items()):
        clean = [r for r in members if not r["flags"]]
        main = {metric: _summary([r[metric] for r in clean]) for metric in BUILD_METRICS}
        everything = {metric: _summary([r[metric] for r in members]) for metric in BUILD_METRICS}
        entries.append({"implementation": impl, "n": n, "m": m, "ef_construction": efc, **main,
                        "builds": len(clean), "all_builds": everything, "flagged_builds": len(members) - len(clean),
                        "differs": main != everything})
    return entries


def _check_compatible(records: list[dict]) -> None:
    if not records:
        raise ReportError("no runs given")
    for field, getter in (("mode", lambda r: r["mode"]), ("dataset", lambda r: r["dataset"]["name"]),
                          ("protocol", lambda r: r["protocol"]["version"])):
        values = {getter(r) for r in records}
        if len(values) > 1:
            raise ReportError(f"runs differ in {field}: {sorted(values)}")


def changed_measurement_files(commit_a: str, commit_b: str, repo=REPO) -> list[str]:
    """兩個 commit 之間，實作與量測程式（MEASUREMENT_PATHS）有差異的檔案。"""
    try:
        out = subprocess.run(["git", "diff", "--name-only", commit_a, commit_b, "--", *MEASUREMENT_PATHS],
                             cwd=repo, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        raise ReportError(f"cannot compare commits {commit_a} and {commit_b}: {err}") from err
    return out.split()


def _check_mergeable(records: list[dict], labels: list[str], changed_files) -> None:
    """013 §5 的合併前檢查：run 狀態、跨 run 重複的 build、版本一致。"""
    for label, record in zip(labels, records):
        if record["status"] not in REPORTABLE_STATUS:
            raise ReportError(f"{label} has status {record['status']!r}; rerun the whole config and pass the failed "
                              "run with --excluded (013 §5)")
    seen = {}
    for label, record in zip(labels, records):
        for child in record["children"]:
            build = child["build"]
            key = (child["implementation"], build["n"], build["m"], build["ef_construction"], build["seed"],
                   build["shuffle"])
            if key in seen:
                raise ReportError(f"build {key} appears in both {seen[key]} and {label}")
            seen[key] = label
    if len(records) < 2:
        return
    for label, record in zip(labels[1:], records[1:]):
        for field in DATASET_IDENTITY:
            if record["dataset"].get(field) != records[0]["dataset"].get(field):
                raise ReportError(f"{label} differs from {labels[0]} in dataset {field}")
        for n, subset in record.get("subsets", {}).items():
            other = records[0].get("subsets", {}).get(n)
            if other is not None and subset.get("gt_source") != other.get("gt_source"):
                raise ReportError(f"{label} differs from {labels[0]} in the ground truth of n = {n}")
    first = records[0]["metadata"]
    for label, record in zip(labels[1:], records[1:]):
        meta = record["metadata"]
        for field, getter in (("the hnswlib extension", lambda m: m["extension_sha256"]["hnswlib"]),
                              ("the faiss extension", lambda m: m["extension_sha256"]["faiss"]),
                              ("package versions", lambda m: m["versions"]),
                              ("rust_hnsw.BUILD_INFO", lambda m: m["build_info"]),
                              # 014 補記：允許自動或高效能電力模式，但一份報告只合併同一個模式的 run。
                              ("the power mode", lambda m: ((m.get("formal_checks") or {}).get("power") or {}).get("powermode")),
                              # 018：70W 與 140W 都允許，但一份報告只合併同一個變壓器的 run。
                              ("the power adapter", lambda m: ((m.get("formal_checks") or {}).get("power") or {}).get("adapter"))):
            if getter(meta) != getter(first):
                raise ReportError(f"{label} differs from {labels[0]} in {field}")
        commit_a, commit_b = first["git"]["commit"], meta["git"]["commit"]
        if commit_a != commit_b:
            changed = changed_measurement_files(commit_a, commit_b) if changed_files is None else \
                changed_files(commit_a, commit_b)
            if changed:
                raise ReportError(f"{label} (commit {commit_b[:12]}) and {labels[0]} (commit {commit_a[:12]}) differ in "
                                  f"implementation or measurement code: {changed}")


def build_report(records: list[dict], labels: list[str] | None = None, changed_files=None) -> dict:
    """`labels` 用於錯誤訊息（預設 run #1…）；`changed_files(a, b)` 預設以 git 比較兩個 commit（測試可替換）。"""
    _check_compatible(records)
    labels = [f"run #{i + 1}" for i in range(len(records))] if labels is None else list(labels)
    _check_mergeable(records, labels, changed_files)
    rows = point_rows(records)
    original = [r for r in rows if r["shuffle"] is None]
    all_aggregates = aggregate(original)
    clean_aggregates = aggregate([r for r in original if not r["flags"]])
    front = frontier(all_aggregates, clean_aggregates)
    dataset = records[0]["dataset"]["name"]
    implementations = sorted({a["implementation"] for a in all_aggregates})
    in_cell = [a for a in all_aggregates if (a["m"], a["ef_construction"]) == AUX_SCALING_CELL]
    clean_in_cell = [a for a in clean_aggregates if (a["m"], a["ef_construction"]) == AUX_SCALING_CELL]
    failed = [
        {"implementation": c["implementation"], **{k: c["build"][k] for k in ("n", "m", "ef_construction", "seed", "shuffle")},
         "status": c["status"], "partial": c["partial"]}
        for record in records for c in record["children"] if c["status"] != "ok"
    ]
    return {
        "mode": records[0]["mode"],
        "dataset": records[0]["dataset"]["name"],
        "protocol": records[0]["protocol"]["version"],
        "run_status": [r["status"] for r in records],
        "targets": [_target_key(t) for t in TARGETS],
        "frontier": front,
        "aux_scaling_cell": list(AUX_SCALING_CELL),
        "aux_scaling_frontier": frontier(in_cell, clean_in_cell, compared=AUX_SCALING_METRICS),
        "excluded_runs": [],
        "ratios": ratios(front),
        "variance_ranges": variance_ranges(clean_aggregates, dataset, implementations),
        "variance_ranges_all_builds": variance_ranges(all_aggregates, dataset, implementations),
        "gate": gate_reading(clean_aggregates, all_aggregates, dataset),
        "seed_variance": all_aggregates,
        "clean_aggregates": clean_aggregates,
        "order_variance": order_variance(rows),
        "scaling": scaling(build_rows(records)),
        "recall_vs_ef": recall_vs_ef(all_aggregates),
        "incomplete_points": [
            {k: r[k] for k in ("implementation", "n", "m", "ef_construction", "seed", "shuffle", "ef_search", "status",
                               "incomplete_queries")}
            for r in rows if r["status"] != "ok" or r["incomplete"]
        ],
        "flagged_points": [
            {k: r[k] for k in ("implementation", "n", "m", "ef_construction", "seed", "shuffle", "ef_search", "flags")}
            for r in rows if r["flags"]
        ],
        "failed_children": failed,
    }


# ---- 輸出 ----


def _fmt(value, scale=1.0, digits=3) -> str:
    return "—" if value is None else f"{value / scale:.{digits}f}"


def _reading_cell(reading: dict | None, metric: str, scale: float, digits: int) -> str:
    if not reading or not reading.get("reached"):
        return "未達到"
    item = reading[metric]
    if item is None:
        return "—"
    mark = "（不足 k）" if item["incomplete"] else ""
    # 013 §3 第 1 點：多個 build 時附上跨 seed 的 min–max 與 build 數。
    spread = "" if item["builds"] < 2 else \
        f" [{item['min'] / scale:.{digits}f}, {item['max'] / scale:.{digits}f}]（{item['builds']} builds）"
    return (f"{item['value'] / scale:.{digits}f}{spread} @ M={item['m']}, efC={item['ef_construction']}, "
            f"ef={item['ef_search']}{mark}")


GATE_VERDICTS = {"gap": f"有差距（{GATE_COMPARED} 較快）", "rust_faster": "rust-hnsw 較快",
                 "within_quantization": "在 ef 梯子的量化範圍內，無法區分", "uncertain": "不確定（seed 範圍未分開或資料不足）",
                 "undetermined": "無法判定（有一方未達到目標）"}
GATE_CELL_STATUS = {"missing": "沒有未標記的 build", "not_reached": "未達到目標", "unbounded": "第一個 ef 就已達到，上界未知",
                    "few_builds": f"未標記的 build 少於 {GATE_MIN_BUILDS} 個"}


def _variance_rows(ranges: dict) -> list[str]:
    def span(summary, scale=1.0, digits=0):
        return "—" if summary is None else (f"{summary['median'] / scale:.{digits}f} "
                                           f"[{summary['min'] / scale:.{digits}f}, {summary['max'] / scale:.{digits}f}]")
    rows = []
    for target, entries in ranges.items():
        for e in entries:
            cell = f"| {target} | ({e['m']}, {e['ef_construction']}) | {e['implementation']} |"
            if e["missing"]:
                rows.append(f"{cell} — | 0 | 沒有未標記的 build | | | | |")
                continue
            mark = "（不足 k）" if e["incomplete"] else ""
            rows.append(f"{cell} {e['ef_search']}{mark} | {e['builds']} | {span(e['recall10'], digits=4)} | {span(e['qps'])} | "
                        f"{span(e['p50_ns'], 1e3, 1)} | {span(e['p95_ns'], 1e3, 1)} | "
                        f"{'是' if e['same_region'] else '否（最大 ef 仍未達到，013 §3 fallback）'} |")
    return rows


def _bracket_cell(bracket: dict) -> str:
    if not bracket["reached"]:
        return "未達到"
    upper = "上界未知" if bracket["upper"] is None else f"{bracket['upper']:.0f}"
    return f"[{bracket['lower']:.0f}, {upper}]"


def _gate_cells(cells: list[dict]) -> str:
    if not cells:
        return "—"
    parts = []
    for c in cells:
        sides = []
        for role in ("faster", "slower"):
            side = c[role]
            if side["status"] in ("ok", "few_builds"):
                text = f"{side['implementation']} ef {side['ef_search']} {side['value']:.0f}（{side['builds']} builds）"
                sides.append(text if side["status"] == "ok" else f"{text}，{GATE_CELL_STATUS['few_builds']}")
            else:
                sides.append(f"{side['implementation']}：{GATE_CELL_STATUS[side['status']]}")
        verdict = "分開" if c["separated"] else "未分開"
        parts.append(f"({c['m']}, {c['ef_construction']})：{' vs '.join(sides)} → {verdict}")
    return "；".join(parts)


def markdown(report: dict) -> str:
    lines = [f"# Benchmark 報告：{report['dataset']}（{report['mode']}）", ""]
    if report["mode"] != "formal":
        lines += [f"> **{report['mode']} 輸出只用來驗證工具本身，數字不得用於報告或任何結論。**", ""]
    lines += [f"- 協定：{report['protocol']}；run 狀態：{', '.join(report['run_status'])}",
              "- 規則：decision 010（id-based Recall@10、跨 seed median、不內插、不外推；被標記的點不進入主要讀值）。", ""]

    lines += ["## Frontier（主要讀值）", "",
              "index memory 以 phys_footprint 的增量為主要指標，index RSS（rss 的增量）為輔助指標（decision 014）。", "",
              "| 實作 | n | 目標 | QPS | P50 (µs) | P95 (µs) | index memory，phys_footprint (MiB) | index RSS，輔助 (MiB) | 排除標記點是否改變讀值 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for impl, per_n in sorted(report["frontier"].items()):
        for n, per_target in sorted(per_n.items(), key=lambda item: int(item[0])):
            for target, entry in per_target.items():
                main = entry["main"]
                lines.append(f"| {impl} | {n} | {target} | {_reading_cell(main, 'qps', 1, 0)} | "
                             f"{_reading_cell(main, 'p50_ns', 1e3, 1)} | {_reading_cell(main, 'p95_ns', 1e3, 1)} | "
                             f"{_reading_cell(main, 'index_footprint_bytes', 2**20, 1)} | "
                             f"{_reading_cell(main, 'index_memory_bytes', 2**20, 1)} | {'是' if entry['differs'] else '否'} |")
    differing = [(impl, n, target, entry["all_points"])
                 for impl, per_n in sorted(report["frontier"].items())
                 for n, per_target in sorted(per_n.items(), key=lambda item: int(item[0]))
                 for target, entry in per_target.items() if entry["differs"]]
    if differing:
        lines += ["", "## 包含被標記點的讀值（只列出與主要讀值不同者，010 §3）", "",
                  "| 實作 | n | 目標 | QPS | P50 (µs) | P95 (µs) | index memory，phys_footprint (MiB) | index RSS，輔助 (MiB) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for impl, n, target, reading in differing:
            lines.append(f"| {impl} | {n} | {target} | {_reading_cell(reading, 'qps', 1, 0)} | "
                         f"{_reading_cell(reading, 'p50_ns', 1e3, 1)} | {_reading_cell(reading, 'p95_ns', 1e3, 1)} | "
                         f"{_reading_cell(reading, 'index_footprint_bytes', 2**20, 1)} | "
                         f"{_reading_cell(reading, 'index_memory_bytes', 2**20, 1)} |")
    m, efc = report["aux_scaling_cell"]
    lines += ["", f"## 輔助 scaling 讀值：每個 n 只用 (M, efC) = ({m}, {efc}) 的點（013 §4，不取代主要讀值）", "",
              "主要讀值的 n 之間格點數不同；這裡固定同一格，只用來呈現這個混淆。", "",
              "| 實作 | n | 目標 | QPS | P50 (µs) | P95 (µs) | 排除標記點是否改變讀值 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for impl, per_n in sorted(report["aux_scaling_frontier"].items()):
        for n, per_target in sorted(per_n.items(), key=lambda item: int(item[0])):
            for target, entry in per_target.items():
                main = entry["main"]
                lines.append(f"| {impl} | {n} | {target} | {_reading_cell(main, 'qps', 1, 0)} | "
                             f"{_reading_cell(main, 'p50_ns', 1e3, 1)} | {_reading_cell(main, 'p95_ns', 1e3, 1)} | "
                             f"{'是' if entry['differs'] else '否'} |")
    aux_differing = [(impl, n, target, entry["all_points"])
                     for impl, per_n in sorted(report["aux_scaling_frontier"].items())
                     for n, per_target in sorted(per_n.items(), key=lambda item: int(item[0]))
                     for target, entry in per_target.items() if entry["differs"]]
    if aux_differing:
        lines += ["", "### 包含被標記點的輔助讀值（只列出與上表不同者，010 §3）", "",
                  "| 實作 | n | 目標 | QPS | P50 (µs) | P95 (µs) |", "| --- | --- | --- | --- | --- | --- |"]
        for impl, n, target, reading in aux_differing:
            lines.append(f"| {impl} | {n} | {target} | {_reading_cell(reading, 'qps', 1, 0)} | "
                         f"{_reading_cell(reading, 'p50_ns', 1e3, 1)} | {_reading_cell(reading, 'p95_ns', 1e3, 1)} |")
    lines += ["", "## QPS 比值（相對 rust-hnsw；兩者都達到目標才計算）", "", "| n | 目標 | 實作 | 倍數 |", "| --- | --- | --- | --- |"]
    for n, per_target in sorted(report["ratios"].items(), key=lambda item: int(item[0])):
        for target, entry in per_target.items():
            for impl, value in entry.items():
                lines.append(f"| {n} | {target} | {impl} | {_fmt(value, digits=2)} |")

    if report["variance_ranges"]:
        lines += ["", f"## 主要結論附上的 seed 範圍（013 §3；n = {VARIANCE_N}，各實作在自己的最小達標 ef；median [min, max]）", "",
                  "排除被標記的 build（010 §2）。其他 n 只有 1 個 seed：它們的讀值以本表作為代理範圍；不同 n，大 n 的 seed 變異可能不同（013 §3）。", "",
                  "| 目標 | (M, efC) | 實作 | ef | builds | Recall@10 | QPS | P50 (µs) | P95 (µs) | 與讀值同一 recall 區間 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        lines += _variance_rows(report["variance_ranges"])
        differing = {target: [e for e, other in zip(entries, report["variance_ranges_all_builds"][target]) if e != other]
                     for target, entries in report["variance_ranges"].items()}
        if any(differing.values()):
            lines += ["", "### 包含被標記 build 的 seed 範圍（只列出與上表不同者，010 §3）", "",
                      "| 目標 | (M, efC) | 實作 | ef | builds | Recall@10 | QPS | P50 (µs) | P95 (µs) | 與讀值同一 recall 區間 |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            keys = {target: {(e["implementation"], e["m"], e["ef_construction"]) for e in entries}
                    for target, entries in differing.items()}
            lines += _variance_rows({target: [e for e in entries if (e["implementation"], e["m"], e["ef_construction"])
                                              in keys[target]]
                                     for target, entries in report["variance_ranges_all_builds"].items()})
    if report["gate"]:
        lines += ["", f"## Gate 讀值（decision 015；n = {VARIANCE_N}；建議判讀，是否進 Advanced 由使用者決定）", "",
                  "QPS 範圍：不內插，各 (M, efC) 格點中 recall 第一次達到目標的 ef 是下界、前一個 ef 是上界（015 §1）。"
                  f"seed 範圍：每個附上的 variance 格點上，較快一方在交點之後最慢的 seed 必須快於較慢一方在交點之前最快的 seed（各至少 {GATE_MIN_BUILDS} 個未標記的 build）。", "",
                  f"| 目標 | 主要 | rust-hnsw QPS 範圍 | {GATE_COMPARED} QPS 範圍 | variance 格點的 seed 範圍 | 判讀 | 包含被標記 build 的判讀 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for target, g in report["gate"].items():
            main = g["main"]
            lines.append(f"| {target} | {'是' if g['primary'] else ''} | {_bracket_cell(main['brackets'][BASELINE])} | "
                         f"{_bracket_cell(main['brackets'][GATE_COMPARED])} | {_gate_cells(main['cells'])} | "
                         f"{GATE_VERDICTS[main['verdict']]} | "
                         f"{GATE_VERDICTS[g['all_builds']['verdict']] + '（與主要判讀不同）' if g['differs'] else '相同'} |")

    lines += ["", "## recall 隨 efSearch（010 §6）", "", "| 實作 | n | M | efC | 系統性惡化 | 最大相鄰下降 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for c in report["recall_vs_ef"]:
        lines.append(f"| {c['implementation']} | {c['n']} | {c['m']} | {c['ef_construction']} | "
                     f"{'是' if c['systematic_deterioration'] else '否'} | {c['max_adjacent_drop']:.4f} |")

    lines += ["", "## seed variance（原始插入順序；median [min, max]）", "",
              "| 實作 | n | M | efC | ef | builds | Recall@10 | Recall@1 | QPS | 被標記的 build |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a in report["seed_variance"]:
        def span(summary, digits=4):
            return "—" if summary is None else f"{summary['median']:.{digits}f} [{summary['min']:.{digits}f}, {summary['max']:.{digits}f}]"
        lines.append(f"| {a['implementation']} | {a['n']} | {a['m']} | {a['ef_construction']} | {a['ef_search']} | "
                     f"{a['builds']} | {span(a['recall10'])} | {span(a['recall1'])} | {span(a['qps'], 0)} | {a['flagged_builds']} |")

    if report["order_variance"]:
        lines += ["", "## insertion order variance", "", "| 實作 | n | M | efC | ef | seed | shuffles | Recall@10 | QPS | 備註 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for o in report["order_variance"]:
            r, q = o["recall10"], o["qps"]
            lines.append(f"| {o['implementation']} | {o['n']} | {o['m']} | {o['ef_construction']} | {o['ef_search']} | "
                         f"{o['seed']} | {o['shuffles']} | {r['median']:.4f} [{r['min']:.4f}, {r['max']:.4f}] | "
                         f"{q['median']:.0f} | {o['note']} |")

    lines += ["", "## scaling（每個 build 計一次；排除被標記的 build；median [min, max]）", "",
              "| 實作 | n | M | efC | build (s) | peak RSS (MiB) | index memory，phys_footprint (MiB) | index RSS，輔助 (MiB) | builds | 被標記的 build |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in report["scaling"]:
        def span(summary, scale, digits):
            return "—" if summary is None else (f"{summary['median'] / scale:.{digits}f} "
                                               f"[{summary['min'] / scale:.{digits}f}, {summary['max'] / scale:.{digits}f}]")
        lines.append(f"| {s['implementation']} | {s['n']} | {s['m']} | {s['ef_construction']} | {span(s['build_ns'], 1e9, 3)} | "
                     f"{span(s['peak_rss_bytes'], 2**20, 1)} | {span(s['index_footprint_bytes'], 2**20, 1)} | "
                     f"{span(s['index_memory_bytes'], 2**20, 1)} | {s['builds']} | {s['flagged_builds']} |")

    if report.get("figures"):
        lines += ["", "## 圖", "",
                  "frontier 與 scaling 圖排除被標記的點（與主要讀值相同）；圖只呈現形狀，數字以上面的表為準（010 §2：不內插）。", ""]
        lines += [f"- `{name}`" for name in report["figures"]]

    if report.get("run_configs"):
        lines += ["", "## 合併的 run", ""]
        lines += [f"- {c['run']}：{c['status']}，config {c['config']}"
                  + (f"（blob {c['formal_config']['blob'][:12]}）" if c.get("formal_config") else "") for c in report["run_configs"]]
    for title, key in (("結果不足 k 的點", "incomplete_points"), ("被標記的點（不進入主要讀值）", "flagged_points"),
                       ("失敗的子 process", "failed_children"),
                       ("被排除的 run（aborted / invalid / error，或 decision 016 登記作廢；整份 config 已重跑，資料不合併，013 §5）", "excluded_runs")):
        lines += ["", f"## {title}", ""]
        items = report[key]
        lines += [f"- {json.dumps(item, ensure_ascii=False)}" for item in items] if items else ["（無）"]
    return "\n".join(lines) + "\n"


def plots(report: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    curves = defaultdict(list)
    for a in report["seed_variance"]:
        curves[(a["n"], a["m"], a["ef_construction"])].append(a)
    for (n, m, efc), members in sorted(curves.items()):
        fig, ax = plt.subplots(figsize=(6, 4))
        for impl in sorted({a["implementation"] for a in members}):
            series = sorted((a for a in members if a["implementation"] == impl), key=lambda a: a["ef_search"])
            efs = [a["ef_search"] for a in series]
            med = np.array([a["recall10"]["median"] for a in series])
            low = med - np.array([a["recall10"]["min"] for a in series])
            high = np.array([a["recall10"]["max"] for a in series]) - med
            ax.errorbar(efs, med, yerr=[low, high], marker="o", capsize=3, label=impl)
        ax.set_xscale("log")
        ax.set_xlabel("efSearch")
        ax.set_ylabel("Recall@10 (median over seeds, min–max)")
        ax.set_title(f"{report['dataset']} ({report['mode']}): n={n}, M={m}, efC={efc}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = out_dir / f"recall_vs_ef_n{n}_m{m}_efc{efc}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    for n in sorted({a["n"] for a in report["seed_variance"]}):
        fig, ax = plt.subplots(figsize=(6, 4))
        for impl in sorted({a["implementation"] for a in report["seed_variance"]}):
            points = [a for a in report["seed_variance"] if a["n"] == n and a["implementation"] == impl and a["qps"]]
            if not points:
                continue
            ax.scatter([a["recall10"]["median"] for a in points], [a["qps"]["median"] for a in points], label=impl, s=18)
        for target in TARGETS:
            ax.axvline(target, color="grey", linestyle=":", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel("Recall@10 (median over seeds)")
        ax.set_ylabel("QPS (median over seeds)")
        ax.set_title(f"{report['dataset']} ({report['mode']}): n={n}, all (M, efC, ef) points")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = out_dir / f"recall_qps_n{n}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    written += frontier_plots(report, out_dir, plt)
    written += scaling_plots(report, out_dir, plt)
    return written


# 三個實作的固定顏色與標記（顏色跟著實作走，不隨排序或篩選改變；色盲可辨識度以 dataviz 的 validator 驗證過，
# 青綠對白底的對比不足 3:1，所以每條線另有標記形狀與線尾的直接標籤）。
IMPL_STYLE = {
    "rust-hnsw": {"color": "#2a78d6", "marker": "o"},
    "hnswlib": {"color": "#eb6834", "marker": "s"},
    "faiss-hnsw": {"color": "#1baf7a", "marker": "^"},
}
FRONTIER_PANELS = (  # (指標, 比例, 單位標籤, 越高越好)
    ("qps", 1.0, "QPS (median over seeds)", True),
    ("p50_ns", 1e3, "P50 latency (µs)", False),
    ("p95_ns", 1e3, "P95 latency (µs)", False),
    ("index_footprint_bytes", 2**20, "index memory, phys_footprint (MiB)", False),
)


def _plain_log_ticks(axis):
    """log 軸的刻度以一般數字顯示（不用 6×10¹ 這種科學記號）。"""
    from matplotlib.ticker import FuncFormatter

    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if str(f"{v:g}")[0] in "25" else ""))


def _n_ticks(ax, ns):
    labels = {n: (f"{n / 1e6:.2f}M" if n >= 1_000_000 else f"{n // 1000}k") for n in ns}
    from matplotlib.ticker import NullLocator

    ax.set_xticks(list(ns))
    ax.set_xticklabels([labels[n] for n in ns])
    ax.xaxis.set_minor_locator(NullLocator())


def _style_axes(ax):
    ax.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b4b2aa")
    ax.tick_params(colors="#52514e", labelsize=8)


def _impl_line(ax, impl, xs, ys):
    style = IMPL_STYLE.get(impl, {"color": "#52514e", "marker": "o"})
    ax.plot(xs, ys, color=style["color"], marker=style["marker"], markersize=5, linewidth=2,
            markeredgecolor="#fcfcfb", markeredgewidth=1, label=impl)
    ax.annotate(impl, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color="#52514e",
                va="center")


def frontier_plots(report: dict, out_dir: Path, plt) -> list[Path]:
    """每個 n：recall–QPS、recall–P50 / P95、recall–index memory 的 frontier（排除被標記的點，與主要讀值相同，010 §2–3）。

    圖只呈現 frontier 的形狀；數字讀值以 report.md 的表為準（目標 recall 上的讀值不內插，010 §2）。
    """
    clean = report.get("clean_aggregates") or []
    written = []
    for n in sorted({a["n"] for a in clean}):
        impls = [impl for impl in IMPL_STYLE if any(a["implementation"] == impl and a["n"] == n for a in clean)]
        if not impls:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        for ax, (metric, scale, label, higher) in zip(axes.flat, FRONTIER_PANELS):
            for impl in impls:
                front = pareto_front([a for a in clean if a["implementation"] == impl and a["n"] == n], metric, higher)
                if front:
                    _impl_line(ax, impl, [a["recall10"]["median"] for a in front],
                               [a[metric]["median"] / scale for a in front])
            for target in TARGETS:
                ax.axvline(target, color="#b4b2aa", linestyle=":", linewidth=1)
            if metric != "index_footprint_bytes":
                ax.set_yscale("log")
                _plain_log_ticks(ax.yaxis)
            ax.set_xlabel("Recall@10 (median over seeds)", fontsize=8, color="#52514e")
            ax.set_ylabel(label, fontsize=8, color="#52514e")
            _style_axes(ax)
        axes.flat[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"{report['dataset']} ({report['mode']}): n={n}, recall frontiers over all (M, efC, ef); "
                     "flagged points excluded", fontsize=10)
        fig.tight_layout()
        path = out_dir / f"frontier_n{n}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def scaling_plots(report: dict, out_dir: Path, plt) -> list[Path]:
    """scaling with N（010 §5）：固定 (M, efC) 的 build time 與 index memory，以及各目標 recall 的 QPS 讀值（主要讀值）。"""
    rows = [s for s in report.get("scaling", []) if (s["m"], s["ef_construction"]) == AUX_SCALING_CELL]
    ns = sorted({s["n"] for s in rows if s.get("build_ns")})  # 只有被標記 build 的 n 沒有主要讀值，不畫
    if len(ns) < 2:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    m, efc = AUX_SCALING_CELL
    for ax, (metric, scale, label) in zip(axes[:2], (("build_ns", 1e9, "build time (s)"),
                                                     ("index_footprint_bytes", 2**20, "index memory, phys_footprint (MiB)"))):
        for impl in IMPL_STYLE:
            series = sorted((s for s in rows if s["implementation"] == impl and s.get(metric)), key=lambda s: s["n"])
            if series:
                _impl_line(ax, impl, [s["n"] for s in series], [s[metric]["median"] / scale for s in series])
        ax.set_xscale("log")
        ax.set_yscale("log")
        _n_ticks(ax, ns)
        _plain_log_ticks(ax.yaxis)
        ax.set_xlabel("n", fontsize=8, color="#52514e")
        ax.set_ylabel(f"{label}, (M, efC) = ({m}, {efc})", fontsize=8, color="#52514e")
        _style_axes(ax)
    ax = axes[2]
    for impl in IMPL_STYLE:
        per_n = report["frontier"].get(impl, {})
        for target, linestyle in zip(("0.90", "0.95"), ("-", "--")):
            pts = [(int(n), per_n[n][target]["main"]["qps"]["value"]) for n in sorted(per_n, key=int)
                   if per_n[n][target]["main"].get("reached") and per_n[n][target]["main"].get("qps")]
            if len(pts) >= 1:
                style = IMPL_STYLE[impl]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=style["color"], marker=style["marker"],
                        linestyle=linestyle, linewidth=2, markersize=5, markeredgecolor="#fcfcfb",
                        label=f"{impl} @ {target}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _n_ticks(ax, sorted({int(n) for per_n in report["frontier"].values() for n, per_target in per_n.items()
                         if any(entry["main"].get("reached") for entry in per_target.values())}))
    _plain_log_ticks(ax.yaxis)
    ax.set_xlabel("n", fontsize=8, color="#52514e")
    ax.set_ylabel("QPS at target Recall@10 (frontier reading)", fontsize=8, color="#52514e")
    ax.legend(fontsize=7, frameon=False)
    _style_axes(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{report['dataset']} ({report['mode']}): scaling with n; flagged builds and points excluded", fontsize=10)
    fig.tight_layout()
    path = out_dir / "scaling.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [path]


def _load(run_dir) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))


def excluded_runs(run_dirs) -> list[dict]:
    """只有失敗的 run（013 §5）與 decision 016 登記作廢的 run 可以被排除；其他 complete / incomplete 的 run 不得排除，避免事後挑選。"""
    entries = []
    for run_dir in run_dirs:
        record = _load(run_dir)
        voided = VOIDED_RUNS.get(Path(run_dir).name)
        if record["status"] not in FAILED_STATUS and voided is None:
            raise ReportError(f"{run_dir} has status {record['status']!r}; only aborted / invalid / error runs and runs "
                              "voided by a decision (VOIDED_RUNS) can be excluded")
        entry = {"run": str(run_dir), "status": record["status"], "status_reasons": record["status_reasons"]}
        if voided is not None:
            entry["voided"] = voided
        entries.append(entry)
    return entries


def _config_path(record: dict) -> str | None:
    """formal run 的 config 路徑（repo 相對，008 §1.8 的 commit 檢查寫入）；smoke 沒有。"""
    checks = record.get("metadata", {}).get("formal_checks") or {}
    return (checks.get("config") or {}).get("path")


def check_formal_set(records: list[dict], run_dirs, excluded_dirs, roots=None) -> None:
    """formal 報告（013 §5、010 §3）：恰好包含該資料集每一份 formal config 各一個 run；
    同一份 config 另外的 formal run 不得被略過——complete / incomplete 的一律拒絕，失敗的必須以 --excluded 列出。"""
    dataset = records[0]["dataset"]["name"]
    expected = FORMAL_CONFIGS.get(dataset)
    if expected is None:
        raise ReportError(f"no formal configs are registered for {dataset!r} (013 §5)")
    paths = [_config_path(r) for r in records]
    for run_dir, path in zip(run_dirs, paths):
        if path not in expected:
            raise ReportError(f"{run_dir} ran config {path!r}, which is not a formal config of {dataset} (013 §5)")
    for path in expected:
        count = paths.count(path)
        if count != 1:
            raise ReportError(f"{path} contributes {count} runs; every formal config must contribute exactly one (013 §5)")
    given = {Path(d).resolve() for d in run_dirs} | {Path(d).resolve() for d in excluded_dirs}
    roots = {RUNS_ROOT, *(Path(d).resolve().parent for d in run_dirs)} if roots is None else roots
    for root in roots:
        for path in sorted(Path(root).glob("*/run.json")):
            if path.parent.resolve() in given:
                continue
            try:
                other = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if other.get("mode") != "formal" or other.get("dataset", {}).get("name") != dataset \
                    or _config_path(other) not in expected:
                continue
            if other.get("status") in FAILED_STATUS:
                raise ReportError(f"{path.parent} is a failed formal run of {_config_path(other)}; list it with --excluded")
            if path.parent.name in VOIDED_RUNS:
                raise ReportError(f"{path.parent} was voided by a decision ({VOIDED_RUNS[path.parent.name]}); "
                                  "list it with --excluded")
            raise ReportError(f"{path.parent} is another formal run of {_config_path(other)}; a completed formal run "
                              "cannot be replaced or dropped (010 §3)")


def write_report(run_dirs, out_dir, excluded=(), roots=None) -> Path:
    records = [_load(d) for d in run_dirs]
    report = build_report(records, labels=[str(d) for d in run_dirs])
    if report["mode"] == "formal":
        check_formal_set(records, run_dirs, excluded, roots=roots)
    report["excluded_runs"] = excluded_runs(excluded)
    report["runs"] = [str(d) for d in run_dirs]
    report["run_configs"] = [{"run": str(d), "status": r["status"], "config": r.get("config", {}).get("path"),
                              "formal_config": (r.get("metadata", {}).get("formal_checks") or {}).get("config")}
                             for d, r in zip(run_dirs, records)]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report["figures"] = [path.name for path in plots(report, out)]
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="runner 的輸出目錄（含 run.json）")
    parser.add_argument("--out", help="輸出目錄；預設為第一個 run 目錄下的 report/")
    parser.add_argument("--excluded", action="append", default=[],
                        help="已整份重跑的失敗 run（aborted / invalid / error）：只列出狀態與原因，資料不合併")
    args = parser.parse_args(argv)
    out = write_report(args.runs, args.out or Path(args.runs[0]) / "report", excluded=args.excluded)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
