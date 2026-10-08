"""從 runner 的 run.json 產生報告（decision 010 的比較規則）。

用法：
    .venv/bin/python -m benchmarks.report <run 目錄> [<run 目錄> ...] [--out <輸出目錄>]

- 主要比較：每個（實作、n）以所有 (M, efC, ef) 點組成 frontier，每個點取跨 seed 的 median（只用原始插入順序）；
  對每個目標 recall，在 median Recall@10 ≥ 目標的點中取 QPS 最高、P50 / P95 最低、index memory 最低。不內插、不外推。
- 有 hygiene 標記的點不進入主要讀值；若排除與否會改變讀值，兩者都保留（010 §3）。
- index memory 另列 phys_footprint 的讀值作為輔助欄位：macOS 上已釋放的大型配置仍算在 rss（NEXT.md B1，待使用者決定），
  主要讀值仍是 008 §8 的 `index_memory_bytes`。
- smoke 的報告只用來驗證工具本身，不得用於任何結論。
- 合併多個 run 之前（013 §5）：只接受 complete / incomplete 的 run；同一個 build 不得出現在多個 run；
  各 run 的 hnswlib / faiss 擴充模組、套件版本、`rust_hnsw.BUILD_INFO` 必須相同，commit 不同時實作與量測程式不得有差異。
  aborted / invalid / error 的 run 整份重跑，原本的 run 只能以 `--excluded` 列出（不合併資料）。
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGETS = (0.90, 0.95, 0.99)
# 008 §9 的 hygiene 標記；子 process、build、點三層的標記都合併到點上（010 §3）。
HYGIENE_FLAGS = ("load", "thermal", "swap", "multithread", "nondeterministic", "power", "memory")
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
REPO = Path(__file__).resolve().parents[1]
# 實作與量測程式：合併的 run 之間（不同 commit 時）這些路徑不得有差異（013 §5）。
MEASUREMENT_PATHS = (
    "crates", "python", "Cargo.toml", "Cargo.lock", "requirements-dev.txt",
    *(f"benchmarks/{name}" for name in ("run.py", "run_child.py", "adapters.py", "interface.py", "protocol.py",
                                        "evaluate.py", "metrics.py", "environment.py", "datasets.py", "config.py")),
)
# 輔助 scaling 讀值（013 §4）：每個 n 只用這一格 (M, efC) 的點讀 frontier，只用來呈現格點數不同造成的混淆。
AUX_SCALING_CELL = (16, 200)


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
        return {"value": chosen[metric]["median"], "m": chosen["m"], "ef_construction": chosen["ef_construction"],
                "ef_search": chosen["ef_search"], "incomplete": chosen["incomplete"], "builds": chosen["builds"]}

    return {"reached": True, "qps": pick("qps", max), "p50_ns": pick("p50_ns", min), "p95_ns": pick("p95_ns", min),
            "index_memory_bytes": pick("index_memory_bytes", min),
            "index_footprint_bytes": pick("index_footprint_bytes", min)}


def frontier(all_aggregates: list[dict], clean_aggregates: list[dict]) -> dict:
    result = defaultdict(dict)
    keys = sorted({(a["implementation"], a["n"]) for a in all_aggregates})
    for impl, n in keys:
        everything = [a for a in all_aggregates if (a["implementation"], a["n"]) == (impl, n)]
        clean = [a for a in clean_aggregates if (a["implementation"], a["n"]) == (impl, n)]
        per_target = {}
        for target in TARGETS:
            main, all_points = _reading(clean, target), _reading(everything, target)
            per_target[_target_key(target)] = {"main": main, "all_points": all_points, "differs": main != all_points}
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
    first = records[0]["metadata"]
    for label, record in zip(labels[1:], records[1:]):
        meta = record["metadata"]
        for field, getter in (("the hnswlib extension", lambda m: m["extension_sha256"]["hnswlib"]),
                              ("the faiss extension", lambda m: m["extension_sha256"]["faiss"]),
                              ("package versions", lambda m: m["versions"]),
                              ("rust_hnsw.BUILD_INFO", lambda m: m["build_info"])):
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
        "aux_scaling_frontier": frontier(in_cell, clean_in_cell),
        "excluded_runs": [],
        "ratios": ratios(front),
        "seed_variance": all_aggregates,
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
    return f"{item['value'] / scale:.{digits}f} @ M={item['m']}, efC={item['ef_construction']}, ef={item['ef_search']}{mark}"


def markdown(report: dict) -> str:
    lines = [f"# Benchmark 報告：{report['dataset']}（{report['mode']}）", ""]
    if report["mode"] != "formal":
        lines += [f"> **{report['mode']} 輸出只用來驗證工具本身，數字不得用於報告或任何結論。**", ""]
    lines += [f"- 協定：{report['protocol']}；run 狀態：{', '.join(report['run_status'])}",
              "- 規則：decision 010（id-based Recall@10、跨 seed median、不內插、不外推；被標記的點不進入主要讀值）。", ""]

    lines += ["## Frontier（主要讀值）", "",
              "index footprint 是輔助欄位（phys_footprint；不含已釋放但仍算在 rss 的 page，見 NEXT.md B1），主要讀值仍是 index memory（rss）。", "",
              "| 實作 | n | 目標 | QPS | P50 (µs) | P95 (µs) | index memory (MiB) | index footprint (MiB) | 排除標記點是否改變讀值 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for impl, per_n in sorted(report["frontier"].items()):
        for n, per_target in sorted(per_n.items(), key=lambda item: int(item[0])):
            for target, entry in per_target.items():
                main = entry["main"]
                lines.append(f"| {impl} | {n} | {target} | {_reading_cell(main, 'qps', 1, 0)} | "
                             f"{_reading_cell(main, 'p50_ns', 1e3, 1)} | {_reading_cell(main, 'p95_ns', 1e3, 1)} | "
                             f"{_reading_cell(main, 'index_memory_bytes', 2**20, 1)} | "
                             f"{_reading_cell(main, 'index_footprint_bytes', 2**20, 1)} | {'是' if entry['differs'] else '否'} |")
    differing = [(impl, n, target, entry["all_points"])
                 for impl, per_n in sorted(report["frontier"].items())
                 for n, per_target in sorted(per_n.items(), key=lambda item: int(item[0]))
                 for target, entry in per_target.items() if entry["differs"]]
    if differing:
        lines += ["", "## 包含被標記點的讀值（只列出與主要讀值不同者，010 §3）", "",
                  "| 實作 | n | 目標 | QPS | P50 (µs) | P95 (µs) | index memory (MiB) | index footprint (MiB) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for impl, n, target, reading in differing:
            lines.append(f"| {impl} | {n} | {target} | {_reading_cell(reading, 'qps', 1, 0)} | "
                         f"{_reading_cell(reading, 'p50_ns', 1e3, 1)} | {_reading_cell(reading, 'p95_ns', 1e3, 1)} | "
                         f"{_reading_cell(reading, 'index_memory_bytes', 2**20, 1)} | "
                         f"{_reading_cell(reading, 'index_footprint_bytes', 2**20, 1)} |")
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
    lines += ["", "## QPS 比值（相對 rust-hnsw；兩者都達到目標才計算）", "", "| n | 目標 | 實作 | 倍數 |", "| --- | --- | --- | --- |"]
    for n, per_target in sorted(report["ratios"].items(), key=lambda item: int(item[0])):
        for target, entry in per_target.items():
            for impl, value in entry.items():
                lines.append(f"| {n} | {target} | {impl} | {_fmt(value, digits=2)} |")

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
              "| 實作 | n | M | efC | build (s) | peak RSS (MiB) | index memory (MiB) | index footprint (MiB) | builds | 被標記的 build |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in report["scaling"]:
        def span(summary, scale, digits):
            return "—" if summary is None else (f"{summary['median'] / scale:.{digits}f} "
                                               f"[{summary['min'] / scale:.{digits}f}, {summary['max'] / scale:.{digits}f}]")
        lines.append(f"| {s['implementation']} | {s['n']} | {s['m']} | {s['ef_construction']} | {span(s['build_ns'], 1e9, 3)} | "
                     f"{span(s['peak_rss_bytes'], 2**20, 1)} | {span(s['index_memory_bytes'], 2**20, 1)} | "
                     f"{span(s['index_footprint_bytes'], 2**20, 1)} | {s['builds']} | {s['flagged_builds']} |")

    for title, key in (("結果不足 k 的點", "incomplete_points"), ("被標記的點（不進入主要讀值）", "flagged_points"),
                       ("失敗的子 process", "failed_children"),
                       ("被排除的 run（aborted / invalid / error；整份 config 已重跑，資料不合併，013 §5）", "excluded_runs")):
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
    return written


def _load(run_dir) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))


def excluded_runs(run_dirs) -> list[dict]:
    """只有失敗的 run 可以被排除（013 §5）；complete / incomplete 的 run 不得排除，避免事後挑選。"""
    entries = []
    for run_dir in run_dirs:
        record = _load(run_dir)
        if record["status"] not in FAILED_STATUS:
            raise ReportError(f"{run_dir} has status {record['status']!r}; only aborted / invalid / error runs can be "
                              "excluded")
        entries.append({"run": str(run_dir), "status": record["status"], "status_reasons": record["status_reasons"]})
    return entries


def write_report(run_dirs, out_dir, excluded=()) -> Path:
    records = [_load(d) for d in run_dirs]
    report = build_report(records, labels=[str(d) for d in run_dirs])
    report["excluded_runs"] = excluded_runs(excluded)
    report["runs"] = [str(d) for d in run_dirs]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    plots(report, out)
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
