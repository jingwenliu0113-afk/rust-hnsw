"""Advanced C 的真實資料 correctness smoke 的核對（decision 019 §9 第 6 步）。

用法（smoke run 完成之後；`--port` 會在真實資料上重建 FAISS 的移植，所以先通過 machine_quiet，再在 heavy lock 下執行，019 §5）：
    ./tools/with-mac-lock .venv/bin/python -m benchmarks.run benchmarks/configs/<glove|text>-quant-smoke.toml --mode smoke
    .venv/bin/python -m benchmarks.quant_smoke <smoke run 目錄…> --core-runs <該資料集在 019 §2 登記的 Core formal run 目錄…>
    ./tools/with-mac-lock .venv/bin/python -m benchmarks.quant_smoke <smoke run 目錄…> --core-runs <…> --port

- smoke 的內容固定在 `benchmarks/configs/<glove|text>-quant-smoke.toml`（019 §9 第 6 步：seed 1、五個臂、
  GloVe ef {64, 304, 768}、文字 ef {24, 48, 512}、smoke 模式）；run 的 config 與 ef 必須就是它們。
- 核對（019 §9 第 6 步只檢查這些）：
  - 結果格式、010 §6、null call 與 warm-up：由 runner 在 smoke 模式本身檢查；這裡照列 run 與每個子 process 的狀態與標記。
  - F / R / P 與 Core 的 ids 逐位元相同：與 C7 同一個比對（`quant_report.c7_core`），seed 1、每個共同的 ef；另比對 C 內 F 與 R（所有 ef）。
  - Q 的移植：Q 的子 process 完成；`--port` 另外以 Q adapter 的同一段建圖程式重建 seed 1 的移植，在 `del flat` 之前核對 graph
    （C8 的同一個比對，`quant_diag.faiss_graph_differences`），並確認重建的 Q 在 e* 的 ids 與 smoke run 相同。
  - 每個子 process 的時間：照列，只用來依 019 §5 上調 timeout。
- smoke 之後不得修改假設、門檻、格點、ef、seed 或讀法；只能修 correctness bug（寫補記）或上調 timeout。
  smoke 中 C8 不一致時只能記為不適用，不得改用 §3 的備案（019 §9 第 6 步）。
- 不寫任何檔案；結果以 JSON 印出。全部通過時 exit code 為 0。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks import adapters, environment, profile, quant_diag, quant_report
from benchmarks.quant_report import (
    ANCHOR_SEED,
    E_STAR,
    EF_CONSTRUCTION,
    GLOVE,
    PROTOCOL,
    TEXT,
    F,
    K,
    M,
    N,
    P,
    Q,
    R,
    S,
)

# 019 §9 第 6 步的 smoke：config 的路徑與登記的 ef。
SMOKE_CONFIGS = {GLOVE: "benchmarks/configs/glove-quant-smoke.toml", TEXT: "benchmarks/configs/text-quant-smoke.toml"}
SMOKE_EFS = {GLOVE: (64, 304, 768), TEXT: (24, 48, 512)}
SMOKE_ARMS = (S, P, F, Q, R)


def _seconds(child) -> float | None:
    """子 process 的時間（run.py 以 `time.time()` 記錄 `started_at`、`ended_at`）。"""
    start, end = child.get("started_at"), child.get("ended_at")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    return float(end - start)


def check_run(label: str, record: dict) -> dict:
    """run 本身：smoke 模式、協定、登記的 config 與 ef、五個臂各一個 seed 1 的 build，以及每個子 process 的狀態與時間。"""
    dataset = record["dataset"]["name"]
    problems = []
    if dataset not in SMOKE_CONFIGS:
        problems.append(f"{dataset!r} is not a dataset of Advanced C (019 §5)")
    if record.get("mode") != "smoke":
        problems.append(f"mode {record.get('mode')!r}; the correctness smoke runs in smoke mode (019 §9 step 6)")
    if record["protocol"]["version"] != PROTOCOL:
        problems.append(f"protocol {record['protocol']['version']}, not {PROTOCOL} (019 §5)")
    path = (record.get("config") or {}).get("path")
    if dataset in SMOKE_CONFIGS and path != SMOKE_CONFIGS[dataset]:
        problems.append(f"config {path!r} is not {SMOKE_CONFIGS[dataset]} (019 §9 step 6)")
    efs = sorted((record.get("config") or {}).get("parsed", {}).get("ef_search", []))
    if dataset in SMOKE_EFS and efs != list(SMOKE_EFS[dataset]):
        problems.append(f"ef_search {efs} is not the registered {list(SMOKE_EFS[dataset])} (019 §9 step 6)")
    children = []
    for child in record["children"]:
        build = child["build"]
        children.append({"implementation": child["implementation"], "status": child["status"],
                         "seed": build["seed"], "in_cell": quant_report._in_cell(build), "seconds": _seconds(child),
                         "flags": sorted({*child["flags"], *build.get("flags", [])}),
                         "points": {p["ef_search"]: p["k10"]["status"] for p in child["points"]}})
    arms = sorted(c["implementation"] for c in children if c["in_cell"] and c["seed"] == ANCHOR_SEED)
    if arms != sorted(SMOKE_ARMS):
        problems.append(f"expected one seed-1 build of each arm {sorted(SMOKE_ARMS)}; found {arms}")
    if record["status"] != "complete":
        problems.append(f"run status {record['status']!r}: {record.get('status_reasons')}")
    for child in children:
        if child["status"] != "ok":
            problems.append(f"{child['implementation']}: child status {child['status']!r}")
        bad = {ef: status for ef, status in child["points"].items() if status != "ok"}
        if bad:
            problems.append(f"{child['implementation']}: points that are not ok {bad}")
    return {"run": label, "dataset": dataset, "status": record["status"], "children": children, "problems": problems}


def ids_against_core(labelled, core: dict, dataset: str) -> dict:
    """F / R / P（seed 1）在每個共同 ef 的 `ids_k10` 對照 Core 正式 run；C 內 F 與 R 在所有 ef 逐位元相同（C7 的同一個比對）。"""
    c_ids = quant_report.load_c_ids(labelled)
    efs = list(SMOKE_EFS[dataset])
    comparisons = quant_report.c7_core(c_ids, core, efs, [(F, (ANCHOR_SEED,)), (R, (ANCHOR_SEED,)), (P, (ANCHOR_SEED,))])
    for ef in efs:
        result = quant_report._compare(c_ids.get((F, ANCHOR_SEED, ef)), c_ids.get((R, ANCHOR_SEED, ef)))
        comparisons.append({"against": "C seed 1 rust-hnsw", "implementation": F, "seed": ANCHOR_SEED, "ef_search": ef,
                            "result": "c_missing" if result == "core_missing" else result})
    return {"status": quant_report._c7_status(comparisons),
            "identical": sum(c["result"] == "identical" for c in comparisons),
            "not_identical": [c for c in comparisons if c["result"] != "identical"]}


def port_check(data, queries, e_star: int, smoke_q_ids) -> dict:
    """以 Q adapter 的同一段建圖程式重建 seed 1 的移植：`del flat` 之前核對 graph（C8 的比對），重建的 Q 在 e* 的 ids 與 smoke 相同。"""
    import faiss

    faiss.omp_set_num_threads(1)
    flat = adapters.faiss_hnsw_flat(data, M, EF_CONSTRUCTION, ANCHOR_SEED)
    index = adapters.faiss_sq8_from_flat(flat, data, M)
    differences = quant_diag.faiss_graph_differences(flat.hnsw, index.hnsw)  # del flat 之前（019 §7）
    del flat
    index.hnsw.efSearch = e_star
    faiss.omp_set_num_threads(1)
    _, ids = index.search(queries, K)
    ids = np.asarray(ids, dtype=np.int64)
    return {"graph_identical": not differences, "graph_differences": differences, "e_star": e_star,
            "ids_match_at_e_star": None if smoke_q_ids is None else quant_diag._same(ids, smoke_q_ids)}


def _q_ids(labelled, e_star):
    for label, record in labelled:
        for child in record["children"]:
            if child["implementation"] != Q or not quant_report._in_cell(child["build"]):
                continue
            for point in child["points"]:
                if point["ef_search"] == e_star:
                    with np.load(Path(label) / point["file"]) as arrays:
                        return np.asarray(arrays["ids_k10"]) if "ids_k10" in arrays.files else None
    return None


def check(labelled, core_dirs, port: bool = False, load=profile.load_dataset) -> dict:
    """`labelled`：[(smoke run 目錄, run 紀錄)]，每個資料集最多一個。`load(dataset, n)` 讀入登記的資料（只在 `port` 時使用）。"""
    datasets = [record["dataset"]["name"] for _, record in labelled]
    if len(set(datasets)) != len(datasets):
        raise SystemExit(f"give at most one smoke run per dataset; got {datasets}")
    runs = [check_run(label, record) for label, record in labelled]
    known = [d for d in datasets if d in SMOKE_CONFIGS]
    core = quant_report.load_core(core_dirs, known) if known else {}
    result = {"tool": "benchmarks.quant_smoke", "decision": "019 §9 step 6", "runs": runs, "ids": {}, "port": {}}
    for dataset in known:
        group = [(label, record) for label, record in labelled if record["dataset"]["name"] == dataset]
        result["ids"][dataset] = ids_against_core(group, core[dataset], dataset)
        if port:
            data, queries, _, _ = load(dataset, N)
            e_star = E_STAR[dataset]
            result["port"][dataset] = port_check(data, queries, e_star, _q_ids(group, e_star))
    result["ok"] = (all(not r["problems"] for r in runs) and len(known) == len(datasets)
                    and all(entry["status"] == quant_report.CONSISTENT for entry in result["ids"].values())
                    and all(entry["graph_identical"] and entry["ids_match_at_e_star"] for entry in result["port"].values()))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="C 的 smoke run 目錄（每個資料集最多一個）")
    parser.add_argument("--core-runs", nargs="+", required=True, help="這些資料集在 019 §2 登記的 Core formal run 目錄（全部列出）")
    parser.add_argument("--port", action="store_true",
                        help="在真實資料上重建 seed 1 的 FAISS 移植並核對（需要 heavy lock，019 §5）")
    args = parser.parse_args(argv)
    if args.port:
        try:
            environment.check_lock()
        except environment.PreflightError as err:
            raise SystemExit(f"--port rebuilds the FAISS port on real data and needs the heavy lock (019 §5): {err}") from None
    labelled = [(str(d), json.loads((Path(d) / "run.json").read_text(encoding="utf-8"))) for d in args.runs]
    try:
        result = check(labelled, args.core_runs, port=args.port)
    except quant_report.ReportError as err:
        raise SystemExit(str(err)) from None
    print(json.dumps(result, indent=1, ensure_ascii=False, default=quant_report._json_default))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
