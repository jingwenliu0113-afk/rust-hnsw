"""Advanced C 的不計時診斷（decision 019 §7）：距離評估次數 D、recall 損失的分解、天花板與結構大小。

用法（10 份 C formal config 都完成之後；先通過 machine_quiet，再在 heavy lock 下、單執行緒執行）：
    .venv/bin/python -m benchmarks.machine_quiet --powermode 2      # 印出 yes 才開始（019 §5）
    ./tools/with-mac-lock env VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m benchmarks.quant_diag --dataset <glove|text> \\
        <10 份 C formal config 的 run 目錄（兩個資料集）…> [--excluded <已整份重跑的失敗 run>…]

- 019 §7：診斷在全部 10 份 C formal config 完成之後才執行。所以兩個資料集的 run 都要列出，先通過與 C 報告相同的合併前檢查
  （`quant_report.load_formal_set`），只診斷 `--dataset` 指定的那一個資料集。

- 每個 seed（Rust 1–5）以 `QuantHnswIndex.build_pair` 建一次圖；F 與 S 在 C 的每個 ef 執行 `search_stats`（k = 10，得到 D、V）
  與 k = ef 的搜尋（得到 W）。兩者的前 10 個 ids 必須與 formal run 同一個 (seed, 臂, ef) 的 `ids_k10` 逐位元相同，
  診斷才與 formal 綁定；不符的 (seed, 臂) 在 H1、H2 中無效（019 §2、§7），照實列出。
- T_X：把 W 以 f64 的精確距離（登記的 unit vectors，1 − q·x）重排，tie-break 依 001 的 (distance, id)，取前 10 個的 Recall@10。
  numpy 的重排分塊進行，不建立 query 數 × n 的大矩陣。
- FAISS（seeds 1–3）：以 Q adapter 的同一段建圖程式建 Flat 並移植，在 `del flat` 之前核對 graph（C8）；
  重建的 Q 在 e* 的 k = 10 ids 必須與 formal 相同（否則該 seed 的 C8 無法判定）。
- 天花板 c：S 的 `exact_search` 對 frozen ground truth 的 Recall@10 / Recall@1；c_FAISS 以 `IndexScalarQuantizer` 做 brute force，
  只描述。brute force 只作分析基準（CLAUDE.md），不作效能比較。
- 證據條件（比照 015 §2）：git 乾淨、binding 晚於 HEAD、`gate_profile.PROFILED_PATHS` 與 C formal run 的 commit 沒有差異、
  BUILD_INFO 與套件版本相同、由祖先持有 heavy lock、`VECLIB_MAXIMUM_THREADS=1`。輸出 `results/quant/diag-<glove|text>.json`，拒絕覆寫。
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from benchmarks import adapters, dataprep, environment, gate_profile, profile, quant_report
from benchmarks.metrics import recall_at_k
from benchmarks.quant_report import (
    E_STAR,
    EF_CONSTRUCTION,
    FAISS_SEEDS,
    RUST_SEEDS,
    F,
    K,
    M,
    N,
    P,
    Q,
    S,
    decompose,
)

CHUNK_BYTES = 64 << 20  # 重排時每塊 f64 候選向量的上限
VECLIB_ENV = "VECLIB_MAXIMUM_THREADS"


# ---- 重排（T_X） ----


def rerank_top(candidates: np.ndarray, data: np.ndarray, queries: np.ndarray, k: int = K,
               chunk_bytes: int = CHUNK_BYTES) -> np.ndarray:
    """每個 query 的候選（`candidates`，-1 為 padding）以 f64 的 1 − q·x 重排，tie-break 依 (distance, id)，回傳前 k 個 id。

    `data` 與 `queries` 是登記的 unit vectors（只把用到的列轉成 f64）。候選不足 k 個時以 -1 補齊。
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    nq, width = candidates.shape
    out = np.full((nq, k), -1, dtype=np.int64)
    rows = max(1, chunk_bytes // max(1, width * data.shape[1] * 8))
    for start in range(0, nq, rows):
        block = candidates[start:start + rows]
        valid = block >= 0
        safe = np.where(valid, block, 0)
        vectors = np.asarray(data[safe], dtype=np.float64)
        q = np.asarray(queries[start:start + rows], dtype=np.float64)
        distance = 1.0 - np.einsum("rcd,rd->rc", vectors, q)
        distance[~valid] = np.inf
        ids = np.where(valid, block, np.iinfo(np.int64).max)
        order = np.lexsort((ids, distance), axis=-1)[:, :k]
        top = np.take_along_axis(block, order, axis=1)
        top[~np.take_along_axis(valid, order, axis=1)] = -1
        out[start:start + rows, :top.shape[1]] = top
    return out


def rerank_recall(candidates, data, queries, gt, k: int = K) -> float:
    """T_X（019 §2）：重排之後前 k 個的 id-based Recall@k。"""
    return recall_at_k(rerank_top(candidates, data, queries, k), gt, k)


def duplicate_code_fraction(codes: np.ndarray) -> float:
    """codes 至少與另一列完全相同的列比例（019 對實驗的影響：codes 相同的向量在 SQ8 距離上變成 tie；只描述）。"""
    rows = np.ascontiguousarray(codes).view(np.dtype((np.void, codes.shape[1] * codes.itemsize))).ravel()
    _, inverse, counts = np.unique(rows, return_inverse=True, return_counts=True)
    return float((counts[inverse] > 1).sum() / codes.shape[0])


# ---- formal run ----


def formal_points(labelled) -> tuple[dict, dict]:
    """(實作, seed, ef) → formal 的 `ids_k10`（沒有時 None）與 Recall@10（`k10.recall`；incomplete 時 None）。"""
    ids, recalls = {}, {}
    for label, record in labelled:
        for child in record["children"]:
            build = child["build"]
            if not quant_report._in_cell(build):
                continue
            for point in child["points"]:
                key = (child["implementation"], build["seed"], point["ef_search"])
                path = Path(label) / point["file"]
                if not path.exists():
                    raise SystemExit(f"{path} is listed in {label}/run.json but missing")
                with np.load(path) as arrays:
                    ids[key] = np.asarray(arrays["ids_k10"]) if "ids_k10" in arrays.files else None
                recalls[key] = point["k10"]["recall"]
    return ids, recalls


def _same(a, b) -> bool:
    return a is not None and b is not None and a.shape == b.shape and bool(np.array_equal(a, b))


# ---- Rust（F、S） ----


def rust_seed(data, queries, gt, seed: int, efs, formal_ids: dict, formal_recalls: dict) -> tuple[dict, object]:
    """一個 seed 的 F、S 診斷；另外回傳 S 的 index（天花板與 codes 比對用）。"""
    import rust_hnsw

    fp32, sq8 = rust_hnsw.QuantHnswIndex.build_pair(data, M, EF_CONSTRUCTION, seed)
    arms = {}
    for arm, index in ((F, fp32), (S, sq8)):
        points = {}
        for ef in efs:
            ids10, _, kernel_calls, visited = index.search_stats(queries, K, ef)
            # W_X（019 §2）：k = ef 的完整候選表；它的前 10 個就是 k = 10 的結果，兩者都與 formal 核對。
            candidates, _ = index.search(queries, ef, ef)
            formal = formal_ids.get((arm, seed, ef))
            stats_match, w_match = _same(np.asarray(ids10, np.int64), formal), _same(np.asarray(candidates[:, :K], np.int64), formal)
            points[str(ef)] = {
                "ef_search": ef,
                "w_width": int(candidates.shape[1]),
                "kernel_calls_mean": float(np.mean(kernel_calls)),
                "visited_inserts_mean": float(np.mean(visited)),
                "w_prefix_matches_stats": bool(np.array_equal(candidates[:, :K], ids10)),
                "stats_ids_match": None if formal is None else stats_match,
                "w_prefix_ids_match": None if formal is None else w_match,
                "formal_ids_match": None if formal is None else stats_match and w_match,
                "mismatched_queries": None if formal is None or formal.shape != ids10.shape else
                int((np.asarray(ids10) != formal).any(axis=1).sum()),
                "rerank_recall10": rerank_recall(candidates, data, queries, gt),
                "formal_recall10": formal_recalls.get((arm, seed, ef)),
            }
        matches = [p["formal_ids_match"] for p in points.values()]
        arms[arm] = {"ids_valid": False not in matches and True in matches,
                     "mismatched_ef": [p["ef_search"] for p in points.values() if p["formal_ids_match"] is False],
                     "missing_formal_ef": [p["ef_search"] for p in points.values() if p["formal_ids_match"] is None],
                     "points": points}
    codes = sq8.codes()
    entry = {"arms": arms,
             "memory_accounting": {F: dict(fp32.memory_accounting()), S: dict(sq8.memory_accounting())},
             "codes_sha256": hashlib.sha256(np.ascontiguousarray(codes).tobytes()).hexdigest()}
    del fp32
    return entry, sq8


def decompositions(rust: dict) -> list[dict]:
    """019 §2 的 L_total、L_trav、L_rank、δ_F（float；判定由 quant_report 以精確的分數重算）。"""
    out = []
    for seed, entry in rust.items():
        f, s = entry["arms"][F], entry["arms"][S]
        for ef, point_f in f["points"].items():
            point_s = s["points"][ef]
            values = (point_f["formal_recall10"], point_s["formal_recall10"], point_f["rerank_recall10"], point_s["rerank_recall10"])
            if None in values:
                continue
            out.append({"seed": int(seed), "ef_search": int(ef), "ids_valid": f["ids_valid"] and s["ids_valid"],
                        **decompose(*values)})
    return out


# ---- FAISS（P、Q） ----


def faiss_graph_differences(flat_hnsw, quant_hnsw) -> list[str]:
    """C8：移植後 Q 的 `hnsw.levels`、`offsets`、`neighbors`、`entry_point`、`max_level` 是否與 Flat 完全相同。"""
    import faiss

    differences = []
    for name in ("levels", "offsets", "neighbors"):
        a, b = faiss.vector_to_array(getattr(flat_hnsw, name)), faiss.vector_to_array(getattr(quant_hnsw, name))
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            differences.append(name)
    for name in ("entry_point", "max_level"):
        if getattr(flat_hnsw, name) != getattr(quant_hnsw, name):
            differences.append(name)
    return differences


def faiss_struct_bytes(index, codebook: bool) -> dict:
    """019 §7 的 FAISS 陣列大小：storage codes、`hnsw.neighbors`、`offsets`、`levels`（Q 另列 codebook）。不複製陣列。"""
    import faiss

    storage = faiss.downcast_index(index.storage)
    out = {"codes_bytes": storage.codes.size(), "neighbors_bytes": index.hnsw.neighbors.size() * 4,
           "offsets_bytes": index.hnsw.offsets.size() * 8, "levels_bytes": index.hnsw.levels.size() * 4}
    if codebook:
        out["codebook_bytes"] = storage.sq.trained.size() * 4
    out["total_bytes"] = sum(out.values())
    return out


def faiss_seed(data, queries, seed: int, e_star: int, formal_q_ids, s_codes, s_codebook) -> dict:
    import faiss

    faiss.omp_set_num_threads(1)
    flat = adapters.faiss_hnsw_flat(data, M, EF_CONSTRUCTION, seed)
    index = adapters.faiss_sq8_from_flat(flat, data, M)
    differences = faiss_graph_differences(flat.hnsw, index.hnsw)  # del flat 之前（019 §7）
    sizes = {P: faiss_struct_bytes(flat, codebook=False), Q: faiss_struct_bytes(index, codebook=True)}
    del flat
    index.hnsw.efSearch = e_star
    faiss.omp_set_num_threads(1)
    _, ids = index.search(queries, K)
    ids = np.asarray(ids, dtype=np.int64)
    storage = faiss.downcast_index(index.storage)
    n, d = data.shape
    q_codes = faiss.vector_to_array(storage.codes).reshape(n, d)
    trained = faiss.vector_to_array(storage.sq.trained)
    vmin, vdiff = s_codebook
    return {"graph_identical": not differences, "graph_differences": differences,
            "ids_match_at_e_star": None if formal_q_ids is None else _same(ids, formal_q_ids),
            "codes_agreement": float((q_codes == s_codes).mean()),
            "codebook_identical": trained[:d].tobytes() == np.asarray(vmin).tobytes()
            and trained[d:].tobytes() == np.asarray(vdiff).tobytes(),
            "struct_bytes": sizes}


# ---- 天花板 ----


def ceiling(sq8_index, data, queries, gt) -> dict:
    """c（S 的 exact_search，同一個 kernel、(distance, id) tie-break）與 c_FAISS（只描述）。"""
    import faiss

    ids, _ = sq8_index.exact_search(queries, K)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexScalarQuantizer(data.shape[1], faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    index.sq.rangestat = faiss.ScalarQuantizer.RS_minmax
    index.sq.rangestat_arg = 0
    index.train(data)
    index.add(data)
    _, faiss_ids = index.search(queries, K)
    return {"recall10": recall_at_k(ids, gt, K), "recall1": recall_at_k(ids, gt, 1),
            "faiss_recall10": recall_at_k(faiss_ids, gt, K), "faiss_recall1": recall_at_k(faiss_ids, gt, 1)}


# ---- 全部 ----


def diagnose(dataset: str, labelled, data, queries, gt, rust_seeds=RUST_SEEDS, faiss_seeds=FAISS_SEEDS) -> dict:
    """§7 的全部診斷（不檢查執行環境；main 先檢查證據條件）。`labelled`：[(run 目錄, run 紀錄)]。"""
    records = [record for _, record in labelled]
    efs = sorted(records[0]["config"]["parsed"]["ef_search"])
    e_star = E_STAR[dataset]
    formal_ids, formal_recalls = formal_points(labelled)
    rust, kept = {}, None
    for seed in rust_seeds:
        entry, sq8 = rust_seed(data, queries, gt, seed, efs, formal_ids, formal_recalls)
        rust[str(seed)] = entry
        if kept is None:
            kept = sq8  # codes 與 seed 無關（同一份資料訓練），天花板只算一次
        else:
            del sq8
    codes = kept.codes()
    codebook = kept.codebook()
    shas = {entry["codes_sha256"] for entry in rust.values()}
    faiss_part = {str(seed): faiss_seed(data, queries, seed, e_star, formal_ids.get((Q, seed, e_star)), codes, codebook)
                  for seed in faiss_seeds}
    return {
        "tool": "benchmarks.quant_diag", "decision": "019 §7", "dataset": dataset, "n": N, "m": M,
        "ef_construction": EF_CONSTRUCTION, "e_star": e_star, "ef_search": efs, "k": K,
        "formal_runs": [{"run": str(label), "commit": record["metadata"]["git"]["commit"],
                         "config": (record.get("config") or {}).get("path")} for label, record in labelled],
        "rust": rust, "decomposition": decompositions(rust), "faiss": faiss_part,
        "ceiling": ceiling(kept, data, queries, gt),
        "codes": {"sha256": sorted(shas)[0] if len(shas) == 1 else sorted(shas), "identical_across_seeds": len(shas) == 1,
                  "duplicate_row_fraction": duplicate_code_fraction(codes)},
    }


def evidence_conditions(records: list[dict]) -> dict:
    """019 §7 的證據條件；不符就拒絕（SystemExit），不寫任何輸出。"""
    import rust_hnsw

    if os.environ.get(VECLIB_ENV) != "1":
        raise SystemExit(f"run with {VECLIB_ENV}=1 (single-threaded diagnostics, 019 §7)")
    try:
        owner = environment.check_lock()
        code = profile.check_code_state()
    except environment.PreflightError as err:
        raise SystemExit(f"quant_diag needs the heavy lock, a clean checkout and a fresh binding (019 §7): {err}") from None
    for commit in sorted({r["metadata"]["git"]["commit"] for r in records}):
        changed = gate_profile.changed_profiled_files(commit, code["commit"])
        if changed:
            raise SystemExit(f"profiled code differs from the C formal commit {commit[:12]}: {changed} (019 §7)")
    if any(dict(rust_hnsw.BUILD_INFO) != r["metadata"]["build_info"] for r in records):
        raise SystemExit("rust_hnsw.BUILD_INFO differs from the C formal runs (019 §7)")
    versions = {name: importlib.metadata.version(name) for name in gate_profile.COMPARED_VERSIONS}
    differing = sorted({name for r in records for name in gate_profile.COMPARED_VERSIONS
                        if versions[name] != r["metadata"]["versions"].get(name)})
    if differing:
        raise SystemExit(f"package versions differ from the C formal runs: {differing} (019 §7)")
    return {"lock_owner": owner, "commit": code["commit"], VECLIB_ENV: os.environ[VECLIB_ENV], "versions": versions}


def main(argv=None) -> int:
    import rust_hnsw

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="10 份 C formal config 的 run 目錄（兩個資料集都要列出，019 §7）")
    parser.add_argument("--dataset", required=True, type=quant_report.dataset_name, help="要診斷的資料集（glove 或 text）")
    parser.add_argument("--excluded", action="append", default=[], help="已整份重跑的失敗 run（與 quant_report 相同）")
    parser.add_argument("--out", help="輸出路徑；預設為 results/quant/diag-<glove|text>.json")
    args = parser.parse_args(argv)
    dataset = args.dataset
    out = Path(args.out) if args.out else quant_report.diag_path(dataset)
    if out.exists():
        raise SystemExit(f"{out} exists; an earlier diagnosis must not be overwritten (019 §7)")
    try:
        groups = quant_report.load_formal_set(args.runs, excluded=args.excluded)
    except quant_report.ReportError as err:
        raise SystemExit(f"the given runs are not the complete set of the 10 C formal configs (019 §7): {err}") from None
    labelled = groups[dataset]
    conditions = evidence_conditions([record for _, record in labelled])
    data, queries, gt, dataset_record = profile.load_dataset(dataset, N)
    record = diagnose(dataset, labelled, data, queries, gt)
    record.update(evidence_conditions=conditions, git=dataprep.git_state(), build_info=dict(rust_hnsw.BUILD_INFO),
                  dataset_record=dataset_record)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
