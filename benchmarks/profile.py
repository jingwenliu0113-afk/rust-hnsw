"""為 profiling 產生穩定的工作負載：透過同一層 adapter 重複執行 build 或 search。

只用來找出時間花在哪裡，不是 benchmark：不輸出任何 latency / QPS 數字。
要取樣的階段開始時，stdout 會印出 `READY <pid>`；`scripts/profile.sh` 以此時機開始取樣。

用法：
    python -m benchmarks.profile --impl rust-hnsw --phase search --seconds 10 \\
        (--dataset NAME --n N | --data X.npy --queries Q.npy | --synthetic N,DIM,NQ) \\
        --m 16 --ef-construction 100 --seed 1 --ef 100 --k 10 [--meta out.json]

作為 Gate 證據時（decision 009）用 `--dataset`：與 runner 相同的登記向量（不重新正規化）與前 n 列子集，
並以 `--meta` 在取樣前寫出參數、資料集 checksum、git 狀態、binding 的編譯資訊、是否持有 heavy lock，
以及 search 階段在該 ef 的 Recall@10（id-based，003），用來確認各實作是在相近的 recall 下取樣。
"""

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks import dataprep, datasets, environment, metrics, protocol
from benchmarks.adapters import FaissHnsw, Hnswlib, RustHnsw
from benchmarks.interface import prepare_vectors

ADAPTERS = {"rust-hnsw": RustHnsw, "hnswlib": Hnswlib, "faiss-hnsw": FaissHnsw}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--phase", required=True, choices=["build", "search"])
    parser.add_argument("--seconds", required=True, type=float)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="登記的資料集名稱（decision 011）；搭配 --n")
    source.add_argument("--data", help="train 向量的 .npy（float，2-D）")
    source.add_argument("--synthetic", help="N,DIM,NQ：只用於驗證工具本身")
    parser.add_argument("--n", type=int, help="搭配 --dataset：取 train 的前 n 列（與 runner 的子集相同，008 §4）")
    parser.add_argument("--queries", help="query 向量的 .npy（搭配 --data）")
    parser.add_argument("--meta", help="取樣前寫出 profiling 紀錄的 JSON 路徑")
    # 與 decision 007 一致：參數一律明確給定，沒有預設值。
    parser.add_argument("--m", required=True, type=int)
    parser.add_argument("--ef-construction", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--ef", type=int)
    parser.add_argument("--k", type=int)
    args = parser.parse_args(argv)
    if args.data and not args.queries and args.phase == "search":
        parser.error("--phase search with --data requires --queries")
    if args.phase == "search" and (args.ef is None or args.k is None):
        parser.error("--phase search requires --ef and --k")
    if (args.dataset is None) != (args.n is None):
        parser.error("--dataset and --n must be given together")
    return args


def load_dataset(name: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """登記資料集的前 n 列、全部 queries，以及 n 的 exact top-10（與 runner 相同的規則）。"""
    from benchmarks.run import _frozen_ground_truth, vectors_for_run

    dataset = datasets.REGISTRY[name]
    loaded = datasets.load(dataset)
    train = vectors_for_run(loaded.train, loaded.prepared)
    queries = vectors_for_run(loaded.queries, loaded.prepared)
    if not protocol.K <= n <= train.shape[0]:
        raise SystemExit(f"--n must satisfy {protocol.K} <= n <= {train.shape[0]}")
    data = np.ascontiguousarray(train[:n])
    digests = {f: environment.sha256_file(path) for f, path in datasets.files(dataset).items()}
    frozen = getattr(dataset, "ground_truth", {}).get(n)
    if frozen is not None:
        gt, gt_source = _frozen_ground_truth(dataset, frozen, data, queries, digests)
    else:
        import rust_hnsw

        exact = rust_hnsw.ExactIndex(data.shape[1])
        exact.add(data)
        gt, gt_source = exact.search(queries, protocol.K)[0], {"kind": "computed"}
    record = {"name": name, "n": n, "n_queries": int(queries.shape[0]), "dim": int(data.shape[1]),
              "files_sha256": digests, "gt_source": gt_source}
    return data, queries, gt, record


def lock_state() -> dict:
    try:
        return {"held_by_ancestor": True, "owner": environment.check_lock()}
    except environment.PreflightError as err:
        return {"held_by_ancestor": False, "reason": str(err)}


def evidence_conditions(args) -> dict | None:
    """作為 Gate 證據的取樣（--dataset 且祖先持有 heavy lock；009）適用 formal run 的條件（014）：
    電力條件不符就拒絕；背景 heavy workload 或 load 超過門檻就等待，最多 10 分鐘，仍不成立就拒絕。只驗證工具的取樣不檢查。"""
    if not args.dataset or not lock_state()["held_by_ancestor"]:
        return None
    power = environment.power_state()
    try:
        environment.check_formal_power(power)
    except environment.PreflightError as err:
        raise SystemExit(f"Gate profiling needs the formal power conditions (decision 014): {err}") from None
    ok, samples, background = environment.wait_for_load(
        lambda: os.getloadavg()[0], protocol.MAX_LOAD_1MIN, protocol.LOAD_WAIT_S, protocol.LOAD_POLL_S, time.sleep,
        time.monotonic, read_background=lambda: environment.background_processes(
            exclude=environment.ancestor_pids(os.getpid())))
    if not ok:
        raise SystemExit(f"load or background workload {[b['command'] for b in background]} did not clear "
                         f"within {protocol.LOAD_WAIT_S} s (decision 014)")
    return {"power": {key: power[key] for key in ("power_source", "powermode", "adapter", "adapter_watts", "battery")},
            "gate_samples": samples}


def profile_record(args, dataset_record, recall, conditions=None) -> dict:
    import rust_hnsw

    return {
        "evidence_conditions": conditions,
        "args": vars(args),
        "dataset": dataset_record,
        "recall_at_10": recall,
        "git": dataprep.git_state(),
        "build_info": dict(rust_hnsw.BUILD_INFO),
        "python": sys.version,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "hnswlib", "faiss-cpu")},
        "lock": lock_state(),
        "loadavg": os.getloadavg(),
        "machine": environment.machine_metadata(),
    }


def load_vectors(args):
    if args.synthetic:
        n, dim, nq = (int(x) for x in args.synthetic.split(","))
        rng = np.random.default_rng(0)
        return prepare_vectors(rng.standard_normal((n, dim))), prepare_vectors(rng.standard_normal((nq, dim)))
    data = prepare_vectors(np.load(args.data))
    queries = prepare_vectors(np.load(args.queries)) if args.queries else None
    return data, queries


def main(argv=None):
    args = parse_args(argv)
    gt, dataset_record = None, None
    if args.dataset:
        data, queries, gt, dataset_record = load_dataset(args.dataset, args.n)
    else:
        data, queries = load_vectors(args)

    def new_adapter():
        return ADAPTERS[args.impl](m=args.m, ef_construction=args.ef_construction, seed=args.seed)

    if args.phase == "build":
        conditions = evidence_conditions(args)
        if args.meta:
            Path(args.meta).write_text(json.dumps(profile_record(args, dataset_record, None, conditions), indent=2) + "\n")
        print(f"READY {os.getpid()}", flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            new_adapter().build(data)
    else:
        adapter = new_adapter()
        adapter.build(data)
        adapter.set_ef_search(args.ef)
        recall = None
        if gt is not None:
            # 不計時；只用來確認取樣時的 recall 區間（009：相近的 Recall@K）。
            recall = metrics.recall_at_k(adapter.search(queries, protocol.K)[0], gt, protocol.K)
        conditions = evidence_conditions(args)
        if args.meta:
            Path(args.meta).write_text(json.dumps(profile_record(args, dataset_record, recall, conditions), indent=2) + "\n")
        print(f"READY {os.getpid()}", flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            adapter.search(queries, args.k)


if __name__ == "__main__":
    main()
