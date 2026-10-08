"""為 profiling 產生穩定的工作負載：透過同一層 adapter 重複執行 build 或 search。

只用來找出時間花在哪裡，不是 benchmark：不輸出任何 latency / QPS 數字。
要取樣的階段開始時，stdout 會印出 `READY <pid>`；`scripts/profile.sh` 以此時機開始取樣。

用法：
    python -m benchmarks.profile --impl rust-hnsw --phase search --seconds 10 \\
        (--data X.npy --queries Q.npy | --synthetic N,DIM,NQ) \\
        --m 16 --ef-construction 100 --seed 1 --ef 100 --k 10
"""

import argparse
import os
import time

import numpy as np

from benchmarks.adapters import FaissHnsw, Hnswlib, RustHnsw
from benchmarks.interface import prepare_vectors

ADAPTERS = {"rust-hnsw": RustHnsw, "hnswlib": Hnswlib, "faiss-hnsw": FaissHnsw}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--phase", required=True, choices=["build", "search"])
    parser.add_argument("--seconds", required=True, type=float)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="train 向量的 .npy（float，2-D）")
    source.add_argument("--synthetic", help="N,DIM,NQ：只用於驗證工具本身")
    parser.add_argument("--queries", help="query 向量的 .npy（搭配 --data）")
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
    return args


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
    data, queries = load_vectors(args)

    def new_adapter():
        return ADAPTERS[args.impl](m=args.m, ef_construction=args.ef_construction, seed=args.seed)

    if args.phase == "build":
        print(f"READY {os.getpid()}", flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            new_adapter().build(data)
    else:
        adapter = new_adapter()
        adapter.build(data)
        adapter.set_ef_search(args.ef)
        print(f"READY {os.getpid()}", flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            adapter.search(queries, args.k)


if __name__ == "__main__":
    main()
