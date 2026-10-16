"""rust_hnsw 的快速上手 Demo：建 index、批次搜尋、以 exact search 算 Recall@10、存檔與讀回。

只示範 API。資料是隨機產生的 synthetic 向量，印出的數字**不是研究結果**，不得用於任何結論
（正式結果見 README 的「正式結果」各節與 results/reports/）。

用法（先 `./scripts/test.sh` 或 `maturin develop --release` 安裝 binding）：
    .venv/bin/python examples/quickstart.py [--n 20000] [--dim 64] [--queries 200]
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

import rust_hnsw


def unit_rows(x: np.ndarray) -> np.ndarray:
    """cosine 距離的 index 要求輸入為 unit vectors（decision 001）。"""
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    """id-based Recall@k（decision 003）：每個 query 的前 k 個結果中，命中 exact 前 k 個的比例；-1 不算命中。"""
    hits = sum(len(set(f[:k][f[:k] >= 0]) & set(t[:k])) for f, t in zip(found, truth))
    return hits / (len(found) * k)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(0)
    data = unit_rows(rng.standard_normal((args.n, args.dim)))
    queries = unit_rows(rng.standard_normal((args.queries, args.dim)))

    # 1. exact search：ground truth（brute force 只作 ground truth 與 correctness baseline）。
    exact = rust_hnsw.ExactIndex(args.dim)
    exact.add(data)
    truth, _ = exact.search(queries, args.k)

    # 2. HNSW：參數一律明確給定。
    index = rust_hnsw.HnswIndex(args.dim, m=16, ef_construction=200, seed=1)
    start = time.perf_counter()
    index.add(data)
    print(f"built {len(index)} vectors (dim {index.dim}, max level {index.max_level}) "
          f"in {time.perf_counter() - start:.2f} s")

    # 3. 批次搜尋：ef_search 越大，recall 越高、越慢。
    results = {}
    for ef in (16, 64, 256):
        ids, distances = index.search(queries, args.k, ef)
        results[ef] = recall_at_k(ids, truth, args.k)
        print(f"ef_search={ef:4d}  Recall@{args.k}={results[ef]:.3f}  nearest distance of query 0={distances[0, 0]:.4f}")

    # 4. 存檔與讀回：讀回的 index 給出完全相同的結果。
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.hnsw"
        index.save(path)
        loaded = rust_hnsw.HnswIndex.load(path)
        same = all(np.array_equal(a, b) for a, b in zip(index.search(queries, args.k, 64), loaded.search(queries, args.k, 64)))
        print(f"saved {path.stat().st_size} bytes; reloaded index gives identical results: {same}")

    print("（synthetic 資料的 Demo，數字不是研究結果。）")
    return {"recall": results, "reload_identical": same}


if __name__ == "__main__":
    main()
