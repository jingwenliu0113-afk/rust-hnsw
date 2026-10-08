"""父 process 在子 process 結束後計算的指標與資料集檢查（decision 008 §4、§6、§7）。

這些計算都不在子 process 內做：tie-aware recall 的 f64 複本會主導 peak RSS（008 §8）。
"""

import numpy as np

from benchmarks.metrics import TIE_EPS, recall_at_k, tie_aware_recall_at_k
from benchmarks.protocol import PERCENTILE_METHOD


def map_ids(ids: np.ndarray, perm: np.ndarray | None) -> np.ndarray:
    """把打亂順序後的 id 映射回子集的列號；padding 的 -1 保持 -1。

    不能寫成 `perm[ids]`：numpy 會把 -1 當成最後一個元素，padding 就變成合法 id。
    """
    ids = np.asarray(ids, dtype=np.int64)
    if perm is None:
        return ids.copy()
    return np.where(ids >= 0, np.asarray(perm)[np.maximum(ids, 0)], -1)


def duplicate_row_fraction(x: np.ndarray) -> float:
    """至少與另一列 f32 bit pattern 完全相同的列比例（在 `prepare_vectors` 之後計算，003、004）。"""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    _, inverse, counts = np.unique(bits, axis=0, return_inverse=True, return_counts=True)
    return float((counts[inverse.reshape(-1)] > 1).sum() / x.shape[0])


def rank1_tie_queries(data: np.ndarray, queries: np.ndarray, gt: np.ndarray, eps: float = TIE_EPS) -> int:
    """exact 第 1 名與第 2 名的 f64 距離差 ≤ eps 的 query 數（rank 1 有 tie，008 §4）。"""
    data64 = np.asarray(data, dtype=np.float64)
    queries64 = np.asarray(queries, dtype=np.float64)
    first = 1.0 - np.einsum("ij,ij->i", data64[gt[:, 0]], queries64)
    second = 1.0 - np.einsum("ij,ij->i", data64[gt[:, 1]], queries64)
    return int((second - first <= eps).sum())


def qps_summary(qps_ns: np.ndarray, num_queries: int) -> dict:
    qps = [float(num_queries / (ns / 1e9)) for ns in np.asarray(qps_ns, dtype=np.float64)]
    return {"qps": qps, "qps_median": float(np.median(qps))}


def _percentile(x, q) -> float:
    return float(np.percentile(x, q, method=PERCENTILE_METHOD))


def latency_summary(latency_ns: np.ndarray) -> dict:
    """P50 / P95：合併所有重跑的樣本，以及每次重跑各自的值。"""
    samples = np.asarray(latency_ns)
    return {
        "p50_ns": _percentile(samples, 50),
        "p95_ns": _percentile(samples, 95),
        "per_repeat_p50_ns": [_percentile(row, 50) for row in samples],
        "per_repeat_p95_ns": [_percentile(row, 95) for row in samples],
    }


def sample_summary(samples_ns: np.ndarray) -> dict:
    return {"p50_ns": _percentile(samples_ns, 50), "p95_ns": _percentile(samples_ns, 95)}


def k_metrics(ids: np.ndarray, gt: np.ndarray, data: np.ndarray, queries: np.ndarray, k: int) -> dict:
    return {
        "recall": recall_at_k(ids, gt, k),
        "tie_aware_recall": tie_aware_recall_at_k(ids, gt, data, queries, k),
        "incomplete_queries": int((ids[:, :k] < 0).any(axis=1).sum()),
    }
