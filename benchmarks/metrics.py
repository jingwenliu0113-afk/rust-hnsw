"""Recall 指標（decision 003）。

- `recall_at_k`：主要指標，id-based。
- `tie_aware_recall_at_k`：次要診斷，量化 tie（重複向量）對 id-based recall 的影響。
"""

import numpy as np

# tie-aware 判定的 cosine distance 絕對容忍值，只吸收不同 f64 計算路徑的 rounding 差異。
TIE_EPS = 1e-6


def _check_ids(found_ids, true_ids, k: int) -> tuple[np.ndarray, np.ndarray]:
    found = np.asarray(found_ids)
    true = np.asarray(true_ids)
    if found.ndim != 2 or true.ndim != 2:
        raise ValueError("found_ids and true_ids must be 2-D")
    if found.shape[0] != true.shape[0]:
        raise ValueError(f"number of queries differs: {found.shape[0]} vs {true.shape[0]}")
    if not 1 <= k <= min(found.shape[1], true.shape[1]):
        raise ValueError(f"k must satisfy 1 <= k <= {min(found.shape[1], true.shape[1])}, got {k}")
    if (true[:, :k] < 0).any():
        raise ValueError("ground truth must not contain padding (-1)")
    return found[:, :k], true[:, :k]


def recall_at_k(found_ids, true_ids, k: int) -> float:
    """平均每個 query 的 `|返回前 k 個 id ∩ exact 前 k 個 id| / k`。`-1` 永遠不算命中。"""
    found, true = _check_ids(found_ids, true_ids, k)
    hits = (found[:, :, None] == true[:, None, :]).any(axis=2)
    return float(hits.sum() / (found.shape[0] * k))


def tie_aware_recall_at_k(found_ids, true_ids, data, queries, k: int, eps: float = TIE_EPS) -> float:
    """返回項的 exact 距離不超過「exact 第 k 近距離 + eps」就算命中。

    `data` 與 `queries` 必須是 `prepare_vectors` 的輸出（unit vectors）。
    所有距離都在這裡以 f64 重新計算，不使用實作回報的距離。
    """
    found, true = _check_ids(found_ids, true_ids, k)
    data64 = np.asarray(data, dtype=np.float64)
    queries64 = np.asarray(queries, dtype=np.float64)
    if queries64.shape[0] != found.shape[0]:
        raise ValueError(f"number of queries differs: {queries64.shape[0]} vs {found.shape[0]}")

    def exact_distance(column_ids: np.ndarray) -> np.ndarray:
        return 1.0 - np.einsum("ij,ij->i", data64[column_ids], queries64)

    threshold = exact_distance(true[:, k - 1]) + eps
    hits = 0
    for j in range(k):
        column = found[:, j]
        valid = column >= 0
        distance = exact_distance(np.where(valid, column, 0))
        hits += int((valid & (distance <= threshold)).sum())
    return float(hits / (found.shape[0] * k))
