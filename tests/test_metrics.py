"""Recall 定義的人工可驗證測試（decision 003）。"""

import numpy as np
import pytest

from benchmarks.interface import prepare_vectors
from benchmarks.metrics import TIE_EPS, recall_at_k, tie_aware_recall_at_k


def ids(rows):
    return np.asarray(rows, dtype=np.int64)


def unit(rows):
    x = np.asarray(rows, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def circle(degrees):
    rad = np.deg2rad(degrees)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32)


# ---- id-based Recall@K（主要指標） ----


def test_perfect_recall():
    true = ids([[0, 1, 2], [3, 4, 5]])
    assert recall_at_k(true, true, 3) == 1.0


def test_partial_recall_is_averaged_over_queries():
    found = ids([[0, 1, 2], [5, 6, 7]])
    true = ids([[0, 1, 3], [7, 8, 9]])
    # query 0 命中 {0, 1} → 2/3；query 1 命中 {7} → 1/3；平均 0.5
    assert recall_at_k(found, true, 3) == pytest.approx(0.5)


def test_order_within_top_k_does_not_matter():
    assert recall_at_k(ids([[2, 1, 0]]), ids([[0, 1, 2]]), 3) == 1.0


def test_only_first_k_columns_are_used():
    found = ids([[0, 8, 9, 1]])
    true = ids([[0, 1, 2, 3]])
    assert recall_at_k(found, true, 1) == 1.0
    assert recall_at_k(found, true, 2) == pytest.approx(1 / 2)
    assert recall_at_k(found, true, 4) == pytest.approx(2 / 4)


def test_recall_at_1_example():
    found = ids([[4, 0], [1, 2], [7, 3]])
    true = ids([[4, 9], [2, 1], [7, 3]])
    # top-1：4 = 4 命中、1 ≠ 2 未命中、7 = 7 命中
    assert recall_at_k(found, true, 1) == pytest.approx(2 / 3)


def test_padding_never_counts_as_hit():
    assert recall_at_k(ids([[0, -1, -1]]), ids([[0, 1, 2]]), 3) == pytest.approx(1 / 3)


def test_rejects_invalid_k():
    with pytest.raises(ValueError, match="k"):
        recall_at_k(ids([[0, 1]]), ids([[0, 1, 2]]), 3)
    with pytest.raises(ValueError, match="k"):
        recall_at_k(ids([[0, 1]]), ids([[0, 1]]), 0)


def test_rejects_mismatched_query_counts():
    with pytest.raises(ValueError, match="queries"):
        recall_at_k(ids([[0], [1]]), ids([[0]]), 1)


def test_rejects_padding_in_ground_truth():
    with pytest.raises(ValueError, match="ground truth"):
        recall_at_k(ids([[0, 1]]), ids([[0, -1]]), 2)


# ---- tie-aware Recall@K（次要診斷） ----


def test_tie_aware_counts_equally_close_duplicate():
    # id 0 與 id 1 完全相同；ground truth 依 id tie-break 選 0。
    data = unit([[1, 0], [1, 0], [0, 1]])
    queries = unit([[1, 0]])
    found, true = ids([[1]]), ids([[0]])
    assert recall_at_k(found, true, 1) == 0.0
    assert tie_aware_recall_at_k(found, true, data, queries, 1) == 1.0


def test_tie_aware_uses_kth_true_distance_as_threshold():
    data = circle([0, 10, 20, 30])
    queries = circle([0])
    true = ids([[0, 1]])
    # id 2（20°）比第 2 近的 id 1（10°）遠 → 不算命中
    assert tie_aware_recall_at_k(ids([[0, 2]]), true, data, queries, 2) == pytest.approx(0.5)
    assert tie_aware_recall_at_k(ids([[1, 0]]), true, data, queries, 2) == 1.0
    # k = 1 時門檻是 0° 的距離，10° 的點不算
    assert tie_aware_recall_at_k(ids([[1, 0]]), true, data, queries, 1) == 0.0


def test_tie_aware_ignores_padding():
    data = circle([0, 10, 20])
    queries = circle([0])
    found = ids([[0, -1]])
    assert tie_aware_recall_at_k(found, ids([[0, 1]]), data, queries, 2) == pytest.approx(0.5)


def test_tie_aware_eps_is_applied():
    data = circle([0, 10, 90])
    queries = circle([0])
    found, true = ids([[2]]), ids([[0]])
    assert tie_aware_recall_at_k(found, true, data, queries, 1) == 0.0
    assert tie_aware_recall_at_k(found, true, data, queries, 1, eps=2.0) == 1.0


def test_tie_aware_rejects_mismatched_queries():
    data = circle([0, 10])
    with pytest.raises(ValueError, match="queries"):
        tie_aware_recall_at_k(ids([[0], [1]]), ids([[0], [1]]), data, circle([0]), 1)


def test_tie_aware_recall_converts_only_the_needed_rows_without_changing_results():
    """008 §8：不再把整份 data 轉成 f64；結果必須與舊的整份轉換完全相同。"""
    from benchmarks import datasets
    from benchmarks.evaluate import rank1_tie_queries

    loaded = datasets.load(datasets.REGISTRY["smoke-duplicates"])
    data, queries = prepare_vectors(loaded.train), prepare_vectors(loaded.queries)
    data64, queries64 = data.astype(np.float64), queries.astype(np.float64)
    order = np.argsort(1.0 - queries64 @ data64.T, axis=1, kind="stable")
    true = order[:, :10]
    found = np.roll(order[:, :10], 1, axis=1)

    def old_formula(k):
        threshold = 1.0 - np.einsum("ij,ij->i", data64[true[:, k - 1]], queries64) + TIE_EPS
        hits = sum(int((1.0 - np.einsum("ij,ij->i", data64[found[:, j]], queries64) <= threshold).sum()) for j in range(k))
        return hits / (found.shape[0] * k)

    for k in (1, 10):
        assert tie_aware_recall_at_k(found, true, data, queries, k) == old_formula(k)
    first = 1.0 - np.einsum("ij,ij->i", data64[true[:, 0]], queries64)
    second = 1.0 - np.einsum("ij,ij->i", data64[true[:, 1]], queries64)
    assert rank1_tie_queries(data, queries, true) == int((second - first <= TIE_EPS).sum())
