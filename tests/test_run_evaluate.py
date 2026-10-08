"""runner 的協定常數與父 process 的指標計算（decision 008 §4、§6、§7、§10）。"""

import numpy as np
import pytest

from benchmarks import evaluate, protocol
from benchmarks.interface import prepare_vectors
from benchmarks.metrics import TIE_EPS
from helpers import random_unit, tie_data


def test_protocol_constants_match_decision_008():
    """改動任何一項都必須另寫 decision log（008「對實驗的影響」）。"""
    assert protocol.protocol_record() == {
        "version": "008-v1",
        "k": 10,
        "recall_at_1_k": 1,
        "batch_warmup_calls": 1,
        "qps_repeats": 5,
        "latency_warmup_queries": 100,
        "latency_repeats": 3,
        "min_queries": 100,
        "library_warmup_rows": 16,
        "null_call": {"n": 1, "k": 1, "ef": 1, "warmup": 100, "samples": 1000},
        "python_loop": {"samples": 1000, "mean_calls": 1_000_000},
        "percentile_method": "linear",
        "gc": "gc.collect() then gc.disable() around the QPS and latency sections",
        "max_load_1min": 4.0,
        "load_wait_s": 600,
        "load_poll_s": 15,
        "single_thread_tolerance": {"relative": 0.02, "absolute_s": 0.001},
        "memory_tolerance": {"relative": 0.02, "absolute_bytes": 1 << 20},
        "tie_eps": TIE_EPS,
    }


# ---- id 映射（004、008 §4） ----


def test_map_ids_applies_the_permutation():
    perm = np.array([2, 0, 1])
    ids = np.array([[0, 1], [2, 0]])
    np.testing.assert_array_equal(evaluate.map_ids(ids, perm), [[2, 0], [1, 2]])


def test_map_ids_keeps_padding():
    """`perm[ids]` 會把 -1 當成最後一個元素（這裡是 1），padding 就變成合法 id。"""
    perm = np.array([2, 0, 1])
    ids = np.array([[0, -1], [-1, -1]])
    np.testing.assert_array_equal(evaluate.map_ids(ids, perm), [[2, -1], [-1, -1]])


def test_map_ids_without_permutation_is_identity():
    ids = np.array([[3, -1]])
    np.testing.assert_array_equal(evaluate.map_ids(ids, None), ids)


# ---- 資料集檢查（003、004、008 §4） ----


def test_duplicate_row_fraction_counts_rows_that_have_an_identical_copy():
    a, b, c = [1.0, 0.0], [0.0, 1.0], [0.6, 0.8]
    x = np.array([a, a, b, c, c, c], dtype=np.float32)
    assert evaluate.duplicate_row_fraction(x) == pytest.approx(5 / 6)
    assert evaluate.duplicate_row_fraction(random_unit(1, 100, 8)) == 0.0


def test_scalar_multiples_become_duplicates_after_preparation():
    x = prepare_vectors(np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 1.0]]))
    assert evaluate.duplicate_row_fraction(x) == pytest.approx(2 / 3)


def test_rank1_tie_queries_uses_exact_f64_distances():
    data = tie_data()
    queries = prepare_vectors([[1.0, 0.0], [0.6, 0.8]])
    gt = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]])
    # 兩個 query 的第 1、2 名都是完全相同的向量。
    assert evaluate.rank1_tie_queries(data, queries, gt) == 2

    data = random_unit(2, 200, 8)
    queries = random_unit(3, 20, 8)
    order = np.argsort(1.0 - queries.astype(np.float64) @ data.astype(np.float64).T, axis=1, kind="stable")
    assert evaluate.rank1_tie_queries(data, queries, order[:, :10]) == 0


# ---- QPS 與 latency 的彙整 ----


def test_qps_summary():
    summary = evaluate.qps_summary(np.array([1e9, 2e9, 5e8, 1e9, 1e9]), num_queries=1000)
    assert summary["qps"] == [1000.0, 500.0, 2000.0, 1000.0, 1000.0]
    assert summary["qps_median"] == 1000.0


def test_latency_summary_uses_linear_percentiles_over_pooled_and_per_repeat_samples():
    rng = np.random.default_rng(0)
    samples = rng.integers(1_000, 100_000, size=(3, 101))
    summary = evaluate.latency_summary(samples)
    assert summary["p50_ns"] == np.percentile(samples, 50, method="linear")
    assert summary["p95_ns"] == np.percentile(samples, 95, method="linear")
    assert summary["per_repeat_p50_ns"] == [float(np.percentile(r, 50, method="linear")) for r in samples]
    assert summary["per_repeat_p95_ns"] == [float(np.percentile(r, 95, method="linear")) for r in samples]


def test_p50_of_an_even_sample_is_interpolated():
    """method 必須固定：nearest-rank 在這裡會得到 2 或 3，linear 是 2.5。"""
    samples = np.array([[1, 2, 3, 4]])
    assert evaluate.latency_summary(samples)["p50_ns"] == 2.5
