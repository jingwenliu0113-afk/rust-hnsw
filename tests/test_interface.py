"""統一 batch interface 的輸入準備與輸出驗證（decision 002）。"""

import numpy as np
import pytest

from benchmarks.interface import check_search_result, prepare_vectors

# ---- prepare_vectors ----


def test_prepare_vectors_returns_unit_float32_c_contiguous():
    x = prepare_vectors([[3.0, 4.0], [0.0, -2.0]])
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]
    assert x.shape == (2, 2)
    np.testing.assert_allclose(x, [[0.6, 0.8], [0.0, -1.0]], atol=1e-7)


def test_prepare_vectors_accepts_non_contiguous_input():
    base = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    x = prepare_vectors(base[:, ::2])
    assert x.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-6)


def test_prepare_vectors_does_not_modify_input():
    original = np.array([[3.0, 4.0]], dtype=np.float32)
    prepare_vectors(original)
    np.testing.assert_array_equal(original, [[3.0, 4.0]])


def test_prepare_vectors_handles_extreme_magnitudes():
    # f32 平方會 underflow / overflow，norm 必須用 f64 算。
    x = prepare_vectors(np.array([[1e-30, 1e-30], [1e30, 1e30]], dtype=np.float32))
    np.testing.assert_allclose(x, np.full((2, 2), np.sqrt(0.5)), rtol=1e-6)


@pytest.mark.parametrize(
    "bad, message",
    [
        (np.zeros((0, 3)), "empty"),
        (np.zeros((3, 0)), "empty"),
        (np.ones(3), "2-D"),
        ([[1.0, 2.0], [0.0, 0.0]], "row 1 is a zero vector"),
        ([[1.0, np.nan]], "row 0 contains NaN or Inf"),
        ([[1.0, 2.0], [np.inf, 0.0]], "row 1 contains NaN or Inf"),
        ([[1e300, 1.0]], "row 0 contains NaN or Inf"),  # 轉 float32 後 overflow
    ],
)
def test_prepare_vectors_rejects_invalid_input(bad, message):
    with pytest.raises(ValueError, match=message):
        prepare_vectors(bad)


# ---- check_search_result ----


def result(id_rows, dist_rows):
    return np.array(id_rows, dtype=np.int64), np.array(dist_rows, dtype=np.float32)


def test_check_search_result_accepts_valid_result():
    ids, dists = result([[3, 1, -1], [0, 2, 4]], [[0.1, 0.2, np.inf], [0.0, 0.0, 1.5]])
    check_search_result(ids, dists, num_queries=2, k=3)


def test_check_search_result_accepts_empty_batch():
    ids = np.zeros((0, 5), dtype=np.int64)
    dists = np.zeros((0, 5), dtype=np.float32)
    check_search_result(ids, dists, num_queries=0, k=5)


def test_check_search_result_allows_tiny_rounding_outside_range():
    ids, dists = result([[0, 1]], [[-1e-7, 2.0000001]])
    check_search_result(ids, dists, num_queries=1, k=2)


@pytest.mark.parametrize(
    "ids, dists, message",
    [
        (np.array([[0, 1]], dtype=np.int32), np.array([[0.1, 0.2]], dtype=np.float32), "int64"),
        (np.array([[0, 1]], dtype=np.int64), np.array([[0.1, 0.2]], dtype=np.float64), "float32"),
        (*result([[0, 1, 2]], [[0.1, 0.2, 0.3]]), "shape"),
        (*result([[0, -2]], [[0.1, 0.2]]), "invalid id"),
        (*result([[-1, 1]], [[np.inf, 0.2]]), "padding"),
        (*result([[1, -1]], [[0.1, 0.5]]), "padding"),
        (*result([[1, 2]], [[0.1, np.nan]]), "finite"),
        (*result([[1, 2]], [[0.1, np.inf]]), "finite"),
        (*result([[1, 2]], [[-0.5, 0.1]]), "range"),
        (*result([[1, 2]], [[0.1, 2.5]]), "range"),
        (*result([[1, 2]], [[0.2, 0.1]]), "sorted"),
        (*result([[1, 1]], [[0.1, 0.1]]), "duplicate"),
    ],
)
def test_check_search_result_rejects_invalid_result(ids, dists, message):
    with pytest.raises(ValueError, match=message):
        check_search_result(ids, dists, num_queries=1, k=2)
