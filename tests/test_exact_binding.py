"""Rust exact index 的 Python binding 與 adapter 測試。

需要先 build binding：`./scripts/test.sh`（或 `maturin develop`）。
"""

import numpy as np
import pytest

import rust_hnsw
from benchmarks.adapters import RustExact
from benchmarks.interface import AnnIndex, check_search_result, prepare_vectors


def circle(degrees):
    rad = np.deg2rad(degrees)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32)


def numpy_exact(data, queries, k):
    """獨立的 numpy 參考實作，依 decision 001：
    f64 正規化後存成 f32，再以 f64 算 dot，依 (distance, id) 穩定排序。"""

    def to_unit_f32(x):
        x64 = x.astype(np.float64)
        return (x64 / np.linalg.norm(x64, axis=1, keepdims=True)).astype(np.float32)

    dist = 1.0 - to_unit_f32(queries).astype(np.float64) @ to_unit_f32(data).astype(np.float64).T
    order = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return order.astype(np.int64), np.take_along_axis(dist, order, axis=1).astype(np.float32)


def test_hand_verifiable_unit_circle_through_python():
    index = rust_hnsw.ExactIndex(2)
    index.add(circle([0, 30, 60, 90, 180]))
    ids, dists = index.search(circle([10, 100]), 5)

    assert ids.dtype == np.int64
    assert dists.dtype == np.float32
    assert ids.shape == dists.shape == (2, 5)
    np.testing.assert_array_equal(ids, [[0, 1, 2, 3, 4], [3, 2, 1, 4, 0]])
    np.testing.assert_allclose(dists[0], 1 - np.cos(np.deg2rad([10, 20, 50, 80, 170])), atol=1e-6)


def test_len_and_dim():
    index = rust_hnsw.ExactIndex(3)
    assert index.dim == 3
    assert len(index) == 0
    index.add(np.ones((4, 3), dtype=np.float32))
    assert len(index) == 4


def test_empty_query_batch():
    index = rust_hnsw.ExactIndex(2)
    index.add(circle([0, 90]))
    ids, dists = index.search(np.zeros((0, 2), dtype=np.float32), 1)
    assert ids.shape == dists.shape == (0, 1)


def test_rejects_non_float32_input():
    index = rust_hnsw.ExactIndex(2)
    with pytest.raises(TypeError):
        index.add(np.ones((2, 2), dtype=np.float64))


@pytest.mark.parametrize(
    "array",
    [
        np.ones((4, 4), dtype=np.float32)[:, ::2],
        np.asfortranarray(np.arange(1, 7, dtype=np.float32).reshape(3, 2)),
    ],
    ids=["strided", "fortran-order"],
)
def test_rejects_non_c_contiguous_input(array):
    index = rust_hnsw.ExactIndex(2)
    with pytest.raises(ValueError, match="C-contiguous"):
        index.add(array)
    index.add(circle([0, 90]))
    with pytest.raises(ValueError, match="C-contiguous"):
        index.search(array, 1)


def test_rejects_wrong_dimension():
    # (2, 3) 攤平後長度 6 剛好是 dim 2 的倍數，binding 必須檢查 shape 而不是只看長度。
    index = rust_hnsw.ExactIndex(2)
    with pytest.raises(ValueError, match="dimension mismatch"):
        index.add(np.ones((2, 3), dtype=np.float32))
    index.add(circle([0, 90]))
    with pytest.raises(ValueError, match="dimension mismatch"):
        index.search(np.ones((2, 3), dtype=np.float32), 1)


def test_errors_map_to_value_error():
    with pytest.raises(ValueError, match="dimension must be positive"):
        rust_hnsw.ExactIndex(0)
    index = rust_hnsw.ExactIndex(2)
    with pytest.raises(ValueError, match="index is empty"):
        index.search(circle([0]), 1)
    index.add(circle([0, 90]))
    with pytest.raises(ValueError, match="row 1 is a zero vector"):
        index.add(np.array([[1, 0], [0, 0]], dtype=np.float32))
    assert len(index) == 2
    with pytest.raises(ValueError, match="k must satisfy"):
        index.search(circle([0]), 3)


def test_matches_numpy_reference_on_random_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1000, 32)).astype(np.float32)
    queries = rng.standard_normal((50, 32)).astype(np.float32)
    index = rust_hnsw.ExactIndex(32)
    index.add(data)

    ids, dists = index.search(queries, 10)
    ref_ids, ref_dists = numpy_exact(data, queries, 10)
    np.testing.assert_array_equal(ids, ref_ids)
    np.testing.assert_allclose(dists, ref_dists, atol=1e-6)


def test_rust_exact_adapter_conforms_to_interface():
    rng = np.random.default_rng(1)
    data = prepare_vectors(rng.standard_normal((300, 16)))
    queries = prepare_vectors(rng.standard_normal((20, 16)))

    adapter = RustExact()
    assert isinstance(adapter, AnnIndex)
    assert adapter.name == "rust-exact"
    adapter.build(data)
    ids, dists = adapter.search(queries, 10)
    check_search_result(ids, dists, num_queries=20, k=10)

    direct = rust_hnsw.ExactIndex(16)
    direct.add(data)
    np.testing.assert_array_equal(ids, direct.search(queries, 10)[0])
