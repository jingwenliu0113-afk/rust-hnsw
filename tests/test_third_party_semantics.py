"""釘住 hnswlib 0.8.0 與 faiss-cpu 1.15.1 的原生語意（decision 007「實測語意」）。

這些不是 adapter 的測試，而是 adapter 設計所依賴的事實。
library 升級後若有測試紅燈，代表語意改變了，必須回頭修訂 decision 007 與 adapters。
"""

import importlib.metadata
import math
import os
import threading

import faiss
import hnswlib
import numpy as np
import pytest

from benchmarks.interface import prepare_vectors
from helpers import duplicate_data, random_unit, tie_data

SHORT_RESULTS_MESSAGE = "Cannot return the results in a contiguous 2D array"
FLT_MAX = float(np.finfo(np.float32).max)


@pytest.fixture(autouse=True)
def restore_faiss_threads():
    """omp_set_num_threads 只影響呼叫的 thread（per-thread 設定）；每個測試結束後還原 main thread 的值。"""
    before = faiss.omp_get_max_threads()
    yield
    faiss.omp_set_num_threads(before)


def build_hnswlib(data, m=8, ef_construction=40, seed=1):
    index = hnswlib.Index(space="ip", dim=data.shape[1])
    index.init_index(max_elements=data.shape[0], ef_construction=ef_construction, M=m, random_seed=seed)
    index.set_num_threads(1)
    index.add_items(data, np.arange(data.shape[0]), num_threads=1)
    return index


def build_faiss(data, m=8, ef_construction=40, seed=None):
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(data.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    if seed is not None:
        index.hnsw.rng = faiss.RandomGenerator(seed)
    index.add(data)
    return index


def test_pinned_versions():
    assert importlib.metadata.version("hnswlib") == "0.8.0"
    assert importlib.metadata.version("faiss-cpu") == "1.15.1"


# ---- hnswlib ----


def test_hnswlib_defaults():
    index = hnswlib.Index(space="ip", dim=4)
    index.init_index(max_elements=10)
    assert (index.M, index.ef_construction, index.ef) == (16, 200, 10)
    assert index.num_threads > 1  # 預設使用全部核心
    index.set_num_threads(1)
    assert index.num_threads == 1


def test_hnswlib_layer_capacity_and_level_multiplier():
    # hnswalg.h：maxM0_ = 2M、mult_ = 1/ln M（與 004 相同）。
    index = hnswlib.Index(space="ip", dim=4)
    index.init_index(max_elements=10, M=6, ef_construction=20, random_seed=1)
    state = index.__getstate__()[0]
    assert (state["M"], state["max_M"], state["max_M0"]) == (6, 6, 12)
    assert state["mult"] == pytest.approx(1 / math.log(6))


def test_hnswlib_ip_space_returns_one_minus_inner_product_ascending():
    data = random_unit(0, 500, 16)
    queries = random_unit(1, 5, 16)
    index = build_hnswlib(data)
    index.set_ef(200)
    labels, dists = index.knn_query(queries, k=5, num_threads=1)
    assert labels.dtype == np.uint64
    assert dists.dtype == np.float32
    assert np.all(np.diff(dists, axis=1) >= 0)
    ip = np.einsum("qd,qkd->qk", queries.astype(np.float64), data[labels.astype(np.int64)].astype(np.float64))
    np.testing.assert_allclose(dists, 1 - ip, atol=1e-6)


def test_hnswlib_uses_max_of_ef_and_k():
    data = random_unit(0, 500, 16)
    queries = random_unit(1, 5, 16)
    index = build_hnswlib(data)
    results = {}
    for ef in (1, 5, 10, 20):
        index.set_ef(ef)
        results[ef] = index.knn_query(queries, k=10, num_threads=1)[0]
    np.testing.assert_array_equal(results[1], results[10])
    np.testing.assert_array_equal(results[5], results[10])
    assert not np.array_equal(results[10], results[20])


def test_hnswlib_raises_for_the_whole_batch_when_any_query_is_short():
    data = prepare_vectors(duplicate_data(12, 300, 6))
    queries = prepare_vectors(np.random.default_rng(13).standard_normal((10, 6)))
    index = build_hnswlib(data, m=4, ef_construction=20, seed=12)
    index.set_ef(300)
    with pytest.raises(RuntimeError, match=SHORT_RESULTS_MESSAGE):
        index.knn_query(queries, k=80, num_threads=1)

    # 不是每個 query 都不足：只要有一個不足，整批就失敗。
    outcomes = []
    for query in queries:
        try:
            index.knn_query(query[None, :], k=80, num_threads=1)
            outcomes.append(True)
        except RuntimeError:
            outcomes.append(False)
    assert any(outcomes) and not all(outcomes), outcomes


def test_hnswlib_k_greater_than_n_raises():
    index = build_hnswlib(tie_data(), m=4, ef_construction=20)
    index.set_ef(50)
    with pytest.raises(RuntimeError, match=SHORT_RESULTS_MESSAGE):
        index.knn_query(prepare_vectors([[1.0, 0.0]]), k=11, num_threads=1)


def test_hnswlib_knn_query_does_not_check_query_dimension():
    # add_items 會檢查維度，knn_query 不會：它依 index 的 dim 讀每一列（decision 007）。
    # 這裡只用比 dim 寬的 query，避免觸發越界讀取。
    index = build_hnswlib(random_unit(0, 100, 8))
    index.set_ef(16)
    labels, _ = index.knn_query(np.ones((2, 16), dtype=np.float32), k=5, num_threads=1)
    assert labels.shape == (2, 5)
    with pytest.raises(RuntimeError, match="Wrong dimensionality"):
        index.add_items(np.ones((2, 16), dtype=np.float32), np.arange(100, 102), num_threads=1)


def test_hnswlib_is_deterministic_with_the_same_seed():
    data = random_unit(0, 500, 16)
    queries = random_unit(1, 5, 16)
    a, b = build_hnswlib(data, seed=5), build_hnswlib(data, seed=5)
    a.set_ef(20)
    b.set_ef(20)
    np.testing.assert_array_equal(a.knn_query(queries, k=10, num_threads=1)[0], b.knn_query(queries, k=10, num_threads=1)[0])


# ---- FAISS ----


def test_faiss_defaults():
    index = faiss.IndexHNSWFlat(4, 8, faiss.METRIC_INNER_PRODUCT)
    assert (index.hnsw.efConstruction, index.hnsw.efSearch) == (40, 16)
    assert index.metric_type == faiss.METRIC_INNER_PRODUCT


@pytest.mark.skipif("OMP_NUM_THREADS" in os.environ, reason="環境變數會改變新 thread 的預設值")
def test_faiss_thread_setting_is_per_thread():
    # OpenMP 的 nthreads 是 per-thread ICV：main thread 設成 1，不影響其他 thread（decision 007 §6）。
    faiss.omp_set_num_threads(1)
    assert faiss.omp_get_max_threads() == 1
    seen = {}
    thread = threading.Thread(target=lambda: seen.setdefault("n", faiss.omp_get_max_threads()))
    thread.start()
    thread.join()
    assert seen["n"] > 1


def test_faiss_checks_query_dimension_with_assert_only():
    index = build_faiss(random_unit(0, 100, 8))
    with pytest.raises(AssertionError):
        index.search(np.ones((2, 16), dtype=np.float32), 5)


def test_faiss_returns_inner_product_similarity_descending():
    data = random_unit(0, 500, 16)
    queries = random_unit(1, 5, 16)
    index = build_faiss(data)
    index.hnsw.efSearch = 200
    sims, ids = index.search(queries, 5)
    assert ids.dtype == np.int64
    assert sims.dtype == np.float32
    assert np.all(np.diff(sims, axis=1) <= 0)
    ip = np.einsum("qd,qkd->qk", queries.astype(np.float64), data[ids].astype(np.float64))
    np.testing.assert_allclose(sims, ip, atol=1e-6)


def test_faiss_ef_below_k_is_not_equivalent_to_k():
    data = random_unit(0, 500, 16)
    queries = random_unit(1, 5, 16)
    index = build_faiss(data)
    results = {}
    for ef in (1, 5, 10):
        index.hnsw.efSearch = ef
        results[ef] = index.search(queries, 10)[1]
    assert not np.array_equal(results[1], results[10])
    assert not np.array_equal(results[5], results[10])


def test_faiss_pads_when_ef_is_much_smaller_than_k_on_ordinary_data():
    # efSearch ≪ k 時一般資料也會回傳少於 k 個結果；runner 的 ef ≥ k 檢查是必要條件（006、007 §4）。
    index = build_faiss(random_unit(0, 5000, 16), m=8, ef_construction=40)
    queries = random_unit(1, 50, 16)
    index.hnsw.efSearch = 1
    assert (index.search(queries, 100)[1] == -1).any()
    index.hnsw.efSearch = 100
    assert not (index.search(queries, 100)[1] == -1).any()


def test_faiss_pads_with_minus_one_and_negative_flt_max():
    index = build_faiss(tie_data(), m=4, ef_construction=20)
    index.hnsw.efSearch = 50
    sims, ids = index.search(prepare_vectors([[1.0, 0.0]]), 11)
    assert ids[0, -1] == -1
    assert sims[0, -1] == -FLT_MAX
    assert np.all(ids[0, :-1] >= 0)


def test_faiss_pads_short_results_on_duplicates():
    data = prepare_vectors(duplicate_data(12, 300, 6))
    queries = prepare_vectors(np.random.default_rng(13).standard_normal((10, 6)))
    index = build_faiss(data, m=4, ef_construction=20)
    index.hnsw.efSearch = 300
    sims, ids = index.search(queries, 80)
    padding = ids == -1
    assert padding.any()
    assert np.all(sims[padding] == -FLT_MAX)


def test_faiss_layer_capacities_and_level_probabilities():
    m = 4
    index = faiss.IndexHNSWFlat(8, m, faiss.METRIC_INNER_PRODUCT)
    cum = faiss.vector_to_array(index.hnsw.cum_nneighbor_per_level)[:4]
    assert cum.tolist() == [0, 2 * m, 3 * m, 4 * m]
    probas = faiss.vector_to_array(index.hnsw.assign_probas)
    assert probas[0] == pytest.approx(1 - 1 / m)


def test_faiss_new_nodes_keep_more_than_m_layer0_neighbors():
    # Rust 與 hnswlib 的新點在 layer 0 只選 M 個鄰居；FAISS 以 2M 為上限（decision 007）。
    # 高維隨機資料上 heuristic 幾乎不 prune，所以每個點的 layer-0 degree 都 > M。
    m = 8
    index = build_faiss(random_unit(1, 3000, 512), m=m, ef_construction=64, seed=1)
    offsets = faiss.vector_to_array(index.hnsw.offsets).astype(np.int64)
    neighbors = faiss.vector_to_array(index.hnsw.neighbors)
    cum = faiss.vector_to_array(index.hnsw.cum_nneighbor_per_level).astype(np.int64)
    degrees = np.array([(neighbors[start + cum[0] : start + cum[1]] >= 0).sum() for start in offsets[:-1]])
    assert degrees.min() > m


def test_faiss_insertion_order_is_not_row_order():
    # Rust 與 hnswlib 依 row 順序插入，entry 是最高層中 id 最小者；FAISS 先依 level 分組再插入（decision 007）。
    data = random_unit(0, 3000, 8)
    entry_not_smallest = 0
    for seed in range(40):
        index = build_faiss(data, m=4, seed=seed)
        levels = faiss.vector_to_array(index.hnsw.levels)
        top = np.flatnonzero(levels == levels.max())
        if len(top) > 1 and index.hnsw.entry_point != top.min():
            entry_not_smallest += 1
    assert entry_not_smallest > 0


def test_faiss_level_rng_seed_is_settable():
    data = random_unit(0, 2000, 8)

    def levels(seed):
        index = build_faiss(data, m=4, seed=seed)
        return faiss.vector_to_array(index.hnsw.levels)

    np.testing.assert_array_equal(levels(7), levels(7))
    np.testing.assert_array_equal(levels(None), levels(None))
    assert not np.array_equal(levels(7), levels(None))


# ---- tie 的順序（decision 007 第 7 點：不保證，各 library 不同） ----


def test_tie_order_differs_between_libraries():
    query = prepare_vectors([[1.0, 0.0]])
    # hnswlib：回傳集合內一定是 (distance, label) 升冪（searchKnn 的 result heap 與 knnQuery 從尾端回填）。
    h = build_hnswlib(tie_data(), m=4, ef_construction=20)
    h.set_ef(50)
    labels, dists = h.knn_query(query, k=5, num_threads=1)
    assert labels[0].tolist() == [0, 1, 2, 3, 4]
    assert np.all(dists == 0)

    # FAISS：只釘住這個 2-D fixture 上的觀察，不是一般保證（一般資料上順序會混合）。
    f = build_faiss(tie_data(), m=4, ef_construction=20)
    f.hnsw.efSearch = 50
    sims, ids = f.search(query, 5)
    assert ids[0].tolist() == [4, 3, 2, 1, 0]
    assert np.all(sims == 1)


def test_k1_answer_differs_from_first_column_of_k10_on_ties():
    """decision 008 §7：tie 時 k = 1 的答案不等於 k = 10 的第一欄，所以 Recall@1 另外跑 k = 1。"""
    query = prepare_vectors([[1.0, 0.0]])
    f = build_faiss(tie_data(), m=4, ef_construction=20)
    f.hnsw.efSearch = 16
    assert f.search(query, 10)[1][0, 0] == 4
    assert f.search(query, 1)[1][0, 0] != 4

    h = build_hnswlib(tie_data(), m=4, ef_construction=20)
    h.set_ef(16)
    assert h.knn_query(query, k=10, num_threads=1)[0][0, 0] == 0
    assert h.knn_query(query, k=1, num_threads=1)[0][0, 0] != 0


# ---- efConstruction < M（decision 008 §3：runner 要求 efC ≥ M） ----


def test_ef_construction_below_m_is_max_for_hnswlib_but_not_for_faiss():
    data = random_unit(3, 3000, 16)
    queries = random_unit(4, 200, 16)

    def hnswlib_ids(ef_construction):
        index = build_hnswlib(data, m=16, ef_construction=ef_construction)
        index.set_ef(20)
        return index.knn_query(queries, k=10, num_threads=1)[0]

    def faiss_ids(ef_construction):
        index = build_faiss(data, m=16, ef_construction=ef_construction, seed=1)
        index.hnsw.efSearch = 20
        return index.search(queries, 10)[1]

    # hnswlib 實際使用 max(efC, M)（與 Rust 相同，decision 004）；FAISS 沒有這條規則。
    np.testing.assert_array_equal(hnswlib_ids(4), hnswlib_ids(16))
    assert not np.array_equal(faiss_ids(4), faiss_ids(16))
