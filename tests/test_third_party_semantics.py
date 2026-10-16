"""釘住 hnswlib 0.8.0 與 faiss-cpu 1.15.1 的原生語意（decision 007「實測語意」）。

這些不是 adapter 的測試，而是 adapter 設計所依賴的事實。
library 升級後若有測試紅燈，代表語意改變了，必須回頭修訂 decision 007 與 adapters。
Advanced C 的 `IndexHNSWSQ` 語意、移植與 codec 比對（decision 019 §4）在檔案最後一節；全部只用 synthetic 資料。
"""

import gc
import importlib.metadata
import itertools
import math
import os
import threading

import faiss
import hnswlib
import numpy as np
import pytest

import rust_hnsw
from benchmarks.adapters import faiss_hnsw_flat, faiss_sq8_from_flat
from benchmarks.interface import DISTANCE_RANGE_TOL, check_search_result, prepare_vectors
from helpers import duplicate_data, other_thread_cpu, random_unit, tie_data

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


# ---- Advanced C：FAISS 的 SQ8 baseline（decision 019 §3、§4；只用 synthetic 資料） ----
#
# 臂 Q（`faiss-hnsw-sq8`）的設計所依賴的事實，比照 007 以 [測試] 登記。
# 移植可不可行只由本節的 synthetic 測試判定（019 §3），所以「移植」一段直接測 adapter 使用的
# `faiss_hnsw_flat` / `faiss_sq8_from_flat`；其餘以原生 API 釘住 FAISS 本身的行為。

QT_8BIT = faiss.ScalarQuantizer.QT_8bit
RS_MINMAX = faiss.ScalarQuantizer.RS_minmax
# 019 §4 要求與 Flat 完全相同的 graph 欄位（另加 entry_point、max_level）。
GRAPH_FIELDS = ("levels", "offsets", "neighbors")


def single_threaded_q(data, m=8, ef_construction=40, seed=1):
    """以 adapter 的移植函式建 Q，回傳 `(flat, q)`。先設成單執行緒：FAISS 多執行緒的 add 不是決定性的。"""
    faiss.omp_set_num_threads(1)
    flat = faiss_hnsw_flat(data, m, ef_construction, seed)
    return flat, faiss_sq8_from_flat(flat, data, m)


def hnsw_graph(index) -> dict:
    hnsw = index.hnsw
    graph = {name: faiss.vector_to_array(getattr(hnsw, name)) for name in GRAPH_FIELDS}
    graph.update(entry_point=hnsw.entry_point, max_level=hnsw.max_level)
    return graph


def assert_same_graph(a, b) -> None:
    graph_a, graph_b = hnsw_graph(a), hnsw_graph(b)
    for name in GRAPH_FIELDS:
        np.testing.assert_array_equal(graph_a[name], graph_b[name], err_msg=name)
    assert (graph_a["entry_point"], graph_a["max_level"]) == (graph_b["entry_point"], graph_b["max_level"])


def minmax_quantizer(d):
    """與 Q 的 storage 相同設定的獨立 ScalarQuantizer：QT_8bit、RS_minmax、rangestat_arg = 0（019 §3）。"""
    sq = faiss.ScalarQuantizer(d, QT_8BIT)
    sq.rangestat = RS_MINMAX
    sq.rangestat_arg = 0
    return sq


def trained_codebook(sq, d):
    """QT_8bit（per-dim）的 `trained`：前 d 個是 vmin，後 d 個是 vdiff = vmax − vmin。"""
    trained = faiss.vector_to_array(sq.trained)
    assert trained.shape == (2 * d,)
    return trained[:d], trained[d:]


def test_faiss_sq_indexes_must_be_trained_before_add():
    data = random_unit(0, 100, 8)
    for index in (faiss.IndexHNSWSQ(8, QT_8BIT, 4, faiss.METRIC_INNER_PRODUCT),
                  faiss.IndexScalarQuantizer(8, QT_8BIT, faiss.METRIC_INNER_PRODUCT)):
        assert not index.is_trained
        with pytest.raises(RuntimeError, match="is_trained"):
            index.add(data)
        assert index.ntotal == 0


def test_faiss_sq_codec_settings_take_effect():
    # white-box：Q 的 storage 確實是 QT_8bit、RS_minmax、rangestat_arg = 0，訓練結果就是逐維的 min 與 max − min（f32），
    # 不外擴範圍、不使用 query（019 §3）。
    data = random_unit(1, 500, 16)
    flat, q = single_threaded_q(data)
    storage = faiss.downcast_index(q.storage)
    assert isinstance(storage, faiss.IndexScalarQuantizer)
    assert (storage.sq.qtype, storage.sq.rangestat, storage.sq.rangestat_arg) == (QT_8BIT, RS_MINMAX, 0.0)
    assert storage.sq.code_size == storage.code_size == 16  # 每維 1 byte
    assert q.metric_type == storage.metric_type == faiss.METRIC_INNER_PRODUCT
    vmin, vdiff = trained_codebook(storage.sq, 16)
    np.testing.assert_array_equal(vmin, data.min(axis=0))
    np.testing.assert_array_equal(vdiff, data.max(axis=0) - data.min(axis=0))

    # 這兩個欄位真的會改變訓練結果，所以明確設定不是空操作。
    for rangestat, arg in ((RS_MINMAX, 0.1), (faiss.ScalarQuantizer.RS_meanstd, 1.0)):
        other = minmax_quantizer(16)
        other.rangestat, other.rangestat_arg = rangestat, arg
        other.train(data)
        assert not np.array_equal(trained_codebook(other, 16)[0], vmin), (rangestat, arg)

    # FAISS 的預設本來就是這兩個值；adapter 仍明確設定（019 §3）。
    default = faiss.IndexHNSWSQ(16, QT_8BIT, 8, faiss.METRIC_INNER_PRODUCT)
    default_storage = faiss.downcast_index(default.storage)
    assert (default_storage.sq.rangestat, default_storage.sq.rangestat_arg) == (RS_MINMAX, 0.0)


@pytest.mark.parametrize("n", [1, 16])
def test_faiss_sq_trains_and_searches_tiny_indexes(n):
    # runner 的 null call（n = 1、k = 1、ef = 1）與 library warm-up（前 16 列）都會走到這裡（008 §6.3、§6.4）。
    data = random_unit(2, n, 8)
    queries = random_unit(3, 5, 8)
    flat, q = single_threaded_q(data)
    del flat
    q.hnsw.efSearch = 1
    sims, ids = q.search(queries, 1)
    assert ids.shape == (5, 1)
    assert (ids >= 0).all() and np.isfinite(sims).all()
    if n == 1:
        assert (ids == 0).all()
    # 019 §7 的 c_FAISS（IndexScalarQuantizer 的 brute force）也一樣能 train 與 search。
    exact = faiss.IndexScalarQuantizer(8, QT_8BIT, faiss.METRIC_INNER_PRODUCT)
    exact.sq.rangestat = RS_MINMAX
    exact.train(data)
    exact.add(data)
    assert (exact.search(queries, 1)[1] >= 0).all()


def test_faiss_sq_zero_range_dimension_encodes_to_zero_and_decodes_to_vmin():
    # vdiff = 0 的維度：code 一律為 0，解碼得到 vmin 本身（與 019 §3 的 codec 相同）。
    data = random_unit(4, 50, 8).copy()
    data[:, 3] = data[0, 3]
    sq = minmax_quantizer(8)
    sq.train(data)
    vmin, vdiff = trained_codebook(sq, 8)
    assert vdiff[3] == 0 and (np.delete(vdiff, 3) > 0).all()
    codes = sq.compute_codes(data)
    assert (codes[:, 3] == 0).all()
    np.testing.assert_array_equal(sq.decode(codes)[:, 3], np.full(50, vmin[3], dtype=np.float32))

    # n = 1（runner 的 null call）：每一維都是 vdiff = 0，重建就是原向量本身（逐位元）。
    one = np.ascontiguousarray(data[:1])
    sq_one = minmax_quantizer(8)
    sq_one.train(one)
    assert (trained_codebook(sq_one, 8)[1] == 0).all()
    assert (sq_one.compute_codes(one) == 0).all()
    np.testing.assert_array_equal(sq_one.decode(sq_one.compute_codes(one)), one)


def test_faiss_sq_similarity_can_exceed_one_and_clipping_restores_the_range():
    # 估計的 ip 可能 > 1（query 就是某個資料點時最明顯）。未夾住的 1 − ip 會讓 check_search_result 拋出 ValueError，
    # 整個子 process 因此失敗（019 背景事實 4）；adapter 先轉成 1 − ip 再原地夾到 [0, 2]（019 §3）。
    data = random_unit(5, 2000, 16)
    queries = np.ascontiguousarray(data[:300])
    flat, q = single_threaded_q(data)
    q.hnsw.efSearch = 64
    sims, ids = q.search(queries, 10)
    assert sims.max() > 1 + DISTANCE_RANGE_TOL
    assert (np.diff(sims, axis=1) <= 0).all()  # 依估計的 similarity 降冪
    distances = 1.0 - sims
    with pytest.raises(ValueError, match="out of range"):
        check_search_result(ids, distances, 300, 10)
    # 夾住是單調的：排序不變，格式檢查通過。
    check_search_result(ids, np.clip(distances, 0.0, 2.0), 300, 10)


def test_faiss_sq_pads_short_results_with_minus_one_and_negative_flt_max():
    # 與 Flat 相同的補齊值；adapter 的 clip 會先把 −FLT_MAX 變成 2，最後才改成 +inf（019 §3）。
    data = prepare_vectors(duplicate_data(12, 300, 6))
    queries = prepare_vectors(np.random.default_rng(13).standard_normal((10, 6)))
    flat, q = single_threaded_q(data, m=4, ef_construction=20, seed=12)
    q.hnsw.efSearch = 300
    sims, ids = q.search(queries, 80)
    padding = ids == -1
    assert padding.any()
    assert np.all(sims[padding] == -FLT_MAX)


def test_faiss_sq_ef_search_semantics_match_flat():
    # 與 IndexHNSWFlat 相同（007 第 4 點）：efSearch < k 不等於 max(efSearch, k)；efSearch ≪ k 時一般資料也會補齊。
    # efSearch 是 hnsw 結構的欄位：移植時一併複製（Flat 的預設 16），之後兩者各自設定。
    data = random_unit(0, 5000, 16)
    queries = random_unit(1, 50, 16)
    flat, q = single_threaded_q(data)
    assert q.hnsw.efSearch == flat.hnsw.efSearch == 16
    q.hnsw.efSearch = 37
    assert flat.hnsw.efSearch == 16

    results = {}
    for ef in (1, 5, 10):
        q.hnsw.efSearch = ef
        results[ef] = q.search(queries, 10)[1]
    assert not np.array_equal(results[1], results[10])
    assert not np.array_equal(results[5], results[10])

    q.hnsw.efSearch = 1
    assert (q.search(queries, 100)[1] == -1).any()
    q.hnsw.efSearch = 100
    assert not (q.search(queries, 100)[1] == -1).any()


def test_faiss_sq_is_deterministic_with_the_same_seed():
    data = random_unit(6, 2000, 16)
    queries = random_unit(7, 50, 16)
    outputs = []
    for _ in range(2):
        flat, q = single_threaded_q(data, seed=9)
        del flat
        q.hnsw.efSearch = 32
        outputs.append(q.search(queries, 10))
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    np.testing.assert_array_equal(outputs[0][0].view(np.uint32), outputs[1][0].view(np.uint32))


# ---- 單執行緒（007 §6；019 §4：build 的 train / add / 移植與 search 都驗，也驗跨 thread 呼叫） ----


def transplant_steps(data, queries, large):
    """依 019 §3 的移植步驟逐步執行（呼叫端先設定 OpenMP 的執行緒數），回傳每一步的 (其他 thread 的 CPU 秒數, 整個 process 的 CPU 秒數)。

    10,000 列的 train / add 只需不到 1 ms，多執行緒時也量不出差別；所以另外在 `large`（20 萬列）上量一次獨立的 storage。
    """
    steps, made = {}, {}
    steps["flat build"] = other_thread_cpu(lambda: made.setdefault("flat", faiss_hnsw_flat(data, 8, 40, 1)))
    q = faiss.IndexHNSWSQ(data.shape[1], QT_8BIT, 8, faiss.METRIC_INNER_PRODUCT)

    def copy_graph():
        q.hnsw = made["flat"].hnsw
        q.ntotal = data.shape[0]

    steps["graph copy"] = other_thread_cpu(copy_graph)
    storage = faiss.downcast_index(q.storage)
    steps["storage train"] = other_thread_cpu(lambda: storage.train(data))
    steps["storage add"] = other_thread_cpu(lambda: storage.add(data))
    q.is_trained = True
    q.hnsw.efSearch = 100
    steps["search"] = other_thread_cpu(lambda: q.search(queries, 10))
    large_storage = faiss.IndexScalarQuantizer(large.shape[1], QT_8BIT, faiss.METRIC_INNER_PRODUCT)
    steps["large train"] = other_thread_cpu(lambda: large_storage.train(large))
    steps["large add"] = other_thread_cpu(lambda: large_storage.add(large))
    return steps


def thread_fixture():
    data = random_unit(40, 10000, 32)
    return data, random_unit(41, 3000, 32), np.tile(data, (20, 1))


def test_faiss_sq_transplant_and_search_are_single_threaded_with_one_omp_thread():
    data, queries, large = thread_fixture()
    faiss.omp_set_num_threads(1)
    for step, (other, total) in transplant_steps(data, queries, large).items():
        assert other <= 0.02 * total + 1e-3, (step, other, total)


def test_faiss_sq_steps_use_other_threads_when_omp_allows():
    # 正對照：上一個測試的單執行緒斷言有鑑別力（train 與 graph 的複製本來就是單執行緒，不列入）。
    data, queries, large = thread_fixture()
    faiss.omp_set_num_threads(4)
    steps = transplant_steps(data, queries, large)
    for step in ("flat build", "search", "large add"):
        other, total = steps[step]
        assert other > 0.3 * total, (step, other, total)


@pytest.mark.skipif("OMP_NUM_THREADS" in os.environ, reason="環境變數會改變新 thread 的預設值")
def test_faiss_sq_search_in_another_thread_needs_its_own_setting():
    # OpenMP 的 nthreads 是 per-thread 設定：main thread 設成 1 之後，其他 thread 的 search 仍是多執行緒，
    # 必須在該 thread 再設定一次；所以 adapter 在每次 search 開頭都呼叫 omp_set_num_threads(1)（019 §4）。
    data, queries, _ = thread_fixture()
    flat, q = single_threaded_q(data)
    q.hnsw.efSearch = 100
    measured = {}

    def worker():
        measured["default"] = other_thread_cpu(lambda: q.search(queries, 10))
        faiss.omp_set_num_threads(1)
        measured["set"] = other_thread_cpu(lambda: q.search(queries, 10))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    other, total = measured["default"]
    assert other > 0.3 * total, (other, total)
    other, total = measured["set"]
    assert other <= 0.02 * total + 1e-3, (other, total)


# ---- 移植（019 §3「Q 的移植」；可行性只由這些 synthetic 測試判定） ----

TRANSPLANT_CASES = {  # 名稱 → (資料, M, efC, seed)：多層圖、M = 4、重複向量（會補齊）、n ≤ 2M + 1、warm-up 與 null call 的大小
    "multi-level": (lambda: random_unit(8, 3000, 16), 8, 40, 1),
    "m4": (lambda: random_unit(9, 2000, 8), 4, 20, 7),
    "duplicates": (lambda: prepare_vectors(duplicate_data(12, 300, 6)), 4, 20, 12),
    "n17-m8": (lambda: random_unit(10, 17, 8), 8, 32, 1),
    "n16": (lambda: random_unit(11, 16, 8), 16, 200, 1),
    "n1": (lambda: random_unit(12, 1, 8), 16, 200, 1),
}


@pytest.mark.parametrize("case", list(TRANSPLANT_CASES))
def test_transplant_copies_the_graph_exactly(case):
    make, m, ef_construction, seed = TRANSPLANT_CASES[case]
    data = make()
    flat, q = single_threaded_q(data, m, ef_construction, seed)
    assert_same_graph(q, flat)
    for name in ("cum_nneighbor_per_level", "assign_probas"):
        np.testing.assert_array_equal(faiss.vector_to_array(getattr(q.hnsw, name)),
                                      faiss.vector_to_array(getattr(flat.hnsw, name)), err_msg=name)
    assert (q.hnsw.efConstruction, q.hnsw.efSearch) == (flat.hnsw.efConstruction, flat.hnsw.efSearch)
    assert q.ntotal == flat.ntotal == len(data)
    if case == "multi-level":
        assert flat.hnsw.max_level >= 2


def test_transplant_is_a_copy_that_survives_releasing_the_flat():
    # SWIG setter 做的是 C++ 結構複製（019 背景事實 7）：之後改動或釋放 Flat，Q 的 graph 與結果都不變。
    data = random_unit(13, 3000, 16)
    queries = random_unit(14, 100, 16)
    flat, q = single_threaded_q(data)
    q.hnsw.efSearch = 64
    graph = hnsw_graph(q)
    before = q.search(queries, 10)

    neighbors = faiss.vector_to_array(flat.hnsw.neighbors)
    neighbors[:] = -1
    faiss.copy_array_to_vector(neighbors, flat.hnsw.neighbors)
    flat.hnsw.entry_point = (flat.hnsw.entry_point + 1) % len(data)
    flat.hnsw.efSearch = 5
    del flat
    gc.collect()

    after = q.search(queries, 10)
    for name in GRAPH_FIELDS:
        np.testing.assert_array_equal(hnsw_graph(q)[name], graph[name], err_msg=name)
    assert (q.hnsw.entry_point, q.hnsw.efSearch) == (graph["entry_point"], 64)
    np.testing.assert_array_equal(after[1], before[1])
    np.testing.assert_array_equal(after[0].view(np.uint32), before[0].view(np.uint32))


def test_training_only_the_storage_leaves_the_outer_index_untrained():
    # 019 §3 第 3 步的理由：只 train storage 時，外層 IndexHNSWSQ 的 is_trained 仍是 False；
    # 原生的 IndexHNSWSQ.train 則會一併設成 True。
    data = random_unit(15, 200, 8)
    q = faiss.IndexHNSWSQ(8, QT_8BIT, 4, faiss.METRIC_INNER_PRODUCT)
    storage = faiss.downcast_index(q.storage)
    storage.train(data)
    assert storage.is_trained and not q.is_trained
    native = faiss.IndexHNSWSQ(8, QT_8BIT, 4, faiss.METRIC_INNER_PRODUCT)
    native.train(data)
    assert native.is_trained


def test_transplanted_index_is_trained_searchable_and_writable(tmp_path):
    data = random_unit(16, 1000, 16)
    queries = random_unit(17, 50, 16)
    flat, q = single_threaded_q(data)
    del flat
    storage = faiss.downcast_index(q.storage)
    assert q.is_trained and storage.is_trained
    assert q.ntotal == storage.ntotal == 1000
    # storage 存的 codes 就是 compute_codes 的結果（codec 比對用的就是這個函式）。
    stored = faiss.vector_to_array(storage.codes).reshape(1000, 16)
    np.testing.assert_array_equal(stored, storage.sq.compute_codes(data))

    q.hnsw.efSearch = 32
    sims, ids = q.search(queries, 10)
    assert (ids >= 0).all()
    path = tmp_path / "q.index"
    faiss.write_index(q, str(path))
    loaded = faiss.read_index(str(path))
    assert isinstance(loaded, faiss.IndexHNSWSQ)
    assert loaded.is_trained and loaded.ntotal == 1000
    assert_same_graph(loaded, q)
    loaded.hnsw.efSearch = 32
    loaded_sims, loaded_ids = loaded.search(queries, 10)
    np.testing.assert_array_equal(loaded_ids, ids)
    np.testing.assert_array_equal(loaded_sims.view(np.uint32), sims.view(np.uint32))


def sign_cube(d=8, n_queries=40, seed=0):
    """量化不改變排序的 synthetic 資料（019 §4「移植」：在量化誤差不改變排序的資料上，Q 的 ids 與 P 相同）。

    data：{−a, +a}^d 的全部 2^d 個頂點（a = 1/√d，所以是 unit vectors），打散順序。每一維的 vmin = −a、vmax = +a，
    codes 只有 0 與 255，重建 x̂ = x + (a/255)·1：每個 query 的估計 ip 只差同一個常數，排序不變。
    queries：(±2^π(j)) 正規化；不同頂點的 ip 兩兩不同，間隔約 5e-3，遠大於 float32 的 rounding。
    """
    rng = np.random.default_rng(seed)
    vertices = np.array(list(itertools.product((-1.0, 1.0), repeat=d)))
    data = np.ascontiguousarray(vertices[rng.permutation(len(vertices))] / np.sqrt(d), dtype=np.float32)
    weights = 2.0 ** np.arange(d)
    queries = [weights[rng.permutation(d)] * rng.choice((-1.0, 1.0), d) for _ in range(n_queries)]
    return data, prepare_vectors(queries)


def test_transplant_returns_the_flat_ids_when_quantization_preserves_the_ranking():
    data, queries = sign_cube()
    # 前提：量化不改變任何 query 對全部資料點的排序（brute force 逐一比對），而且 ip 之間的間隔遠大於 rounding。
    exact = faiss.IndexFlatIP(data.shape[1])
    exact.add(data)
    exact_sims, exact_ids = exact.search(queries, len(data))
    quantized = faiss.IndexScalarQuantizer(data.shape[1], QT_8BIT, faiss.METRIC_INNER_PRODUCT)
    quantized.sq.rangestat = RS_MINMAX
    quantized.train(data)
    quantized.add(data)
    assert (faiss.vector_to_array(quantized.codes) % 255 == 0).all()  # codes 只有 0 與 255
    np.testing.assert_array_equal(quantized.search(queries, len(data))[1], exact_ids)
    assert np.diff(exact_sims, axis=1).max() < -1e-3

    approximate = False
    for m, seed in ((4, 1), (4, 2), (8, 1), (8, 2)):
        flat, q = single_threaded_q(data, m=m, ef_construction=20, seed=seed)
        for ef in (1, 2, 5, 10, 16, 32, 100, 256):
            flat.hnsw.efSearch = q.hnsw.efSearch = ef
            for k in (1, 10, 50):
                np.testing.assert_array_equal(q.search(queries, k)[1], flat.search(queries, k)[1],
                                              err_msg=f"M = {m}, seed = {seed}, ef = {ef}, k = {k}")
        flat.hnsw.efSearch = 10
        approximate |= not np.array_equal(flat.search(queries, 10)[1], exact_ids[:, :10])
    # 這幾張圖在小 ef 時確實是近似搜尋（走訪的路徑會影響結果），不是窮舉。
    assert approximate


def test_transplant_ids_differ_from_flat_when_quantization_changes_the_ranking():
    # 對照：一般資料上量化誤差會改變排序，Q 與 Flat 的 ids 不完全相同；上一個測試的相同不是恆真。
    data = random_unit(0, 2000, 16)
    queries = random_unit(1, 200, 16)
    flat, q = single_threaded_q(data)
    flat.hnsw.efSearch = q.hnsw.efSearch = 64
    assert not np.array_equal(q.search(queries, 10)[1], flat.search(queries, 10)[1])


# ---- codec：Rust 的 sq8_encode 對 FAISS 的 compute_codes（019 §4、C8；只記錄，不擋流程） ----


def constant_dimension_data():
    data = random_unit(25, 200, 8).copy()
    data[:, 3] = data[0, 3]
    return data


CODEC_INPUTS = {
    "random-16d": lambda: random_unit(20, 1000, 16),
    "random-100d": lambda: random_unit(21, 2000, 100),
    "random-768d": lambda: random_unit(22, 300, 768),
    "n1": lambda: random_unit(23, 1, 8),
    "n16": lambda: random_unit(24, 16, 8),
    "duplicates": lambda: prepare_vectors(duplicate_data(12, 300, 6)),
    "constant-dimension": constant_dimension_data,
}


def faiss_codec(x):
    sq = minmax_quantizer(x.shape[1])
    sq.train(x)
    vmin, vdiff = trained_codebook(sq, x.shape[1])
    return vmin, vdiff, sq.compute_codes(x)


def codec_agreement(rust, reference) -> dict:
    """codec 輸出的比對：codes 的一致比例（逐元素與整列）與 codebook 是否逐位元相同（019 §4、C8）。"""
    (vmin, vdiff, codes), (ref_vmin, ref_vdiff, ref_codes) = rust, reference
    assert codes.shape == ref_codes.shape and codes.dtype == ref_codes.dtype == np.uint8
    same = codes == ref_codes
    return {
        "codes_equal_fraction": float(same.mean()),
        "rows_equal_fraction": float(same.all(axis=1).mean()),
        "vmin_identical": bool(np.array_equal(vmin.view(np.uint32), ref_vmin.view(np.uint32))),
        "vdiff_identical": bool(np.array_equal(vdiff.view(np.uint32), ref_vdiff.view(np.uint32))),
    }


@pytest.mark.parametrize("name", list(CODEC_INPUTS))
def test_rust_sq8_codec_against_faiss_compute_codes(name, record_property):
    # 在同一個輸入陣列上比對，照實記錄（junit 的 property；`pytest -s` 會印出），不擋流程（019 §4、C8）。
    # 撰寫時實測（faiss-cpu 1.15.1）：每一組輸入的 codes 一致比例都是 1.0，vmin、vdiff 逐位元相同。
    x = CODEC_INPUTS[name]()
    agreement = codec_agreement(rust_hnsw.sq8_encode(x), faiss_codec(x))
    for key, value in agreement.items():
        record_property(key, value)
    print(f"[019 C8 codec] {name}: {agreement}")
    assert 0.0 <= agreement["codes_equal_fraction"] <= 1.0


def test_codec_agreement_reflects_any_difference():
    # 比對本身的鑑別力：改一個 code 或 codebook 的最後一個 bit，記錄的數字就會改變。
    x = random_unit(20, 1000, 16)
    vmin, vdiff, codes = rust_hnsw.sq8_encode(x)
    assert codec_agreement((vmin, vdiff, codes), (vmin, vdiff, codes)) == {
        "codes_equal_fraction": 1.0, "rows_equal_fraction": 1.0, "vmin_identical": True, "vdiff_identical": True}
    changed = codes.copy()
    changed[3, 5] ^= 1
    agreement = codec_agreement((vmin, vdiff, changed), (vmin, vdiff, codes))
    assert agreement["codes_equal_fraction"] == 1 - 1 / codes.size
    assert agreement["rows_equal_fraction"] == 1 - 1 / len(codes)
    for index, key in ((0, "vmin_identical"), (1, "vdiff_identical")):
        book = [vmin.copy(), vdiff.copy()]
        book[index].view(np.uint32)[0] ^= 1
        assert not codec_agreement((*book, codes), (vmin, vdiff, codes))[key]
