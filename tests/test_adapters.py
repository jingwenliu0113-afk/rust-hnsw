"""三個 HNSW adapter（Rust、hnswlib、FAISS）的契約與語意等價測試（decision 002、006、007）。

這些是 correctness 測試，不是 benchmark：CPU time 只用來確認單執行緒設定生效。
"""

import threading

import faiss
import hnswlib
import numpy as np
import pytest

import rust_hnsw
from benchmarks.adapters import FaissHnsw, Hnswlib, RustHnsw
from benchmarks.interface import HnswAnnIndex, IncompleteResultsError, check_search_result, prepare_vectors
from benchmarks.metrics import recall_at_k
from helpers import duplicate_data, other_thread_cpu, random_unit, tie_data

ADAPTERS = {
    "rust-hnsw": RustHnsw,
    "hnswlib": Hnswlib,
    "faiss-hnsw": FaissHnsw,
}


@pytest.fixture(params=list(ADAPTERS))
def adapter_cls(request):
    return ADAPTERS[request.param]


@pytest.fixture(autouse=True)
def restore_faiss_threads():
    before = faiss.omp_get_max_threads()
    yield
    faiss.omp_set_num_threads(before)


def built(cls, data, m=8, ef_construction=64, seed=5):
    adapter = cls(m=m, ef_construction=ef_construction, seed=seed)
    adapter.build(data)
    return adapter


class CallRecorder:
    """包住 adapter 的原生 index，記錄每次原生 search 收到的 queries shape；其他屬性原樣轉發。"""

    def __init__(self, inner):
        self._inner = inner
        self.calls = []

    def search(self, queries, *args, **kwargs):
        self.calls.append(np.shape(queries))
        return self._inner.search(queries, *args, **kwargs)

    def knn_query(self, queries, *args, **kwargs):
        self.calls.append(np.shape(queries))
        return self._inner.knn_query(queries, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def exact_ids(data, queries, k):
    exact = rust_hnsw.ExactIndex(data.shape[1])
    exact.add(data)
    return exact.search(queries, k)[0]


# ---- 契約（三個 adapter 都要滿足） ----


def test_conforms_to_the_hnsw_interface(adapter_cls):
    data = random_unit(13, 400, 16)
    queries = random_unit(14, 20, 16)
    adapter = built(adapter_cls, data)
    assert isinstance(adapter, HnswAnnIndex)
    assert adapter.name in ADAPTERS
    adapter.set_ef_search(32)
    ids, dists = adapter.search(queries, 10)
    check_search_result(ids, dists, num_queries=20, k=10)
    # 用 ip / INNER_PRODUCT，不用 hnswlib 的 "cosine"（會在計時路徑內再正規化，decision 007 §1）。
    if adapter_cls is Hnswlib:
        assert adapter._index.space == "ip"
    if adapter_cls is FaissHnsw:
        assert adapter._index.metric_type == faiss.METRIC_INNER_PRODUCT


def test_one_native_call_per_search(adapter_cls):
    # 整批 queries 只做一次原生呼叫：adapter 內沒有 per-query 迴圈（002、007 §3）。
    data = random_unit(13, 400, 16)
    adapter = built(adapter_cls, data)
    adapter.set_ef_search(32)
    recorder = CallRecorder(adapter._index)
    adapter._index = recorder
    adapter.search(random_unit(14, 20, 16), 10)
    assert recorder.calls == [(20, 16)]


def test_build_requires_prepared_vectors(adapter_cls):
    # hnswlib / FAISS 只在 unit vectors 上等價於 cosine；Rust 會自行正規化而掩蓋錯誤（decision 007 §1）。
    data = random_unit(1, 200, 8)
    for bad in (data * 1.05, data.astype(np.float64), np.asfortranarray(data)):
        with pytest.raises(ValueError, match="prepare_vectors"):
            adapter_cls(m=8, ef_construction=40, seed=1).build(bad)


def test_k_validation_is_uniform(adapter_cls):
    # 原生行為不一（hnswlib k > n 整批 raise、FAISS 補 -1、k = 0 各自不同）；adapter 一律 ValueError。
    data = random_unit(1, 200, 8)
    adapter = built(adapter_cls, data)
    adapter.set_ef_search(16)
    queries = random_unit(2, 3, 8)
    for bad_k in (0, len(data) + 1):
        with pytest.raises(ValueError, match="k must satisfy") as info:
            adapter.search(queries, bad_k)
        assert not isinstance(info.value, IncompleteResultsError)
    ids, _ = adapter.search(queries, np.int64(10))
    assert ids.shape == (3, 10)


def test_reported_distances_are_cosine_distances(adapter_cls):
    # 三者回傳同一種量：unit vectors 上的 1 − dot（decision 007 第 2 點）。
    data = random_unit(1, 1000, 16)
    queries = random_unit(2, 30, 16)
    adapter = built(adapter_cls, data)
    adapter.set_ef_search(64)
    ids, dists = adapter.search(queries, 10)
    exact = 1 - np.einsum("qd,qkd->qk", queries.astype(np.float64), data[ids].astype(np.float64))
    np.testing.assert_allclose(dists, exact, atol=1e-5)


def test_high_ef_recall_is_near_perfect(adapter_cls):
    data = random_unit(1, 2000, 16)
    queries = random_unit(2, 100, 16)
    adapter = built(adapter_cls, data, m=12, ef_construction=100, seed=3)
    adapter.set_ef_search(200)
    ids, _ = adapter.search(queries, 10)
    assert recall_at_k(ids, exact_ids(data, queries, 10), 10) >= 0.99


def test_small_graph_with_large_ef_equals_exact(adapter_cls):
    # ef ≥ n 時 search 會窮舉 entry point 可達的所有節點；這組小圖在此 seed 下全部可達，
    # 所以三者都應與 exact 一致（隨機資料沒有 tie）。
    m, n = 8, 17
    data = random_unit(30, n, 8)
    queries = random_unit(31, 20, 8)
    adapter = built(adapter_cls, data, m=m, ef_construction=32, seed=1)
    adapter.set_ef_search(n)
    ids, _ = adapter.search(queries, n)
    np.testing.assert_array_equal(ids, exact_ids(data, queries, n))


def test_sweep_semantics_repeated_set_ef_search(adapter_cls):
    # 同一個 adapter 以非單調順序反覆 set_ef_search，每次都等同於新建後設定一次（006）。
    data = random_unit(13, 400, 16)
    queries = random_unit(14, 20, 16)
    adapter = built(adapter_cls, data)

    def fresh(ef):
        other = built(adapter_cls, data)
        other.set_ef_search(ef)
        return other.search(queries, 10)

    for ef in (64, 10, 16, 32):
        adapter.set_ef_search(ef)
        got_ids, got_dists = adapter.search(queries, 10)
        want_ids, want_dists = fresh(ef)
        np.testing.assert_array_equal(got_ids, want_ids)
        np.testing.assert_array_equal(got_dists.view(np.uint32), want_dists.view(np.uint32))

    # 設定失敗時保留前一個值（值與型別錯誤都一樣）。
    adapter.set_ef_search(64)
    with pytest.raises(ValueError):
        adapter.set_ef_search(0)
    with pytest.raises(TypeError):
        adapter.set_ef_search(None)
    np.testing.assert_array_equal(adapter.search(queries, 10)[0], fresh(64)[0])
    assert not np.array_equal(fresh(10)[0], fresh(64)[0])


def test_ef_can_be_set_before_build(adapter_cls):
    # ef = 37 避開 hnswlib 預設 10 與 FAISS 預設 16，否則「build 沒套用 ef」的錯誤抓不到。
    data = random_unit(13, 400, 16)
    queries = random_unit(14, 50, 16)
    early = adapter_cls(m=8, ef_construction=64, seed=5)
    early.set_ef_search(37)
    early.build(data)
    late = built(adapter_cls, data)
    late.set_ef_search(37)
    np.testing.assert_array_equal(early.search(queries, 10)[0], late.search(queries, 10)[0])
    if adapter_cls is Hnswlib:
        assert early._index.ef == 37
    if adapter_cls is FaissHnsw:
        assert early._index.hnsw.efSearch == 37


def test_search_requires_set_ef_search(adapter_cls):
    adapter = built(adapter_cls, random_unit(15, 50, 4))
    with pytest.raises(RuntimeError, match="set_ef_search"):
        adapter.search(random_unit(16, 2, 4), 5)


def test_params_must_be_integers(adapter_cls):
    for kwargs in (
        dict(m=None, ef_construction=64, seed=5),
        dict(m=8, ef_construction=None, seed=5),
        dict(m=8, ef_construction=64, seed=None),
        dict(m=8.0, ef_construction=64, seed=5),
    ):
        with pytest.raises(TypeError):
            adapter_cls(**kwargs)


def test_set_ef_search_validates_immediately(adapter_cls):
    adapter = adapter_cls(m=8, ef_construction=64, seed=5)
    for bad in (10.5, None):
        with pytest.raises(TypeError):
            adapter.set_ef_search(bad)
    for bad in (0, -3):
        with pytest.raises(ValueError):
            adapter.set_ef_search(bad)


def test_query_dimension_mismatch_raises_value_error(adapter_cls):
    # hnswlib 原生不檢查（會越界讀取）、FAISS 只用 assert；adapter 一律回 ValueError（decision 007）。
    adapter = built(adapter_cls, random_unit(1, 100, 8))
    adapter.set_ef_search(16)
    for shape in ((3, 4), (2, 16)):
        with pytest.raises(ValueError, match="dimension"):
            adapter.search(np.ones(shape, dtype=np.float32), 5)


def test_same_seed_is_deterministic(adapter_cls):
    data = random_unit(6, 500, 8)
    queries = random_unit(7, 20, 8)
    a, b = built(adapter_cls, data, seed=9), built(adapter_cls, data, seed=9)
    a.set_ef_search(20)
    b.set_ef_search(20)
    ids_a, dists_a = a.search(queries, 10)
    ids_b, dists_b = b.search(queries, 10)
    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_array_equal(dists_a.view(np.uint32), dists_b.view(np.uint32))


def test_params_reach_the_library(adapter_cls):
    # M、efConstruction、seed 都必須真的傳進 library（不依賴 library default，decision 007 第 5 點）。
    data = random_unit(13, 400, 16)
    queries = random_unit(14, 200, 16)  # 200 筆：seed 變體也有足夠多的列不同

    def result(**params):
        adapter = built(adapter_cls, data, **params)
        adapter.set_ef_search(16)
        return adapter.search(queries, 10)[0]

    baseline = result(m=8, ef_construction=64, seed=5)
    for variant in (dict(m=4, ef_construction=64, seed=5), dict(m=8, ef_construction=8, seed=5), dict(m=8, ef_construction=64, seed=6)):
        assert not np.array_equal(result(**variant), baseline), variant

    adapter = built(adapter_cls, data, m=6, ef_construction=48, seed=5)
    if adapter_cls is RustHnsw:
        assert (adapter._index.m, adapter._index.ef_construction, adapter._index.seed) == (6, 48, 5)
    if adapter_cls is Hnswlib:
        assert (adapter._index.M, adapter._index.ef_construction) == (6, 48)
    if adapter_cls is FaissHnsw:
        assert adapter._index.hnsw.efConstruction == 48
        assert adapter._index.hnsw.nb_neighbors(0) == 12


def force_multithreaded_defaults(adapter):
    """模擬環境把執行緒數設成多執行緒：FAISS 的 OpenMP 與 hnswlib 的 index 預設。"""
    faiss.omp_set_num_threads(4)
    if isinstance(adapter, Hnswlib) and adapter._index is not None:
        adapter._index.num_threads = 4


def test_build_and_search_are_single_threaded(adapter_cls):
    # 單執行緒時所有 CPU time 都在呼叫的 thread 上；其他 thread 的 CPU time 為 0，與機器負載無關。
    data = random_unit(40, 10000, 32)
    queries = random_unit(41, 3000, 32)
    adapter = adapter_cls(m=8, ef_construction=40, seed=1)
    force_multithreaded_defaults(adapter)
    other, total = other_thread_cpu(lambda: adapter.build(data))
    assert other <= 0.02 * total + 1e-3, (other, total)

    force_multithreaded_defaults(adapter)
    adapter.set_ef_search(100)
    force_multithreaded_defaults(adapter)
    other, total = other_thread_cpu(lambda: adapter.search(queries, 10))
    assert other <= 0.02 * total + 1e-3, (other, total)


def test_faiss_search_from_another_thread_is_single_threaded():
    # OpenMP 的 nthreads 是 per-thread 設定：在 main thread build，在另一個 thread search 仍須單執行緒。
    data = random_unit(40, 10000, 32)
    queries = random_unit(41, 3000, 32)
    adapter = built(FaissHnsw, data, m=8, ef_construction=40, seed=1)
    adapter.set_ef_search(100)
    measured = {}

    def worker():
        measured["cpu"] = other_thread_cpu(lambda: adapter.search(queries, 10))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    other, total = measured["cpu"]
    assert other <= 0.02 * total + 1e-3, (other, total)


def test_other_thread_cpu_detects_parallel_execution():
    # 正對照：多執行緒時其他 thread 的 CPU time 佔大部分，證明上面的單執行緒斷言有鑑別力。
    data = random_unit(40, 10000, 32)
    queries = random_unit(41, 3000, 32)

    index = hnswlib.Index(space="ip", dim=32)
    index.init_index(max_elements=len(data), ef_construction=40, M=8, random_seed=1)
    other, total = other_thread_cpu(lambda: index.add_items(data, np.arange(len(data)), num_threads=4))
    assert other > 0.3 * total, (other, total)

    faiss.omp_set_num_threads(4)
    f = faiss.IndexHNSWFlat(32, 8, faiss.METRIC_INNER_PRODUCT)
    f.add(data)
    f.hnsw.efSearch = 100
    other, total = other_thread_cpu(lambda: f.search(queries, 10))
    assert other > 0.3 * total, (other, total)


# ---- 各 library 的轉換（decision 007 第 2–4、7 點） ----


def short_result_fixture():
    data = prepare_vectors(duplicate_data(12, 300, 6))
    queries = prepare_vectors(np.random.default_rng(13).standard_normal((10, 6)))
    return data, queries


def test_rust_and_faiss_pad_short_results_with_minus_one_and_inf():
    data, queries = short_result_fixture()
    for cls in (RustHnsw, FaissHnsw):
        adapter = built(cls, data, m=4, ef_construction=20, seed=12)
        adapter.set_ef_search(300)
        ids, dists = adapter.search(queries, 80)
        check_search_result(ids, dists, num_queries=10, k=80)
        assert (ids == -1).any(), cls.name
        assert np.all(dists[ids == -1] == np.inf), cls.name


def test_faiss_adapter_converts_similarity_and_sentinel():
    # 原生 FAISS 回傳 similarity 與 −FLT_MAX；adapter 轉成 1 − ip 與 +inf，ids 不變。
    data, queries = short_result_fixture()
    adapter = built(FaissHnsw, data, m=4, ef_construction=20, seed=12)
    adapter.set_ef_search(300)
    ids, dists = adapter.search(queries, 80)

    raw = faiss.IndexHNSWFlat(6, 4, faiss.METRIC_INNER_PRODUCT)
    raw.hnsw.efConstruction = 20
    raw.hnsw.rng = faiss.RandomGenerator(12)
    raw.add(data)
    raw.hnsw.efSearch = 300
    raw_sims, raw_ids = raw.search(queries, 80)

    np.testing.assert_array_equal(ids, raw_ids)
    valid = raw_ids >= 0
    np.testing.assert_array_equal(dists[valid], (1 - raw_sims[valid]).astype(np.float32))
    assert np.all(raw_sims[~valid] == -np.finfo(np.float32).max)
    assert np.all(dists[~valid] == np.inf)


def test_hnswlib_adapter_raises_incomplete_results_without_fallback():
    data, queries = short_result_fixture()
    adapter = built(Hnswlib, data, m=4, ef_construction=20, seed=12)
    adapter.set_ef_search(300)
    recorder = CallRecorder(adapter._index)
    adapter._index = recorder
    with pytest.raises(IncompleteResultsError):
        adapter.search(queries, 80)
    assert issubclass(IncompleteResultsError, RuntimeError)
    # 整批只呼叫一次原生 knn_query，失敗後沒有改成逐筆查詢（007 §3）。
    assert recorder.calls == [(10, 6)]
    # 能完整回答的 query 仍正常回傳。
    ids, dists = adapter.search(queries[:1], 5)
    check_search_result(ids, dists, num_queries=1, k=5)


def test_hnswlib_adapter_passes_other_errors_through():
    # 只有「結果不足 k」的訊息會轉成 IncompleteResultsError，其他 RuntimeError 原樣拋出。
    class FailingIndex:
        def knn_query(self, queries, k, num_threads):
            raise RuntimeError("some other hnswlib failure")

    adapter = built(Hnswlib, random_unit(1, 100, 8))
    adapter.set_ef_search(16)
    adapter._index = FailingIndex()
    with pytest.raises(RuntimeError, match="some other hnswlib failure") as info:
        adapter.search(random_unit(2, 3, 8), 5)
    assert not isinstance(info.value, IncompleteResultsError)


def test_ef_below_k_semantics_per_adapter():
    # Rust、hnswlib：實際 ef = max(ef, k)；FAISS：efSearch < k 的結果不同（decision 007 第 4 點）。
    data = random_unit(0, 500, 16)
    queries = random_unit(1, 5, 16)
    for cls, equivalent in ((RustHnsw, True), (Hnswlib, True), (FaissHnsw, False)):
        adapter = built(cls, data, m=8, ef_construction=40, seed=1)
        adapter.set_ef_search(1)
        low = adapter.search(queries, 10)[0]
        adapter.set_ef_search(10)
        at_k = adapter.search(queries, 10)[0]
        assert np.array_equal(low, at_k) == equivalent, cls.name


def test_tie_order_is_passed_through_unchanged():
    # adapter 不重新排序 tie（decision 007 第 7 點）。
    query = prepare_vectors([[1.0, 0.0]])
    expected = {RustHnsw: [0, 1, 2, 3, 4], Hnswlib: [0, 1, 2, 3, 4], FaissHnsw: [4, 3, 2, 1, 0]}
    for cls, order in expected.items():
        adapter = built(cls, tie_data(), m=4, ef_construction=20, seed=1)
        adapter.set_ef_search(50)
        ids, dists = adapter.search(query, 5)
        assert ids[0].tolist() == order, cls.name
        assert np.all(dists == 0), cls.name


# ---- Recall@1 的來源（decision 008 §7） ----


def test_first_column_equals_k1_answer_without_ties(adapter_cls):
    data = random_unit(41, 2000, 16)
    queries = random_unit(42, 200, 16)
    adapter = built(adapter_cls, data)
    for ef in (10, 37):
        adapter.set_ef_search(ef)
        np.testing.assert_array_equal(adapter.search(queries, 10)[0][:, 0], adapter.search(queries, 1)[0][:, 0])


def test_first_column_differs_from_k1_only_by_tie_order_on_duplicates(adapter_cls):
    # 200 個方向各 15 份完全相同的副本：k = 1 與第一欄的 id 可以不同，但距離必須相同。
    rng = np.random.default_rng(43)
    directions = prepare_vectors(rng.standard_normal((200, 16)))
    data = np.ascontiguousarray(directions[rng.permutation(np.arange(3000) % 200)])
    queries = random_unit(44, 300, 16)
    adapter = built(adapter_cls, data, m=16, ef_construction=100)
    adapter.set_ef_search(16)
    ids10, dists10 = adapter.search(queries, 10)
    ids1, dists1 = adapter.search(queries, 1)
    np.testing.assert_allclose(dists10[:, 0], dists1[:, 0], rtol=0, atol=1e-6)
    differs = ids10[:, 0] != ids1[:, 0]
    np.testing.assert_array_equal(data[ids10[differs, 0]], data[ids1[differs, 0]])


# ---- 原生序列化大小（decision 008 §8：只作輔助欄位） ----


def test_save_native_writes_the_library_format(adapter_cls, tmp_path):
    data = random_unit(45, 300, 8)
    adapter = built(adapter_cls, data)
    path = tmp_path / "index.bin"
    adapter.save_native(str(path))
    assert path.stat().st_size > data.nbytes

    if adapter_cls is RustHnsw:
        count = len(rust_hnsw.HnswIndex.load(str(path)))
    elif adapter_cls is Hnswlib:
        loaded = hnswlib.Index(space="ip", dim=8)
        loaded.load_index(str(path))
        count = loaded.get_current_count()
    else:
        count = faiss.read_index(str(path)).ntotal
    assert count == 300
