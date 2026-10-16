"""Advanced C 三個新 adapter（F `rust-hnsw-fp32g`、S `rust-hnsw-sq8`、Q `faiss-hnsw-sq8`）的合約測試（decision 019 §4）。

合約（019 §4 pytest 第 1 點）：輸出格式、build 的 unit-norm 檢查、單執行緒、決定性、batch 與 single 一致、
每次 search 只有一次 FFI 呼叫、n = 1 / 16（runner 的 null call 與 library warm-up）。
另外驗證登記項，以及成對臂的 graph 與既有 adapter 相同（019 §3：S 與 F、Q 與 P 是同一張圖；F 與 R 逐位元相同）。
與 FP32 語意等價的測試（tests/test_adapters.py 的 cosine distance、高 ef recall、小圖等於 exact）不套用到這三個 adapter。
這些是 correctness 測試，不是 benchmark：只用 synthetic 資料，CPU time 只用來確認單執行緒。
"""

import threading

import faiss
import numpy as np
import pytest

from benchmarks import config, profile, protocol, run, run_child
from benchmarks.adapters import (
    FaissHnsw,
    FaissHnswSq8,
    RustHnsw,
    RustHnswFp32g,
    RustHnswSq8,
    faiss_hnsw_flat,
    faiss_sq8_from_flat,
)
from benchmarks.interface import HnswAnnIndex, IncompleteResultsError, check_search_result, prepare_vectors
from helpers import other_thread_cpu, random_unit
from test_adapters import CallRecorder, built, force_multithreaded_defaults, short_result_fixture
from test_third_party_semantics import sign_cube

QUANT_ADAPTERS = {
    "rust-hnsw-fp32g": RustHnswFp32g,
    "rust-hnsw-sq8": RustHnswSq8,
    "faiss-hnsw-sq8": FaissHnswSq8,
}
LIBRARIES = {"rust-hnsw-fp32g": "rust_hnsw", "rust-hnsw-sq8": "rust_hnsw", "faiss-hnsw-sq8": "faiss"}


@pytest.fixture(params=list(QUANT_ADAPTERS))
def adapter_cls(request):
    return QUANT_ADAPTERS[request.param]


@pytest.fixture(autouse=True)
def restore_faiss_threads():
    before = faiss.omp_get_max_threads()
    yield
    faiss.omp_set_num_threads(before)


def same_bits(a, b) -> None:
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1].view(np.uint32), b[1].view(np.uint32))


# ---- 登記項（019 §4「登記項」） ----


def test_registered_in_config_run_and_profile():
    for name, cls in QUANT_ADAPTERS.items():
        path, library = config.IMPLEMENTATIONS[name]
        assert run_child._resolve(path) is cls
        assert cls.name == name
        assert library == LIBRARIES[name]
        assert library in run_child.MEASURED_LIBRARIES
    # 每個實作名稱都有 kernel 的事實紀錄（寫進 run.json）。
    assert set(run.DISTANCE_KERNELS) == set(config.IMPLEMENTATIONS)
    # 019 §8：只取樣 F、S；FAISS 沒有符號表，Q 不取樣。
    assert profile.ADAPTERS["rust-hnsw-fp32g"] is RustHnswFp32g
    assert profile.ADAPTERS["rust-hnsw-sq8"] is RustHnswSq8
    assert "faiss-hnsw-sq8" not in profile.ADAPTERS


def test_configs_accept_the_new_implementations():
    text = f"""
schema_version = 1
dataset = "smoke-gaussian"
implementations = {list(QUANT_ADAPTERS)!r}
ef_search = [10]
child_timeout_s = 60

[[sweep]]
n = [300]
m = [8]
ef_construction = [32]
seeds = [1]
shuffles = []
""".replace("'", '"')
    assert config.parse_config(text).implementations == tuple(QUANT_ADAPTERS)


# ---- 輸出格式 ----


def test_conforms_to_the_hnsw_interface(adapter_cls):
    data = random_unit(13, 400, 16)
    queries = random_unit(14, 20, 16)
    adapter = built(adapter_cls, data)
    assert isinstance(adapter, HnswAnnIndex)
    adapter.set_ef_search(32)
    ids, dists = adapter.search(queries, 10)
    assert ids.dtype == np.int64 and dists.dtype == np.float32
    check_search_result(ids, dists, num_queries=20, k=10)
    if adapter_cls is FaissHnswSq8:
        assert isinstance(adapter._index, faiss.IndexHNSWSQ)
        assert adapter._index.metric_type == faiss.METRIC_INNER_PRODUCT
    else:
        assert adapter._index.representation == adapter_cls._representation


def test_distances_stay_in_range_when_estimates_leave_it(adapter_cls):
    # query 就是資料點時，SQ8 的估計距離可能 < 0：S 在 Rust 內、Q 在 adapter 內夾到 [0, 2]（019 §3），
    # 格式檢查因此通過（否則 ValueError 會讓整個子 process 失敗，019 背景事實 4）。F 不夾，與 Core 逐位元相同。
    data = random_unit(5, 2000, 16)
    queries = np.ascontiguousarray(data[:300])
    adapter = built(adapter_cls, data)
    adapter.set_ef_search(64)
    ids, dists = adapter.search(queries, 10)
    check_search_result(ids, dists, num_queries=300, k=10)
    if adapter_cls is not RustHnswFp32g:
        assert dists.min() == 0.0 and dists.max() <= 2.0
    if adapter_cls is FaissHnswSq8:
        # 這組資料上 FAISS 原生的估計 similarity 確實 > 1，夾住是必要的。
        assert adapter._index.search(queries, 10)[0].max() > 1.0


def test_pads_short_results_with_minus_one_and_inf(adapter_cls):
    data, queries = short_result_fixture()
    adapter = built(adapter_cls, data, m=4, ef_construction=20, seed=12)
    adapter.set_ef_search(300)
    ids, dists = adapter.search(queries, 80)
    check_search_result(ids, dists, num_queries=10, k=80)
    assert (ids == -1).any()
    assert np.all(dists[ids == -1] == np.inf)
    assert np.all(np.isfinite(dists[ids >= 0]))


def test_faiss_sq8_converts_clips_and_pads_the_native_output():
    # adapter 的輸出 = clip(1 − 原生 similarity, 0, 2)，補齊改成 +inf；ids 原樣傳回（019 §3）。
    data, queries = short_result_fixture()
    adapter = built(FaissHnswSq8, data, m=4, ef_construction=20, seed=12)
    adapter.set_ef_search(300)
    ids, dists = adapter.search(queries, 80)

    faiss.omp_set_num_threads(1)
    flat = faiss_hnsw_flat(data, 4, 20, 12)
    raw = faiss_sq8_from_flat(flat, data, 4)
    del flat
    raw.hnsw.efSearch = 300
    raw_sims, raw_ids = raw.search(queries, 80)

    np.testing.assert_array_equal(ids, raw_ids)
    valid = raw_ids >= 0
    expected = np.clip(np.float32(1.0) - raw_sims, np.float32(0.0), np.float32(2.0))
    np.testing.assert_array_equal(dists[valid].view(np.uint32), expected[valid].view(np.uint32))
    assert np.all(raw_sims[~valid] == -np.finfo(np.float32).max)
    assert np.all(dists[~valid] == np.inf)


def test_faiss_sq8_clips_estimated_distances_above_two():
    # q = −x：估計的 similarity 會略小於 −1，原生距離 1 − ip > 2；adapter 必須把它夾到恰好 2（019 §3 的上界）。
    data = random_unit(1, 100, 8)
    queries = prepare_vectors(-data)
    adapter = built(FaissHnswSq8, data, m=4, ef_construction=20, seed=1)
    adapter.set_ef_search(100)
    ids, dists = adapter.search(queries, 100)

    faiss.omp_set_num_threads(1)
    flat = faiss_hnsw_flat(data, 4, 20, 1)
    raw = faiss_sq8_from_flat(flat, data, 4)
    del flat
    raw.hnsw.efSearch = 100
    raw_sims, raw_ids = raw.search(queries, 100)
    assert np.all(raw_ids >= 0) and (raw_sims < -1).any()  # 前提：原生的距離確實超過 2
    np.testing.assert_array_equal(ids, raw_ids)
    expected = np.clip(np.float32(1.0) - raw_sims, np.float32(0.0), np.float32(2.0))
    np.testing.assert_array_equal(dists.view(np.uint32), expected.view(np.uint32))
    assert dists.max() == np.float32(2.0) and (dists == np.float32(2.0)).sum() == (raw_sims < -1).sum()
    check_search_result(ids, dists, num_queries=len(queries), k=100)


# ---- build 的輸入檢查與參數 ----


def test_build_requires_prepared_vectors(adapter_cls):
    data = random_unit(1, 200, 8)
    for bad in (data * 1.05, data.astype(np.float64), np.asfortranarray(data)):
        with pytest.raises(ValueError, match="prepare_vectors"):
            adapter_cls(m=8, ef_construction=40, seed=1).build(bad)


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


def test_search_requires_set_ef_search(adapter_cls):
    adapter = built(adapter_cls, random_unit(15, 50, 4))
    with pytest.raises(RuntimeError, match="set_ef_search"):
        adapter.search(random_unit(16, 2, 4), 5)


def test_k_validation_is_uniform(adapter_cls):
    data = random_unit(1, 200, 8)
    adapter = built(adapter_cls, data)
    adapter.set_ef_search(16)
    queries = random_unit(2, 3, 8)
    for bad_k in (0, len(data) + 1):
        with pytest.raises(ValueError, match="k must satisfy") as info:
            adapter.search(queries, bad_k)
        assert not isinstance(info.value, IncompleteResultsError)
    assert adapter.search(queries, np.int64(10))[0].shape == (3, 10)


def test_query_dimension_mismatch_raises_value_error(adapter_cls):
    adapter = built(adapter_cls, random_unit(1, 100, 8))
    adapter.set_ef_search(16)
    for shape in ((3, 4), (2, 16)):
        with pytest.raises(ValueError, match="dimension"):
            adapter.search(np.ones(shape, dtype=np.float32), 5)


def test_sweep_semantics_repeated_set_ef_search(adapter_cls):
    # runner 對同一個 adapter 依序設定每個 ef：每次都等同於新建後設定一次（006）。ef 也可以在 build 之前設定。
    data = random_unit(13, 400, 16)
    queries = random_unit(14, 20, 16)
    adapter = built(adapter_cls, data)

    def fresh(ef, before_build=False):
        other = adapter_cls(m=8, ef_construction=64, seed=5)
        if before_build:
            other.set_ef_search(ef)
            other.build(data)
        else:
            other.build(data)
            other.set_ef_search(ef)
        return other.search(queries, 10)

    for ef in (64, 10, 16, 37):
        adapter.set_ef_search(ef)
        got = adapter.search(queries, 10)
        same_bits(got, fresh(ef))
        same_bits(got, fresh(ef, before_build=True))
    assert not np.array_equal(fresh(10)[0], fresh(64)[0])

    # 設定失敗時保留前一個值。
    adapter.set_ef_search(64)
    with pytest.raises(ValueError):
        adapter.set_ef_search(0)
    same_bits(adapter.search(queries, 10), fresh(64))


# ---- 決定性、batch 與 single、一次 FFI 呼叫 ----


def test_same_seed_is_deterministic(adapter_cls):
    data = random_unit(6, 500, 8)
    queries = random_unit(7, 20, 8)
    a, b = built(adapter_cls, data, seed=9), built(adapter_cls, data, seed=9)
    a.set_ef_search(20)
    b.set_ef_search(20)
    same_bits(a.search(queries, 10), b.search(queries, 10))


def test_batch_equals_single_queries(adapter_cls):
    # runner 的 batch_vs_single 檢查（008 §6.7）：整批 search 與逐筆 search 的 ids、distances 逐位元相同。
    data = random_unit(41, 2000, 16)
    queries = random_unit(42, 100, 16)
    adapter = built(adapter_cls, data)
    for ef in (10, 37):
        adapter.set_ef_search(ef)
        for k in (1, 10):
            batch = adapter.search(queries, k)
            singles = [adapter.search(queries[i : i + 1], k) for i in range(len(queries))]
            same_bits(batch, (np.vstack([s[0] for s in singles]), np.vstack([s[1] for s in singles])))


def test_one_native_call_per_search(adapter_cls):
    # 整批 queries 只做一次原生呼叫：adapter 內沒有 per-query 迴圈（002、007 §3）。
    adapter = built(adapter_cls, random_unit(13, 400, 16))
    adapter.set_ef_search(32)
    recorder = CallRecorder(adapter._index)
    adapter._index = recorder
    adapter.search(random_unit(14, 20, 16), 10)
    assert recorder.calls == [(20, 16)]


# ---- 單執行緒（007 §6；019 §4） ----


def test_build_and_search_are_single_threaded(adapter_cls):
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


def test_faiss_sq8_search_from_another_thread_is_single_threaded():
    # OpenMP 的 nthreads 是 per-thread 設定：在 main thread build，在另一個 thread search 仍須單執行緒（007 §6）。
    data = random_unit(40, 10000, 32)
    queries = random_unit(41, 3000, 32)
    adapter = built(FaissHnswSq8, data, m=8, ef_construction=40, seed=1)
    adapter.set_ef_search(100)
    measured = {}

    def worker():
        measured["cpu"] = other_thread_cpu(lambda: adapter.search(queries, 10))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    other, total = measured["cpu"]
    assert other <= 0.02 * total + 1e-3, (other, total)


# ---- n = 1 / 16：runner 的 null call 與 library warm-up（008 §6.3、§6.4；019 背景事實 5） ----


def test_runner_null_call_and_library_warmup(adapter_cls):
    data = random_unit(50, 300, 16)
    queries = random_unit(51, 5, 16)
    params = {"m": 16, "ef_construction": 200, "seed": 1}

    # library warm-up：前 16 列建 index、ef = 1、search 一次（與 run_child.main 相同）。
    warm = adapter_cls(**params)
    warm.build(np.ascontiguousarray(data[: protocol.LIBRARY_WARMUP_ROWS]))
    warm.set_ef_search(1)
    ids, dists = warm.search(queries[0:1], 1)
    check_search_result(ids, dists, num_queries=1, k=1)
    assert ids[0, 0] >= 0

    # null call：n = 1 的 index（SQ8 的每一維都是 vmin = vmax），沿用 run_child 本身的函式。
    record, samples = run_child.null_call(adapter_cls, params, data, queries)
    assert record["n"] == 1 and record["dim"] == 16
    assert samples.shape == (protocol.NULL_CALL["samples"],)
    one = adapter_cls(**params)
    one.build(np.ascontiguousarray(data[:1]))
    one.set_ef_search(1)
    ids, dists = one.search(queries, 1)
    check_search_result(ids, dists, num_queries=5, k=1)
    assert (ids == 0).all()


# ---- 成對臂是同一張圖（019 §3；C7a、C8 的前提）；參數確實傳進 library ----

PARAM_SETS = (dict(m=8, ef_construction=64, seed=5), dict(m=4, ef_construction=64, seed=5),
              dict(m=8, ef_construction=8, seed=5), dict(m=8, ef_construction=64, seed=6))


def rust_graph(index) -> tuple:
    """QuantHnswIndex 的 entry point、各點的 level 與各層的 neighbors。"""
    levels = [index.level(i) for i in range(len(index))]
    neighbors = [[index.neighbors(i, layer) for layer in range(levels[i] + 1)] for i in range(len(index))]
    return index.entry_point, levels, neighbors


def faiss_graph(index) -> tuple:
    hnsw = index.hnsw
    arrays = tuple(faiss.vector_to_array(getattr(hnsw, name)).tolist() for name in ("levels", "offsets", "neighbors"))
    return (hnsw.entry_point, hnsw.max_level, *arrays)


def test_fp32g_output_is_bitwise_rust_hnsw():
    # F 與 R 的 ids、distances 逐位元相同（019 §3）；不同參數得到不同結果，所以參數確實傳進 library 且沒有對調。
    data = random_unit(13, 400, 16)
    queries = random_unit(14, 200, 16)
    outputs = []
    for params in PARAM_SETS:
        f, r = built(RustHnswFp32g, data, **params), built(RustHnsw, data, **params)
        f.set_ef_search(16)
        r.set_ef_search(16)
        out = f.search(queries, 10)
        same_bits(out, r.search(queries, 10))
        outputs.append(out[0])
    assert all(not np.array_equal(outputs[0], other) for other in outputs[1:])

    # 會補齊的重複向量也一樣。
    dup, dup_queries = short_result_fixture()
    f, r = built(RustHnswFp32g, dup, m=4, ef_construction=20, seed=12), built(RustHnsw, dup, m=4, ef_construction=20, seed=12)
    f.set_ef_search(300)
    r.set_ef_search(300)
    same_bits(f.search(dup_queries, 80), r.search(dup_queries, 80))


def test_sq8_uses_the_same_graph_as_fp32g():
    data = random_unit(13, 400, 16)
    graphs = []
    for params in PARAM_SETS:
        s, f = built(RustHnswSq8, data, **params), built(RustHnswFp32g, data, **params)
        assert rust_graph(s._index) == rust_graph(f._index)
        graphs.append(rust_graph(s._index))
    assert all(graphs[0] != other for other in graphs[1:])


def test_faiss_sq8_uses_the_same_graph_as_faiss_hnsw():
    # Q 與 P 以相同參數建同一張 Flat graph，再移植（019 §3、C8）；efConstruction 與 layer-0 容量也一併相同。
    data = random_unit(13, 400, 16)
    graphs = []
    for params in PARAM_SETS:
        q, p = built(FaissHnswSq8, data, **params), built(FaissHnsw, data, **params)
        assert faiss_graph(q._index) == faiss_graph(p._index)
        assert q._index.hnsw.efConstruction == p._index.hnsw.efConstruction == params["ef_construction"]
        assert q._index.hnsw.nb_neighbors(0) == 2 * params["m"]
        graphs.append(faiss_graph(q._index))
    assert all(graphs[0] != other for other in graphs[1:])


def test_faiss_sq8_returns_the_faiss_hnsw_ids_when_quantization_preserves_the_ranking():
    # 019 §4「移植」第 3 點在 adapter 層的版本：量化不改變排序的資料上（前提由 test_third_party_semantics 檢查），
    # Q 與 P 的 adapter 回傳相同的 ids，包括 ef < k 的點。
    data, queries = sign_cube()
    for params in (dict(m=4, ef_construction=20, seed=1), dict(m=8, ef_construction=20, seed=2)):
        q, p = built(FaissHnswSq8, data, **params), built(FaissHnsw, data, **params)
        for ef in (1, 5, 10, 32, 256):
            q.set_ef_search(ef)
            p.set_ef_search(ef)
            for k in (1, 10, 50):
                np.testing.assert_array_equal(q.search(queries, k)[0], p.search(queries, k)[0], err_msg=f"{params} ef = {ef}, k = {k}")


# ---- 原生序列化（019 §4：Rust 的兩個新 adapter 沒有 save_native；Q 有） ----


def test_rust_quant_adapters_have_no_save_native():
    for cls in (RustHnswFp32g, RustHnswSq8):
        assert not hasattr(cls, "save_native")
        assert not hasattr(built(cls, random_unit(45, 50, 8)), "save_native")


def test_faiss_sq8_save_native_writes_a_readable_index(tmp_path):
    data = random_unit(45, 300, 8)
    queries = random_unit(46, 20, 8)
    adapter = built(FaissHnswSq8, data)
    adapter.set_ef_search(32)
    path = tmp_path / "index.bin"
    adapter.save_native(str(path))
    assert path.stat().st_size > 0
    loaded = faiss.read_index(str(path))
    assert isinstance(loaded, faiss.IndexHNSWSQ)
    assert loaded.ntotal == 300 and loaded.is_trained
    loaded.hnsw.efSearch = 32
    np.testing.assert_array_equal(loaded.search(queries, 10)[1], adapter.search(queries, 10)[0])
