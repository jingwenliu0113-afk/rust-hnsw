"""Rust HNSW 的 Python binding、efSearch 介面與 RustHnsw adapter（decision 006）。

需要先 build binding：`./scripts/test.sh`（或 `maturin develop`）。
synthetic 資料只用於 correctness，不作 benchmark 結論。
"""

import threading
import time

import numpy as np
import pytest

import rust_hnsw
from benchmarks.adapters import RustExact, RustHnsw
from benchmarks.interface import AnnIndex, HnswAnnIndex, check_search_result
from benchmarks.metrics import recall_at_k
from helpers import duplicate_data, random_unit


def circle(degrees):
    rad = np.deg2rad(degrees)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32)


def build(data, **params):
    index = rust_hnsw.HnswIndex(data.shape[1], **params)
    index.add(data)
    return index


# ---- 建構與屬性 ----


def test_defaults_match_rust_params():
    index = rust_hnsw.HnswIndex(4)
    assert (index.dim, index.m, index.ef_construction, index.seed) == (4, 16, 200, 100)
    assert len(index) == 0
    assert index.max_level is None


def test_explicit_params_and_max_level():
    index = build(random_unit(0, 300, 8), m=4, ef_construction=40, seed=7)
    assert (index.m, index.ef_construction, index.seed) == (4, 40, 7)
    assert len(index) == 300
    # level 只由 seed、m、n 決定（004）；3 由獨立的 SplitMix64 參考計算交叉確認。
    assert index.max_level == 3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(dim=0), "dimension must be positive"),
        (dict(dim=4, m=1), "m must be in"),
        (dict(dim=4, m=2**31), "m must be in"),
        (dict(dim=4, ef_construction=0), "ef_construction must be in"),
    ],
)
def test_invalid_params_raise_value_error(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rust_hnsw.HnswIndex(**kwargs)


# ---- 搜尋 ----


def test_hand_verifiable_unit_circle_through_python():
    # 與 Rust 測試相同：M = 2 → M0 = 4，n = 5 ≤ M0 + 1，ef ≥ n 時必須與 exact 一致。
    index = build(circle([0, 30, 60, 90, 180]), m=2, ef_construction=10, seed=1)
    ids, dists = index.search(circle([10, 100]), 5, 5)
    assert ids.dtype == np.int64
    assert dists.dtype == np.float32
    assert ids.shape == dists.shape == (2, 5)
    np.testing.assert_array_equal(ids, [[0, 1, 2, 3, 4], [3, 2, 1, 4, 0]])
    np.testing.assert_allclose(dists[0], 1 - np.cos(np.deg2rad([10, 20, 50, 80, 170])), atol=1e-6)


def test_recall_against_exact_on_synthetic_data():
    data = random_unit(1, 2000, 16)
    queries = random_unit(2, 100, 16)
    exact = rust_hnsw.ExactIndex(16)
    exact.add(data)
    true_ids, _ = exact.search(queries, 10)

    index = build(data, m=12, ef_construction=100, seed=3)
    recalls = [recall_at_k(index.search(queries, 10, ef)[0], true_ids, 10) for ef in (10, 40, 160)]
    assert recalls[-1] >= 0.95, recalls
    # 這個固定資料集約為 0.88 對 1.0。這裡檢查 ef 有傳進核心，不是主張 recall 逐點嚴格單調
    # （SPEC 只要求整體改善或飽和）。
    assert recalls[0] < recalls[-1], recalls


def test_ef_below_k_is_raised_to_exactly_k():
    # 使用 recall 測試的資料集：ef 在 10 與 20 之間結果確實不同。
    index = build(random_unit(1, 2000, 16), m=12, ef_construction=100, seed=3)
    queries = random_unit(2, 20, 16)
    low = index.search(queries, 10, 1)
    assert low[0].shape == (20, 10)
    np.testing.assert_array_equal(low[0], index.search(queries, 10, 10)[0])
    # 提高到恰好 k（004、006：max(efSearch, k)），不是更大的值。
    assert not np.array_equal(low[0], index.search(queries, 10, 20)[0])


def test_short_results_are_padded_with_minus_one_and_inf():
    data = duplicate_data(12, 300, 6)
    index = build(data, m=4, ef_construction=20, seed=12)
    queries = np.ascontiguousarray(np.random.default_rng(13).standard_normal((10, 6)), dtype=np.float32)
    ids, dists = index.search(queries, 80, 300)
    check_search_result(ids, dists, num_queries=10, k=80)
    assert (ids == -1).any(), "this dataset should produce padded rows (decision 004)"
    assert np.all(np.isinf(dists[ids == -1]))


def test_empty_query_batch_and_empty_index():
    index = rust_hnsw.HnswIndex(2)
    with pytest.raises(ValueError, match="index is empty"):
        index.search(circle([0]), 1, 10)
    index.add(circle([0, 90]))
    ids, dists = index.search(np.zeros((0, 2), dtype=np.float32), 1, 10)
    assert ids.shape == dists.shape == (0, 1)


def test_input_validation_matches_exact_binding():
    index = rust_hnsw.HnswIndex(2)
    with pytest.raises(TypeError):
        index.add(np.ones((2, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="C-contiguous"):
        index.add(np.asfortranarray(np.arange(1, 7, dtype=np.float32).reshape(3, 2)))
    with pytest.raises(ValueError, match="dimension mismatch"):
        index.add(np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="row 1 is a zero vector"):
        index.add(np.array([[1, 0], [0, 0]], dtype=np.float32))
    assert len(index) == 0
    index.add(circle([0, 90]))
    with pytest.raises(ValueError, match="k must satisfy"):
        index.search(circle([0]), 3, 10)
    with pytest.raises(ValueError, match="C-contiguous"):
        index.search(np.ones((4, 4), dtype=np.float32)[:, ::2], 1, 10)


def test_same_seed_gives_identical_results():
    data = random_unit(6, 500, 8)
    queries = random_unit(7, 20, 8)
    a = build(data, m=6, ef_construction=40, seed=9)
    b = build(data, m=6, ef_construction=40, seed=9)
    ids_a, dists_a = a.search(queries, 10, 20)
    ids_b, dists_b = b.search(queries, 10, 20)
    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_array_equal(dists_a.view(np.uint32), dists_b.view(np.uint32))


# ---- GIL（002、006：計算期間釋放 GIL，與 hnswlib 一致） ----


def main_thread_ticks_during(call):
    """在背景 thread 執行 `call`，回傳 main thread 在其執行期間中段 60% 內跑了幾圈。"""
    window = {}

    def worker():
        window["start"] = time.perf_counter()
        call()
        window["end"] = time.perf_counter()

    ticks = []
    thread = threading.Thread(target=worker)
    thread.start()
    while thread.is_alive():
        ticks.append(time.perf_counter())
    thread.join()
    start, end = window["start"], window["end"]
    lo, hi = start + 0.2 * (end - start), end - 0.2 * (end - start)
    return sum(lo < t < hi for t in ticks)


def test_search_releases_gil():
    data = random_unit(20, 3000, 32)
    index = build(data, m=8, ef_construction=40, seed=1)
    assert main_thread_ticks_during(lambda: index.search(data, 10, 200)) > 0


def test_add_releases_gil():
    data = random_unit(21, 3000, 32)
    index = rust_hnsw.HnswIndex(32, m=8, ef_construction=100, seed=1)
    assert main_thread_ticks_during(lambda: index.add(data)) > 0


# ---- save / load ----


def test_save_load_round_trip(tmp_path):
    data = random_unit(8, 1200, 8)
    index = build(data, m=4, ef_construction=40, seed=7)
    path = tmp_path / "index.rhnsw"
    index.save(path)
    loaded = rust_hnsw.HnswIndex.load(path)

    assert (loaded.dim, loaded.m, loaded.ef_construction, loaded.seed) == (8, 4, 40, 7)
    assert len(loaded) == len(index)
    assert index.max_level == 5  # seed 7、m 4、n 1200（獨立參考計算交叉確認）
    assert loaded.max_level == index.max_level
    queries = random_unit(9, 50, 8)
    for ef in (1, 16, 64):
        ids_a, dists_a = index.search(queries, 10, ef)
        ids_b, dists_b = loaded.search(queries, 10, ef)
        np.testing.assert_array_equal(ids_a, ids_b)
        np.testing.assert_array_equal(dists_a.view(np.uint32), dists_b.view(np.uint32))

    # 繼續插入後仍與從未存檔的 index 相同（decision 005）。
    more = random_unit(10, 100, 8)
    index.add(more)
    loaded.add(more)
    again_a, again_b = tmp_path / "a.rhnsw", tmp_path / "b.rhnsw"
    index.save(str(again_a))
    loaded.save(str(again_b))
    assert again_a.read_bytes() == again_b.read_bytes()


def test_load_errors_are_mapped(tmp_path):
    with pytest.raises(FileNotFoundError):
        rust_hnsw.HnswIndex.load(tmp_path / "missing.rhnsw")

    path = tmp_path / "index.rhnsw"
    build(random_unit(11, 50, 4), m=4, ef_construction=20, seed=1).save(path)
    original = path.read_bytes()

    path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="truncated"):
        rust_hnsw.HnswIndex.load(path)

    corrupted = bytearray(original)
    corrupted[100] ^= 0x01
    path.write_bytes(bytes(corrupted))
    with pytest.raises(ValueError):
        rust_hnsw.HnswIndex.load(path)

    path.write_bytes(b"not an index file")
    with pytest.raises(ValueError, match="bad magic"):
        rust_hnsw.HnswIndex.load(path)


def test_save_to_a_directory_raises_os_error(tmp_path):
    index = build(random_unit(12, 20, 4), m=4, ef_construction=20, seed=1)
    target = tmp_path / "index.rhnsw"
    target.mkdir()
    with pytest.raises(OSError):
        index.save(target)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.rhnsw"]


# ---- efSearch 介面與 adapter（decision 006） ----


def test_rust_hnsw_adapter_conforms_to_hnsw_interface():
    data = random_unit(13, 400, 16)
    queries = random_unit(14, 20, 16)
    adapter = RustHnsw(m=8, ef_construction=64, seed=5)
    assert isinstance(adapter, AnnIndex)
    assert isinstance(adapter, HnswAnnIndex)
    assert adapter.name == "rust-hnsw"
    adapter.build(data)
    adapter.set_ef_search(32)
    ids, dists = adapter.search(queries, 10)
    check_search_result(ids, dists, num_queries=20, k=10)

    direct = build(data, m=8, ef_construction=64, seed=5)
    np.testing.assert_array_equal(ids, direct.search(queries, 10, 32)[0])

    # sweep：同一個 adapter 反覆 set_ef_search（非單調順序），每次都要生效，結果與 binding 逐位元相同。
    for ef in (64, 10, 16, 1, 32):
        adapter.set_ef_search(ef)
        got_ids, got_dists = adapter.search(queries, 10)
        check_search_result(got_ids, got_dists, num_queries=20, k=10)
        want_ids, want_dists = direct.search(queries, 10, ef)
        np.testing.assert_array_equal(got_ids, want_ids)
        np.testing.assert_array_equal(got_dists.view(np.uint32), want_dists.view(np.uint32))

    # 設定失敗時保留前一個值（006「檢查後存下」）。
    low, high = direct.search(queries, 10, 10)[0], direct.search(queries, 10, 64)[0]
    assert not np.array_equal(low, high)
    adapter.set_ef_search(64)
    with pytest.raises(ValueError):
        adapter.set_ef_search(0)
    np.testing.assert_array_equal(adapter.search(queries, 10)[0], high)


def test_ef_search_must_be_set_explicitly_and_positive():
    adapter = RustHnsw(m=8, ef_construction=64, seed=5)
    adapter.build(random_unit(15, 50, 4))
    with pytest.raises(RuntimeError, match="set_ef_search"):
        adapter.search(random_unit(16, 2, 4), 5)
    for bad in (0, -3):
        with pytest.raises(ValueError, match="ef"):
            adapter.set_ef_search(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(m=None, ef_construction=64, seed=5),
        dict(m=8, ef_construction=None, seed=5),
        dict(m=8, ef_construction=64, seed=None),
        dict(m=8.0, ef_construction=64, seed=5),
    ],
)
def test_rust_hnsw_adapter_requires_integer_params(kwargs):
    # 三套實作的預設參數不同；adapter 不得默默套用 binding 的預設值（006）。
    with pytest.raises(TypeError):
        RustHnsw(**kwargs)


def test_set_ef_search_rejects_non_integers_immediately():
    adapter = RustHnsw(m=8, ef_construction=64, seed=5)
    for bad in (10.5, 10.0, None):
        with pytest.raises(TypeError):
            adapter.set_ef_search(bad)


def test_exact_adapter_is_not_an_hnsw_index():
    assert not isinstance(RustExact(), HnswAnnIndex)
