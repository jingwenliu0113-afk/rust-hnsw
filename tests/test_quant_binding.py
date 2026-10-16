"""Advanced C 的 Python binding：`QuantHnswIndex` 與 `sq8_encode`（decision 019 §4「Binding」）。

需要先 build binding：`./scripts/test.sh`（或 `maturin develop`）。
只用 synthetic 資料，只作 correctness，不作 benchmark 結論；不碰 datasets/ 的真實資料（019 §9）。

「與 Rust 端一致」以兩種方式檢查：
- F 對 Core 的 `HnswIndex` binding 逐位元相同（同一張 graph、同一個 FP32 kernel，019 §3）。
- 以 numpy 依 019 §3 的文字逐位元重寫正規化、codec 與兩個 kernel 的算術（f32 依序加總以 `np.cumsum` 表示），
  binding 的輸出必須與它逐位元相同。
"""

import threading
import time

import numpy as np
import pytest

import rust_hnsw
from benchmarks.interface import check_search_result
from helpers import duplicate_data, random_unit, tie_data

QuantHnswIndex = rust_hnsw.QuantHnswIndex
REPRESENTATIONS = ("fp32", "sq8")


def circle(degrees):
    rad = np.deg2rad(degrees)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32)


def core(data, m, ef_construction, seed):
    index = rust_hnsw.HnswIndex(data.shape[1], m=m, ef_construction=ef_construction, seed=seed)
    index.add(data)
    return index


def assert_same_output(a, b):
    """兩個 `(ids, distances)` 逐位元相同。"""
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1].view(np.uint32), b[1].view(np.uint32))


def graph(index):
    """entry point 與每個節點各層的鄰居清單（測試用的 graph 存取方法）。"""
    return index.entry_point, [
        [index.neighbors(i, level) for level in range(index.level(i) + 1)] for i in range(len(index))
    ]


# ---- 依 019 §3 文字重寫的 numpy 參考（逐位元） ----


def rust_unit(x):
    """`distance::normalize` 的逐位元重寫：f64 依序累加平方和、開根號，各分量以 f64 相除後存成 f32。"""
    x64 = np.asarray(x, dtype=np.float32).astype(np.float64)
    norm = np.sqrt(np.cumsum(x64 * x64, axis=1)[:, -1:])
    return (x64 / norm).astype(np.float32)


def numpy_sq8(x):
    """019 §3 的 codec：逐維 vmin / vmax，vdiff = vmax − vmin（f32）；vdiff = 0 → 0，否則 (255·clip(t)) 向零截斷。"""
    vmin, vmax = x.min(axis=0), x.max(axis=0)
    vdiff = vmax - vmin
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip((x - vmin) / vdiff, np.float32(0), np.float32(1))
    codes = np.where(vdiff == 0, np.float32(0), np.float32(255) * t).astype(np.uint8)
    return vmin, vdiff, codes


def f32_sum(x):
    """最後一軸依序以 f32 加總（與 `iter().sum::<f32>()` 相同的順序與捨入）。"""
    return np.cumsum(x, axis=-1, dtype=np.float32)[..., -1]


def fp32_raw(unit_queries, unit_data):
    """F 的 kernel：`1 − dot`，依序以 f32 累加。shape 為 `(num_queries, n)`。"""
    return np.float32(1) - f32_sum(unit_queries[:, None, :] * unit_data[None, :, :])


def sq8_raw(unit_queries, codes, vmin, vdiff):
    """S 的 kernel（019 §3）：`1 − (b + Σ_j w_j · code_j)`，夾住之前的原始值。shape 為 `(num_queries, n)`。"""
    w = unit_queries * (vdiff / np.float32(255))
    q64, vmin64, vdiff64 = unit_queries.astype(np.float64), vmin.astype(np.float64), vdiff.astype(np.float64)
    b = np.cumsum(q64 * (vmin64 + 0.5 * vdiff64 / 255.0), axis=1)[:, -1].astype(np.float32)
    total = f32_sum(w[:, None, :] * codes[None, :, :].astype(np.float32))
    return np.float32(1) - (b[:, None] + total)


def clamp(raw):
    return np.clip(raw, np.float32(0), np.float32(2))


def raw_distances(index, data, queries):
    """以 numpy 參考計算 index 的原始距離（data 為 build 時的輸入）。"""
    unit_queries = rust_unit(queries)
    if index.representation == "fp32":
        return fp32_raw(unit_queries, rust_unit(data))
    vmin, vdiff = index.codebook()
    return sq8_raw(unit_queries, index.codes(), vmin, vdiff)


def output(index, raw):
    """輸出的距離：F 不夾，S 夾到 [0, 2]（019 §3）。"""
    return raw if index.representation == "fp32" else clamp(raw)


def reference_exact(index, data, queries, k):
    """依原始距離的 (distance, id) 取前 k 個（001 的 tie-break），輸出距離依表示處理。"""
    raw = raw_distances(index, data, queries)
    ids = np.arange(raw.shape[1])
    order = np.stack([np.lexsort((ids, row))[:k] for row in raw]).astype(np.int64)
    return order, output(index, np.take_along_axis(raw, order, axis=1))


# ---- 建構與參數 ----


def test_build_requires_every_parameter_explicitly():
    data = random_unit(0, 50, 4)
    with pytest.raises(TypeError):
        QuantHnswIndex.build(data, 8, 40, 1)  # 缺 representation
    with pytest.raises(TypeError):
        QuantHnswIndex.build(data, m=8, ef_construction=40, representation="fp32")  # 缺 seed
    with pytest.raises(TypeError):
        QuantHnswIndex.build_pair(data, 8, 40)
    for bad in (dict(m=None), dict(m=8.0), dict(ef_construction=None), dict(seed=None), dict(seed=1.5)):
        kwargs = dict(m=8, ef_construction=40, seed=1, representation="fp32") | bad
        with pytest.raises(TypeError):
            QuantHnswIndex.build(data, **kwargs)
    with pytest.raises(TypeError):
        QuantHnswIndex.build(data, 8, 40, 1, None)
    with pytest.raises(TypeError, match="cannot create"):
        QuantHnswIndex()


@pytest.mark.parametrize("name", ["FP32", "Sq8", "int8", "", "fp16"])
def test_unknown_representation_raises_value_error(name):
    with pytest.raises(ValueError, match="representation must be"):
        QuantHnswIndex.build(random_unit(0, 10, 4), 8, 40, 1, name)


@pytest.mark.parametrize(
    "shape, params, message",
    [
        ((3, 0), (8, 40, 1), "dimension must be positive"),
        ((10, 4), (1, 40, 1), "m must be in"),
        ((10, 4), (2**31, 40, 1), "m must be in"),
        ((10, 4), (8, 0, 1), "ef_construction must be in"),
    ],
)
def test_invalid_build_params_raise_the_core_errors(shape, params, message):
    data = np.ones(shape, dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        QuantHnswIndex.build(data, *params, "sq8")
    with pytest.raises(ValueError, match=message):
        QuantHnswIndex.build_pair(data, *params)


def test_build_input_validation_matches_core():
    with pytest.raises(TypeError):
        QuantHnswIndex.build(np.ones((2, 2), dtype=np.float64), 8, 40, 1, "fp32")
    with pytest.raises(ValueError, match="C-contiguous"):
        QuantHnswIndex.build(np.asfortranarray(np.arange(1, 7, dtype=np.float32).reshape(3, 2)), 8, 40, 1, "fp32")
    with pytest.raises(ValueError, match="row 1 is a zero vector"):
        QuantHnswIndex.build(np.array([[1, 0], [0, 0]], dtype=np.float32), 8, 40, 1, "sq8")
    with pytest.raises(ValueError, match="row 0 contains NaN or Inf"):
        QuantHnswIndex.build_pair(np.array([[np.inf, 0], [0, 1]], dtype=np.float32), 8, 40, 1)


def test_attributes_and_representation_data():
    data = random_unit(1, 300, 8)
    fp32 = QuantHnswIndex.build(data, 8, 40, 1, "fp32")
    sq8 = QuantHnswIndex.build(data, 8, 40, 1, "sq8")
    for index, name in ((fp32, "fp32"), (sq8, "sq8")):
        assert (len(index), index.dim, index.representation) == (300, 8, name)
    assert fp32.codes() is None and fp32.codebook() is None

    codes = sq8.codes()
    assert codes.dtype == np.uint8 and codes.shape == (300, 8) and codes.flags.c_contiguous
    vmin, vdiff = sq8.codebook()
    assert vmin.dtype == vdiff.dtype == np.float32
    assert vmin.shape == vdiff.shape == (8,)
    # 回傳的是複製：修改它不影響 index。
    codes[:] = 0
    vmin[:] = 0
    assert sq8.codes().any() and sq8.codebook()[0].any()


# ---- 搜尋 ----


def test_hand_verifiable_unit_circle_through_python():
    # 與 Core 的 binding 測試相同：M = 2 → M0 = 4，n = 5 ≤ M0 + 1，ef ≥ n 時必須與 exact 一致。
    # 各點的角度差大，SQ8 的誤差（每維 ≤ vdiff / 510）不改變排序。
    want = np.array([[0, 1, 2, 3, 4], [3, 2, 1, 4, 0]])
    true = 1 - np.cos(np.deg2rad([[10, 20, 50, 80, 170], [10, 40, 70, 80, 100]]))
    for name, atol in (("fp32", 1e-6), ("sq8", 1e-2)):
        index = QuantHnswIndex.build(circle([0, 30, 60, 90, 180]), 2, 10, 1, name)
        ids, dists = index.search(circle([10, 100]), 5, 5)
        assert ids.dtype == np.int64 and dists.dtype == np.float32
        assert ids.shape == dists.shape == (2, 5)
        np.testing.assert_array_equal(ids, want)
        np.testing.assert_allclose(dists, true, atol=atol)


def equivalence_fixtures():
    """(名稱, data, (m, efC, seed), queries, [(k, ef), ...])。"""
    rng = np.random.default_rng(13)
    random16 = random_unit(1, 2000, 16)
    multi = random_unit(8, 1200, 8)
    duplicates = duplicate_data(12, 300, 6)
    return [
        ("random 16-d", random16, (12, 100, 3), np.concatenate([random_unit(2, 40, 16), random16[:5]]),
         [(1, 1), (10, 10), (10, 40), (10, 160), (50, 64)]),
        ("multi-layer", multi, (4, 40, 7), np.concatenate([random_unit(9, 30, 8), multi[:5]]),
         [(1, 16), (10, 1), (10, 64)]),
        ("duplicates (padded)", duplicates, (4, 20, 12),
         np.ascontiguousarray(rng.standard_normal((10, 6)), dtype=np.float32), [(10, 10), (80, 300)]),
        ("ties", tie_data(), (4, 20, 0), np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32),
         [(5, 5), (10, 10)]),
    ]


@pytest.mark.parametrize("name, data, params, queries, grid", equivalence_fixtures(),
                         ids=[f[0] for f in equivalence_fixtures()])
def test_fp32_arm_is_bitwise_identical_to_the_core_binding(name, data, params, queries, grid):
    reference = core(data, *params)
    fp32 = QuantHnswIndex.build(data, *params, "fp32")
    pair_fp32, _ = QuantHnswIndex.build_pair(data, *params)
    assert fp32.entry_point is not None and fp32.level(fp32.entry_point) == reference.max_level
    for k, ef in grid:
        want = reference.search(queries, k, ef)
        assert_same_output(fp32.search(queries, k, ef), want)
        assert_same_output(pair_fp32.search(queries, k, ef), want)


@pytest.mark.parametrize("name, data, params, queries, grid", equivalence_fixtures(),
                         ids=[f[0] for f in equivalence_fixtures()])
def test_build_pair_matches_build_and_shares_one_graph(name, data, params, queries, grid):
    pair = QuantHnswIndex.build_pair(data, *params)
    assert [index.representation for index in pair] == ["fp32", "sq8"]
    built = [QuantHnswIndex.build(data, *params, rep) for rep in REPRESENTATIONS]
    # 四個 index 是同一張 graph（019 §3：建完 FP32 圖才轉換）。
    graphs = [graph(index) for index in (*pair, *built)]
    assert all(g == graphs[0] for g in graphs[1:])
    np.testing.assert_array_equal(pair[1].codes(), built[1].codes())
    for a, b in zip(pair[1].codebook(), built[1].codebook()):
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    for k, ef in grid:
        for a, b in zip(pair, built):
            assert_same_output(a.search(queries, k, ef), b.search(queries, k, ef))


def test_graph_accessors():
    data = random_unit(8, 1200, 8)
    index = QuantHnswIndex.build(data, 4, 40, 7, "sq8")
    entry = index.entry_point
    assert index.level(entry) == 5  # seed 7、m 4、n 1200，與 Core binding 測試的 max_level 相同
    assert max(index.level(i) for i in range(len(index))) == 5
    for i in range(len(index)):
        for level in range(index.level(i) + 1):
            neighbors = index.neighbors(i, level)
            assert len(neighbors) <= (8 if level == 0 else 4)
            assert len(set(neighbors)) == len(neighbors) and i not in neighbors
            assert all(0 <= j < len(index) and index.level(j) >= level for j in neighbors)
        assert index.neighbors(i, index.level(i) + 1) is None
    assert index.level(len(index)) is None and index.neighbors(len(index), 0) is None
    empty = QuantHnswIndex.build(np.zeros((0, 3), dtype=np.float32), 4, 20, 1, "fp32")
    assert empty.entry_point is None and len(empty) == 0


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_exact_search_is_bitwise_identical_to_an_independent_numpy_brute_force(representation):
    data = random_unit(3, 300, 16)
    # 存進去的向量本身當 query：原始距離接近 0，S 的原始值可能為負（輸出夾到 0）。
    queries = np.concatenate([random_unit(4, 15, 16), data[:10]])
    index = QuantHnswIndex.build(data, 8, 40, 5, representation)
    for k in (1, 10, 300):
        assert_same_output(index.exact_search(queries, k), reference_exact(index, data, queries, k))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_search_distances_follow_the_numpy_kernel_and_raw_order(representation):
    data = random_unit(5, 1000, 16)
    queries = np.concatenate([random_unit(6, 20, 16), data[:10]])
    index = QuantHnswIndex.build(data, 8, 64, 2, representation)
    raw = raw_distances(index, data, queries)
    saw_negative_raw = False
    for k, ef in ((1, 10), (10, 10), (10, 100), (40, 64)):
        ids, dists = index.search(queries, k, ef)
        got_raw = np.take_along_axis(raw, ids, axis=1)
        np.testing.assert_array_equal(dists.view(np.uint32), output(index, got_raw).view(np.uint32))
        # 每一列依原始距離的 (distance, id) 嚴格遞增：夾住只在輸出時做，不改變順序。
        assert np.all((got_raw[:, :-1] < got_raw[:, 1:])
                      | ((got_raw[:, :-1] == got_raw[:, 1:]) & (ids[:, :-1] < ids[:, 1:])))
        saw_negative_raw |= bool((got_raw < 0).any())
        if representation == "sq8":
            assert dists.min() >= 0 and dists.max() <= 2
    if representation == "sq8":
        assert saw_negative_raw, "the fixture should exercise the clamp"


def test_sq8_differs_from_fp32_but_stays_close():
    data = random_unit(5, 1000, 16)
    queries = random_unit(6, 20, 16)
    fp32, sq8 = QuantHnswIndex.build_pair(data, 8, 64, 2)
    (_, d_fp32), (_, d_sq8) = fp32.exact_search(queries, 10), sq8.exact_search(queries, 10)
    assert not np.array_equal(d_fp32, d_sq8)
    np.testing.assert_allclose(d_sq8, d_fp32, atol=0.05)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_small_graph_with_large_ef_equals_exact_search(representation):
    # n ≤ M0 + 1：layer 0 從不 prune，graph 連通；ef ≥ n 時走訪涵蓋所有點，結果必須等於同一個 kernel 的 brute force。
    for seed in range(3):
        data = random_unit(30 + seed, 17, 8)
        queries = np.concatenate([random_unit(40 + seed, 10, 8), data[:3]])
        index = QuantHnswIndex.build(data, 8, 16, seed, representation)
        for k in (1, 5, 17):
            assert_same_output(index.search(queries, k, 17), index.exact_search(queries, k))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_short_results_are_padded_with_minus_one_and_inf(representation):
    data = duplicate_data(12, 300, 6)
    index = QuantHnswIndex.build(data, 4, 20, 12, representation)
    queries = np.ascontiguousarray(np.random.default_rng(13).standard_normal((10, 6)), dtype=np.float32)
    ids, dists = index.search(queries, 80, 300)
    check_search_result(ids, dists, num_queries=10, k=80)
    assert (ids == -1).any(), "this dataset should produce padded rows (decision 004)"
    assert np.all(np.isinf(dists[ids == -1])) and np.all(np.isfinite(dists[ids != -1]))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_same_seed_gives_identical_results(representation):
    data = random_unit(6, 500, 8)
    queries = random_unit(7, 20, 8)
    a = QuantHnswIndex.build(data, 6, 40, 9, representation)
    b = QuantHnswIndex.build(data, 6, 40, 9, representation)
    assert graph(a) == graph(b)
    assert_same_output(a.search(queries, 10, 20), b.search(queries, 10, 20))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_search_input_validation_matches_core(representation):
    data = circle([0, 90])
    index = QuantHnswIndex.build(data, 4, 20, 1, representation)
    reference = core(data, 4, 20, 1)
    calls = {
        "search": (lambda q, k: index.search(q, k, 10), lambda q, k: reference.search(q, k, 10)),
        "search_stats": (lambda q, k: index.search_stats(q, k, 10), lambda q, k: reference.search(q, k, 10)),
        "exact_search": (lambda q, k: index.exact_search(q, k), lambda q, k: reference.search(q, k, 10)),
    }
    cases = [
        (np.ones((1, 2), dtype=np.float64), 1, TypeError, None),
        (np.ones((4, 4), dtype=np.float32)[:, ::2], 1, ValueError, "C-contiguous"),
        (np.asfortranarray(np.arange(1, 7, dtype=np.float32).reshape(3, 2)), 1, ValueError, "C-contiguous"),
        (np.ones((1, 3), dtype=np.float32), 1, ValueError, "dimension mismatch"),
        (np.array([[1, 0], [0, 0]], dtype=np.float32), 1, ValueError, "row 1 is a zero vector"),
        (np.array([[np.nan, 0]], dtype=np.float32), 1, ValueError, "row 0 contains NaN or Inf"),
        (circle([0]), 0, ValueError, "k must satisfy"),
        (circle([0]), 3, ValueError, "k must satisfy"),
    ]
    for name, (call, core_call) in calls.items():
        for queries, k, error, message in cases:
            for fn in (call, core_call):
                with pytest.raises(error, match=message):
                    fn(queries, k)
        ids, dists = call(np.zeros((0, 2), dtype=np.float32), 1)[:2]
        assert ids.shape == dists.shape == (0, 1), name

    empty = QuantHnswIndex.build(np.zeros((0, 2), dtype=np.float32), 4, 20, 1, representation)
    for fn in (lambda: empty.search(circle([0]), 1, 10), lambda: empty.search_stats(circle([0]), 1, 10),
               lambda: empty.exact_search(circle([0]), 1)):
        with pytest.raises(ValueError, match="index is empty"):
            fn()


# ---- search_stats（019 §7 的 D 與 V） ----


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_search_stats_returns_the_search_results(representation):
    data = random_unit(8, 1200, 8)
    queries = np.concatenate([random_unit(9, 30, 8), data[:5]])
    index = QuantHnswIndex.build(data, 4, 40, 7, representation)
    for k, ef in ((1, 1), (10, 10), (10, 64), (64, 64)):
        ids, dists, kernel_calls, visited_inserts = index.search_stats(queries, k, ef)
        assert_same_output((ids, dists), index.search(queries, k, ef))
        for counts in (kernel_calls, visited_inserts):
            assert counts.dtype == np.uint64 and counts.shape == (len(queries),)
        # 多層的圖：每個 query 至少評估 entry point 與 layer 0 的 k 個結果。
        assert np.all(kernel_calls >= k) and np.all(visited_inserts >= 1)


def test_search_stats_counts_every_visit_when_ef_covers_the_whole_graph():
    # n = 17 ≤ M0 + 1 = 17：layer 0 連通。ef ≥ n 時每個點都被展開一次：
    # V = 1（entry）+ Σ layer 0 的 degree；D = n + 上層 greedy 的評估數（只有一層時 D = n）。
    for seed, single_layer in ((1, True), (0, False)):
        data = random_unit(30, 17, 8)
        queries = np.concatenate([random_unit(31, 10, 8), data[:3]])
        pair = QuantHnswIndex.build_pair(data, 8, 16, seed)
        fp32 = pair[0]
        assert (fp32.level(fp32.entry_point) == 0) == single_layer
        degree_sum = sum(len(fp32.neighbors(i, 0)) for i in range(17))
        for index in pair:
            for k in (1, 10):
                _, _, kernel_calls, visited_inserts = index.search_stats(queries, k, 17)
                np.testing.assert_array_equal(visited_inserts, 1 + degree_sum)
                if single_layer:
                    np.testing.assert_array_equal(kernel_calls, 17)
                else:
                    assert np.all(kernel_calls > 17)


# ---- memory_accounting（019 §7 的 M_struct） ----


@pytest.mark.parametrize("n, dim", [(300, 8), (1000, 16), (17, 8), (1, 4), (50, 100)])
def test_memory_accounting_difference_is_3nd_minus_8d(n, dim):
    data = random_unit(70 + n, n, dim)
    keys = {"struct_bytes", "graph_bytes", "vectors_bytes", "codes_bytes", "codebook_bytes", "total_bytes"}
    pairs = {
        "build_pair": [index.memory_accounting() for index in QuantHnswIndex.build_pair(data, 6, 40, 1)],
        "build": [QuantHnswIndex.build(data, 6, 40, 1, rep).memory_accounting() for rep in REPRESENTATIONS],
    }
    for name, (fp32, sq8) in pairs.items():
        for acc in (fp32, sq8):
            assert set(acc) == keys
            assert acc["total_bytes"] == sum(acc[key] for key in keys - {"total_bytes"})
        assert (fp32["vectors_bytes"], fp32["codes_bytes"], fp32["codebook_bytes"]) == (4 * n * dim, 0, 0)
        assert (sq8["vectors_bytes"], sq8["codes_bytes"], sq8["codebook_bytes"]) == (0, n * dim, 8 * dim)
        assert fp32["graph_bytes"] == sq8["graph_bytes"] > 0
        assert fp32["total_bytes"] - sq8["total_bytes"] == 3 * n * dim - 8 * dim, name
    # build_pair 的兩者都由 clone 轉換，鄰居清單的 capacity 收緊成長度，graph 不會比 build 的大。
    assert pairs["build_pair"][0]["graph_bytes"] <= pairs["build"][0]["graph_bytes"]
    # 絕對值（64 位元）：clone 之後 graph = n × 24（Node）+ Σ_v (level_v + 1) × 24（每層鄰居清單的 Vec header）+ Σ len × 4。
    fp32 = QuantHnswIndex.build_pair(data, 6, 40, 1)[0]
    levels = [fp32.level(i) for i in range(n)]
    neighbours = sum(len(fp32.neighbors(i, level)) for i in range(n) for level in range(levels[i] + 1))
    assert pairs["build_pair"][0]["graph_bytes"] == 24 * n + 24 * sum(level + 1 for level in levels) + 4 * neighbours
    struct = {acc["struct_bytes"] for accs in pairs.values() for acc in accs}
    assert len(struct) == 1 and struct.pop() > 0  # QuantIndex 本身的大小：與表示、建法無關


# ---- sq8_encode（019 §3 的 codec） ----


def test_sq8_encode_hand_verifiable_case():
    # 第 0 維 [0, 2, 1]：vmin 0、vdiff 2；第 1 維全為 1：vdiff = 0 → code 0；第 2 維 [5, −1, 2]：vmin −1、vdiff 6。
    # t = 0.5 時 255·t = 127.5，向零截斷為 127。
    data = np.array([[0, 1, 5], [2, 1, -1], [1, 1, 2]], dtype=np.float32)
    vmin, vdiff, codes = rust_hnsw.sq8_encode(data)
    assert vmin.dtype == vdiff.dtype == np.float32 and codes.dtype == np.uint8
    assert vmin.shape == vdiff.shape == (3,) and codes.shape == (3, 3)
    np.testing.assert_array_equal(vmin, [0, 1, -1])
    np.testing.assert_array_equal(vdiff, [2, 0, 6])
    np.testing.assert_array_equal(codes, [[0, 0, 255], [255, 0, 0], [127, 0, 127]])


def test_sq8_encode_is_bitwise_identical_to_an_independent_numpy_codec():
    rng = np.random.default_rng(50)
    # 各維的尺度與位移不同，含一個常數維（vdiff = 0）；不正規化（codec 本身不要求單位長度）。
    data = (rng.standard_normal((500, 12)) * rng.uniform(0.01, 100, 12) + rng.uniform(-5, 5, 12)).astype(np.float32)
    data[:, 4] = 0.25
    for x in (data, random_unit(51, 400, 7)):
        got, want = rust_hnsw.sq8_encode(x), numpy_sq8(x)
        for a, b in zip(got, want):
            np.testing.assert_array_equal(a, b)
        assert got[2].min() == 0 and got[2].max() == 255


def test_index_codes_are_the_codec_of_the_stored_unit_vectors():
    # 019 §3：在 `add` 正規化之後的 unit vectors 上訓練。rust_unit 逐位元重寫 Core 的正規化。
    data = random_unit(52, 600, 10) * np.float32(3.5)  # 刻意不是 unit：index 先正規化
    sq8 = QuantHnswIndex.build(np.ascontiguousarray(data), 8, 40, 1, "sq8")
    vmin, vdiff, codes = rust_hnsw.sq8_encode(rust_unit(data))
    np.testing.assert_array_equal(sq8.codes(), codes)
    for a, b in zip(sq8.codebook(), (vmin, vdiff)):
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    # 在未正規化的輸入上訓練會得到不同的 codebook（確認上面的比對有鑑別力）。
    assert not np.array_equal(rust_hnsw.sq8_encode(np.ascontiguousarray(data))[0], vmin)


def test_sq8_encode_input_validation():
    with pytest.raises(TypeError):
        rust_hnsw.sq8_encode(np.ones((2, 2), dtype=np.float64))
    with pytest.raises(TypeError):
        rust_hnsw.sq8_encode(np.ones(4, dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        rust_hnsw.sq8_encode(np.ones((4, 4), dtype=np.float32)[:, ::2])
    with pytest.raises(ValueError, match="C-contiguous"):
        rust_hnsw.sq8_encode(np.asfortranarray(np.arange(1, 7, dtype=np.float32).reshape(3, 2)))
    with pytest.raises(ValueError, match="dimension must be positive"):
        rust_hnsw.sq8_encode(np.zeros((3, 0), dtype=np.float32))
    for bad, row in ((np.nan, 1), (np.inf, 2), (-np.inf, 0)):
        data = np.ones((3, 2), dtype=np.float32)
        data[row, 1] = bad
        with pytest.raises(ValueError, match=f"row {row} contains NaN or Inf"):
            rust_hnsw.sq8_encode(data)
    # 有限但 vmax − vmin 溢位成 Inf：sq8_train_encode 會 panic，binding 先轉成 ValueError。
    with pytest.raises(ValueError, match="dimension 1 overflows"):
        rust_hnsw.sq8_encode(np.array([[0, -3e38], [1, 3e38]], dtype=np.float32))
    # n = 0：codebook 全為 0，codes 為 (0, d)（與 Rust 的 sq8_train_encode 相同）。
    vmin, vdiff, codes = rust_hnsw.sq8_encode(np.zeros((0, 3), dtype=np.float32))
    np.testing.assert_array_equal(vmin, [0, 0, 0])
    np.testing.assert_array_equal(vdiff, [0, 0, 0])
    assert codes.shape == (0, 3) and codes.dtype == np.uint8


# ---- GIL（002、006：計算期間釋放 GIL） ----


def main_thread_ticks_during(call):
    """在背景 thread 執行 `call`，回傳 main thread 在其執行期間中段 60% 內跑了幾圈（同 test_hnsw_binding）。

    沒有釋放 GIL 時，main thread 仍會在呼叫返回之後、worker 記下結束時間之前跑一段（約一個 switch interval，5–8 ms），
    所以呼叫本身必須遠長於它（下面的呼叫在本機約 100–500 ms），這段尾巴才不會落進中段 60%。
    """
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


def test_build_and_build_pair_release_gil():
    data = random_unit(21, 3000, 32)
    assert main_thread_ticks_during(lambda: QuantHnswIndex.build(data, 8, 100, 1, "sq8")) > 0
    assert main_thread_ticks_during(lambda: QuantHnswIndex.build_pair(data, 8, 100, 1)) > 0


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_search_methods_release_gil(representation):
    data = random_unit(20, 3000, 32)
    index = QuantHnswIndex.build(data, 8, 40, 1, representation)
    assert main_thread_ticks_during(lambda: index.search(data, 10, 200)) > 0
    assert main_thread_ticks_during(lambda: index.search_stats(data, 10, 200)) > 0
    queries = np.concatenate([data, data])
    assert main_thread_ticks_during(lambda: index.exact_search(queries, 10)) > 0


def test_sq8_encode_releases_gil():
    data = np.random.default_rng(22).standard_normal((400_000, 64)).astype(np.float32)
    assert main_thread_ticks_during(lambda: rust_hnsw.sq8_encode(data)) > 0


# ---- 既有的介面不變（019 §4） ----


def public(cls):
    return sorted(name for name in dir(cls) if not name.startswith("_"))


def test_module_only_adds_the_quant_names():
    assert sorted(rust_hnsw.__all__) == ["BUILD_INFO", "ExactIndex", "HnswIndex", "QuantHnswIndex", "sq8_encode"]
    # 既有的兩個類別沒有多出任何方法或屬性（BUILD_INFO 的內容由 test_hnsw_binding 檢查）。
    assert public(rust_hnsw.HnswIndex) == [
        "add", "dim", "ef_construction", "load", "m", "max_level", "save", "search", "seed"]
    assert public(rust_hnsw.ExactIndex) == ["add", "dim", "search"]
    assert public(QuantHnswIndex) == [
        "build", "build_pair", "codebook", "codes", "dim", "entry_point", "exact_search", "level",
        "memory_accounting", "neighbors", "representation", "search", "search_stats"]
