"""Advanced C 的不計時診斷（decision 019 §7）：重排與分解的算術、與 formal run 的 ids 核對、FAISS 的 graph 核對與天花板。

全部是 synthetic 資料（隨機 unit vectors 與 sign-cube）；formal run 由同一組 adapter 在 synthetic 資料上產生，只用來驗證核對邏輯。
"""

import json
from fractions import Fraction
from pathlib import Path

import faiss
import numpy as np
import pytest

from benchmarks import adapters, config, environment, profile, quant_diag, quant_report
from benchmarks.metrics import recall_at_k
from benchmarks.quant_report import E_STAR, F, P, Q, S
from helpers import random_unit
from test_third_party_semantics import sign_cube

SYNTHETIC = "synthetic-quant"
SMALL_N = 600
EFS = (16, 32, 64)


@pytest.fixture(autouse=True)
def synthetic_cell(monkeypatch):
    """synthetic 資料只有 600 列：把 C 的格點換成 n = 600（M、efC 仍是 16、200），e* 取 32。"""
    monkeypatch.setattr(quant_report, "N", SMALL_N)
    monkeypatch.setitem(E_STAR, SYNTHETIC, 32)
    before = faiss.omp_get_max_threads()
    yield
    faiss.omp_set_num_threads(before)


@pytest.fixture(scope="module")
def vectors():
    data = random_unit(41, SMALL_N, 16)
    queries = random_unit(42, 40, 16)
    exact = __import__("rust_hnsw").ExactIndex(16)
    exact.add(data)
    gt, _ = exact.search(queries, 10)
    return data, queries, gt


def formal_child(directory, impl, seed, efs, data, queries, gt, group=0, flags=()):
    """一個 formal 子 process 的紀錄：以真正的 adapter 建圖並搜尋，points 檔只存 ids_k10（與 run.py 相同）。"""
    path, _ = config.IMPLEMENTATIONS[impl]
    adapter = getattr(adapters, path.split(":")[1])(m=16, ef_construction=200, seed=seed)
    adapter.build(data)
    child_id = f"g{group:03d}-{impl}"
    points = []
    (directory / "points").mkdir(parents=True, exist_ok=True)
    for ef in efs:
        adapter.set_ef_search(ef)
        ids, _ = adapter.search(queries, 10)
        ids = np.asarray(ids, dtype=np.int64)
        ids1, _ = adapter.search(queries, 1)
        file = f"points/{child_id}-ef{ef}.npz"
        np.savez(directory / file, ids_k10=ids, ids_k1=np.asarray(ids1, dtype=np.int64))
        recall = recall_at_k(ids, gt, 10)
        points.append({"ef_search": ef, "file": file, "flags": list(flags),
                       "k10": {"status": "ok", "incomplete": False, "incomplete_queries": 0, "recall": recall,
                               "tie_aware_recall": recall, "qps": {"qps": [1000.0] * 5, "qps_median": 1000.0},
                               "latency": {"p50_ns": 1000.0, "p95_ns": 2000.0}},
                       "k1": {"status": "ok", "recall": recall_at_k(ids1, gt, 1), "tie_aware_recall": None}})
    return {"child_id": child_id, "implementation": impl, "status": "ok", "partial": False, "flags": [],
            "build": {"n": len(data), "m": 16, "ef_construction": 200, "seed": seed, "shuffle": None, "build_ns": 1e9,
                      "peak_rss_bytes": 1e9, "index_memory_bytes": 1e8, "index_footprint_bytes": 1e8,
                      "index_footprint_relieved_bytes": 9e7, "index_memory_relieved_bytes": 9e7, "flags": []},
            "points": points}


def formal_run(directory, dataset, seed, impls, efs, data, queries, gt, commit="c" * 40):
    directory.mkdir(parents=True)
    children = [formal_child(directory, impl, seed, efs, data, queries, gt, group=i) for i, impl in enumerate(impls)]
    record = {"mode": "formal", "status": "complete", "status_reasons": [], "protocol": {"version": "008-v5"},
              "dataset": {"name": dataset, "n_queries": len(queries), "dim": data.shape[1]},
              "config": {"path": f"benchmarks/configs/quant-x-seed-{seed}.toml", "parsed": {"ef_search": list(efs)}},
              "metadata": {"git": {"commit": commit}}, "children": children}
    (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return str(directory), record


# ---- 重排（T_X）與分解的算術 ----


def test_rerank_breaks_ties_by_id_and_keeps_padding_out():
    data = np.array([[1, 0], [0, 1], [1, 0], [0.6, 0.8]], dtype=np.float32)
    queries = np.array([[1, 0]], dtype=np.float32)
    candidates = np.array([[3, 2, 1, 0, -1]])
    # 距離：id 0、2 為 0（tie，依 id），id 3 為 0.4，id 1 為 1；padding 不算候選。
    assert quant_diag.rerank_top(candidates, data, queries, k=3).tolist() == [[0, 2, 3]]
    assert quant_diag.rerank_top(candidates, data, queries, k=5).tolist() == [[0, 2, 3, 1, -1]]
    assert quant_diag.rerank_top(np.array([[-1, -1]]), data, queries, k=2).tolist() == [[-1, -1]]


def test_rerank_matches_an_independent_f64_sort_in_every_chunking(vectors):
    data, queries, gt = vectors
    index = __import__("rust_hnsw").QuantHnswIndex.build(data, 16, 200, 1, "sq8")
    candidates, _ = index.search(queries, 64, 64)
    expected = []
    for q, row in zip(queries.astype(np.float64), candidates):
        distance = 1.0 - data[row].astype(np.float64) @ q
        expected.append(sorted(zip(distance, row))[:10])
    expected = np.array([[i for _, i in row] for row in expected])
    for chunk in (1, 64 * 16 * 8 * 3, quant_diag.CHUNK_BYTES):
        np.testing.assert_array_equal(quant_diag.rerank_top(candidates, data, queries, chunk_bytes=chunk), expected)
    assert quant_diag.rerank_recall(candidates, data, queries, gt) == recall_at_k(expected, gt, 10)
    # 候選就是整個資料集時，重排等於 exact search：T = 1。
    everything = np.tile(np.arange(SMALL_N), (len(queries), 1))
    assert quant_diag.rerank_recall(everything, data, queries, gt) == 1.0


@pytest.mark.parametrize("values", [(0.95, 0.93, 0.951, 0.94), (0.5, 0.6, 0.4, 0.7), (1.0, 1.0, 1.0, 1.0)])
def test_the_decomposition_satisfies_the_identity(values):
    exact = [Fraction(v).limit_denominator(1000) for v in values]
    parts = quant_report.decompose(*exact)
    assert parts["L_total"] == parts["L_trav"] + parts["L_rank"] - parts["delta_F"]
    r_f, r_s, t_f, t_s = exact
    assert parts == {"L_total": r_f - r_s, "L_trav": t_f - t_s, "L_rank": t_s - r_s, "delta_F": t_f - r_f}


def test_duplicate_code_rows_are_counted():
    codes = np.array([[1, 2], [1, 2], [3, 4], [1, 3], [3, 4], [0, 0]], dtype=np.uint8)
    assert quant_diag.duplicate_code_fraction(codes) == 4 / 6
    assert quant_diag.duplicate_code_fraction(np.zeros((3, 2), dtype=np.uint8)) == 1.0


# ---- Rust：與 formal run 的 ids 核對、D、V 與 M_struct ----


@pytest.fixture(scope="module")
def formal(vectors, tmp_path_factory):
    data, queries, gt = vectors
    directory = tmp_path_factory.mktemp("formal") / "run1"
    return [formal_run(directory, SYNTHETIC, 1, (S, F, Q), EFS, data, queries, gt)]


def test_the_diagnosis_is_bound_to_the_formal_ids(vectors, formal):
    data, queries, gt = vectors
    ids, recalls = quant_diag.formal_points(formal)
    entry, sq8 = quant_diag.rust_seed(data, queries, gt, 1, EFS, ids, recalls)
    import rust_hnsw

    pair = dict(zip((F, S), rust_hnsw.QuantHnswIndex.build_pair(data, 16, 200, 1)))
    for arm in (F, S):
        arm_entry = entry["arms"][arm]
        assert arm_entry["ids_valid"] and arm_entry["mismatched_ef"] == [] and arm_entry["missing_formal_ef"] == []
        for ef in EFS:
            point = arm_entry["points"][str(ef)]
            assert point["formal_ids_match"] and point["stats_ids_match"] and point["w_prefix_ids_match"]
            assert point["w_prefix_matches_stats"] and point["mismatched_queries"] == 0
            assert point["formal_recall10"] == recalls[(arm, 1, ef)]
            # T_X 由 k = ef 的完整候選表 W_X 重排（019 §2）：獨立重算，候選表的寬度就是 ef。
            candidates, _ = pair[arm].search(queries, ef, ef)
            assert candidates.shape == (len(queries), ef) and point["w_width"] == ef
            assert point["rerank_recall10"] == quant_diag.rerank_recall(candidates, data, queries, gt)
    # 鑑別力：這組資料上 S 的重排救回了一部分損失（T_S > R_S）。若 W 只有 10 個候選，T 恆等於 R。
    s_64 = entry["arms"][S]["points"]["64"]
    assert s_64["rerank_recall10"] > s_64["formal_recall10"]
    # D 與 V 是 search_stats 的平均（與 search_batch 同一次走訪）。
    _, _, d, v = sq8.search_stats(queries, 10, 32)
    point = entry["arms"][S]["points"]["32"]
    assert point["kernel_calls_mean"] == float(np.mean(d)) and point["visited_inserts_mean"] == float(np.mean(v))
    # M_struct：F − S = 3·n·d − 8·d（019 §2 C1，由單元測試保證的會計恆等式）。
    accounting = entry["memory_accounting"]
    assert accounting[F]["total_bytes"] - accounting[S]["total_bytes"] == 3 * SMALL_N * 16 - 8 * 16
    assert entry["codes_sha256"] == __import__("hashlib").sha256(sq8.codes().tobytes()).hexdigest()


def test_a_mismatch_invalidates_only_that_seed_and_arm(vectors, formal):
    data, queries, gt = vectors
    ids, recalls = quant_diag.formal_points(formal)
    ids = dict(ids)
    corrupted = ids[(S, 1, 64)].copy()
    corrupted[3, [0, 1]] = corrupted[3, [1, 0]]  # 兩個 id 對調：同一組 id、順序不同也算不符
    ids[(S, 1, 64)] = corrupted
    del ids[(F, 1, 16)]
    entry, _ = quant_diag.rust_seed(data, queries, gt, 1, EFS, ids, recalls)
    s = entry["arms"][S]
    assert not s["ids_valid"] and s["mismatched_ef"] == [64] and s["points"]["64"]["mismatched_queries"] == 1
    assert s["points"]["32"]["formal_ids_match"] is True
    f = entry["arms"][F]
    assert f["ids_valid"] and f["missing_formal_ef"] == [16] and f["points"]["16"]["formal_ids_match"] is None
    # quant_report 的 D 只在核對通過時有效。
    diag = {"rust": {"1": entry}}
    assert quant_report.d_value(diag, S, 1, 32) is None  # 該 (seed, 臂) 整個無效
    assert quant_report.d_value(diag, F, 1, 16) is None  # formal 沒有這個點
    assert quant_report.d_value(diag, F, 1, 32) == f["points"]["32"]["kernel_calls_mean"]


def test_an_arm_without_any_formal_point_is_not_valid(vectors, formal):
    data, queries, gt = vectors
    entry, _ = quant_diag.rust_seed(data, queries, gt, 2, EFS, {}, {})
    assert not entry["arms"][F]["ids_valid"] and entry["arms"][F]["missing_formal_ef"] == list(EFS)


# ---- FAISS：C8 的 graph 核對與結構大小 ----


def test_the_faiss_port_is_checked_before_the_flat_is_freed(vectors, formal):
    data, queries, gt = vectors
    ids, _ = quant_diag.formal_points(formal)
    sq8 = __import__("rust_hnsw").QuantHnswIndex.build(data, 16, 200, 1, "sq8")
    entry = quant_diag.faiss_seed(data, queries, 1, 32, ids[(Q, 1, 32)], sq8.codes(), sq8.codebook())
    assert entry["graph_identical"] and entry["graph_differences"] == [] and entry["ids_match_at_e_star"] is True
    n, d = data.shape
    p, q = entry["struct_bytes"][P], entry["struct_bytes"][Q]
    assert (p["codes_bytes"], q["codes_bytes"], q["codebook_bytes"]) == (n * d * 4, n * d, 8 * d)
    assert p["neighbors_bytes"] == q["neighbors_bytes"] and p["offsets_bytes"] == (n + 1) * 8 and p["levels_bytes"] == n * 4
    assert p["total_bytes"] - q["total_bytes"] == 3 * n * d - 8 * d
    assert entry["codes_agreement"] == pytest.approx(1.0, abs=0.01) and 0.0 <= entry["codes_agreement"] <= 1.0
    wrong = ids[(Q, 1, 32)].copy()
    wrong[0, 0] = wrong[0, 1]
    assert quant_diag.faiss_seed(data, queries, 1, 32, wrong, sq8.codes(), sq8.codebook())["ids_match_at_e_star"] is False
    assert quant_diag.faiss_seed(data, queries, 1, 32, None, sq8.codes(), sq8.codebook())["ids_match_at_e_star"] is None


def test_graph_differences_are_named():
    data = random_unit(5, 300, 8)
    faiss.omp_set_num_threads(1)
    a = adapters.faiss_hnsw_flat(data, 8, 32, 1)
    b = adapters.faiss_hnsw_flat(data, 8, 32, 2)
    assert quant_diag.faiss_graph_differences(a.hnsw, a.hnsw) == []
    assert "neighbors" in quant_diag.faiss_graph_differences(a.hnsw, b.hnsw)


# ---- 天花板 ----


def test_the_ceiling_is_one_when_quantization_preserves_the_ranking():
    data, queries = sign_cube()
    exact = __import__("rust_hnsw").ExactIndex(data.shape[1])
    exact.add(data)
    gt, _ = exact.search(queries, 10)
    sq8 = __import__("rust_hnsw").QuantHnswIndex.build(data, 8, 32, 1, "sq8")
    assert quant_diag.ceiling(sq8, data, queries, gt) == {"recall10": 1.0, "recall1": 1.0, "faiss_recall10": 1.0,
                                                         "faiss_recall1": 1.0}


def test_the_ceiling_uses_the_sq8_exact_search(vectors):
    data, queries, gt = vectors
    sq8 = __import__("rust_hnsw").QuantHnswIndex.build(data, 16, 200, 1, "sq8")
    ids, _ = sq8.exact_search(queries, 10)
    result = quant_diag.ceiling(sq8, data, queries, gt)
    assert result["recall10"] == recall_at_k(ids, gt, 10) and result["recall1"] == recall_at_k(ids[:, :1], gt, 1)


# ---- 全部與執行條件 ----


def test_the_full_diagnosis_on_synthetic_runs(vectors, tmp_path):
    data, queries, gt = vectors
    labelled = [formal_run(tmp_path / f"run{seed}", SYNTHETIC, seed, (F, S, Q) if seed == 1 else (S, F), EFS, data, queries, gt)
                for seed in (1, 2)]
    result = quant_diag.diagnose(SYNTHETIC, labelled, data, queries, gt, rust_seeds=(1, 2), faiss_seeds=(1,))
    assert result["e_star"] == 32 and result["ef_search"] == list(EFS)
    assert [Path(r["run"]).name for r in result["formal_runs"]] == ["run1", "run2"]
    assert all(result["rust"][str(s)]["arms"][arm]["ids_valid"] for s in (1, 2) for arm in (F, S))
    assert result["codes"]["identical_across_seeds"] and isinstance(result["codes"]["sha256"], str)
    assert result["faiss"]["1"]["graph_identical"] and result["faiss"]["1"]["ids_match_at_e_star"]
    assert len(result["decomposition"]) == 2 * len(EFS)
    for item in result["decomposition"]:
        assert item["L_total"] == pytest.approx(item["L_trav"] + item["L_rank"] - item["delta_F"], abs=1e-12)
        assert item["ids_valid"]
    json.dumps(result)  # 可以寫成 JSON


def test_the_evidence_conditions_refuse_without_single_thread_or_lock(monkeypatch):
    monkeypatch.delenv(quant_diag.VECLIB_ENV, raising=False)
    with pytest.raises(SystemExit, match="VECLIB_MAXIMUM_THREADS=1"):
        quant_diag.evidence_conditions([])
    monkeypatch.setenv(quant_diag.VECLIB_ENV, "1")

    def no_lock():
        raise environment.PreflightError("lock owner pid 1 is not an ancestor")

    monkeypatch.setattr(environment, "check_lock", no_lock)
    with pytest.raises(SystemExit, match="heavy lock"):
        quant_diag.evidence_conditions([])


def test_the_evidence_conditions_compare_code_build_info_and_versions(monkeypatch):
    import rust_hnsw

    monkeypatch.setenv(quant_diag.VECLIB_ENV, "1")
    monkeypatch.setattr(environment, "check_lock", lambda: "pid=1")
    monkeypatch.setattr(profile, "check_code_state", lambda: {"commit": "h" * 40})
    versions = {name: __import__("importlib.metadata").metadata.version(name) for name in ("hnswlib", "faiss-cpu", "numpy")}
    good = {"metadata": {"git": {"commit": "c" * 40}, "build_info": dict(rust_hnsw.BUILD_INFO), "versions": versions}}
    monkeypatch.setattr(quant_diag.gate_profile, "changed_profiled_files", lambda a, b: [])
    assert quant_diag.evidence_conditions([good])["commit"] == "h" * 40
    monkeypatch.setattr(quant_diag.gate_profile, "changed_profiled_files", lambda a, b: ["crates/hnsw/src/hnsw/quant.rs"])
    with pytest.raises(SystemExit, match="profiled code differs"):
        quant_diag.evidence_conditions([good])
    monkeypatch.setattr(quant_diag.gate_profile, "changed_profiled_files", lambda a, b: [])
    with pytest.raises(SystemExit, match="BUILD_INFO"):
        quant_diag.evidence_conditions([{"metadata": {**good["metadata"], "build_info": {"profile": "debug"}}}])
    with pytest.raises(SystemExit, match="package versions"):
        quant_diag.evidence_conditions([{"metadata": {**good["metadata"], "versions": {**versions, "numpy": "0"}}}])


def test_main_refuses_to_overwrite_an_earlier_diagnosis(vectors, tmp_path):
    data, queries, gt = vectors
    label, _ = formal_run(tmp_path / "run1", SYNTHETIC, 1, (F,), (16,), data, queries, gt)
    out = tmp_path / "diag.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="must not be overwritten"):
        quant_diag.main([label, "--dataset", "glove", "--out", str(out)])
    assert out.read_text() == "{}"


def test_main_refuses_runs_that_are_not_the_complete_c_formal_set(vectors, tmp_path):
    data, queries, gt = vectors
    label, _ = formal_run(tmp_path / "run1", SYNTHETIC, 1, (F,), (16,), data, queries, gt)
    with pytest.raises(SystemExit, match="complete set of the 10 C formal configs"):
        quant_diag.main([label, "--dataset", "glove", "--out", str(tmp_path / "diag.json")])
    assert not (tmp_path / "diag.json").exists()


def test_main_needs_all_ten_c_formal_runs_before_diagnosing_either_dataset(tmp_path, monkeypatch):
    # 019 §7：診斷在全部 10 份 C formal config 完成之後才執行。GloVe 的 5 份完整、文字還沒有跑時必須拒絕。
    from test_quant_report import Scenario, materialize

    from benchmarks import report

    # CLI 預設也會掃描 repo 的 results/runs（formal 的完整性檢查）；只掃描這個測試自己的 run，不受本機真實 run 的影響。
    monkeypatch.setattr(report, "RUNS_ROOT", tmp_path / "runs")
    args = materialize(Scenario(), tmp_path)
    glove = [d for d in args["run_dirs"] if "glove" in d.name]
    out = tmp_path / "diag.json"
    with pytest.raises(SystemExit, match="both datasets"):
        quant_diag.main([*map(str, glove), "--dataset", "glove", "--out", str(out)])
    # 10 份都在時，完整性的檢查通過，接著才檢查執行環境（這裡因為沒有設定單執行緒而停下）。
    monkeypatch.delenv(quant_diag.VECLIB_ENV, raising=False)
    with pytest.raises(SystemExit, match="VECLIB_MAXIMUM_THREADS=1"):
        quant_diag.main([*map(str, args["run_dirs"]), "--dataset", "text", "--out", str(out)])
    assert not out.exists()
    with pytest.raises(SystemExit):
        quant_diag.main([*map(str, args["run_dirs"]), "--dataset", "other", "--out", str(out)])



def test_rerank_uses_f64_where_f32_cannot_separate_the_candidates():
    # 兩個只差兩個 ulp 的 768 維向量：f64 的距離 id 1 較近；f32 的 einsum 分不出（或順序相反），tie-break 會選 id 0。
    rng = np.random.default_rng(0)
    for _ in range(2):
        q = rng.standard_normal(768).astype(np.float32)
        q /= np.linalg.norm(q)
        a = rng.standard_normal(768).astype(np.float32)
        a /= np.linalg.norm(a)
        b = a.copy()
        i, j = rng.integers(0, 768, 2)
        b[i] = np.nextafter(b[i], np.float32(10))
        b[j] = np.nextafter(b[j], np.float32(-10))
    data = np.stack([a, b])
    d64 = 1 - data.astype(np.float64) @ q.astype(np.float64)
    d32 = 1 - np.einsum("rcd,rd->rc", data[None], q[None])[0]
    assert d64[1] < d64[0] and d32[1] >= d32[0]  # 前提
    assert quant_diag.rerank_top(np.array([[0, 1]]), data, q[None], k=2).tolist() == [[1, 0]]


class _FakeIndex:
    """search_stats 與 search 的前 10 個不一致的假 index：只用來確認兩者都要與 formal 核對（019 §7 第 3 步）。"""

    def __init__(self, ids, shifted):
        self.ids, self.shifted = ids, shifted

    def search_stats(self, queries, k, ef):
        return self.ids, None, np.ones(len(queries), np.uint64), np.ones(len(queries), np.uint64)

    def search(self, queries, k, ef):
        w = np.concatenate([self.ids, self.ids], axis=1)[:, :k].copy()
        if self.shifted:
            w[:, :10] = w[:, :10][:, ::-1]
        return w, None

    def memory_accounting(self):
        return {"total_bytes": 0}

    def codes(self):
        return np.zeros((1, 1), np.uint8)


def test_both_the_stats_ids_and_the_w_prefix_are_checked(vectors, monkeypatch):
    import rust_hnsw

    data, queries, gt = vectors
    ids = np.tile(np.arange(10, dtype=np.int64), (len(queries), 1))

    class FakeQuant:
        @staticmethod
        def build_pair(*args):
            return _FakeIndex(ids, shifted=True), _FakeIndex(ids, shifted=False)

    monkeypatch.setattr(rust_hnsw, "QuantHnswIndex", FakeQuant)
    formal = {(arm, 1, 16): ids for arm in (F, S)}
    entry, _ = quant_diag.rust_seed(data, queries, gt, 1, (16,), formal, {})
    f, s = entry["arms"][F]["points"]["16"], entry["arms"][S]["points"]["16"]
    assert f["stats_ids_match"] and not f["w_prefix_ids_match"] and f["formal_ids_match"] is False
    assert not f["w_prefix_matches_stats"] and not entry["arms"][F]["ids_valid"]
    assert s["formal_ids_match"] is True and entry["arms"][S]["ids_valid"]


def test_the_candidate_table_is_searched_with_k_equal_to_ef(vectors, monkeypatch):
    # 019 §2：W_X 是 k = ef 的搜尋結果（layer 0 的完整結果表），重排的輸入是它，不是 search_stats 的 k = 10 ids。
    import rust_hnsw

    data, queries, gt = vectors
    calls, reranked = [], []
    real = rust_hnsw.QuantHnswIndex

    class Spy:
        def __init__(self, index):
            self.index = index

        def search_stats(self, queries, k, ef):
            calls.append(("search_stats", k, ef))
            return self.index.search_stats(queries, k, ef)

        def search(self, queries, k, ef):
            calls.append(("search", k, ef))
            return self.index.search(queries, k, ef)

        def __getattr__(self, name):
            return getattr(self.index, name)

    class FakeQuant:
        @staticmethod
        def build_pair(*args):
            return tuple(Spy(index) for index in real.build_pair(*args))

    original = quant_diag.rerank_top

    def spy_rerank(candidates, *args, **kwargs):
        reranked.append(np.asarray(candidates).shape[1])
        return original(candidates, *args, **kwargs)

    monkeypatch.setattr(rust_hnsw, "QuantHnswIndex", FakeQuant)
    monkeypatch.setattr(quant_diag, "rerank_top", spy_rerank)
    quant_diag.rust_seed(data, queries, gt, 1, EFS, {}, {})
    assert [c for c in calls if c[0] == "search"] == [("search", ef, ef) for _ in (F, S) for ef in EFS]
    assert [c for c in calls if c[0] == "search_stats"] == [("search_stats", 10, ef) for _ in (F, S) for ef in EFS]
    assert reranked == [ef for _ in (F, S) for ef in EFS]


def test_codes_agreement_and_codebook_are_compared_with_the_faiss_storage(vectors, formal):
    # C8 的 codes 一致比例與 codebook 是否相同（只描述），以 FAISS 自己的 codec 在同一份資料上獨立算出期望值。
    data, queries, gt = vectors
    ids, _ = quant_diag.formal_points(formal)
    sq = faiss.IndexScalarQuantizer(data.shape[1], faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    sq.sq.rangestat = faiss.ScalarQuantizer.RS_minmax
    sq.sq.rangestat_arg = 0
    sq.train(data)
    faiss_codes = sq.sa_encode(data)
    trained = faiss.vector_to_array(sq.sq.trained)
    d = data.shape[1]
    codebook = (trained[:d].copy(), trained[d:].copy())
    same = quant_diag.faiss_seed(data, queries, 1, 32, ids[(Q, 1, 32)], faiss_codes, codebook)
    assert same["codes_agreement"] == 1.0 and same["codebook_identical"] is True
    changed = faiss_codes.copy()
    changed[7, 3] ^= 1
    vmin = codebook[0].copy()
    vmin[2] = np.nextafter(vmin[2], np.float32(1))
    other = quant_diag.faiss_seed(data, queries, 1, 32, ids[(Q, 1, 32)], changed, (vmin, codebook[1]))
    assert other["codes_agreement"] == 1.0 - 1.0 / changed.size and other["codebook_identical"] is False
    assert quant_diag.faiss_seed(data, queries, 1, 32, ids[(Q, 1, 32)], faiss_codes,
                                 (codebook[0], codebook[1] * 2))["codebook_identical"] is False


def test_the_faiss_ceiling_is_faiss_own_brute_force():
    # 重複向量讓 SQ8 距離出現 tie：Rust 以 (distance, id) 決定，FAISS 的順序不同，所以兩個天花板的 Recall@1 不同。
    import rust_hnsw

    from benchmarks.interface import prepare_vectors
    from helpers import duplicate_data

    data, queries = prepare_vectors(duplicate_data(5, 300, 8)), random_unit(6, 30, 8)
    exact = rust_hnsw.ExactIndex(8)
    exact.add(data)
    gt, _ = exact.search(queries, 10)
    sq8 = rust_hnsw.QuantHnswIndex.build(data, 8, 32, 1, "sq8")
    result = quant_diag.ceiling(sq8, data, queries, gt)
    index = faiss.IndexScalarQuantizer(8, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    index.sq.rangestat = faiss.ScalarQuantizer.RS_minmax
    index.train(data)
    index.add(data)
    _, faiss_ids = index.search(queries, 10)
    rust_ids, _ = sq8.exact_search(queries, 10)
    assert result["faiss_recall10"] == recall_at_k(faiss_ids, gt, 10) and result["faiss_recall1"] == recall_at_k(faiss_ids, gt, 1)
    assert result["recall1"] == recall_at_k(rust_ids, gt, 1)
    assert result["faiss_recall1"] != result["recall1"]  # 前提：兩者真的不同


def test_codes_that_differ_across_seeds_are_reported(vectors, tmp_path, monkeypatch):
    data, queries, gt = vectors
    labelled = [formal_run(tmp_path / f"run{seed}", SYNTHETIC, seed, (F, S, Q) if seed == 1 else (S, F), EFS, data, queries, gt)
                for seed in (1, 2)]
    real = quant_diag.rust_seed

    def different_sha(*args):
        entry, sq8 = real(*args)
        if args[3] == 2:
            entry["codes_sha256"] = "0" * 64
        return entry, sq8

    monkeypatch.setattr(quant_diag, "rust_seed", different_sha)
    result = quant_diag.diagnose(SYNTHETIC, labelled, data, queries, gt, rust_seeds=(1, 2), faiss_seeds=(1,))
    assert result["codes"]["identical_across_seeds"] is False
    assert result["codes"]["sha256"] == sorted({result["rust"]["1"]["codes_sha256"], "0" * 64})


def test_the_decomposition_marks_a_seed_valid_only_when_both_arms_are_valid():
    def arm(valid, r, t):
        return {"ids_valid": valid, "points": {"32": {"formal_recall10": r, "rerank_recall10": t}}}

    rust = {"1": {"arms": {F: arm(True, 0.9, 0.91), S: arm(False, 0.8, 0.85)}},
            "2": {"arms": {F: arm(False, 0.9, 0.91), S: arm(True, 0.8, 0.85)}},
            "3": {"arms": {F: arm(True, 0.9, 0.91), S: arm(True, 0.8, 0.85)}}}
    items = quant_diag.decompositions(rust)
    assert [(i["seed"], i["ids_valid"]) for i in items] == [(1, False), (2, False), (3, True)]
    assert items[0]["L_total"] == pytest.approx(0.1) and items[0]["L_rank"] == pytest.approx(0.05)


@pytest.mark.parametrize("name", ["levels", "offsets", "neighbors", "entry_point", "max_level"])
def test_each_graph_field_is_compared(name):
    data = random_unit(5, 300, 8)
    faiss.omp_set_num_threads(1)
    flat = adapters.faiss_hnsw_flat(data, 8, 32, 1)
    index = adapters.faiss_sq8_from_flat(flat, data, 8)
    assert quant_diag.faiss_graph_differences(flat.hnsw, index.hnsw) == []
    if name in ("entry_point", "max_level"):
        setattr(index.hnsw, name, getattr(index.hnsw, name) + 1)
    else:
        vector = getattr(index.hnsw, name)
        values = faiss.vector_to_array(vector)
        values[-1] += 1
        faiss.copy_array_to_vector(values, vector)
    assert quant_diag.faiss_graph_differences(flat.hnsw, index.hnsw) == [name]
