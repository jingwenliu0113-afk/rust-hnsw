"""資料集準備：manifest、frozen ground truth、GloVe（HDF5）轉換與 runner 的使用方式（decision 011）。

全部以 tmp_path 中的小型合成檔案測試；真正的資料集只在準備時下載，不進 Git。
"""

import json

import h5py
import numpy as np
import pytest

import rust_hnsw
from benchmarks import dataprep, datasets, prepare_glove
from benchmarks.interface import prepare_vectors
from benchmarks.run import run
from test_runner import STUBS, stub_config


def exact_top(train, queries, k):
    index = rust_hnsw.ExactIndex(train.shape[1])
    index.add(train)
    return index.search(queries, k)[0]


# ---- frozen ground truth ----


def test_frozen_ground_truth_is_the_exact_top_100_for_each_n():
    train = prepare_vectors(np.random.default_rng(0).standard_normal((300, 8)))
    queries = prepare_vectors(np.random.default_rng(1).standard_normal((20, 8)))
    gts = dataprep.exact_ground_truth(train, queries, [150, 300], k=100)
    assert set(gts) == {150, 300}
    assert gts[150].dtype == np.int64 and gts[150].shape == (20, 100)
    np.testing.assert_array_equal(gts[150], exact_top(train[:150], queries, 100))
    # top-100 的前 10 欄等於 k = 10 的答案（tie-break 是全序）。
    np.testing.assert_array_equal(gts[300][:, :10], exact_top(train, queries, 10))


def test_verify_frozen_ground_truth_detects_a_wrong_file():
    train = prepare_vectors(np.random.default_rng(0).standard_normal((200, 8)))
    queries = prepare_vectors(np.random.default_rng(1).standard_normal((120, 8)))
    gt = dataprep.exact_ground_truth(train, queries, [200], k=100)[200]
    dataprep.verify_frozen_ground_truth(gt, train, queries)
    tampered = gt.copy()
    tampered[7, [0, 1]] = tampered[7, [1, 0]]
    with pytest.raises(ValueError, match="frozen ground truth"):
        dataprep.verify_frozen_ground_truth(tampered, train, queries)
    with pytest.raises(ValueError, match="shape"):
        dataprep.verify_frozen_ground_truth(gt[:, :5], train, queries)


def test_file_record_has_checksum_shape_and_dtype(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.zeros((3, 4), dtype=np.float32))
    record = dataprep.file_record(path)
    assert record["shape"] == [3, 4] and record["dtype"] == "float32"
    assert len(record["sha256"]) == 64 and record["bytes"] == path.stat().st_size


# ---- GloVe（ann-benchmarks HDF5） ----


def write_hdf5(path, train, test, neighbors, distance="angular"):
    with h5py.File(path, "w") as f:
        f.attrs["distance"] = distance
        f.attrs["dimension"] = train.shape[1]
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)
        f.create_dataset("neighbors", data=neighbors)
        f.create_dataset("distances", data=np.zeros(neighbors.shape))


def test_glove_conversion_writes_unit_vectors_and_keeps_row_order(tmp_path):
    rng = np.random.default_rng(2)
    train = rng.standard_normal((50, 6)) * 3
    test = rng.standard_normal((12, 6))
    neighbors = rng.integers(0, 50, size=(12, 100))
    raw = tmp_path / "g.hdf5"
    write_hdf5(raw, train, test, neighbors)
    info = prepare_glove.convert(raw, tmp_path / "out")
    out_train = np.load(tmp_path / "out" / "train.npy")
    out_queries = np.load(tmp_path / "out" / "queries.npy")
    np.testing.assert_array_equal(out_train, prepare_vectors(train))
    np.testing.assert_array_equal(out_queries, prepare_vectors(test))
    assert out_train.dtype == np.float32 and out_train.flags.c_contiguous
    np.testing.assert_array_equal(np.load(tmp_path / "out" / "neighbors.npy"), neighbors.astype(np.int64))
    assert info == {"n_train": 50, "n_queries": 12, "dim": 6, "distance": "angular"}


def test_glove_conversion_rejects_non_angular_and_zero_vectors(tmp_path):
    rng = np.random.default_rng(3)
    raw = tmp_path / "l2.hdf5"
    write_hdf5(raw, rng.standard_normal((20, 4)), rng.standard_normal((3, 4)), np.zeros((3, 100), int), "euclidean")
    with pytest.raises(ValueError, match="angular"):
        prepare_glove.convert(raw, tmp_path / "a")
    train = rng.standard_normal((20, 4))
    train[5] = 0.0
    raw = tmp_path / "zero.hdf5"
    write_hdf5(raw, train, rng.standard_normal((3, 4)), np.zeros((3, 100), int))
    with pytest.raises(ValueError, match="zero vector"):
        prepare_glove.convert(raw, tmp_path / "b")


def test_glove_manifest_matches_the_decision():
    manifest = prepare_glove.manifest_template()
    assert manifest["name"] == "glove-100-angular"
    assert manifest["source"]["url"] == "http://ann-benchmarks.com/glove-100-angular.hdf5"
    assert manifest["ground_truth"]["n"] == [100_000, 300_000, 1_183_514]
    assert manifest["ground_truth"]["k"] == 100


# ---- manifest → registry → runner ----


def make_npy_dataset(tmp_path, n=300, nq=120, dim=8, frozen_n=(300,)):
    rng = np.random.default_rng(4)
    directory = tmp_path / "ds"
    directory.mkdir()
    train = prepare_vectors(rng.standard_normal((n, dim)))
    queries = prepare_vectors(rng.standard_normal((nq, dim)))
    np.save(directory / "train.npy", train)
    np.save(directory / "queries.npy", queries)
    gts = dataprep.exact_ground_truth(train, queries, list(frozen_n), k=100)
    files = {"train.npy": dataprep.file_record(directory / "train.npy"),
             "queries.npy": dataprep.file_record(directory / "queries.npy")}
    for m, gt in gts.items():
        np.save(directory / f"gt_n{m}.npy", gt)
        files[f"gt_n{m}.npy"] = dataprep.file_record(directory / f"gt_n{m}.npy")
    manifest = {"name": "tiny-npy", "kind": "npy", "metric": "angular", "directory": str(directory),
                "train": "train.npy", "queries": "queries.npy", "neighbors": None, "files": files,
                "ground_truth": {"k": 100, "n": list(frozen_n), "files": {str(m): f"gt_n{m}.npy" for m in frozen_n}}}
    path = tmp_path / "tiny-npy.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, train, queries


def test_manifests_register_npy_datasets(tmp_path):
    path, train, _ = make_npy_dataset(tmp_path)
    registry = datasets.load_registry(tmp_path)
    dataset = registry["tiny-npy"]
    assert isinstance(dataset, datasets.NpyDataset)
    assert dataset.ground_truth == {300: "gt_n300.npy"}
    assert set(dataset.sha256) == {"train.npy", "queries.npy", "gt_n300.npy"}
    np.testing.assert_array_equal(datasets.load(dataset).train, train)
    assert "smoke-gaussian" in registry  # synthetic 的 smoke 資料集仍在


def test_repo_manifests_load():
    registry = datasets.load_registry()
    for name, dataset in registry.items():
        assert dataset.name == name


def test_runner_uses_verified_frozen_ground_truth(tmp_path):
    _, train, queries = make_npy_dataset(tmp_path, frozen_n=(300,))
    registry = datasets.load_registry(tmp_path)
    text = stub_config(["exact-stub"], n=(200, 300)).replace('"smoke-gaussian"', '"tiny-npy"')
    result = run(text, "smoke", tmp_path / "runs", implementations=STUBS, registry=registry)
    subsets = result.record["subsets"]
    assert subsets["300"]["gt_source"]["kind"] == "frozen"
    assert subsets["300"]["gt_source"]["file"] == "gt_n300.npy"
    assert subsets["200"]["gt_source"]["kind"] == "computed"
    np.testing.assert_array_equal(np.load(result.run_dir / subsets["300"]["gt_file"]), exact_top(train, queries, 10))
    for point in [p for c in result.record["children"] for p in c["points"]]:
        assert point["k10"]["recall"] == 1.0


def test_runner_refuses_a_tampered_frozen_ground_truth(tmp_path):
    _, _, _ = make_npy_dataset(tmp_path, frozen_n=(300,))
    gt_path = tmp_path / "ds" / "gt_n300.npy"
    gt = np.load(gt_path)
    gt[3, [0, 1]] = gt[3, [1, 0]]
    np.save(gt_path, gt)
    registry = datasets.load_registry(tmp_path)
    text = stub_config(["exact-stub"], n=(300,)).replace('"smoke-gaussian"', '"tiny-npy"')
    with pytest.raises(ValueError, match="SHA-256"):
        run(text, "smoke", tmp_path / "runs", implementations=STUBS, registry=registry)


def test_registered_vectors_are_used_unchanged_and_validated():
    from benchmarks.run import vectors_for_run

    x = prepare_vectors(np.random.default_rng(5).standard_normal((50, 7)))
    assert vectors_for_run(x, prepared=True) is x
    with pytest.raises(ValueError, match="unit"):
        vectors_for_run(x * 2, prepared=True)
    with pytest.raises(ValueError, match="float32"):
        vectors_for_run(x.astype(np.float64), prepared=True)
    raw = np.random.default_rng(6).standard_normal((5, 3))
    np.testing.assert_array_equal(vectors_for_run(raw, prepared=False), prepare_vectors(raw))
