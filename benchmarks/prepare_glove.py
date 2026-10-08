"""準備 glove-100-angular（decision 011 §2）：下載 ann-benchmarks 的 HDF5、轉成 `.npy`、frozen ground truth、manifest。

用法（在 repo root；不需要 Mac heavy lock，這是資料準備）：
    .venv/bin/python -m benchmarks.prepare_glove [--download]

- 保留 ann-benchmarks 的 train 列順序（已是隨機順序），所以 runner 的「前 n 列」是巢狀的隨機子集。
- 網站沒有公布 checksum：第一次下載時記錄原始檔的 SHA-256 與大小，之後必須相符。
"""

import argparse
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from benchmarks import dataprep
from benchmarks.interface import prepare_vectors

NAME = "glove-100-angular"
URL = "http://ann-benchmarks.com/glove-100-angular.hdf5"
RAW_NAME = "glove-100-angular.hdf5"
GT_N = (100_000, 300_000, 1_183_514)


def manifest_template() -> dict:
    return {
        "name": NAME,
        "kind": "npy",
        "metric": "angular",
        "decision": "docs/decisions/011-datasets.md",
        "directory": f"datasets/{NAME}",
        "train": "train.npy",
        "queries": "queries.npy",
        "neighbors": "neighbors.npy",
        "source": {
            "url": URL,
            "raw_file": f"raw/{RAW_NAME}",
            "license": "GloVe vectors (glove.twitter.27B, 100d): Public Domain Dedication and License v1.0; "
                       "ann-benchmarks: MIT",
            "split": "ann-benchmarks train_test_split(X, test_size=10000, random_state=1): rows are already shuffled",
        },
        "subset_rule": "first n rows of train.npy (original ann-benchmarks order)",
        "ground_truth": {"k": dataprep.GT_K, "n": list(GT_N), "files": {}},
        "files": {},
    }


def download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(URL, timeout=60) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    partial.rename(path)


def convert(raw_path, out_dir) -> dict:
    """HDF5 → train.npy / queries.npy（unit vectors，float32）與 neighbors.npy（int64）。"""
    import h5py

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_path, "r") as f:
        distance = f.attrs["distance"]
        distance = distance.decode() if isinstance(distance, bytes) else str(distance)
        if distance != "angular":
            raise ValueError(f"expected an angular dataset, got distance = {distance!r}")
        # prepare_vectors 遇到零向量或非有限值會 raise：不默默丟棄任何列（011 §1）。
        train = prepare_vectors(f["train"][:])
        queries = prepare_vectors(f["test"][:])
        neighbors = np.ascontiguousarray(f["neighbors"][:], dtype=np.int64)
    np.save(out / "train.npy", train)
    np.save(out / "queries.npy", queries)
    np.save(out / "neighbors.npy", neighbors)
    return {"n_train": int(train.shape[0]), "n_queries": int(queries.shape[0]), "dim": int(train.shape[1]),
            "distance": distance}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="原始檔不存在時從 ann-benchmarks 下載")
    args = parser.parse_args(argv)

    # 在開始時記錄：準備要跑很久，期間 repo 若有變動，結束時的 git 狀態就不是實際執行的程式。
    prepared_with = dataprep.prepared_with("benchmarks/prepare_glove.py", ["numpy", "h5py"])
    directory = dataprep.DATASETS_DIR / NAME
    raw = directory / "raw" / RAW_NAME
    if not raw.exists():
        if not args.download:
            raise SystemExit(f"{raw} not found; rerun with --download")
        download(raw)

    manifest_path = dataprep.MANIFEST_DIR / f"{NAME}.json"
    manifest = manifest_template()
    raw_record = dataprep.file_record(raw)
    if manifest_path.exists():
        import json

        previous = json.loads(manifest_path.read_text(encoding="utf-8"))["source"]
        if previous.get("raw_sha256") not in (None, raw_record["sha256"]):
            raise SystemExit(f"{raw} SHA-256 differs from the manifest: {raw_record['sha256']}")
    manifest["source"].update(raw_sha256=raw_record["sha256"], raw_bytes=raw_record["bytes"])

    info = convert(raw, directory)
    manifest["shape"] = info
    train = np.load(directory / "train.npy")
    queries = np.load(directory / "queries.npy")
    for n, gt in dataprep.exact_ground_truth(train, queries, GT_N).items():
        np.save(directory / f"gt_n{n}.npy", gt)
        manifest["ground_truth"]["files"][str(n)] = f"gt_n{n}.npy"
    for name in ["train.npy", "queries.npy", "neighbors.npy", *manifest["ground_truth"]["files"].values()]:
        manifest["files"][name] = dataprep.file_record(directory / name)
    manifest["prepared_with"] = prepared_with
    dataprep.write_manifest(manifest, manifest_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
