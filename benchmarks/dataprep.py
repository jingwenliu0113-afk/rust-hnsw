"""資料集準備的共用工具（decision 011 §1）：manifest、檔案 checksum、frozen ground truth。

資料檔一律放在 `datasets/<名稱>/`（不進 Git）；manifest 放在 `benchmarks/manifests/`（進 Git）。
"""

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO / "datasets"
MANIFEST_DIR = REPO / "benchmarks" / "manifests"

# frozen ground truth 的深度；runner 只用前 10 欄（tie-break 是全序，所以等於 k = 10 的答案）。
GT_K = 100
# runner 使用 frozen ground truth 前，以 ExactIndex 重新計算的 query 數（固定取前 100 個）。
GT_VERIFY_QUERIES = 100
GT_VERIFY_K = 10


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path) -> dict:
    """npy 檔的 SHA-256、大小、shape 與 dtype；其他檔案只有前兩項。"""
    path = Path(path)
    record = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        record.update(shape=list(array.shape), dtype=str(array.dtype))
    return record


def exact_ground_truth(train: np.ndarray, queries: np.ndarray, n_values, k: int = GT_K) -> dict[int, np.ndarray]:
    """每個 n 在 train 的前 n 列上，以 ExactIndex（001 的 f64 與 (distance, id) tie-break）計算 top-k。"""
    import rust_hnsw

    result = {}
    for n in sorted(set(int(n) for n in n_values)):
        if not k <= n <= train.shape[0]:
            raise ValueError(f"n = {n} must satisfy {k} <= n <= {train.shape[0]}")
        index = rust_hnsw.ExactIndex(train.shape[1])
        index.add(train[:n])
        result[n] = np.ascontiguousarray(index.search(queries, k)[0], dtype=np.int64)
        del index
    return result


def verify_frozen_ground_truth(gt: np.ndarray, train_subset: np.ndarray, queries: np.ndarray) -> None:
    """frozen ground truth 必須仍是 exact 答案：重新計算前 100 個 query 的 top-10，必須完全相同。"""
    import rust_hnsw

    if gt.ndim != 2 or gt.shape[0] != queries.shape[0] or gt.shape[1] < GT_VERIFY_K:
        raise ValueError(f"frozen ground truth has shape {gt.shape}, expected ({queries.shape[0]}, >= {GT_VERIFY_K})")
    sample = min(GT_VERIFY_QUERIES, queries.shape[0])
    index = rust_hnsw.ExactIndex(train_subset.shape[1])
    index.add(train_subset)
    expected = index.search(np.ascontiguousarray(queries[:sample]), GT_VERIFY_K)[0]
    if not np.array_equal(gt[:sample, :GT_VERIFY_K], expected):
        raise ValueError("frozen ground truth does not match exact search on the first queries")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def git_state() -> dict:
    """HEAD 與 `git status` 的逐行清單（與 run.json 的 008 §10 相同旗標），看得出 dirty 的是哪些檔案。"""
    return {"commit": _git("rev-parse", "HEAD").strip(),
            "status": _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()}


def prepared_with(script: str, packages) -> dict:
    """manifest 的準備環境：程式版本、Python、套件版本，以及計算 frozen ground truth 的 rust_hnsw build。"""
    import rust_hnsw

    return {
        "script": script,
        "git": git_state(),
        "python": sys.version,
        "versions": {name: importlib.metadata.version(name) for name in packages},
        "rust_hnsw_build_info": dict(rust_hnsw.BUILD_INFO),
    }


def write_manifest(manifest: dict, path) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def relative_to_repo(path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)
