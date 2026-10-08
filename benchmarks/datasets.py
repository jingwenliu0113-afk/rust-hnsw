"""benchmark 資料集的登記與載入（decision 008 §1、§4）。

- `SyntheticDataset`：只給 smoke 驗證 runner 本身（SPEC §5：隨機向量不作主要結論），formal 一律拒絕。
- `NpyDataset`：真實資料集，由 `benchmarks/manifests/<名稱>.json` 登記名稱、檔案與 SHA-256（decision 011）。
  train 的列順序就是「前 n 列」子集的順序；抽樣規則由資料集 decision 在準備檔案時落實（008 §4）。
  manifest 進 Git，資料檔放在 `datasets/`（不進 Git）。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from benchmarks.dataprep import MANIFEST_DIR, REPO


@dataclass(frozen=True)
class SyntheticDataset:
    name: str
    n_train: int
    n_queries: int
    dim: int
    seed: int
    # 非 None 時，train 由這麼多個方向重複組成（decision 004 的重複向量 stress）；queries 仍是隨機方向。
    duplicate_directions: int | None = None
    metric: str = "angular"


@dataclass(frozen=True)
class NpyDataset:
    name: str
    directory: Path
    train: str
    queries: str
    sha256: dict = field(default_factory=dict)
    metric: str = "angular"
    # 資料集自帶的 ground truth（只作 sanity check，008 §4）。
    neighbors: str | None = None
    # frozen ground truth：n → 檔名（top-100，decision 011 §1）。
    ground_truth: dict = field(default_factory=dict)


@dataclass
class LoadedDataset:
    train: np.ndarray
    queries: np.ndarray
    provided_neighbors: np.ndarray | None
    # True：檔案本身就是 prepare_vectors 的輸出（unit vectors，float32）；runner 只驗證、不重新正規化，
    # 因為對 float32 再正規化一次不保證逐位元相同，會與 frozen ground truth 用的向量不一致（011 §1）。
    prepared: bool = False


SYNTHETIC: dict[str, SyntheticDataset] = {
    "smoke-gaussian": SyntheticDataset(name="smoke-gaussian", n_train=600, n_queries=100, dim=16, seed=0),
    "smoke-duplicates": SyntheticDataset(
        name="smoke-duplicates", n_train=300, n_queries=100, dim=6, seed=0, duplicate_directions=5
    ),
}


def dataset_from_manifest(manifest: dict) -> NpyDataset:
    return NpyDataset(
        name=manifest["name"],
        directory=REPO / manifest["directory"],
        train=manifest["train"],
        queries=manifest["queries"],
        sha256={name: record["sha256"] for name, record in manifest["files"].items()},
        metric=manifest["metric"],
        neighbors=manifest.get("neighbors"),
        ground_truth={int(n): name for n, name in manifest.get("ground_truth", {}).get("files", {}).items()},
    )


def load_registry(manifest_dir=MANIFEST_DIR) -> dict[str, SyntheticDataset | NpyDataset]:
    """smoke 用的 synthetic 資料集，加上每個 manifest 登記的真實資料集。"""
    registry: dict = dict(SYNTHETIC)
    for path in sorted(Path(manifest_dir).glob("*.json")):
        dataset = dataset_from_manifest(json.loads(path.read_text(encoding="utf-8")))
        if dataset.name in registry:
            raise ValueError(f"duplicate dataset name {dataset.name!r} in {path}")
        registry[dataset.name] = dataset
    return registry


REGISTRY = load_registry()


def is_synthetic(dataset) -> bool:
    return isinstance(dataset, SyntheticDataset)


def files(dataset) -> dict[str, Path]:
    """要計算 SHA-256 的檔案；synthetic 沒有檔案。"""
    if is_synthetic(dataset):
        return {}
    names = [dataset.train, dataset.queries] + ([dataset.neighbors] if dataset.neighbors else [])
    names += [name for _, name in sorted(dataset.ground_truth.items())]
    return {name: Path(dataset.directory) / name for name in names}


def load(dataset) -> LoadedDataset:
    """回傳原始（尚未 `prepare_vectors`）的 train 與 queries。"""
    if is_synthetic(dataset):
        rng = np.random.default_rng(dataset.seed)
        if dataset.duplicate_directions is None:
            train = rng.standard_normal((dataset.n_train, dataset.dim))
        else:
            base = rng.standard_normal((dataset.duplicate_directions, dataset.dim))
            train = base[rng.permutation(np.arange(dataset.n_train) % dataset.duplicate_directions)]
        queries = rng.standard_normal((dataset.n_queries, dataset.dim))
        return LoadedDataset(train=train, queries=queries, provided_neighbors=None)
    paths = files(dataset)
    neighbors = np.load(paths[dataset.neighbors]) if dataset.neighbors else None
    return LoadedDataset(
        train=np.load(paths[dataset.train]),
        queries=np.load(paths[dataset.queries]),
        provided_neighbors=neighbors,
        prepared=True,
    )
