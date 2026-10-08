"""benchmark 資料集的登記與載入（decision 008 §1、§4）。

- `SyntheticDataset`：只給 smoke 驗證 runner 本身（SPEC §5：隨機向量不作主要結論），formal 一律拒絕。
- `NpyDataset`：真實資料集，由資料集的 decision log 登記名稱、檔案與 SHA-256。
  train 的列順序就是「前 n 列」子集的順序；抽樣規則由資料集 decision 在準備檔案時落實（008 §4）。

目前還沒有登記任何真實資料集（需要使用者決定下載哪一組，見 NEXT.md）。
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


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


@dataclass
class LoadedDataset:
    train: np.ndarray
    queries: np.ndarray
    provided_neighbors: np.ndarray | None


REGISTRY: dict[str, SyntheticDataset | NpyDataset] = {
    "smoke-gaussian": SyntheticDataset(name="smoke-gaussian", n_train=600, n_queries=100, dim=16, seed=0),
    "smoke-duplicates": SyntheticDataset(
        name="smoke-duplicates", n_train=300, n_queries=100, dim=6, seed=0, duplicate_directions=5
    ),
}


def is_synthetic(dataset) -> bool:
    return isinstance(dataset, SyntheticDataset)


def files(dataset) -> dict[str, Path]:
    """要計算 SHA-256 的檔案；synthetic 沒有檔案。"""
    if is_synthetic(dataset):
        return {}
    names = [dataset.train, dataset.queries] + ([dataset.neighbors] if dataset.neighbors else [])
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
        train=np.load(paths[dataset.train]), queries=np.load(paths[dataset.queries]), provided_neighbors=neighbors
    )
