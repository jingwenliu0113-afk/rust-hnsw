"""各實作的 `AnnIndex` adapter。每次 `search` 都只做一次 FFI 呼叫。"""

import operator

import numpy as np


class RustExact:
    """Rust exact brute-force index：ground truth 與 correctness baseline。"""

    name = "rust-exact"

    def __init__(self) -> None:
        self._index = None

    def build(self, data: np.ndarray) -> None:
        import rust_hnsw

        self._index = rust_hnsw.ExactIndex(data.shape[1])
        self._index.add(data)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self._index.search(queries, k)


class RustHnsw:
    """本專案的 Rust HNSW（decision 004、006）。"""

    name = "rust-hnsw"

    def __init__(self, m: int, ef_construction: int, seed: int) -> None:
        # 必須是整數：None 或 float 立即 TypeError，不默默套用 binding 的預設值（006）。
        self._params = {
            "m": operator.index(m),
            "ef_construction": operator.index(ef_construction),
            "seed": operator.index(seed),
        }
        self._index = None
        self._ef_search = None

    def build(self, data: np.ndarray) -> None:
        import rust_hnsw

        self._index = rust_hnsw.HnswIndex(data.shape[1], **self._params)
        self._index.add(data)

    def set_ef_search(self, ef: int) -> None:
        # 先檢查再存下：失敗時保留前一個值（006）。
        ef = operator.index(ef)
        if ef < 1:
            raise ValueError(f"ef_search must be >= 1, got {ef}")
        self._ef_search = ef

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._ef_search is None:
            raise RuntimeError("call set_ef_search before search (decision 006)")
        return self._index.search(queries, k, self._ef_search)
