"""各實作的 `AnnIndex` adapter。每次 `search` 都只做一次 FFI 呼叫。"""

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
