"""各實作的 `AnnIndex` adapter。每次 `search` 都只做一次 FFI 呼叫。"""

import operator

import numpy as np

from benchmarks.interface import IncompleteResultsError


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


# ---- HNSW adapters（decision 006、007） ----


def _integer_params(m, ef_construction, seed) -> dict:
    """參數必須是整數：None 或 float 立即 TypeError，不默默套用 library 的預設值。"""
    return {
        "m": operator.index(m),
        "ef_construction": operator.index(ef_construction),
        "seed": operator.index(seed),
    }


def _checked_ef(ef) -> int:
    ef = operator.index(ef)
    if ef < 1:
        raise ValueError(f"ef_search must be >= 1, got {ef}")
    return ef


def _require_ef(ef_search) -> None:
    if ef_search is None:
        raise RuntimeError("call set_ef_search before search (decision 006)")


# data 的 ‖v‖² 與 1 的最大容許偏差；prepare_vectors 的輸出實測偏差在 1e-6 以內（decision 007）。
_UNIT_NORM_TOLERANCE = 1e-5


def _check_data(data: np.ndarray) -> None:
    """build 的輸入必須是 prepare_vectors 的輸出（decision 007 §1）。

    hnswlib 的 ip 與 FAISS 的 IP 只在 unit vectors 上等價於 cosine；Rust 會在內部再正規化，
    剛好掩蓋這個錯誤，所以三個 adapter 都在 build 時檢查（O(n·d)，不在 search 的計時路徑上）。
    """
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError(f"data must be a non-empty 2-D array, got shape {data.shape}")
    if data.dtype != np.float32 or not data.flags.c_contiguous:
        raise ValueError("data must be prepare_vectors output: a C-contiguous float32 array")
    squared_norms = np.einsum("ij,ij->i", data, data, dtype=np.float64)
    if np.abs(squared_norms - 1.0).max() > _UNIT_NORM_TOLERANCE:
        raise ValueError("data must be prepare_vectors output: rows must have unit L2 norm")


def _check_query(queries: np.ndarray, k, dim: int, n: int) -> int:
    """hnswlib 的 knn_query 不檢查維度（會越界讀取），FAISS 只用 assert；k 的行為三者也不同。

    統一成與 Rust binding 相同的 ValueError（decision 007 §6.5）。dtype / layout 由 runner 在計時前檢查一次。
    """
    if queries.ndim != 2 or queries.shape[1] != dim:
        raise ValueError(f"dimension mismatch: expected {dim}, got shape {queries.shape}")
    k = operator.index(k)
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= {n}, got k = {k}")
    return k


class RustHnsw:
    """本專案的 Rust HNSW（decision 004、006）。"""

    name = "rust-hnsw"

    def __init__(self, m: int, ef_construction: int, seed: int) -> None:
        self._params = _integer_params(m, ef_construction, seed)
        self._index = None
        self._ef_search = None

    def build(self, data: np.ndarray) -> None:
        import rust_hnsw

        _check_data(data)
        self._index = rust_hnsw.HnswIndex(data.shape[1], **self._params)
        self._index.add(data)

    def set_ef_search(self, ef: int) -> None:
        # 先檢查再存下：失敗時保留前一個值（006）。
        self._ef_search = _checked_ef(ef)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        _require_ef(self._ef_search)
        return self._index.search(queries, k, self._ef_search)

    def save_native(self, path: str) -> None:
        """decision 005 的格式；只用於 008 §8 的輔助欄位 serialized_bytes。"""
        self._index.save(path)


# hnswlib 0.8.0 在任一 query 結果不足 k 時，對整批丟出的 RuntimeError 訊息（decision 007）。
_HNSWLIB_SHORT_RESULTS = "Cannot return the results in a contiguous 2D array"


class Hnswlib:
    """hnswlib（decision 007）：space="ip"、單執行緒，回傳的已是 1 − ip distance。"""

    name = "hnswlib"

    def __init__(self, m: int, ef_construction: int, seed: int) -> None:
        self._params = _integer_params(m, ef_construction, seed)
        self._index = None
        self._ef_search = None

    def build(self, data: np.ndarray) -> None:
        import hnswlib

        _check_data(data)
        index = hnswlib.Index(space="ip", dim=data.shape[1])
        index.init_index(
            max_elements=data.shape[0],
            ef_construction=self._params["ef_construction"],
            M=self._params["m"],
            random_seed=self._params["seed"],
        )
        index.set_num_threads(1)
        index.add_items(data, np.arange(data.shape[0]), num_threads=1)
        if self._ef_search is not None:
            index.set_ef(self._ef_search)
        self._index = index
        self._n, self._dim = data.shape

    def set_ef_search(self, ef: int) -> None:
        self._ef_search = _checked_ef(ef)
        if self._index is not None:
            self._index.set_ef(self._ef_search)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        _require_ef(self._ef_search)
        k = _check_query(queries, k, self._dim, self._n)
        try:
            labels, distances = self._index.knn_query(queries, k=k, num_threads=1)
        except RuntimeError as err:
            # 整批失敗就是整批失敗：不改成逐筆查詢（decision 007 第 3 點）。
            if _HNSWLIB_SHORT_RESULTS in str(err):
                raise IncompleteResultsError(f"hnswlib returned fewer than k = {k} results for some query") from err
            raise
        # uint64 labels zero-copy 轉成 int64（id < 2^63）。
        return labels.view(np.int64), distances

    def save_native(self, path: str) -> None:
        self._index.save_index(path)


class FaissHnsw:
    """FAISS IndexHNSWFlat + METRIC_INNER_PRODUCT（decision 007）：單執行緒，similarity 轉成 distance。"""

    name = "faiss-hnsw"

    def __init__(self, m: int, ef_construction: int, seed: int) -> None:
        self._params = _integer_params(m, ef_construction, seed)
        self._index = None
        self._ef_search = None

    def build(self, data: np.ndarray) -> None:
        import faiss

        _check_data(data)
        # OpenMP 的 nthreads 是 per-thread 設定：只對呼叫的 thread 生效（decision 007 §6）。
        self._omp_set_num_threads = faiss.omp_set_num_threads
        self._omp_set_num_threads(1)
        index = faiss.IndexHNSWFlat(data.shape[1], self._params["m"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self._params["ef_construction"]
        index.hnsw.rng = faiss.RandomGenerator(self._params["seed"])
        index.add(data)
        if self._ef_search is not None:
            index.hnsw.efSearch = self._ef_search
        self._index = index
        self._n, self._dim = data.shape

    def set_ef_search(self, ef: int) -> None:
        self._ef_search = _checked_ef(ef)
        if self._index is not None:
            self._index.hnsw.efSearch = self._ef_search

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        _require_ef(self._ef_search)
        k = _check_query(queries, k, self._dim, self._n)
        # 每次都設定：search 可能在與 build 不同的 thread 呼叫；成本約數十 ns（decision 007 §6）。
        self._omp_set_num_threads(1)
        sims, ids = self._index.search(queries, k)
        # similarity → cosine distance（in place）；補齊的 −FLT_MAX 改成 +inf（002）。
        np.subtract(1.0, sims, out=sims)
        padding = ids < 0
        if padding.any():
            sims[padding] = np.inf
        return ids, sims

    def save_native(self, path: str) -> None:
        import faiss

        faiss.write_index(self._index, path)
