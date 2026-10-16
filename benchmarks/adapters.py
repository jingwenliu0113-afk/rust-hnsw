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


# ---- Advanced C：vector quantization（decision 019 §3、§4）；上面的四個 adapter 不改 ----


class _RustQuantHnsw:
    """F（`rust-hnsw-fp32g`）與 S（`rust-hnsw-sq8`）共用的小基底（decision 019 §3）。

    以 Core 的 `HnswIndex::new` + `add` 建圖，再轉成子類別的 `_representation`；F 與 S 走同一份新的搜尋路徑。
    沒有 `save_native`：019 不新增檔案格式，runner 把 `serialized_bytes` 記為 null（019 §4、§11 的 005）。
    """

    name: str
    _representation: str

    def __init__(self, m: int, ef_construction: int, seed: int) -> None:
        self._params = _integer_params(m, ef_construction, seed)
        self._index = None
        self._ef_search = None

    def build(self, data: np.ndarray) -> None:
        import rust_hnsw

        _check_data(data)
        self._index = rust_hnsw.QuantHnswIndex.build(data, representation=self._representation, **self._params)

    def set_ef_search(self, ef: int) -> None:
        # 與 RustHnsw 相同：先檢查再存下，失敗時保留前一個值（006）。
        self._ef_search = _checked_ef(ef)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        _require_ef(self._ef_search)
        return self._index.search(queries, k, self._ef_search)


class RustHnswFp32g(_RustQuantHnsw):
    """臂 F：FP32 向量原樣移入新的搜尋路徑；ids、distances 與 `rust-hnsw` 逐位元相同（019 §3、§4）。"""

    name = "rust-hnsw-fp32g"
    _representation = "fp32"


class RustHnswSq8(_RustQuantHnsw):
    """臂 S：同一個 FP32 build 之後轉成 SQ8；回傳的估計 cosine distance 已在 Rust 內夾到 [0, 2]（019 §3）。"""

    name = "rust-hnsw-sq8"
    _representation = "sq8"


def faiss_hnsw_flat(data: np.ndarray, m: int, ef_construction: int, seed: int):
    """臂 Q 的移植來源：以與 `FaissHnsw.build` 相同的參數與步驟建 `IndexHNSWFlat`（019 §3）。

    `FaissHnsw` 不改，所以這裡重寫同樣的四行；兩者的 graph 相同由測試保證。呼叫端先設定單執行緒（007 §6）。
    019 §7 的診斷以這個函式與 `faiss_sq8_from_flat` 重建 Q，與 adapter 是同一段建圖程式。
    """
    import faiss

    flat = faiss.IndexHNSWFlat(data.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    flat.hnsw.efConstruction = ef_construction
    flat.hnsw.rng = faiss.RandomGenerator(seed)
    flat.add(data)
    return flat


def faiss_sq8_from_flat(flat, data: np.ndarray, m: int):
    """019 §3「Q 的移植」的第 1–3 步：把 `flat` 的 HNSW graph 移植到 `IndexHNSWSQ(d, QT_8bit, M, METRIC_INNER_PRODUCT)`。

    第 4 步（`del flat`）由呼叫端執行：019 §7 要在釋放之前核對 graph（C8）。
    `data` 必須是建 `flat` 的同一份資料；storage 以它 train 與 add。
    """
    import faiss

    index = faiss.IndexHNSWSQ(data.shape[1], faiss.ScalarQuantizer.QT_8bit, m, faiss.METRIC_INNER_PRODUCT)
    # 1. SWIG setter 做 C++ 結構複製：之後釋放 flat 不影響 index（tests/test_third_party_semantics.py 釘住）。
    index.hnsw = flat.hnsw
    index.ntotal = data.shape[0]
    # 2. codec 對齊 019 §3：per-dim min–max、不外擴範圍（FAISS 的預設也是這兩個值，這裡仍明確設定）。
    #    storage 是 index 擁有的物件；downcast 得到的 proxy 不擁有它，只在 index 存活期間使用。
    storage = faiss.downcast_index(index.storage)
    storage.sq.rangestat = faiss.ScalarQuantizer.RS_minmax
    storage.sq.rangestat_arg = 0
    storage.train(data)
    storage.add(data)
    # 3. 只 train storage 時，外層 index 的 is_trained 仍是 False（同上，由測試釘住）。
    index.is_trained = True
    return index


class FaissHnswSq8:
    """臂 Q：與 `faiss-hnsw` 同一張 graph，向量改為 FAISS 的 SQ8（`QT_8bit` + `RS_minmax`，019 §3）。單執行緒。

    FAISS 回傳的是估計的 similarity，可能 > 1；轉成 1 − ip 之後原地夾到 [0, 2]，再把補齊的列設為 +inf（019 §3）。
    """

    name = "faiss-hnsw-sq8"

    def __init__(self, m: int, ef_construction: int, seed: int) -> None:
        self._params = _integer_params(m, ef_construction, seed)
        self._index = None
        self._ef_search = None

    def build(self, data: np.ndarray) -> None:
        import faiss

        _check_data(data)
        # OpenMP 的 nthreads 是 per-thread 設定（007 §6）：Flat 的建圖、移植與 storage 的 train / add 都在這個 thread。
        self._omp_set_num_threads = faiss.omp_set_num_threads
        self._omp_set_num_threads(1)
        flat = faiss_hnsw_flat(data, **self._params)
        index = faiss_sq8_from_flat(flat, data, self._params["m"])
        # 4. 釋放 Flat（含它的 FP32 向量）；之後的記憶體量測只剩 Q（019 §3）。
        del flat
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
        # 每次都設定：search 可能在與 build 不同的 thread 呼叫（007 §6）。
        self._omp_set_num_threads(1)
        sims, ids = self._index.search(queries, k)
        # similarity → 估計的 cosine distance，原地夾到 [0, 2]（夾住是單調的，排序不變）；
        # 補齊的 −FLT_MAX 先變成 2，最後改成 +inf（002、019 §3）。
        np.subtract(1.0, sims, out=sims)
        np.clip(sims, 0.0, 2.0, out=sims)
        padding = ids < 0
        if padding.any():
            sims[padding] = np.inf
        return ids, sims

    def save_native(self, path: str) -> None:
        import faiss

        faiss.write_index(self._index, path)
