"""runner 測試用的 stub adapter，在子 process 中以 `tests.stub_adapters:<類別>` 載入。

只依賴 numpy，不 import 任何受測的 library。行為固定，讓 runner 的語意（id 映射、Recall@1 的來源、
結果不足 k、crash 的處理）可以被精確斷言。
"""

import os

import numpy as np

from benchmarks.interface import IncompleteResultsError


def _log_call(k: int, num_queries: int) -> None:
    path = os.environ.get("STUB_CALL_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{k},{num_queries}\n")


class ExactStub:
    """以 f64 brute force 回傳 exact top-k（(distance, id) 升冪），id 為 build 時的 row 編號。"""

    name = "exact-stub"

    def __init__(self, m: int, ef_construction: int, seed: int) -> None:
        self._data = None
        self._ef = None

    def build(self, data: np.ndarray) -> None:
        self._data = np.array(data, dtype=np.float64)

    def set_ef_search(self, ef: int) -> None:
        self._ef = ef

    def _exact(self, queries: np.ndarray, k: int):
        distances = 1.0 - np.asarray(queries, dtype=np.float64) @ self._data.T
        order = np.argsort(distances, axis=1, kind="stable")[:, :k]
        return order.astype(np.int64), np.take_along_axis(distances, order, axis=1).astype(np.float32)

    def search(self, queries: np.ndarray, k: int):
        _log_call(k, queries.shape[0])
        return self._exact(queries, k)

    def save_native(self, path: str) -> None:
        # 以檔案物件寫入：np.save(path) 會自動加上 .npy 副檔名。
        with open(path, "wb") as f:
            np.save(f, self._data)


class TieStub(ExactStub):
    """k = 10 的第一欄是 exact 第 2 名（距離與第 1 名相同），k = 1 才回傳 exact 第 1 名。"""

    name = "tie-stub"

    def search(self, queries: np.ndarray, k: int):
        ids, distances = self._exact(queries, k)
        if k > 1:
            ids[:, [0, 1]] = ids[:, [1, 0]]
            distances[:, 1] = distances[:, 0]
        return ids, distances


class IncompleteStub(ExactStub):
    """模擬 hnswlib：k = 10 整批 raise IncompleteResultsError；k = 1 正常回傳。"""

    name = "incomplete-stub"

    def search(self, queries: np.ndarray, k: int):
        _log_call(k, queries.shape[0])
        if k > 1:
            raise IncompleteResultsError("stub: fewer than k results")
        return self._exact(queries, k)


class PaddingStub(ExactStub):
    """模擬 Rust / FAISS 的補齊：k = 10 時最後一欄一律是 -1 / +inf。"""

    name = "padding-stub"

    def search(self, queries: np.ndarray, k: int):
        ids, distances = self._exact(queries, k)
        if k > 1:
            ids[:, -1] = -1
            distances[:, -1] = np.inf
        return ids, distances


class CrashStub(ExactStub):
    """ef ≥ 20 時直接 abort，模擬 crash：之前完成的 ef 點必須保留。"""

    name = "crash-stub"

    def set_ef_search(self, ef: int) -> None:
        if ef >= 20:
            os.abort()
        super().set_ef_search(ef)
