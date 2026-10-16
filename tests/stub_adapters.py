"""runner 測試用的 stub adapter，在子 process 中以 `tests.stub_adapters:<類別>` 載入。

只依賴 numpy，不 import 任何受測的 library。行為固定，讓 runner 的語意（id 映射、Recall@1 的來源、
結果不足 k、crash 的處理）可以被精確斷言。
"""

import gc
import os
import threading
import time

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


def _log(event: str) -> None:
    path = os.environ.get("STUB_CALL_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(event + "\n")


class TracingStub(ExactStub):
    """記錄每次 build / set_ef_search / search / save_native，用來釘住 008 §6 的呼叫順序與次數。"""

    name = "tracing-stub"

    def build(self, data):
        _log(f"build:{data.shape[0]}")
        super().build(data)

    def set_ef_search(self, ef):
        _log(f"ef:{ef}")
        super().set_ef_search(ef)

    def search(self, queries, k):
        _log(f"search:{k},{queries.shape[0]},{self._ef},{int(gc.isenabled())}")
        return self._exact(queries, k)

    def save_native(self, path):
        _log("save")
        super().save_native(path)


class MemoryStub(ExactStub):
    """第一次 build（library warm-up）時配置 64 MiB 並一直保留；第一次 k = 10 search 時再配置 128 MiB。

    baseline 與 peak RSS 的時點正確時：index 記憶體與 build 期間的高水位增量都很小，搜尋後的高水位多出 128 MiB。
    """

    name = "memory-stub"
    _kept = []

    def build(self, data):
        if not MemoryStub._kept:
            MemoryStub._kept.append(np.random.default_rng(0).random(64 * 2**20 // 8))
        super().build(data)

    def search(self, queries, k):
        if k > 1 and not hasattr(self, "_buffer"):
            self._buffer = np.random.default_rng(1).random(128 * 2**20 // 8)
        return self._exact(queries, k)


class NondeterministicStub(ExactStub):
    """每個 ef 的第一次整批 k = 10 search 與所有單筆 k = 10 search，把最後一欄換成 -1 / +inf。"""

    name = "nondeterministic-stub"

    def set_ef_search(self, ef):
        self._batch_calls = 0
        super().set_ef_search(ef)

    def search(self, queries, k):
        ids, distances = self._exact(queries, k)
        if k > 1:
            batch = queries.shape[0] > 1
            if batch:
                self._batch_calls += 1
            if not batch or self._batch_calls == 1:
                ids[:, -1] = -1
                distances[:, -1] = np.inf
        return ids, distances


def _spin_in_another_thread(seconds: float) -> None:
    def spin():
        end = time.thread_time() + seconds
        while time.thread_time() < end:
            pass

    worker = threading.Thread(target=spin)
    worker.start()
    worker.join()


class MultithreadStub(ExactStub):
    """build 與整批 k = 10 search 時，另一個 thread 消耗 CPU：模擬單執行緒設定失效。"""

    name = "multithread-stub"

    def build(self, data):
        _spin_in_another_thread(0.05)
        super().build(data)

    def search(self, queries, k):
        if k > 1 and queries.shape[0] > 1:
            _spin_in_another_thread(0.02)
        return self._exact(queries, k)


class SlowStub(ExactStub):
    """ef ≥ 20 時卡住 60 秒，讓測試在子 process 執行中送出中止訊號。"""

    name = "slow-stub"

    def set_ef_search(self, ef):
        if ef >= 20:
            time.sleep(60)
        super().set_ef_search(ef)


class ExitStub(ExactStub):
    """ef ≥ 20 時以非零碼直接結束（不是 signal）。"""

    name = "exit-stub"

    def set_ef_search(self, ef):
        if ef >= 20:
            os._exit(3)
        super().set_ef_search(ef)


class NoSaveStub:
    """沒有 `save_native` 的 adapter（019 的 rust-hnsw-fp32g、rust-hnsw-sq8 就是這樣）；其餘行為與 ExactStub 相同。

    以組合而不是繼承 ExactStub，類別上才真的沒有 `save_native`。runner 必須把 serialized_bytes 記為 null（019 §4）。
    """

    name = "no-save-stub"

    def __init__(self, m: int, ef_construction: int, seed: int) -> None:
        self._inner = ExactStub(m, ef_construction, seed)

    def build(self, data: np.ndarray) -> None:
        self._inner.build(data)

    def set_ef_search(self, ef: int) -> None:
        self._inner.set_ef_search(ef)

    def search(self, queries: np.ndarray, k: int):
        return self._inner.search(queries, k)


class TracingNoSaveStub(NoSaveStub):
    """TracingStub 的呼叫紀錄，但沒有 `save_native`：019 §4 的 relief 必須照樣在所有計時點之後才做。"""

    name = "tracing-no-save-stub"

    def build(self, data):
        _log(f"build:{data.shape[0]}")
        super().build(data)

    def set_ef_search(self, ef):
        _log(f"ef:{ef}")
        super().set_ef_search(ef)

    def search(self, queries, k):
        _log(f"search:{k},{queries.shape[0]},{self._inner._ef},{int(gc.isenabled())}")
        return self._inner._exact(queries, k)
