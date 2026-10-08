"""測試共用的資料與量測工具。synthetic 資料只用於 correctness，不作 benchmark 結論。"""

import time

import numpy as np

from benchmarks.interface import prepare_vectors


def random_unit(seed, n, dim):
    """`n` 個 `dim` 維的 unit vectors（`prepare_vectors` 的輸出）。"""
    return prepare_vectors(np.random.default_rng(seed).standard_normal((n, dim)))


def duplicate_data(seed, n, dim):
    """5 個方向各重複多次並打散，未正規化（decision 004：會出現結果不足 k）。"""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((5, dim)).astype(np.float32)
    return np.ascontiguousarray(base[rng.permutation(np.arange(n) % 5)])


def tie_data():
    """id 0–4 都是 [1, 0]，id 5–9 都是 [0, 1]：query [1, 0] 的前 5 名距離全為 0。"""
    return prepare_vectors(np.repeat([[1.0, 0.0], [0.0, 1.0]], 5, axis=0))


def other_thread_cpu(fn):
    """執行 `fn`，回傳 (其他 thread 消耗的 CPU 秒數, 整個 process 消耗的 CPU 秒數)。

    單執行緒時，所有工作都在呼叫的 thread 上，第一個值為 0；與 wall time、機器負載無關。
    """
    process_before, thread_before = time.process_time(), time.thread_time()
    fn()
    process_used = time.process_time() - process_before
    thread_used = time.thread_time() - thread_before
    return process_used - thread_used, process_used
