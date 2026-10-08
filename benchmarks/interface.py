"""統一的 Python batch interface（decision 002；efSearch 介面見 decision 006）。

所有 ANN 實作都透過 `AnnIndex` adapter 被測：
- 輸入：`prepare_vectors` 產生的 float32、C-contiguous、L2 正規化 2-D array。
  harness 在計時之外呼叫一次。
- 輸出：`(ids, distances)`，ids 為 int64、distances 為 float32 cosine distance，
  shape 都是 `(num_queries, k)`；不足 k 時尾端補 `-1` / `+inf`。`check_search_result` 驗證全部規則。
"""

from typing import Protocol, runtime_checkable

import numpy as np

# 回報的 cosine distance 允許超出 [0, 2] 的 rounding 誤差。
DISTANCE_RANGE_TOL = 1e-5


class IncompleteResultsError(RuntimeError):
    """實作無法對某個 query 回傳 k 個結果，而且只能整批失敗（例如 hnswlib，decision 007）。

    adapter 不得改成逐筆查詢或做任何 fallback；runner 把該點記為「結果不足 k」。
    """


@runtime_checkable
class AnnIndex(Protocol):
    name: str

    def build(self, data: np.ndarray) -> None:
        """以 `prepare_vectors` 的輸出建 index；id 為 row 編號。"""
        ...

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """一次 FFI 呼叫處理整批 queries，回傳 `(ids, distances)`。"""
        ...


@runtime_checkable
class HnswAnnIndex(AnnIndex, Protocol):
    def set_ef_search(self, ef: int) -> None:
        """設定之後 `search` 使用的 efSearch（decision 006）。

        沒有預設值：必須在 `search` 之前呼叫，且 `ef ≥ 1`。harness 在計時外呼叫。
        """
        ...


def prepare_vectors(x) -> np.ndarray:
    """轉成 float32、C-contiguous 的 unit vectors（norm 以 f64 計算），不修改輸入。

    非 2-D、空的、含 NaN / Inf（含轉 float32 後 overflow）或零向量時 raise ValueError。
    """
    # 轉 float32 時的 overflow 會變成 Inf，下面會明確檢查並報錯。
    with np.errstate(over="ignore"):
        x32 = np.asarray(x, dtype=np.float32)
    if x32.ndim != 2:
        raise ValueError(f"expected a 2-D array, got {x32.ndim}-D")
    if x32.shape[0] == 0 or x32.shape[1] == 0:
        raise ValueError(f"array is empty: shape {x32.shape}")

    non_finite = ~np.isfinite(x32).all(axis=1)
    if non_finite.any():
        raise ValueError(f"row {np.flatnonzero(non_finite)[0]} contains NaN or Inf")

    x64 = x32.astype(np.float64)
    norms = np.linalg.norm(x64, axis=1, keepdims=True)
    zero = norms[:, 0] == 0.0
    if zero.any():
        raise ValueError(f"row {np.flatnonzero(zero)[0]} is a zero vector")

    return np.ascontiguousarray(x64 / norms, dtype=np.float32)


def check_search_result(ids: np.ndarray, distances: np.ndarray, num_queries: int, k: int) -> None:
    """驗證 adapter 輸出符合 decision 002 的格式；不符合時 raise ValueError。"""
    if ids.dtype != np.int64:
        raise ValueError(f"ids must be int64, got {ids.dtype}")
    if distances.dtype != np.float32:
        raise ValueError(f"distances must be float32, got {distances.dtype}")
    expected = (num_queries, k)
    if ids.shape != expected or distances.shape != expected:
        raise ValueError(f"expected shape {expected}, got ids {ids.shape}, distances {distances.shape}")

    if (ids < -1).any():
        raise ValueError("invalid id: ids must be >= 0, or -1 for padding")

    padding = ids == -1
    if (padding[:, :-1] & ~padding[:, 1:]).any():
        raise ValueError("padding (-1) must only appear at the tail of a row")
    if not np.all(distances[padding] == np.inf):
        raise ValueError("padding (-1) must have distance +inf")

    valid = distances[~padding]
    if not np.isfinite(valid).all():
        raise ValueError("distances of valid ids must be finite")
    if ((valid < -DISTANCE_RANGE_TOL) | (valid > 2.0 + DISTANCE_RANGE_TOL)).any():
        raise ValueError("cosine distance out of range [0, 2]")

    if (distances[:, 1:] < distances[:, :-1]).any():
        raise ValueError("distances must be sorted in ascending order within each row")

    sorted_ids = np.sort(ids, axis=1)
    if ((sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] >= 0)).any():
        raise ValueError("duplicate id within a row")
