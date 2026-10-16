"""Core FP32 的 golden test（decision 019 §4、§9 第 2 步）：在動任何 Advanced C 的程式之前固定。

在確定性的 synthetic fixture 上，`HnswIndex.search` 的 ids 與 distances 的 bytes、以及 decision 005 v1 存檔的 bytes，
SHA-256 必須與登記值完全相同。Advanced C 只新增模組與類別，Core 的 FP32 路徑在原始碼層級不動；
這個測試失敗就代表 Core 的行為改變了，必須先查明，不得更新登記值來讓它通過。
"""

import hashlib

import numpy as np
import pytest

import rust_hnsw

DIM = 12
# (m, ef_construction, seed, ef_search, k) → (search 的 SHA-256, 存檔的 SHA-256)。2026-10-16 在 commit 4c5cbc2 上登記。
GOLDEN = {
    (8, 64, 7, 32, 10): ("dcca86ff8359e3becfb6737c1a3eef7dc5769fd4e76af1b9df22e0673ceed188",
                         "66c7240f38f357acbaa812172e8d07db64a6c85d9e61ef2cc82207d2fc55f91b"),
    (16, 200, 1, 10, 5): ("05578e67a503fd95e91be79a08554f0dfdfd6e3d0d2dc82e5e20d93d09db4d0f",
                          "3824d2d8e2d6187dcfdb7e896813948c1257d517a55695d64153dbbe0b267a7e"),
    (4, 16, 3, 100, 20): ("98b55674ca226cb46a8d98e02c6ac5a7bb560fee983f8533909513f1c6ed9cb9",
                          "0b9899cd78b0cc65a49c9a4b95e4b04bdbd855949ce5fb87a78ba49bb3a3b175"),
}


def lcg_matrix(rows: int, cols: int, seed: int) -> np.ndarray:
    """不依賴 numpy 亂數串流的確定性 fixture：32-bit LCG，值域 [-1, 1)。"""
    state = seed
    out = np.empty((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            state = (1664525 * state + 1013904223) % (1 << 32)
            out[i, j] = state / (1 << 31) - 1.0
    return out


def unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="module")
def fixture():
    data = unit(lcg_matrix(600, DIM, 1))
    data[100] = data[7]  # 重複向量：tie 的處理也被固定（001 的 (distance, id)）
    return data, unit(lcg_matrix(40, DIM, 2))


@pytest.mark.parametrize("params", sorted(GOLDEN))
def test_core_fp32_search_and_file_bytes_are_unchanged(fixture, tmp_path, params):
    data, queries = fixture
    m, efc, seed, ef, k = params
    index = rust_hnsw.HnswIndex(DIM, m=m, ef_construction=efc, seed=seed)
    index.add(data)
    ids, distances = index.search(queries, k, ef)
    assert ids.dtype == np.int64 and distances.dtype == np.float32
    path = tmp_path / "golden.hnsw"
    index.save(path)
    search_sha, file_sha = GOLDEN[params]
    assert hashlib.sha256(ids.tobytes() + distances.tobytes()).hexdigest() == search_sha
    assert hashlib.sha256(path.read_bytes()).hexdigest() == file_sha
