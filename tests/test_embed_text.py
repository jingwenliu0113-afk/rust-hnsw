"""EmbeddingGemma 2 的 embedding 產生（decision 011 §3.4）。

純函式的測試在 benchmark 用的 .venv 執行；載入模型的測試需要 `.venv-embed`（torch、sentence-transformers）
以及本機快取中固定 revision 的模型，否則 skip。`scripts/test.sh` 會在 `.venv-embed` 存在時另外執行這個檔案。
"""

import numpy as np
import pytest

from benchmarks import embed_text as et


def test_pinned_model_and_runtime_settings():
    assert et.MODEL_ID == "google/embeddinggemma-2"
    assert et.REVISION == "914f7f89142e33e77833254d9c9b90c3cef7303b"
    assert et.MODEL_FILE_SHA256 == "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
    assert (et.DIM, et.DTYPE, et.DEVICE, et.THREADS, et.SEED) == (768, "float32", "cpu", 8, 0)
    assert et.BATCH_SIZE == 1  # 不 padding：向量與批次組成無關


def test_requirements_pin_the_packages_that_affect_the_output():
    pins = et.parse_pins(et.REQUIREMENTS.read_text(encoding="utf-8"))
    for name in ("torch", "transformers", "sentence-transformers", "tokenizers", "safetensors", "huggingface-hub"):
        assert name in pins


def test_installed_versions_must_match_the_pins():
    pins = et.parse_pins("torch==2.14.1  # 註解\n# tokenizers==0.1\ntokenizers==0.23.2\n")
    assert pins == {"torch": "2.14.1", "tokenizers": "0.23.2"}
    assert et.mismatched_pins(pins, {"torch": "2.14.1", "tokenizers": "0.23.2"}.get) == []
    assert et.mismatched_pins(pins, {"torch": "2.14.1", "tokenizers": "0.23.3"}.get) == [
        "tokenizers 0.23.3 != pinned 0.23.2"
    ]


@pytest.mark.parametrize("dtype", ["float16", "fp16", "half"])
def test_fp16_is_refused(dtype):
    with pytest.raises(ValueError, match="fp16"):
        et.check_dtype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_bf16_and_fp32_are_allowed(dtype):
    et.check_dtype(dtype)


def test_normalization_is_done_in_f64_and_stored_as_f32():
    x = np.array([[3.0, 4.0], [1e-3, 0.0]], dtype=np.float32)
    out = et.l2_normalize(x)
    assert out.dtype == np.float32 and out.flags.c_contiguous
    np.testing.assert_allclose(out, [[0.6, 0.8], [1.0, 0.0]], rtol=0, atol=1e-7)


@pytest.mark.parametrize("bad", [[[0.0, 0.0]], [[np.nan, 1.0]], [[np.inf, 1.0]]])
def test_zero_or_non_finite_embeddings_fail_loudly(bad):
    with pytest.raises(ValueError):
        et.l2_normalize(np.array(bad, dtype=np.float32))


def test_embeddings_are_checked_for_dimension():
    with pytest.raises(ValueError, match="768"):
        et.check_embeddings(np.ones((2, 512), dtype=np.float32))


# ---- 以下需要 .venv-embed 與快取中的模型 ----


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    try:
        return et.load_model(local_files_only=True)
    except OSError:
        pytest.skip("model revision not in the local cache")


TEXTS = [
    "title: Paris | text: Paris is the capital and largest city of France.",
    "title: Rust | text: Rust is a programming language focused on safety.",
    "title: HNSW | text: Hierarchical navigable small world graphs support approximate nearest neighbor search.",
]


def test_model_produces_unit_768d_embeddings_deterministically(model):
    first = et.encode(model, TEXTS)
    second = et.encode(model, TEXTS)
    assert first.shape == (3, 768) and first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first.astype(np.float64), axis=1), 1.0, rtol=0, atol=1e-5)


def test_similar_texts_are_closer(model):
    vectors = et.encode(model, [*TEXTS, "title: France | text: The capital of France is Paris."])
    similarity = vectors @ vectors[3]
    assert int(np.argmax(similarity[:3])) == 0


def test_vectors_do_not_depend_on_batch_composition(model):
    # 長度不一的 40 段文字：舊設定（batch 32，依長度排序後 padding）下，前 32 筆單獨重新 encode 會差約 2e-7。
    words = "the quick brown fox jumps over the lazy dog".split()
    texts = [f"title: Item {i} | text: " + " ".join(words[j % len(words)] for j in range(4 + 6 * i)) for i in range(40)]
    full = et.encode(model, texts)
    np.testing.assert_array_equal(et.encode(model, texts[: et.DETERMINISM_SAMPLE]), full[: et.DETERMINISM_SAMPLE])
    np.testing.assert_array_equal(et.encode(model, texts[::-1])[::-1], full)
