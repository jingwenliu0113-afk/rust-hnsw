"""以 EmbeddingGemma 2 產生文字 embedding（decision 011 §3.4）。在獨立的 `.venv-embed` 執行。

用法（通常由 `benchmarks.prepare_text` 呼叫）：
    .venv-embed/bin/python -m benchmarks.embed_text --job base.jsonl:train.npy --job queries.jsonl:queries.npy --meta embedding.json

- 輸入 JSONL 的每一行有 `input` 欄位（已是 `title: … | text: …` 格式）。
- 固定 model revision、fp32、CPU、thread 數與 seed；不用 fp16（model card 明示會產生 NaN 或劣化的 embedding）。
- 完整 768 維，不做 Matryoshka truncation；以 f64 計算 norm 正規化後存成 float32。
- batch size 1：每段文字單獨 forward、不 padding，向量只取決於文字與固定的 runtime，與批次組成無關
  （有無 padding 的輸出差約 2e-7，011 §3.4 補記）。
- 對每個 job 的前 32 筆單獨重新 encode 一次，必須逐位元相同。
- 開始前比對已安裝版本與 `requirements-embed.txt`，不符就拒絕。
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements-embed.txt"

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
MODEL_FILE = "model.safetensors"
MODEL_FILE_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
DIM = 768
DTYPE = "float32"
DEVICE = "cpu"
THREADS = 8
SEED = 0
BATCH_SIZE = 1
DETERMINISM_SAMPLE = 32
UNIT_NORM_TOLERANCE = 1e-5

_FP16 = {"float16", "fp16", "half"}
_ALLOWED = {"float32", "bfloat16"}


def check_dtype(dtype: str) -> None:
    if dtype in _FP16:
        raise ValueError("fp16 is not allowed for EmbeddingGemma 2 (NaN or degraded embeddings); use bfloat16 or float32")
    if dtype not in _ALLOWED:
        raise ValueError(f"dtype must be one of {sorted(_ALLOWED)}, got {dtype!r}")


def parse_pins(requirements_text: str) -> dict[str, str]:
    """`name==version` 的固定版本；與 `environment.parse_pins` 規則相同。

    這裡不 import benchmarks.environment：它會連帶 import datasets，在 import 時讀 manifests。
    """
    pins = {}
    for line in requirements_text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def mismatched_pins(pins: dict[str, str], installed) -> list[str]:
    """`installed(name)` 回傳已安裝的版本（例如 `importlib.metadata.version`）。"""
    return [f"{name} {installed(name)} != pinned {version}" for name, version in sorted(pins.items())
            if installed(name) != version]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """以 f64 計算 norm 並正規化，存成 float32；零向量或非有限值 raise（不默默丟棄）。"""
    x64 = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x64).all():
        raise ValueError("embeddings contain NaN or Inf")
    norms = np.linalg.norm(x64, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("embeddings contain a zero vector")
    return np.ascontiguousarray(x64 / norms, dtype=np.float32)


def check_embeddings(x: np.ndarray) -> None:
    if x.ndim != 2 or x.shape[1] != DIM:
        raise ValueError(f"expected {DIM}-dimensional embeddings, got shape {x.shape}")
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("embeddings must be finite float32")
    squared = np.einsum("ij,ij->i", x, x, dtype=np.float64)
    if np.abs(squared - 1.0).max() > UNIT_NORM_TOLERANCE:
        raise ValueError("embeddings must be unit vectors")


def load_model(local_files_only: bool = False):
    import torch
    from sentence_transformers import SentenceTransformer

    check_dtype(DTYPE)
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    model = SentenceTransformer(
        MODEL_ID,
        revision=REVISION,
        device=DEVICE,
        model_kwargs={"dtype": getattr(torch, DTYPE)},
        local_files_only=local_files_only,
    )
    model.eval()
    return model


def encode(model, texts: list[str]) -> np.ndarray:
    import torch

    with torch.inference_mode():
        raw = model.encode(
            texts, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
    return l2_normalize(raw)


def model_file_sha256() -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL_ID, MODEL_FILE, revision=REVISION, local_files_only=True)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_lengths(model, texts: list[str]) -> list[int]:
    return [len(ids) for ids in model.tokenizer(texts)["input_ids"]]


def read_inputs(path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["input"] for line in f if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", action="append", required=True, help="輸入 JSONL:輸出 .npy")
    parser.add_argument("--meta", required=True, help="寫出產生紀錄的 JSON")
    args = parser.parse_args(argv)

    started = time.time()
    pins = parse_pins(REQUIREMENTS.read_text(encoding="utf-8"))
    mismatched = mismatched_pins(pins, importlib.metadata.version)
    if mismatched:
        raise SystemExit(f"installed versions differ from {REQUIREMENTS.name}: {', '.join(mismatched)}")
    model = load_model()
    digest = model_file_sha256()
    if digest != MODEL_FILE_SHA256:
        raise SystemExit(f"{MODEL_FILE} SHA-256 {digest} does not match the pinned {MODEL_FILE_SHA256}")
    max_seq_length = getattr(model, "max_seq_length", None)

    jobs = []
    for job in args.job:
        source, _, target = job.partition(":")
        texts = read_inputs(source)
        lengths = token_lengths(model, texts)
        if max_seq_length is not None and max(lengths) > max_seq_length:
            raise SystemExit(f"{source}: {max(lengths)} tokens exceed max_seq_length {max_seq_length}")
        job_start = time.time()
        vectors = encode(model, texts)
        seconds = time.time() - job_start
        again = encode(model, texts[:DETERMINISM_SAMPLE])
        if not np.array_equal(again, vectors[:DETERMINISM_SAMPLE]):
            raise SystemExit(f"{source}: re-encoding the first {DETERMINISM_SAMPLE} texts gave different bits")
        check_embeddings(vectors)
        np.save(target, vectors)
        jobs.append({"input": Path(source).name, "output": Path(target).name, "count": len(texts),
                     "max_tokens": max(lengths), "mean_tokens": float(np.mean(lengths)), "seconds": seconds,
                     "deterministic_sample": DETERMINISM_SAMPLE})

    import torch

    meta = {
        "model": MODEL_ID,
        "revision": REVISION,
        "model_file_sha256": digest,
        "dim": DIM,
        "dtype": DTYPE,
        "device": DEVICE,
        "threads": THREADS,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "padding": "none (one text per forward pass)",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "normalization": "sentence-transformers Normalize, then f64 L2 normalization stored as float32",
        "prompt": "none added by the library; inputs are 'title: {title} | text: {passage}'",
        "max_seq_length": max_seq_length,
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in pins},
        },
        "jobs": jobs,
        "seconds_total": time.time() - started,
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
