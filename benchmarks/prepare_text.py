"""準備真實文字 workload：Simple English Wikipedia + EmbeddingGemma 2（decision 011 §3）。

用法（在 repo root；資料準備不需要 Mac heavy lock）：
    .venv/bin/python -m benchmarks.prepare_text --name simplewiki-embeddinggemma2-pilot --n-base 10000 --n-queries 1000

步驟：
1. 驗證原始 parquet 的 SHA-256，建立 deterministic、disjoint 的 base / query passage（`text_corpus`）。
2. 在 `.venv-embed` 以 `benchmarks.embed_text` 產生 768 維 embedding（fp32、CPU）。
3. 以 ExactIndex 產生 frozen ground truth，寫出 manifest（`benchmarks/manifests/<名稱>.json`）。
"""

import argparse
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np

from benchmarks import dataprep, text_corpus

CORPUS = {
    "dataset": "wikimedia/wikipedia",
    "config": "20231101.simple",
    "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "file": "20231101.simple/train-00000-of-00001.parquet",
    "sha256": "31bded16768a47c286becd292079122f5d7d4397a17b87d4250a00ccd581e6f0",
    "bytes": 156_885_218,
    "license": "CC BY-SA 3.0 and GFDL",
}
RAW = dataprep.DATASETS_DIR / "simplewiki" / "raw" / "train-00000-of-00001.parquet"
EMBED_PYTHON = dataprep.REPO / ".venv-embed" / "bin" / "python"


def default_embed_command(jobs: list[tuple[Path, Path]], meta: Path) -> list[str]:
    command = [str(EMBED_PYTHON), "-m", "benchmarks.embed_text", "--meta", str(meta)]
    for source, target in jobs:
        command += ["--job", f"{source}:{target}"]
    return command


def write_passages(path: Path, passages) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in passages:
            record = {"passage_id": p.passage_id, "article_id": p.article_id, "title": p.title, "text": p.text,
                      "input": text_corpus.format_document(p.title, p.text)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare(name: str, raw: Path, n_base: int, n_queries: int, gt_n, embed_command=default_embed_command,
            datasets_dir=dataprep.DATASETS_DIR, manifest_dir=dataprep.MANIFEST_DIR, corpus=CORPUS) -> Path:
    digest = dataprep.sha256_file(raw)
    if digest != corpus["sha256"]:
        raise ValueError(f"{raw}: SHA-256 {digest} does not match the pinned corpus {corpus['sha256']}")

    split = text_corpus.build_split(text_corpus.read_parquet_articles(raw), n_base, n_queries)
    directory = Path(datasets_dir) / name
    directory.mkdir(parents=True, exist_ok=True)
    write_passages(directory / "base.jsonl", split.base)
    write_passages(directory / "queries.jsonl", split.queries)

    meta_path = directory / "embedding.json"
    jobs = [(directory / "base.jsonl", directory / "train.npy"), (directory / "queries.jsonl", directory / "queries.npy")]
    subprocess.run(embed_command(jobs, meta_path), cwd=dataprep.REPO, check=True)
    embedding = json.loads(meta_path.read_text(encoding="utf-8"))

    train = np.load(directory / "train.npy")
    queries = np.load(directory / "queries.npy")
    if train.shape[0] != n_base or queries.shape[0] != n_queries or train.shape[1] != queries.shape[1]:
        raise ValueError(f"unexpected embedding shapes {train.shape} and {queries.shape}")
    gt_files = {}
    for n, gt in dataprep.exact_ground_truth(train, queries, gt_n).items():
        np.save(directory / f"gt_n{n}.npy", gt)
        gt_files[str(n)] = f"gt_n{n}.npy"

    files = ["train.npy", "queries.npy", *gt_files.values(), "base.jsonl", "queries.jsonl", "embedding.json"]
    manifest = {
        "name": name,
        "kind": "npy",
        "metric": "angular",
        "decision": "docs/decisions/011-datasets.md",
        "directory": dataprep.relative_to_repo(directory),
        "train": "train.npy",
        "queries": "queries.npy",
        "neighbors": None,
        "source": {**corpus, "raw_file": dataprep.relative_to_repo(raw)},
        "preprocessing": {
            "paragraphs": "split on newline, NFC, collapse whitespace, truncate at the last space within max_chars, "
                          "then keep >= min_chars",
            "min_chars": text_corpus.MIN_CHARS,
            "max_chars": text_corpus.MAX_CHARS,
            "input_format": "title: {title} | text: {passage}",
        },
        "split": split.metadata(),
        "embedding": embedding,
        "subset_rule": "first n rows of train.npy (base passages in salted-SHA-256 article order)",
        "ground_truth": {"k": dataprep.GT_K, "n": sorted(int(n) for n in gt_files), "files": gt_files},
        "files": {name_: dataprep.file_record(directory / name_) for name_ in files},
        # NFC 與空白的定義取決於 Python 內建的 Unicode 資料庫，所以一併記錄它的版本。
        "prepared_with": {**dataprep.prepared_with("benchmarks/prepare_text.py", ["numpy", "pyarrow"]),
                          "unicodedata": unicodedata.unidata_version},
    }
    path = Path(manifest_dir) / f"{name}.json"
    dataprep.write_manifest(manifest, path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--n-base", type=int, required=True)
    parser.add_argument("--n-queries", type=int, required=True)
    parser.add_argument("--gt-n", type=int, action="append", help="frozen ground truth 的 n（預設為 n-base）")
    args = parser.parse_args(argv)
    if not RAW.exists():
        raise SystemExit(f"{RAW} not found; download {CORPUS['file']} at revision {CORPUS['revision']} first")
    if not EMBED_PYTHON.exists():
        raise SystemExit(f"{EMBED_PYTHON} not found; create .venv-embed from requirements-embed.txt first")
    path = prepare(args.name, RAW, args.n_base, args.n_queries, args.gt_n or [args.n_base])
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
