"""文字 workload 的準備流程（decision 011 §3）：以假的 embedding 指令與小型 parquet 走完整條流程。

真正的 embedding 由 `.venv-embed` 產生；這裡只驗證切分、檔案、frozen ground truth、manifest 與 runner 的銜接。
"""

import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks import dataprep, datasets, prepare_text
from benchmarks.run import run
from test_runner import STUBS, stub_config

FAKE_EMBED = """
import json, sys, numpy as np
args = sys.argv[1:]
meta = args[args.index("--meta") + 1]
jobs = [args[i + 1] for i, a in enumerate(args) if a == "--job"]
for k, job in enumerate(jobs):
    source, target = job.split(":")
    texts = [json.loads(line)["input"] for line in open(source, encoding="utf-8")]
    x = np.random.default_rng(k).standard_normal((len(texts), 12))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    np.save(target, x.astype(np.float32))
json.dump({"model": "fake", "jobs": jobs}, open(meta, "w"))
"""


def fake_embed(jobs, meta):
    command = [sys.executable, "-c", FAKE_EMBED, "--meta", str(meta)]
    for source, target in jobs:
        command += ["--job", f"{source}:{target}"]
    return command


@pytest.fixture
def corpus(tmp_path):
    rows = {"id": [], "title": [], "text": []}
    for i in range(400):
        rows["id"].append(str(i))
        rows["title"].append(f"Title {i}")
        rows["text"].append("\n".join(f"article {i} paragraph {j} " + "word " * 50 for j in range(2)))
    path = tmp_path / "wiki.parquet"
    pq.write_table(pa.table(rows), path)
    return path, {**prepare_text.CORPUS, "sha256": dataprep.sha256_file(path)}


def test_preparation_writes_files_ground_truth_and_a_registrable_manifest(tmp_path, corpus):
    raw, pinned = corpus
    manifest_path = prepare_text.prepare("tiny-text", raw, n_base=300, n_queries=100, gt_n=[300],
                                         embed_command=fake_embed, datasets_dir=tmp_path / "data",
                                         manifest_dir=tmp_path, corpus=pinned)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    directory = tmp_path / "data" / "tiny-text"
    assert manifest["split"]["n_base"] == 300 and manifest["split"]["n_queries"] == 100
    assert manifest["ground_truth"] == {"k": 100, "n": [300], "files": {"300": "gt_n300.npy"}}
    for name, record in manifest["files"].items():
        assert record["sha256"] == dataprep.sha256_file(directory / name)
    base = [json.loads(line) for line in open(directory / "base.jsonl", encoding="utf-8")]
    queries = [json.loads(line) for line in open(directory / "queries.jsonl", encoding="utf-8")]
    assert base[0]["input"] == f"title: {base[0]['title']} | text: {base[0]['text']}"
    assert {q["article_id"] for q in queries}.isdisjoint({b["article_id"] for b in base})
    gt = np.load(directory / "gt_n300.npy")
    dataprep.verify_frozen_ground_truth(gt, np.load(directory / "train.npy"), np.load(directory / "queries.npy"))

    registry = datasets.load_registry(tmp_path)
    text = stub_config(["exact-stub"], n=(300,)).replace('"smoke-gaussian"', '"tiny-text"')
    result = run(text, "smoke", tmp_path / "runs", implementations=STUBS, registry=registry)
    assert result.record["subsets"]["300"]["gt_source"]["kind"] == "frozen"
    assert result.record["status"] == "complete"


def test_preparation_refuses_an_unpinned_corpus(tmp_path, corpus):
    raw, pinned = corpus
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_text.prepare("x", raw, 10, 2, [10], embed_command=fake_embed, datasets_dir=tmp_path,
                             manifest_dir=tmp_path, corpus={**pinned, "sha256": "0" * 64})
