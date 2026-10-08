"""benchmark runner 的 config 解析與驗證（decision 008 §3）。"""

from pathlib import Path

import pytest

from benchmarks.config import Build, ConfigError, check_dataset_fit, execution_order, parse_config

VALID = """
schema_version = 1
dataset = "smoke-gaussian"
implementations = ["rust-hnsw", "hnswlib", "faiss-hnsw"]
ef_search = [32, 10]
child_timeout_s = 60

[[sweep]]
n = [300, 600]
m = [8]
ef_construction = [32]
seeds = [1, 2]
shuffles = [7]
"""


def with_line(old: str, new: str) -> str:
    assert old in VALID
    return VALID.replace(old, new)


def test_valid_config_expands_seed_and_order_variance_without_full_cross():
    config = parse_config(VALID)
    assert config.dataset == "smoke-gaussian"
    assert config.ef_search == (10, 32)  # runner 排序
    assert config.child_timeout_s == 60
    assert config.builds == (
        Build(n=300, m=8, ef_construction=32, seed=1, shuffle=None),
        Build(n=300, m=8, ef_construction=32, seed=2, shuffle=None),
        Build(n=300, m=8, ef_construction=32, seed=1, shuffle=7),
        Build(n=600, m=8, ef_construction=32, seed=1, shuffle=None),
        Build(n=600, m=8, ef_construction=32, seed=2, shuffle=None),
        Build(n=600, m=8, ef_construction=32, seed=1, shuffle=7),
    )
    assert config.text == VALID


def test_execution_order_rotates_implementations_per_build():
    order = execution_order(parse_config(VALID))
    assert [group for group, _, _ in order] == list(range(6))
    assert order[0][2] == ("rust-hnsw", "hnswlib", "faiss-hnsw")
    assert order[1][2] == ("hnswlib", "faiss-hnsw", "rust-hnsw")
    assert order[2][2] == ("faiss-hnsw", "rust-hnsw", "hnswlib")
    assert order[3][2] == ("rust-hnsw", "hnswlib", "faiss-hnsw")


def test_multiple_sweeps_are_concatenated():
    text = VALID + "\n[[sweep]]\nn = [300]\nm = [16]\nef_construction = [64]\nseeds = [3]\nshuffles = []\n"
    config = parse_config(text)
    assert config.builds[-1] == Build(n=300, m=16, ef_construction=64, seed=3, shuffle=None)
    assert len(config.builds) == 7


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (with_line("schema_version = 1", "schema_version = 2"), "schema_version"),
        (VALID + "k = 1\n", "unknown key"),
        (with_line("child_timeout_s = 60", "child_timeout_s = 60\nef_serach = [10]"), "unknown key"),
        (with_line("shuffles = [7]", "shuffles = [7]\nk = 10"), "unknown key"),
        (with_line('dataset = "smoke-gaussian"\n', ""), "missing"),
        (with_line("seeds = [1, 2]\n", ""), "missing"),
        (with_line("m = [8]", "m = [true]"), "integer"),
        (with_line("child_timeout_s = 60", "child_timeout_s = true"), "integer"),
        (with_line("ef_search = [32, 10]", "ef_search = [32.0, 10]"), "integer"),
        (with_line("m = [8]", "m = 8"), "list"),
        (with_line("m = [8]", "m = []"), "empty"),
        (with_line("seeds = [1, 2]", "seeds = [1, 1]"), "duplicate"),
        (with_line("shuffles = [7]", "shuffles = [7, 7]"), "duplicate"),
        (with_line("ef_search = [32, 10]", "ef_search = [10, 10]"), "duplicate"),
        (with_line("m = [8]", "m = [1]"), "m"),
        (with_line("m = [8]", "m = [2147483648]"), "m"),
        (with_line("ef_construction = [32]", "ef_construction = [7]"), "ef_construction"),
        (with_line("ef_search = [32, 10]", "ef_search = [9, 32]"), "ef_search"),
        (with_line("n = [300, 600]", "n = [9]"), "n"),
        (with_line("seeds = [1, 2]", "seeds = [-1]"), "seed"),
        (with_line("seeds = [1, 2]", "seeds = [9223372036854775808]"), "seed"),
        (with_line("shuffles = [7]", "shuffles = [-7]"), "shuffle"),
        (with_line("child_timeout_s = 60", "child_timeout_s = 0"), "child_timeout_s"),
        (with_line('"faiss-hnsw"]', '"faiss"]'), "implementation"),
        (with_line('["rust-hnsw", "hnswlib", "faiss-hnsw"]', '["rust-hnsw", "rust-hnsw"]'), "duplicate"),
        (with_line('dataset = "smoke-gaussian"', 'dataset = "no-such-dataset"'), "dataset"),
        (VALID.split("[[sweep]]")[0], "sweep"),
        (VALID + "\n[[sweep]]\nn = [300]\nm = [8]\nef_construction = [32]\nseeds = [2]\nshuffles = []\n", "duplicate build"),
        ("not = [valid toml", "TOML"),
        (with_line("schema_version = 1", "schema_version = true"), "schema_version"),
        (with_line("schema_version = 1", "schema_version = 1.0"), "schema_version"),
        (with_line("seeds = [1, 2]", "seeds = []"), "empty"),
        (with_line('["rust-hnsw", "hnswlib", "faiss-hnsw"]', "[]"), "implementations"),
        (with_line('dataset = "smoke-gaussian"', "dataset = 3"), "dataset"),
        (with_line('["rust-hnsw", "hnswlib", "faiss-hnsw"]', '["rust-hnsw", 3]'), "implementation"),
        (VALID.replace("[[sweep]]", "[sweep]"), "sweep"),
    ],
)
def test_invalid_configs_are_rejected(text, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(text)


def test_config_boundaries_are_accepted():
    text = with_line("n = [300, 600]", "n = [10]")
    text = text.replace("m = [8]", "m = [2]").replace("ef_construction = [32]", "ef_construction = [2]")
    text = text.replace("ef_search = [32, 10]", "ef_search = [10]").replace("seeds = [1, 2]", "seeds = [0]")
    text = text.replace("shuffles = [7]", "shuffles = []")
    config = parse_config(text)
    assert config.builds == (Build(n=10, m=2, ef_construction=2, seed=0, shuffle=None),)
    assert config.ef_search == (10,)


def test_shuffles_may_be_empty():
    config = parse_config(with_line("shuffles = [7]", "shuffles = []"))
    assert all(build.shuffle is None for build in config.builds)


def test_dataset_fit_checks_n_and_query_count():
    config = parse_config(VALID)
    check_dataset_fit(config, n_train=600, n_queries=100)
    with pytest.raises(ConfigError, match="n = 600"):
        check_dataset_fit(config, n_train=599, n_queries=100)
    with pytest.raises(ConfigError, match="queries"):
        check_dataset_fit(config, n_train=600, n_queries=99)


def test_shipped_configs_are_valid():
    paths = sorted((Path(__file__).resolve().parents[1] / "benchmarks" / "configs").glob("*.toml"))
    assert paths
    for path in paths:
        parse_config(path.read_text(encoding="utf-8"))
