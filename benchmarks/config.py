"""benchmark config 的解析與驗證（decision 008 §3）。驗證都在任何量測之前完成，不符就拒絕執行。"""

import tomllib
from dataclasses import dataclass

from benchmarks import datasets
from benchmarks.protocol import K, MIN_QUERIES

# 實作名稱 → (adapter 的 "module:類別", 子 process 唯一允許 import 的受測 library)。
IMPLEMENTATIONS = {
    "rust-hnsw": ("benchmarks.adapters:RustHnsw", "rust_hnsw"),
    "hnswlib": ("benchmarks.adapters:Hnswlib", "hnswlib"),
    "faiss-hnsw": ("benchmarks.adapters:FaissHnsw", "faiss"),
    # Advanced C（decision 019 §3、§4）：F、S、Q。
    "rust-hnsw-fp32g": ("benchmarks.adapters:RustHnswFp32g", "rust_hnsw"),
    "rust-hnsw-sq8": ("benchmarks.adapters:RustHnswSq8", "rust_hnsw"),
    "faiss-hnsw-sq8": ("benchmarks.adapters:FaissHnswSq8", "faiss"),
}

# 與 004 / 005 一致：2 ≤ M ≤ 2^31 − 1。
M_MAX = 2**31 - 1
SEED_MAX = 2**63 - 1

_TOP_KEYS = {"schema_version", "dataset", "implementations", "ef_search", "child_timeout_s", "sweep"}
_SWEEP_KEYS = {"n", "m", "ef_construction", "seeds", "shuffles"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Build:
    n: int
    m: int
    ef_construction: int
    seed: int
    # None 表示原始順序；否則為 `np.random.default_rng(shuffle).permutation(n)` 的 seed。
    shuffle: int | None


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    implementations: tuple[str, ...]
    ef_search: tuple[int, ...]
    child_timeout_s: int
    builds: tuple[Build, ...]
    raw: dict
    text: str


def _check_keys(table: dict, allowed: set, where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {unknown}")
    missing = sorted(allowed - set(table))
    if missing:
        raise ConfigError(f"missing key(s) in {where}: {missing}")


def _int(value, name: str, low: int, high: int | None = None) -> int:
    # TOML 的 true 是 bool，而 bool 是 int 的子類別：一律以 type(v) is int 檢查。
    if type(value) is not int:
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if value < low or (high is not None and value > high):
        bound = f">= {low}" if high is None else f"in [{low}, {high}]"
        raise ConfigError(f"{name} must be {bound}, got {value}")
    return value


def _int_list(value, name: str, low: int, high: int | None = None, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, list):
        raise ConfigError(f"{name} must be a list, got {value!r}")
    if not value and not allow_empty:
        raise ConfigError(f"{name} must not be empty")
    items = [_int(v, name, low, high) for v in value]
    if len(set(items)) != len(items):
        raise ConfigError(f"{name} has duplicate values: {value}")
    return items


def parse_config(text: str, implementations=IMPLEMENTATIONS, registry=None) -> RunConfig:
    registry = datasets.REGISTRY if registry is None else registry
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as err:
        raise ConfigError(f"invalid TOML: {err}") from err

    if "sweep" not in raw:
        raise ConfigError("config needs at least one [[sweep]] table")
    _check_keys(raw, _TOP_KEYS, "config")
    if raw["schema_version"] != 1 or type(raw["schema_version"]) is not int:
        raise ConfigError(f"schema_version must be 1, got {raw['schema_version']!r}")

    dataset = raw["dataset"]
    if not isinstance(dataset, str):
        raise ConfigError(f"dataset must be a string, got {dataset!r}")
    if dataset not in registry:
        raise ConfigError(f"unknown dataset {dataset!r}; registered: {sorted(registry)}")

    names = raw["implementations"]
    if not isinstance(names, list) or not names:
        raise ConfigError("implementations must be a non-empty list")
    for name in names:
        if not isinstance(name, str):
            raise ConfigError(f"implementation names must be strings, got {name!r}")
        if name not in implementations:
            raise ConfigError(f"unknown implementation {name!r}; expected one of {sorted(implementations)}")
    if len(set(names)) != len(names):
        raise ConfigError(f"implementations has duplicate values: {names}")

    # 006 / 007 §4：ef < k 時三者語意不同，必須在計時前擋下。
    ef_search = sorted(_int_list(raw["ef_search"], "ef_search", K))
    child_timeout_s = _int(raw["child_timeout_s"], "child_timeout_s", 1)

    sweeps = raw["sweep"]
    if not isinstance(sweeps, list) or not sweeps:
        raise ConfigError("config needs at least one [[sweep]] table")
    builds: list[Build] = []
    for index, sweep in enumerate(sweeps):
        where = f"sweep #{index + 1}"
        if not isinstance(sweep, dict):
            raise ConfigError(f"{where} must be a table")
        _check_keys(sweep, _SWEEP_KEYS, where)
        ns = _int_list(sweep["n"], f"{where}: n", K)
        ms = _int_list(sweep["m"], f"{where}: m", 2, M_MAX)
        efcs = _int_list(sweep["ef_construction"], f"{where}: ef_construction", 1)
        seeds = _int_list(sweep["seeds"], f"{where}: seed", 0, SEED_MAX)
        shuffles = _int_list(sweep["shuffles"], f"{where}: shuffle", 0, SEED_MAX, allow_empty=True)
        for m in ms:
            for efc in efcs:
                # Rust 與 hnswlib 實際使用 max(efC, M)，FAISS 沒有這條規則（008 §3）。
                if efc < m:
                    raise ConfigError(f"{where}: ef_construction must be >= m, got ef_construction = {efc} < m = {m}")
        for n in ns:
            for m in ms:
                for efc in efcs:
                    # 不做 seed × order 的全交叉（010 §4）。
                    builds.extend(Build(n, m, efc, seed, None) for seed in seeds)
                    builds.extend(Build(n, m, efc, seeds[0], shuffle) for shuffle in shuffles)

    seen = set()
    for build in builds:
        if build in seen:
            raise ConfigError(f"duplicate build across sweeps: {build}")
        seen.add(build)

    return RunConfig(
        dataset=dataset,
        implementations=tuple(names),
        ef_search=tuple(ef_search),
        child_timeout_s=child_timeout_s,
        builds=tuple(builds),
        raw=raw,
        text=text,
    )


def check_dataset_fit(config: RunConfig, n_train: int, n_queries: int) -> None:
    too_large = sorted({b.n for b in config.builds if b.n > n_train})
    if too_large:
        raise ConfigError(f"n = {too_large[0]} exceeds the {n_train} train rows of {config.dataset}")
    if n_queries < MIN_QUERIES:
        raise ConfigError(f"need at least {MIN_QUERIES} queries, {config.dataset} has {n_queries}")


def execution_order(config: RunConfig) -> list[tuple[int, Build, tuple[str, ...]]]:
    """外層為 build、內層為實作；第 g 個 build 把實作清單輪換 g mod（實作數）位（008 §5）。"""
    names = config.implementations
    order = []
    for group, build in enumerate(config.builds):
        shift = group % len(names)
        order.append((group, build, names[shift:] + names[:shift]))
    return order
