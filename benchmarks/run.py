"""benchmark runner（decision 008）：父 process 負責驗證、ground truth、執行順序、子 process 管理與指標計算。

用法（在 repo root）：
    .venv/bin/python -m benchmarks.run benchmarks/configs/smoke.toml --mode smoke
    ./tools/with-mac-lock .venv/bin/python -m benchmarks.run <formal config> --mode formal

- smoke 只驗證 runner 本身，數字不得用於報告。
- formal 必須在 with-mac-lock 之下、git 乾淨、低負載、已登記的真實資料集上執行（008 §1、§2）。
- 每個（build、實作）在獨立的子 process（`benchmarks.run_child`）中量測；recall 由這裡在子 process 結束後計算。
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from benchmarks import datasets, environment, evaluate, protocol
from benchmarks.config import IMPLEMENTATIONS, ConfigError, check_dataset_fit, execution_order, parse_config
from benchmarks.environment import PreflightError
from benchmarks.interface import prepare_vectors

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO / "results" / "runs"
# 寫入 run.json 的環境變數（008 §5）；其他變數不記錄，避免寫出不相干的機密。
CHILD_ENV_PREFIXES = ("OMP_", "KMP_", "RUSTFLAGS", "CARGO_", "PYTHON", "VIRTUAL_ENV")
# formal 模式比對 requirements-dev.txt 的 pin（007 的語意是在這些版本上實測的）。
VERSION_CHECKED = ("hnswlib", "faiss-cpu", "numpy", "maturin")
# 007 §8：報告時要一起呈現的 distance kernel 事實。
DISTANCE_KERNELS = {
    "rust-hnsw": "scalar f32 loop (decision 004; SIMD is Optional), release defaults without target-cpu=native",
    "hnswlib": "arm64: no hand-written SIMD; multiply auto-vectorized, sequential scalar reduction (decision 007 §8)",
    "faiss-hnsw": "official NEON wheel (faiss.get_compile_options)",
}
HNSWLIB_COMPILE_FLAGS = "-O3 -march=native, inferred from hnswlib 0.8.0 setup.py (decision 007); not observed"


class RunAborted(Exception):
    """SIGTERM 或 load 門檻：中止整個 run，仍寫出 run.json。"""


@dataclass
class RunResult:
    exit_code: int
    run_dir: Path
    record: dict


def default_child_command(spec_path) -> list[str]:
    return [sys.executable, "-m", "benchmarks.run_child", str(spec_path)]


def extension_files() -> dict[str, str]:
    """三個受測擴充模組的 `.so` 路徑（父 process 可以 import；子 process 是全新的 interpreter）。"""
    import faiss
    import hnswlib
    import rust_hnsw

    return {
        "rust_hnsw": rust_hnsw.rust_hnsw.__file__,
        "hnswlib": hnswlib.__file__,
        "faiss": sys.modules["faiss._swigfaiss"].__file__,
    }


def formal_preflight(context: dict) -> dict:
    """008 §1 的全部前置檢查；任一項不符就 raise PreflightError（不寫任何輸出）。"""
    import rust_hnsw

    owner = environment.check_lock()
    environment.check_git_clean(REPO)
    environment.check_venv(sys.prefix, REPO)
    pins = environment.parse_pins((REPO / "requirements-dev.txt").read_text(encoding="utf-8"))
    environment.check_versions({name: pins[name] for name in VERSION_CHECKED}, importlib.metadata.version)
    environment.check_build_info(rust_hnsw.BUILD_INFO)
    head, head_time = environment.git_head(REPO)
    so_path = extension_files()["rust_hnsw"]
    environment.check_binding_fresh(so_path, head_time)
    digests = environment.check_dataset_registered(context["dataset"])
    context.update(owner=owner, head=head, so_sha256=environment.sha256_file(so_path))
    return {"lock_owner": owner, "head": head, "rust_hnsw_sha256": context["so_sha256"], "dataset_sha256": digests}


def formal_recheck(context: dict) -> None:
    """每個子 process 啟動前重新確認（008 §1）：lock、HEAD、工作目錄、被量測的 binary 都沒有變。"""
    _, owner = environment.read_lock_owner()
    if owner != context["owner"]:
        raise PreflightError("lock owner.txt changed during the run")
    head, _ = environment.git_head(REPO)
    if head != context["head"]:
        raise PreflightError(f"HEAD changed during the run: {context['head']} -> {head}")
    environment.check_git_clean(REPO)
    if environment.sha256_file(extension_files()["rust_hnsw"]) != context["so_sha256"]:
        raise PreflightError("rust_hnsw extension module changed during the run")


@dataclass
class Hooks:
    """測試用的注入點；預設值就是正式行為。"""

    child_command: Callable = default_child_command
    formal_preflight: Callable = formal_preflight
    formal_recheck: Callable = formal_recheck
    load_reader: Callable = field(default=lambda: os.getloadavg()[0])
    sleep: Callable = time.sleep
    monotonic: Callable = time.monotonic
    snapshot: Callable = environment.snapshot


# ---- run 的組成步驟 ----


def _new_run_dir(output_root: Path, mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1000):
        name = f"{stamp}-{mode}" if attempt == 0 else f"{stamp}-{attempt}-{mode}"
        try:
            (output_root / name).mkdir()
            return output_root / name
        except FileExistsError:
            continue
    raise RuntimeError("cannot create a unique run directory")


def _dataset_record(dataset, train: np.ndarray, queries: np.ndarray) -> dict:
    record = {
        "name": dataset.name,
        "kind": "synthetic" if datasets.is_synthetic(dataset) else "npy",
        "n_train": int(train.shape[0]),
        "n_queries": int(queries.shape[0]),
        "dim": int(train.shape[1]),
        "metric": dataset.metric,
        "files_sha256": {name: environment.sha256_file(path) for name, path in datasets.files(dataset).items()},
    }
    if datasets.is_synthetic(dataset):
        record["generator"] = asdict(dataset)
    return record


def _metadata(formal_checks, snapshot: dict) -> dict:
    import faiss
    import rust_hnsw

    files = extension_files()
    head, head_time = environment.git_head(REPO)
    clock = time.get_clock_info("perf_counter")
    return {
        "machine": environment.machine_metadata(),
        "python": {"version": sys.version, "executable": sys.executable, "prefix": sys.prefix},
        "versions": {name: importlib.metadata.version(name) for name in VERSION_CHECKED},
        "faiss_compile_options": faiss.get_compile_options(),
        "build_info": dict(rust_hnsw.BUILD_INFO),
        "extension_files": files,
        "extension_sha256": {name: environment.sha256_file(path) for name, path in files.items()},
        "distance_kernels": DISTANCE_KERNELS,
        "hnswlib_compile_flags": HNSWLIB_COMPILE_FLAGS,
        "git": {"commit": head, "commit_time": head_time, "status": environment.git_status(REPO)},
        "clock": {
            "implementation": clock.implementation,
            "resolution_s": clock.resolution,
            "monotonic": clock.monotonic,
            "adjustable": clock.adjustable,
        },
        "child_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(CHILD_ENV_PREFIXES)},
        "formal_checks": formal_checks,
        "start": {**snapshot, "top_processes": environment.top_processes()},
        "end": None,
    }


def _provided_gt_agreement(dataset, loaded, n: int, n_train: int, gt: np.ndarray) -> dict:
    """資料集自帶的 ground truth 只在條件成立時比對（008 §4）；只作 sanity check。"""
    if loaded.provided_neighbors is None:
        return {"status": "not_applicable", "reason": "dataset has no provided ground truth"}
    if dataset.metric not in ("angular", "cosine"):
        return {"status": "not_applicable", "reason": f"dataset metric is {dataset.metric}"}
    if n != n_train:
        return {"status": "not_applicable", "reason": "n is a subset of train"}
    provided = np.asarray(loaded.provided_neighbors)
    if provided.shape[0] != gt.shape[0] or provided.shape[1] < protocol.K:
        return {"status": "not_applicable", "reason": f"provided ground truth has shape {provided.shape}"}
    return {"status": "computed", "recall_of_provided_at_10": evaluate.recall_at_k(provided, gt, protocol.K)}


def _prepare_subsets(config, dataset, loaded, train, queries, run_dir: Path) -> tuple[dict, dict]:
    """每個 n 的 ground truth 與資料集檢查；全部在第一個子 process 啟動前完成（008 §4）。"""
    import rust_hnsw

    (run_dir / "gt").mkdir()
    records, gts = {}, {}
    for n in sorted({build.n for build in config.builds}):
        subset = train[:n]
        exact = rust_hnsw.ExactIndex(train.shape[1])
        exact.add(subset)
        gt, _ = exact.search(queries, protocol.K)
        del exact
        gt_file = f"gt/n{n}.npy"
        np.save(run_dir / gt_file, gt)
        gts[n] = gt
        records[str(n)] = {
            "n": n,
            "gt_file": gt_file,
            "duplicate_row_fraction": evaluate.duplicate_row_fraction(subset),
            "rank1_tie_queries": evaluate.rank1_tie_queries(subset, queries, gt),
            "tie_eps": protocol.TIE_EPS,
            "provided_gt_agreement": _provided_gt_agreement(dataset, loaded, n, train.shape[0], gt),
        }
    return records, gts


def _read_jsonl(path: Path) -> list[dict]:
    """子 process 的紀錄；crash 時最後一行可能不完整，略過。"""
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return lines


def _exceeds(other_s, process_s) -> bool:
    return protocol.exceeds_single_thread_tolerance(other_s, process_s)


def _evaluate_point(line: dict, child_id: str, perm, gt, subset, queries, run_dir: Path) -> dict:
    """把子 process 的原始 ids 映射回子集列號，計算 recall 與計時摘要（008 §4、§6、§7）。"""
    nq = queries.shape[0]
    point_id = f"{child_id}-ef{line['ef_search']}"
    raw_path = Path(line["raw_file"])
    raw = dict(np.load(raw_path))
    arrays: dict[str, np.ndarray] = {}
    flags = []

    source = line["k10"]
    if source["status"] == "ok":
        ids10 = evaluate.map_ids(raw["ids_k10"], perm)
        metrics = evaluate.k_metrics(ids10, gt, subset, queries, protocol.K)
        cpu = source["other_thread_cpu"]
        k10 = {
            "status": "ok",
            "incomplete": metrics["incomplete_queries"] > 0,
            **metrics,
            "qps": evaluate.qps_summary(raw["qps_ns"], nq),
            "latency": evaluate.latency_summary(raw["latency_ns"]),
            "repeat_mismatch_queries": source["repeat_mismatch_queries"],
            "batch_vs_single_mismatch_queries": source["batch_vs_single_mismatch_queries"],
            "other_thread_cpu": cpu,
        }
        arrays.update(ids_k10=ids10, qps_ns=raw["qps_ns"], latency_ns=raw["latency_ns"])
        if _exceeds(cpu["qps_other_s"], cpu["qps_process_s"]) or _exceeds(
            cpu["latency_other_s"], cpu["latency_process_s"]
        ):
            flags.append("multithread")
        if k10["repeat_mismatch_queries"] or k10["batch_vs_single_mismatch_queries"]:
            flags.append("nondeterministic")
    else:
        # hnswlib 整批失敗：沒有 recall / QPS / latency，不足 k 的 query 數無法取得（007 §3）。
        k10 = {
            "status": "incomplete",
            "incomplete": True,
            "incomplete_queries": None,
            "recall": None,
            "tie_aware_recall": None,
            "qps": None,
            "latency": None,
            "repeat_mismatch_queries": None,
            "batch_vs_single_mismatch_queries": None,
            "other_thread_cpu": None,
        }

    if line["k1"]["status"] == "ok":
        ids1 = evaluate.map_ids(raw["ids_k1"], perm)
        metrics = evaluate.k_metrics(ids1, gt, subset, queries, protocol.RECALL_AT_1_K)
        k1 = {"status": "ok", "incomplete": metrics["incomplete_queries"] > 0, **metrics}
        arrays["ids_k1"] = ids1
    else:
        k1 = {"status": "incomplete", "incomplete": True, "incomplete_queries": None, "recall": None, "tie_aware_recall": None}

    differs = None
    if "ids_k10" in arrays and "ids_k1" in arrays:
        differs = int((arrays["ids_k1"][:, 0] != arrays["ids_k10"][:, 0]).sum())
    if line["loadavg_end"][0] > protocol.MAX_LOAD_1MIN:
        flags.append("load")

    point_file = f"points/{point_id}.npz"
    np.savez(run_dir / point_file, **arrays)
    raw_path.unlink()
    return {
        "point_id": point_id,
        "ef_search": line["ef_search"],
        "file": point_file,
        "start_time": line["start_time"],
        "end_time": line["end_time"],
        "loadavg_start": line["loadavg_start"],
        "loadavg_end": line["loadavg_end"],
        "k10": k10,
        "k1": k1,
        "k1_differs_from_k10_first_column": differs,
        "flags": flags,
    }


def _child_record(lines: list[dict], build_params: dict) -> tuple[dict, dict]:
    """由子 process 的 JSONL 組出 build 的量測欄位（008 §8）與 prelude。"""
    by_type = {line["type"]: line for line in lines if line["type"] != "point"}
    prelude, built, end = by_type.get("prelude"), by_type.get("build"), by_type.get("end")
    build = dict(build_params)
    if prelude is not None:
        build["baseline"] = prelude["baseline"]
    if built is not None:
        baseline = prelude["baseline"]
        build.update(
            build_ns=built["build_ns"],
            other_thread_cpu=built["other_thread_cpu"],
            peak_rss_bytes=built["peak_rss_bytes"],
            peak_rss_minus_baseline_bytes=built["peak_rss_bytes"] - baseline["ru_maxrss_bytes"],
            after_build=built["after_build"],
            index_memory_bytes=built["after_build"]["rss_bytes"] - baseline["rss_bytes"],
        )
    if end is not None:
        build.update(
            peak_rss_after_search_bytes=end["peak_rss_after_search_bytes"], serialized_bytes=end["serialized_bytes"]
        )
    return build, {"prelude": prelude, "built": built, "end": end}


def _run_child(*, mode, group, build, name, implementations, config, context, hooks, scratch, data_path, queries_path,
               perm, perm_sha, gt, subset, queries, run_dir, child_env) -> dict:
    child_id = f"g{group:03d}-{name}"
    if mode == "formal":
        hooks.formal_recheck(context)
        ok, samples = environment.wait_for_load(
            hooks.load_reader, protocol.MAX_LOAD_1MIN, protocol.LOAD_WAIT_S, protocol.LOAD_POLL_S,
            hooks.sleep, hooks.monotonic,
        )
        if not ok:
            raise RunAborted(f"load average stayed above {protocol.MAX_LOAD_1MIN} for {protocol.LOAD_WAIT_S} s")
        load_gate = {"enforced": True, "ok": True, "samples": samples}
    else:
        load_gate = {"enforced": False, "ok": None, "samples": [(0.0, hooks.load_reader())]}

    child_dir = run_dir / "children" / child_id
    raw_dir = child_dir / "raw"
    raw_dir.mkdir(parents=True)
    adapter, library = implementations[name]
    spec = {
        "child_id": child_id,
        "implementation": name,
        "adapter": adapter,
        "library": library,
        "params": {"m": build.m, "ef_construction": build.ef_construction, "seed": build.seed},
        "data_path": str(data_path),
        "queries_path": str(queries_path),
        "ef_search": list(config.ef_search),
        "jsonl_path": str(child_dir / "points.jsonl"),
        "raw_dir": str(raw_dir),
        "samples_path": str(child_dir / "samples.npz"),
        "scratch_dir": str(scratch),
    }
    spec_path = child_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    argv = hooks.child_command(spec_path)

    before = hooks.snapshot()
    started = time.time()
    returncode = None
    with open(child_dir / "log.txt", "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                argv, cwd=REPO, env=child_env, stdout=log, stderr=subprocess.STDOUT, timeout=config.child_timeout_s
            )
            returncode = proc.returncode
            status = "ok" if returncode == 0 else ("signal" if returncode < 0 else "exit_nonzero")
        except subprocess.TimeoutExpired:
            status = "timeout"
    ended = time.time()
    after = hooks.snapshot()

    lines = _read_jsonl(child_dir / "points.jsonl")
    build_params = {**asdict(build), "permutation_sha256": perm_sha}
    build_record, parts = _child_record(lines, build_params)
    if status == "ok" and parts["end"] is None:
        status = "missing_output"

    points = [
        _evaluate_point(line, child_id, perm, gt, subset, queries, run_dir)
        for line in lines
        if line["type"] == "point"
    ]
    shutil.rmtree(raw_dir, ignore_errors=True)

    flags = []
    if before["therm_warnings"] or after["therm_warnings"]:
        flags.append("thermal")
    if after["swap_used_bytes"] > before["swap_used_bytes"]:
        flags.append("swap")
    cpu = build_record.get("other_thread_cpu")
    if cpu is not None and _exceeds(cpu["other_s"], cpu["process_s"]):
        flags.append("multithread")

    prelude = parts["prelude"] or {}
    samples_file = child_dir / "samples.npz"
    return {
        "child_id": child_id,
        "group": group,
        "implementation": name,
        "build": build_record,
        "status": status,
        "returncode": returncode,
        "partial": status != "ok" and bool(points),
        "started_at": started,
        "ended_at": ended,
        "load_gate": load_gate,
        "before": before,
        "after": after,
        "flags": flags,
        "loaded_libraries": prelude.get("loaded_libraries"),
        "python_loop": prelude.get("python_loop"),
        "null_call": prelude.get("null_call"),
        "samples_file": str(samples_file.relative_to(run_dir)) if samples_file.exists() else None,
        "log": str((child_dir / "log.txt").relative_to(run_dir)),
        "points": points,
    }


def _file_digests(run_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(run_dir)): environment.sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "run.json"
    }


def _raise_aborted(signum, frame):
    raise RunAborted(f"received signal {signal.Signals(signum).name}")


def run(config_text: str, mode: str, output_root, *, implementations=IMPLEMENTATIONS, registry=None,
        hooks: Hooks | None = None) -> RunResult:
    """執行一次 run。config 或 formal 前置檢查不符時 raise（ConfigError / PreflightError），不寫任何輸出。"""
    if mode not in ("smoke", "formal"):
        raise ValueError(f"mode must be 'smoke' or 'formal', got {mode!r}")
    hooks = hooks or Hooks()
    registry = datasets.REGISTRY if registry is None else registry
    config = parse_config(config_text, implementations, registry)
    dataset = registry[config.dataset]

    context = {"dataset": dataset}
    formal_checks = hooks.formal_preflight(context) if mode == "formal" else None

    loaded = datasets.load(dataset)
    train = prepare_vectors(loaded.train)
    queries = prepare_vectors(loaded.queries)
    check_dataset_fit(config, train.shape[0], queries.shape[0])

    run_dir = _new_run_dir(Path(output_root), mode)
    (run_dir / "points").mkdir()
    order = execution_order(config)
    record = {
        "status": "running",
        "status_reasons": [],
        "mode": mode,
        "protocol": protocol.protocol_record(),
        "config": {
            "text": config.text,
            "sha256": hashlib.sha256(config.text.encode("utf-8")).hexdigest(),
            "parsed": config.raw,
        },
        "dataset": _dataset_record(dataset, train, queries),
        "metadata": _metadata(formal_checks, hooks.snapshot()),
        "subsets": {},
        "execution_order": [
            {"group": group, **asdict(build), "implementations": list(names)} for group, build, names in order
        ],
        "children": [],
        "files": {},
    }

    scratch = Path(tempfile.mkdtemp(prefix="rust-hnsw-run-"))
    child_env = dict(os.environ)
    in_main_thread = threading.current_thread() is threading.main_thread()
    previous_handler = signal.signal(signal.SIGTERM, _raise_aborted) if in_main_thread else None
    error = None
    try:
        record["subsets"], gts = _prepare_subsets(config, dataset, loaded, train, queries, run_dir)
        queries_path = scratch / "queries.npy"
        np.save(queries_path, queries)
        for group, build, names in order:
            subset = train[: build.n]
            if build.shuffle is None:
                perm, perm_sha, data = None, None, subset
            else:
                perm = np.random.default_rng(build.shuffle).permutation(build.n)
                perm_sha = hashlib.sha256(perm.astype("<i8").tobytes()).hexdigest()
                data = np.ascontiguousarray(subset[perm])
            data_path = scratch / f"g{group:03d}.npy"
            np.save(data_path, data)
            del data
            try:
                for name in names:
                    child = _run_child(
                        mode=mode, group=group, build=build, name=name, implementations=implementations,
                        config=config, context=context, hooks=hooks, scratch=scratch, data_path=data_path,
                        queries_path=queries_path, perm=perm, perm_sha=perm_sha, gt=gts[build.n], subset=subset,
                        queries=queries, run_dir=run_dir, child_env=child_env,
                    )
                    record["children"].append(child)
                    if child["status"] != "ok":
                        record["status_reasons"].append(f"{child['child_id']}: {child['status']}")
            finally:
                data_path.unlink(missing_ok=True)
        record["status"] = "incomplete" if record["status_reasons"] else "complete"
    except PreflightError as err:
        record["status"] = "invalid"
        record["status_reasons"].append(f"formal recheck failed: {err}")
    except (RunAborted, KeyboardInterrupt) as err:
        record["status"] = "aborted"
        record["status_reasons"].append(f"aborted: {err or type(err).__name__}")
    except BaseException as err:
        record["status"] = "error"
        record["status_reasons"].append(f"error: {err!r}")
        error = err
    finally:
        if in_main_thread:
            signal.signal(signal.SIGTERM, previous_handler)
        shutil.rmtree(scratch, ignore_errors=True)
        record["metadata"]["end"] = {**hooks.snapshot(), "top_processes": environment.top_processes()}
        record["files"] = _file_digests(run_dir)
        text = json.dumps(record, indent=1, ensure_ascii=False)
        (run_dir / "run.json").write_text(text, encoding="utf-8")
    if error is not None:
        raise error
    return RunResult(exit_code=0 if record["status"] == "complete" else 1, run_dir=run_dir, record=json.loads(text))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="TOML config（decision 008 §3）")
    parser.add_argument("--mode", required=True, choices=["smoke", "formal"])
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args(argv)
    text = Path(args.config).read_text(encoding="utf-8")
    try:
        result = run(text, args.mode, Path(args.output_root))
    except ConfigError as err:
        print(f"config error: {err}", file=sys.stderr)
        return 2
    except PreflightError as err:
        print(f"formal preflight failed: {err}", file=sys.stderr)
        return 3
    print(f"{result.record['status']}: {result.run_dir}")
    for reason in result.record["status_reasons"]:
        print(f"  {reason}")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
