"""Gate 證據的 profiling 計畫（decision 015 §2）：依正式 run 選各實作的 ef，產生在 heavy lock 下依序取樣的指令。

用法（計畫寫在 results/ 下，不讓 repo 變 dirty）：
    .venv/bin/python -m benchmarks.gate_profile plan <該資料集全部 formal run 目錄…> > results/profiles/gate-<資料集>-plan.sh
    ./tools/with-mac-lock bash results/profiles/gate-<資料集>-plan.sh
    .venv/bin/python -m benchmarks.gate_profile check <同一組 run 目錄…>

- 格點：n = 100,000、(M, efC) = (16, 200)、seed 1、原始插入順序（兩個資料集的完整 variance 格點，013 §1）。
- ef：該 build 在正式 run 中 Recall@10 ≥ 0.95 的最小 ef（各實作各自的，007 §4）。任一實作在 ef 清單內沒有達到 0.95 時，
  三個實作**一起**改用 0.90，讓取樣仍在相近的 recall（009）。recall 不受 hygiene 標記影響（標記只關係到計時），所以被標記的 build 也用來選 ef。
- 每個實作取樣 search（30 秒）與 build（300 秒）兩個階段（009），一律經由 `scripts/profile.sh`。
  計畫一開始就拒絕覆寫已存在的同名輸出：上一次嘗試的檔案要先移開，不得混進新的證據。
- `check`：每個（實作、階段）的取樣前紀錄、`sample` 原始報告與排行都存在，而且取樣晚於紀錄；
  紀錄的參數與計畫完全相同、由祖先持有 heavy lock、通過 formal 的電力與背景條件（014 補記）、git 乾淨、
  與正式 run 之間被取樣的程式（`PROFILED_PATHS`：實作、binding、adapter、資料載入與 recall 計算）沒有差異、
  `rust_hnsw.BUILD_INFO` 與套件版本相同；
  search 階段的 Recall@10 與正式 run 同一個 build 的選定點完全相同（同一個 `metrics.recall_at_k`，三個實作的 build 在相同 seed 下都是確定的）。
  不符時照實列出，該取樣不得作為 Gate 證據，直到查明原因。
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROFILES = REPO / "results" / "profiles"
N, M, EF_CONSTRUCTION, SEED = 100_000, 16, 200, 1
TARGET, FALLBACK_TARGET = 0.95, 0.90
K = 10
PHASES = ("search", "build")
SECONDS = {"search": 30, "build": 300}
IMPLEMENTATIONS = ("rust-hnsw", "hnswlib", "faiss-hnsw")
SHORT_NAMES = {"glove-100-angular": "glove", "simplewiki-embeddinggemma2-100k": "text"}
OUTPUT_SUFFIXES = (".json", ".log", ".sample.txt", ".top.txt")
COMPARED_VERSIONS = ("hnswlib", "faiss-cpu", "numpy")
# 015 補記（2026-10-15）：與正式 run 比較的是被取樣的程式路徑。runner 的有效性規則（run.py、protocol.py、environment.py）
# 在 017、018 之後已改過，但它們不在 profiling 的執行路徑上，不影響「時間花在哪」。
PROFILED_PATHS = ("crates", "python", "Cargo.toml", "Cargo.lock", "requirements-dev.txt", "benchmarks/adapters.py",
                  "benchmarks/interface.py", "benchmarks/datasets.py", "benchmarks/metrics.py")


class PlanError(ValueError):
    pass


def _cell_points(records: list[dict], impl: str) -> list[dict]:
    """正式 run 中 (N, M, efC, seed 1, 原始順序) 的那一個 build 的點（依 ef 排序）。"""
    children = [c for r in records for c in r["children"] if c["implementation"] == impl
                and (c["build"]["n"], c["build"]["m"], c["build"]["ef_construction"], c["build"]["seed"],
                     c["build"]["shuffle"]) == (N, M, EF_CONSTRUCTION, SEED, None)]
    if len(children) != 1:
        raise PlanError(f"{impl}: expected exactly one build at n={N}, M={M}, efC={EF_CONSTRUCTION}, seed {SEED}, "
                        f"original order; found {len(children)}")
    points = [p for p in children[0]["points"] if p["k10"]["status"] == "ok" and p["k10"]["recall"] is not None]
    return sorted(points, key=lambda p: p["ef_search"])


def _check_records(records: list[dict]) -> str:
    if not records:
        raise PlanError("no runs given")
    if {r["mode"] for r in records} != {"formal"}:
        raise PlanError("Gate profiling selects ef from formal runs only (009)")
    datasets = {r["dataset"]["name"] for r in records}
    if len(datasets) != 1:
        raise PlanError(f"runs mix datasets: {sorted(datasets)}")
    return datasets.pop()


def select_ef(records: list[dict]) -> dict:
    """每個實作：Recall@10 ≥ 目標的最小 ef。目標是 TARGET；任一實作在 ef 清單內沒有達到時，三者一起改用 FALLBACK_TARGET。"""
    _check_records(records)
    points = {impl: _cell_points(records, impl) for impl in IMPLEMENTATIONS}
    for target in (TARGET, FALLBACK_TARGET):
        reaching = {impl: [p for p in pts if p["k10"]["recall"] >= target] for impl, pts in points.items()}
        if all(reaching.values()):
            return {impl: {"ef": pts[0]["ef_search"], "recall10": pts[0]["k10"]["recall"], "target": target}
                    for impl, pts in reaching.items()}
    missing = [impl for impl, pts in points.items() if not any(p["k10"]["recall"] >= FALLBACK_TARGET for p in pts)]
    raise PlanError(f"{missing} reach neither {TARGET} nor {FALLBACK_TARGET} in the formal ef list")


def profile_name(dataset: str, impl: str, phase: str) -> str:
    return f"gate-{SHORT_NAMES.get(dataset, dataset)}-{impl}-{phase}"


def expected_args(dataset: str, impl: str, phase: str, selected: dict) -> dict:
    """profile.py 的參數；取樣前紀錄的 `args` 必須與此相同（`seconds`、`meta` 等其他欄位不比）。"""
    args = {"impl": impl, "phase": phase, "dataset": dataset, "n": N, "m": M, "ef_construction": EF_CONSTRUCTION,
            "seed": SEED, "ef": None, "k": None}
    if phase == "search":
        args.update(ef=selected[impl]["ef"], k=K)
    return args


def plan_commands(dataset: str, selected: dict) -> list[list[str]]:
    """search 先、build 後；同一階段依 IMPLEMENTATIONS 的順序。"""
    commands = []
    for phase in PHASES:
        for impl in IMPLEMENTATIONS:
            args = expected_args(dataset, impl, phase, selected)
            command = ["./scripts/profile.sh", profile_name(dataset, impl, phase), str(SECONDS[phase]), "--"]
            for key in ("impl", "phase", "dataset", "n", "m", "ef_construction", "seed", "ef", "k"):
                if args[key] is not None:
                    command += [f"--{key.replace('_', '-')}", str(args[key])]
            commands.append(command)
    return commands


def plan_script(records: list[dict]) -> str:
    dataset = _check_records(records)
    selected = select_ef(records)
    lines = ["#!/usr/bin/env bash", "# decision 015 §2：在 ./tools/with-mac-lock 下執行。", "set -euo pipefail",
             f"cd {shlex.quote(str(REPO))}"]
    for impl, choice in selected.items():
        note = "" if choice["target"] == TARGET else f"（有實作在 ef 清單內未達 {TARGET}，三者一起改用 {FALLBACK_TARGET}）"
        lines.append(f"# {impl}: ef {choice['ef']}，正式 run 的 Recall@10 = {choice['recall10']:.4f}{note}")
    outputs = [f"results/profiles/{profile_name(dataset, impl, phase)}{suffix}"
               for phase in PHASES for impl in IMPLEMENTATIONS for suffix in OUTPUT_SUFFIXES]
    lines.append("# 拒絕覆寫上一次嘗試的輸出（先移開，不得混進新的證據）。")
    lines.append(f"for f in {' '.join(shlex.quote(o) for o in outputs)}; do "
                 'if [ -e "$f" ]; then echo "exists: $f" >&2; exit 1; fi; done')
    lines += [shlex.join(command) for command in plan_commands(dataset, selected)]
    return "\n".join(lines) + "\n"


def changed_profiled_files(commit_a: str, commit_b: str, repo=REPO) -> list[str]:
    """兩個 commit 之間，被取樣的程式（PROFILED_PATHS）有差異的檔案。"""
    try:
        out = subprocess.run(["git", "diff", "--name-only", commit_a, commit_b, "--", *PROFILED_PATHS],
                             cwd=repo, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        raise PlanError(f"cannot compare commits {commit_a} and {commit_b}: {err}") from err
    return out.split()


def _check_one(records, dataset, impl, phase, selected, profiles, changed_files) -> dict:
    stem = Path(profiles) / profile_name(dataset, impl, phase)
    result = {"implementation": impl, "phase": phase, "problems": []}
    problems = result["problems"]
    meta_path = stem.with_name(stem.name + ".json")
    if not meta_path.exists():
        problems.append(f"{meta_path} not found")
        result["ok"] = False
        return result
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for suffix in (".sample.txt", ".top.txt"):
        path = stem.with_name(stem.name + suffix)
        if not path.exists():
            problems.append(f"{path} not found")
        elif path.stat().st_mtime < meta_path.stat().st_mtime:
            problems.append(f"{path} is older than {meta_path.name} (left over from an earlier attempt)")
        elif not path.read_text(encoding="utf-8", errors="replace").strip():
            problems.append(f"{path} is empty")  # 009 補記：排行是空的就不是證據
    args = meta.get("args", {})
    for key, value in expected_args(dataset, impl, phase, selected).items():
        if args.get(key) != value:
            problems.append(f"args.{key} = {args.get(key)!r}, expected {value!r}")
    if meta.get("lock", {}).get("held_by_ancestor") is not True:
        problems.append("the heavy lock was not held by an ancestor")
    if meta.get("evidence_conditions") is None:
        problems.append("the formal power / background conditions were not checked")
    git = meta.get("git", {})
    if git.get("status") != []:
        problems.append(f"git status was not clean: {git.get('status')}")
    if git.get("commit") is None:
        problems.append("no git commit recorded")
    else:
        for commit in sorted({r["metadata"]["git"]["commit"] for r in records}):
            changed = changed_files(commit, git["commit"])
            if changed:
                problems.append(f"profiled code differs from formal commit {commit[:12]}: {changed}")
    if any(meta.get("build_info") != r["metadata"]["build_info"] for r in records):
        problems.append("rust_hnsw.BUILD_INFO differs from the formal run")
    versions = meta.get("versions", {})
    differing = sorted({name for r in records for name in COMPARED_VERSIONS
                        if versions.get(name) != r["metadata"]["versions"].get(name)})
    if differing:
        problems.append(f"package versions differ from the formal run: {differing}")
    if phase == "search":
        recall = meta.get("recall_at_10")
        result.update(ef=selected[impl]["ef"], formal_recall10=selected[impl]["recall10"], profile_recall10=recall)
        if recall != selected[impl]["recall10"]:
            problems.append(f"Recall@10 {recall!r} differs from the formal run's {selected[impl]['recall10']!r}")
    result["ok"] = not problems
    return result


def check(records: list[dict], profiles=None, changed_files=None) -> list[dict]:
    """每個（實作、階段）的取樣是否符合 015 §2 的證據條件；`changed_files(a, b)` 預設以 git 比較（測試可替換）。"""
    profiles = PROFILES if profiles is None else profiles
    changed_files = changed_profiled_files if changed_files is None else changed_files
    dataset = _check_records(records)
    selected = select_ef(records)
    return [_check_one(records, dataset, impl, phase, selected, profiles, changed_files)
            for phase in PHASES for impl in IMPLEMENTATIONS]


def _load(run_dir) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "check"])
    parser.add_argument("runs", nargs="+", help="同一個資料集的 formal run 目錄")
    args = parser.parse_args(argv)
    records = [_load(d) for d in args.runs]
    if args.command == "plan":
        sys.stdout.write(plan_script(records))
        return 0
    results = check(records)
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
