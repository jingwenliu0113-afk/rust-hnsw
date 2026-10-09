"""Gate 證據的 profiling 計畫（decision 015 §2）：依正式 run 選各實作的 ef，產生在 heavy lock 下依序取樣的指令。

用法：
    .venv/bin/python -m benchmarks.gate_profile plan <該資料集全部 formal run 目錄…> > plan.sh
    ./tools/with-mac-lock bash plan.sh
    .venv/bin/python -m benchmarks.gate_profile check <同一組 run 目錄…>

- 格點：n = 100,000、(M, efC) = (16, 200)、seed 1、原始插入順序（兩個資料集的完整 variance 格點，013 §1）。
- ef：該 build 在正式 run 中 Recall@10 ≥ 0.95 的最小 ef（各實作各自的，007 §4）；ef 清單內都沒有達到時改用 0.90，並在計畫中註明。
  recall 不受 hygiene 標記影響（標記只關係到計時），所以被標記的 build 也用來選 ef。
- 每個實作取樣 search（30 秒）與 build（300 秒）兩個階段（009），一律經由 `scripts/profile.sh`。
- `check`：取樣前寫出的 JSON 中，search 階段的 ef 與 Recall@10 必須與正式 run 同一個 build 的選定點相同
  （三個實作的 build 在相同 seed 下都是確定的），而且由祖先持有 heavy lock、通過 formal 的電力與背景條件（014 補記）；
  不符時照實列出，該取樣不得作為 Gate 證據，直到查明原因。
"""

import argparse
import json
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROFILES = REPO / "results" / "profiles"
N, M, EF_CONSTRUCTION, SEED = 100_000, 16, 200, 1
TARGET, FALLBACK_TARGET = 0.95, 0.90
K = 10
SECONDS = {"search": 30, "build": 300}
IMPLEMENTATIONS = ("rust-hnsw", "hnswlib", "faiss-hnsw")
SHORT_NAMES = {"glove-100-angular": "glove", "simplewiki-embeddinggemma2-100k": "text"}


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


def select_ef(records: list[dict]) -> dict:
    """每個實作：Recall@10 ≥ TARGET 的最小 ef；ef 清單內都沒有達到時，改用 FALLBACK_TARGET。"""
    if {r["mode"] for r in records} != {"formal"}:
        raise PlanError("Gate profiling selects ef from formal runs only (009)")
    datasets = {r["dataset"]["name"] for r in records}
    if len(datasets) != 1:
        raise PlanError(f"runs mix datasets: {sorted(datasets)}")
    selected = {}
    for impl in IMPLEMENTATIONS:
        points = _cell_points(records, impl)
        for target in (TARGET, FALLBACK_TARGET):
            reaching = [p for p in points if p["k10"]["recall"] >= target]
            if reaching:
                selected[impl] = {"ef": reaching[0]["ef_search"], "recall10": reaching[0]["k10"]["recall"],
                                  "target": target}
                break
        else:
            raise PlanError(f"{impl} reaches neither {TARGET} nor {FALLBACK_TARGET} in the formal ef list")
    return selected


def profile_name(dataset: str, impl: str, phase: str) -> str:
    return f"gate-{SHORT_NAMES.get(dataset, dataset)}-{impl}-{phase}"


def plan_commands(dataset: str, selected: dict) -> list[list[str]]:
    """search 先、build 後；同一階段依 IMPLEMENTATIONS 的順序。"""
    commands = []
    for phase in ("search", "build"):
        for impl in IMPLEMENTATIONS:
            args = ["./scripts/profile.sh", profile_name(dataset, impl, phase), str(SECONDS[phase]), "--",
                    "--impl", impl, "--phase", phase, "--dataset", dataset, "--n", str(N), "--m", str(M),
                    "--ef-construction", str(EF_CONSTRUCTION), "--seed", str(SEED)]
            if phase == "search":
                args += ["--ef", str(selected[impl]["ef"]), "--k", str(K)]
            commands.append(args)
    return commands


def plan_script(records: list[dict]) -> str:
    dataset = records[0]["dataset"]["name"]
    selected = select_ef(records)
    lines = ["#!/usr/bin/env bash", "# decision 015 §2：在 ./tools/with-mac-lock 下執行。", "set -euo pipefail",
             f'cd {shlex.quote(str(REPO))}']
    for impl, choice in selected.items():
        note = "" if choice["target"] == TARGET else f"（ef 清單內未達 {TARGET}，改用 {FALLBACK_TARGET}）"
        lines.append(f"# {impl}: ef {choice['ef']}，正式 run 的 Recall@10 = {choice['recall10']:.4f}{note}")
    lines += [shlex.join(command) for command in plan_commands(dataset, selected)]
    return "\n".join(lines) + "\n"


def check(records: list[dict], profiles=None) -> list[dict]:
    """search 取樣的 Recall@10 與正式 run 相同才可作為證據。"""
    profiles = PROFILES if profiles is None else profiles
    dataset = records[0]["dataset"]["name"]
    selected = select_ef(records)
    results = []
    for impl, choice in selected.items():
        path = Path(profiles) / f"{profile_name(dataset, impl, 'search')}.json"
        if not path.exists():
            results.append({"implementation": impl, "ok": False, "reason": f"{path} not found"})
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        recall = meta.get("recall_at_10")
        # 同一個 metrics.recall_at_k、同一個確定的 build：數值必須完全相同。
        ok = (recall == choice["recall10"] and meta.get("args", {}).get("ef") == choice["ef"]
              and meta.get("lock", {}).get("held_by_ancestor") is True and meta.get("evidence_conditions") is not None)
        results.append({"implementation": impl, "ok": ok, "ef": choice["ef"], "formal_recall10": choice["recall10"],
                        "profile_recall10": recall, "profile_ef": meta.get("args", {}).get("ef"), "lock": meta.get("lock"),
                        "evidence_conditions": meta.get("evidence_conditions") is not None})
    return results


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
