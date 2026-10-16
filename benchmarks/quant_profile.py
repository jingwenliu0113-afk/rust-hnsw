"""Advanced C 的 profiling（decision 019 §8）：取樣計畫、取樣後的證據核對、self-time 的分類規則與 H1 前提的反組譯檢查。

用法（計畫與輸出寫在 results/ 下，不讓 repo 變 dirty）：
    .venv/bin/python -m benchmarks.quant_profile plan --dataset <glove|text> <10 份 C formal config 的 run 目錄…> \\
        [--excluded <已整份重跑的失敗 run>…] > results/profiles/quant-<glove|text>-plan.sh
    .venv/bin/python -m benchmarks.machine_quiet --powermode 2      # 印出 yes 才開始（019 §5）
    ./tools/with-mac-lock bash results/profiles/quant-<glove|text>-plan.sh
    .venv/bin/python -m benchmarks.quant_profile check --dataset <glove|text> <同一組 run 目錄…> [--excluded …]
    .venv/bin/python -m benchmarks.quant_profile premise [--out results/quant/premise.json]   # 019 §9 第 4 步，formal 之前

- plan 與 check 在全部 10 份 C formal config 完成之後才執行（019 §9 第 8 步：診斷之後）：兩個資料集的 run 都要列出，
  先通過與 C 報告相同的合併前檢查（`quant_report.load_formal_set`），再對 `--dataset` 指定的資料集產生計畫或核對。

- 取樣（每個資料集：seed 1、(100k, 16, 200)、原始順序；只取樣 search 階段，每次 30 秒）：
  R、F、S 在 e\\*，以及 S 在它自己的同 recall ef（C 的 seed 1 中 S 第一次 Recall@10 ≥ 0.95 的 ef；沒有達到改用 0.90；
  與 e\\* 相同或都沒有達到就略過）。不取樣 build 與 FAISS（019 §8）。
- `check`：重用 `gate_profile.check_evidence`（輸出檔、參數、heavy lock、formal 的電力與背景條件、git 乾淨、
  `PROFILED_PATHS` 與 C formal run 沒有差異、BUILD_INFO 與套件版本、取樣時的 Recall@10 與 formal 的同一點完全相同）；
  另外確認取樣在高效能模式（016）、用的是 C formal run 的同一個變壓器（018、019 §5），以及排行可以解析。
- 分類規則（`classify`）在取樣之前固定，依序取第一個符合的類別；看過取樣之後的重新歸類只能當註記。
- `premise`：以 `otool -tV` 反組譯 binding，檢查兩個 kernel 的加總是依序的純量相依鏈（019 §2 的機械判準），
  並記錄 S 的 kernel 中出現的轉換指令。不符時 H1 記為「前提不成立」；不為了符合前提而改寫 kernel。
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

from benchmarks import dataprep, environment, gate_profile, profile_report, quant_report
from benchmarks.gate_profile import PlanError
from benchmarks.quant_report import DIAG_DIR, E_STAR, EF_CONSTRUCTION, F, M, N, POWERMODE, R, S, SHORT_NAMES

REPO = Path(__file__).resolve().parents[1]
PROFILES = REPO / "results" / "profiles"
PREMISE_OUT = DIAG_DIR / "premise.json"
SEED, K, SECONDS = 1, 10, 30
PROFILED = (R, F, S)
SAME_RECALL_TARGETS = (0.95, 0.90)
OUTPUT_SUFFIXES = gate_profile.OUTPUT_SUFFIXES

# ---- 分類規則（019 §8，取樣之前固定） ----

CATEGORIES = ("kernel", "visited_set", "heap", "traversal", "allocation", "other")
OTHER_REPORT_SHARE = 0.05
OTHER_TOP = 10


def classify(row: profile_report.Row) -> str:
    """依 019 §8 的順序取第一個符合的類別。「符號」是 `Row.name`（pretty_symbol 的輸出），「映像檔」是 `Row.image`。

    沒有符號的列（`??? +0x…`）同樣依這些規則分類：除非映像檔符合第 5 條，否則歸入「其他」，不得略過。
    """
    name = row.name
    if "rhnsw_kernel_" in name or "rhnsw_kernel_" in row.symbol:
        return "kernel"
    if name.startswith("hashbrown::") or "core::hash::sip" in name:
        return "visited_set"
    if name.startswith("alloc::collections::binary_heap::"):
        return "heap"
    if any(part in name for part in ("search_layer", "greedy_search", "search_unit")):
        return "traversal"
    if row.image == "libsystem_malloc.dylib" or name.startswith("_platform_mem"):
        return "allocation"
    return "other"


def read_rows(sample_path) -> list[profile_report.Row]:
    """`sample` 原始報告的 top-of-stack 全部列（排行檔只有前 20 名，分母必須是全部樣本）。"""
    return profile_report.top_of_stack(Path(sample_path).read_text(encoding="utf-8", errors="replace"))


def categorize(rows: list[profile_report.Row]) -> dict:
    """各類別的樣本數；「其他」超過 5% 時列出其中的前 10 名（019 §8）。占比 = 樣本數 / total。"""
    counts = Counter()
    others = []
    for row in rows:
        category = classify(row)
        counts[category] += row.count
        if category == "other":
            others.append(row)
    total = sum(counts.values())
    top = []
    if total and counts["other"] / total > OTHER_REPORT_SHARE:
        top = [{"name": r.name, "symbol": r.symbol, "image": r.image, "count": r.count, "share": r.count / total}
               for r in sorted(others, key=lambda r: -r.count)[:OTHER_TOP]]
    return {"total": total, "counts": {category: counts[category] for category in CATEGORIES}, "other_top": top}


# ---- 計畫 ----


def profile_name(dataset: str, impl: str, ef: int) -> str:
    """`results/profiles/quant-<glove|text>-<實作>-ef<ef>`（019 §8）。"""
    return f"quant-{SHORT_NAMES.get(dataset, dataset)}-{impl}-ef{ef}"


def select_samples(records: list[dict]) -> tuple[list[dict], list[str]]:
    """019 §8 的取樣清單（依序）與略過的說明。recall 取自 C formal run 中 seed 1 的那一個 build（不受 hygiene 標記影響）。"""
    dataset = gate_profile._check_records(records)
    if dataset not in E_STAR:
        raise PlanError(f"{dataset!r} is not a dataset of Advanced C (019 §5)")
    e_star = E_STAR[dataset]
    samples, notes = [], []
    for impl in PROFILED:
        point = next((p for p in gate_profile._cell_points(records, impl) if p["ef_search"] == e_star), None)
        if point is None:
            raise PlanError(f"{impl}: the seed 1 build has no complete point at e* = {e_star}")
        samples.append({"implementation": impl, "ef": e_star, "recall10": point["k10"]["recall"], "role": "e_star"})
    points = gate_profile._cell_points(records, S)
    for target in SAME_RECALL_TARGETS:
        reaching = [p for p in points if p["k10"]["recall"] >= target]
        if reaching:
            chosen = reaching[0]
            if chosen["ef_search"] == e_star:
                notes.append(f"{S} 的同 recall ef（{target}）就是 e* = {e_star}，略過")
            else:
                samples.append({"implementation": S, "ef": chosen["ef_search"], "recall10": chosen["k10"]["recall"],
                                "role": f"same_recall_{target:.2f}"})
            break
    else:
        notes.append(f"{S} 在 ef 清單內沒有達到 {SAME_RECALL_TARGETS[-1]}，略過同 recall 的取樣")
    return samples, notes


def expected_args(dataset: str, sample: dict) -> dict:
    """profile.py 的參數；取樣前紀錄的 `args` 必須與此相同。"""
    return {"impl": sample["implementation"], "phase": "search", "dataset": dataset, "n": N, "m": M,
            "ef_construction": EF_CONSTRUCTION, "seed": SEED, "ef": sample["ef"], "k": K}


def plan_commands(dataset: str, samples: list[dict]) -> list[list[str]]:
    commands = []
    for sample in samples:
        args = expected_args(dataset, sample)
        command = ["./scripts/profile.sh", profile_name(dataset, sample["implementation"], sample["ef"]), str(SECONDS), "--"]
        for key in ("impl", "phase", "dataset", "n", "m", "ef_construction", "seed", "ef", "k"):
            command += [f"--{key.replace('_', '-')}", str(args[key])]
        commands.append(command)
    return commands


def plan_script(records: list[dict]) -> str:
    dataset = gate_profile._check_records(records)
    samples, notes = select_samples(records)
    lines = ["#!/usr/bin/env bash",
             "# decision 019 §8：先以 benchmarks.machine_quiet --powermode 2 確認（019 §5），再在 ./tools/with-mac-lock 下執行。",
             "set -euo pipefail", f"cd {shlex.quote(str(REPO))}"]
    for sample in samples:
        lines.append(f"# {sample['implementation']}（{sample['role']}）：ef {sample['ef']}，C formal run 的 seed 1 Recall@10 = "
                     f"{sample['recall10']:.4f}")
    lines += [f"# {note}" for note in notes]
    outputs = [f"results/profiles/{profile_name(dataset, s['implementation'], s['ef'])}{suffix}"
               for s in samples for suffix in OUTPUT_SUFFIXES]
    lines.append("# 拒絕覆寫上一次嘗試的輸出（先移開，不得混進新的證據）。")
    lines.append(f"for f in {' '.join(shlex.quote(o) for o in outputs)}; do "
                 'if [ -e "$f" ]; then echo "exists: $f" >&2; exit 1; fi; done')
    lines += [shlex.join(command) for command in plan_commands(dataset, samples)]
    return "\n".join(lines) + "\n"


# ---- 取樣後的核對 ----


def _formal_adapters(records: list[dict]) -> list:
    adapters = []
    for record in records:
        adapter = (((record["metadata"].get("formal_checks") or {}).get("power") or {}).get("adapter"))
        if adapter not in adapters:
            adapters.append(adapter)
    return adapters


def check(records: list[dict], profiles=None, changed_files=None) -> list[dict]:
    """每個取樣是否符合 015 §2 與 019 §8 的證據條件；`changed_files(a, b)` 預設以 git 比較 PROFILED_PATHS（測試可替換）。"""
    profiles = PROFILES if profiles is None else Path(profiles)
    changed_files = gate_profile.changed_profiled_files if changed_files is None else changed_files
    dataset = gate_profile._check_records(records)
    samples, _ = select_samples(records)
    adapters = _formal_adapters(records)
    results = []
    for sample in samples:
        stem = Path(profiles) / profile_name(dataset, sample["implementation"], sample["ef"])
        result = {"implementation": sample["implementation"], "ef": sample["ef"], "role": sample["role"], "problems": []}
        problems = result["problems"]
        meta = gate_profile.check_evidence(records, stem, expected_args(dataset, sample), sample["recall10"], changed_files,
                                           problems)
        if meta is not None:
            result.update(formal_recall10=sample["recall10"], profile_recall10=meta.get("recall_at_10"))
            power = (meta.get("evidence_conditions") or {}).get("power") or {}
            if meta.get("evidence_conditions") is not None and power.get("powermode") != POWERMODE:
                problems.append(f"power mode {power.get('powermode')!r}; C profiling uses high-performance mode {POWERMODE} "
                                "(decision 016, 019 §5)")
            if meta.get("evidence_conditions") is not None and (len(adapters) != 1 or power.get("adapter") != adapters[0]):
                problems.append(f"power adapter {power.get('adapter')!r} differs from the C formal runs' {adapters} (018, 019 §5)")
            sample_path = stem.with_name(stem.name + ".sample.txt")
            if sample_path.exists():
                try:
                    if not read_rows(sample_path):
                        problems.append(f"{sample_path} has an empty top-of-stack section")
                except ValueError as err:
                    problems.append(f"{sample_path}: {err}")
        result["ok"] = not problems
        results.append(result)
    return results


# ---- H1 前提：反組譯（019 §2） ----

KERNELS = ("rhnsw_kernel_fp32", "rhnsw_kernel_sq8")
# b + Σ（019 §3 的距離公式）是 SQ8 kernel 唯一允許的、不屬於累加的 fadd：它以累加暫存器為來源，寫入另一個暫存器。
FINAL_ADDS = {"rhnsw_kernel_fp32": 0, "rhnsw_kernel_sq8": 1}
_LABEL = re.compile(r"^(\S.*):$")
_SCALAR = re.compile(r"^s(\d+)$")
FUSED = ("fmadd", "fmsub", "fnmadd", "fnmsub", "fmla", "fmls")
CONVERSIONS = ("ucvtf", "scvtf", "fcvt", "ushll", "uxtl", "sshll", "sxtl", "tbl")


def kernel_bodies(otool_text: str) -> dict:
    """`otool -tV` 的輸出中，名稱含 `rhnsw_kernel_fp32` / `rhnsw_kernel_sq8` 的函式：label → [(mnemonic, operands)]。"""
    bodies, current = {}, None
    for line in otool_text.splitlines():
        label = _LABEL.match(line)
        if label and "\t" not in line:
            name = label.group(1)
            current = name if any(kernel in name for kernel in KERNELS) else None
            if current is not None:
                bodies[current] = []
            continue
        if current is None:
            continue
        fields = line.split("\t")
        if len(fields) < 2 or not fields[1]:
            continue
        operands = fields[2].split(";")[0].strip() if len(fields) > 2 else ""
        bodies[current].append((fields[1].strip(), operands))
    return bodies


def check_kernel(instructions, final_adds: int) -> dict:
    """019 §2 的機械判準：每次累加都是 `fadd sN, sN, sM`、sN 是同一個累加暫存器；不得出現 `faddp`、向量 `fadd`（例如 `fadd.4s`）
    或 `fadda` 作為累加。乘法與 u8→f32 轉換可以是向量指令。

    具體化（019 沒有寫明，固定在檢查之前）：
    - 函式內任何 `faddp` / `fadda` / 帶排列的 `fadd.<T>` / 運算元不全是 s 暫存器的 `fadd`、以及融合乘加（fmadd、fmla…）都算違反；
    - 「累加暫存器」是 `fadd sN, sN, sM` 中最常出現的 N；
    - 其他的純量 `fadd` 只允許「以累加暫存器為來源、寫入另一個暫存器」的最後一次加法，次數必須恰好是 `final_adds`
      （SQ8 的 b + Σ 是 1 次，FP32 是 0 次）。
    """
    violations, adds = [], []
    for mnemonic, operands in instructions:
        base = mnemonic.split(".")[0]
        if base in ("faddp", "fadda") or base in FUSED or (base == "fadd" and "." in mnemonic):
            violations.append(f"{mnemonic} {operands}")
            continue
        if mnemonic != "fadd":
            continue
        regs = [op.strip() for op in operands.split(",")]
        matched = [_SCALAR.match(op) for op in regs]
        if len(regs) != 3 or not all(matched):
            violations.append(f"{mnemonic} {operands}")
            continue
        adds.append(tuple(int(m.group(1)) for m in matched))
    chains = Counter(d for d, a, b in adds if d == a and b != d)
    result = {"instructions": len(instructions), "accumulator": None, "accumulations": 0, "final_adds": [],
              "violations": violations}
    if not chains:
        violations.append("no accumulation of the form fadd sN, sN, sM")
        result["ok"] = False
        return result
    (acc, count), *rest = chains.most_common()
    if rest and rest[0][1] == count:
        violations.append(f"no single accumulator register: {dict(chains)}")
    result.update(accumulator=f"s{acc}", accumulations=count)
    for d, a, b in adds:
        if d == a == acc and b != acc:
            continue
        if d != acc and acc in (a, b):
            result["final_adds"].append(f"fadd s{d}, s{a}, s{b}")
        else:
            violations.append(f"fadd s{d}, s{a}, s{b} (not an accumulation into s{acc})")
    if len(result["final_adds"]) != final_adds:
        violations.append(f"expected {final_adds} final add(s) consuming s{acc}, found {result['final_adds']}")
    result["ok"] = not violations
    return result


def premise(otool_text: str) -> dict:
    bodies = kernel_bodies(otool_text)
    kernels = {}
    for kernel in KERNELS:
        matches = [name for name in bodies if name.endswith(kernel)]
        if len(matches) != 1:
            kernels[kernel] = {"ok": False, "symbols": matches,
                               "violations": [f"expected exactly one function named *{kernel}, found {len(matches)}"]}
            continue
        entry = {"symbol": matches[0], **check_kernel(bodies[matches[0]], FINAL_ADDS[kernel])}
        if kernel == "rhnsw_kernel_sq8":
            # 只作紀錄（019 §8 的靜態佐證）：u8 → f32 的轉換指令。
            entry["conversions"] = dict(sorted(Counter(m for m, _ in bodies[matches[0]]
                                                       if m.split(".")[0].startswith(CONVERSIONS)).items()))
        kernels[kernel] = entry
    status = "成立" if all(k["ok"] for k in kernels.values()) else "前提不成立"
    return {"kernels": kernels, "status": status}


def run_premise(out_path=None, so_path=None) -> dict:
    """反組譯目前安裝的 binding（或 `so_path`），寫出紀錄；拒絕覆寫上一次的輸出。"""
    import rust_hnsw

    out = Path(out_path or PREMISE_OUT)
    if out.exists():
        raise SystemExit(f"{out} exists; move it away first (an earlier check must not be overwritten)")
    so = Path(so_path or rust_hnsw.rust_hnsw.__file__)
    text = subprocess.run(["otool", "-tV", str(so)], check=True, capture_output=True, text=True).stdout
    record = {"tool": "benchmarks.quant_profile premise", "decision": "019 §2, §8", "binding": str(so),
              "binding_sha256": environment.sha256_file(so), "git": dataprep.git_state(),
              "build_info": dict(rust_hnsw.BUILD_INFO), "command": f"otool -tV {so}", **premise(text)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "check", "premise"])
    parser.add_argument("runs", nargs="*", help="10 份 C formal config 的 run 目錄（plan、check；兩個資料集都要列出）")
    parser.add_argument("--dataset", type=quant_report.dataset_name, help="plan、check 的資料集（glove 或 text）")
    parser.add_argument("--excluded", action="append", default=[], help="已整份重跑的失敗 run（與 quant_report 相同）")
    parser.add_argument("--out", help="premise 的輸出路徑；預設為 results/quant/premise.json")
    args = parser.parse_args(argv)
    if args.command == "premise":
        record = run_premise(args.out)
        print(json.dumps({"status": record["status"], **{k: v["ok"] for k, v in record["kernels"].items()}}, ensure_ascii=False))
        return 0 if record["status"] == "成立" else 1
    if not args.runs or args.dataset is None:
        parser.error(f"{args.command} needs --dataset and the run directories of all 10 C formal configs")
    try:
        groups = quant_report.load_formal_set(args.runs, excluded=args.excluded)
    except quant_report.ReportError as err:
        raise SystemExit(f"the given runs are not the complete set of the 10 C formal configs (019 §9 step 8): {err}") from None
    records = [record for _, record in groups[args.dataset]]
    if args.command == "plan":
        sys.stdout.write(plan_script(records))
        return 0
    results = check(records)
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
