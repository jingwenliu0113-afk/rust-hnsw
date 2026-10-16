"""比對兩個 commit 中 hnsw crate 的 Core 函式機器碼（decision 019 補記：Core 的原始碼不變，但 codegen 改變）。

用法（在 repo root）：
    .venv/bin/python scripts/probe_core_codegen.py <rev a> <rev b> [--out <json>]

- 以 `git archive` 取出兩個 rev 的 `crates/hnsw`、`Cargo.toml`、`Cargo.lock`，workspace 只留 `crates/hnsw`
  （hnsw crate 沒有相依，根目錄的 Cargo.toml 也沒有 [profile]，所以與 maturin 編譯的 binding 中的 hnsw crate 是同一份設定）。
- `cargo build --release --offline -p hnsw --lib`，再以 `ar x` 拆出 rlib 的 object、`otool -tV` 反組譯每個函式。
- 指令序列先正規化（位址、區域標籤、crate disambiguator `Cs…_`、`.llvm.<hash>`），再比對名稱不含 quant 模組的函式。
- 未連結的 object 中 `bl` 的目標只是位址，所以這裡只比對指令序列與指令數；呼叫了哪些函式要在連結後的 binding 上看。
- 只描述，不判定任何東西；只編譯零相依的 hnsw crate（各約數秒），不是重工作。
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _normalize_name(name: str) -> str:
    return re.sub(r"\.llvm\.\d+", ".llvm", re.sub(r"Cs[0-9A-Za-z]+_", "CS_", name))


def _normalize(ins: str) -> str:
    ins = re.sub(r"0x[0-9a-f]+", "ADDR", ins)
    ins = re.sub(r"\bltmp\d+\b|\blloh\d+\b|\bLBB\d+_\d+\b|\bl_[A-Za-z0-9_.]+", "LBL", ins)
    return _normalize_name(ins)


def disassemble(rev: str, work: Path) -> dict[str, list[str]]:
    work = work / rev
    src = work / "src"
    src.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", rev, "crates/hnsw", "Cargo.toml", "Cargo.lock"], cwd=REPO,
                             check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(src, filter="data")
    manifest = src / "Cargo.toml"
    manifest.write_text(re.sub(r"members = \[[^\]]*\]", 'members = ["crates/hnsw"]', manifest.read_text()))
    target = work / "target"
    subprocess.run(["cargo", "build", "--release", "--offline", "-p", "hnsw", "--lib"], cwd=src, check=True,
                   env={**os.environ, "CARGO_TARGET_DIR": str(target)}, capture_output=True)
    (rlib,) = (target / "release" / "deps").glob("libhnsw-*.rlib")
    objects = work / "obj"
    objects.mkdir()
    subprocess.run(["ar", "x", str(rlib)], cwd=objects, check=True)
    funcs: dict[str, list[str]] = {}
    for obj in sorted(objects.glob("*.o")):
        text = subprocess.run(["otool", "-tV", str(obj)], check=True, capture_output=True, text=True).stdout
        current = None
        for line in text.splitlines():
            if line.endswith(":") and not line.startswith(("\t", " ")):
                name = line[:-1]
                current = None if name.endswith(".o") or name.startswith("(__TEXT") else _normalize_name(name)
                if current is not None:
                    funcs.setdefault(current, [])
                continue
            parts = line.split("\t", 1)
            if current is not None and len(parts) == 2:
                funcs[current].append(_normalize(parts[1]))
    return funcs


def _demangle(names: list[str]) -> dict[str, str]:
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    return dict(zip(names, out))


def compare(a: dict, b: dict) -> dict:
    core = lambda funcs: {n for n in funcs if "5quant" not in n}  # noqa: E731（v0 / legacy mangling 中的模組名）
    a_core, b_core = core(a), core(b)
    shared = sorted(a_core & b_core)
    names = _demangle(sorted(a_core | b_core))
    differ = [n for n in shared if a[n] != b[n]]
    return {
        "identical": len(shared) - len(differ),
        "different": [{"function": names[n], "instructions": [len(a[n]), len(b[n])]} for n in differ],
        "only_in_a": sorted(names[n] for n in a_core - b_core),
        "only_in_b": sorted(names[n] for n in b_core - a_core),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rev_a")
    parser.add_argument("rev_b")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    revs = [subprocess.run(["git", "rev-parse", r], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
            for r in (args.rev_a, args.rev_b)]
    rustc = subprocess.run(["rustc", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        a, b = (disassemble(r, Path(tmp) / side) for r, side in zip(revs, "ab"))
    record = {"tool": "scripts/probe_core_codegen.py", "decision": "019 (implementation addendum)",
              "rev_a": revs[0], "rev_b": revs[1], "rustc": rustc, **compare(a, b)}
    text = json.dumps(record, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(f"identical {record['identical']}, different {len(record['different'])}, "
          f"only in a {len(record['only_in_a'])}, only in b {len(record['only_in_b'])}", file=sys.stderr)
    for d in record["different"]:
        print(f"  {d['function'][:120]}: {d['instructions'][0]} -> {d['instructions'][1]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
