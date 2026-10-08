"""解析 macOS `sample` 報告的 "Sort by top of stack" 段落，整理成 self-time 排行。

用法：python -m benchmarks.profile_report <sample 輸出檔> [--top N]
"""

import argparse
import re
from dataclasses import dataclass

_SECTION = "Sort by top of stack"
_ROW = re.compile(r"^\s+(?P<symbol>\S.*?)\s+\(in (?P<image>[^)]+)\)\s+(?P<count>\d+)\s*$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Rust v0 mangling 中不屬於路徑名稱的片段：crate 雜湊（Cs…_）、back reference（B…_）、impl 的 disambiguator（Ms…_）。
_V0_NOISE = re.compile(r"Cs[0-9A-Za-z]+_|B[0-9A-Za-z]*_|(?<=M)s[0-9A-Za-z]*_")


@dataclass(frozen=True)
class Row:
    name: str
    symbol: str
    image: str
    count: int
    share: float


def pretty_symbol(symbol: str) -> str:
    """把 Rust v0 mangled 名稱整理成 `a::b::c`（近似，只用於閱讀）；其他名稱原樣回傳。"""
    if not symbol.startswith("_R"):
        return symbol
    text = _V0_NOISE.sub("", symbol[2:])
    parts = []
    i = 0
    while i < len(text):
        if text[i].isdigit():
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
            length = int(text[i:j])
            candidate = text[j : j + length]
            if length >= 2 and len(candidate) == length and _IDENT.fullmatch(candidate):
                parts.append(candidate)
                i = j + length
                continue
            i = j
        else:
            i += 1
    return "::".join(parts) if parts else symbol


def top_of_stack(report: str) -> list[Row]:
    """回傳 top-of-stack 的每一列（依 sample 的原始順序），share 為占該段總樣本數的比例。"""
    lines = report.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(_SECTION))
    except StopIteration:
        raise ValueError("report has no 'Sort by top of stack' section") from None
    parsed = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        match = _ROW.match(line)
        if match:
            parsed.append((match["symbol"], match["image"], int(match["count"])))
    total = sum(count for _, _, count in parsed) or 1
    return [Row(pretty_symbol(symbol), symbol, image, count, count / total) for symbol, image, count in parsed]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    with open(args.report, encoding="utf-8", errors="replace") as f:
        rows = top_of_stack(f.read())
    for row in rows[: args.top]:
        print(f"{row.share:6.1%}  {row.count:7d}  {row.name}  [{row.image}]")


if __name__ == "__main__":
    main()
