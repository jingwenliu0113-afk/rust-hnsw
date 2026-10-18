"""README 的關鍵數字與判定必須與 docs/results.md 一致，不是手寫的宣稱。

- `README.md`（英文入口）與 `README.zh-TW.md`（繁體中文版）都以 `docs/results.md` 核對。docs/results.md 是四份正式報告的
  逐值轉錄，與 README 一起進版控；它與本機報告的一致性由 `tests/test_results_doc.py` 核對。本測試不需要 `results/`，
  只讀 repo 內的文件，乾淨的 clone 也能執行。
- 不一致時以 docs/results.md 為準（它又以報告為準，報告已封版）：要改的是 README，不是轉錄或報告。
- README 為了易讀，部分數值取較少位數（例如 P50 / P95 取整數 µs、ρ_eval 取 2–3 位小數）。這類數值只容許「顯示位數的半個單位」的差；
  QPS、記憶體、ef 與判定一律逐字比對。
- 另外核對：兩份 README 的同一張表引用相同的數字、事先登記與事後描述的限定語都還在；公開文件（`PUBLIC_DOCS`）的相對連結與
  頁內錨點都存在，而且不含本機路徑與 email。
- 只讀檔，不產生任何讀值。
"""

import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "results.md"
READMES = {"en": ROOT / "README.md", "zh": ROOT / "README.zh-TW.md"}
LANGS = tuple(READMES)
# 要公開的文件（decision log 001–019 不在這裡：它們是開發紀錄，保留原樣）。
PUBLIC_DOCS = (
    "README.md",
    "README.zh-TW.md",
    "docs/results.md",
    "docs/architecture.md",
    "docs/reproduce.md",
    "docs/figures/README.md",
    "docs/decisions/README.md",
    "THIRD_PARTY_NOTICES.md",
    "LICENSE",
)
LINKED_DOCS = tuple(p for p in PUBLIC_DOCS if p.endswith(".md"))
CORE = ("glove-100-angular", "simplewiki-embeddinggemma2-100k")
TARGETS = ("0.90", "0.95", "0.99")
IMPLS = ("rust-hnsw", "hnswlib", "faiss-hnsw")

# 各語言 README 中的章節標題（標題文字的開頭）與表頭（前幾欄）。
CORE_SECTION = {
    "glove-100-angular": {"en": "glove-100-angular", "zh": "正式結果：glove-100-angular"},
    "simplewiki-embeddinggemma2-100k": {
        "en": "simplewiki-embeddinggemma2-100k",
        "zh": "正式結果：simplewiki-embeddinggemma2-100k",
    },
}
FRONTIER_HEADER = {"en": ("Target Recall@10", "Implementation", "QPS"), "zh": ("目標 Recall@10", "實作", "QPS")}
GATE_HEADER = {"en": ("Condition (SPEC)",), "zh": ("條件（SPEC）",)}
VERDICT_HEADER = {"en": ("Row", "Verdict"), "zh": ("列", "判定")}
QUANT_FRONTIER_HEADER = {"en": ("Dataset", "Arm", "ef"), "zh": ("資料集", "實作", "ef")}
STATUS = {  # docs/results.md 的判定 → README 的寫法
    "成立": {"en": "holds", "zh": "成立"},
    "不成立": {"en": "does not hold", "zh": "不成立"},
    "無法判定": {"en": "undetermined", "zh": "無法判定"},
}
TEXT_LABEL = {"en": "Text", "zh": "文字"}
ARM_IMPL = {"F": "rust-hnsw-fp32g", "S": "rust-hnsw-sq8", "P": "faiss-hnsw", "Q": "faiss-hnsw-sq8", "R": "rust-hnsw（錨點）"}


# ---------------------------------------------------------------- 讀檔與解析


def strip_code(text: str) -> str:
    """去掉 fenced code block（含縮排的），其中的 `#` 與連結不算標題或連結。"""
    return re.sub(r"^[ \t]*```.*?^[ \t]*```[^\n]*$", "", text, flags=re.M | re.S)


def readme(lang: str) -> str:
    return READMES[lang].read_text()


def flat(text: str) -> str:
    """把換行與縮排壓成單一空白，讓跨行的句子也能以子字串比對。"""
    return " ".join(text.split())


def plain(text: str) -> str:
    """flat 之後再去掉粗體標記，用於限定語的比對。"""
    return flat(text).replace("**", "")


def _cells(line: str) -> list[str]:
    inner = line.strip()[1:-1]
    return [c.strip().replace("\\|", "|").replace("\\*", "*") for c in re.split(r"(?<!\\)\|", inner)]


def tables(text: str) -> list[list[list[str]]]:
    """文件中所有表（code block 之外），每張表第一列是表頭，不含分隔列。"""
    out, current = [], None
    for line in strip_code(text).splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = _cells(s)
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            if current is None:
                current = []
                out.append(current)
            current.append(cells)
        else:
            current = None
    return out


def find_tables(text: str, header: tuple[str, ...]) -> list[list[list[str]]]:
    return [t for t in tables(text) if tuple(t[0][: len(header)]) == header]


def one_table(text: str, header: tuple[str, ...]) -> list[list[str]]:
    hits = find_tables(text, header)
    assert len(hits) == 1, (header, len(hits))
    return hits[0][1:]


def section(text: str, title_prefix: str) -> str:
    """標題以 title_prefix 開頭的章節（到下一個同層或更上層的標題為止），不含標題本身。"""
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if re.match(rf"#{{1,6}} {re.escape(title_prefix)}", line)]
    assert len(hits) == 1, (title_prefix, hits)
    start = hits[0]
    level = len(lines[start]) - len(lines[start].lstrip("#"))
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = re.match(r"(#{1,6}) ", lines[j])
        if m and len(m.group(1)) <= level:
            end = j
            break
    return "\n".join(lines[start + 1 : end])


def doc_text() -> str:
    return DOC_PATH.read_text()


def doc_table(name: str) -> list[list[str]]:
    """docs/results.md 中 `<!-- check:name -->` 之後的第一張表（不含表頭與分隔列）。"""
    lines = doc_text().splitlines()
    tag = f"<!-- check:{name} -->"
    assert lines.count(tag) == 1, tag
    i = lines.index(tag) + 1
    while not lines[i].startswith("|"):
        i += 1
    rows = []
    while i < len(lines) and lines[i].startswith("|"):
        rows.append(_cells(lines[i]))
        i += 1
    return rows[2:]


DOC_CELL = re.compile(
    r"^(?P<v>[\d.]+)(?: \[(?P<lo>[\d.]+), (?P<hi>[\d.]+)\]（(?P<n>\d+) builds）)?"
    r" @ M=(?P<m>\d+), efC=(?P<efc>\d+), ef=(?P<ef>\d+)$"
)
README_CELL = re.compile(
    r"^(?P<v>[\d,]+(?:\.\d+)?)(?: \[(?P<lo>[\d,]+), (?P<hi>[\d,]+)\])?\s*[（(](?P<params>\d+(?: / \d+)*)[）)]$"
)


def doc_cell(cell: str) -> re.Match:
    m = DOC_CELL.match(cell)
    assert m, cell
    return m


def readme_cell(cell: str) -> tuple[str, str | None, str | None, tuple[str, ...]]:
    m = README_CELL.match(cell)
    assert m, cell
    return m["v"], m["lo"], m["hi"], tuple(m["params"].split(" / "))


def plain_int(s: str) -> str:
    return s.replace(",", "")


def shown(display: str, exact: float) -> bool:
    """display 是 exact 依顯示位數取位的結果（容許半個單位）。"""
    d = display.replace(",", "")
    places = len(d.split(".")[1]) if "." in d else 0
    return abs(float(d) - exact) <= 0.5 * 10**-places + 1e-9


def number_tokens(text: str) -> list[str]:
    """一段 Markdown 中的數字（去掉連結網址與千分位），排序後回傳。"""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return sorted(t.replace(",", "") for t in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text))


def comma(n: str | int) -> str:
    return f"{int(n):,}"


# ---------------------------------------------------------------- Core


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("name", CORE)
def test_core_frontier_tables_match_results_doc(name, lang):
    """frontier 表：QPS（含 [min, max]）、記憶體與每格的 (M, efC, ef) 逐字相同；P50 / P95 只取位不同。"""
    rows = one_table(section(readme(lang), CORE_SECTION[name][lang]), FRONTIER_HEADER[lang])
    doc = {(r[1], r[0]): r for r in doc_table(f"core-frontier:{name}")}
    assert len(rows) == len(doc) == 9
    for target, impl, qps, p50, p95, mem in rows:
        d = doc[(target, impl)]
        v, lo, hi, params = readme_cell(qps)
        dq = doc_cell(d[2])
        assert (plain_int(v), lo and plain_int(lo), hi and plain_int(hi)) == (dq["v"], dq["lo"], dq["hi"]), (name, target, impl)
        assert params == (dq["m"], dq["efc"], dq["ef"]), (name, target, impl, "QPS")
        for cell, dcell in ((p50, d[3]), (p95, d[4])):
            v, _, _, params = readme_cell(cell)
            dm = doc_cell(dcell)
            assert shown(v, float(dm["v"])), (name, target, impl, cell, dcell)
            assert params == (dm["m"], dm["efc"], dm["ef"]), (name, target, impl, cell)
        v, _, _, params = readme_cell(mem)
        dm = doc_cell(d[5])
        assert v == dm["v"] and params == (dm["m"], dm["efc"]), (name, target, impl, mem)


@pytest.mark.parametrize("lang", LANGS)
def test_headline_qps_at_095(lang):
    q = {
        name: [doc_cell(r[2])["v"] for i in IMPLS for r in doc_table(f"core-frontier:{name}") if r[0] == i and r[1] == "0.95"]
        for name in CORE
    }
    g, t = ([comma(x) for x in q[name]] for name in CORE)
    text = flat(readme(lang))
    if lang == "en":
        assert f"rust-hnsw {g[0]}, hnswlib {g[1]}, faiss-hnsw {g[2]} on GloVe; {t[0]}, {t[1]}, {t[2]} on text" in text
        assert f"QPS {' / '.join(g)} (rust-hnsw / hnswlib / faiss-hnsw)" in text
        assert f"QPS {' / '.join(t)};" in text
    else:
        assert f"GloVe 上 rust-hnsw {g[0]}、hnswlib {g[1]}、faiss-hnsw {g[2]}；文字上 {t[0]}、{t[1]}、{t[2]}" in text


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("name", CORE)
def test_core_qps_ratios(name, lang):
    ratios = {r[1]: (r[2], r[3]) for r in doc_table(f"core-ratios:{name}") if r[0] == "100000"}
    faiss = " / ".join(ratios[t][0] for t in TARGETS)
    hnswlib = " / ".join(ratios[t][1] for t in TARGETS)
    text = flat(section(readme(lang), CORE_SECTION[name][lang]))
    assert f"hnswlib ÷ rust-hnsw = {hnswlib}" in text
    assert f"faiss-hnsw ÷ rust-hnsw = {faiss}" in text
    if lang == "en":  # 只有英文版有 Key results 表
        assert f"hnswlib ÷ rust-hnsw = {ratios['0.95'][1]}" in flat(section(readme(lang), "Key results"))


def _span(values: list[str], places: int) -> str:
    xs = [float(v) for v in values]
    return f"{min(xs):.{places}f}–{max(xs):.{places}f}"


@pytest.mark.parametrize("lang", LANGS)
def test_ratio_ranges_quoted_in_prose(lang):
    text_rows = [r for r in doc_table("core-ratios:simplewiki-embeddinggemma2-100k") if r[0] in ("10000", "30000")]
    hnswlib_small, faiss_small = _span([r[3] for r in text_rows], 2), _span([r[2] for r in text_rows], 2)
    assert (hnswlib_small, faiss_small) == ("1.03–1.07", "2.89–4.52")
    text_section = flat(section(readme(lang), CORE_SECTION["simplewiki-embeddinggemma2-100k"][lang]))
    assert hnswlib_small in text_section and faiss_small in text_section
    # 「FAISS 在兩者都最快」的倍數：兩個資料集都只取 n = 100,000 的 frontier 讀值比值（同一個口徑）。
    glove_faiss = _span([r[2] for r in doc_table("core-ratios:glove-100-angular") if r[0] == "100000"], 2)
    text_faiss = _span([r[2] for r in doc_table("core-ratios:simplewiki-embeddinggemma2-100k") if r[0] == "100000"], 2)
    assert (glove_faiss, text_faiss) == ("1.64–2.23", "2.85–3.21")
    whole = flat(readme(lang))
    phrase = {
        "en": f"at n = 100,000: {glove_faiss}× on GloVe, {text_faiss}× on text",
        "zh": f"n = 100,000 的 frontier 讀值比值 faiss-hnsw ÷ rust-hnsw：GloVe {glove_faiss} 倍、文字 {text_faiss} 倍",
    }[lang]
    assert phrase in whole


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("name", CORE)
def test_gate_brackets_and_codes(name, lang):
    rows = {r[0]: r for r in doc_table(f"core-gate:{name}")}
    primary = rows["0.95"]
    assert primary[1] == "是"
    rust, hnswlib = (re.findall(r"\d+", primary[i]) for i in (2, 3))
    text = flat(section(readme(lang), CORE_SECTION[name][lang]))
    assert f"[{comma(rust[0])}, {comma(rust[1])}]" in text
    assert f"[{comma(hnswlib[0])}, {comma(hnswlib[1])}]" in text
    codes = {t: rows[t][5].split("：")[0].strip("`") for t in TARGETS}
    assert codes["0.95"] != "gap"  # 依 015，主要目標不是 gap → 條件 3 不成立
    gate_row3 = [r for r in one_table(readme(lang), GATE_HEADER[lang]) if r[0].startswith("3.")][0][2]
    if name == "glove-100-angular":
        expected = {
            "en": f"GloVe's primary target 0.95 is `{codes['0.95']}` (0.90 `{codes['0.90']}`, 0.99 `{codes['0.99']}`)",
            "zh": f"GloVe 主要目標 0.95 為 `{codes['0.95']}`（0.90 為 `{codes['0.90']}`、0.99 為 `{codes['0.99']}`）",
        }[lang]
        assert expected in gate_row3
    else:
        assert len(set(codes.values())) == 1
        code = codes["0.95"]
        expected = {"en": f"all three text targets are `{code}`", "zh": f"文字三個目標都是 `{code}`"}[lang]
        assert expected in gate_row3


@pytest.mark.parametrize("lang", LANGS)
def test_gate_summary_verdicts(lang):
    doc = {r[0].split(".")[0]: r[2].strip("*") for r in doc_table("gate-summary")}
    assert doc == {"1": "成立", "2": "成立", "3": "不成立", "4": "成立"}
    rows = {r[0].split(".")[0]: r[1].strip("*") for r in one_table(readme(lang), GATE_HEADER[lang])}
    assert rows == {k: STATUS[v][lang] for k, v in doc.items()}


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("name", CORE)
def test_core_build_time_and_memory(name, lang):
    rows = {r[0]: r for r in doc_table(f"core-scaling:{name}") if r[1] == "100000"}
    text = flat(section(readme(lang), CORE_SECTION[name][lang]))
    m = re.search(
        r"build ([\d.]+) / ([\d.]+) / ([\d.]+) s[，,]\s*peak RSS ([\d,.]+) / ([\d,.]+) / ([\d,.]+) MiB[，,]\s*"
        r"index memory ([\d.]+) / ([\d.]+) / ([\d.]+) MiB",
        text,
    )
    assert m, (name, lang)
    for k, impl in enumerate(IMPLS):
        build, rss, mem = (rows[impl][i].split(" ")[0] for i in (2, 3, 4))
        assert shown(m[1 + k], float(build)), (impl, m[1 + k], build)
        assert shown(m[4 + k], float(rss)), (impl, m[4 + k], rss)
        assert m[7 + k] == mem, (impl, m[7 + k], mem)


@pytest.mark.parametrize("lang", LANGS)
def test_text_variance_grid_qps(lang):
    rows = [r for r in doc_table("core-seed-ranges:simplewiki-embeddinggemma2-100k") if r[1] == "(16, 200)"]
    text = flat(readme(lang))
    for target in TARGETS:
        by = {r[2]: r for r in rows if r[0] == target}
        h, r = by["hnswlib"], by["rust-hnsw"]
        assert h[3] == r[3]  # 同一個 ef
        assert (h[4], r[4]) == ("5", "4")  # rust-hnsw 的 seed 1 被標記，不是 5 個 seed
        hq, rq = h[6].split(" ")[0], r[6].split(" ")[0]
        phrase = {"en": f"{comma(hq)} vs {comma(rq)} (ef {h[3]})", "zh": f"{comma(hq)} 對 {comma(rq)}（ef {h[3]}）"}[lang]
        assert phrase in text, phrase
    builds = {"en": "(hnswlib 5 builds; rust-hnsw 4: seed 1's build is flagged `background`",
              "zh": "（hnswlib 5 個 build；rust-hnsw 4 個：seed 1 的 build 被標記 `background`"}[lang]
    assert builds in text
    # 被標記的 build：(16, 200) seed 1 原始順序的 rust-hnsw 與 faiss-hnsw；Gate 判讀含不含都相同。
    flagged = {(r[0], r[2], r[3], r[4]) for r in doc_table("core-flagged:simplewiki-embeddinggemma2-100k")}
    assert {("rust-hnsw", "(16, 200)", "1", "原始"), ("faiss-hnsw", "(16, 200)", "1", "原始")} <= flagged
    assert all(r[6] == "相同" for r in doc_table("core-gate:simplewiki-embeddinggemma2-100k"))
    (changed,) = doc_table("core-frontier-flagged:simplewiki-embeddinggemma2-100k")
    main = {(r[0], r[1]): r for r in doc_table("core-frontier:simplewiki-embeddinggemma2-100k")}[(changed[0], changed[1])]
    a, b = doc_cell(changed[2]), doc_cell(main[2])
    sep = {"en": " instead of ", "zh": "，主要讀值是 "}[lang]
    paren = {"en": (" (", ")"), "zh": ("（", "）")}[lang]
    phrase = (f"{comma(a['v'])} [{comma(a['lo'])}, {comma(a['hi'])}]{paren[0]}{a['n']} builds{paren[1]}{sep}"
              f"{comma(b['v'])} [{comma(b['lo'])}, {comma(b['hi'])}]{paren[0]}{b['n']} builds{paren[1]}")
    assert phrase in text, phrase


@pytest.mark.parametrize("lang", LANGS)
def test_post_hoc_ratios_are_traceable_and_labelled(lang):
    doc = flat(doc_text())
    assert "median 比值 1.22–1.38" in doc and "文字上 hnswlib 只快 6–8%" in doc
    text = flat(readme(lang))
    label = {"en": "Post-hoc observation (not a reading)", "zh": "看到結果之後的觀察（不是判讀）"}[lang]
    assert label in text
    assert "1.22–1.38" in text and "6–8%" in text


@pytest.mark.parametrize("lang", LANGS)
def test_glove_300k_flagged_reading(lang):
    row = [r for r in doc_table("core-other-n:glove-100-angular") if r[:3] == ["rust-hnsw", "300000", "0.95"]][0]
    assert row[3] == "未達到" and row[4] == "是"
    qps, ef = doc_cell(row[5])["v"], doc_cell(row[5])["ef"]
    assert f"{qps} QPS @ ef {ef}" in flat(readme(lang))


# ---------------------------------------------------------------- Advanced C


VERDICT_ROWS = [  # (docs/results.md 的列, README en 的列開頭, README zh 的列開頭, 預期的判定)
    ("H1 前提（反組譯）", "H1 premise", "H1 前提", "成立"),
    ("H1a 文字", "H1a", "H1a", "成立"),
    ("H1b GloVe", "H1b", "H1b", "無法判定"),
    ("H1c 維度對比", "H1c", "H1c", "無法判定"),
    ("H1d profiler（seed 1）", "H1d", "H1d", "無法判定"),
    ("H1 整體", "H1 overall", "H1 整體", "無法判定"),
    ("H2 GloVe", "H2 GloVe", "H2 GloVe", "成立"),
    ("H2 文字", "H2 text", "H2 文字", "成立"),
    ("H2 整體", "H2 overall", "H2 整體", "成立"),
]


def verdict_row(lang: str, prefix: str) -> list[str]:
    rows = [r for r in one_table(readme(lang), VERDICT_HEADER[lang]) if r[0].strip("*").startswith(prefix)]
    assert len(rows) == 1, (lang, prefix, rows)
    return rows[0]


def judgement() -> dict[str, list[str]]:
    return {r[0]: r for r in doc_table("quant-judgement")}


@pytest.mark.parametrize("lang", LANGS)
def test_hypothesis_verdicts(lang):
    """H1a 成立、H1b 無法判定、H1 整體無法判定、H2 成立：docs/results.md 與兩份 README 一致。"""
    doc = judgement()
    for doc_row, en, zh, expected in VERDICT_ROWS:
        assert doc[doc_row][1] == expected, doc_row
        row = verdict_row(lang, {"en": en, "zh": zh}[lang])
        assert row[1].strip("*") == STATUS[expected][lang], (lang, doc_row, row[1])


@pytest.mark.parametrize("lang", LANGS)
def test_quantization_frontier_readings(lang):
    doc = judgement()

    def code(dataset: str, pair: str, target: str) -> str:
        return doc[f"quantization frontier {dataset} {pair} {target}"][1]

    sf, qp = {"en": ("S vs F", "Q vs P"), "zh": ("S 對 F", "Q 對 P")}[lang]
    glove, text_ = {"en": ("GloVe", "text"), "zh": ("GloVe", "文字")}[lang]
    open_, close_, semi = {"en": (" (", ")", "; "), "zh": ("（", "）", "；")}[lang]
    lead = {"en": "quantization frontier, ", "zh": "quantization frontier，"}[lang]
    cases = [
        (f"{lead}{sf}{open_}{glove} 0.90 / 0.95{close_}", [("glove", "S 對 F")]),
        (f"{lead}{sf}{open_}{text_} 0.90 / 0.95{close_}", [("text", "S 對 F")]),
        (f"{lead}{qp}{open_}{glove} 0.90 / 0.95{semi}{text_} 0.90 / 0.95{close_}", [("glove", "Q 對 P"), ("text", "Q 對 P")]),
    ]
    for label, pairs in cases:
        expected = "; " if lang == "en" else "；"
        cell = expected.join(" / ".join(f"`{code(d, p, t)}`" for t in ("0.90", "0.95")) for d, p in pairs)
        assert verdict_row(lang, label)[1] == cell, (lang, label)
        # 015 §1：包含被標記 build 的判讀與主要判讀不同時要並列，不自行擇一。
        for d, p in pairs:
            alt = {}
            for t in ("0.90", "0.95"):
                m = re.search(r"包含被標記 build：(?:(\w+)（不同）|相同)", doc[f"quantization frontier {d} {p} {t}"][3])
                assert m, (d, p, t)
                if m[1]:
                    alt[t] = m[1]
            if not alt:
                continue
            assert sorted(alt) == ["0.90", "0.95"] and (d, p) == ("glove", "Q 對 P")
            lead_alt = {"en": "Including flagged builds, GloVe 0.90 / 0.95 is ",
                        "zh": "包含被標記 build 時，GloVe 0.90 / 0.95 為 "}[lang]
            assert lead_alt + " / ".join(f"`{alt[t]}`" for t in ("0.90", "0.95")) in verdict_row(lang, label)[2]
    pre_registered = {
        "en": "Including flagged builds, GloVe's Q vs P reading is `within_quantization` (different;",
        "zh": "包含被標記 build 時，GloVe 的 Q 對 P 判讀為 `within_quantization`（不同；",
    }[lang]
    assert pre_registered in flat(readme(lang))


@pytest.mark.parametrize("lang", LANGS)
def test_h1_rho_eval(lang):
    glove = doc_table("quant-h1:quant-glove")
    text_ = doc_table("quant-h1:quant-text")
    clean_t = [float(r[5]) for r in text_ if r[1] == "是"]
    clean_g = sorted(float(r[5]) for r in glove if r[1] == "是")
    flagged_g = sorted(float(r[5]) for r in glove if r[1] == "否")
    assert (len(clean_t), len(clean_g)) == (5, 2)
    h1a = verdict_row(lang, "H1a")[2]
    span = f"[{min(clean_t):.3f}, {max(clean_t):.3f}]"
    assert h1a == {"en": f"5 clean pairs, {span}", "zh": f"5 個乾淨配對，{span}"}[lang]
    sep = "、" if lang == "zh" else ", "
    h1b = verdict_row(lang, "H1b")[2]
    assert f"{len(clean_g)}" in h1b and sep.join(f"{x:.2f}" for x in clean_g) in h1b
    flagged = sep.join(f"{x:.2f}" for x in flagged_g)
    assert {"en": f"({flagged})", "zh": f"（{flagged}）"}[lang] in flat(readme(lang))
    clean_seeds = [r[0] for r in glove if r[1] == "是"]
    assert clean_seeds == ["2", "3"]
    phrase = {"en": "only seeds 2 and 3 as clean pairs", "zh": "只剩 seeds 2、3 兩個乾淨配對"}[lang]
    assert phrase in flat(readme(lang))


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize(("name", "prefix"), [("quant-glove", "H2 GloVe"), ("quant-text", {"en": "H2 text", "zh": "H2 文字"})])
def test_h2_loss_decomposition(name, prefix, lang):
    rows = doc_table(f"quant-h2:{name}")
    valid = [r for r in rows if r[1] == "是"]
    l_total = sorted(float(r[3]) for r in valid)
    l_trav = [float(r[4]) for r in valid]
    thr = sorted(float(r[7]) for r in valid)
    assert all(float(r[4]) <= float(r[7]) for r in valid)  # 「成立」的條件
    cell = verdict_row(lang, prefix if isinstance(prefix, str) else prefix[lang])[2]
    m = re.search(r"(\d+)\D+L_total ≈ ([\d.]+)\D+L_trav ≤ ([\d.]+)\D+≈ ([\d.]+)", cell)
    assert m, cell
    assert int(m[1]) == len(valid)
    assert shown(m[2], l_total[len(l_total) // 2])
    assert shown(m[3], max(l_trav))
    assert shown(m[4], thr[len(thr) // 2])


@pytest.mark.parametrize("lang", LANGS)
def test_c1_memory_tolerance(lang):
    rows = doc_table("quant-c1")
    out = [r for r in rows if r[5] == "**否**"]
    assert len(out) == 1
    pair, seed, _, deviation, tolerance = out[0][:5]
    assert (pair, seed) == ("glove F − S", "5")
    doc = judgement()
    assert doc["C1 記憶體 glove F − S"][1] == "不一致"
    assert [doc[k][1] for k in ("C1 記憶體 glove P − Q", "C1 記憶體 text F − S", "C1 記憶體 text P − Q")] == ["一致"] * 3
    text = flat(readme(lang))
    assert f"{comma(deviation)} B" in text and f"{comma(tolerance)} B" in text
    assert f"{int(deviation) / 2**20:.2f} MiB" in text  # 換算約 3.84 MiB
    assert f"{int(tolerance) / 2**20:.0f} MiB" in text
    phrase = {"en": "GloVe's F − S is **inconsistent**: only seed 5", "zh": "GloVe 的 F − S **不一致**：只有 seed 5"}[lang]
    assert phrase in text
    others = {"en": "GloVe's P − Q and text's F − S and P − Q are consistent", "zh": "GloVe 的 P − Q、文字的 F − S 與 P − Q 都一致"}[lang]
    assert others in text
    # C1 比的是診斷用的 ΔM_relief，不是主要指標 M_014。
    scope = {"en": "M_relief is only a diagnostic, and the primary memory metric M_014 is not judged by C1",
             "zh": "M_relief 只是診斷，主要記憶體指標 M_014 不由 C1 判定"}[lang]
    assert scope in text
    assert "ΔM_relief = M_relief(FP32 臂) − M_relief(SQ8 臂)" in doc_text()


@pytest.mark.parametrize("lang", LANGS)
def test_quant_frontier_table(lang):
    rows = one_table(readme(lang), QUANT_FRONTIER_HEADER[lang])
    assert len(rows) == 10
    docs = {
        "GloVe": {r[0]: r for r in doc_table("quant-frontier:quant-glove") if r[1] == "0.95"},
        TEXT_LABEL[lang]: {r[0]: r for r in doc_table("quant-frontier:quant-text") if r[1] == "0.95"},
    }
    for dataset, arm, ef, qps, p50, m014 in rows:
        d = docs[dataset][ARM_IMPL[arm[0]]]
        dq = doc_cell(d[2])
        m = re.fullmatch(r"([\d,]+)(?:\s*[（(](\d+) builds[）)])?", qps)
        assert m, qps
        assert (plain_int(m[1]), m[2], ef) == (dq["v"], dq["n"], dq["ef"]), (dataset, arm)
        assert shown(p50, float(doc_cell(d[3])["v"])), (dataset, arm, p50)
        assert m014 == doc_cell(d[5])["v"], (dataset, arm, m014)
    # 排除被標記 build 會改變讀值的臂（GloVe 的 F、S、P），README 也列出包含被標記 build 的讀值（010 §3）。
    main = {(r[0], r[1]): r for r in doc_table("quant-frontier:quant-glove")}
    flagged = {(r[0], r[1]): r for r in doc_table("quant-frontier-flagged:quant-glove")}
    assert {k[0] for k in flagged} == {"rust-hnsw-fp32g", "rust-hnsw-sq8", "faiss-hnsw"}
    assert all(main[k][6] == "是" for k in flagged)
    text = flat(readme(lang))
    for impl, target in (("faiss-hnsw", "0.95"), ("rust-hnsw-fp32g", "0.90"), ("rust-hnsw-sq8", "0.95")):
        a, b = doc_cell(flagged[(impl, target)][2]), doc_cell(main[(impl, target)][2])
        assert a["v"] != b["v"]
        phrase = {
            "en": f"{comma(a['v'])} ({a['n']} builds) instead of {comma(b['v'])} ({b['n']} builds)",
            "zh": f"{comma(a['v'])}（{a['n']} builds），主要讀值是 {comma(b['v'])}（{b['n']} builds）",
        }[lang]
        assert phrase in text, phrase


@pytest.mark.parametrize("lang", LANGS)
def test_speed_memory_tradeoff(lang):
    rows = {r[0]: r for r in doc_table("quant-tradeoff")}
    text = flat(readme(lang))
    vals = {}
    for dataset in ("文字", "GloVe"):
        _, target, f, s, fm, sm, _, reading = rows[dataset]
        assert target == "0.95" and reading == "`uncertain`"
        vals[dataset] = (f.split(" ")[0], s.split(" ")[0], fm.split(" ")[0], sm.split(" ")[0])
        assert f"{vals[dataset][2]} → {vals[dataset][3]} MiB" in text
    f, s, _, _ = vals["文字"]
    assert {"en": f"QPS {comma(s)} vs {comma(f)} at 0.95", "zh": f"0.95 時 QPS {comma(s)} 對 {comma(f)}"}[lang] in text
    if lang == "en":
        f, s, _, _ = vals["GloVe"]
        assert f"{comma(s)} vs {comma(f)} QPS" in text
    for target in ("0.95",):
        for dataset in ("glove", "text"):
            assert judgement()[f"quantization frontier {dataset} S 對 F {target}"][1] == "uncertain"
    # 記憶體的事後描述只引用 docs/results.md 已有的值（不另外換算）。
    retention = re.search(r"GloVe 的 Q 為 ([\d、]+) B，P 為 ([\d、]+) B（seeds 1–3）", doc_text())
    assert retention
    q_ret, p_ret = (" / ".join(comma(x) for x in g.split("、")) for g in retention.groups())
    sep = {"en": " B for Q against ", "zh": " B，P 為 "}[lang]
    assert f"{q_ret}{sep}{p_ret} B" in text
    mem = {r[1]: r for r in doc_table("quant-memory:quant-glove")}
    q_rel, p_rel = mem["faiss-hnsw-sq8"][4], mem["faiss-hnsw"][4]
    phrase = {"en": f"{q_rel} MiB for Q against {p_rel} MiB for P", "zh": f"Q 為 {q_rel} MiB，P 為 {p_rel} MiB"}[lang]
    assert phrase in text, phrase
    delta = [r for r in doc_table("quant-delta-m") if r[:3] == ["文字", "Q 對 P", "delta_m014"]][0][3].split(" ")[0]
    assert f"{delta} MiB" in text
    rss = {r[1]: [float(x) for x in r[6].split(" / ")] for r in doc_table("quant-memory:quant-text")}
    s_rss, f_rss = rss["rust-hnsw-sq8"], rss["rust-hnsw-fp32g"]
    s_span = f"{min(s_rss):,.1f}–{max(s_rss):,.1f} MiB"
    f_span = f"{min(f_rss):,.1f}–{max(f_rss):,.1f} MiB"
    assert s_span in text and f_span in text, (s_span, f_span)
    assert "1,023.2–1,023.6 MiB" == s_span and "950.1–950.9 MiB" == f_span


@pytest.mark.parametrize("lang", LANGS)
def test_consistency_check_numbers(lang):
    checks = {r[0]: r[2] for r in doc_table("quant-checks")}

    def first(pattern: str, row: str, places: int) -> str:
        return f"{float(re.search(pattern, checks[row])[1]):.{places}f}"

    c4 = [first(r"= ([\d.]+)", f"C4 路徑失真 {d}", 2) for d in ("glove", "text")]
    c6 = [_span(re.findall(r"ef \d+ ([\d.]+)", checks[f"C6 漂移 {d}"]), 2) for d in ("glove", "text")]
    ceil = [re.search(r"c = ([\d.]+)", checks[f"C3 天花板 {d}"])[1] for d in ("glove", "text")]  # 照報告的位數，不取位
    assert ceil == ["0.98048", "0.9935"]
    text = flat(readme(lang))
    phrases = {
        "en": [f"F ÷ R QPS is {c4[0]} on GloVe and {c4[1]} on text", f"{c6[0]} on GloVe and {c6[1]} on text",
               f"GloVe {ceil[0]}, text {ceil[1]}"],
        "zh": [f"F ÷ R 的 QPS 為 GloVe {c4[0]}、文字 {c4[1]}", f"GloVe {c6[0]}、文字 {c6[1]}", f"GloVe {ceil[0]}、文字 {ceil[1]}"],
    }[lang]
    for phrase in phrases:
        assert phrase in text, phrase
    h1d = judgement()["H1d profiler（seed 1）"][3]
    kappa_g, nu_g = re.search(r"GloVe κ = ([\d.]+)、ν = ([\d.]+)", h1d).groups()
    kappa_t = re.search(r"文字 κ = ([\d.]+)", h1d)[1]
    for v in (f"κ = {float(kappa_g):.2f}", f"ν = {float(nu_g):.2f}", f"κ = {float(kappa_t):.2f}"):
        assert v in text, v
    for doc_row, status in (("C5 Gate 重現 glove", "不一致"), ("C5 Gate 重現 text", "不一致"), ("C4 路徑失真 glove", "一致")):
        assert judgement()[doc_row][1] == status


@pytest.mark.parametrize("lang", LANGS)
def test_recall_cost_follows_h2_medians(lang):
    medians = []
    for name in ("quant-glove", "quant-text"):
        xs = sorted(float(r[3]) for r in doc_table(f"quant-h2:{name}") if r[1] == "是")
        medians.append(f"{xs[len(xs) // 2]:.3f}")
    assert medians == ["0.014", "0.005"]
    phrase = {"en": f"about {medians[0]} (GloVe) and {medians[1]} (text)", "zh": f"約 {medians[0]}（GloVe）與 {medians[1]}（文字）"}[lang]
    assert phrase in flat(readme(lang))


@pytest.mark.parametrize("lang", LANGS)
def test_background_flags_name_the_right_seeds_and_arms(lang):
    rows = doc_table("quant-hygiene")
    background = sorted((r[1][0], r[0]) for r in rows if r[2] == "`background`")
    assert background == [("F", "4"), ("F", "5"), ("S", "1")]
    phrase = {
        "en": "seed 1's S and seeds 4 and 5's F were flagged `background`",
        "zh": "GloVe 的 seed 1 的 S、seeds 4、5 的 F 在子 process 層級被標記為 `background`",
    }[lang]
    text = flat(readme(lang))
    assert phrase in text
    assert {"en": "3 seeds (1, 4, 5)", "zh": "3 個 seed（1、4、5）"}[lang] in text
    assert "文字的 5 個 run 沒有任何標記" in flat(doc_text())
    assert {"en": "Text's C runs carry no flags at all", "zh": "文字的 C run 沒有任何標記"}[lang] in text


# ---------------------------------------------------------------- 兩種語言、出處與連結


PARALLEL_TABLES = [  # (說明, zh 表頭的前幾欄, en 表頭的前幾欄, 張數)
    ("frontier", ("目標 Recall@10", "實作", "QPS"), ("Target Recall@10", "Implementation", "QPS"), 2),
    ("tests", ("測什麼", "在哪", "數量"), ("What is tested", "Where", "Count"), 1),
    ("real-data correctness", ("資料集", "n", "M / efC"), ("Dataset", "n", "M / efC"), 1),
    ("gate summary", ("條件（SPEC）",), ("Condition (SPEC)",), 1),
    ("gate profiling", ("資料集 / 階段",), ("Dataset / phase",), 1),
    ("verdict table", ("列", "判定"), ("Row", "Verdict"), 1),
    ("arms", ("代號", "實作"), ("Code", "Implementation"), 1),
    ("quant frontier", ("資料集", "實作", "ef"), ("Dataset", "Arm", "ef"), 1),
    ("quant profiling", ("資料集", "實作", "kernel"), ("Dataset", "Arm", "kernel"), 1),
    ("glossary", ("詞", "意思"), ("Term", "Meaning"), 1),
    ("simd and flags", ("實作", "編譯"), ("Implementation", "Build"), 1),
    ("spec metrics", ("SPEC 指標",), ("SPEC metric",), 1),
    ("engineering", ("項目", "內容"), ("What", "Details"), 1),
    ("pre-registration timeline", ("事先登記的內容",), ("Pre-registered content",), 1),
    ("not done", ("項目", "狀態"), ("Item", "Status"), 1),
    ("documentation map", ("想看什麼",), ("What you want to see",), 1),
]


@pytest.mark.parametrize(("label", "zh_header", "en_header", "count"), PARALLEL_TABLES, ids=[p[0] for p in PARALLEL_TABLES])
def test_both_languages_quote_the_same_numbers_in_each_table(label, zh_header, en_header, count):
    """同一張表在兩份 README 中逐列含有相同的數字（只比數字，不比文字）。"""
    zh, en = find_tables(readme("zh"), zh_header), find_tables(readme("en"), en_header)
    assert len(zh) == len(en) == count, label
    for zt, et in zip(zh, en):
        assert len(zt) == len(et), label
        for zr, er in zip(zt[1:], et[1:]):
            assert number_tokens(" ".join(zr)) == number_tokens(" ".join(er)), (label, zr[0], er[0])
    if label == "tests":  # 合計列等於各列的和（兩份 README 各自核對）
        for t in (zh[0], en[0]):
            counts = {"Rust": 0, "Python": 0}
            for row in t[1:-1]:
                kind, n = re.fullmatch(r"(Rust|Python) (\d+)", row[2]).groups()
                counts[kind] += int(n)
            assert re.findall(r"\d+", t[-1][2]) == [str(counts["Rust"]), str(counts["Python"])], t[-1]


def test_at_a_glance_matches_the_60_second_summary():
    zh = section(readme("zh"), "60 秒看懂")
    en = section(readme("en"), "At a glance")
    assert number_tokens(zh) == number_tokens(en)


REPORTS = [f"results/reports/{n}/report.md" for n in (*CORE, "quant-glove", "quant-text")]


@pytest.mark.parametrize("lang", LANGS)
def test_source_reports_are_named(lang):
    text = readme(lang)
    for path in REPORTS:
        assert f"`{path}`" in text, path
    assert "docs/results.md" in text


def slug(heading: str) -> str:
    """GitHub 的標題錨點：小寫，只保留字母、數字、標記、空白、- 與 _，空白換成 -。"""
    kept = (ch for ch in heading.strip().lower() if ch in "-_ " or unicodedata.category(ch)[0] in "LMN")
    return "".join(kept).replace(" ", "-")


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    out = set()
    for m in re.finditer(r"^#{1,6}\s+(.*?)\s*#*\s*$", strip_code(path.read_text()), re.M):
        s = slug(re.sub(r"`", "", m.group(1)))
        if s in seen:
            seen[s] += 1
            out.add(f"{s}-{seen[s]}")
        else:
            seen[s] = 0
            out.add(s)
    return out


def links(doc: str) -> list[str]:
    """文件中的相對連結（code block 之外；不含 http: 等有 scheme 的網址）。"""
    text = strip_code((ROOT / doc).read_text())
    return [t for t in re.findall(r"\]\(([^)\s]+)\)", text) if not re.match(r"[a-z][a-z0-9+.-]*:", t)]


@pytest.mark.parametrize("doc", LINKED_DOCS)
def test_relative_links_point_to_existing_files(doc):
    found = links(doc)
    assert found, doc
    base = (ROOT / doc).parent
    for target in found:
        path = target.split("#", 1)[0]
        if path:
            assert (base / path).exists(), f"{doc}: {target}"


@pytest.mark.parametrize("doc", LINKED_DOCS)
def test_anchors_point_to_existing_headings(doc):
    base = (ROOT / doc).parent
    for target in links(doc):
        path, _, frag = target.partition("#")
        if not frag:
            continue
        dest = base / path if path else ROOT / doc
        if dest.suffix == ".md":
            assert frag in anchors(dest), f"{doc}: {target}"


@pytest.mark.parametrize("lang", LANGS)
def test_language_switch_and_license(lang):
    text = readme(lang)
    switch = {"en": "**English** | [繁體中文](README.zh-TW.md)", "zh": "[English](README.md) | **繁體中文**"}[lang]
    assert switch in text
    assert "[![License: BSD-3-Clause]" in text and "](LICENSE)" in text
    assert "THIRD_PARTY_NOTICES.md" in text


# 本機路徑的前綴；heavy lock 的固定位置是 tools/with-mac-lock 寫死的共用路徑，不是個人路徑，先去掉再檢查。
LOCAL_PATH_PREFIXES = ("/Users/", "/home/", "/private/", "/tmp/", "/var/folders/", "~/Desktop", "C:\\Users\\")
SHARED_PATHS = ("/tmp/ai-portfolio-mac-heavy.lock",)
EMAIL = re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


@pytest.mark.parametrize("doc", PUBLIC_DOCS)
def test_no_local_paths_or_personal_data(doc):
    """不寫死任何帳號名稱或 email：以通用的路徑前綴、執行時的家目錄與 email 的形式檢查。"""
    text = (ROOT / doc).read_text()
    for shared in SHARED_PATHS:
        text = text.replace(shared, "")
    home = Path.home()
    for bad in (*LOCAL_PATH_PREFIXES, *([str(home)] if len(home.parts) > 2 else [])):
        assert bad not in text, (doc, bad)
    assert not EMAIL.findall(text), (doc, EMAIL.findall(text))
    # 對專案作者的稱呼：繁中寫「專案作者」、英文寫 the owner / the author。
    # docs/decisions/README.md 逐字引用 decision 的標題（其中有開發時的用語），並在該檔說明，所以不檢查用語。
    if doc != "docs/decisions/README.md":
        for word in ("使用者", "the user", "The user"):
            assert word not in text, (doc, word)


QUALIFIERS = {
    "en": [
        "does not mean the Gate passed",
        "Entering Advanced C was the owner's decision on 2026-10-16 under this special case",
        "H1a holds, H1b is undetermined, H1 overall is undetermined; H2 holds",
        "The record only proves that they did not belong to this run",
        "unrelated to Advanced C's vector quantization (SQ8)",
        "an observation, not validated by an intervention",
        "post-hoc description",
        "is a reading, not a verdict",
        "it cannot be stated as \"SQ8 is faster at 100-d\"",
        "The public repository cannot reproduce the formal numbers",
        "the owner decided to report the result as it is and not to add seeds",
        "frontier readings at n = 100,000 for the target Recall@10 ≥ 0.95",
        "`std::collections::HashSet<u32>`",
        "not the Gate evidence",
        "it only decides when to start",
    ],
    "zh": [
        "不代表 Gate 通過",
        "進入 Advanced C 是專案作者 2026-10-16 依這個特例做的決定",
        "H1a 成立、H1b 無法判定、H1 整體無法判定；H2 成立",
        "紀錄只證明它們不屬於本次 run",
        "與 Advanced C 的 vector quantization（SQ8）無關",
        "觀察，沒有以介入實驗驗證",
        "看到結果之後的描述",
        "只是讀值，不是判定",
        "不能寫成「SQ8 在 100 維上更快」",
        "無法從公開 repo 重產正式數字",
        "專案作者 2026-10-19 決定照實報告，不補 seed",
        "n = 100,000、目標 Recall@10 ≥ 0.95 的 frontier 讀值",
        "`std::collections::HashSet<u32>`",
        "不是 Gate 證據",
        "它只決定開始的時機",
    ],
}


@pytest.mark.parametrize("lang", LANGS)
def test_qualifiers_stay_in_the_readme(lang):
    text = plain(readme(lang))
    for phrase in QUALIFIERS[lang]:
        assert phrase in text, phrase


@pytest.mark.parametrize("lang", LANGS)
def test_gate_and_advanced_c_sections_say_the_gate_did_not_pass(lang):
    """Gate 總表與 Advanced C 兩節各自寫明：進入 C 是專案作者的決定，不代表 Gate 通過。"""
    gate, advc = {
        "en": ("Gate summary", "Advanced C: vector quantization"),
        "zh": ("Gate 四條件總表", "Advanced C：vector quantization"),
    }[lang]
    phrase = {"en": "does not mean the Gate passed", "zh": "不代表 Gate 通過"}[lang]
    owner = {"en": "the owner", "zh": "專案作者"}[lang]
    for title in (gate, advc):
        body = plain(section(readme(lang), title))
        assert phrase in body and owner in body, title
