"""docs/results.md 是本機正式報告的逐字轉錄：本測試把文件中的數字與判定和報告逐項核對。

- 報告（`results/reports/*/report.md`、`report.json`）不進版控（decision 008 §12）。乾淨的 clone 沒有報告時，
  需要報告的測試以 `pytest.skip` 跳過並說明原因；不需要報告的檢查（例如文件中沒有本機路徑）照常執行。
- 文件中每張被核對的表前面有 `<!-- check:<名稱> -->` 標記。報告與文件不一致時，以報告為準（報告已封版）。
- 只讀檔，不產生任何讀值。
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "results.md"
REPORTS = ROOT / "results" / "reports"
RUNS = ROOT / "results" / "runs"
CORE = ("glove-100-angular", "simplewiki-embeddinggemma2-100k")
QUANT = ("quant-glove", "quant-text")
QUANT_DATASET = {"quant-glove": "glove-100-angular", "quant-text": "simplewiki-embeddinggemma2-100k"}
QUANT_LABEL = {"quant-glove": "GloVe", "quant-text": "文字"}
TARGETS = ("0.90", "0.95", "0.99")
PARAM = re.compile(r"@ M=(\d+), efC=(\d+), ef=(\d+)")
MIB = 1024 * 1024


# ---------------------------------------------------------------- 文件與報告的讀取


def doc_text() -> str:
    return DOC_PATH.read_text()


def _cells(line: str) -> list[str]:
    inner = line.strip()[1:-1]
    return [c.strip().replace("\\|", "|").replace("\\*", "*") for c in re.split(r"(?<!\\)\|", inner)]


def doc_table(name: str) -> list[list[str]]:
    """`<!-- check:name -->` 之後的第一張表（不含表頭與分隔列）。"""
    lines = doc_text().splitlines()
    tag = f"<!-- check:{name} -->"
    assert lines.count(tag) == 1, f"docs/results.md 應該剛好有一個 {tag}"
    i = lines.index(tag) + 1
    rows = []
    while i < len(lines) and not lines[i].startswith("|"):
        i += 1
    while i < len(lines) and lines[i].startswith("|"):
        rows.append(_cells(lines[i]))
        i += 1
    assert len(rows) >= 2, name
    return rows[2:]


def doc_block(name: str) -> list[str]:
    """`<!-- check:name -->` 之後、下一個空行之前的文字行。"""
    lines = doc_text().splitlines()
    i = lines.index(f"<!-- check:{name} -->") + 1
    out = []
    while i < len(lines) and lines[i].strip():
        out.append(lines[i].replace("\\*", "*"))
        i += 1
    return out


def need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"本機沒有 {path.relative_to(ROOT)}：results/ 不進版控（decision 008 §12），乾淨的 clone 無法核對轉錄")
    return path


def report_json(name: str) -> dict:
    return json.loads(need(REPORTS / name / "report.json").read_text())


def report_md(name: str) -> tuple[str, dict[str, list[list[list[str]]]]]:
    """回傳 (全文, {章節標題: [表, ...]})；每張表不含分隔列，第一列是表頭。"""
    text = need(REPORTS / name / "report.md").read_text()
    sections: dict[str, list] = {}
    current, table = "", None
    for line in text.splitlines():
        if line.startswith("#"):
            current = line.lstrip("#").strip()
            sections.setdefault(current, [])
            table = None
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip()[1:-1].split("|")]
            if all(re.fullmatch(r"-+", c) for c in cells if c):
                continue
            if table is None:
                table = []
                sections.setdefault(current, []).append(table)
            table.append(cells)
        else:
            table = None
    return text, sections


def md_table(name: str, prefix: str, index: int = 0) -> list[list[str]]:
    _, sections = report_md(name)
    hits = [k for k in sections if k.startswith(prefix)]
    assert len(hits) == 1, (name, prefix, hits)
    return sections[hits[0]][index][1:]


def run_json(run: str) -> dict:
    return json.loads(need(RUNS / run / "run.json").read_text())


def qps0(cell: str) -> str:
    return cell.split(" @ ")[0]


# ---------------------------------------------------------------- 不需要報告的檢查


def test_doc_has_no_local_paths_or_personal_data():
    """不寫死任何帳號名稱或 email：以通用的路徑前綴、執行時的家目錄與 email 的形式檢查。"""
    text = doc_text()
    home = Path.home()
    for bad in ("/Users/", "/home/", "/private/", "/tmp/", "/var/folders/", *([str(home)] if len(home.parts) > 2 else [])):
        assert bad not in text, bad
    emails = re.findall(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}", text)
    assert not emails, emails
    assert "使用者" not in text  # 對專案作者的稱呼一律寫「專案作者」


def test_doc_marks_post_hoc_descriptions_and_owner_decision():
    text = doc_text()
    assert "看到結果之後的描述，**不是判定**" in text
    assert "這**不代表 Gate 通過**" in text
    assert "進入 Advanced C 是專案作者的決定" in text
    assert "紀錄只證明它們不屬於本次 run" in text
    assert "只剩 seeds 2、3 兩個乾淨配對" in text
    assert "與 Advanced C 的 vector quantization（SQ8）無關" in text


# ---------------------------------------------------------------- 來源


SHA_ROWS = [(n, f) for n in CORE + QUANT for f in ("report.md", "report.json")]


@pytest.mark.parametrize(("name", "fname"), SHA_ROWS)
def test_sha256_of_local_reports(name, fname):
    listed = {r[0].strip("`"): r[1].strip("`") for r in doc_table("sha256")}
    rel = f"results/reports/{name}/{fname}"
    assert rel in listed
    digest = hashlib.sha256(need(ROOT / rel).read_bytes()).hexdigest()
    assert listed[rel] == digest, f"{rel} 與 docs/results.md 列出的 SHA-256 不同：文件轉錄的不是這份報告"


def test_run_list_matches_reports():
    rows = doc_table("runs")
    for name in CORE:
        text, _ = report_md(name)
        merged = re.findall(r"^- results/runs/(\S+)：(\w+)，config (\S+)（blob \w+）", text, re.M)
        doc_rows = [(r[1].strip("`"), r[2], r[3].strip("`")) for r in rows if r[0] == name]
        assert doc_rows == merged
        j = report_json(name)
        assert [r[0] for r in doc_rows] == [Path(p).name for p in j["runs"]]
        assert all(r[4] == j["protocol"] for r in rows if r[0] == name)
    for name in QUANT:
        text, _ = report_md(name)
        part = text.split("## 合併的 run", 1)[1].split("##", 1)[0]
        merged = re.findall(r"^- results/runs/(\S+)", part, re.M)
        doc_rows = [r for r in rows if r[0] == name]
        assert [r[1].strip("`") for r in doc_rows] == merged
        j = report_json(name)
        assert [r[1].strip("`") for r in doc_rows] == [Path(p).name for p in j["datasets"][QUANT_DATASET[name]]["runs"]]
        assert all(r[4] == j["protocol"] for r in doc_rows)


def test_run_list_commits_and_configs_match_run_records():
    for r in doc_table("runs"):
        rec = run_json(r[1].strip("`"))
        assert rec["status"] == r[2]
        assert rec["config"]["path"] == r[3].strip("`")
        assert rec["protocol"]["version"] == r[4]
        assert rec["metadata"]["git"]["commit"].startswith(r[5].strip("`"))
        assert rec["metadata"]["git"]["status"] == ""


# ---------------------------------------------------------------- Core


@pytest.mark.parametrize("name", CORE)
def test_core_frontier_is_verbatim(name):
    report = md_table(name, "Frontier（主要讀值）")
    expected = [[r[0], r[2], r[3], r[4], r[5], r[6], r[8]] for r in report if r[1] == "100000"]
    assert doc_table(f"core-frontier:{name}") == expected
    assert len(expected) == 9
    flagged = [[r[0], r[2], r[3], r[4], r[5], r[6]] for r in md_table(name, "包含被標記點的讀值") if r[1] == "100000"]
    assert doc_table(f"core-frontier-flagged:{name}") == flagged


@pytest.mark.parametrize("name", CORE)
def test_core_frontier_qps_matches_report_json(name):
    frontier = report_json(name)["frontier"]
    for impl, target, qps, *_ in doc_table(f"core-frontier:{name}"):
        main = frontier[impl]["100000"][target]["main"]
        if not main["reached"]:
            assert qps == "未達到"
            continue
        q = main["qps"]
        head = f"{q['value']:.0f}"
        if q["builds"] >= 2:
            head += f" [{q['min']:.0f}, {q['max']:.0f}]（{q['builds']} builds）"
        assert qps == f"{head} @ M={q['m']}, efC={q['ef_construction']}, ef={q['ef_search']}"


@pytest.mark.parametrize("name", CORE)
def test_core_recall_at_frontier_points(name):
    frontier = {(r[0], r[1]): r[2] for r in doc_table(f"core-frontier:{name}")}
    seed_variance = md_table(name, "seed variance")
    by_point = {(r["implementation"], r["m"], r["ef_construction"], r["ef_search"]): r
                for r in report_json(name)["seed_variance"] if r["n"] == 100000}
    rows = doc_table(f"core-recall:{name}")
    assert len(rows) == 9
    for impl, target, m, efc, ef, builds, r10, r1 in rows:
        assert PARAM.search(frontier[(impl, target)]).groups() == (m, efc, ef)
        hit = [s for s in seed_variance if s[0] == impl and s[1] == "100000" and s[2:5] == [m, efc, ef]]
        assert len(hit) == 1 and hit[0][5:8] == [builds, r10, r1]
        j = by_point[(impl, int(m), int(efc), int(ef))]
        for cell, key in ((r10, "recall10"), (r1, "recall1")):
            s = j[key]
            assert cell == f"{s['median']:.4f} [{s['min']:.4f}, {s['max']:.4f}]"


@pytest.mark.parametrize("name", CORE)
def test_core_qps_ratios(name):
    ratios = report_json(name)["ratios"]
    rows = doc_table(f"core-ratios:{name}")
    assert len(rows) == sum(len(v) for v in ratios.values())
    for n, target, faiss, hnswlib in rows:
        for cell, impl in ((faiss, "faiss-hnsw"), (hnswlib, "hnswlib")):
            v = ratios[n][target][impl]
            assert cell == ("—" if v is None else f"{v:.2f}")


@pytest.mark.parametrize("name", CORE)
def test_core_gate_readings(name):
    gate = report_json(name)["gate"]
    report = {r[0]: r for r in md_table(name, "Gate 讀值")}
    rows = doc_table(f"core-gate:{name}")
    assert [r[0] for r in rows] == list(TARGETS)
    for target, primary, rust, hnsw, seeds, verdict, all_builds in rows:
        g = gate[target]
        assert primary == ("是" if g["primary"] else "—")
        for cell, impl in ((rust, "rust-hnsw"), (hnsw, "hnswlib")):
            b = g["main"]["brackets"][impl]
            assert cell == f"[{b['lower']:.0f}, {b['upper']:.0f}]"
        assert verdict == f"`{g['main']['verdict']}`：{report[target][5]}"
        assert [rust, hnsw, seeds, all_builds] == [report[target][2], report[target][3], report[target][4], report[target][6]]
        assert (all_builds == "相同") == (not g["differs"])


@pytest.mark.parametrize("name", CORE)
def test_core_seed_ranges_are_verbatim(name):
    assert doc_table(f"core-seed-ranges:{name}") == md_table(name, "主要結論附上的 seed 範圍")
    assert doc_table(f"core-seed-ranges-flagged:{name}") == md_table(name, "包含被標記 build 的 seed 範圍")


@pytest.mark.parametrize("name", CORE)
def test_core_insertion_order_variance(name):
    report = md_table(name, "insertion order variance")
    rows = doc_table(f"core-order:{name}")
    assert len(rows) * 3 == len(report)
    for ef, rust, hnsw, faiss in rows:
        cells = {r[0]: r for r in report if r[4] == ef}
        assert all(cells[i][5] == "1" and cells[i][6] == "[None, 101, 102, 103, 104, 105]" for i in cells)
        assert [rust, hnsw, faiss] == [cells["rust-hnsw"][7], cells["hnswlib"][7], cells["faiss-hnsw"][7]]


@pytest.mark.parametrize("name", CORE)
def test_core_recall_vs_ef_has_no_deterioration(name):
    curves = report_json(name)["recall_vs_ef"]
    assert all(not c["systematic_deterioration"] and c["max_adjacent_drop"] == 0.0 for c in curves)
    (line,) = doc_block(f"core-recall-vs-ef:{name}")
    cells = sorted({(c["n"], c["m"], c["ef_construction"]) for c in curves})
    assert f"共 {len(curves)} 列（3 個實作 × {len(cells)} 個 (n, M, efC)）" in line
    assert "、".join(f"({a}, {b}, {c})" for a, b, c in cells) in line


@pytest.mark.parametrize("name", CORE)
def test_core_scaling_rows(name):
    expected = [[r[0], r[1], r[4], r[5], r[6], r[8], r[9]] for r in md_table(name, "scaling") if r[2:4] == ["16", "200"]]
    assert doc_table(f"core-scaling:{name}") == expected


OTHER_N = {"glove-100-angular": {"300000", "1183514"}, "simplewiki-embeddinggemma2-100k": {"10000", "30000"}}


@pytest.mark.parametrize("name", CORE)
def test_core_other_n_and_flagged_readings(name):
    flagged = {(r[0], r[1], r[2]): r[3] for r in md_table(name, "包含被標記點的讀值")}
    expected = [[r[0], r[1], r[2], r[3], r[8], flagged.get((r[0], r[1], r[2]), "（與主要讀值相同）")]
                for r in md_table(name, "Frontier（主要讀值）") if r[1] in OTHER_N[name]]
    assert doc_table(f"core-other-n:{name}") == expected


@pytest.mark.parametrize("name", CORE)
def test_core_flagged_points_grouping(name):
    points = report_json(name)["flagged_points"]
    assert f"被標記的點共 {len(points)} 個" in doc_text()
    expected = {(p["implementation"], p["n"], p["m"], p["ef_construction"], p["seed"], p["shuffle"], p["ef_search"],
                 tuple(p["flags"])) for p in points}
    seen = set()
    for impl, n, cell, seed, shuffle, parts in doc_table(f"core-flagged:{name}"):
        m, efc = map(int, re.fullmatch(r"\((\d+), (\d+)\)", cell).groups())
        shuf = None if shuffle == "原始" else int(shuffle)
        for part in parts.split("；"):
            efs, flags = re.fullmatch(r"ef ([\d、]+)：(.+)", part).groups()
            for ef in efs.split("、"):
                key = (impl, int(n), m, efc, int(seed), shuf, int(ef), tuple(f.strip("`") for f in flags.split(", ")))
                assert key not in seen
                seen.add(key)
    assert seen == expected


@pytest.mark.parametrize("name", CORE)
def test_core_excluded_runs(name):
    excluded = report_json(name)["excluded_runs"]
    rows = doc_table(f"core-excluded:{name}")
    assert [(r[0].strip("`"), r[1]) for r in rows] == [(Path(e["run"]).name, e["status"]) for e in excluded]
    if name == "glove-100-angular":
        assert [r[2] for r in rows] == ["；".join(e["status_reasons"]) for e in excluded]
    else:
        assert [r[2] for r in rows] == ["作廢（voided）" if e.get("voided") else "排除" for e in excluded]
        for r, e in zip(rows, excluded):
            if e.get("voided"):
                assert e["voided"].split(":")[0] in r[3]


def test_gate_summary_matches_reports():
    rows = {r[0][:2]: r for r in doc_table("gate-summary")}
    glove, text = report_json("glove-100-angular"), report_json("simplewiki-embeddinggemma2-100k")
    g = {t: glove["gate"][t]["main"]["verdict"] for t in TARGETS}
    x = {t: text["gate"][t]["main"]["verdict"] for t in TARGETS}
    cond3 = rows["3."]
    assert f"GloVe `{g['0.95']}`、文字 `{x['0.95']}`" in cond3[1]
    assert f"GloVe 0.90 `{g['0.90']}`、0.99 `{g['0.99']}`" in cond3[1]
    assert x["0.90"] == x["0.99"] and f"文字 0.90、0.99 都是 `{x['0.90']}`" in cond3[1]
    # 015：只有主要目標判為 gap 的資料集才算「有差距」。
    assert g["0.95"] != "gap" and x["0.95"] != "gap" and cond3[2] == "**不成立**"
    curves = glove["recall_vs_ef"] + text["recall_vs_ef"]
    assert all(not c["systematic_deterioration"] for c in curves) and rows["2."][2] == "成立"
    assert rows["1."][2] == "成立" and rows["4."][2] == "成立"


# ---------------------------------------------------------------- Advanced C


def judgement() -> dict:
    a, b = report_json("quant-glove")["judgement"], report_json("quant-text")["judgement"]
    assert a == b, "兩份 C 報告的判定表應該相同（019 §6）"
    return a


def test_quant_judgement_rows_and_statuses():
    rows = judgement()["rows"]
    doc = doc_table("quant-judgement")
    assert [(r[0], r[1], r[2]) for r in doc] == [(r["row"], r["status"], r["note"]) for r in rows]
    assert f"列出報告的全部 {len(rows)} 列" in doc_text()
    for name in QUANT:
        assert [r[:3] for r in md_table(name, "完整判定表")] == [r[:3] for r in doc]


def test_quant_headline_verdicts():
    status = {r["row"]: r["status"] for r in judgement()["rows"]}
    assert status["H1 整體"] == "無法判定" and status["H1a 文字"] == "成立"
    assert status["H1b GloVe"] == status["H1c 維度對比"] == status["H1d profiler（seed 1）"] == "無法判定"
    assert status["H2 GloVe"] == status["H2 文字"] == status["H2 整體"] == "成立"
    text = doc_text()
    assert "**H1 整體：無法判定**。H1a（文字）**成立**；H1b（GloVe）**無法判定**" in text
    assert "H1d **無法判定**：GloVe seed 1 的 S 在 e\\* 有 `background` 標記" in text
    assert "**H2 整體：成立**（GloVe 成立、文字成立）" in text


def test_quant_judgement_numbers():
    nums = {r["row"]: r["numbers"] for r in judgement()["rows"]}
    doc = {r[0]: r[3] for r in doc_table("quant-judgement")}
    for row in ("H1a 文字", "H1b GloVe"):
        for v in nums[row]["rho_eval"]["values"]:
            assert f"{v:.4f}" in doc[row], (row, v)
        assert f"乾淨配對{'只有 ' if row == 'H1b GloVe' else ' '}{nums[row]['clean_pairs']} 個" in doc[row]
    assert f"κ = {nums['H1d profiler（seed 1）']['glove']['kappa']:.4f}" in doc["H1d profiler（seed 1）"]
    for row in ("H2 GloVe", "H2 文字"):
        assert f"median L_total {nums[row]['median_L_total']}" in doc[row]
    for row, n in nums.items():
        if row.startswith("C1"):
            assert f"Δ_theory {n['theory_bytes']} B，容許 {n['tolerance_bytes']:.0f} B" in doc[row]
            for p in n["per_seed"]:
                assert (f"seed {p['seed']} 偏差 {p['deviation']} B 超出" in doc[row]) == (not p["within"])
        elif row.startswith("C3"):
            assert f"天花板 c = {n['ceiling']}，最大 Recall@10 = {n['max_recall10']}" in doc[row]
        elif row.startswith("C4"):
            assert f"= {n['qps_f_over_r']}" in doc[row]
        elif row.startswith("C7"):
            assert doc[row] == f"逐位元相同 {n['identical']} 個，不相同 {len(n['not_identical'])} 個"
        elif row.startswith("C5"):
            for k in ("traversal", "visited_set", "heap"):
                assert str(n["gate_shares"][k]) in doc[row]
    checks = doc_table("quant-checks")
    assert checks == [[r[0], r[1], r[3]] for r in doc_table("quant-judgement") if re.match(r"C[2-8]", r[0])]


@pytest.mark.parametrize("name", QUANT)
def test_quant_h1_pairs(name):
    rows = doc_table(f"quant-h1:{name}")
    assert rows == md_table(name, "H1")
    pairs = report_json(name)["datasets"][QUANT_DATASET[name]]["h1"]["pairs"]
    for r, p in zip(rows, pairs, strict=True):
        assert r == [str(p["seed"]), "是" if p["clean"] else "否", ", ".join(p["flags"]) or "—",
                     f"{p['rho_qps']:.4f}", f"{p['d_ratio']:.4f}", f"{p['rho_eval']:.4f}", "是" if p["usable"] else "否"]
    text, _ = report_md(name)
    (line,) = re.findall(r"^- (H1d（seed 1）：.*)$", text, re.M)
    assert f"- {line}" in doc_text().replace("\\*", "*")


def test_quant_glove_clean_pairs_and_flagged_rho():
    pairs = report_json("quant-glove")["datasets"]["glove-100-angular"]["h1"]["pairs"]
    assert [p["seed"] for p in pairs if p["clean"]] == [2, 3]
    flagged = "、".join(f"{p['rho_eval']:.4f}" for p in pairs if not p["clean"])
    assert f"seeds 1、4、5）的 ρ_eval 為 {flagged}" in doc_text()


@pytest.mark.parametrize("name", QUANT)
def test_quant_h2_per_seed(name):
    rows = doc_table(f"quant-h2:{name}")
    assert rows == md_table(name, "H2")
    seeds = report_json(name)["datasets"][QUANT_DATASET[name]]["h2"]["seeds"]
    for r, s in zip(rows, seeds, strict=True):
        assert int(r[0]) == s["seed"] and (r[1] == "是") == s["valid"]
        for cell, key in zip(r[3:7], ("L_total", "L_trav", "L_rank", "delta_F")):
            assert cell == f"{s['exact'][key]:.5f}"


def test_quant_c1_per_seed_deviations():
    expected = []
    for r in judgement()["rows"]:
        if r["row"].startswith("C1"):
            n = r["numbers"]
            for p in n["per_seed"]:
                expected.append([r["row"].replace("C1 記憶體 ", ""), str(p["seed"]), str(p["delta_relief"]), str(p["deviation"]),
                                 f"{n['tolerance_bytes']:.0f}", "是" if p["within"] else "**否**",
                                 "、".join(f"`{f}`" for f in p["flags"]) or "—", str(p["delta_m014"])])
    rows = doc_table("quant-c1")
    assert rows == expected
    outside = [(r[0], r[1], r[3]) for r in rows if r[5] != "是"]
    assert outside == [("glove F − S", "5", "4030488")]
    assert "只有 seed 5 偏離 4030488 B（換算約 3.84 MiB），超過 max(5% × Δ_theory, 2 MiB) = 2097152 B" in doc_text()


@pytest.mark.parametrize("name", QUANT)
def test_quant_frontier_is_verbatim(name):
    rows = doc_table(f"quant-frontier:{name}")
    assert rows == md_table(name, "讀值 1", 0)
    assert len(rows) == 15
    frontier = report_json(name)["datasets"][QUANT_DATASET[name]]["frontier"]
    for impl, target, qps, *_ in rows:
        main = frontier[impl.replace("（錨點）", "")]["100000"][target]["main"]
        if not main["reached"]:
            assert qps == "未達到"
            continue
        q = main["qps"]
        assert qps.startswith(f"{q['value']:.0f}") and qps.endswith(f"@ M=16, efC=200, ef={q['ef_search']}")


def test_quant_frontier_flagged_and_tables_are_verbatim():
    assert doc_table("quant-frontier-flagged:quant-glove") == md_table("quant-glove", "讀值 1", 1)
    for name, ratios, recall, qf in (("quant-glove", 3, 2, 4), ("quant-text", 2, 1, 3)):
        label = QUANT_LABEL[name]
        assert doc_table(f"quant-ratios:{name}") == [[label] + r for r in md_table(name, "讀值 1", ratios)]
        e_star = str(report_json(name)["datasets"][QUANT_DATASET[name]]["e_star"])
        assert doc_table(f"quant-recall:{name}") == [[label] + r for r in md_table(name, "讀值 1", recall) if r[1] == e_star]
        assert doc_table(f"quant-qf:{name}") == md_table(name, "讀值 1", qf)


def test_quant_frontier_verdicts_match_judgement():
    status = {r["row"]: r["status"] for r in judgement()["rows"]}
    for name, key in (("quant-glove", "glove"), ("quant-text", "text")):
        for pair, target, *_, verdict, _ in doc_table(f"quant-qf:{name}"):
            assert status[f"quantization frontier {key} {pair} {target}"] == verdict


def test_quant_tradeoff_numbers_come_from_the_frontier():
    status = {r["row"]: r["status"] for r in judgement()["rows"]}
    for row in doc_table("quant-tradeoff"):
        name = "quant-text" if row[0] == "文字" else "quant-glove"
        cells = {(r[0], r[1]): r for r in md_table(name, "讀值 1", 0)}
        ratios = {(r[0], r[1]): r for r in md_table(name, "讀值 1", 3 if name == "quant-glove" else 2)}
        f, s = cells[("rust-hnsw-fp32g", row[1])], cells[("rust-hnsw-sq8", row[1])]
        assert row[2] == f"{qps0(f[2])}（ef {PARAM.search(f[2]).group(3)}）"
        assert row[3] == f"{qps0(s[2])}（ef {PARAM.search(s[2]).group(3)}）"
        assert row[4:6] == [qps0(f[5]), qps0(s[5])]
        r = ratios[("S 對 F", row[1])]
        assert row[6] == f"{r[2]} / {r[5]}"
        key = "text" if name == "quant-text" else "glove"
        assert row[7] == f"`{status[f'quantization frontier {key} S 對 F {row[1]}']}`"
    text = doc_text()
    assert "S 1769 對 F 2277 QPS，M_014 從 318.8 降到 99.0 MiB" in text
    assert "S 2299 對 F 1877 QPS，M_014 從 105.4 降到 76.1 MiB" in text


@pytest.mark.parametrize("name", QUANT)
def test_quant_profiling_is_verbatim(name):
    assert doc_table(f"quant-profile:{name}") == md_table(name, "profiling")
    text, _ = report_md(name)
    for line in re.findall(r"^- (rust-hnsw-(?:fp32g|sq8) ef \d+ 每 query.*)$", text, re.M):
        assert f"- {line}" in doc_text()


@pytest.mark.parametrize("name", QUANT)
def test_quant_memory_summary(name):
    memory = md_table(name, "記憶體")
    builds = report_json(name)["datasets"][QUANT_DATASET[name]]["builds"]
    rows = doc_table(f"quant-memory:{name}")
    assert sorted(r[1] for r in rows) == sorted({b["implementation"] for b in builds})
    for row in rows:
        arm = [r for r in memory if r[0] == row[1]]
        assert row[2:7] == [" / ".join(r[i] for r in arm) for i in (1, 2, 3, 4, 8)]
        js = [b for b in builds if b["implementation"] == row[1]]
        assert row[3] == " / ".join(f"{b['footprint'] / MIB:.1f}" for b in js)
        assert row[6] == " / ".join(f"{b['peak_rss'] / MIB:.1f}" for b in js)
        flags = "、".join(f"seed {b['seed']} `{', '.join(b['flags'])}`" for b in js if b["flags"]) or "—"
        assert row[7] == flags


def test_quant_delta_m_is_verbatim():
    expected = []
    for name in QUANT:
        for pair in ("S 對 F", "Q 對 P"):
            expected += [[QUANT_LABEL[name], pair] + r for r in md_table(name, pair, 1)]
    assert doc_table("quant-delta-m") == expected


def test_quant_hygiene_matches_run_records():
    code = {"rust-hnsw-fp32g": "F", "rust-hnsw-sq8": "S", "faiss-hnsw": "P", "faiss-hnsw-sq8": "Q", "rust-hnsw": "R"}
    expected = []
    for name in QUANT:
        for run in report_json(name)["datasets"][QUANT_DATASET[name]]["runs"]:
            for c in run_json(Path(run).name)["children"]:
                points = [(p["ef_search"], p["flags"]) for p in c["points"] if p["flags"]]
                if not (c["flags"] or c["build"]["flags"] or points):
                    continue
                assert name == "quant-glove", "文字的 C run 應該沒有任何標記"
                expected.append([str(c["build"]["seed"]), f"{code[c['implementation']]}（`{c['implementation']}`）",
                                 "、".join(f"`{f}`" for f in c["flags"]) or "—",
                                 "、".join(f"`{f}`" for f in c["build"]["flags"]) or "—",
                                 "、".join(f"ef {ef} `{'`, `'.join(fl)}`" for ef, fl in points) or "—"])
    expected.sort(key=lambda r: int(r[0]))
    assert [r[:5] for r in doc_table("quant-hygiene")] == expected


def test_quant_hygiene_matches_report_flags():
    rows = doc_table("quant-hygiene")
    builds = report_json("quant-glove")["datasets"]["glove-100-angular"]["builds"]
    flagged_builds = {(b["implementation"], b["seed"]) for b in builds if b["flags"]}
    doc_builds = {(re.search(r"`(.+)`", r[1]).group(1), int(r[0])) for r in rows if r[2] != "—" or r[3] != "—"}
    assert doc_builds == flagged_builds
    sq8_seed5 = [r for r in rows if r[0] == "5" and "rust-hnsw-sq8" in r[1]]
    assert len(sq8_seed5) == 1 and "ef 96" not in sq8_seed5[0][4]
    assert [int(e) for e in re.findall(r"ef (\d+)", sq8_seed5[0][4])] == [64, 128, 152, 192, 256, 304]
    assert all(not b["flags"] for b in report_json("quant-text")["datasets"][QUANT_DATASET["quant-text"]]["builds"])


def test_quant_expectations_are_verbatim():
    text, _ = report_md("quant-glove")
    part = text.split("## 附帶預期", 1)[1].split("##", 1)[0]
    lines = [f"- {x}" for x in re.findall(r"^- (.+→ 符合)$", part, re.M)]
    assert doc_block("quant-expectations") == lines
    assert all(e["met"] for e in judgement()["expectations"]) and len(lines) == len(judgement()["expectations"])
