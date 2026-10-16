"""Advanced C 的 10 份 formal config（decision 019 附錄、§5）：與 019 附錄逐字相同、都能解析、執行順序等於 §5 的表，
以及 report.py 的登記表參數（019 §4：Core 報告重產時忽略 C 的 run，C 的登記表拒絕 Core 的 run）。

範本、ef 清單、檔案表與 §5 的順序表都直接從 docs/decisions/019 讀出，不在測試裡另寫一份。
"""

import re
from pathlib import Path

import pytest

from benchmarks import config, quant_report
from benchmarks import report as report_module
from benchmarks.report import ReportError, check_formal_set, write_report
from test_report import RUST_CURVE, curve, make_record

REPO = Path(__file__).resolve().parents[1]
DECISION = (REPO / "docs" / "decisions" / "019-advanced-c-vector-quantization.md").read_text(encoding="utf-8")
CONFIGS = REPO / "benchmarks" / "configs"


def _appendix() -> str:
    return DECISION[DECISION.index("## 附錄：formal config（逐字登記）"):]


def registered_configs() -> dict[str, dict]:
    """019 附錄的範本代入檔案表之後的全文，依檔名。"""
    appendix = _appendix()
    template = re.search(r"```toml\n(.*?)```", appendix, re.S).group(1)
    ef_lists = dict(re.findall(r"^- `([a-z0-9-]+)`：`(\[[0-9, ]+\])`$", appendix, re.M))
    rows = re.findall(r"^\| `(quant-[a-z]+-seed-\d\.toml)` \| ([a-z0-9-]+) \| (\d) \| `(\[.*?\])` \| (\d+) \|$", appendix, re.M)
    configs = {}
    for name, dataset, seed, implementations, timeout in rows:
        text = (template.replace("<資料集>", dataset).replace("<s>", seed).replace("<implementations>", implementations)
                .replace("<ef_search>", ef_lists[dataset]).replace("<timeout>", timeout))
        configs[name] = {"dataset": dataset, "seed": int(seed), "text": text, "timeout": int(timeout)}
    return configs


def section5_orders() -> dict[tuple[str, int], tuple[str, ...]]:
    """019 §5 的表：(資料集, seed) → implementations 的順序（代號經 §2 的表換成實作名稱）。"""
    codes = dict(re.findall(r"^\| ([RFSPQ]) \| `([a-z0-9-]+)` \|", DECISION, re.M))
    section = DECISION[DECISION.index("### 5. 格點、ef 與 formal config"):DECISION.index("### 6. 讀值與判定規則")]
    glove = {int(seed): tuple(codes[c.strip()] for c in order.split(","))
             for seed, order in re.findall(r"^\| `quant-glove-seed-\d\.toml` \| (\d) \| ([RFSPQ, ]+) \|", section, re.M)}
    assert "與 GloVe 同一個 seed 的順序相同" in section
    orders = {}
    for seed, names in glove.items():
        orders[(quant_report.GLOVE, seed)] = names
        orders[(quant_report.TEXT, seed)] = names
    return orders


def test_the_appendix_is_read_completely():
    configs = registered_configs()
    assert len(configs) == 10
    assert {c["dataset"] for c in configs.values()} == {quant_report.GLOVE, quant_report.TEXT}
    assert len(section5_orders()) == 10


def test_the_shipped_configs_match_the_decision_verbatim():
    """019 附錄：內容必須與範本代入後完全相同；唯一的例外是依 §5 上調的 child_timeout_s（600 的倍數、不低於登記值）。"""
    shipped = {p.name for p in CONFIGS.glob("quant-*.toml")}
    registered = registered_configs()
    assert shipped == set(registered)
    for name, entry in registered.items():
        text = (CONFIGS / name).read_text(encoding="utf-8")
        timeout = int(re.search(r"^child_timeout_s = (\d+)$", text, re.M).group(1))
        assert timeout >= entry["timeout"] and timeout % 600 == 0, name
        normalized = re.sub(r"^child_timeout_s = \d+$", f"child_timeout_s = {entry['timeout']}", text, flags=re.M)
        assert normalized == entry["text"], name


def test_the_registry_lists_exactly_the_ten_configs():
    registered = registered_configs()
    assert {path for paths in quant_report.QUANT_CONFIGS.values() for path in paths} == {
        f"benchmarks/configs/{name}" for name in registered}
    for dataset, paths in quant_report.QUANT_CONFIGS.items():
        assert len(paths) == 5
        assert {registered[Path(p).name]["dataset"] for p in paths} == {dataset}
    # 與 Core 的登記表沒有交集：兩份報告互不合併對方的 run。
    core = {p for paths in report_module.FORMAL_CONFIGS.values() for p in paths}
    assert not core & {p for paths in quant_report.QUANT_CONFIGS.values() for p in paths}


@pytest.mark.parametrize("name", sorted(registered_configs()))
def test_each_config_parses_and_runs_in_the_registered_order(name):
    entry = registered_configs()[name]
    parsed = config.parse_config((CONFIGS / name).read_text(encoding="utf-8"))
    assert parsed.dataset == entry["dataset"]
    assert parsed.builds == (config.Build(100_000, 16, 200, entry["seed"], None),)
    # 單一 build：008 §5 的輪換位移是 0，執行順序就是 implementations 的順序（019 §5）。
    ((group, build, names),) = config.execution_order(parsed)
    assert group == 0 and names == section5_orders()[(entry["dataset"], entry["seed"])]
    assert list(parsed.ef_search) == sorted(parsed.ef_search)
    e_star = quant_report.E_STAR[entry["dataset"]]
    assert e_star in parsed.ef_search


def test_the_ef_lists_are_identical_within_each_dataset_and_cover_e_star():
    lists = {}
    for name, entry in registered_configs().items():
        lists.setdefault(entry["dataset"], set()).add(config.parse_config((CONFIGS / name).read_text(encoding="utf-8")).ef_search)
    assert lists == {quant_report.GLOVE: {(64, 96, 128, 152, 192, 256, 304, 384, 512, 768)},
                     quant_report.TEXT: {(24, 32, 48, 64, 96, 128, 192, 256, 384, 512)}}


def test_rust_seeds_and_faiss_seeds_follow_section_5():
    arms = {}
    for name, entry in registered_configs().items():
        for impl in config.parse_config((CONFIGS / name).read_text(encoding="utf-8")).implementations:
            arms.setdefault((entry["dataset"], impl), set()).add(entry["seed"])
    for dataset in (quant_report.GLOVE, quant_report.TEXT):
        assert arms[(dataset, quant_report.F)] == arms[(dataset, quant_report.S)] == set(quant_report.RUST_SEEDS)
        assert arms[(dataset, quant_report.P)] == arms[(dataset, quant_report.Q)] == set(quant_report.FAISS_SEEDS)
        assert arms[(dataset, quant_report.R)] == {quant_report.ANCHOR_SEED}


# ---- report.py 的登記表參數（019 §4） ----

GLOVE = quant_report.GLOVE


def _formal_run(directory, config_path, seed, impl="rust-hnsw", protocol="008-v2"):
    import json

    record = make_record([curve(impl, RUST_CURVE, seed=seed)], dataset=GLOVE)
    record["protocol"]["version"] = protocol
    record["metadata"]["formal_checks"] = {"config": {"path": config_path, "blob": "b" * 40}}
    record["config"] = {"path": config_path}
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return directory, record


def test_regenerating_the_core_report_ignores_c_runs(tmp_path):
    runs = [_formal_run(tmp_path / "runs" / f"core{i}", path, seed=i + 1)[0]
            for i, path in enumerate(report_module.FORMAL_CONFIGS[GLOVE])]
    # C 的 formal run 放在同一個 results/runs 底下：Core 的登記表不掃描它，報告照常產生、內容不含 C 的臂。
    for i, path in enumerate(quant_report.QUANT_CONFIGS[GLOVE]):
        _formal_run(tmp_path / "runs" / f"c{i}", path, seed=50 + i, impl=quant_report.S, protocol="008-v5")
    out = write_report(runs, tmp_path / "report", roots=[tmp_path / "runs"])
    import json

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert [c["config"] for c in report["run_configs"]] == list(report_module.FORMAL_CONFIGS[GLOVE])
    assert set(report["frontier"]) == {"rust-hnsw"}
    # 把 C 的 run 混進 Core 報告則被拒絕：協定版本不同（_check_compatible），而且不在 Core 的登記表中。
    with pytest.raises(ReportError, match="protocol"):
        write_report([*runs, tmp_path / "runs" / "c0"], tmp_path / "r2", roots=[tmp_path / "runs"])
    records = [json.loads((d / "run.json").read_text(encoding="utf-8")) for d in (*runs, tmp_path / "runs" / "c0")]
    with pytest.raises(ReportError, match="not a formal config"):
        check_formal_set(records, [*runs, tmp_path / "runs" / "c0"], [], roots=[tmp_path / "runs"])


def test_the_c_registry_rejects_core_runs(tmp_path):
    core = [_formal_run(tmp_path / "runs" / f"core{i}", path, seed=i + 1) for i, path in
            enumerate(report_module.FORMAL_CONFIGS[GLOVE])]
    c = [_formal_run(tmp_path / "runs" / f"c{i}", path, seed=50 + i, impl=quant_report.S, protocol="008-v5") for i, path in
         enumerate(quant_report.QUANT_CONFIGS[GLOVE])]
    with pytest.raises(ReportError, match="not a formal config"):
        check_formal_set([r for _, r in core[:1]], [d for d, _ in core[:1]], [], roots=[tmp_path / "runs"],
                         registry=quant_report.QUANT_CONFIGS)
    # C 的登記表只掃描 C 的 config：同一目錄下的 Core run 不影響 C 的檢查。
    check_formal_set([r for _, r in c], [d for d, _ in c], [], roots=[tmp_path / "runs"], registry=quant_report.QUANT_CONFIGS)
    with pytest.raises(ReportError, match="exactly one"):
        check_formal_set([r for _, r in c[:-1]], [d for d, _ in c[:-1]], [], roots=[tmp_path / "runs"],
                         registry=quant_report.QUANT_CONFIGS)
    # 預設仍是 Core 的登記表。
    with pytest.raises(ReportError, match="not a formal config"):
        check_formal_set([r for _, r in c], [d for d, _ in c], [], roots=[tmp_path / "runs"])


def test_the_new_arms_have_fixed_plot_styles_and_core_styles_are_unchanged():
    assert report_module.IMPL_STYLE["rust-hnsw"] == {"color": "#2a78d6", "marker": "o"}
    assert report_module.IMPL_STYLE["hnswlib"] == {"color": "#eb6834", "marker": "s"}
    assert report_module.IMPL_STYLE["faiss-hnsw"] == {"color": "#1baf7a", "marker": "^"}
    assert list(report_module.IMPL_STYLE)[:3] == ["rust-hnsw", "hnswlib", "faiss-hnsw"]
    new = [report_module.IMPL_STYLE[name] for name in (quant_report.F, quant_report.S, quant_report.Q)]
    styles = [*new, report_module.IMPL_STYLE["rust-hnsw"], report_module.IMPL_STYLE["faiss-hnsw"]]
    assert len({s["color"] for s in styles}) == 5 and len({s["marker"] for s in styles}) == 5


# ---- 真實資料的 correctness smoke（019 §9 第 6 步） ----


def section9_smoke() -> dict:
    """019 §9 第 6 步的 smoke 內容：資料集 → ef；另外確認 seed 1、(16, 200)、五個臂與 smoke 模式寫在同一步。"""
    section = DECISION[DECISION.index("6. **真實資料的 correctness smoke**"):DECISION.index("7. **formal**")]
    glove, text = re.search(r"ef 為 GloVe \{([0-9, ]+)\}、文字 \{([0-9, ]+)\}", section).groups()
    assert "(16, 200) seed 1" in section and "五個臂都跑" in section and "使用 smoke 模式" in section
    return {quant_report.GLOVE: tuple(int(x) for x in glove.split(", ")),
            quant_report.TEXT: tuple(int(x) for x in text.split(", "))}


def test_the_smoke_configs_follow_section_9_step_6():
    from benchmarks import quant_smoke

    efs = section9_smoke()
    assert quant_smoke.SMOKE_EFS == efs
    codes = dict(re.findall(r"^\| ([RFSPQ]) \| `([a-z0-9-]+)` \|", DECISION, re.M))
    registered = registered_configs()
    for dataset, path in quant_smoke.SMOKE_CONFIGS.items():
        parsed = config.parse_config((REPO / path).read_text(encoding="utf-8"))
        assert parsed.dataset == dataset and parsed.ef_search == efs[dataset]
        assert parsed.builds == (config.Build(100_000, 16, 200, 1, None),)
        assert sorted(parsed.implementations) == sorted(codes.values())  # 五個臂
        assert quant_report.E_STAR[dataset] in parsed.ef_search
        # implementations 的順序與 timeout 沿用同一個資料集的 seed 1 formal config。
        seed_1 = next(e for e in registered.values() if e["dataset"] == dataset and e["seed"] == 1)
        formal = config.parse_config(seed_1["text"])
        assert parsed.implementations == formal.implementations
        assert parsed.child_timeout_s == formal.child_timeout_s
        # smoke 的 config 不在 C 的 formal 登記表中，C 的報告不會合併它。
        assert path not in {p for paths in quant_report.QUANT_CONFIGS.values() for p in paths}
