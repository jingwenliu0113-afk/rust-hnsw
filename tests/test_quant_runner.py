"""runner 的 008-v5 改動（decision 019 §4）：`serialized_bytes` 可以為 null，以及子 process 最後的 relief 診斷。

- adapter 沒有 save_native 時，run.json 寫出 null；runner 的其他欄位與報告照常處理。
- relief 在所有計時點與 save_native 之後才做（呼叫序列），之後才記錄 after_relief；
  build 紀錄的 `index_*_relieved_bytes` 相對 baseline，不影響主要指標與 hygiene 標記。
- 全部是 synthetic 資料（smoke-gaussian 與小型隨機向量）；數字不作任何結論。
"""

import ctypes
import json

import numpy as np
import pytest

from benchmarks import adapters, protocol, run_child
from benchmarks import run as run_module
from benchmarks.interface import prepare_vectors
from benchmarks.report import write_report
from benchmarks.run import run
from stub_adapters import NoSaveStub, TracingNoSaveStub, TracingStub, _log
from test_runner import STUBS, stub_config

QUANT_STUBS = {
    **STUBS,
    "no-save-stub": ("tests.stub_adapters:NoSaveStub", "numpy"),
    "tracing-no-save-stub": ("tests.stub_adapters:TracingNoSaveStub", "numpy"),
}
MEMORY_KEYS = {"rss_bytes", "ru_maxrss_bytes", "phys_footprint_bytes", "reusable_bytes", "compressed_bytes",
               "compressed_lifetime_bytes"}
MiB = 2**20


def test_the_stubs_model_adapters_without_save_native():
    # 前提：stub 與 019 的 F、S adapter 一樣，類別上沒有 save_native；有紀錄的 stub 才有。
    for cls in (NoSaveStub, TracingNoSaveStub, adapters.RustHnswFp32g, adapters.RustHnswSq8):
        assert not hasattr(cls, "save_native"), cls
    for cls in (TracingStub, adapters.RustHnsw, adapters.FaissHnsw, adapters.FaissHnswSq8):
        assert callable(getattr(cls, "save_native")), cls


# ---- serialized_bytes = null（019 §4 第 1 點） ----


@pytest.fixture(scope="module")
def stub_run(tmp_path_factory):
    return run(stub_config(["no-save-stub", "exact-stub"], ef_search=(10, 20)), "smoke",
               tmp_path_factory.mktemp("quant-runner"), implementations=QUANT_STUBS)


def test_an_adapter_without_save_native_records_null_serialized_bytes(stub_run):
    record = stub_run.record
    assert record["status"] == "complete" and stub_run.exit_code == 0
    by_name = {child["implementation"]: child for child in record["children"]}
    without, with_save = by_name["no-save-stub"], by_name["exact-stub"]
    assert without["status"] == "ok" and not without["partial"]
    assert without["build"]["serialized_bytes"] is None
    assert with_save["build"]["serialized_bytes"] > 0
    # null 不影響其他欄位：每個 ef 點都量完，記憶體與高水位照常記錄。
    for child in (without, with_save):
        assert [point["ef_search"] for point in child["points"]] == [10, 20]
        assert child["build"]["peak_rss_after_search_bytes"] >= child["build"]["peak_rss_bytes"]
    # run.json 寫出的是 JSON 的 null（不是 0、不是缺欄位）。
    on_disk = json.loads((stub_run.run_dir / "run.json").read_text(encoding="utf-8"))
    builds = {child["implementation"]: child["build"] for child in on_disk["children"]}
    assert "serialized_bytes" in builds["no-save-stub"] and builds["no-save-stub"]["serialized_bytes"] is None


def test_relieved_memory_is_recorded_relative_to_the_baseline(stub_run):
    for child in stub_run.record["children"]:
        build = child["build"]
        baseline, relieved = build["baseline"], build["after_relief"]
        assert set(relieved) == MEMORY_KEYS == set(build["after_build"])
        assert build["index_footprint_relieved_bytes"] == relieved["phys_footprint_bytes"] - baseline["phys_footprint_bytes"]
        assert build["index_memory_relieved_bytes"] == relieved["rss_bytes"] - baseline["rss_bytes"]
        # 主要指標（014）的定義不變：仍是 build 之後的讀數。
        assert build["index_footprint_bytes"] == build["after_build"]["phys_footprint_bytes"] - baseline["phys_footprint_bytes"]
        assert build["index_memory_bytes"] == build["after_build"]["rss_bytes"] - baseline["rss_bytes"]


def test_the_run_records_protocol_008_v5_with_the_relief_timing(stub_run):
    assert stub_run.record["protocol"]["version"] == "008-v5"
    assert stub_run.record["protocol"]["memory_relief"] == protocol.MEMORY_RELIEF


def test_the_report_handles_null_serialized_bytes(stub_run, tmp_path):
    out = write_report([stub_run.run_dir], tmp_path / "report")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "smoke" and report["protocol"] == "008-v5"
    assert {row["implementation"] for row in report["scaling"]} == {"no-save-stub", "exact-stub"}
    assert report["frontier"]["no-save-stub"]["300"]["0.99"]["all_points"]["reached"] is True
    assert (out / "report.md").stat().st_size > 0


def child_lines(after_relief, serialized_bytes):
    """手寫的子 process 紀錄：baseline、build 之後與 relief 之後的讀數都不同，才分得出欄位取自哪一個時點。"""
    baseline = {"rss_bytes": 100 * MiB, "ru_maxrss_bytes": 100 * MiB, "phys_footprint_bytes": 100 * MiB}
    after_build = {"rss_bytes": 150 * MiB, "ru_maxrss_bytes": 200 * MiB, "phys_footprint_bytes": 140 * MiB}
    return [{"type": "prelude", "baseline": baseline},
            {"type": "build", "build_ns": 1, "other_thread_cpu": {"other_s": 0.0, "process_s": 1.0},
             "peak_rss_bytes": 200 * MiB, "after_build": after_build, "loadavg_start": [1.0, 0, 0],
             "loadavg_end": [1.0, 0, 0]},
            {"type": "end", "peak_rss_after_search_bytes": 210 * MiB, "serialized_bytes": serialized_bytes,
             "after_relief": after_relief}]


@pytest.mark.parametrize("serialized_bytes", [None, 12345])
def test_child_record_takes_the_relieved_fields_from_the_end_line(serialized_bytes):
    # relief 之後 phys_footprint 比 rss 多 20 MiB：若 hygiene 標記誤用這兩個讀數，就會標記 memory（008 §8）。
    after_relief = {"rss_bytes": 110 * MiB, "ru_maxrss_bytes": 210 * MiB, "phys_footprint_bytes": 130 * MiB}
    build, parts = run_module._child_record(child_lines(after_relief, serialized_bytes), {"n": 300})
    assert build["serialized_bytes"] == serialized_bytes
    assert build["after_relief"] == after_relief
    assert build["index_footprint_relieved_bytes"] == 30 * MiB
    assert build["index_memory_relieved_bytes"] == 10 * MiB
    # 主要指標與 hygiene 標記不受 relief 影響（019 §4、§6）。
    assert build["index_footprint_bytes"] == 40 * MiB and build["index_memory_bytes"] == 50 * MiB
    assert build["flags"] == []
    assert parts["end"]["after_relief"] == after_relief


def test_a_child_without_an_end_line_has_no_relieved_fields():
    # relief 之前就結束的子 process：沒有 end 行，也就沒有 after_relief（runner 另記 missing_output）。
    build, _ = run_module._child_record(child_lines({}, None)[:2], {"n": 300})
    assert "after_relief" not in build and "index_footprint_relieved_bytes" not in build
    assert "serialized_bytes" not in build


# ---- relief 的時點（019 §4 第 2 點）：呼叫序列 ----


def run_child_in_process(tmp_path, monkeypatch, adapter_path):
    """在同一個 process 內執行 run_child.main，替換計時、記憶體與 relief，讓它們與 adapter 的呼叫寫進同一份序列。

    `clock` = 一次 perf_counter_ns（計時區間的邊界）；`memory` = 一次 _memory()；`relief` = 一次 relief。
    pytest 的 process 已經 import 過受測的 library，所以只在這裡略過「只 import 受測 library」的檢查（另有子 process 的測試）。
    """
    rng = np.random.default_rng(0)
    data_path, queries_path = tmp_path / "data.npy", tmp_path / "queries.npy"
    np.save(data_path, prepare_vectors(rng.standard_normal((300, 8))))
    np.save(queries_path, prepare_vectors(rng.standard_normal((100, 8))))
    (tmp_path / "raw").mkdir()
    spec = {"child_id": "g000-trace", "implementation": "trace", "adapter": adapter_path, "library": "numpy",
            "params": {"m": 8, "ef_construction": 32, "seed": 1}, "data_path": str(data_path),
            "queries_path": str(queries_path), "ef_search": [10, 20], "jsonl_path": str(tmp_path / "points.jsonl"),
            "raw_dir": str(tmp_path / "raw"), "samples_path": str(tmp_path / "samples.npz"), "scratch_dir": str(tmp_path)}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    log = tmp_path / "calls.log"
    monkeypatch.setenv("STUB_CALL_LOG", str(log))
    monkeypatch.setattr(run_child, "MEASURED_LIBRARIES", ())
    real_clock, real_memory, real_relief = run_child.perf_counter_ns, run_child._memory, run_child._pressure_relief
    readings = []

    def clock():
        _log("clock")
        return real_clock()

    def memory():
        _log("memory")
        readings.append(real_memory())
        return readings[-1]

    def relief():
        _log("relief")
        real_relief()

    monkeypatch.setattr(run_child, "perf_counter_ns", clock)
    monkeypatch.setattr(run_child, "_memory", memory)
    monkeypatch.setattr(run_child, "_pressure_relief", relief)
    assert run_child.main([str(spec_path)]) == 0
    lines = [json.loads(line) for line in (tmp_path / "points.jsonl").read_text(encoding="utf-8").splitlines()]
    return log.read_text(encoding="utf-8").split(), readings, lines


def expected_sequence(save: bool) -> list[str]:
    # 與 test_runner.test_child_call_sequence_matches_008 相同的 008 §6 序列，加上記憶體的三個時點與 relief。
    expected = ["build:16", "ef:1", "search:1,1,1,1", "build:1", "ef:1"] + ["search:1,1,1,1"] * 1100
    expected += ["memory", "build:300", "memory"]
    for ef in (10, 20):
        expected += [f"ef:{ef}", f"search:10,100,{ef},1"] + [f"search:10,100,{ef},0"] * 5
        expected += [f"search:10,1,{ef},0"] * 400 + [f"search:1,100,{ef},1"]
    return expected + (["save"] if save else []) + ["relief", "memory"]


@pytest.mark.parametrize(("adapter_path", "save"), [("tests.stub_adapters:TracingStub", True),
                                                    ("tests.stub_adapters:TracingNoSaveStub", False)])
def test_relief_comes_after_every_timed_point_and_save_native(tmp_path, monkeypatch, adapter_path, save):
    events, readings, lines = run_child_in_process(tmp_path, monkeypatch, adapter_path)
    assert [event for event in events if event != "clock"] == expected_sequence(save)
    relief = events.index("relief")
    assert events.count("relief") == 1
    # 不在任何計時區間內：relief 之後再也沒有 perf_counter_ns，也沒有 adapter 的呼叫。
    assert relief > max(i for i, event in enumerate(events) if event == "clock")
    assert all(event == "memory" for event in events[relief + 1:])
    if save:
        assert events.index("save") < relief
    # after_relief 就是 relief 之後的那一次讀數；serialized_bytes 依 adapter 是否有 save_native。
    assert [line["type"] for line in lines] == ["start", "prelude", "build", "point", "point", "end"]
    assert len(readings) == 3
    assert lines[1]["baseline"] == readings[0] and lines[2]["after_build"] == readings[1]
    end = lines[-1]
    assert end["after_relief"] == readings[2]
    assert (end["serialized_bytes"] is None) is (not save)
    if save:
        assert end["serialized_bytes"] > 0
    assert not list(tmp_path.glob("*.index"))


def test_pressure_relief_uses_the_probe_script_ctypes_call(monkeypatch):
    # 019 §4：沿用 scripts/probe_reusable_memory.py 的寫法：libSystem 的 malloc_zone_pressure_relief(NULL, 0)。
    opened, calls = [], []

    class Function:
        argtypes = restype = None

        def __call__(self, *args):
            calls.append(args)
            return 0

    class Library:
        def __init__(self, path):
            opened.append(path)
            self.malloc_zone_pressure_relief = Function()

    libraries = []
    monkeypatch.setattr(run_child.ctypes, "CDLL", lambda path: libraries.append(Library(path)) or libraries[-1])
    run_child._pressure_relief()
    assert opened == ["/usr/lib/libSystem.B.dylib"]
    function = libraries[0].malloc_zone_pressure_relief
    assert function.argtypes == [ctypes.c_void_p, ctypes.c_size_t]
    assert function.restype is ctypes.c_size_t
    assert calls == [(None, 0)]


def test_the_real_pressure_relief_call_succeeds():
    run_child._pressure_relief()


# ---- 019 的新 adapter 經過真正的 runner（synthetic 的 smoke-gaussian） ----


def test_the_quant_adapters_complete_a_runner_pass(tmp_path):
    # 階段 3 登記了 F、S、Q：F、S 沒有 save_native，008-v4 的 runner 會在所有 ef 點之後以 AttributeError 失敗。
    names = ["rust-hnsw", "rust-hnsw-fp32g", "rust-hnsw-sq8", "faiss-hnsw-sq8"]
    result = run(stub_config(names, ef_search=(10, 20)), "smoke", tmp_path)
    assert result.record["status"] == "complete"
    by_name = {child["implementation"]: child for child in result.record["children"]}
    for name in names:
        child = by_name[name]
        assert child["status"] == "ok" and [p["ef_search"] for p in child["points"]] == [10, 20]
        assert set(child["build"]["after_relief"]) == MEMORY_KEYS
        assert child["loaded_libraries"] == ["faiss" if name.startswith("faiss") else "rust_hnsw"]
    assert by_name["rust-hnsw-fp32g"]["build"]["serialized_bytes"] is None
    assert by_name["rust-hnsw-sq8"]["build"]["serialized_bytes"] is None
    assert by_name["rust-hnsw"]["build"]["serialized_bytes"] > 0
    assert by_name["faiss-hnsw-sq8"]["build"]["serialized_bytes"] > 0
    # F 與 R 是同一張圖、同一個距離：經過 runner 之後的 ids 仍逐位元相同（C7a 的機制）。
    for f_point, r_point in zip(by_name["rust-hnsw-fp32g"]["points"], by_name["rust-hnsw"]["points"]):
        f_arrays, r_arrays = np.load(result.run_dir / f_point["file"]), np.load(result.run_dir / r_point["file"])
        np.testing.assert_array_equal(f_arrays["ids_k10"], r_arrays["ids_k10"])
        np.testing.assert_array_equal(f_arrays["ids_k1"], r_arrays["ids_k1"])
