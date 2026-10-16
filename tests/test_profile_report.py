"""macOS `sample` 報告的解析與 Rust v0 符號整理。"""

import pytest

from benchmarks.profile_report import pretty_symbol, top_of_stack

SAMPLE_REPORT = """\
Call graph:
    1000 Thread_1   DispatchQueue_1: com.apple.main-thread  (serial)
      1000 start  (in dyld) + 6076  [0x18a1b5c4c]

Total number in stack (recursive counted multiple times):

Sort by top of stack, same collapsed (when >= 5):
        _RNvMs_NtCs5kyc4sahh7U_4hnsw4hnswNtB4_9HnswIndex12search_layer  (in rust_hnsw.cpython-313-darwin.so)        600
        _RNvMs1_NtCsib8wo0YhOiF_9hashbrown3mapINtB5_7HashMapmuNtNtNtCs1SSuMAdGM58_3std4hash6random11RandomStateE6insertCs5kyc4sahh7U_4hnsw  (in rust_hnsw.cpython-313-darwin.so)        300
        free_small  (in libsystem_malloc.dylib)        100

Binary Images:
       0x100b34000 -        0x100fbbff3 +python3.13 (0) <C5DF> /opt/homebrew/*/python3.13
"""


def test_pretty_symbol_extracts_rust_path_segments():
    assert (
        pretty_symbol("_RNvMs_NtCs5kyc4sahh7U_4hnsw4hnswNtB4_9HnswIndex12search_layer")
        == "hnsw::hnsw::HnswIndex::search_layer"
    )
    hashmap = pretty_symbol(
        "_RNvMs1_NtCsib8wo0YhOiF_9hashbrown3mapINtB5_7HashMapmuNtNtNtCs1SSuMAdGM58_3std4hash6random11RandomStateE6insertCs5kyc4sahh7U_4hnsw"
    )
    assert hashmap.startswith("hashbrown::map::HashMap")
    assert "insert" in hashmap


def test_pretty_symbol_keeps_non_rust_names():
    assert pretty_symbol("free_small") == "free_small"


def test_top_of_stack_parses_counts_and_shares():
    rows = top_of_stack(SAMPLE_REPORT)
    assert [r.count for r in rows] == [600, 300, 100]
    assert [r.image for r in rows] == [
        "rust_hnsw.cpython-313-darwin.so",
        "rust_hnsw.cpython-313-darwin.so",
        "libsystem_malloc.dylib",
    ]
    assert rows[0].name == "hnsw::hnsw::HnswIndex::search_layer"
    assert rows[0].share == pytest.approx(0.6)
    assert sum(r.share for r in rows) == pytest.approx(1.0)


def test_top_of_stack_requires_the_section():
    with pytest.raises(ValueError, match="top of stack"):
        top_of_stack("Call graph:\n")


STRIPPED_REPORT = """\
Sort by top of stack, same collapsed (when >= 5):
        ???  (in libfaiss.dylib)  load address 0x109d5c000 + 0x2da19c  [0x10a03619c]        300
        ???  (in libfaiss.dylib)  load address 0x109d5c000 + 0x7fb20  [0x109ddbb20]        100
        free_tiny  (in libsystem_malloc.dylib)        100
        ???  (in <unknown binary>)  [0x1a5df40dc]        100

"""


def test_rows_without_symbols_are_kept_with_their_offset():
    # 009 補記：pip 安裝的 libfaiss.dylib 沒有符號表；這些列原本被略過，free_tiny 因此被算成 100%。
    rows = top_of_stack(STRIPPED_REPORT)
    assert [(r.name, r.image, r.count) for r in rows] == [
        ("??? +0x2da19c", "libfaiss.dylib", 300), ("??? +0x7fb20", "libfaiss.dylib", 100),
        ("free_tiny", "libsystem_malloc.dylib", 100), ("??? [0x1a5df40dc]", "<unknown binary>", 100)]
    assert rows[2].share == pytest.approx(100 / 600)


def test_an_unparsed_line_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError, match="unparsed"):
        top_of_stack("Sort by top of stack, same collapsed (when >= 5):\n        something odd here\n\n")
