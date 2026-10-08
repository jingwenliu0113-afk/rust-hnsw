"""文字 corpus 的前處理與 base / query 切分（decision 011 §3.2、§3.3）。"""

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks import text_corpus as tc

LONG = "word " * 60  # 300 個字元


def test_paragraphs_are_normalized_filtered_and_truncated_at_a_space():
    text = "Title line\n\n" + "a\tb  " + "x" * 196 + "\nshort\n" + ("lorem ipsum " * 120)
    passages = tc.article_passages("42", text)
    assert [pid for pid, _ in passages] == ["42:0", "42:1"]
    assert passages[0][1].startswith("a b x")  # tab 與連續空白合併
    assert len(passages[1][1]) <= tc.MAX_CHARS
    assert passages[1][1].endswith("ipsum") or passages[1][1].endswith("lorem")  # 不切斷單字
    assert not passages[1][1].endswith(" ")


def test_truncation_never_yields_a_short_passage():
    # 前 1,001 字元內唯一的空白在第 2 個字元：截斷後只剩 "Hi"，必須丟棄。
    assert tc.article_passages("1", "Hi " + "x" * 1200) == []
    passages = tc.article_passages("2", "\n".join(["Hi " + "x" * 1200, "word " * 300, LONG]))
    assert all(tc.MIN_CHARS <= len(p) <= tc.MAX_CHARS for _, p in passages)
    assert [pid for pid, _ in passages] == ["2:0", "2:1"]


def test_unicode_is_nfc_normalized():
    decomposed = "Café " + "x" * 200
    (_, passage), = tc.article_passages("1", decomposed)
    assert passage.startswith("Café ")


def test_article_order_is_a_salted_sha256():
    key = tc.article_key("123")
    assert key == hashlib.sha256(b"rust-hnsw-text-v1:123").hexdigest()


def articles(count, passages_per_article=3, duplicate_text=None):
    rows = []
    for i in range(count):
        paragraphs = [f"article {i} paragraph {j} " + LONG for j in range(passages_per_article)]
        if duplicate_text is not None and i % 2 == 0:
            paragraphs.append(duplicate_text)
        rows.append((str(i), f"Title {i}", "\n".join(paragraphs)))
    return rows


def test_split_is_deterministic_disjoint_and_article_level():
    split = tc.build_split(articles(200), n_base=300, n_queries=40)
    again = tc.build_split(list(reversed(articles(200))), n_base=300, n_queries=40)
    assert [p.passage_id for p in split.base] == [p.passage_id for p in again.base]
    assert [p.passage_id for p in split.queries] == [p.passage_id for p in again.queries]
    assert len(split.base) == 300 and len(split.queries) == 40
    query_articles = {p.article_id for p in split.queries}
    assert len(query_articles) == 40  # 每篇 query 文章只取一個 passage
    assert query_articles.isdisjoint({p.article_id for p in split.base})
    assert {p.text for p in split.queries}.isdisjoint({p.text for p in split.base})
    # query 取排序最前面的文章，而且是該文章的第一個 passage。
    order = sorted({str(i) for i in range(200)}, key=tc.article_key)
    assert [p.article_id for p in split.queries] == order[:40]
    assert all(p.passage_id.endswith(":0") for p in split.queries)


def test_base_follows_article_order_then_paragraph_order():
    split = tc.build_split(articles(50), n_base=30, n_queries=5)
    order = sorted({str(i) for i in range(50)}, key=tc.article_key)[5:]
    expected = [f"{a}:{j}" for a in order for j in range(3)][:30]
    assert [p.passage_id for p in split.base] == expected


def test_identical_texts_are_kept_once_across_the_whole_corpus():
    duplicate = "shared boilerplate paragraph " + LONG
    split = tc.build_split(articles(100, duplicate_text=duplicate), n_base=250, n_queries=10)
    texts = [p.text for p in split.base + split.queries]
    # 第一份若在 query 文章的非首段，它不會輸出，但其他副本仍會被移除（避免洩漏）：所以最多一份。
    assert texts.count(tc.normalize_paragraph(duplicate)) <= 1
    assert split.duplicates_removed > 0
    assert len(texts) == len(set(texts))


def test_split_fails_loudly_when_the_corpus_is_too_small():
    with pytest.raises(ValueError, match="base"):
        tc.build_split(articles(10), n_base=1000, n_queries=2)
    with pytest.raises(ValueError, match="quer"):
        tc.build_split(articles(3), n_base=1, n_queries=10)


def test_document_format_matches_the_model_card():
    assert tc.format_document("Paris", "Paris is the capital.") == "title: Paris | text: Paris is the capital."


def test_split_metadata_has_rule_counts_and_id_checksums():
    split = tc.build_split(articles(80), n_base=100, n_queries=8)
    meta = split.metadata()
    assert meta["key_prefix"] == "rust-hnsw-text-v1:"
    assert (meta["n_base"], meta["n_queries"]) == (100, 8)
    joined = "\n".join(p.passage_id for p in split.base).encode()
    assert meta["base_ids_sha256"] == hashlib.sha256(joined).hexdigest()
    assert meta["min_chars"] == 200 and meta["max_chars"] == 1000


def test_reads_articles_from_parquet(tmp_path):
    table = pa.table({"id": ["1", "2"], "url": ["u1", "u2"], "title": ["A", "B"], "text": ["x " + LONG, "y " + LONG]})
    path = tmp_path / "w.parquet"
    pq.write_table(table, path)
    assert list(tc.read_parquet_articles(path)) == [("1", "A", "x " + LONG), ("2", "B", "y " + LONG)]
