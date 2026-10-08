"""文字 corpus 的前處理與 base / query 切分（decision 011 §3.2、§3.3）。

只依賴標準函式庫與 pyarrow，不依賴任何亂數產生器：文章的順序由加鹽的 SHA-256 決定，換套件版本也能重現。
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass

MIN_CHARS = 200
MAX_CHARS = 1000
KEY_PREFIX = "rust-hnsw-text-v1:"

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Passage:
    passage_id: str
    article_id: str
    title: str
    text: str


def normalize_paragraph(text: str) -> str:
    """NFC 正規化、連續空白合併成一個空格、去掉頭尾空白。"""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def truncate_at_space(text: str, limit: int = MAX_CHARS) -> str:
    """長於 limit 時，截在 limit 字元以內的最後一個空白處（不切斷單字）。"""
    if len(text) <= limit:
        return text
    head = text[: limit + 1]
    cut = head.rfind(" ")
    return (head[:cut] if cut > 0 else text[:limit]).rstrip()


def article_passages(article_id: str, text: str) -> list[tuple[str, str]]:
    """一篇文章的 passage：以換行切段，正規化，截到 1,000 字元，再保留 ≥ 200 字元的段落。

    先截斷再過濾：截斷點若落在 200 字元之前（前段很長沒有空白），截斷後的短句也會被丟棄。
    """
    kept = []
    for paragraph in text.split("\n"):
        passage = truncate_at_space(normalize_paragraph(paragraph))
        if len(passage) >= MIN_CHARS:
            kept.append(passage)
    return [(f"{article_id}:{index}", passage) for index, passage in enumerate(kept)]


def article_key(article_id: str) -> str:
    return hashlib.sha256((KEY_PREFIX + article_id).encode("utf-8")).hexdigest()


def format_document(title: str, passage: str) -> str:
    """EmbeddingGemma 2 model card 的 document 格式；base 與 query 都用這個格式（011 §3.4）。"""
    return f"title: {title} | text: {passage}"


def _ids_sha256(passages: list[Passage]) -> str:
    return hashlib.sha256("\n".join(p.passage_id for p in passages).encode("utf-8")).hexdigest()


@dataclass
class Split:
    base: list[Passage]
    queries: list[Passage]
    articles_seen: int
    duplicates_removed: int

    def metadata(self) -> dict:
        return {
            "rule": "articles ordered by sha256(key_prefix + article id); queries = first passage of the first "
                    "n_queries articles; base = all passages of the remaining articles in order, first n_base; "
                    "identical passage texts kept once (earliest article in that order)",
            "key_prefix": KEY_PREFIX,
            "min_chars": MIN_CHARS,
            "max_chars": MAX_CHARS,
            "n_base": len(self.base),
            "n_queries": len(self.queries),
            "articles_seen": self.articles_seen,
            "duplicates_removed": self.duplicates_removed,
            "base_ids_sha256": _ids_sha256(self.base),
            "query_ids_sha256": _ids_sha256(self.queries),
        }


def build_split(articles, n_base: int, n_queries: int) -> Split:
    """deterministic 而且 disjoint 的 base / query（以文章為單位，011 §3.3）。

    `articles` 為 (id, title, text) 的 iterable；文章 id 唯一（Wikipedia page id）時，輸入順序不影響結果。
    """
    ordered = sorted(((article_key(str(a)), str(a), t, x) for a, t, x in articles), key=lambda item: item[0])
    seen_texts: set[str] = set()
    duplicates = 0
    queries: list[Passage] = []
    base: list[Passage] = []
    for _, article_id, title, text in ordered:
        passages = []
        for passage_id, passage in article_passages(article_id, text):
            if passage in seen_texts:
                duplicates += 1
                continue
            seen_texts.add(passage)
            passages.append(Passage(passage_id, article_id, title, passage))
        if not passages:
            continue
        if len(queries) < n_queries:
            # query 文章只取第一個 passage；它的其他 passage 也不得進入 base。
            queries.append(passages[0])
        elif len(base) < n_base:
            base.extend(passages[: n_base - len(base)])
        else:
            break
    if len(queries) < n_queries:
        raise ValueError(f"corpus has only {len(queries)} usable query articles, need {n_queries}")
    if len(base) < n_base:
        raise ValueError(f"corpus has only {len(base)} base passages, need {n_base}")
    split = Split(base=base, queries=queries, articles_seen=len(ordered), duplicates_removed=duplicates)
    _assert_disjoint(split)
    return split


def _assert_disjoint(split: Split) -> None:
    for field in ("passage_id", "article_id", "text"):
        if {getattr(p, field) for p in split.queries} & {getattr(p, field) for p in split.base}:
            raise AssertionError(f"base and queries share a {field}")


def read_parquet_articles(path):
    """Wikipedia parquet（欄位 id、title、text）逐列產生 (id, title, text)。"""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["id", "title", "text"])
    for article_id, title, text in zip(*(table.column(name).to_pylist() for name in ("id", "title", "text"))):
        yield str(article_id), title, text
