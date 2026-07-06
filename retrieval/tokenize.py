"""Query tokenization.

IMPORTANT: sparse retrieval only works if the query is tokenized the same way
the chunks' ``word_frequency`` maps were built. The default below (lowercase,
unicode word characters) is a reasonable baseline; if your ingestion pipeline
used something else (stemming, stopword removal, different casing), pass that
tokenizer to :class:`retrieval.sparse.BM25Index` / ``HybridRetriever`` instead.
"""

from __future__ import annotations

import re
from typing import Callable, List

Tokenizer = Callable[[str], List[str]]

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def default_tokenizer(text: str) -> List[str]:
    """Lowercase and split on unicode word characters."""
    return _WORD_RE.findall(text.lower())
