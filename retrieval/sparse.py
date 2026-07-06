"""Sparse (lexical) retrieval.

The default implementation is BM25 built directly from the per-chunk
``word_frequency`` maps you already store, so no re-tokenization of chunk text
is needed at index time. An inverted index is used so query cost scales with
the number of chunks that *contain a query term*, not with corpus size.

For very large corpora, implement :class:`SparseIndex` on top of a real search
engine (Elasticsearch/OpenSearch, Lucene, or Palantir Foundry's built-in
full-text search) and pass that to ``HybridRetriever`` — the fusion logic only
needs a ranked list of (chunk_id, score) back.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Protocol, Tuple

from retrieval.tokenize import Tokenizer, default_tokenizer
from retrieval.types import Chunk, ScoredChunk


class SparseIndex(Protocol):
    """Anything that can return a lexical ranking for a query string."""

    def search(self, query: str, top_k: int) -> List[ScoredChunk]:
        """Return up to ``top_k`` chunks ranked by lexical relevance."""
        ...


class BM25Index:
    """In-memory BM25 (Okapi) index over pre-computed word frequencies.

    Args:
        chunks: The corpus. Only ``chunk_id`` and ``word_frequency`` are used.
        tokenizer: Must match the tokenization used to build ``word_frequency``.
        k1: BM25 term-frequency saturation (typical range 1.2-2.0).
        b: BM25 length normalization (0 = none, 1 = full).
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        tokenizer: Tokenizer = default_tokenizer,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._tokenizer = tokenizer
        self._k1 = k1
        self._b = b

        # term -> list of (chunk_id, term_frequency)
        self._postings: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # chunk_id -> chunk length in tokens
        self._doc_len: Dict[str, int] = {}

        for chunk in chunks:
            length = sum(chunk.word_frequency.values())
            self._doc_len[chunk.chunk_id] = length
            for term, tf in chunk.word_frequency.items():
                self._postings[term].append((chunk.chunk_id, tf))

        self._n_docs = len(self._doc_len)
        self._avg_doc_len = (
            sum(self._doc_len.values()) / self._n_docs if self._n_docs else 0.0
        )

        # Precompute IDF per term (BM25+ style floor at 0 to avoid negative
        # scores for terms present in most documents).
        self._idf: Dict[str, float] = {
            term: max(
                0.0,
                math.log(
                    (self._n_docs - len(postings) + 0.5) / (len(postings) + 0.5) + 1.0
                ),
            )
            for term, postings in self._postings.items()
        }

    def search(self, query: str, top_k: int) -> List[ScoredChunk]:
        scores: Dict[str, float] = defaultdict(float)
        for term in self._tokenizer(query):
            idf = self._idf.get(term)
            if idf is None or idf == 0.0:
                continue
            for chunk_id, tf in self._postings[term]:
                norm = 1.0 - self._b + self._b * (
                    self._doc_len[chunk_id] / self._avg_doc_len
                )
                scores[chunk_id] += idf * (tf * (self._k1 + 1.0)) / (
                    tf + self._k1 * norm
                )

        top = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
        return [ScoredChunk(chunk_id=cid, score=score) for cid, score in top]
