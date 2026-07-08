"""BM25 (Okapi) scoring over precomputed per-chunk term frequencies.

Foundry Python Functions have no persistent inverted index, so this module
does not build one. Instead it scores an already-retrieved candidate set
using statistics computed at ingestion:

- per-chunk ``{term: count}`` maps and token counts (stored on the chunk),
- corpus-level document frequencies and average length (stored on the
  ``SearchTermStats`` / ``SearchCorpusMeta`` object types).

When corpus statistics are unavailable (ingestion enrichment not deployed
yet), :func:`stats_from_candidates` estimates them from the candidate set.
Candidate-set IDF is biased — every candidate matched at least one query
term — but it still produces a far better ranking than ad-hoc scoring, and
the ranking is all RRF fusion consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class Bm25Params:
    """k1: term-frequency saturation (1.2-2.0). b: length normalization."""

    k1: float = 1.5
    b: float = 0.75


@dataclass(frozen=True)
class CorpusStats:
    """Corpus-level inputs to BM25.

    ``document_frequency`` only needs entries for the query's terms.
    """

    n_docs: int
    avg_doc_len: float
    document_frequency: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredDoc:
    doc_id: str
    score: float


def idf(n_docs: int, document_frequency: int) -> float:
    """BM25+ style IDF, floored at 0 so ubiquitous terms never go negative."""
    if n_docs <= 0:
        return 0.0
    df = min(max(document_frequency, 0), n_docs)
    return max(0.0, math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0))


def stats_from_candidates(
    candidates: Sequence[Tuple[str, Mapping[str, int]]],
    query_terms: Sequence[str],
) -> CorpusStats:
    """Fallback corpus statistics estimated from the candidate set alone."""
    n = len(candidates)
    total_len = sum(sum(counts.values()) for _, counts in candidates)
    dfs = {
        term: sum(1 for _, counts in candidates if counts.get(term, 0) > 0)
        for term in query_terms
    }
    return CorpusStats(
        n_docs=n,
        avg_doc_len=(total_len / n) if n else 0.0,
        document_frequency=dfs,
    )


def score(
    query_terms: Sequence[str],
    term_counts: Mapping[str, int],
    doc_len: int,
    stats: CorpusStats,
    params: Bm25Params = Bm25Params(),
) -> float:
    """BM25 score of one document for the given (deduplicated) query terms."""
    if not query_terms or doc_len <= 0 or stats.n_docs <= 0:
        return 0.0
    avg_len = stats.avg_doc_len if stats.avg_doc_len > 0 else float(doc_len)
    norm = 1.0 - params.b + params.b * (doc_len / avg_len)

    total = 0.0
    for term in query_terms:
        tf = term_counts.get(term, 0)
        if tf <= 0:
            continue
        term_idf = idf(stats.n_docs, stats.document_frequency.get(term, 0))
        total += term_idf * (tf * (params.k1 + 1.0)) / (tf + params.k1 * norm)
    return total


def rank_candidates(
    query_terms: Sequence[str],
    candidates: Sequence[Tuple[str, Mapping[str, int], int]],
    stats: CorpusStats,
    params: Bm25Params = Bm25Params(),
) -> List[ScoredDoc]:
    """Score and rank candidates: (doc_id, term_counts, doc_len) tuples.

    Documents scoring 0 (no query-term overlap after analysis) are dropped.
    Ties break on doc_id so results are deterministic across invocations.
    """
    scored = [
        ScoredDoc(doc_id=doc_id, score=score(query_terms, counts, doc_len, stats, params))
        for doc_id, counts, doc_len in candidates
    ]
    scored = [item for item in scored if item.score > 0.0]
    scored.sort(key=lambda item: (-item.score, item.doc_id))
    return scored
