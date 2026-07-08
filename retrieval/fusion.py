"""Weighted Reciprocal Rank Fusion of multiple ranked lists.

RRF combines rankings without needing the underlying scores to be
comparable — exactly the situation with a dense cosine similarity and a
BM25 score. Each retriever contributes ``weight / (k + rank)`` for every
document it ranked:

    fused(d) = sum_over_retrievers  w_r / (k + rank_r(d))

``k`` (typically 60) dampens how much the very top ranks dominate; higher
``k`` flattens the contribution curve and rewards documents that appear in
several lists over documents that top a single list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class FusedDoc:
    doc_id: str
    score: float
    # Retriever name -> 1-based rank, for every list the doc appeared in.
    ranks: Dict[str, int]


def to_ranks(ranked_ids: Sequence[str]) -> Dict[str, int]:
    """Ranked id list -> {id: 1-based rank}; first occurrence wins."""
    ranks: Dict[str, int] = {}
    for position, doc_id in enumerate(ranked_ids, start=1):
        ranks.setdefault(doc_id, position)
    return ranks


def reciprocal_rank_fusion(
    rankings: Mapping[str, Mapping[str, int]],
    weights: Mapping[str, float],
    k: int = 60,
) -> List[FusedDoc]:
    """Fuse ``{retriever: {doc_id: rank}}`` into one ranking.

    Retrievers missing from ``weights`` default to weight 1.0. Ties break on
    the best single-list rank, then doc_id, so output is deterministic.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    fused: Dict[str, FusedDoc] = {}
    for retriever, doc_ranks in rankings.items():
        weight = weights.get(retriever, 1.0)
        if weight <= 0.0:
            continue
        for doc_id, rank in doc_ranks.items():
            previous = fused.get(doc_id)
            contribution = weight / (k + rank)
            if previous is None:
                fused[doc_id] = FusedDoc(
                    doc_id=doc_id, score=contribution, ranks={retriever: rank}
                )
            else:
                fused[doc_id] = FusedDoc(
                    doc_id=doc_id,
                    score=previous.score + contribution,
                    ranks={**previous.ranks, retriever: rank},
                )

    return sorted(
        fused.values(),
        key=lambda doc: (-doc.score, min(doc.ranks.values()), doc.doc_id),
    )
