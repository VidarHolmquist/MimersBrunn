"""Reciprocal Rank Fusion (RRF).

RRF merges ranked lists using only ranks, never raw scores, which is exactly
why it works well for hybrid search: BM25 scores and cosine similarities live
on incomparable scales, but ranks are always comparable.

    RRF(d) = sum over rankings r of  weight_r / (k + rank_r(d))

where ``rank_r(d)`` is 1-based and ``k`` (default 60, from the original paper
by Cormack et al.) dampens the influence of top ranks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """Fuse ranked lists of ids into a single ranking.

    Args:
        rankings: One list of ids per retriever, best first. Ids missing from
            a list simply contribute nothing from that list.
        k: RRF damping constant; larger values flatten the rank contribution.
        weights: Optional per-ranking weights (e.g. to favor dense over
            sparse). Defaults to 1.0 for every ranking.

    Returns:
        (id, fused_score) pairs sorted by fused score, best first. Ties are
        broken by id so results are deterministic.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"got {len(weights)} weights for {len(rankings)} rankings"
        )

    scores: Dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += weight / (k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
