"""Hybrid retriever: sparse + dense candidate generation fused with RRF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from retrieval.dense import BruteForceDenseIndex, DenseIndex
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.sparse import BM25Index, SparseIndex
from retrieval.tokenize import Tokenizer, default_tokenizer
from retrieval.types import Chunk, ScoredChunk

Embedder = Callable[[str], Sequence[float]]


@dataclass(frozen=True)
class RetrievalResult:
    """One fused search hit, with per-retriever diagnostics.

    ``sparse_rank`` / ``dense_rank`` are 1-based positions within each
    retriever's candidate list, or ``None`` if that retriever did not surface
    the chunk at all. They are useful for debugging fusion behavior.
    """

    chunk: Chunk
    fused_score: float
    sparse_rank: Optional[int]
    sparse_score: Optional[float]
    dense_rank: Optional[int]
    dense_score: Optional[float]


class HybridRetriever:
    """Runs sparse and dense search and merges the rankings with RRF.

    Args:
        chunks: The corpus. Chunk ids must be unique.
        embedder: Callable that embeds a query string. Optional if every call
            to :meth:`search` supplies ``query_embedding`` directly.
        sparse_index: Override the default in-memory BM25 index (e.g. with an
            Elasticsearch- or Foundry-backed implementation).
        dense_index: Override the default brute-force cosine index (e.g. with
            a FAISS- or vector-DB-backed implementation).
        tokenizer: Used by the default BM25 index; must match how the chunks'
            ``word_frequency`` maps were produced. Ignored if ``sparse_index``
            is supplied.
        rrf_k: RRF damping constant (60 is the standard default).
        sparse_weight / dense_weight: Relative weight of each ranking in the
            fusion. Equal by default.
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        embedder: Optional[Embedder] = None,
        sparse_index: Optional[SparseIndex] = None,
        dense_index: Optional[DenseIndex] = None,
        tokenizer: Tokenizer = default_tokenizer,
        rrf_k: int = 60,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        chunk_list = list(chunks)
        self._chunks_by_id: Dict[str, Chunk] = {c.chunk_id: c for c in chunk_list}
        if len(self._chunks_by_id) != len(chunk_list):
            raise ValueError("chunk_id values must be unique across the corpus")

        self._embedder = embedder
        self._sparse = sparse_index or BM25Index(chunk_list, tokenizer=tokenizer)
        self._dense = dense_index or BruteForceDenseIndex(chunk_list)
        self._rrf_k = rrf_k
        self._weights = [sparse_weight, dense_weight]

    def search(
        self,
        query: str,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: int = 10,
        candidate_pool: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Hybrid search for ``query``.

        Args:
            query: The query string (used for sparse search, and for dense
                search via the embedder when ``query_embedding`` is absent).
            query_embedding: Pre-computed query vector. If omitted, the
                retriever's ``embedder`` is called; providing it saves that
                call and is the recommended path when you embed queries
                elsewhere anyway.
            top_k: Number of fused results to return.
            candidate_pool: How many candidates to pull from *each* retriever
                before fusing. Defaults to ``max(50, 5 * top_k)``. A pool
                larger than ``top_k`` matters: it lets a chunk ranked
                moderately by both retrievers beat a chunk ranked highly by
                only one, which is the point of hybrid search.
        """
        if query_embedding is None:
            if self._embedder is None:
                raise ValueError(
                    "no query_embedding given and no embedder configured; "
                    "provide one of the two"
                )
            query_embedding = self._embedder(query)

        pool = candidate_pool if candidate_pool is not None else max(50, 5 * top_k)

        sparse_hits = self._sparse.search(query, top_k=pool)
        dense_hits = self._dense.search(query_embedding, top_k=pool)

        fused = reciprocal_rank_fusion(
            rankings=[
                [hit.chunk_id for hit in sparse_hits],
                [hit.chunk_id for hit in dense_hits],
            ],
            k=self._rrf_k,
            weights=self._weights,
        )

        sparse_by_id = _index_hits(sparse_hits)
        dense_by_id = _index_hits(dense_hits)

        results: List[RetrievalResult] = []
        for chunk_id, fused_score in fused[:top_k]:
            sparse_hit = sparse_by_id.get(chunk_id)
            dense_hit = dense_by_id.get(chunk_id)
            results.append(
                RetrievalResult(
                    chunk=self._chunks_by_id[chunk_id],
                    fused_score=fused_score,
                    sparse_rank=sparse_hit[0] if sparse_hit else None,
                    sparse_score=sparse_hit[1] if sparse_hit else None,
                    dense_rank=dense_hit[0] if dense_hit else None,
                    dense_score=dense_hit[1] if dense_hit else None,
                )
            )
        return results


def _index_hits(hits: Sequence[ScoredChunk]) -> Dict[str, tuple]:
    """Map chunk_id -> (1-based rank, score)."""
    return {hit.chunk_id: (rank, hit.score) for rank, hit in enumerate(hits, start=1)}
