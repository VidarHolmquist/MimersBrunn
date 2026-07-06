"""Hybrid sparse + dense retrieval with Reciprocal Rank Fusion (RRF).

Typical usage::

    from retrieval import Chunk, HybridRetriever

    retriever = HybridRetriever(chunks, embedder=my_embedding_fn)
    results = retriever.search("how do I reset my password?", top_k=10)

See README.md for scaling notes and how to swap in ANN backends.
"""

from retrieval.dense import BruteForceDenseIndex, DenseIndex
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.retriever import HybridRetriever, RetrievalResult
from retrieval.sparse import BM25Index, SparseIndex
from retrieval.types import Chunk, ScoredChunk

__all__ = [
    "BM25Index",
    "BruteForceDenseIndex",
    "Chunk",
    "DenseIndex",
    "HybridRetriever",
    "RetrievalResult",
    "ScoredChunk",
    "SparseIndex",
    "reciprocal_rank_fusion",
]
