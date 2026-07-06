"""Dense (vector) retrieval.

The default implementation is exact cosine similarity over a NumPy matrix.
This is simple, has no external dependencies beyond NumPy, and is fast enough
up to roughly a few hundred thousand chunks on a single machine.

Beyond that, implement :class:`DenseIndex` on top of an ANN library or a
vector database (FAISS, hnswlib, pgvector, or whatever vector search Palantir
Foundry exposes) and pass it to ``HybridRetriever`` — the rest of the pipeline
only needs ranked (chunk_id, score) pairs back.
"""

from __future__ import annotations

from typing import Iterable, List, Protocol, Sequence

import numpy as np

from retrieval.types import Chunk, ScoredChunk


class DenseIndex(Protocol):
    """Anything that can return a vector-similarity ranking for a query vector."""

    def search(self, query_embedding: Sequence[float], top_k: int) -> List[ScoredChunk]:
        """Return up to ``top_k`` chunks ranked by similarity to the query."""
        ...


class BruteForceDenseIndex:
    """Exact cosine-similarity search over all chunk embeddings.

    Embeddings are L2-normalized once at build time, so each query is a single
    matrix-vector product followed by a partial sort (``argpartition``), i.e.
    O(n_chunks * dim) per query.
    """

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        chunks = list(chunks)
        self._chunk_ids: List[str] = [c.chunk_id for c in chunks]

        if chunks:
            matrix = np.asarray([c.embedding for c in chunks], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            self._matrix = matrix / norms
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)

    def search(self, query_embedding: Sequence[float], top_k: int) -> List[ScoredChunk]:
        if not self._chunk_ids:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0.0:
            query = query / norm

        similarities = self._matrix @ query

        top_k = min(top_k, len(self._chunk_ids))
        candidate_idx = np.argpartition(similarities, -top_k)[-top_k:]
        ranked_idx = candidate_idx[np.argsort(similarities[candidate_idx])[::-1]]

        return [
            ScoredChunk(chunk_id=self._chunk_ids[i], score=float(similarities[i]))
            for i in ranked_idx
        ]
