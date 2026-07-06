"""Core data types shared by the retrieval modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text.

    Attributes:
        chunk_id: Stable, unique identifier for the chunk. Ranked lists from
            the sparse and dense indexes are joined on this id during fusion,
            so it must be unique across the whole corpus.
        source: Where the chunk came from (document name, URL, dataset row, ...).
        text: The raw chunk text, returned to the caller with results.
        embedding: Dense vector for the chunk. All chunks must use the same
            embedding model and dimensionality as the query embedder.
        word_frequency: Term -> count mapping for the chunk, produced by the
            same tokenizer/normalization that will be applied to queries.
        metadata: Optional extra fields (page number, language, timestamps, ...)
            useful for filtering or display; not used for scoring.
    """

    chunk_id: str
    source: str
    text: str
    embedding: Sequence[float]
    word_frequency: Mapping[str, int]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk id with a score from a single index (sparse or dense)."""

    chunk_id: str
    score: float
