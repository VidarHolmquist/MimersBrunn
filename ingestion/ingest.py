"""Ingestion-time enrichment for hybrid search — run once per pipeline build.

Everything expensive happens here so the retrieval function stays fast:

1. Build the decompounding lexicon from the whole corpus.
2. Analyze every chunk with the Swedish analyzer and store:
   - ``analyzed_text``   space-joined analyzed tokens; index this property
     for full-text search in the Ontology so query stems match chunk stems.
   - ``word_frequency``  JSON ``{term: count}`` — the BM25 term-frequency
     input, so the retrieval function never re-tokenizes chunk text.
   - ``token_count``     chunk length for BM25 length normalization.
3. Compute corpus-level statistics that make runtime BM25 exact:
   - one row per term with its document frequency (-> ``SearchTermStats``
     object type; the retrieval function looks up only the query's terms),
   - a singleton meta row with chunk count, average length, analyzer
     version, and the serialized lexicon (-> ``SearchCorpusMeta``).

Deploy as a Python transform in a Code Repository. Example::

    from transforms.api import transform_df, Input, Output

    @transform_df(
        Output("/MimersBrunn/data/chunks_enriched"),
        Output("/MimersBrunn/data/search_term_stats"),
        Output("/MimersBrunn/data/search_corpus_meta"),
        chunks=Input("/MimersBrunn/data/chunks"),
    )
    def enrich(ctx, chunks_out, term_stats_out, meta_out, chunks):
        df = chunks.dataframe().toPandas()
        result = enrich_corpus(zip(df["chunk_id"], df["chunked_text"]))
        # join result.chunks back onto df by chunk_id, write all three
        # outputs, then back the ontology object types with these datasets.

NOTE ON INCREMENTAL RUNS: the lexicon and document frequencies are
corpus-global, so this transform should run as a full snapshot. BM25 is
robust to slightly stale statistics — if the corpus grows continuously,
a scheduled (e.g. nightly) full rebuild is fine while new chunks arrive
incrementally with only steps 1-2 applied.

If the embedding model ever changes, force a full rebuild of embeddings as
well — mixed vector spaces fail silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from shared.swedish_analyzer import (
    ANALYZER_VERSION,
    SwedishAnalyzer,
    build_lexicon,
    lexicon_to_json,
)

# Primary key of the singleton meta row / SearchCorpusMeta object.
CORPUS_META_ID = "corpus"


@dataclass(frozen=True)
class ChunkEnrichment:
    """Per-chunk columns to add to the chunks dataset."""

    chunk_id: str
    analyzed_text: str
    word_frequency_json: str
    token_count: int


@dataclass(frozen=True)
class TermStats:
    """One row per analyzed term -> SearchTermStats object type."""

    term: str
    document_frequency: int


@dataclass(frozen=True)
class CorpusMeta:
    """Singleton corpus row -> SearchCorpusMeta object type."""

    meta_id: str
    n_chunks: int
    avg_token_count: float
    analyzer_version: str
    lexicon_json: str


@dataclass(frozen=True)
class EnrichmentResult:
    chunks: List[ChunkEnrichment]
    term_stats: List[TermStats]
    meta: CorpusMeta


def enrich_corpus(
    id_text_pairs: Iterable[Tuple[str, Optional[str]]],
    lexicon: Optional[Set[str]] = None,
    min_lexicon_count: int = 3,
) -> EnrichmentResult:
    """Analyze a corpus of (chunk_id, text) pairs into search-ready columns.

    Two passes: the first builds the decompounding lexicon from the corpus
    (skipped if ``lexicon`` is supplied, e.g. when re-using a frozen lexicon
    for an incremental run), the second analyzes each chunk and accumulates
    document frequencies.
    """
    pairs = [(chunk_id, text or "") for chunk_id, text in id_text_pairs]

    if lexicon is None:
        lexicon = build_lexicon(
            (text for _, text in pairs), min_count=min_lexicon_count
        )
    analyzer = SwedishAnalyzer(lexicon=lexicon)

    chunks: List[ChunkEnrichment] = []
    document_frequency: Dict[str, int] = {}
    total_tokens = 0

    for chunk_id, text in pairs:
        counts = analyzer.term_counts(text)
        token_count = sum(counts.values())
        total_tokens += token_count
        for term in counts:
            document_frequency[term] = document_frequency.get(term, 0) + 1
        chunks.append(
            ChunkEnrichment(
                chunk_id=chunk_id,
                analyzed_text=" ".join(analyzer.tokenize(text)),
                word_frequency_json=json.dumps(
                    counts, ensure_ascii=False, sort_keys=True
                ),
                token_count=token_count,
            )
        )

    n_chunks = len(chunks)
    meta = CorpusMeta(
        meta_id=CORPUS_META_ID,
        n_chunks=n_chunks,
        avg_token_count=(total_tokens / n_chunks) if n_chunks else 0.0,
        analyzer_version=ANALYZER_VERSION,
        lexicon_json=lexicon_to_json(lexicon),
    )
    term_stats = [
        TermStats(term=term, document_frequency=df)
        for term, df in sorted(document_frequency.items())
    ]
    return EnrichmentResult(chunks=chunks, term_stats=term_stats, meta=meta)
