"""Hybrid dense + sparse document search — the runtime Palantir Function.

Per query this does the minimum work possible; everything heavy was
precomputed by ``ingestion/ingest.py``:

  dense side    nearest-neighbor search over ``chunkedEmbedding`` (Foundry
                embeds the query and searches the vector index).
  sparse side   the query is analyzed with the same Swedish analyzer used
                at ingestion, candidates are fetched by term filter over
                the pre-analyzed text, then rescored with true BM25 using
                corpus statistics (document frequencies, average length)
                looked up from the ontology — a handful of point reads.
  fusion        weighted Reciprocal Rank Fusion of the two rankings.

GRACEFUL DEGRADATION: the function also runs against an ontology that only
has the original properties (``chunkedText``/``chunkedEmbedding``). Without
``analyzedText``/``wordFrequency`` it filters on raw text and tokenizes
candidates on the fly; without ``SearchTermStats``/``SearchCorpusMeta`` it
estimates BM25 statistics from the candidate set. Deploying the ingestion
enrichment upgrades quality without touching this function.

Expected ontology additions (see README for the full setup):

  DocumentChunk (existing) gains:
      analyzedText   String, full-text indexed — filter target for sparse
      wordFrequency  String (JSON {term: count})
      tokenCount     Integer

  SearchTermStats (new): term (PK, String), documentFrequency (Integer)
  SearchCorpusMeta (new, singleton row "corpus"): metaId (PK), nChunks
      (Integer), avgTokenCount (Double), analyzerVersion (String),
      lexiconJson (String)
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from functools import reduce
from operator import or_
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from foundry_sdk_runtime import AllowBetaFeatures
from functions.api import Double, Integer, function
from ontology_sdk import FoundryClient
from ontology_sdk.ontology.objects import DocumentChunk

from retrieval import bm25
from retrieval.fusion import reciprocal_rank_fusion, to_ranks
from shared.swedish_analyzer import (
    ANALYZER_VERSION,
    SwedishAnalyzer,
    lexicon_from_json,
)

logger = logging.getLogger(__name__)

# The corpus-stats object types only exist once the ingestion enrichment is
# deployed; the function degrades gracefully without them.
try:
    from ontology_sdk.ontology.objects import SearchCorpusMeta, SearchTermStats
except ImportError:  # pragma: no cover - depends on ontology state
    SearchCorpusMeta = None
    SearchTermStats = None

CORPUS_META_ID = "corpus"
# Foundry keeps the Python worker warm between invocations, so module-level
# caches survive; refresh corpus meta on this interval to pick up rebuilds.
_META_CACHE_TTL_SECONDS = 600.0
# contains_any_term degrades with very long term lists; queries rarely have
# more meaningful terms than this anyway.
_MAX_FILTER_TERMS = 24


@dataclass
class HybridSearchResult:
    chunk_id: str
    chunked_text: str | None
    plan_reference: str | None
    chunk_number: Integer | None
    document_type: str | None
    dense_rank: Integer | None
    sparse_rank: Integer | None
    rrf_score: Double
    dense_score: Double | None
    sparse_score: Double | None


# ---------------------------------------------------------------------------
# Corpus meta + analyzer (cached across warm invocations)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CorpusMeta:
    n_chunks: int
    avg_token_count: float
    analyzer: SwedishAnalyzer


_meta_cache: Tuple[float, Optional[_CorpusMeta]] | None = None


def _load_corpus_meta(client: FoundryClient) -> Optional[_CorpusMeta]:
    """Fetch the singleton SearchCorpusMeta row; None when unavailable."""
    global _meta_cache
    now = time.monotonic()
    if _meta_cache is not None and now - _meta_cache[0] < _META_CACHE_TTL_SECONDS:
        return _meta_cache[1]

    meta: Optional[_CorpusMeta] = None
    if SearchCorpusMeta is not None:
        try:
            rows = list(client.ontology.objects.SearchCorpusMeta.take(num_items=1))
            if rows:
                row = rows[0]
                if row.analyzer_version != ANALYZER_VERSION:
                    # Stats built by a different analyzer are worse than no
                    # stats: stems would not line up with query stems.
                    logger.warning(
                        "Corpus analyzer version %s != function version %s; "
                        "ignoring precomputed stats. Redeploy ingestion or "
                        "this function so the versions match.",
                        row.analyzer_version,
                        ANALYZER_VERSION,
                    )
                else:
                    meta = _CorpusMeta(
                        n_chunks=int(row.n_chunks or 0),
                        avg_token_count=float(row.avg_token_count or 0.0),
                        analyzer=SwedishAnalyzer(
                            lexicon=lexicon_from_json(row.lexicon_json)
                        ),
                    )
        except Exception:
            logger.exception("Failed to load SearchCorpusMeta; degrading.")

    _meta_cache = (now, meta)
    return meta


def _load_term_document_frequencies(
    client: FoundryClient, terms: Sequence[str]
) -> Dict[str, int]:
    """Point-read document frequencies for the query's terms."""
    if SearchTermStats is None or not terms:
        return {}
    try:
        term_prop = SearchTermStats.object_type.term
        term_filter = reduce(or_, (term_prop == term for term in terms))
        rows = client.ontology.objects.SearchTermStats.where(term_filter).take(
            num_items=len(terms)
        )
        return {
            row.term: int(row.document_frequency or 0)
            for row in rows
            if row.term
        }
    except Exception:
        logger.exception("Failed to load SearchTermStats; degrading.")
        return {}


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------

def _optional_chunk_property(name: str):
    """Ontology property handle, or None if the property does not exist."""
    try:
        return getattr(DocumentChunk.object_type, name)
    except Exception:
        return None


def _properties_to_load() -> list:
    properties = [
        DocumentChunk.object_type.chunk_id,
        DocumentChunk.object_type.chunked_text,
        DocumentChunk.object_type.plan_reference,
        DocumentChunk.object_type.chunk_number,
        DocumentChunk.object_type.type,
    ]
    for name in ("word_frequency", "token_count"):
        prop = _optional_chunk_property(name)
        if prop is not None:
            properties.append(prop)
    return properties


def _dense_candidates(
    client: FoundryClient, query: str, count: int
) -> Tuple[List[str], Dict[str, DocumentChunk], Dict[str, float]]:
    """Vector search: ranked chunk ids, chunks by id, similarity scores."""
    with AllowBetaFeatures():  # nearest_neighbors is beta in the OSDK
        chunks = list(
            client.ontology.objects.DocumentChunk.nearest_neighbors(
                query=query,
                num_neighbors=count,
                vector_property=DocumentChunk.object_type.chunked_embedding,
            ).take(num_items=count, properties=_properties_to_load())
        )

    ranked_ids: List[str] = []
    by_id: Dict[str, DocumentChunk] = {}
    scores: Dict[str, float] = {}
    for chunk in chunks:
        if not chunk.chunk_id or chunk.chunk_id in by_id:
            continue
        ranked_ids.append(chunk.chunk_id)
        by_id[chunk.chunk_id] = chunk
        similarity = chunk.get_score()
        if similarity is not None:
            scores[chunk.chunk_id] = float(similarity)
    return ranked_ids, by_id, scores


def _surface_tokens(query: str) -> List[str]:
    """Unstemmed lowercase tokens, deduplicated, for filtering raw chunkedText.

    Stemmed terms would not literally occur in the raw text, so the fallback
    filter must use surface forms.
    """
    seen: Set[str] = set()
    tokens: List[str] = []
    for word in re.findall(r"[\wåäöÅÄÖ]+", query.lower()):
        if len(word) < 2 or word in seen:
            continue
        seen.add(word)
        tokens.append(word)
    return tokens


def _sparse_candidate_chunks(
    client: FoundryClient,
    query: str,
    analyzed_terms: Sequence[str],
    count: int,
) -> List[DocumentChunk]:
    """Fetch keyword candidates via term filter on the best available property.

    Prefers the ingestion-produced ``analyzedText`` (query stems match chunk
    stems, compounds match their parts); falls back to raw ``chunkedText``
    with unstemmed tokens when enrichment is not deployed.
    """
    analyzed_text_prop = _optional_chunk_property("analyzed_text")
    if analyzed_text_prop is not None and analyzed_terms:
        filter_prop, filter_terms = analyzed_text_prop, list(analyzed_terms)
    else:
        filter_prop = DocumentChunk.object_type.chunked_text
        filter_terms = _surface_tokens(query)

    if not filter_terms:
        return []
    term_filter = filter_prop.contains_any_term(filter_terms[:_MAX_FILTER_TERMS])
    return list(
        client.ontology.objects.DocumentChunk.where(term_filter).take(
            num_items=count, properties=_properties_to_load()
        )
    )


# ---------------------------------------------------------------------------
# BM25 rescoring
# ---------------------------------------------------------------------------

def _chunk_term_counts(
    chunk: DocumentChunk, analyzer: SwedishAnalyzer
) -> Mapping[str, int]:
    """Precomputed word_frequency JSON, or analyze the raw text on the fly."""
    payload = getattr(chunk, "word_frequency", None)
    if payload:
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                return {
                    str(term): int(count)
                    for term, count in data.items()
                    if isinstance(count, (int, float)) and count > 0
                }
        except (TypeError, ValueError):
            logger.warning("Bad word_frequency JSON on chunk %s", chunk.chunk_id)
    return analyzer.term_counts(chunk.chunked_text)


def _bm25_ranking(
    client: FoundryClient,
    analyzer: SwedishAnalyzer,
    corpus_meta: Optional[_CorpusMeta],
    query_terms: Sequence[str],
    candidates: Sequence[DocumentChunk],
) -> Tuple[List[str], Dict[str, DocumentChunk], Dict[str, float]]:
    """Rescore sparse candidates with BM25; returns ranked ids/chunks/scores."""
    prepared: List[Tuple[str, Mapping[str, int], int]] = []
    by_id: Dict[str, DocumentChunk] = {}
    for chunk in candidates:
        if not chunk.chunk_id or chunk.chunk_id in by_id:
            continue
        counts = _chunk_term_counts(chunk, analyzer)
        doc_len = int(getattr(chunk, "token_count", None) or 0) or sum(counts.values())
        by_id[chunk.chunk_id] = chunk
        prepared.append((chunk.chunk_id, counts, doc_len))

    if corpus_meta is not None and corpus_meta.n_chunks > 0:
        dfs = _load_term_document_frequencies(client, query_terms)
        candidate_dfs = bm25.stats_from_candidates(
            [(doc_id, counts) for doc_id, counts, _ in prepared], query_terms
        ).document_frequency
        # A term seen in candidates but absent from (possibly stale) stats
        # must not get an unrealistically huge IDF.
        stats = bm25.CorpusStats(
            n_docs=corpus_meta.n_chunks,
            avg_doc_len=corpus_meta.avg_token_count,
            document_frequency={
                term: max(dfs.get(term, 0), candidate_dfs.get(term, 0))
                for term in query_terms
            },
        )
    else:
        stats = bm25.stats_from_candidates(
            [(doc_id, counts) for doc_id, counts, _ in prepared], query_terms
        )

    ranked = bm25.rank_candidates(query_terms, prepared, stats)
    return (
        [doc.doc_id for doc in ranked],
        by_id,
        {doc.doc_id: doc.score for doc in ranked},
    )


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def _clamped_int(value: Integer, fallback: int, minimum: int = 1, maximum: int = 500) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(parsed, maximum))


def _non_negative_float(value: Double, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(parsed) or parsed < 0.0:
        return fallback
    return parsed


# ---------------------------------------------------------------------------
# The function
# ---------------------------------------------------------------------------

@function(api_name="hybridDocumentSearch")
def hybrid_document_search(
    query: str,
    max_results: Integer = 10,
    dense_candidate_count: Integer = 50,
    sparse_candidate_count: Integer = 150,
    rrf_k: Integer = 60,
    dense_weight: Double = 1.0,
    sparse_weight: Double = 1.0,
) -> list[HybridSearchResult]:
    """Hybrid dense+sparse chunk search fused with weighted RRF.

    ``sparse_candidate_count`` is intentionally larger than the dense count:
    the term filter is recall-oriented and BM25 rescoring is what orders it,
    so giving the rescorer more candidates costs little and helps accuracy.
    """
    normalized_query = (query or "").strip()
    if not normalized_query:
        return []

    max_results_i = _clamped_int(max_results, fallback=10, maximum=100)
    dense_count = _clamped_int(dense_candidate_count, fallback=50)
    sparse_count = _clamped_int(sparse_candidate_count, fallback=150)
    rrf_k_i = _clamped_int(rrf_k, fallback=60, minimum=0, maximum=10_000)
    dense_weight_f = _non_negative_float(dense_weight, fallback=1.0)
    sparse_weight_f = _non_negative_float(sparse_weight, fallback=1.0)

    client = FoundryClient()
    corpus_meta = _load_corpus_meta(client)
    analyzer = corpus_meta.analyzer if corpus_meta else SwedishAnalyzer()
    query_terms = analyzer.unique_terms(normalized_query)

    dense_ids, dense_chunks, dense_scores = _dense_candidates(
        client, normalized_query, dense_count
    )

    sparse_ids: List[str] = []
    sparse_chunks: Dict[str, DocumentChunk] = {}
    sparse_scores: Dict[str, float] = {}
    if query_terms:
        candidates = _sparse_candidate_chunks(
            client, normalized_query, query_terms, sparse_count
        )
        sparse_ids, sparse_chunks, sparse_scores = _bm25_ranking(
            client, analyzer, corpus_meta, query_terms, candidates
        )

    dense_ranks = to_ranks(dense_ids)
    sparse_ranks = to_ranks(sparse_ids)
    fused = reciprocal_rank_fusion(
        rankings={"dense": dense_ranks, "sparse": sparse_ranks},
        weights={"dense": dense_weight_f, "sparse": sparse_weight_f},
        k=rrf_k_i,
    )

    chunks_by_id = {**sparse_chunks, **dense_chunks}
    logger.info(
        "hybridDocumentSearch: %d dense + %d sparse candidates -> %d fused "
        "(corpus stats: %s)",
        len(dense_ids),
        len(sparse_ids),
        len(fused),
        "precomputed" if corpus_meta else "candidate-estimated",
    )

    results: List[HybridSearchResult] = []
    for doc in fused[:max_results_i]:
        chunk = chunks_by_id.get(doc.doc_id)
        if chunk is None:
            continue
        results.append(
            HybridSearchResult(
                chunk_id=doc.doc_id,
                chunked_text=chunk.chunked_text,
                plan_reference=chunk.plan_reference,
                chunk_number=chunk.chunk_number,
                document_type=chunk.type,
                dense_rank=dense_ranks.get(doc.doc_id),
                sparse_rank=sparse_ranks.get(doc.doc_id),
                rrf_score=Double(doc.score),
                dense_score=(
                    Double(dense_scores[doc.doc_id])
                    if doc.doc_id in dense_scores
                    else None
                ),
                sparse_score=(
                    Double(sparse_scores[doc.doc_id])
                    if doc.doc_id in sparse_scores
                    else None
                ),
            )
        )
    return results
