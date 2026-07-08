# MimersBrunn — Full Codebase (pastable bundle)

GENERATED FILE — do not edit by hand; regenerate with
`python tools/make_bundle.py`.

This is the complete source of the hybrid Swedish RAG retrieval codebase,
for pasting into AIP Assist / an AI FDE session that cannot access the Git
repo. Use it together with `docs/AI_FDE_PROMPT.md`, which explains the
architecture and the four deployment phases; paste PART A during Phase 1
and PART B during Phase 3.

Rules for whoever receives this bundle:

- Every file is delimited by a `FILE:` heading giving its repo-relative
  path. Reproduce file contents exactly.
- The ONLY permitted edits are the import-path adjustments called out in
  each part's intro (Foundry code repositories have their own package
  layouts). Do not rename functions, parameters, properties, or the
  `hybridDocumentSearch` API name.
- `swedish_analyzer.py` is deployed twice (transforms repo and functions
  repo) and the two copies must be identical.
- Single external dependency for both repos: `snowballstemmer` (pure
  Python).

---

## PART A — Ingestion (paste into the Python TRANSFORMS Code Repository)

Both files go into the transforms repo. `ingest.py` imports the analyzer as `from shared.swedish_analyzer import ...`; adjust that import to wherever the files land in the repo's package structure (e.g. put both in one package and import as siblings). Change nothing else. Dependency: `snowballstemmer`.

### FILE: `shared/swedish_analyzer.py`

````python
"""Swedish text analyzer: lowercase + stopwords + compound splitting + stemming.

Swedish forms compounds by concatenation ("felsökningsguide" = felsökning +
guide), so a query for one part never lexically matches the compound unless
the analyzer splits it. This module provides:

- :class:`SwedishAnalyzer` — lowercases, drops stopwords, splits compounds
  against a lexicon, and stems with the Snowball Swedish stemmer. When a
  compound is split, the stemmed *whole* word is emitted as well as its
  parts, so an exact compound query still matches.
- :func:`build_lexicon` — derives the decompounding lexicon from the corpus
  itself, so no external Swedish dictionary is needed and domain terms are
  covered automatically.

THE ANALYZER CONTRACT: the exact same analyzer (same version, same lexicon)
must be applied to chunk text at ingestion and to queries at retrieval.
``ANALYZER_VERSION`` is stored with the corpus stats at ingestion and checked
at query time so a drifted deployment fails loudly instead of silently
returning bad matches.

This file is the single source of truth. In Foundry, the ingestion transform
repo and the retrieval functions repo are separate code repositories — copy
this file into both (or publish it as a shared library) and keep the copies
identical. Bump ``ANALYZER_VERSION`` on any behavioral change and re-run the
full ingestion pipeline.

Requires the ``snowballstemmer`` package (pure Python, no binary deps).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

import snowballstemmer

# Bump on ANY change that affects token output (stopwords, splitting rules,
# stemmer, regex). Ingestion stores this with the corpus; retrieval refuses
# to trust precomputed stats built with a different version.
ANALYZER_VERSION = "sv-1"

_WORD_RE = re.compile(r"[\wåäöÅÄÖ]+", re.UNICODE)

# Deliberately small: over-aggressive stopword lists hurt technical
# documentation, where words like "under"/"över" can be meaningful.
SWEDISH_STOPWORDS: Set[str] = {
    "och", "i", "att", "det", "som", "en", "ett", "på", "är", "av", "för",
    "med", "till", "den", "de", "har", "inte", "om", "man", "kan", "ska",
    "skall", "vid", "eller", "från", "så", "vi", "du", "ni", "sig", "sin",
    "sitt", "sina", "denna", "detta", "dessa", "vara", "blir", "bli", "då",
    "när", "här", "där", "efter", "innan", "samt", "även", "också",
}

# Swedish technical docs mix in English; drop the highest-frequency English
# function words too.
ENGLISH_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "with",
}

DEFAULT_STOPWORDS: Set[str] = SWEDISH_STOPWORDS | ENGLISH_STOPWORDS


def build_lexicon(
    texts: Iterable[str],
    min_count: int = 3,
    min_length: int = 4,
    stopwords: Optional[Set[str]] = None,
) -> Set[str]:
    """Derive a decompounding lexicon from the corpus itself.

    Words that occur standalone at least ``min_count`` times are considered
    valid compound parts. The corpus's own vocabulary is exactly the
    vocabulary users will query with, and it automatically includes domain
    terms no general dictionary knows.

    Returns *stemmed* forms; :class:`SwedishAnalyzer` stems candidate parts
    before lookup, which also neutralizes the compound linking "s"
    ("arbets-" and "arbete" both stem to "arbet").
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    stemmer = snowballstemmer.stemmer("swedish")

    counts: Counter = Counter()
    for text in texts:
        for token in _WORD_RE.findall(text.lower()):
            if len(token) >= min_length and token not in stopwords:
                counts[token] += 1

    return {
        stemmer.stemWord(token)
        for token, count in counts.items()
        if count >= min_count
    }


def lexicon_to_json(lexicon: Set[str]) -> str:
    """Serialize a lexicon for storage (e.g. on the corpus meta object)."""
    return json.dumps(sorted(lexicon), ensure_ascii=False)


def lexicon_from_json(payload: Optional[str]) -> Set[str]:
    """Inverse of :func:`lexicon_to_json`; tolerates None/empty/bad input."""
    if not payload:
        return set()
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return set()
    return {str(item) for item in data} if isinstance(data, list) else set()


class SwedishAnalyzer:
    """Tokenizer for both ingestion and query analysis.

    Pipeline per token: lowercase -> stopword filter -> compound split
    (if a lexicon is given) -> Snowball stem. When a compound is split, the
    stemmed whole word is emitted as well as its parts, so an exact compound
    query still matches ("felsökningsguide" emits
    ["felsökningsguid", "felsökning", "guid"]).

    Args:
        lexicon: Stemmed valid compound parts, from :func:`build_lexicon`.
            Without it, no decompounding happens (stemming still does).
        stopwords: Surface-form words to drop. Defaults to
            :data:`DEFAULT_STOPWORDS`.
        min_part_length: Shortest surface substring considered a compound
            part. Below 4 the splitter starts inventing nonsense splits.
        max_parts: Give up on splits needing more than this many parts;
            genuine Swedish compounds rarely exceed three.
    """

    def __init__(
        self,
        lexicon: Optional[Set[str]] = None,
        stopwords: Optional[Set[str]] = None,
        min_part_length: int = 4,
        max_parts: int = 3,
    ) -> None:
        self._lexicon = lexicon or set()
        self._stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self._min_part = min_part_length
        self._max_parts = max_parts
        self._stemmer = snowballstemmer.stemmer("swedish")

    def tokenize(self, text: Optional[str]) -> List[str]:
        """Analyzed tokens for a text, compounds expanded, in reading order."""
        if not text:
            return []
        output: List[str] = []
        for token in _WORD_RE.findall(text.lower()):
            if len(token) < 2 or token in self._stopwords:
                continue
            output.append(self._stemmer.stemWord(token))
            parts = self._decompound(token)
            if parts is not None:
                output.extend(parts)
        return output

    def unique_terms(self, text: Optional[str]) -> List[str]:
        """Analyzed tokens, deduplicated, first-seen order. For queries."""
        seen: Set[str] = set()
        terms: List[str] = []
        for term in self.tokenize(text):
            if term not in seen:
                seen.add(term)
                terms.append(term)
        return terms

    def term_counts(self, text: Optional[str]) -> Dict[str, int]:
        """Term -> frequency map. For ingestion (stored as word_frequency)."""
        return dict(Counter(self.tokenize(text)))

    def _decompound(self, token: str) -> Optional[List[str]]:
        """Split ``token`` into known parts; None if it doesn't decompose.

        Dynamic programming over split positions, preferring the split with
        the fewest parts (fewer, longer parts are more likely to be the
        linguistically correct reading than many short ones).
        """
        if not self._lexicon or len(token) < 2 * self._min_part:
            return None

        n = len(token)
        # best[i] = (number of parts, parts list) for token[:i], or None.
        best: List[Optional[Tuple[int, List[str]]]] = [None] * (n + 1)
        best[0] = (0, [])

        for end in range(self._min_part, n + 1):
            for start in range(0, end - self._min_part + 1):
                if best[start] is None:
                    continue
                n_parts = best[start][0] + 1
                if n_parts > self._max_parts:
                    continue
                if best[end] is not None and best[end][0] <= n_parts:
                    continue
                part_stem = self._lookup(token[start:end])
                if part_stem is not None:
                    best[end] = (n_parts, best[start][1] + [part_stem])

        if best[n] is None or best[n][0] < 2:
            return None
        return best[n][1]

    def _lookup(self, part: str) -> Optional[str]:
        """Return the stemmed lexicon entry for a surface part, if any.

        Stemming before lookup also absorbs the compound linking "s"
        (fogemorfem): "arbets" stems to "arbet", same as "arbete". As a
        fallback, an explicit trailing "s" is stripped and retried, for
        joints the stemmer does not remove.
        """
        stem = self._stemmer.stemWord(part)
        if stem in self._lexicon:
            return stem
        if part.endswith("s"):
            stem = self._stemmer.stemWord(part[:-1])
            if stem in self._lexicon:
                return stem
        return None
````

### FILE: `ingestion/ingest.py`

````python
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
````

---

## PART B — Runtime (paste into the Python FUNCTIONS Code Repository)

All four files go into the functions repo, plus a second identical copy of `shared/swedish_analyzer.py` from PART A (it must stay byte-identical to the transforms copy — `ANALYZER_VERSION` guards drift at runtime). `hybrid_search.py` imports `from retrieval import bm25`, `from retrieval.fusion import ...` and `from shared.swedish_analyzer import ...`; adjust those import paths to the repo's package structure, change nothing else. Dependency: `snowballstemmer`.

### FILE: `retrieval/__init__.py`

````python
"""Runtime retrieval package: BM25 rescoring, RRF fusion, and the Foundry
Function (``hybrid_search``). ``hybrid_search`` imports Foundry SDKs and is
only importable inside a Foundry Functions repository; ``bm25`` and
``fusion`` are pure Python and unit-tested locally."""
````

### FILE: `retrieval/bm25.py`

````python
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
````

### FILE: `retrieval/fusion.py`

````python
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
````

### FILE: `retrieval/hybrid_search.py`

````python
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
````

---

## PART C — Unit tests (optional; run locally or in CI, not in Foundry)

Pure-Python tests for the analyzer, BM25, fusion, and ingestion logic. They document expected behavior; `pytest` from the repo root runs them (29 tests).

### FILE: `tests/test_swedish_analyzer.py`

````python
from shared.swedish_analyzer import (
    SwedishAnalyzer,
    build_lexicon,
    lexicon_from_json,
    lexicon_to_json,
)


def make_analyzer() -> SwedishAnalyzer:
    corpus = [
        "felsökning av radiosystemet kräver systematisk felsökning",
        "felsökning i fält görs enligt guide",
        "en guide för underhåll och en guide för installation",
        "guide till felsökning",
    ]
    return SwedishAnalyzer(lexicon=build_lexicon(corpus, min_count=2))


def test_lowercases_and_drops_stopwords():
    analyzer = SwedishAnalyzer()
    tokens = analyzer.tokenize("Detta ÄR en Radio och en Antenn")
    assert "detta" not in tokens
    assert "och" not in tokens
    assert "radio" in tokens


def test_stemming_conflates_inflections():
    analyzer = SwedishAnalyzer()
    assert analyzer.tokenize("radiosystemet") == analyzer.tokenize("radiosystem")


def test_compound_emits_whole_and_parts():
    analyzer = make_analyzer()
    tokens = analyzer.tokenize("felsökningsguide")
    # The stemmed whole word plus both parts.
    assert len(tokens) == 3
    whole, part_a, part_b = tokens
    assert part_a in analyzer.tokenize("felsökning")
    assert part_b in analyzer.tokenize("guide")
    # Query for one part matches the compound document.
    assert set(analyzer.tokenize("felsökning")) & set(tokens)


def test_no_lexicon_means_no_decompounding():
    analyzer = SwedishAnalyzer()
    assert len(analyzer.tokenize("felsökningsguide")) == 1


def test_unique_terms_preserve_order():
    analyzer = SwedishAnalyzer()
    assert analyzer.unique_terms("radio antenn radio") == analyzer.unique_terms(
        "radio antenn"
    )


def test_term_counts_counts_expanded_tokens():
    analyzer = make_analyzer()
    counts = analyzer.term_counts("felsökningsguide felsökningsguide")
    assert all(count == 2 for count in counts.values())
    assert len(counts) == 3


def test_lexicon_json_roundtrip():
    lexicon = {"felsökning", "guid", "antenn"}
    assert lexicon_from_json(lexicon_to_json(lexicon)) == lexicon
    assert lexicon_from_json(None) == set()
    assert lexicon_from_json("not json") == set()


def test_empty_and_none_input():
    analyzer = SwedishAnalyzer()
    assert analyzer.tokenize(None) == []
    assert analyzer.tokenize("") == []
    assert analyzer.term_counts(None) == {}
````

### FILE: `tests/test_bm25.py`

````python
from retrieval import bm25


STATS = bm25.CorpusStats(
    n_docs=1000,
    avg_doc_len=100.0,
    document_frequency={"radio": 50, "antenn": 5, "system": 900},
)


def test_idf_favors_rare_terms():
    assert bm25.idf(1000, 5) > bm25.idf(1000, 50) > bm25.idf(1000, 900)


def test_idf_never_negative_and_zero_docs_safe():
    assert bm25.idf(10, 10) >= 0.0
    assert bm25.idf(0, 5) == 0.0


def test_rare_term_match_outscores_common_term_match():
    rare = bm25.score(["antenn"], {"antenn": 1}, 100, STATS)
    common = bm25.score(["system"], {"system": 1}, 100, STATS)
    assert rare > common


def test_term_frequency_saturates():
    one = bm25.score(["radio"], {"radio": 1}, 100, STATS)
    five = bm25.score(["radio"], {"radio": 5}, 100, STATS)
    fifty = bm25.score(["radio"], {"radio": 50}, 100, STATS)
    assert one < five < fifty
    assert (five - one) > (fifty - five) / 9  # diminishing returns


def test_length_normalization_prefers_shorter_doc_at_equal_tf():
    short = bm25.score(["radio"], {"radio": 2}, 50, STATS)
    long = bm25.score(["radio"], {"radio": 2}, 400, STATS)
    assert short > long


def test_no_overlap_scores_zero():
    assert bm25.score(["antenn"], {"radio": 3}, 100, STATS) == 0.0


def test_rank_candidates_orders_and_drops_zero():
    candidates = [
        ("doc_common", {"system": 3}, 100),
        ("doc_rare", {"antenn": 1}, 100),
        ("doc_none", {"kabel": 5}, 100),
    ]
    ranked = bm25.rank_candidates(["antenn", "system"], candidates, STATS)
    assert [doc.doc_id for doc in ranked] == ["doc_rare", "doc_common"]


def test_rank_candidates_ties_break_deterministically():
    candidates = [
        ("doc_b", {"radio": 1}, 100),
        ("doc_a", {"radio": 1}, 100),
    ]
    ranked = bm25.rank_candidates(["radio"], candidates, STATS)
    assert [doc.doc_id for doc in ranked] == ["doc_a", "doc_b"]


def test_stats_from_candidates():
    stats = bm25.stats_from_candidates(
        [("a", {"radio": 2, "antenn": 1}), ("b", {"radio": 1})],
        ["radio", "antenn", "saknas"],
    )
    assert stats.n_docs == 2
    assert stats.avg_doc_len == 2.0
    assert stats.document_frequency == {"radio": 2, "antenn": 1, "saknas": 0}
````

### FILE: `tests/test_fusion.py`

````python
import pytest

from retrieval.fusion import reciprocal_rank_fusion, to_ranks


def test_to_ranks_first_occurrence_wins():
    assert to_ranks(["a", "b", "a", "c"]) == {"a": 1, "b": 2, "c": 4}


def test_doc_in_both_lists_beats_single_list_top():
    fused = reciprocal_rank_fusion(
        rankings={
            "dense": to_ranks(["only_dense", "both"]),
            "sparse": to_ranks(["only_sparse", "both"]),
        },
        weights={"dense": 1.0, "sparse": 1.0},
        k=60,
    )
    assert fused[0].doc_id == "both"
    assert fused[0].ranks == {"dense": 2, "sparse": 2}


def test_weights_shift_the_ranking():
    rankings = {
        "dense": to_ranks(["d1", "d2"]),
        "sparse": to_ranks(["s1", "s2"]),
    }
    sparse_heavy = reciprocal_rank_fusion(
        rankings, weights={"dense": 0.5, "sparse": 2.0}, k=60
    )
    assert sparse_heavy[0].doc_id == "s1"


def test_zero_weight_disables_a_retriever():
    fused = reciprocal_rank_fusion(
        rankings={"dense": to_ranks(["d1"]), "sparse": to_ranks(["s1"])},
        weights={"dense": 1.0, "sparse": 0.0},
        k=60,
    )
    assert [doc.doc_id for doc in fused] == ["d1"]


def test_ties_break_on_best_rank_then_id():
    fused = reciprocal_rank_fusion(
        rankings={"dense": to_ranks(["b", "a"]), "sparse": to_ranks(["a", "b"])},
        weights={},
        k=60,
    )
    # Identical scores and best ranks -> alphabetical.
    assert [doc.doc_id for doc in fused] == ["a", "b"]


def test_negative_k_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(rankings={}, weights={}, k=-1)
````

### FILE: `tests/test_ingest.py`

````python
import json

from ingestion.ingest import CORPUS_META_ID, enrich_corpus
from shared.swedish_analyzer import ANALYZER_VERSION


CORPUS = [
    ("c1", "Felsökning av radiosystemet enligt guide"),
    ("c2", "Guide för installation av antenn"),
    ("c3", "Felsökning och underhåll av antenn i fält, guide finns"),
    ("c4", None),
]


def test_enrich_corpus_produces_all_outputs():
    result = enrich_corpus(CORPUS)

    assert [chunk.chunk_id for chunk in result.chunks] == ["c1", "c2", "c3", "c4"]
    assert result.meta.meta_id == CORPUS_META_ID
    assert result.meta.n_chunks == 4
    assert result.meta.analyzer_version == ANALYZER_VERSION
    assert result.meta.avg_token_count > 0


def test_word_frequency_matches_analyzed_text():
    result = enrich_corpus(CORPUS)
    for chunk in result.chunks:
        counts = json.loads(chunk.word_frequency_json)
        tokens = chunk.analyzed_text.split()
        assert sum(counts.values()) == len(tokens) == chunk.token_count


def test_document_frequency_counts_chunks_not_occurrences():
    result = enrich_corpus(CORPUS)
    dfs = {row.term: row.document_frequency for row in result.term_stats}
    # "guide" (stemmed) appears in three chunks; c3 mentions it once despite
    # also being counted for "felsökning".
    guide_stem = [term for term in dfs if term.startswith("guid")]
    assert guide_stem and dfs[guide_stem[0]] == 3
    assert all(df >= 1 for df in dfs.values())
    assert all(df <= result.meta.n_chunks for df in dfs.values())


def test_empty_text_chunk_gets_zero_tokens():
    result = enrich_corpus(CORPUS)
    empty = result.chunks[3]
    assert empty.token_count == 0
    assert empty.analyzed_text == ""
    assert json.loads(empty.word_frequency_json) == {}


def test_frozen_lexicon_is_reused():
    frozen = {"felsökning", "guid"}
    result = enrich_corpus(CORPUS, lexicon=frozen)
    assert set(json.loads(result.meta.lexicon_json)) == frozen


def test_empty_corpus():
    result = enrich_corpus([])
    assert result.chunks == []
    assert result.term_stats == []
    assert result.meta.n_chunks == 0
    assert result.meta.avg_token_count == 0.0
````
