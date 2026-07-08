# MimersBrunn — Hybrid RAG Retrieval for Swedish Technical Documentation

Hybrid (sparse + dense) document-chunk search for Palantir Foundry, split so
that all heavy computation happens **once at ingestion** and the per-query
Foundry Function stays fast:

```
INGESTION (once, Python transform)          RUNTIME (per query, Foundry Function)
────────────────────────────────           ─────────────────────────────────────
build Swedish decompounding lexicon         analyze query (same analyzer)
analyze every chunk (stem + decompound)     dense:  nearest_neighbors on embedding
store analyzed_text / word_frequency        sparse: term filter -> true BM25 rescore
compute corpus stats (df, avgdl, N)         fuse:   weighted Reciprocal Rank Fusion
        │                                           ▲
        └────────────── Ontology ───────────────────┘
     DocumentChunk (+3 properties), SearchTermStats, SearchCorpusMeta
```

## Repository layout

| Path | Deploys to | Purpose |
|---|---|---|
| `shared/swedish_analyzer.py` | **both** repos (copy, keep identical) | Lowercase → stopwords → compound splitting → Snowball stemming |
| `ingestion/ingest.py` | Python transform (Code Repository) | Per-chunk enrichment + corpus-level BM25 statistics |
| `retrieval/hybrid_search.py` | Python Functions repo | The `hybridDocumentSearch` function |
| `retrieval/bm25.py`, `retrieval/fusion.py` | Python Functions repo | Pure, unit-tested scoring/fusion logic |
| `tests/` | local / CI | `pip install -r requirements.txt && pytest` |

In Foundry, transforms and functions live in separate code repositories, so
`shared/swedish_analyzer.py` must exist in both. Its `ANALYZER_VERSION` is
written into the corpus stats at ingestion and checked at query time — a
version mismatch makes the function ignore stale statistics and log a
warning instead of silently returning stems that don't line up.

## Was the previous sparse implementation any good? (BM25 verdict)

The RRF plumbing and the dense side were sound. The sparse side had two real
problems, both fixed here:

1. **The ad-hoc score** (saturating TF × coverage ÷ √length) had no notion of
   term rarity. In technical documentation, rare terms ("fogemorfem",
   "impedansmatchning") are exactly what identifies the right chunk, while
   semi-common words ("system", "enhet") match everywhere. BM25's IDF handles
   this, and its `k1`/`b` parameters are battle-tested defaults rather than
   invented constants. A full BM25 needs corpus statistics — the reason the
   old code avoided it — so ingestion now precomputes document frequencies
   into `SearchTermStats` and the function point-reads only the query's
   terms (a handful of lookups per query).

2. **Recall for Swedish was capped by raw-text matching.** `contains_any_term`
   on unanalyzed text can't match "felsökning" against "felsökningsguiden" —
   no stemming, no compound splitting, and Swedish compounds by concatenation.
   The analyzer fixes this at ingestion: chunks are stored with an
   `analyzed_text` property (stemmed, compounds expanded into whole + parts),
   and the query is analyzed identically, so the term filter and BM25 operate
   in the same normalized space.

The function **degrades gracefully**: deployed against today's ontology (no
new properties/objects) it falls back to raw-text filtering and estimates
BM25 statistics from the candidate set — already better than the ad-hoc
score. Deploying the ingestion enrichment upgrades it to exact corpus-level
BM25 with no changes to the function.

## Foundry setup

### 1. Ingestion (Python transform repo)

Copy `shared/` and `ingestion/` into a transforms Code Repository, add
`snowballstemmer` to the environment, and wire `enrich_corpus` into a
transform (a template is in the `ingest.py` docstring). It outputs three
datasets: enriched chunks, term statistics, corpus meta. Run it as a **full
snapshot** (the lexicon and document frequencies are corpus-global); BM25
tolerates slightly stale stats, so a nightly rebuild with incremental chunk
additions in between is fine.

### 2. Ontology (Ontology Manager)

* `DocumentChunk` gains three properties backed by the enriched dataset:
  `analyzedText` (String — **enable full-text/keyword indexing**),
  `wordFrequency` (String, JSON), `tokenCount` (Integer).
* New object type `SearchTermStats`: `term` (String, primary key),
  `documentFrequency` (Integer).
* New object type `SearchCorpusMeta`: `metaId` (String, primary key —
  single row `"corpus"`), `nChunks` (Integer), `avgTokenCount` (Double),
  `analyzerVersion` (String), `lexiconJson` (String).

### 3. Retrieval (Python Functions repo)

Copy `shared/` and `retrieval/` into the functions repository, add
`snowballstemmer`, regenerate the OSDK after the ontology changes, and
publish `hybridDocumentSearch`. Corpus meta (including the decompounding
lexicon) is cached in the warm worker for 10 minutes, so steady-state
per-query overhead is: 1 vector search + 1 filtered object query + 1
term-stats point read.

### Tuning knobs

| Parameter | Default | Notes |
|---|---|---|
| `dense_candidate_count` | 50 | Depth of the vector list entering fusion |
| `sparse_candidate_count` | 150 | Recall-oriented; BM25 rescoring orders it, so more is cheap |
| `rrf_k` | 60 | Higher = flatter fusion, rewards appearing in both lists |
| `dense_weight` / `sparse_weight` | 1.0 | Shift toward sparse for exact-terminology queries, dense for paraphrased questions |
| BM25 `k1`=1.5, `b`=0.75 | in `bm25.py` | Standard defaults; only tune with an eval set |

## Roadmap — what to add next, in order of expected impact

1. **Evaluation harness first.** Collect 50–100 real Swedish queries with
   labeled relevant chunks and track recall@k / MRR / nDCG. Every idea below
   is cheap to try and impossible to judge without this. Foundry's AIP Evals
   or a notebook over the function works.
2. **Cross-encoder reranking.** Retrieve the fused top ~50, rerank to top ~8
   with a reranker before generation. Options in Foundry: a hosted reranker
   via AIP model catalog, `BAAI/bge-reranker-v2-m3` (multilingual, handles
   Swedish) deployed as a Foundry model, or an LLM-as-reranker in AIP Logic.
   Typically the single largest accuracy gain after hybrid search.
3. **Embedding model choice.** Verify the model behind `chunkedEmbedding` is
   genuinely multilingual (e.g. `text-embedding-3-large`, `multilingual-e5-
   large`, or a Voyage multilingual model). A monolingual-English embedder
   silently cripples the dense side for Swedish.
4. **Parent-context retrieval.** Rank with small chunks but hand the LLM the
   surrounding page/section (`plan_reference` already points there). Small
   chunks retrieve better; larger contexts generate better.
5. **Query rewriting / expansion.** An AIP Logic step that expands the user
   question with synonyms/abbreviations ("ubåt" ↔ "undervattensbåt", unit
   spellings) before search, or HyDE (embed a hypothetical answer instead of
   the question) for the dense side.
6. **Metadata filtering.** Expose `document_type` / `plan_reference` filters
   as function parameters so callers can scope searches; filtered search is
   both faster and more precise.
7. **MMR diversification.** If top results are near-duplicate chunks from one
   document, apply maximal marginal relevance over the fused list before
   returning.
