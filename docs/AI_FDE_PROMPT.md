# Prompt for Palantir AIP Assist / AI FDE

Copy everything below the line into AIP Assist (or your AI FDE session).
It is self-contained — the assistant cannot see the GitHub repo, so all
context it needs is inlined. Work through it one phase at a time.

---

I am implementing a hybrid (sparse + dense) RAG retrieval system for Swedish
technical documentation in Foundry. The Python code already exists in an
external Git repo and I will paste files in as we go — your job is to guide
me through the Foundry-specific wiring: Code Repositories, Ontology Manager,
OSDK regeneration, and Functions publishing. Guide me **one phase at a time**,
give exact click-paths and code where needed, and wait for me to confirm each
phase works before moving to the next. Where my enrollment's capabilities
matter (beta features, model availability, index settings), tell me how to
check rather than assuming.

## Current state (already working)

- An object type `DocumentChunk` backed by a chunks dataset, with properties
  (API names): `chunkId` (String, primary key), `chunkedText` (String),
  `chunkedEmbedding` (Vector, populated by an embedding model),
  `planReference` (String), `chunkNumber` (Integer), `type` (String).
- A published Python Function `hybridDocumentSearch` that does
  `nearest_neighbors` over `chunkedEmbedding` plus a `contains_any_term`
  keyword filter, fused with reciprocal rank fusion. I am replacing its
  sparse side with true BM25.

## Target architecture

```
INGESTION (Python transform, runs once/nightly)   RUNTIME (Foundry Function, per query)
build Swedish decompounding lexicon from corpus    analyze query with the SAME analyzer
analyze chunks: stem + split compounds             dense:  nearest_neighbors on embedding
store analyzed_text / word_frequency / token_count sparse: term filter on analyzedText
compute corpus stats: doc frequencies, avg length          -> exact BM25 rescore
        │                                          fuse:   weighted RRF
        └───────────────── Ontology ───────────────────────┘
   DocumentChunk (+3 new properties), SearchTermStats, SearchCorpusMeta
```

The Python modules I will paste:

- `swedish_analyzer.py` — analyzer (Snowball Swedish stemming + corpus-derived
  compound splitting), must be byte-identical in the transforms repo and the
  functions repo; it exposes an `ANALYZER_VERSION` constant that ingestion
  stores and the function checks at runtime.
- `ingest.py` — pure function `enrich_corpus(id_text_pairs)` returning
  per-chunk enrichment rows, per-term document-frequency rows, and a
  singleton corpus-meta row. Needs a transform wrapper.
- `hybrid_search.py` + `bm25.py` + `fusion.py` — the runtime function. It
  degrades gracefully if the new ontology pieces don't exist yet, so it can
  be deployed first.
- Only external dependency: `snowballstemmer` (pure Python).

## Phase 1 — Ingestion transform

Help me, in a Python transforms Code Repository:

1. Add `snowballstemmer` to the repo environment (meta.yml / conda deps).
2. Create a transform that reads the existing chunks dataset (columns
   `chunk_id`, `chunked_text`), calls `enrich_corpus`, and writes THREE
   output datasets:
   - `chunks_enriched`: `chunk_id`, `analyzed_text` (String),
     `word_frequency` (String, JSON `{term: count}`), `token_count` (Integer)
     — to be joined/backing the DocumentChunk object type.
   - `search_term_stats`: `term` (String), `document_frequency` (Integer).
   - `search_corpus_meta`: single row — `meta_id` = `"corpus"`,
     `n_chunks` (Integer), `avg_token_count` (Double),
     `analyzer_version` (String), `lexicon_json` (String, can be ~100s of KB).
3. Configure it as a FULL SNAPSHOT transform (the lexicon and document
   frequencies are corpus-global — no incremental), and schedule it nightly.
4. Decide with me: should `chunks_enriched` be joined onto the existing
   chunks dataset in this transform (one backing dataset for DocumentChunk),
   or kept separate with a downstream join? Recommend the simpler option for
   my pipeline.

My corpus is Swedish technical documentation, size on the order of
[FILL IN: number of chunks], so tell me if anything here needs Spark-level
care versus a pandas/lightweight transform.

## Phase 2 — Ontology changes

Help me in Ontology Manager:

1. Add three properties to `DocumentChunk`, backed by the enriched dataset:
   - `analyzedText` (String) — **enable full-text / keyword indexing** on
     this property; it is the target of `contains_any_term` filters. Tell me
     which analyzer/index settings Object Storage v2 offers here and what to
     pick (the text is already pre-stemmed space-separated tokens, so I want
     plain whitespace/standard tokenization — no additional stemming).
   - `wordFrequency` (String — JSON payload, no indexing needed).
   - `tokenCount` (Integer).
2. Create object type `SearchTermStats`: `term` (String, primary key),
   `documentFrequency` (Integer). Backed by `search_term_stats`. It will be
   queried with exact-match filters on `term` (~10-25 terms per query), so
   confirm the primary key gives me efficient point lookups.
3. Create object type `SearchCorpusMeta`: `metaId` (String, primary key),
   `nChunks` (Integer), `avgTokenCount` (Double), `analyzerVersion` (String),
   `lexiconJson` (String). Backed by `search_corpus_meta` (one row).
4. Tell me how to verify indexing/sync has completed before I test.

## Phase 3 — Runtime function

Help me in the Python Functions Code Repository:

1. Add `snowballstemmer` to the environment.
2. Regenerate the ontology SDK so `DocumentChunk`'s new properties and the
   two new object types are available, and confirm the generated Python
   property names are the snake_case forms the code expects:
   `analyzed_text`, `word_frequency`, `token_count`, `term`,
   `document_frequency`, `meta_id`, `n_chunks`, `avg_token_count`,
   `analyzer_version`, `lexicon_json`.
3. The function uses `nearest_neighbors` under `AllowBetaFeatures()` — tell
   me how to confirm this beta is enabled for my enrollment/repo.
4. Publish the updated `hybridDocumentSearch` (same API name and signature:
   `query`, `max_results`, `dense_candidate_count`, `sparse_candidate_count`,
   `rrf_k`, `dense_weight`, `sparse_weight`) without breaking existing
   callers.

## Phase 4 — Verification

Walk me through:

1. Running test queries in the Functions preview — including a Swedish
   compound-word test: a query using one part of a compound (e.g.
   "felsökning") should now retrieve chunks containing the full compound
   (e.g. "felsökningsguiden").
2. Checking the function logs for the line that says corpus stats are
   `precomputed` (good) versus `candidate-estimated` (means Phase 1/2 pieces
   aren't visible to the function yet) and for any analyzer-version-mismatch
   warning.
3. Wiring the function into my AIP Logic / Agent Studio retrieval step.

## Constraints

- The two copies of `swedish_analyzer.py` (transforms repo, functions repo)
  must stay identical; `ANALYZER_VERSION` guards against drift at runtime.
- Do not change the function's API name or parameter names.
- The ingestion transform must remain a full snapshot.
- Also tell me: which embedding model is backing `chunkedEmbedding` in my
  setup, and is it multilingual? If it is English-only, flag it — that
  silently cripples the dense side for Swedish and I should switch models
  and re-embed.
