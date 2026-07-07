# Foundry Integration Guide — Hybrid RAG for Swedish Technical Documentation

Step-by-step instructions for deploying the MimersBrunn hybrid retrieval
pipeline (BM25 + vector search + RRF + page-level reranking) in Palantir
Foundry. Each step names the specific Foundry tool to use.

Target architecture:

```
PIPELINE (offline)                 ONTOLOGY                        FUNCTION (per query)
docs → extract → chunk → embed  →  Chunk / Page object types    →  hybrid search + RRF
(Pipeline Builder + transforms)    (vector + full-text indexes)     (Functions on Objects)
                                                                    ↑ AIP Logic / Agent Studio
```

---

## Phase 0 — Prerequisites (ask your Foundry admin)

1. **Confirm AIP is enabled** on the enrollment and check which
   embedding/LLM models are available in the model catalog
   (AIP Settings → Models). You need at least one multilingual embedding
   model (e.g. `text-embedding-3-large`).
2. **Ask whether the full-text search index supports a Swedish analyzer**
   (stemming + compound splitting). This determines how much the sparse
   side helps for Swedish compound words.
3. If model fine-tuning or a self-hosted reranker is planned later:
   confirm **GPU availability for Jupyter Code Workspaces**.

## Phase 1 — Ingestion pipeline

4. **Upload documents to a Media Set** (Foundry's resource type for
   unstructured files like PDFs). Keep one media set per document
   collection.
5. **Extract text with Pipeline Builder**: add the media set as input,
   use the *Get media references* board, then the *Text extraction*
   transform (`Extract text from PDF`, or the OCR variant for scanned
   documents). Output: one row per page with `document_id`, `page_number`,
   `page_text`. For complex layouts (tables, figures), evaluate
   **AIP Document Intelligence** instead — it orchestrates multiple
   extraction strategies with evaluation.
6. **Write the `pages` dataset** from step 5: columns `page_id`
   (= `document_id` + `page_number`), `document_id`, `title`,
   `page_number`, `full_text`. This is what the LLM will receive at
   generation time.
7. **Chunk the pages** in a **Python transform (Code Repositories)** —
   Pipeline Builder works for simple splitting, but chunking logic
   (target ~200–500 tokens, overlap, respect section boundaries) is easier
   to maintain in code. Output one row per chunk with `chunk_id`
   (primary key, e.g. `page_id-chunkIndex`), `page_id`, `source`,
   `title`, `chunk_index`, `text`.
8. **Tokenize for sparse search** in the same or a following transform:
   apply the Swedish analyzer (lowercase + Snowball Swedish stemming +
   compound splitting) to `text` and store the result as a JSON string
   column `word_frequency`. Record the analyzer version in an
   `analyzer_version` column. *The exact same analyzer must later be
   applied to queries.*
9. **Embed the chunks** in a Python transform using
   `palantir_models.transforms.GenericEmbeddingModelInput` — batch all
   chunk texts through the chosen embedding model, write an `embedding`
   array column and an `embedding_model` version column.
10. **Make transforms incremental** (`@incremental` in transforms) so only
    new/changed documents are re-processed, and **create a schedule** for
    the pipeline. Exception: if the embedding model ever changes, force a
    full rebuild — mixed vector spaces fail silently.

## Phase 2 — Ontology (Ontology Manager)

11. **Create the `Chunk` object type** backed by the chunks dataset:

    | property | type | notes |
    |---|---|---|
    | `chunk_id` | String | primary key |
    | `page_id` | String | grouping key for page-level results |
    | `source` | String | parent document id/path |
    | `title` | String | section heading |
    | `text` | String | enable full-text/keyword search on this property |
    | `word_frequency` | String (JSON) | only needed for in-process BM25 |
    | `embedding` | **Vector** | dimension = model output (e.g. 1024/3072), similarity = cosine |
    | `language` | String | ISO code |
    | `embedding_model` | String | model + version tag |
    | `analyzer_version` | String | tokenizer version tag |
    | `chunk_index` | Integer | position within page |
    | `updated_at` | Timestamp | freshness |

    The **Vector property** is what makes Foundry build and maintain the
    approximate-nearest-neighbor index — configure its *dimension* and
    *similarity function* when creating the property.
12. **Create the `Page` object type** backed by the pages dataset:
    `page_id` (primary key), `document_id`, `title`, `full_text`,
    `page_number`.
13. **Create a link type `Chunk → Page`** on `page_id`, so the query
    function can traverse from hit chunks to their parent pages.

## Phase 3 — Query-time function (Code Repositories → Functions)

14. **Create a Functions on Objects repository** (TypeScript is the most
    mature path; Python functions also work). Import the Chunk and Page
    object types.
15. **Implement `hybridSearch(query: string, topK: Integer)`**:
    1. Embed the query with the same Palantir-provided model as ingestion
       (`@palantir/languagemodelservice` API).
    2. Dense candidates: `Objects.search().chunk()
       .nearestNeighbors(c => c.embedding.near(queryVec, {kValue: 100}))
       .orderByRelevance().take(100)`.
    3. Sparse candidates: keyword search on the `text` property,
       `.orderByRelevance().take(100)`.
    4. Fuse with RRF (port of `retrieval/fusion.py`, ~15 dependency-free
       lines): `score(id) = Σ 1/(60 + rank)`.
    5. Group fused chunks by `page_id`, keep max score per page (dedupe).
    6. Traverse Chunk→Page links; return top 5–10 pages with scores and
       matched-chunk diagnostics.
16. **Publish the function** (tag a version in the repository) so it can
    be referenced by AIP Logic, Agent Studio, and Workshop.

## Phase 4 — RAG assembly

17. **AIP Logic**: create a Logic function that calls `hybridSearch` as a
    tool/board, assembles the returned pages into the LLM prompt (token
    budget ~8–16k, best pages first), and generates the answer with an
    AIP-provided LLM. Alternatively use **Agent Studio** and register
    `hybridSearch` as the agent's retrieval tool.
18. **Front-end**: a **Workshop** app (or Developer Console/OSDK app) that
    calls the Logic function and renders answers with source pages for
    citation/validation.

## Phase 5 — Quality loop (after the baseline works)

19. **Build an evaluation set first**: 50–100 real Swedish queries with
    known correct pages. Measure recall@50 (are the right pages in the
    candidate pool?) and nDCG@10. Store it as a dataset; evaluate via a
    transform so every pipeline change gets scored.
20. **Add a reranker** once eval exists: host `BGE-reranker-v2-m3` as a
    Foundry **model asset** (train/import in a Jupyter Code Workspace,
    publish with a model adapter, serve via a **live deployment**), then
    call it in the function between steps 15.5 and 15.6 on the top ~50
    pages, scoring `(query, title + best chunk text)` pairs.
21. **Only then consider fine-tuning** the embedder or reranker on
    (query, chunk) pairs — synthetic queries can be generated per chunk
    with an AIP LLM in a transform. Fine-tune the reranker before the
    embedder: same training data, no corpus re-embedding on update.
    Justify with recall@50: fine-tune the embedder only if the right pages
    are missing from the candidate pool.

---

## Role of this repository

The `retrieval/` package is the local/prototype implementation of the same
pipeline: use it in a Jupyter Code Workspace against a sample of the chunks
dataset to validate chunking, embeddings, and fusion quality before
building Phases 2–4. `fusion.py` is the permanent core — port it verbatim
into the Phase 3 function. `BM25Index`/`BruteForceDenseIndex` are the
prototype backends that platform indexes replace at scale.

## Foundry documentation references

- Document processing for semantic search: palantir.com/docs/foundry/ontology/document-processing
- Pipeline Builder text extraction: palantir.com/docs/foundry/pb-functions-expression/pdfTextExtractionV1
- AIP Document Intelligence: palantir.com/docs/foundry/document-intelligence/overview
- Palantir-provided models in transforms: palantir.com/docs/foundry/transforms-python-spark/palantir-provided-models
- Semantic search overview (vector properties): palantir.com/docs/foundry/ontology/overview-semantic-search
- Semantic search workflow in Functions: palantir.com/docs/foundry/functions/using-palantir-provided-models-to-create-a-semantic-search-workflow/
- Functions on objects, object sets API: palantir.com/docs/foundry/functions/api-object-sets
- Train models in Jupyter Code Workspaces: palantir.com/docs/foundry/integrate-models/model-asset-code-workspaces
- GPU training: palantir.com/docs/foundry/model-integration/gpu-training
- Ontology-augmented generation: palantir.com/docs/foundry/ontology/ontology-augmented-generation
