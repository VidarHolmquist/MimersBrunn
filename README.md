# MimersBrunn — Hybrid Retrieval

Hybrid sparse + dense retrieval over a corpus of text chunks, merged with
Reciprocal Rank Fusion (RRF). Pure Python, only external dependency is NumPy.

## Quick start

```python
from retrieval import Chunk, HybridRetriever

chunks = [
    Chunk(
        chunk_id="doc1-0",
        source="doc1.pdf",
        text="the quick brown fox ...",
        embedding=[0.1, 0.7, ...],           # from your embedding model
        word_frequency={"the": 2, "quick": 1, "brown": 1, "fox": 1},
    ),
    # ...
]

retriever = HybridRetriever(chunks, embedder=my_embedding_fn)

# Query string only — embedder is called for you:
results = retriever.search("where does the fox live?", top_k=10)

# Or supply a pre-computed query embedding (skips the embedder):
results = retriever.search("where does the fox live?",
                           query_embedding=my_vector, top_k=10)

for r in results:
    print(r.fused_score, r.chunk.source, r.chunk.text[:80])
    print("  sparse:", r.sparse_rank, r.sparse_score,
          " dense:", r.dense_rank, r.dense_score)
```

## How it works

```
query ──┬─> BM25Index (sparse, from word_frequency) ──> ranked list ─┐
        │                                                            ├─> RRF ─> top_k
        └─> embed ─> DenseIndex (cosine similarity)  ──> ranked list ─┘
```

1. **Sparse** (`retrieval/sparse.py`): BM25 built from the per-chunk
   `word_frequency` maps via an inverted index — query cost scales with the
   number of chunks containing the query terms, not with corpus size.
2. **Dense** (`retrieval/dense.py`): exact cosine similarity over an
   L2-normalized NumPy matrix (one matrix–vector product per query).
3. **Fusion** (`retrieval/fusion.py`): RRF, `score(d) = Σ w / (k + rank(d))`,
   with `k = 60` by default. RRF uses ranks only, so the incomparable BM25 and
   cosine score scales never need normalizing.

Each stage retrieves a candidate pool (default `max(50, 5 * top_k)`) before
fusion, so a chunk ranked moderately by *both* retrievers can beat one ranked
highly by only one — that is the point of hybrid search.

## Data model — what your chunks need

Required by this module:

| property         | why                                                            |
|------------------|----------------------------------------------------------------|
| `chunk_id`       | **You must add this if you don't have it.** A stable unique id is what joins the sparse and dense rankings during fusion. |
| `source`         | provenance, returned with results                              |
| `text`           | returned with results                                          |
| `embedding`      | dense search (same model/dimension for all chunks and queries) |
| `word_frequency` | sparse search (BM25 term frequencies)                          |

Strongly recommended additions:

- **Tokenizer consistency**: BM25 only works if queries are tokenized the same
  way `word_frequency` was built. Pass your ingestion tokenizer via the
  `tokenizer=` argument, or record the tokenizer name/version with the corpus.
- **Swedish corpora**: use `retrieval.tokenize_sv.SwedishTokenizer` — it
  stems (Snowball) and splits compounds ("felsökningsguide" also indexes as
  "felsökning" + "guide") against a lexicon derived from your own corpus via
  `build_lexicon()`. Persist the lexicon alongside the corpus and use the
  same tokenizer instance at ingestion and query time.
- **Embedding model version** in `metadata`: mixing vectors from different
  models silently breaks dense search.
- **Filterable metadata** (document type, date, language, ACLs) if you will
  ever need to restrict searches to a subset of the corpus.

Not needed: document length (derived from `word_frequency`), IDF/document
frequencies (computed at index build time).

## Scaling up

The in-memory defaults are fine to roughly **10⁵–10⁶ chunks** on one machine.
Past that, don't rewrite the pipeline — swap the index backends. Both are
small interfaces (`retrieval/sparse.py::SparseIndex`,
`retrieval/dense.py::DenseIndex`):

```python
class FoundryVectorIndex:               # example dense backend
    def search(self, query_embedding, top_k):
        hits = ...  # call FAISS / pgvector / Foundry vector search
        return [ScoredChunk(chunk_id=h.id, score=h.score) for h in hits]

retriever = HybridRetriever(chunks, dense_index=FoundryVectorIndex(...),
                            sparse_index=MySearchEngineIndex(...))
```

- **Dense**: FAISS / hnswlib (in-process ANN), or pgvector / a vector DB.
- **Sparse**: Elasticsearch / OpenSearch, or any engine that returns a ranked
  list for a keyword query.
- **Fusion**: RRF is O(pool size) and never the bottleneck; it stays as is.

### Palantir Foundry notes

Python is the right choice here — Foundry's transforms and Functions runtimes
are Python-first, and this package's minimal dependency footprint (NumPy only)
makes it easy to run inside them. When you move in:

- Keep `fusion.py` and `retriever.py` unchanged; implement `SparseIndex` /
  `DenseIndex` adapters over Foundry's own vector search / full-text search so
  the heavy lifting happens in the platform, not in Python memory.
- The `Chunk` schema maps 1:1 onto an ontology object type or dataset schema;
  `chunk_id` becomes the primary key.

## Why Python?

Python is the standard for retrieval/RAG pipelines: every embedding model,
ANN library (FAISS, hnswlib) and vector DB ships Python bindings first, and
Palantir Foundry's code runtimes are Python-first. The performance-critical
inner loops here are already vectorized NumPy or delegated to a pluggable
backend written in C++/Rust under the hood, so a lower-level language would
add friction without meaningful speedup. The only scenario where you'd
reconsider is building a standalone low-latency search *service* at very high
QPS — and even then the usual answer is a dedicated engine (Elasticsearch,
Vespa, Qdrant) behind these same interfaces, not rewriting this logic.

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```
