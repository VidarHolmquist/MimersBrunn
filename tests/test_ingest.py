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
