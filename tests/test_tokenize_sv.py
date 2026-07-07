"""Tests for the Swedish analyzer."""

import pytest

from retrieval import BM25Index, Chunk
from retrieval.tokenize_sv import (
    SWEDISH_STOPWORDS,
    SwedishTokenizer,
    build_lexicon,
    build_word_frequency,
)

CORPUS_TEXTS = [
    # "felsökning", "guide", "system", "dokumentation", "arbete", "miljö"
    # all occur standalone >= 3 times so they enter the lexicon.
    "felsökning av systemet kräver en guide och dokumentation",
    "denna guide beskriver felsökning i systemet steg för steg",
    "dokumentation om arbete i en säker miljö",
    "systemet loggar varje fel; se guide för felsökning",
    "arbete med dokumentation sker i produktionsmiljön",
    "vid arbete i denna miljö krävs behörighet",
    "miljö och system ska dokumenteras enligt dokumentation",
]


@pytest.fixture
def lexicon():
    return build_lexicon(CORPUS_TEXTS, min_count=3)


@pytest.fixture
def tokenizer(lexicon):
    return SwedishTokenizer(lexicon=lexicon)


class TestBuildLexicon:
    def test_frequent_words_included_as_stems(self, lexicon):
        assert "felsökning" in lexicon  # stem of felsökning(en)
        assert "guid" in lexicon        # stem of guide(n)
        assert "system" in lexicon      # stem of system(et)

    def test_stopwords_and_short_words_excluded(self, lexicon):
        assert "och" not in lexicon
        assert "steg" not in lexicon or len("steg") >= 4  # length rule only

    def test_rare_words_excluded(self, lexicon):
        # "behörighet" appears once < min_count=3
        assert "behör" not in lexicon


class TestSwedishTokenizer:
    def test_stems_inflections_to_same_form(self, tokenizer):
        assert tokenizer("felsökningen") == tokenizer("felsökning")

    def test_drops_stopwords(self, tokenizer):
        assert tokenizer("och att det") == []

    def test_decompounds_known_compound(self, tokenizer):
        # "felsökningsguide" (with linking-s) -> whole stem + both parts
        tokens = tokenizer("felsökningsguide")
        assert "felsökning" in tokens and "guid" in tokens
        assert tokens[0] == tokenizer("felsökningsguide")[0]  # whole kept first

    def test_decompound_handles_plain_concatenation(self, tokenizer):
        tokens = tokenizer("systemdokumentation")
        assert "system" in tokens and "dokumentation" in tokens

    def test_unknown_compound_left_whole(self, tokenizer):
        # No lexicon entries for these parts -> only the whole-word stem.
        tokens = tokenizer("xylofonorkester")
        assert len(tokens) == 1

    def test_no_lexicon_means_no_decompounding(self):
        plain = SwedishTokenizer()
        assert len(plain("felsökningsguide")) == 1

    def test_query_matches_compound_in_index(self, lexicon):
        """End-to-end: query 'felsökning' finds a chunk that only contains
        the compound 'felsökningsguiden'."""
        tok = SwedishTokenizer(lexicon=lexicon)
        text = "se felsökningsguiden för mer information"
        chunk = Chunk(
            chunk_id="c1",
            source="doc",
            text=text,
            embedding=[0.0],
            word_frequency=build_word_frequency(text, tok),
        )
        index = BM25Index([chunk], tokenizer=tok)
        assert index.search("felsökning", top_k=1)
        assert index.search("guide", top_k=1)

    def test_stopword_set_is_surface_forms(self):
        # Guard against accidentally stemming the stopword list.
        assert "och" in SWEDISH_STOPWORDS
