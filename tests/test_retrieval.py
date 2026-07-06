"""Tests for the hybrid retrieval package.

Run with: python -m pytest tests/ -v
"""

from collections import Counter

import pytest

from retrieval import (
    BM25Index,
    BruteForceDenseIndex,
    Chunk,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from retrieval.tokenize import default_tokenizer


def make_chunk(chunk_id: str, text: str, embedding) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source=f"doc-{chunk_id}",
        text=text,
        embedding=embedding,
        word_frequency=dict(Counter(default_tokenizer(text))),
    )


@pytest.fixture
def corpus():
    # Embeddings are hand-made 3-d vectors: axis 0 ~ "animals",
    # axis 1 ~ "cooking", axis 2 ~ "finance".
    return [
        make_chunk("a", "the quick brown fox jumps over the lazy dog", [1.0, 0.0, 0.1]),
        make_chunk("b", "how to cook pasta with tomato sauce", [0.0, 1.0, 0.0]),
        make_chunk("c", "stock market returns and index funds", [0.0, 0.1, 1.0]),
        make_chunk("d", "dogs and foxes are both canines", [0.9, 0.0, 0.0]),
    ]


class TestBM25Index:
    def test_keyword_match_ranks_first(self, corpus):
        index = BM25Index(corpus)
        hits = index.search("tomato sauce recipe", top_k=4)
        assert hits and hits[0].chunk_id == "b"

    def test_no_matching_terms_returns_empty(self, corpus):
        index = BM25Index(corpus)
        assert index.search("xylophone zeppelin", top_k=4) == []

    def test_respects_top_k(self, corpus):
        index = BM25Index(corpus)
        assert len(index.search("fox dog canines", top_k=1)) == 1

    def test_empty_corpus(self):
        assert BM25Index([]).search("anything", top_k=5) == []


class TestBruteForceDenseIndex:
    def test_nearest_vector_ranks_first(self, corpus):
        index = BruteForceDenseIndex(corpus)
        hits = index.search([0.0, 0.05, 1.0], top_k=4)  # finance-ish query
        assert hits[0].chunk_id == "c"

    def test_scores_are_descending(self, corpus):
        index = BruteForceDenseIndex(corpus)
        hits = index.search([1.0, 0.0, 0.0], top_k=4)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_empty_corpus(self):
        assert BruteForceDenseIndex([]).search([1.0, 0.0], top_k=5) == []


class TestReciprocalRankFusion:
    def test_agreement_wins(self):
        fused = reciprocal_rank_fusion([["x", "y", "z"], ["y", "x", "w"]])
        # x and y each appear at ranks 1 and 2; both beat z and w.
        top_two = {item_id for item_id, _ in fused[:2]}
        assert top_two == {"x", "y"}

    def test_weights_shift_ranking(self):
        fused = reciprocal_rank_fusion(
            [["x"], ["y"]], weights=[1.0, 3.0]
        )
        assert fused[0][0] == "y"

    def test_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["x"]], weights=[1.0, 2.0])

    def test_deterministic_tie_break(self):
        fused = reciprocal_rank_fusion([["b"], ["a"]])
        assert [item_id for item_id, _ in fused] == ["a", "b"]


class TestHybridRetriever:
    def test_search_with_provided_embedding(self, corpus):
        retriever = HybridRetriever(corpus)
        results = retriever.search(
            "fox and dog", query_embedding=[1.0, 0.0, 0.0], top_k=2
        )
        assert len(results) == 2
        # Both sparse (fox/dog terms) and dense (animal axis) agree on a/d.
        assert {r.chunk.chunk_id for r in results} == {"a", "d"}
        assert results[0].fused_score >= results[1].fused_score

    def test_search_uses_embedder_when_no_embedding_given(self, corpus):
        calls = []

        def embedder(query: str):
            calls.append(query)
            return [0.0, 1.0, 0.0]

        retriever = HybridRetriever(corpus, embedder=embedder)
        results = retriever.search("cook pasta", top_k=1)
        assert calls == ["cook pasta"]
        assert results[0].chunk.chunk_id == "b"

    def test_missing_embedder_and_embedding_raises(self, corpus):
        retriever = HybridRetriever(corpus)
        with pytest.raises(ValueError, match="embedder"):
            retriever.search("anything")

    def test_duplicate_chunk_ids_rejected(self, corpus):
        with pytest.raises(ValueError, match="unique"):
            HybridRetriever(corpus + [corpus[0]])

    def test_result_diagnostics(self, corpus):
        retriever = HybridRetriever(corpus)
        results = retriever.search(
            "fox", query_embedding=[1.0, 0.0, 0.0], top_k=4
        )
        top = results[0]
        # The top hit should have been found by at least one retriever,
        # and any reported rank must be 1-based.
        assert top.sparse_rank is not None or top.dense_rank is not None
        for r in results:
            if r.sparse_rank is not None:
                assert r.sparse_rank >= 1 and r.sparse_score is not None
            if r.dense_rank is not None:
                assert r.dense_rank >= 1 and r.dense_score is not None
