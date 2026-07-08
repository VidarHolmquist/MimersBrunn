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
