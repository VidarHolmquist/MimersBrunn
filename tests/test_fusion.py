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
