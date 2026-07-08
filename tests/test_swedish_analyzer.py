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
