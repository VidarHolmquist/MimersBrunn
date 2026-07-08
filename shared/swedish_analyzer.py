"""Swedish text analyzer: lowercase + stopwords + compound splitting + stemming.

Swedish forms compounds by concatenation ("felsökningsguide" = felsökning +
guide), so a query for one part never lexically matches the compound unless
the analyzer splits it. This module provides:

- :class:`SwedishAnalyzer` — lowercases, drops stopwords, splits compounds
  against a lexicon, and stems with the Snowball Swedish stemmer. When a
  compound is split, the stemmed *whole* word is emitted as well as its
  parts, so an exact compound query still matches.
- :func:`build_lexicon` — derives the decompounding lexicon from the corpus
  itself, so no external Swedish dictionary is needed and domain terms are
  covered automatically.

THE ANALYZER CONTRACT: the exact same analyzer (same version, same lexicon)
must be applied to chunk text at ingestion and to queries at retrieval.
``ANALYZER_VERSION`` is stored with the corpus stats at ingestion and checked
at query time so a drifted deployment fails loudly instead of silently
returning bad matches.

This file is the single source of truth. In Foundry, the ingestion transform
repo and the retrieval functions repo are separate code repositories — copy
this file into both (or publish it as a shared library) and keep the copies
identical. Bump ``ANALYZER_VERSION`` on any behavioral change and re-run the
full ingestion pipeline.

Requires the ``snowballstemmer`` package (pure Python, no binary deps).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

import snowballstemmer

# Bump on ANY change that affects token output (stopwords, splitting rules,
# stemmer, regex). Ingestion stores this with the corpus; retrieval refuses
# to trust precomputed stats built with a different version.
ANALYZER_VERSION = "sv-1"

_WORD_RE = re.compile(r"[\wåäöÅÄÖ]+", re.UNICODE)

# Deliberately small: over-aggressive stopword lists hurt technical
# documentation, where words like "under"/"över" can be meaningful.
SWEDISH_STOPWORDS: Set[str] = {
    "och", "i", "att", "det", "som", "en", "ett", "på", "är", "av", "för",
    "med", "till", "den", "de", "har", "inte", "om", "man", "kan", "ska",
    "skall", "vid", "eller", "från", "så", "vi", "du", "ni", "sig", "sin",
    "sitt", "sina", "denna", "detta", "dessa", "vara", "blir", "bli", "då",
    "när", "här", "där", "efter", "innan", "samt", "även", "också",
}

# Swedish technical docs mix in English; drop the highest-frequency English
# function words too.
ENGLISH_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "with",
}

DEFAULT_STOPWORDS: Set[str] = SWEDISH_STOPWORDS | ENGLISH_STOPWORDS


def build_lexicon(
    texts: Iterable[str],
    min_count: int = 3,
    min_length: int = 4,
    stopwords: Optional[Set[str]] = None,
) -> Set[str]:
    """Derive a decompounding lexicon from the corpus itself.

    Words that occur standalone at least ``min_count`` times are considered
    valid compound parts. The corpus's own vocabulary is exactly the
    vocabulary users will query with, and it automatically includes domain
    terms no general dictionary knows.

    Returns *stemmed* forms; :class:`SwedishAnalyzer` stems candidate parts
    before lookup, which also neutralizes the compound linking "s"
    ("arbets-" and "arbete" both stem to "arbet").
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    stemmer = snowballstemmer.stemmer("swedish")

    counts: Counter = Counter()
    for text in texts:
        for token in _WORD_RE.findall(text.lower()):
            if len(token) >= min_length and token not in stopwords:
                counts[token] += 1

    return {
        stemmer.stemWord(token)
        for token, count in counts.items()
        if count >= min_count
    }


def lexicon_to_json(lexicon: Set[str]) -> str:
    """Serialize a lexicon for storage (e.g. on the corpus meta object)."""
    return json.dumps(sorted(lexicon), ensure_ascii=False)


def lexicon_from_json(payload: Optional[str]) -> Set[str]:
    """Inverse of :func:`lexicon_to_json`; tolerates None/empty/bad input."""
    if not payload:
        return set()
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return set()
    return {str(item) for item in data} if isinstance(data, list) else set()


class SwedishAnalyzer:
    """Tokenizer for both ingestion and query analysis.

    Pipeline per token: lowercase -> stopword filter -> compound split
    (if a lexicon is given) -> Snowball stem. When a compound is split, the
    stemmed whole word is emitted as well as its parts, so an exact compound
    query still matches ("felsökningsguide" emits
    ["felsökningsguid", "felsökning", "guid"]).

    Args:
        lexicon: Stemmed valid compound parts, from :func:`build_lexicon`.
            Without it, no decompounding happens (stemming still does).
        stopwords: Surface-form words to drop. Defaults to
            :data:`DEFAULT_STOPWORDS`.
        min_part_length: Shortest surface substring considered a compound
            part. Below 4 the splitter starts inventing nonsense splits.
        max_parts: Give up on splits needing more than this many parts;
            genuine Swedish compounds rarely exceed three.
    """

    def __init__(
        self,
        lexicon: Optional[Set[str]] = None,
        stopwords: Optional[Set[str]] = None,
        min_part_length: int = 4,
        max_parts: int = 3,
    ) -> None:
        self._lexicon = lexicon or set()
        self._stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self._min_part = min_part_length
        self._max_parts = max_parts
        self._stemmer = snowballstemmer.stemmer("swedish")

    def tokenize(self, text: Optional[str]) -> List[str]:
        """Analyzed tokens for a text, compounds expanded, in reading order."""
        if not text:
            return []
        output: List[str] = []
        for token in _WORD_RE.findall(text.lower()):
            if len(token) < 2 or token in self._stopwords:
                continue
            output.append(self._stemmer.stemWord(token))
            parts = self._decompound(token)
            if parts is not None:
                output.extend(parts)
        return output

    def unique_terms(self, text: Optional[str]) -> List[str]:
        """Analyzed tokens, deduplicated, first-seen order. For queries."""
        seen: Set[str] = set()
        terms: List[str] = []
        for term in self.tokenize(text):
            if term not in seen:
                seen.add(term)
                terms.append(term)
        return terms

    def term_counts(self, text: Optional[str]) -> Dict[str, int]:
        """Term -> frequency map. For ingestion (stored as word_frequency)."""
        return dict(Counter(self.tokenize(text)))

    def _decompound(self, token: str) -> Optional[List[str]]:
        """Split ``token`` into known parts; None if it doesn't decompose.

        Dynamic programming over split positions, preferring the split with
        the fewest parts (fewer, longer parts are more likely to be the
        linguistically correct reading than many short ones).
        """
        if not self._lexicon or len(token) < 2 * self._min_part:
            return None

        n = len(token)
        # best[i] = (number of parts, parts list) for token[:i], or None.
        best: List[Optional[Tuple[int, List[str]]]] = [None] * (n + 1)
        best[0] = (0, [])

        for end in range(self._min_part, n + 1):
            for start in range(0, end - self._min_part + 1):
                if best[start] is None:
                    continue
                n_parts = best[start][0] + 1
                if n_parts > self._max_parts:
                    continue
                if best[end] is not None and best[end][0] <= n_parts:
                    continue
                part_stem = self._lookup(token[start:end])
                if part_stem is not None:
                    best[end] = (n_parts, best[start][1] + [part_stem])

        if best[n] is None or best[n][0] < 2:
            return None
        return best[n][1]

    def _lookup(self, part: str) -> Optional[str]:
        """Return the stemmed lexicon entry for a surface part, if any.

        Stemming before lookup also absorbs the compound linking "s"
        (fogemorfem): "arbets" stems to "arbet", same as "arbete". As a
        fallback, an explicit trailing "s" is stripped and retried, for
        joints the stemmer does not remove.
        """
        stem = self._stemmer.stemWord(part)
        if stem in self._lexicon:
            return stem
        if part.endswith("s"):
            stem = self._stemmer.stemWord(part[:-1])
            if stem in self._lexicon:
                return stem
        return None
