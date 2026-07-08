"""Regenerate docs/REPO_BUNDLE.md — the single pastable copy of the codebase.

Run from the repo root after any code change:

    python tools/make_bundle.py

The bundle exists because AIP Assist / the AI FDE cannot access this Git
repo; the whole codebase is pasted into the session instead. Keeping the
bundle generated (never hand-edited) guarantees it matches the repo.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "REPO_BUNDLE.md"

# (section title, intro, files) — order mirrors the deployment phases in
# docs/AI_FDE_PROMPT.md.
SECTIONS = [
    (
        "PART A — Ingestion (paste into the Python TRANSFORMS Code Repository)",
        "Both files go into the transforms repo. `ingest.py` imports the "
        "analyzer as `from shared.swedish_analyzer import ...`; adjust that "
        "import to wherever the files land in the repo's package structure "
        "(e.g. put both in one package and import as siblings). Change "
        "nothing else. Dependency: `snowballstemmer`.",
        ["shared/swedish_analyzer.py", "ingestion/ingest.py"],
    ),
    (
        "PART B — Runtime (paste into the Python FUNCTIONS Code Repository)",
        "All four files go into the functions repo, plus a second identical "
        "copy of `shared/swedish_analyzer.py` from PART A (it must stay "
        "byte-identical to the transforms copy — `ANALYZER_VERSION` guards "
        "drift at runtime). `hybrid_search.py` imports "
        "`from retrieval import bm25`, `from retrieval.fusion import ...` "
        "and `from shared.swedish_analyzer import ...`; adjust those import "
        "paths to the repo's package structure, change nothing else. "
        "Dependency: `snowballstemmer`.",
        [
            "retrieval/__init__.py",
            "retrieval/bm25.py",
            "retrieval/fusion.py",
            "retrieval/hybrid_search.py",
        ],
    ),
    (
        "PART C — Unit tests (optional; run locally or in CI, not in Foundry)",
        "Pure-Python tests for the analyzer, BM25, fusion, and ingestion "
        "logic. They document expected behavior; `pytest` from the repo "
        "root runs them (29 tests).",
        [
            "tests/test_swedish_analyzer.py",
            "tests/test_bm25.py",
            "tests/test_fusion.py",
            "tests/test_ingest.py",
        ],
    ),
]

HEADER = """\
# MimersBrunn — Full Codebase (pastable bundle)

GENERATED FILE — do not edit by hand; regenerate with
`python tools/make_bundle.py`.

This is the complete source of the hybrid Swedish RAG retrieval codebase,
for pasting into AIP Assist / an AI FDE session that cannot access the Git
repo. Use it together with `docs/AI_FDE_PROMPT.md`, which explains the
architecture and the four deployment phases; paste PART A during Phase 1
and PART B during Phase 3.

Rules for whoever receives this bundle:

- Every file is delimited by a `FILE:` heading giving its repo-relative
  path. Reproduce file contents exactly.
- The ONLY permitted edits are the import-path adjustments called out in
  each part's intro (Foundry code repositories have their own package
  layouts). Do not rename functions, parameters, properties, or the
  `hybridDocumentSearch` API name.
- `swedish_analyzer.py` is deployed twice (transforms repo and functions
  repo) and the two copies must be identical.
- Single external dependency for both repos: `snowballstemmer` (pure
  Python).
"""


def main() -> None:
    parts = [HEADER]
    for title, intro, files in SECTIONS:
        parts.append(f"\n---\n\n## {title}\n\n{intro}\n")
        for rel_path in files:
            source = (ROOT / rel_path).read_text(encoding="utf-8").rstrip("\n")
            parts.append(f"\n### FILE: `{rel_path}`\n\n````python\n{source}\n````\n")
    OUTPUT.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
