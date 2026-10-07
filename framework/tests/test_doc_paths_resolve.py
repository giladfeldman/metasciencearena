"""Backtick-quoted repo paths in the working docs must actually exist.

REGRESSION (2026-08-21, /ship Phase 3). TODO.md carried four references to files that
are not there: `contract/pricing.json` and `contract/schemas/run_record.schema.json`
(both live under `framework/contract/`), `scripts/build-data.mjs` (it is under
`leaderboard-app/scripts/`), and a stats-extraction v1 escimate run that was superseded
when that arena moved to task set v2.

None of them broke anything, which is the point. A doc is a claim with a citation, and
this repo's standing rule is that a citation which no longer resolves makes the claim
UNSUPPORTED rather than merely untidy. The failure mode is a reader — human or agent —
following a path that has quietly moved and concluding the feature is gone, or worse,
recreating it somewhere new.

SCOPE is deliberately narrow: the docs a session actually NAVIGATES BY. Dated,
append-only historical ledgers are EXEMPT — docs/superpowers/{handoffs,plans}/ and
LESSONS.md. Those record what was true when written; several of their paths are stale
precisely BECAUSE the repo moved on (tools/build_gold.py was deliberately removed from
the public mirror, contract/ moved under framework/). Rewriting a dated entry to keep a
linter quiet would destroy the only thing that makes it evidence. The distinction is
navigational-vs-historical, not important-vs-unimportant.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Only paths rooted at a real top-level directory are checked, so prose like
#: `players/registry.yaml` is validated while `some/made/up/example` in an illustration
#: is not mistaken for a citation.
_ROOTS = ("arenas", "contract", "docs", "framework", "leaderboard-app", "players",
          "scripts", "taxonomy", "tools")
_PATH_RE = re.compile(r"`((?:" + "|".join(_ROOTS) + r")/[A-Za-z0-9_.@/-]+)`")

#: A trailing '.' or ',' is punctuation, not part of the path.
_TRIM = ".,;:"


def _docs_under_review() -> list[Path]:
    docs = [REPO / "TODO.md"]          # LESSONS.md is historical — see the module docstring
    for d in ("docs/model-access", "docs/design"):
        docs.extend(sorted((REPO / d).glob("*.md")))
    return [p for p in docs if p.exists()]


@pytest.mark.parametrize("doc", _docs_under_review(), ids=lambda p: p.name)
def test_referenced_repo_paths_exist(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8", errors="replace")
    broken: list[str] = []
    for raw in _PATH_RE.findall(text):
        ref = raw.rstrip(_TRIM)
        # Globs and placeholders are patterns, not citations.
        if any(ch in ref for ch in "*<>{}") or ref.endswith("/"):
            continue
        # `framework/discovery.discover_arenas` is a dotted SYMBOL, not a file: the
        # last segment carries a dot but no known file extension.
        last = ref.rsplit("/", 1)[-1]
        if "." in last and not last.rsplit(".", 1)[-1].isalnum():
            continue
        if "." in last and last.rsplit(".", 1)[-1] not in {
                "py", "ts", "tsx", "mjs", "js", "json", "md", "yaml", "yml", "txt",
                "jsonl", "sh", "ps1", "R", "r", "csv", "toml", "lock", "html"}:
            continue
        if not (REPO / ref).exists():
            broken.append(ref)
    assert not broken, (
        f"{doc.relative_to(REPO).as_posix()} cites {len(broken)} path(s) that do not "
        f"exist. A citation that no longer resolves makes the claim UNSUPPORTED, not "
        f"merely stale — fix the path or drop the claim:\n  "
        + "\n  ".join(sorted(set(broken)))
    )
