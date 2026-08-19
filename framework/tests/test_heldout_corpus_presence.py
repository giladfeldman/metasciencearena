"""A missing held-out corpus must FAIL, never silently shrink the benchmark.

The defect this pins, found 2026-08-08. Every PDF arena's real-paper generator
opened with:

    if not HELD_OUT_PMC_DIR.exists():
        return

The corpora are gitignored — working tree only — so `git stash -a`, a branch
switch, or a manual tidy-up removes them without git noticing. When exactly that
happened, `pdf-citation-matching-v1` went from 36 tasks to 6 and reported
success. Nothing raised. 137 tests skipped. The run would have published a
benchmark missing 30 of its 36 tasks, and the leaderboard would have shown
scores computed over the remainder as though nothing had changed.

Same false-green class as a suite quietly collecting 1023 of 1143 tests, and the
same one `framework.publish.scan_for_leaks` already refuses to commit: a check
that cannot see its subject must not certify it.

`framework.holdout.require_corpus` is the class-level fix. Absence raises unless
the caller DECLARES a synthetic-only run, which turns a reduced benchmark from an
accident into a stated choice.
"""
from __future__ import annotations

import pytest

from framework.holdout import (
    SYNTHETIC_ONLY_ENV,
    HeldOutCorpusMissing,
    require_corpus,
)


def test_a_present_corpus_is_returned(tmp_path):
    corpus = tmp_path / "pmc"
    corpus.mkdir()
    assert require_corpus(corpus, arena_id="x-v1", kind="PMC") == corpus


def test_a_missing_corpus_raises_rather_than_returning_none(monkeypatch, tmp_path):
    # The whole point: the caller must not be able to mistake "gone" for "empty".
    monkeypatch.delenv(SYNTHETIC_ONLY_ENV, raising=False)
    with pytest.raises(HeldOutCorpusMissing):
        require_corpus(tmp_path / "absent", arena_id="x-v1", kind="PMC")


def test_the_error_names_the_arena_the_path_and_the_way_out(monkeypatch, tmp_path):
    # An operator hitting this at 2am needs to know WHICH corpus, WHERE it should
    # be, and that the files are gitignored — otherwise the natural response is
    # "git says the tree is clean, so nothing is missing".
    monkeypatch.delenv(SYNTHETIC_ONLY_ENV, raising=False)
    with pytest.raises(HeldOutCorpusMissing) as exc:
        require_corpus(tmp_path / "absent", arena_id="pdf-text-fidelity-v1", kind="PMC")
    msg = str(exc.value)
    assert "pdf-text-fidelity-v1" in msg
    assert "absent" in msg
    assert "GITIGNORED" in msg
    assert SYNTHETIC_ONLY_ENV in msg


def test_synthetic_only_must_be_DECLARED_not_inferred(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(SYNTHETIC_ONLY_ENV, "1")
    assert require_corpus(tmp_path / "absent", arena_id="x-v1", kind="PMC") is None
    # Declaring it still says so out loud: a smaller benchmark must never pass
    # unremarked, even when it was asked for.
    err = capsys.readouterr().err
    assert "SYNTHETIC-ONLY" in err
    assert "not comparable" in err


def test_only_the_exact_optin_value_counts(monkeypatch, tmp_path):
    # "0", "false", "" and a stray non-empty string must all still raise —
    # an env var that happens to be set must not disable a safety check.
    for value in ("0", "false", "", "yes"):
        monkeypatch.setenv(SYNTHETIC_ONLY_ENV, value)
        with pytest.raises(HeldOutCorpusMissing):
            require_corpus(tmp_path / "absent", arena_id="x-v1", kind="PMC")


def test_every_generator_uses_the_helper_rather_than_its_own_check():
    """No generator may reintroduce the bare `if not ...exists(): return`.

    This is the guard that actually holds the line: the helper is worthless if
    the next arena copies the old three-line pattern from a sibling, which is
    exactly how it came to exist in six of them.
    """
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    offenders = []
    for gen in sorted((repo / "arenas").glob("*/generator.py")):
        src = gen.read_text(encoding="utf-8")
        if "HELD_OUT" not in src:
            continue
        # A bare existence test on a HELD_OUT path whose body just bails out.
        for m in re.finditer(
            r"if not (HELD_OUT\w*)\.(?:exists|is_dir)\(\):\s*\n\s*return", src
        ):
            offenders.append(f"{gen.relative_to(repo)} -> {m.group(0).splitlines()[0]}")

    assert offenders == [], (
        "these generators silently emit ZERO real-paper tasks when their corpus is "
        "missing, instead of failing:\n  " + "\n  ".join(offenders)
        + "\nUse framework.holdout.require_corpus(...) so absence raises."
    )
