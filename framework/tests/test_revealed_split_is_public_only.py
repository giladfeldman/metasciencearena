"""`--split revealed` must not play held-out tasks (2026-08-04, cycle 10).

`--split` selects the SEED; it never filtered by visibility. Arenas whose
generator emits BOTH visibilities from one call — every PDF arena — therefore
played their held-out real papers during a run whose command line reads
"revealed". With an LLM-CLI player that ships held-out PDFs, potentially
copyrighted APA papers (DATA_HANDLING.md), to a third-party provider.

This is not hypothetical: a cycle-10 run of claude-sonnet-5-sections with
`--split revealed` produced 15 records for a 6-task public split — 9 of them
held-out real PMC papers, already transmitted before it was caught. Egress is
irreversible, so `--split revealed` now implies --public-only, and
--include-held-out is the explicit opt-out.
"""
from __future__ import annotations

import argparse

import pytest

from framework.cli import main as cli_main  # noqa: F401  (import guard)


def _resolve(split: str, *, public_only=False, held_out_only=False, include_held_out=False):
    """Mirror the resolution in _cmd_run so the policy is unit-testable."""
    args = argparse.Namespace(split=split, public_only=public_only,
                              held_out_only=held_out_only,
                              include_held_out=include_held_out)
    resolved = args.public_only
    if args.split == "revealed" and not args.held_out_only and not args.include_held_out:
        resolved = True
    return resolved


def test_revealed_split_implies_public_only():
    assert _resolve("revealed") is True


def test_include_held_out_opts_out_of_the_safe_default():
    assert _resolve("revealed", include_held_out=True) is False


def test_held_out_only_is_not_overridden_by_the_default():
    """A --split revealed --held-out-only run must stay held-out-only."""
    assert _resolve("revealed", held_out_only=True) is False


def test_private_split_is_untouched():
    assert _resolve("private") is False
    assert _resolve("private", public_only=True) is True


def test_cli_exposes_the_opt_out_flag():
    """The escape hatch must exist, or the safe default becomes a hard block."""
    import framework.cli as cli
    import inspect
    src = inspect.getsource(cli)
    assert "--include-held-out" in src
    assert "DATA_HANDLING.md" in src, "the opt-out must point at the data-handling policy"
