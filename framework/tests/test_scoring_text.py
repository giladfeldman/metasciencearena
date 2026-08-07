"""Tests for framework/scoring/text.py — the shared text-similarity helpers.

Focus: the Levenshtein pure-Python-fallback cell guard (DR-0001). The guard must
raise RuntimeError (a clear, fail-fast error the runner records as a soft adapter
error) instead of hanging for minutes on a ~1.4-billion-cell DP matrix when
rapidfuzz is absent — the exact failure that masked the full-body gold rebuild
on 2026-06-12.
"""
from __future__ import annotations

import builtins

import pytest

from framework.scoring import text as shared_text


def _force_no_rapidfuzz(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("rapidfuzz"):
            raise ImportError("rapidfuzz forced absent for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_levenshtein_identity_and_empty():
    assert shared_text.levenshtein_similarity("abc", "abc") == 1.0
    assert shared_text.levenshtein_similarity("", "") == 1.0
    assert shared_text.levenshtein_similarity("abc", "") == 0.0


def test_levenshtein_basic_value():
    # one substitution out of three chars -> 1 - 1/3
    assert shared_text.levenshtein_similarity("cat", "car") == pytest.approx(1 - 1 / 3)


def test_pure_python_fallback_guard_raises_not_hangs(monkeypatch):
    # DR-0001: with rapidfuzz forced absent, two ~3000-char strings would be a
    # ~9M-cell DP matrix (> the 4M guard) and must raise RuntimeError fast.
    _force_no_rapidfuzz(monkeypatch)
    a = "x" * 3000
    b = "y" * 3000
    with pytest.raises(RuntimeError, match="too large for the pure-Python fallback"):
        shared_text.levenshtein_similarity(a, b)


def test_pure_python_fallback_works_below_guard(monkeypatch):
    # Below the guard the fallback computes a real value (no rapidfuzz needed).
    _force_no_rapidfuzz(monkeypatch)
    assert shared_text.levenshtein_similarity("kitten", "sitting") == pytest.approx(1 - 3 / 7)


def test_shared_and_arena_levenshtein_agree():
    # The arena keeps its own _normalize.levenshtein_similarity for back-compat;
    # it MUST stay numerically in sync with the shared helper (same guard, same
    # math). Import the arena copy via its package path.
    import importlib.util
    from pathlib import Path

    arena_norm = (
        Path(__file__).resolve().parents[2]
        / "arenas" / "pdf-text-fidelity-v1" / "_normalize.py"
    )
    spec = importlib.util.spec_from_file_location("_arena_normalize_for_test", arena_norm)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for a, b in [("kitten", "sitting"), ("abc", "abd"), ("hello world", "hallo word"), ("", "x")]:
        assert mod.levenshtein_similarity(a, b) == pytest.approx(
            shared_text.levenshtein_similarity(a, b)
        ), f"divergence on {a!r} vs {b!r}"
