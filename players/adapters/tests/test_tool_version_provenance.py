"""`resolved_tool_version` must report the INSTALLED DISTRIBUTION version.

Found 2026-08-10 while diagnosing `framework audit`'s tool-version DRIFT for the
grant-submission gap-closure handoff. `liteparse` 2.0.8 ships a stale module
attribute — `liteparse.__version__` is `"2.0.0"` while the distribution metadata
(what pip, the registry declaration, and the audit all compare against) says
`"2.0.8"`.

`module_version()` read `__version__`, so every run record that adapter stamped
carried a version string that named a release the code was NOT running. That is
published provenance: the letters in `docs/outreach/provider-applications/`
claim "every result is published with full provenance", and the leaderboard
ranks on `resolved_tool_version`. A wrong-but-plausible version string fails no
test and crashes nothing — it just silently misattributes a score to the wrong
release of someone's tool.

Distribution metadata is authoritative because it is what actually got
installed; `__version__` is an author-maintained convention that upstream can
and does forget to bump.
"""
from __future__ import annotations

import sys
import types

import pytest

from players.adapters._tool_version import module_version


def test_distribution_metadata_wins_over_a_stale_dunder_version(monkeypatch):
    """The exact liteparse-2.0.8 shape: __version__ stale, metadata correct."""
    fake = types.ModuleType("arena_fake_tool")
    fake.__version__ = "2.0.0"          # what upstream forgot to bump
    monkeypatch.setitem(sys.modules, "arena_fake_tool", fake)
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda dist: "2.0.8" if dist == "arena_fake_tool" else pytest.fail(
            f"unexpected distribution lookup: {dist}"),
    )

    assert module_version("arena_fake_tool") == "arena_fake_tool-2.0.8", (
        "module_version() reported the stale __version__ attribute instead of "
        "the installed distribution version — this is the defect that stamps a "
        "wrong resolved_tool_version into published run records"
    )


def test_falls_back_to_dunder_version_when_no_distribution_metadata(monkeypatch):
    """A module with no installed distribution must still resolve, not vanish.

    Version detection is best-effort by contract (see _tool_version docstring):
    losing it would blank provenance for vendored/namespace modules.
    """
    import importlib.metadata as md

    fake = types.ModuleType("arena_fake_vendored")
    fake.__version__ = "1.4.2"
    monkeypatch.setitem(sys.modules, "arena_fake_vendored", fake)

    def _raise(dist):
        raise md.PackageNotFoundError(dist)

    monkeypatch.setattr("importlib.metadata.version", _raise)

    assert module_version("arena_fake_vendored") == "arena_fake_vendored-1.4.2"


def test_returns_none_when_neither_source_has_a_version(monkeypatch):
    import importlib.metadata as md

    fake = types.ModuleType("arena_fake_bare")  # no __version__ at all
    monkeypatch.setitem(sys.modules, "arena_fake_bare", fake)

    def _raise(dist):
        raise md.PackageNotFoundError(dist)

    monkeypatch.setattr("importlib.metadata.version", _raise)

    assert module_version("arena_fake_bare") is None


@pytest.mark.parametrize("dist", ["liteparse", "docpluck"])
def test_live_pdf_tool_versions_match_their_installed_distribution(dist):
    """The real tools, against the real environment.

    Guards the actual provenance claim rather than a stub: whatever is installed
    on this machine, `resolved_tool_version` must name THAT release.
    """
    import importlib.metadata as md

    try:
        expected = md.version(dist)
    except md.PackageNotFoundError:
        pytest.skip(f"{dist} not installed in this environment")

    assert module_version(dist) == f"{dist}-{expected}", (
        f"{dist}: resolved_tool_version disagrees with the installed "
        f"distribution — run records would misattribute the score"
    )
