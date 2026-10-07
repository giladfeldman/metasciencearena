"""The adapter entry point must actually import a third party's adapters.

WHY THIS TEST EXISTS
--------------------
`framework/runner.py` used to do, at import time:

    try:
        import players.adapters as pkg
    except ImportError:
        return

Installed as a package there IS no `players` directory, so that `return` fired
and **zero adapters registered — silently**. Every player then failed with
"unknown adapter", and nothing in the framework said why. The failure mode is
the dangerous one: no crash, no warning, just an empty registry.

Swapping the hardcoded import for an entry point is only worth anything if the
entry point is actually consumed. A test that merely asserts "runner imports
without players/ present" would pass against the *old* code too — the old code
also imported fine, it just did nothing. So this test drives a fake distribution
through `importlib.metadata` and asserts a class from it lands in the registry.
"""
from __future__ import annotations

import sys
import types

from framework import runner
from framework.player_adapter import _ADAPTER_CLASSES


def test_entry_point_group_name_is_the_published_one():
    # Third parties put this exact string in their pyproject.toml. Renaming it
    # silently breaks every published plugin, so pin it.
    assert runner.ADAPTER_ENTRY_POINT_GROUP == "metasciencearena.adapters"


def test_adapters_load_from_an_entry_point(monkeypatch):
    """A distribution advertising the group gets its modules imported."""
    imported: list[str] = []

    pkg = types.ModuleType("thirdparty_adapters")
    pkg.__path__ = []

    class FakeEP:
        name = "thirdparty"

        def load(self):
            return pkg

    import importlib
    import importlib.metadata as md
    import pkgutil

    monkeypatch.setattr(md, "entry_points", lambda **kw: [FakeEP()] if kw.get("group") == runner.ADAPTER_ENTRY_POINT_GROUP else [])
    monkeypatch.setattr(pkgutil, "iter_modules", lambda path: [(None, "tool_x", False)])
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name) or types.ModuleType(name))
    # Make the in-repo fallback unavailable, so a pass cannot come from it.
    monkeypatch.setitem(sys.modules, "players.adapters", None)

    runner._autoload_adapter_modules()

    assert "thirdparty_adapters.tool_x" in imported, (
        "the entry point was advertised but its adapter module was never imported — "
        "a plugin would register nothing, exactly like the old hardcoded import did"
    )


def test_a_broken_plugin_does_not_block_the_others(monkeypatch):
    """One bad entry point must not stop a good one (DR-0015, generalised)."""
    good = types.ModuleType("good_adapters")
    good.__path__ = []
    imported: list[str] = []

    class BadEP:
        name = "bad"

        def load(self):
            raise RuntimeError("optional dependency missing")

    class GoodEP:
        name = "good"

        def load(self):
            return good

    import importlib
    import importlib.metadata as md
    import pkgutil

    monkeypatch.setattr(md, "entry_points", lambda **kw: [BadEP(), GoodEP()])
    monkeypatch.setattr(pkgutil, "iter_modules", lambda path: [(None, "tool_x", False)])
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "players.adapters", None)

    runner._autoload_adapter_modules()

    assert "good_adapters.tool_x" in imported, "a failing plugin blocked a working one"


def test_in_repo_adapters_still_register():
    """The repo's own adapters are not a distribution — they must still load.

    This is the non-vacuity half: the entry-point machinery must not have
    replaced the in-repo path, or every existing player would stop resolving.
    """
    # `players.adapters` is imported at framework.runner import time.
    assert _ADAPTER_CLASSES, "no adapters registered at all"
    for expected in ("SubprocessCliAdapter", "HttpAdapter", "RCliAdapter"):
        assert expected in _ADAPTER_CLASSES, (
            f"{expected} no longer registers — the entry-point change broke the "
            f"in-repo path that every one of this repo's players uses. "
            f"Known: {sorted(_ADAPTER_CLASSES)}"
        )
