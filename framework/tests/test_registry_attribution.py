"""Attribution sweep (2026-06-13 global-fairness handoff §D).

A leaderboard row must never credit an ARENA-AUTHORED heuristic to the named
tool as if it were the tool's native capability (Finding 5 of the 2026-06-12
handoff: `liteparse-sections/tables-heuristic` are an in-repo regex, not a
liteparse feature). The fix was applied to those two players; this test makes it
a CLASS invariant so the next heuristic adapter added is correct by construction:

  1. Every registry player whose id/adapter_class signals an arena-authored
     heuristic MUST carry ``tool_native: false`` + a non-empty ``wrapper`` badge.
  2. Every ``players/adapters/*_heuristic.py`` adapter file must be referenced by
     at least one such flagged registry entry — so a new heuristic adapter can't
     be wired in without the attribution flags.
"""
from __future__ import annotations

import re
from pathlib import Path

from framework.registry import load_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "players" / "registry.yaml"
ADAPTERS_DIR = REPO_ROOT / "players" / "adapters"

# A player is an arena-authored heuristic when its id or adapter_class says so.
# (regcheck's "shim" only routes auth — its OUTPUT is the regcheck tool's own —
# so "shim" is deliberately NOT a heuristic signal.)
_HEURISTIC_RE = re.compile(r"heuristic", re.IGNORECASE)


def _is_heuristic(entry: dict) -> bool:
    return bool(
        _HEURISTIC_RE.search(entry.get("player_id", ""))
        or _HEURISTIC_RE.search(entry.get("adapter_class", ""))
    )


def test_every_heuristic_player_is_attributed_not_tool_native():
    registry = load_registry(REGISTRY_PATH)
    offenders = []
    for e in registry:
        if not _is_heuristic(e):
            continue
        ok = (e.get("tool_native") is False) and bool(e.get("wrapper"))
        if not ok:
            offenders.append(
                f"{e['player_id']} (tool_native={e.get('tool_native')!r}, "
                f"wrapper={e.get('wrapper')!r})"
            )
    assert not offenders, (
        "Arena-authored heuristic players must declare tool_native:false + a "
        "wrapper badge so the leaderboard doesn't read them as native tool "
        f"capability: {offenders}"
    )


def test_no_unflagged_heuristic_adapter_module():
    """Each players/adapters/*_heuristic.py adapter class is wired to a flagged
    registry entry — a new heuristic adapter can't ship unattributed."""
    # Adapter classes that ARE flagged in the registry.
    registry = load_registry(REGISTRY_PATH)
    flagged_classes = {
        e.get("adapter_class")
        for e in registry
        if _is_heuristic(e) and e.get("tool_native") is False and e.get("wrapper")
    }

    # Public adapter modules (skip private `_helpers` and test_ files).
    heuristic_modules = [
        p for p in ADAPTERS_DIR.glob("*_heuristic.py")
        if not p.name.startswith(("_", "test_"))
    ]
    unflagged = []
    for mod in heuristic_modules:
        text = mod.read_text(encoding="utf-8")
        classes = re.findall(r"class\s+(\w+)\s*\(", text)
        if not any(c in flagged_classes for c in classes):
            unflagged.append(f"{mod.name} (classes={classes})")
    assert not unflagged, (
        "These heuristic adapter modules have no flagged (tool_native:false + "
        f"wrapper) registry entry: {unflagged}"
    )
