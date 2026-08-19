"""A registry key an adapter accepts MUST actually reach that adapter.

Found 2026-08-19 while wiring Docling. `build_adapter` forwards only the keys
listed in the hardcoded `ADAPTER_EXTRA_KWARGS` allowlist and silently drops the
rest, so a registry key that names a real adapter parameter is a no-op unless
someone also remembered to extend the allowlist. Nobody did for
`ocr_enabled` / `ocr_language` / `dpi`.

The live consequence: `liteparse-no-ocr` declares `ocr_enabled: false` and calls
itself "liteparse-2.11.1-no-ocr", but was constructed with `ocr_enabled=True`
(the adapter default) and therefore published a score bit-identical to
`liteparse-default` — 0.7240294726242898 on n=70 for both, ranked 5th and 6th as
though they were an OCR-on/OCR-off comparison that never ran.

This is the "a UI option the code never branches on" defect from the portfolio
rules: silently recording a condition that never happened is worse than not
offering the option.

A per-key allowlist test would just re-encode the same oversight for the next
tool, so both tests below are GENERIC over the real registry.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from framework.player_adapter import (
    ADAPTER_EXTRA_KWARGS,
    _ADAPTER_CLASSES,
    build_adapter,
)
from framework.registry import REQUIRED_FIELDS, load_registry
import framework.runner  # noqa: F401 - importing self-registers every adapter class

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "players" / "registry.yaml"

# Registry keys that are deliberately NOT adapter parameters. Each is read by a
# named consumer outside the adapter, so being dropped by `build_adapter` is
# correct rather than a defect. Adding a name here is a claim you must be able
# to point at a call site for.
CONSUMED_OUTSIDE_THE_ADAPTER = {
    "best_effort",   # framework/cli.py — exempts a player from the symmetry gate
    "tool_native",   # leaderboard attribution: is the score the tool's own capability?
    "wrapper",       # leaderboard attribution: names the arena-authored wrapper
}


@pytest.fixture(scope="module")
def registry() -> list[dict]:
    return load_registry(REGISTRY_PATH)


def test_every_registry_key_is_routed_somewhere(registry):
    """No key may be silently dropped on the floor."""
    known = REQUIRED_FIELDS | set(ADAPTER_EXTRA_KWARGS) | CONSUMED_OUTSIDE_THE_ADAPTER
    unrouted: dict[str, set[str]] = {}
    for entry in registry:
        for key in entry:
            if key not in known:
                unrouted.setdefault(key, set()).add(entry["player_id"])
    assert not unrouted, (
        "registry keys reach no consumer — they are silently dropped by "
        f"build_adapter and read by nothing else: "
        + "; ".join(f"{k} ({', '.join(sorted(v))})" for k, v in sorted(unrouted.items()))
    )


def test_a_key_the_adapter_accepts_is_actually_delivered(registry):
    """The sharp version: if the target adapter's __init__ names a parameter and
    the registry declares it, `build_adapter` must deliver it.

    This is the invariant the liteparse defect broke. It does not depend on
    anyone remembering to update an allowlist — it reads the adapter signature.
    """
    dropped: list[str] = []
    for entry in registry:
        cls = _ADAPTER_CLASSES.get(entry["adapter_class"])
        if cls is None:
            continue  # unknown classes are covered by test_registry_attribution
        try:
            params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtin __init__
            continue
        named = {
            name
            for name, p in params.items()
            if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD) and name != "self"
        }
        for key in entry:
            if key in REQUIRED_FIELDS or key not in named:
                continue
            if key not in ADAPTER_EXTRA_KWARGS:
                dropped.append(
                    f"{entry['player_id']}: declares {key!r}, which "
                    f"{cls.__name__}.__init__ accepts, but ADAPTER_EXTRA_KWARGS "
                    f"drops it — the adapter silently uses its default"
                )
    assert not dropped, "registry config never reaches the adapter:\n  " + "\n  ".join(dropped)


def test_liteparse_no_ocr_is_actually_built_without_ocr(registry):
    """The concrete instance, pinned so it cannot regress.

    Reverting the `ocr_enabled` entry in ADAPTER_EXTRA_KWARGS must turn this red.
    """
    entry = next(e for e in registry if e["player_id"] == "liteparse-no-ocr")
    assert entry["ocr_enabled"] is False, "registry no longer declares ocr_enabled: false"
    adapter = build_adapter(entry)
    assert adapter.ocr_enabled is False, (
        "liteparse-no-ocr was built WITH OCR — its player_version claims "
        "'no-ocr' and its published score would be a duplicate of liteparse-default"
    )


def test_every_registry_entry_can_actually_be_built(registry):
    """The blunt end-to-end check: `build_adapter` must succeed for every entry.

    Added after the ADAPTER_EXTRA_KWARGS fix above surfaced a SECOND instance of
    the same defect class. `LiteparseSectionsHeuristicAdapter` and
    `LiteparseTablesHeuristicAdapter` declare `ocr_enabled` / `ocr_language` /
    `dpi` as class attributes but are not dataclasses, so they inherit
    `PlayerAdapter.__init__`, which rejects those keywords. While the keys were
    being silently dropped this raised nothing; delivering them turned it into a
    TypeError at construction.

    Signature inspection cannot see this (the inherited __init__ does not name
    the parameters), so the only honest check is to build the thing.
    """
    failures: list[str] = []
    for entry in registry:
        if entry["adapter_class"] not in _ADAPTER_CLASSES:
            continue  # optional dep absent; covered by test_registry_attribution
        try:
            build_adapter(entry)
        except Exception as exc:  # noqa: BLE001 - the point is to report all of them
            failures.append(f"{entry['player_id']} ({entry['adapter_class']}): {exc!r}")
    assert not failures, "registry entries that cannot be constructed:\n  " + "\n  ".join(failures)
