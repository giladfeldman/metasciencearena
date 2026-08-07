"""Load ``arenas/<id>/difficulty.yaml`` — the human-readable tier ladder.

``arena.yaml#difficulty_axes`` carries the machine-readable axes; this file
carries the *prose*: what each tier actually traps, in language a reader who is
not a methodologist can follow. The leaderboard needs it to answer "what even IS
``xlat-regression_multi-spss-s0``, and why is it hard?".

One normalized shape
--------------------
The published contract is a block keyed by the arena's **pivot axis** — which is
``tier`` for most arenas but NOT all: the PDF arenas pivot on
``corruption_intensity`` / ``citation_style`` / ``header_complexity`` /
``label_diversity``. The block is named after whatever ``arena.yaml#tier_pivot.axis``
says::

    corruption_intensity:
      0:
        label: clean
        description: >-
          No injected corruption ...

Historically four spellings existed across the 22 arenas (see
``framework/tests/test_difficulty.py`` for the audit). All are now normalized on
disk, but this loader stays tolerant of the legacy forms so a hand-edited arena
cannot silently drop out of the UI:

* ``{label, description}``  — the contract shape
* ``"prose"``               — a bare string, promoted to ``description``
* ``[...]`` at top level    — a list of axis specs, searched for the pivot axis
* ``<axis>: [...]``         — a list, indexed 1-based

Every form yields ``{int: {"label": str, "description": str}}``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

TIER_AXIS = "tier"


def pivot_axis(arena_dir: str | Path) -> str:
    """The axis an arena's heatmap pivots on; ``tier`` when unspecified.

    The UI renders one column per ``tier_pivot.values`` entry, so this is the
    axis whose values need prose. Five arenas pivot on something other than
    ``tier``, which is why this is read rather than assumed.
    """
    path = Path(arena_dir) / "arena.yaml"
    if not path.exists():
        return TIER_AXIS
    manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    axis = ((manifest.get("tier_pivot") or {}).get("axis")) or TIER_AXIS
    return str(axis)


def load_difficulty(arena_dir: str | Path) -> dict[str, Any]:
    """Parse an arena's ``difficulty.yaml``; ``{}`` when absent.

    Raises ``yaml.YAMLError`` on a malformed file rather than swallowing it — a
    silently-empty ladder would render as "no explanation available" and look
    like missing content rather than a broken file.
    """
    path = Path(arena_dir) / "difficulty.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {"_list": data}


def _coerce_entry(raw: Any, fallback_label: str = "") -> dict[str, str] | None:
    """Normalize one tier entry to ``{label, description}``."""
    if isinstance(raw, str):
        text = raw.strip()
        return {"label": fallback_label, "description": text} if text else None
    if isinstance(raw, dict):
        desc = str(raw.get("description") or "").strip()
        label = str(raw.get("label") or fallback_label or "").strip()
        return {"label": label, "description": desc} if desc else None
    return None


def tier_descriptions(
    difficulty: dict[str, Any], axis: str = TIER_AXIS
) -> dict[int, dict[str, str]]:
    """Return ``{axis_value: {label, description}}`` from a loaded difficulty doc.

    ``axis`` is the arena's pivot axis (see :func:`pivot_axis`) — ``tier`` for
    most arenas, ``corruption_intensity`` / ``citation_style`` / others for the
    PDF family. Tolerates the legacy shapes; entries without prose are dropped
    rather than emitted empty, so a caller can treat "present" as "has something
    to show".
    """
    if not difficulty:
        return {}

    raw_tiers: Any = difficulty.get(axis)
    if raw_tiers is None and axis != TIER_AXIS:
        raw_tiers = difficulty.get(TIER_AXIS)

    # Top-level list of axis specs: find the one describing the pivot axis.
    if raw_tiers is None and "_list" in difficulty:
        for spec in difficulty["_list"] or []:
            if isinstance(spec, dict) and (spec.get("id") == axis or axis in spec):
                raw_tiers = spec.get(axis, spec.get("values"))
                break

    if raw_tiers is None:
        return {}

    out: dict[int, dict[str, str]] = {}

    # ``tier:`` as a list — 1-based, matching how tiers are numbered everywhere.
    if isinstance(raw_tiers, list):
        for i, item in enumerate(raw_tiers, start=1):
            value = i
            if isinstance(item, dict) and "value" in item:
                try:
                    value = int(item["value"])
                except (TypeError, ValueError):
                    value = i
            entry = _coerce_entry(item)
            if entry:
                out[value] = entry
        return out

    if isinstance(raw_tiers, dict):
        for key, item in raw_tiers.items():
            try:
                value = int(key)
            except (TypeError, ValueError):
                continue
            entry = _coerce_entry(item)
            if entry:
                out[value] = entry

    return out
