"""Guard the human-readable difficulty ladder (``arenas/*/difficulty.yaml``).

Why this exists
---------------
``difficulty.yaml`` is documented in every arena as the "human-readable mirror of
arena.yaml#difficulty_axes" — the prose that explains *what trap each tier sets*.
It is the content the leaderboard needs to answer "what even IS this task, and
why is it hard?".

Planning audit on 2026-08-06 found the file was never actually consumed by
anything, and had rotted invisibly:

* **11 of 22 failed to parse at all** — an unquoted ``:`` inside a bare prose
  scalar makes PyYAML read the line as a nested mapping
  (``effect-size-conversion-v1/difficulty.yaml:11``:
  ``d<->r with UNEQUAL group sizes (n1, n2): the conversion factor``).
* The 11 that *did* parse used **four mutually incompatible shapes**: nested
  ``{label, description}`` dicts, plain strings, a bare top-level list, and
  ``tier:`` as a list.
* Tier coverage was incomplete even where it parsed — ``code-translation-r-v1``
  documented 6 of its 9 ``tier_pivot`` values, ``replication-target-lookup-v1``
  1 of 6, ``stats-extraction-v1`` 3 of 6.

Nothing read the file, so nothing broke, so nobody noticed. That is exactly the
"a module imported only by its own test is not shipped" failure mode in
CLAUDE.md, inverted: a *data* file nothing consumes rots silently. These tests
are the consumer-side contract, so the rot cannot come back.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from framework.difficulty import load_difficulty, pivot_axis, tier_descriptions

REPO_ROOT = Path(__file__).resolve().parents[2]
ARENA_DIRS = sorted(p.parent for p in REPO_ROOT.glob("arenas/*/arena.yaml"))


@pytest.mark.parametrize("arena_dir", ARENA_DIRS, ids=lambda p: p.name)
def test_difficulty_yaml_parses(arena_dir: Path) -> None:
    """Every difficulty.yaml must be valid YAML.

    Regression: 11/22 raised ScannerError("mapping values are not allowed here")
    from an unquoted colon inside prose.
    """
    path = arena_dir / "difficulty.yaml"
    if not path.exists():
        pytest.skip("no difficulty.yaml")
    try:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - the assert reports it
        pytest.fail(f"{path.relative_to(REPO_ROOT)} is not valid YAML: {exc}")


@pytest.mark.parametrize("arena_dir", ARENA_DIRS, ids=lambda p: p.name)
def test_difficulty_normalizes_to_label_and_description(arena_dir: Path) -> None:
    """The loader must yield {tier: {label, description}} for every arena.

    This is the shape the UI renders, so it is the shape the contract pins —
    regardless of which of the four legacy spellings the file used.
    """
    path = arena_dir / "difficulty.yaml"
    if not path.exists():
        pytest.skip("no difficulty.yaml")
    axis = pivot_axis(arena_dir)
    tiers = tier_descriptions(load_difficulty(arena_dir), axis)
    assert tiers, f"{arena_dir.name}: no '{axis}' entries recovered"
    for value, entry in tiers.items():
        assert isinstance(value, int), f"{arena_dir.name}: {axis} key {value!r} is not an int"
        assert entry.get("description"), f"{arena_dir.name}: {axis} {value} has no description"


@pytest.mark.parametrize("arena_dir", ARENA_DIRS, ids=lambda p: p.name)
def test_every_tier_pivot_value_is_documented(arena_dir: Path) -> None:
    """Each tier shown on the leaderboard heatmap must have prose behind it.

    The heatmap renders one column per ``tier_pivot.values`` entry, so an
    undocumented tier is a column the user cannot interpret. Regression:
    code-translation-r-v1 shipped T7/T8/T9 with no description at all.
    """
    manifest = yaml.safe_load((arena_dir / "arena.yaml").read_text(encoding="utf-8"))
    pivot = manifest.get("tier_pivot") or {}
    values = [v.get("value") for v in (pivot.get("values") or [])]
    if not values:
        pytest.skip("no tier_pivot")
    axis = pivot_axis(arena_dir)
    tiers = tier_descriptions(load_difficulty(arena_dir), axis)
    missing = [v for v in values if v not in tiers]
    assert not missing, (
        f"{arena_dir.name}: tier_pivot '{axis}' values {missing} have no "
        f"difficulty.yaml description (documented: {sorted(tiers)})"
    )
