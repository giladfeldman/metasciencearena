"""Validate display_columns + tier_pivot in every arena.yaml against the schema,
plus the cross-field constraints that JSON Schema can't express on its own."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from framework.paths import schema_path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = schema_path("arena_manifest.schema.json")
ARENA_YAMLS = sorted(REPO_ROOT.glob("arenas/*/arena.yaml"))


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return Draft202012Validator(json.load(f))


@pytest.mark.parametrize("arena_yaml", ARENA_YAMLS, ids=lambda p: p.parent.name)
def test_arena_yaml_validates_against_schema(validator, arena_yaml):
    manifest = yaml.safe_load(arena_yaml.read_text(encoding="utf-8"))
    errors = sorted(validator.iter_errors(manifest), key=lambda e: list(e.path))
    assert not errors, "; ".join(f"{list(e.path)}: {e.message}" for e in errors)


@pytest.mark.parametrize("arena_yaml", ARENA_YAMLS, ids=lambda p: p.parent.name)
def test_display_columns_have_exactly_one_primary(arena_yaml):
    manifest = yaml.safe_load(arena_yaml.read_text(encoding="utf-8"))
    cols = manifest.get("display_columns")
    if cols is None:
        pytest.skip("display_columns optional during phase 1")
    primaries = [c for c in cols if c.get("primary")]
    assert len(primaries) == 1, f"expected exactly one primary column, got {len(primaries)}"


@pytest.mark.parametrize("arena_yaml", ARENA_YAMLS, ids=lambda p: p.parent.name)
def test_display_column_ids_are_unique(arena_yaml):
    manifest = yaml.safe_load(arena_yaml.read_text(encoding="utf-8"))
    cols = manifest.get("display_columns")
    if cols is None:
        pytest.skip("display_columns optional during phase 1")
    ids = [c["id"] for c in cols]
    assert len(ids) == len(set(ids)), f"duplicate column ids: {ids}"


@pytest.mark.parametrize("arena_yaml", ARENA_YAMLS, ids=lambda p: p.parent.name)
def test_tier_pivot_axis_matches_a_difficulty_axis(arena_yaml):
    manifest = yaml.safe_load(arena_yaml.read_text(encoding="utf-8"))
    pivot = manifest.get("tier_pivot")
    if pivot is None:
        pytest.skip("tier_pivot optional during phase 1")
    axis_ids = {a["id"] for a in manifest.get("difficulty_axes", [])}
    assert pivot["axis"] in axis_ids, f"tier_pivot.axis {pivot['axis']!r} not in difficulty_axes ids {axis_ids}"


@pytest.mark.parametrize("arena_yaml", ARENA_YAMLS, ids=lambda p: p.parent.name)
def test_tier_pivot_values_cover_axis_range(arena_yaml):
    manifest = yaml.safe_load(arena_yaml.read_text(encoding="utf-8"))
    pivot = manifest.get("tier_pivot")
    if pivot is None:
        pytest.skip("tier_pivot optional during phase 1")
    axis = next(a for a in manifest["difficulty_axes"] if a["id"] == pivot["axis"])
    declared_values = {v["value"] for v in pivot["values"]}
    expected_values = set(range(axis["min"], axis["max"] + 1))
    assert declared_values == expected_values, (
        f"tier_pivot.values declares {sorted(declared_values)}, "
        f"axis {pivot['axis']} range is {sorted(expected_values)}"
    )
