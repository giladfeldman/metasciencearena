"""Tests for arena discovery."""
import logging

import pytest
from framework.discovery import discover_arenas, load_arena, ArenaDiscoveryError


def test_discover_arenas_finds_fake_arena(fixtures_dir):
    arenas = discover_arenas(fixtures_dir)
    ids = [a["arena_id"] for a in arenas]
    assert "fake-arena" in ids


def test_discover_arenas_logs_warning_on_broken_arena(tmp_path, caplog):
    # DR-0012: a malformed arena.yaml must be SKIPPED but logged, not silently
    # dropped. Write an arena.yaml that fails schema validation.
    bad = tmp_path / "broken-arena"
    bad.mkdir()
    (bad / "arena.yaml").write_text("arena_id: 123\n", encoding="utf-8")  # wrong type / missing fields
    with caplog.at_level(logging.WARNING, logger="framework.discovery"):
        arenas = discover_arenas(tmp_path)
    assert arenas == []
    assert any("Skipping arena" in r.message and "broken-arena" in r.message for r in caplog.records)


def test_discover_arenas_strict_reraises(tmp_path):
    bad = tmp_path / "broken-arena"
    bad.mkdir()
    (bad / "arena.yaml").write_text("arena_id: 123\n", encoding="utf-8")
    with pytest.raises(ArenaDiscoveryError):
        discover_arenas(tmp_path, strict=True)


def test_load_arena_returns_manifest_and_paths(fixtures_dir, fake_arena_dir):
    arena = load_arena(fake_arena_dir)
    assert arena["manifest"]["arena_id"] == "fake-arena"
    assert arena["root"] == fake_arena_dir
    assert arena["input_schema_path"].name == "input.schema.json"
    assert arena["output_schema_path"].name == "output.schema.json"


def test_load_arena_missing_arena_yaml_raises(tmp_path):
    with pytest.raises(ArenaDiscoveryError):
        load_arena(tmp_path)
