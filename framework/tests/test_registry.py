"""Tests for player registry loader."""
import pytest
from framework.registry import load_registry, RegistryError


def test_load_registry_returns_player_dicts(fake_registry_path):
    players = load_registry(fake_registry_path)
    ids = [p["player_id"] for p in players]
    assert "stub-pass" in ids
    assert "stub-fail" in ids


def test_load_registry_rejects_missing_required_fields(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- player_id: only-id\n", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(bad)


def test_load_registry_rejects_duplicate_ids(tmp_path):
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "- {player_id: x, player_type: tool, player_version: '1', adapter_class: A, confidence_strategy: native, deterministic: true}\n"
        "- {player_id: x, player_type: tool, player_version: '1', adapter_class: A, confidence_strategy: native, deterministic: true}\n",
        encoding="utf-8",
    )
    with pytest.raises(RegistryError):
        load_registry(bad)
