"""Tests for FredFridaOnlyAdapter."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from players.adapters.fred_frida_only import FredFridaOnlyAdapter


_CATALOG = Path(__file__).resolve().parents[2] / "arenas" / "replication-target-lookup-v1" / "catalogs" / "fred_frida_v1.json"


def _envelope(direction: str, doi: str) -> dict:
    return {
        "task_id": "t",
        "arena_id": "replication-target-lookup-v1",
        "task_set_version": "v1",
        "difficulty": {"tier": 1},
        "input": {"direction": direction, "doi": doi, "tier": 1},
    }


def test_known_target_returns_match():
    adapter = FredFridaOnlyAdapter(
        player_id="fred-frida-only",
        player_version="day1",
        player_type="tool",
        confidence_strategy="implicit-1.0",
        deterministic=True,
        catalog_path=str(_CATALOG),
    )
    adapter.prepare()
    out = adapter.play_task(_envelope("targets", "10.1177/1745691616674458"), timeout_s=10)
    assert out["direction"] == "targets"
    assert out["matches"]
    assert out["matches"][0]["original_doi"] == "10.1037/0022-3514.54.5.768"
    assert "curated" in out["matches"][0]["provenance"]


def test_unknown_doi_returns_empty_matches():
    adapter = FredFridaOnlyAdapter(
        player_id="fred-frida-only",
        player_version="day1",
        player_type="tool",
        confidence_strategy="implicit-1.0",
        deterministic=True,
        catalog_path=str(_CATALOG),
    )
    adapter.prepare()
    out = adapter.play_task(_envelope("targets", "10.UNKNOWN/zzz"), timeout_s=10)
    assert out["direction"] == "targets"
    assert out["matches"] == []
