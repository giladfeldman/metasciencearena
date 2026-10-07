"""FredFridaOnlyAdapter — deterministic curated-only lookup, no network calls."""
from __future__ import annotations

import json
from pathlib import Path

from framework.player_adapter import PlayerAdapter, register_adapter_class


class FredFridaOnlyAdapter(PlayerAdapter):
    """Looks up DOIs in a vendored FRED/FORRT-style snapshot. No network."""

    catalog_path: Path

    def __init__(self, *args, catalog_path: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalog_path = Path(catalog_path)
        self._catalog: dict | None = None

    def prepare(self) -> None:
        with self.catalog_path.open("r", encoding="utf-8") as f:
            self._catalog = json.load(f)

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._catalog is None:
            self.prepare()
        direction = envelope["input"]["direction"]
        doi = envelope["input"]["doi"]
        bucket = self._catalog.get(direction, {})
        raw = bucket.get(doi, [])
        matches = []
        for entry in raw:
            if direction == "targets":
                matches.append({
                    "replication_doi": doi,
                    "original_doi": entry.get("original_doi"),
                    "outcome": entry.get("outcome", ""),
                    "confidence": 1.0,
                    "evidence": entry.get("label", "FRED/FORRT curated"),
                    "provenance": ["curated"],
                    "abstained": False,
                })
            else:
                matches.append({
                    "replication_doi": entry.get("replication_doi"),
                    "original_doi": doi,
                    "outcome": entry.get("outcome", ""),
                    "confidence": 1.0,
                    "evidence": entry.get("label", "FRED/FORRT curated"),
                    "provenance": ["curated"],
                    "abstained": False,
                })
        return {
            "direction": direction,
            "input_doi": doi,
            "matches": matches,
            "player_strategy_notes": "fred-frida-only (curated snapshot, no network)",
        }


register_adapter_class("FredFridaOnlyAdapter", FredFridaOnlyAdapter)
