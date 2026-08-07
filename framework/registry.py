"""Load + validate player registry."""
from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_FIELDS = {"player_id", "player_type", "player_version", "adapter_class", "confidence_strategy", "deterministic"}
ALLOWED_TYPES = {"platform", "tool", "human-baseline", "ai-model"}
ALLOWED_CONFIDENCE = {"native", "implicit-1.0", "derived"}


class RegistryError(Exception):
    pass


def load_registry(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise RegistryError(f"{path}: registry must be a YAML list")

    seen_ids: set[str] = set()
    for entry in data:
        if not isinstance(entry, dict):
            raise RegistryError(f"{path}: each entry must be a mapping; got {type(entry).__name__}")
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise RegistryError(f"{path}: entry missing fields {sorted(missing)}: {entry}")
        if entry["player_type"] not in ALLOWED_TYPES:
            raise RegistryError(f"{path}: player_type must be one of {sorted(ALLOWED_TYPES)}, got {entry['player_type']}")
        if entry["confidence_strategy"] not in ALLOWED_CONFIDENCE:
            raise RegistryError(f"{path}: confidence_strategy must be one of {sorted(ALLOWED_CONFIDENCE)}, got {entry['confidence_strategy']}")
        if entry["player_id"] in seen_ids:
            raise RegistryError(f"{path}: duplicate player_id {entry['player_id']}")
        seen_ids.add(entry["player_id"])

    return data
