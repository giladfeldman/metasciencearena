"""Discover and validate arenas conforming to the C-level Arena Contract."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from framework.paths import schema_path

logger = logging.getLogger(__name__)

# Resolved through the package, not the checkout: under `pip install` a
# `parents[1]`-derived path lands in site-packages, where `contract/` does not
# exist. See framework/paths.py.
CONTRACT_SCHEMA_PATH = schema_path("arena_manifest.schema.json")


class ArenaDiscoveryError(Exception):
    pass


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_arena(arena_dir: Path) -> dict:
    """Load + validate one arena directory. Returns dict with manifest + paths."""
    manifest_path = arena_dir / "arena.yaml"
    if not manifest_path.is_file():
        raise ArenaDiscoveryError(f"Missing arena.yaml in {arena_dir}")
    manifest = _load_yaml(manifest_path)
    schema = _load_json(CONTRACT_SCHEMA_PATH)
    errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda e: e.path)
    if errors:
        msgs = [f"{manifest_path}: {e.message}" for e in errors]
        raise ArenaDiscoveryError("\n".join(msgs))
    return {
        "manifest": manifest,
        "arena_id": manifest["arena_id"],
        "root": arena_dir,
        "input_schema_path": arena_dir / "schemas" / "input.schema.json",
        "output_schema_path": arena_dir / "schemas" / "output.schema.json",
        "generator_path": arena_dir / "generator.py",
        "scorer_path": arena_dir / "scorer.py",
    }


def discover_arenas(search_root: Path, *, strict: bool = False) -> list[dict]:
    """Find every arena under search_root. Each must have an arena.yaml.

    A directory whose arena.yaml fails schema validation is SKIPPED (so one
    broken arena can't take down the whole tournament) — but the skip is now
    logged at WARNING with the path + reason. Previously it was a silent
    ``continue`` that made a typo'd arena.yaml vanish with zero diagnostic
    output (the 2026-06-06 trap, DR-0012). Pass ``strict=True`` to re-raise
    instead of skipping (useful in CI / `framework audit`).
    """
    arenas: list[dict] = []
    for manifest_path in sorted(search_root.rglob("arena.yaml")):
        try:
            arenas.append(load_arena(manifest_path.parent))
        except ArenaDiscoveryError as exc:
            if strict:
                raise
            logger.warning("Skipping arena %s: %s", manifest_path.parent, exc)
    return arenas
