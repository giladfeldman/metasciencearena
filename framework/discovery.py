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
    _validate_version_ladder(manifest, manifest_path)
    return {
        "manifest": manifest,
        "arena_id": manifest["arena_id"],
        "root": arena_dir,
        "input_schema_path": arena_dir / "schemas" / "input.schema.json",
        "output_schema_path": arena_dir / "schemas" / "output.schema.json",
        "generator_path": arena_dir / "generator.py",
        "scorer_path": arena_dir / "scorer.py",
    }


def output_schema_path_for(arena: dict, task_set_version: str | None) -> Path:
    """The output schema a record on ``task_set_version`` was validated against.

    ``schemas/<version>/output.schema.json`` when the task set declares its own
    answer shape (publication-bias-v1 v2 asks for statistics, v1 for a verdict);
    otherwise the arena-wide ``schemas/output.schema.json``. Mirrored by
    leaderboard-app/scripts/build-data.mjs::outputSchemaPath.
    """
    if task_set_version:
        per_version = Path(arena["root"]) / "schemas" / task_set_version / "output.schema.json"
        if per_version.is_file():
            return per_version
    return Path(arena["output_schema_path"])


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


# --- task-set version ladder ------------------------------------------------
#
# ORDER IS NOT MEANING. Which version the leaderboard builds from is decided by
# `status: current`, never by position in the list. The rule this replaces --
# "the last declared entry wins", in build-data.mjs::resolveTaskSetVersion --
# meant that merely DECLARING a new version made every run of the previous one
# invisible, with nothing failing anywhere.
#
# Arenas that have only ever had one task set declare no status at all; for them
# the legacy last-entry rule still applies, unchanged, so nothing had to be
# touched to keep the other 19 arenas building exactly as before.

#: A version with one of these statuses is immutable: the generator must
#: reproduce it byte-for-byte and the runner refuses to append records to it.
FROZEN_STATUSES = ("frozen", "superseded")


def _validate_version_ladder(manifest: dict, manifest_path: Path) -> None:
    """Fail loudly on a version ladder that cannot be resolved unambiguously.

    JSON Schema cannot express any of these: they are relations BETWEEN entries.
    """
    versions = manifest.get("task_set_versions") or []
    names = [v["version"] for v in versions]

    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ArenaDiscoveryError(
            f"{manifest_path}: duplicate task_set_versions entries: {sorted(dupes)}")

    statuses = [v.get("status") for v in versions]
    declared = [s for s in statuses if s]
    if not declared:
        return  # legacy single-ladder arena; the last-entry rule applies

    if len(declared) != len(versions):
        missing = [v["version"] for v in versions if not v.get("status")]
        raise ArenaDiscoveryError(
            f"{manifest_path}: task_set_versions mixes declared and undeclared "
            f"status; add a status to {missing}")
    n_current = statuses.count("current")
    if n_current != 1:
        raise ArenaDiscoveryError(
            f"{manifest_path}: exactly one task_set_version must have status "
            f"'current', found {n_current} in {names}")

    known = set(names)
    for v in versions:
        for field in ("supersedes", "superseded_by"):
            ref = v.get(field)
            if ref is not None and ref not in known:
                raise ArenaDiscoveryError(
                    f"{manifest_path}: task_set_version {v['version']}.{field} "
                    f"names '{ref}', which is not a declared version {sorted(known)}")
        if v.get("status") in FROZEN_STATUSES and not v.get("generator_fingerprint"):
            # WARNING, not an error: an arena is frozen before the fingerprint is
            # computed (`framework audit --write-fingerprints`).
            # But an unfingerprinted frozen version is UNGUARDED, which is the
            # exact failure mode this whole feature exists to close.
            logger.warning(
                "%s: task_set_version %s is %s but carries no "
                "generator_fingerprint - its immutability is UNGUARDED. Run "
                "`framework audit --write-fingerprints`.",
                manifest_path, v["version"], v["status"])


def version_ladder(manifest: dict) -> list[dict]:
    """Every declared task-set version, with `status` filled in for legacy arenas.

    Entries come back in declaration order and ALWAYS carry a `status`, so no
    caller has to re-implement the fallback.
    """
    versions = [dict(v) for v in (manifest.get("task_set_versions") or [])]
    if not versions:
        return []
    if any(v.get("status") for v in versions):
        return versions
    # Legacy: the last declared entry is current, everything before it is
    # superseded. That is precisely what the old build did, written down.
    for v in versions[:-1]:
        v["status"] = "superseded"
    versions[-1]["status"] = "current"
    return versions


def resolve_task_set_version(manifest: dict, *, default: str = "v1") -> str:
    """The task-set version new runs go to and the leaderboard defaults to."""
    for v in version_ladder(manifest):
        if v["status"] == "current":
            return v["version"]
    return default


def task_set_status(manifest: dict, version: str) -> str | None:
    """Status of one declared version, or None when the arena does not declare it."""
    for v in version_ladder(manifest):
        if v["version"] == version:
            return v["status"]
    return None


def is_frozen(manifest: dict, version: str) -> bool:
    """True when `version` is immutable and must not receive new run records."""
    return task_set_status(manifest, version) in FROZEN_STATUSES
