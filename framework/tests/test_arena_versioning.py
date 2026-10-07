"""Task-set version ladder: resolution, freezing, and the guards around both.

Added 2026-08-21 with the arena-versioning feature. The rule it replaces was
"the last declared entry in ``task_set_versions`` is the one that counts", which
meant that merely DECLARING a successor made every run of the previous version
invisible to the leaderboard, with nothing failing anywhere. An arena that
freezes v1 and publishes v2 needs BOTH to keep rendering, and the frozen one
needs to be provably unchanged.

The resolution rule is deliberately implemented twice -- here and in
``leaderboard-app/scripts/build-data.mjs::versionLadder`` -- because making
version resolution depend on a Python subprocess would inherit the failure mode
that broke every production deploy for four days in 2026-08 (numpy became a
build dependency of the website). Each side is pinned in its own runtime against
the SAME published artifact: this module checks the Python resolution against
``public/data/index.json``, and ``lib/__tests__/task-set-version-agreement.test.ts``
checks the index against ``arena.yaml``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from framework.discovery import (
    ArenaDiscoveryError,
    is_frozen,
    load_arena,
    resolve_task_set_version,
    task_set_status,
    version_ladder,
)
from framework.paths import arenas_root

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_INDEX = REPO_ROOT / "leaderboard-app" / "public" / "data" / "index.json"


def _manifest(entries: list[dict]) -> dict:
    return {"task_set_versions": entries}


# --- resolution --------------------------------------------------------------


def test_legacy_ladder_resolves_to_the_last_entry():
    """An arena that declares no status keeps the old behaviour exactly.

    19 of 22 arenas are in this state; changing what they resolve to would have
    silently re-pointed the whole leaderboard.
    """
    m = _manifest([
        {"version": "v1", "n_tasks": 10, "released_on": "2026-01-01"},
        {"version": "v2", "n_tasks": 10, "released_on": "2026-02-01"},
    ])
    assert resolve_task_set_version(m) == "v2"
    assert [v["status"] for v in version_ladder(m)] == ["superseded", "current"]


def test_status_beats_position():
    """`current` is what resolves -- even when it is not the last entry.

    This is the whole point: a version can be declared before it is current, and
    a frozen predecessor must not be resurrected by being listed last.
    """
    m = _manifest([
        {"version": "v1", "n_tasks": 10, "released_on": "2026-01-01", "status": "current"},
        {"version": "v2", "n_tasks": 10, "released_on": "2026-02-01", "status": "frozen"},
    ])
    assert resolve_task_set_version(m) == "v1"
    assert is_frozen(m, "v2") is True
    assert is_frozen(m, "v1") is False


def test_unknown_version_has_no_status():
    m = _manifest([{"version": "v1", "n_tasks": 1, "released_on": "2026-01-01"}])
    assert task_set_status(m, "v9") is None
    assert is_frozen(m, "v9") is False


# --- ladder validation -------------------------------------------------------


def _write_arena(tmp_path: Path, entries: list[dict]) -> Path:
    """A minimal schema-valid arena.yaml carrying the given version ladder."""
    d = tmp_path / "ladder-arena"
    d.mkdir()
    (d / "arena.yaml").write_text(yaml.safe_dump({
        "arena_id": "ladder-arena",
        "taxonomy_leaf_id": "ladder",
        "version": "0.1.0",
        "owner": "test",
        "license": "MIT",
        "input_artifact": "text",
        "output_to_judge": "text",
        "difficulty_axes": [{"id": "a", "min": 0, "max": 1, "description": "d"}],
        "task_set_versions": entries,
        "display_columns": [{
            "id": "composite", "source": "score.primary", "label": "C",
            "description": "d", "direction": "higher", "format": "f3", "primary": True,
        }],
        "tier_pivot": {
            "axis": "a",
            "composite_formula": "x",
            "values": [{"value": 0, "label": "A", "sub": "a"}],
        },
    }), encoding="utf-8")
    return d


def test_two_current_versions_is_a_hard_error(tmp_path):
    d = _write_arena(tmp_path, [
        {"version": "v1", "n_tasks": 1, "released_on": "2026-01-01", "status": "current"},
        {"version": "v2", "n_tasks": 1, "released_on": "2026-02-01", "status": "current"},
    ])
    with pytest.raises(ArenaDiscoveryError, match="exactly one"):
        load_arena(d)


def test_no_current_version_is_a_hard_error(tmp_path):
    d = _write_arena(tmp_path, [
        {"version": "v1", "n_tasks": 1, "released_on": "2026-01-01", "status": "frozen"},
    ])
    with pytest.raises(ArenaDiscoveryError, match="exactly one"):
        load_arena(d)


def test_partially_declared_status_is_a_hard_error(tmp_path):
    """Mixing declared and undeclared status is ambiguous, so it is refused.

    Left alone it would resolve by the *declared* rule while a reader looking at
    the undeclared entry assumes the positional one -- the two disagree exactly
    when it matters.
    """
    d = _write_arena(tmp_path, [
        {"version": "v1", "n_tasks": 1, "released_on": "2026-01-01"},
        {"version": "v2", "n_tasks": 1, "released_on": "2026-02-01", "status": "current"},
    ])
    with pytest.raises(ArenaDiscoveryError, match="mixes declared and undeclared"):
        load_arena(d)


def test_dangling_supersedes_reference_is_a_hard_error(tmp_path):
    d = _write_arena(tmp_path, [
        {"version": "v1", "n_tasks": 1, "released_on": "2026-01-01", "status": "frozen"},
        {"version": "v2", "n_tasks": 1, "released_on": "2026-02-01", "status": "current",
         "supersedes": "v7"},
    ])
    with pytest.raises(ArenaDiscoveryError, match="not a declared version"):
        load_arena(d)


# --- the real arenas ---------------------------------------------------------


def test_every_real_arena_resolves_exactly_one_current_version():
    arenas = sorted(arenas_root().glob("*/arena.yaml"))
    assert arenas, "no arenas found -- a vacuous pass would prove nothing"
    for path in arenas:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        ladder = version_ladder(manifest)
        assert ladder, f"{path}: empty version ladder"
        current = [v for v in ladder if v["status"] == "current"]
        assert len(current) == 1, f"{path}: {len(current)} current versions"
        assert resolve_task_set_version(manifest) == current[0]["version"]


@pytest.mark.skipif(not PUBLISHED_INDEX.exists(),
                    reason="leaderboard bundle not built (public/data is gitignored)")
def test_python_resolution_matches_the_published_index():
    """The two implementations of the rule must not drift apart.

    Checked against the built artifact rather than by calling the other language,
    so neither test suite acquires a dependency on the other's runtime.
    """
    index = json.loads(PUBLISHED_INDEX.read_text(encoding="utf-8"))
    checked = 0
    for entry in index["arenas"]:
        arena_id = entry["arena_id"]
        if arena_id.startswith("planned:"):
            continue
        path = arenas_root() / arena_id / "arena.yaml"
        if not path.exists():
            continue
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        resolved = resolve_task_set_version(manifest)
        assert resolved == entry["task_set_version"], (
            f"{arena_id}: Python resolves {resolved} but the published index "
            f"advertises {entry['task_set_version']}"
        )
        assert [v["version"] for v in version_ladder(manifest)] == \
            [v["version"] for v in entry.get("task_set_versions", [])], \
            f"{arena_id}: ladder disagrees with the published one"
        checked += 1
    assert checked > 0, "no arenas compared -- a vacuous pass would prove nothing"


def test_runner_refuses_to_write_into_a_frozen_task_set(tmp_path):
    """The runner must refuse before it builds or prepares a single adapter.

    Appending to a frozen version silently rewrites published history: the board
    keeps serving that version's page while the numbers underneath it move.
    """
    from framework.runner import run_tournament

    d = _write_arena(tmp_path, [
        {"version": "v1", "n_tasks": 1, "released_on": "2026-01-01", "status": "frozen"},
        {"version": "v2", "n_tasks": 1, "released_on": "2026-02-01", "status": "current",
         "supersedes": "v1"},
    ])
    (d / "schemas").mkdir()
    (d / "schemas" / "output.schema.json").write_text('{"type": "object"}', encoding="utf-8")
    (d / "generator.py").write_text("def generate(v, seed):\n    return []\n", encoding="utf-8")
    (d / "scorer.py").write_text("def score(o, g):\n    return {'primary': 0.0}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen"):
        run_tournament(
            arena_dir=d,
            task_set_version="v1",
            registry_path=REPO_ROOT / "players" / "registry.yaml",
            player_ids=[],
            output_path=tmp_path / "out.jsonl",
        )
