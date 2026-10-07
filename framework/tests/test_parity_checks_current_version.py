"""The parity gate must examine the version each arena is ACTUALLY SERVING.

WHY THIS FILE EXISTS. `tools/check_parity.py` hardcoded the literal "v1" in both
call sites -- the `--all` loop that sciencearena-qa Phase 5 runs as a gate, and
the single-arena default. So on every arena that had minted a successor the gate
checked a SUPERSEDED task set and never the one being served, and it did so
SILENTLY, because a superseded task set parity-matches perfectly well. Fixed
2026-09-17; on the next run the gate went red on stats-extraction-v1 v5, a break
that had been live in the served task set since v3 (released 2026-08-21) and
invisible for 27 days.

`framework/tests/test_parity.py::test_stats_extraction_arena_passes_parity`
carried the SAME hardcoded "v1", so the whole suite stayed green throughout.
A gate and its test sharing one blind spot is why this check is structural: it
asserts the gate LOOKS AT the right version, which no amount of "does the arena
pass" testing can establish.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from framework.discovery import resolve_task_set_version

REPO_ROOT = Path(__file__).resolve().parents[2]
ARENAS_ROOT = REPO_ROOT / "arenas"


def _load_cli():
    """Import tools/check_parity.py as a module (it is a script, not a package)."""
    path = REPO_ROOT / "tools" / "check_parity.py"
    spec = importlib.util.spec_from_file_location("_check_parity_cli", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _arena_dirs() -> list[Path]:
    return sorted(m.parent for m in ARENAS_ROOT.rglob("arena.yaml"))


def _manifest(arena_dir: Path) -> dict:
    return yaml.safe_load((arena_dir / "arena.yaml").read_text(encoding="utf-8")) or {}


@pytest.mark.skipif(
    not (REPO_ROOT / "tools" / "check_parity.py").is_file(),
    reason="tools/check_parity.py is private QA tooling and is not in the public mirror; "
           "the ladder assertions below still run there",
)
@pytest.mark.parametrize("arena_dir", _arena_dirs(), ids=lambda p: p.name)
def test_cli_resolves_the_current_version_not_a_hardcoded_literal(arena_dir: Path):
    """What the gate selects must equal what `status: current` declares.

    This is the assertion that would have caught the original defect: it compares
    the CLI's own choice against the single source of truth, so reintroducing a
    literal anywhere in `_current_version` fails here for every arena that has a
    successor.
    """
    cli = _load_cli()
    assert cli._current_version(arena_dir) == resolve_task_set_version(_manifest(arena_dir))


def test_at_least_one_arena_has_moved_past_v1():
    """CONTROL -- proves the test above can FAIL, and is not passing vacuously.

    If every arena were still on v1, `_current_version` could go back to
    `return "v1"` and every assertion above would still pass. That is exactly the
    false green that hid this bug for 27 days. This test fails the moment the
    fleet no longer contains a counter-example, telling a future reader that the
    check above has stopped discriminating and needs a synthetic fixture instead.
    """
    moved = {d.name: resolve_task_set_version(_manifest(d))
             for d in _arena_dirs()
             if resolve_task_set_version(_manifest(d)) != "v1"}
    assert moved, (
        "no arena is past v1, so test_cli_resolves_the_current_version_not_a_"
        "hardcoded_literal can no longer distinguish a real resolver from "
        "`return \"v1\"`. Add a synthetic multi-version arena fixture."
    )


def test_hardcoded_v1_resolver_would_fail_the_check():
    """CONTROL (two-sided) -- the ORIGINAL buggy resolver must be rejected here.

    Substituting the pre-2026-09-17 implementation (`lambda _dir: "v1"`) has to
    make the parametrized check fail on at least one arena. Without this, a
    resolver that silently degraded to v1 could pass unnoticed.
    """
    buggy = lambda _arena_dir: "v1"  # noqa: E731 -- the historical implementation
    disagreements = [d.name for d in _arena_dirs()
                     if buggy(d) != resolve_task_set_version(_manifest(d))]
    assert disagreements, (
        "the historical hardcoded-v1 resolver agrees with `status: current` on "
        "every arena, so this suite cannot tell them apart"
    )


@pytest.mark.parametrize("arena_dir", _arena_dirs(), ids=lambda p: p.name)
def test_current_version_is_declared_in_the_ladder(arena_dir: Path):
    """The version the gate selects must be one the arena actually declares.

    This is the checkout-independent half of "does the selected version exist".
    A `status: current` naming a version absent from `task_set_versions` would
    send the gate to generate from an entry nothing was ever minted for -- and
    unlike the directory check below, the ladder is always in git, so this runs
    everywhere including CI.
    """
    manifest = _manifest(arena_dir)
    version = resolve_task_set_version(manifest)
    declared = {v.get("version") for v in (manifest.get("task_set_versions") or [])}
    # An arena with no ladder at all predates versioning and defaults to v1.
    if not declared:
        assert version == "v1"
        return
    assert version in declared, (
        f"{arena_dir.name} resolves to {version}, which is not in its declared "
        f"task_set_versions {sorted(d for d in declared if d)}"
    )


@pytest.mark.parametrize("arena_dir", _arena_dirs(), ids=lambda p: p.name)
def test_current_version_has_a_task_set_directory(arena_dir: Path):
    """The selected version's directory exists -- WHERE ONE CAN EXIST.

    SKIPS on arenas whose `task_sets/` holds no tracked files, and that skip is
    not a formality: `pdf-citation-matching-v1` and `pdf-reference-parsing-v1`
    have ZERO tracked files under `task_sets/` because their held-out corpora and
    private seeds are all gitignored, so git creates no directory in a fresh
    checkout. Asserting unconditionally made this test pass on a workstation and
    fail in CI -- measured, both arenas, in a detached worktree at this commit.

    The ladder assertion above is the guard that always runs. This one adds the
    on-disk check wherever the repository can actually supply it.
    """
    version = resolve_task_set_version(_manifest(arena_dir))
    task_sets = arena_dir / "task_sets"
    if not any(task_sets.glob("v*/")):
        pytest.skip(
            f"{arena_dir.name}: no task_sets/v*/ on disk -- everything under "
            "task_sets/ is gitignored for this arena, so a checkout has no "
            "directory to check. Covered by "
            "test_current_version_is_declared_in_the_ladder instead."
        )
    assert (task_sets / version).is_dir(), (
        f"{arena_dir.name} declares {version} current but "
        f"task_sets/{version}/ does not exist"
    )
