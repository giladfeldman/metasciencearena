"""The committed manifest snapshot must match what publish actually computes.

WHY THIS TEST EXISTS
--------------------
`contract/public_manifest.txt` is committed because
`publish.public_manifest()` CANNOT be computed on a bare checkout: it decides an
arena is publishable partly from its `task_sets/*/.private_seed`, and those are
gitignored secrets. Without them it degrades to 225 files and ZERO arenas — no
error, just a wrong answer.

That is not hypothetical. The `mirror-drift` CI job called it directly and so
reported **275 phantom STALE files on every run since it was added** (measured
2026-08-19 by deleting the seeds locally and re-running). An alarm that is always
on is worse than no alarm: it trains a reader to ignore the one time it is real.

A committed snapshot fixes that only while it stays true, which is what this
test is for. It runs where the seeds live and skips where they do not — the same
posture as the publish gate itself.

The snapshot also makes the published file list REVIEWABLE in a diff, which is
the check that caught `framework/gold/` and `build_gold.py` when no automated
gate could see them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from framework import publish

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "contract" / "public_manifest.txt"

_SEEDS = sorted(REPO.glob("arenas/*/task_sets/*/.private_seed"))
requires_private_seeds = pytest.mark.skipif(
    not _SEEDS,
    reason=("no arenas/*/task_sets/*/.private_seed on disk, so public_manifest() "
            "computes 0 arenas — the snapshot cannot be verified on a bare checkout"),
)


def _snapshot_paths() -> list[str]:
    return [
        line.strip()
        for line in SNAPSHOT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.skipif(
    not SNAPSHOT.is_file(),
    reason=("contract/public_manifest.txt is a SOURCE-repo artifact and is not part "
            "of the published package — this test file ships, its subject does not"),
)
def test_the_snapshot_exists_and_is_not_empty():
    """In the source repo: a missing snapshot breaks the drift job entirely.

    Skipped inside the published package. The snapshot describes what the mirror
    should contain, which is a fact about the SOURCE repo's publishing step; a
    consumer of the package neither has it nor needs it. Caught by the mirrored
    test suite, which runs this file in a tree built from the manifest — the
    same run that proves the public package's own tests pass.
    """
    paths = _snapshot_paths()
    assert len(paths) > 400, f"snapshot has only {len(paths)} paths — suspiciously small"
    assert any(p.startswith("arenas/") for p in paths), (
        "snapshot contains no arenas — it was generated without the private seeds"
    )


@requires_private_seeds
def test_the_snapshot_matches_what_publish_computes():
    """A stale snapshot would publish a wrong file list, silently."""
    computed = sorted(p.as_posix() for p in publish.public_manifest(REPO))
    snapshot = sorted(_snapshot_paths())
    missing = sorted(set(computed) - set(snapshot))
    extra = sorted(set(snapshot) - set(computed))
    assert not missing and not extra, (
        "contract/public_manifest.txt is stale — regenerate it with "
        "`python scripts/write_public_manifest.py`.\n"
        f"  computed but not in snapshot ({len(missing)}): {missing[:10]}\n"
        f"  in snapshot but not computed ({len(extra)}): {extra[:10]}"
    )


@requires_private_seeds
def test_the_snapshot_is_sorted_and_deduplicated():
    """So a diff shows a real change rather than a reordering."""
    paths = _snapshot_paths()
    assert paths == sorted(paths), "snapshot is not sorted — diffs become unreadable"
    assert len(paths) == len(set(paths)), "snapshot contains duplicates"
