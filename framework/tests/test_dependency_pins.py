"""The docpluck the arena RUNS must be the docpluck the arena DECLARES.

WHY THIS FILE EXISTS. The dependency manifest that governs this project
declares an `arena -> docpluck` pin edge and records it as UNCHECKED. The
project's own standing guidance says plainly that
adding a consumer-side contract test at an unchecked edge is worth more than
almost any feature work. This is that test for this edge.

WHAT WENT WRONG WITHOUT IT, measured 2026-08-29 during `/ship`. Three surfaces
that must agree disagreed three ways:

    players/registry.yaml   declared  docpluck-2.4.137-*   (bumped in 585cffb)
    requirements.lock       pinned    2.4.136  (commit a1979aed)
    .venv                   installed 2.4.136

The 362 committed docpluck records were all written at 2.4.137 (checked:
`player_version` equals `resolved_tool_version` on every one, 0 mislabelled), so
the RECORDS were fine and the REPRODUCIBILITY was not: a fresh checkout
installing from the lock would have got 2.4.136 and produced different numbers
under the same label. Nothing in the repo compared the lock to the runtime.

WHAT THIS TEST DOES NOT DO. It does not compare against docpluck's working tree.
That comparison is what makes the fleet row unclearable: docpluck's tree declares
2.4.138 while its newest tag is v2.4.137 and the tree is dirty, so an arena that
pinned "whatever upstream currently is" would be pinning uncommitted work. The
standing directive for the sibling consumer edge is explicit -- consume only a
released tag, never uncommitted work -- and this arena follows it. A residual
`arena -> docpluck` row therefore means DOCPLUCK has unreleased dirty work, not
that this consumer is stale, and it cannot be cleared from this repository.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LOCK = REPO / "requirements.lock"

#: `docpluck @ git+https://…@<40-hex>` — the pin is a COMMIT on purpose. A tag is
#: mutable and four docpluck versions have been live on one machine at once, so a
#: tag reproduces the LABEL while the commit reproduces the LIBRARY.
_PIN = re.compile(
    r"^docpluck\s*@\s*git\+https://\S+?@([0-9a-f]{40})\s*$", re.MULTILINE)


def _locked_commit() -> str:
    m = _PIN.search(LOCK.read_text(encoding="utf-8"))
    assert m, (
        "requirements.lock no longer pins docpluck by 40-hex commit. It must: "
        "`docpluck==<version>` does NOT resolve, because our builds were never "
        "published to PyPI (newest there is 2.4.135).")
    return m.group(1)


def _installed():
    """(version, commit) of the docpluck actually importable, or (None, None).

    Read through `importlib.metadata`, NOT by globbing `<repo>/.venv`. The first
    version of this globbed the venv path and was wrong in exactly the way this
    file warns about elsewhere: it asked a different question from the skip guard
    above, which uses the ambient interpreter. In the public-mirror checkout —
    a temp directory with no `.venv` — the guard saw docpluck installed while the
    glob found nothing, so the test RAN and then failed on a missing commit. It
    was the mirrored suite that caught it, not the local one, because locally the
    two happen to be the same directory. Ask the distribution where it came from.
    """
    try:
        dist = md.distribution("docpluck")
    except md.PackageNotFoundError:
        return None, None
    commit = None
    raw = dist.read_text("direct_url.json")
    if raw:
        commit = json.loads(raw).get("vcs_info", {}).get("commit_id")
    return dist.version, commit


docpluck_absent = pytest.mark.skipif(
    _installed()[0] is None,
    reason="docpluck is not installed in this environment — a fresh checkout that "
           "has not run `pip install -r requirements.lock` is not a pin failure")


def test_the_lock_pins_docpluck_by_commit():
    """The pin must exist and be a commit. Runs even without docpluck installed,
    because a lock that stopped pinning is a defect on any machine."""
    assert len(_locked_commit()) == 40


@docpluck_absent
def test_the_installed_docpluck_is_the_commit_the_lock_pins():
    """THE EDGE. A fresh checkout must get the library the published rows came from.

    This is the assertion that was missing: the lock said a1979aed (2.4.136) while
    the venv ran 2.4.136 and every record claimed 2.4.137.
    """
    version, commit = _installed()
    locked = _locked_commit()
    assert commit is not None, (
        f"docpluck {version} is installed but carries no direct_url.json, so what "
        "it was built from cannot be established. Reinstall from the lock: "
        "`pip install --no-deps --force-reinstall -r requirements.lock`.")
    assert commit == locked, (
        f"docpluck installed from {commit[:12]} but requirements.lock pins "
        f"{locked[:12]}. A published score is attributed to the pinned library; "
        "reinstall from the lock, or update the lock IN THE SAME COMMIT as the "
        "re-run that makes the new version's numbers real.")


@docpluck_absent
def test_the_registry_label_matches_the_installed_version():
    """The provenance LABEL and the runtime must agree BEFORE a run, not after.

    `player_version` is stamped onto records at write time, so bumping it after a
    run mislabels every record that run produced — the exact defect the fleet's
    `provenance-label` edges exist to prevent. Bumping it BEFORE, with the runtime
    already matching, is the correct order and this test pins that order.
    """
    yaml = pytest.importorskip("yaml")
    version, _commit = _installed()
    registry = yaml.safe_load((REPO / "players" / "registry.yaml").read_text(encoding="utf-8"))
    declared = {p["player_version"] for p in registry
                if str(p.get("player_id", "")).startswith("docpluck")}
    assert declared, "no docpluck players in the registry — the probe is broken"
    # `docpluck-2.4.137-academic` must contain the installed `2.4.137`.
    wrong = sorted(d for d in declared if version not in d)
    assert not wrong, (
        f"docpluck {version} is installed but the registry declares {wrong}. "
        "Every record a run writes would carry a label naming a version that did "
        "not produce it. Fix the ORDER: install the version, then bump the label, "
        "then re-run — never the other way round.")
