"""Quarantine is the DEFAULT for a new player, not something to remember.

docs/recovery-regime.md (Phase 3) adopts "quarantine-by-default for new players
and records" and, until 2026-10-07, nothing implemented it: the phrase lived in a
document, which that same document measures as a ~8-day half-life. This test is
the executable half.

THE RULE. Any player whose FIRST published record is dated on or after
`QUARANTINE_ADOPTED_ON` must declare `quarantine:` (shown on the board as
"Provisional") or `quarantine_released:` (the audit that lifted it, on the
record). A player that predates the rule is not retro-quarantined -- that would
be a claim about records this rule never governed.

"Published" is what build-data.mjs serves: a .jsonl directly in
arenas/<a>/runs/<v>/, never a subdirectory.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from framework.paths import arenas_root, registry_path
from framework.registry import QUARANTINE_ADOPTED_ON, load_registry, quarantine_problem


def _first_published_day() -> dict[str, str]:
    first: dict[str, str] = {}
    for path in sorted(arenas_root().glob("*/runs/*/*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                day = str(rec.get("timestamp_utc", ""))[:10]
                pid = rec.get("player_id")
                if pid and day and (pid not in first or day < first[pid]):
                    first[pid] = day
    return first


# Keyed on leaderboard-app/, NOT on the record scan: a broken scan must still fail
# the non-vacuity assert below in the full repo.
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "leaderboard-app").is_dir(),
    reason="the public mirror ships no run records (leaderboard-app/ and runs are private)",
)
def test_every_player_first_published_since_adoption_is_quarantined_or_released():
    by_id = {e["player_id"]: e for e in load_registry(registry_path())}
    first = _first_published_day()
    new = sorted(p for p, d in first.items() if d >= QUARANTINE_ADOPTED_ON)
    # Non-vacuity: the rule was adopted together with the twelve extractor players
    # (2026-10-07), so an empty set means the record scan broke, not that the rule
    # was honoured.
    assert new, ("no player has a published record dated on/after "
                 f"{QUARANTINE_ADOPTED_ON}; the scan of arenas/*/runs/*/*.jsonl is broken")
    unquarantined = [p for p in new
                     if p not in by_id or not (by_id[p].get("quarantine") or by_id[p].get("quarantine_released"))]
    assert not unquarantined, (
        f"{len(unquarantined)} player(s) first published on/after {QUARANTINE_ADOPTED_ON} "
        f"carry neither `quarantine:` nor `quarantine_released:` in players/registry.yaml: "
        f"{unquarantined}. New players publish as Provisional until a blinded audit "
        f"releases them (docs/recovery-regime.md, Phase 3).")


@pytest.mark.parametrize("entry,ok", [
    ({"quarantine": {"since": "2026-10-07", "reason": "new"}}, True),
    ({"quarantine_released": {"on": "2026-11-01", "by": "blinded audit 2026-11"}}, True),
    ({}, True),
    ({"quarantine": True}, False),
    ({"quarantine": {"since": "2026-10-07"}}, False),
    ({"quarantine": {"since": "Oct 7", "reason": "x"}}, False),
    ({"quarantine": {"since": "2026-10-07", "reason": "x"},
      "quarantine_released": {"on": "2026-11-01", "by": "y"}}, False),
    ({"quarantine_released": {"on": "2026-11-01"}}, False),
])
def test_quarantine_declaration_shape(entry, ok):
    assert (quarantine_problem(entry) is None) is ok
