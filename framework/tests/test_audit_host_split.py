"""One ``player_id`` must not span two machines.

WHY A GATE AND NOT JUST A LABEL (2026-08-21).

Recording the host in ``provenance`` makes the machine *visible*. It does not
make it *safe*, because nothing reads it. Both leaderboard engines group strictly
by ``player_id``:

* ``framework/leaderboard.py`` — ``grouped[player_id]``, and
  ``versions[pid] = r.get("player_version")`` keeps the FIRST label it sees.
* ``leaderboard-app/scripts/build-data.mjs`` — ``recordsByPid``, same
  first-record-wins for ``playerVersions``.

So dropping desktop records beside laptop records in one run file publishes a
single row: one machine's version label over the mean of both machines' scores,
with no test failing and nothing visibly wrong. That is the "unconsumed data
rots" pattern — a field nothing consumes — applied to the one axis this project
just spent a module recording.

This gate consumes it.
"""
from framework import audit


def _rec(player_id, fingerprint, task_id="t1"):
    prov = {"tested_at_utc": "2026-08-21T00:00:00Z"}
    if fingerprint is not None:
        prov["host_fingerprint"] = {"fingerprint": fingerprint, "label": f"box-{fingerprint}"}
    return {"player_id": player_id, "task_id": task_id, "provenance": prov}


def test_one_player_on_one_machine_passes():
    rep = audit.host_split_players([
        _rec("p", "aaa", "t1"), _rec("p", "aaa", "t2"), _rec("p", "aaa", "t3"),
    ])
    assert rep.ok
    assert not rep.problems


def test_one_player_across_two_machines_fails():
    """THE defect this exists to catch."""
    rep = audit.host_split_players([
        _rec("local-gpt-oss-20b-q8-grim", "laptopfp", "t1"),
        _rec("local-gpt-oss-20b-q8-grim", "desktopfp", "t2"),
    ])
    assert not rep.ok
    assert len(rep.problems) == 1
    msg = rep.problems[0]
    assert "local-gpt-oss-20b-q8-grim" in msg
    assert "laptopfp" in msg and "desktopfp" in msg


def test_different_players_on_different_machines_is_fine():
    """Two players legitimately run wherever they run. The gate is per player."""
    rep = audit.host_split_players([
        _rec("player-a", "laptopfp"), _rec("player-b", "desktopfp"),
    ])
    assert rep.ok


def test_legacy_plus_one_known_machine_reports_but_does_not_gate():
    """Codex #3 (High), 2026-08-21 — this gated, and it was a false positive.

    A player with old records (written before the fingerprint existed) and new
    records from THE SAME machine failed the gate, because "unrecorded" was
    counted as a second machine. That is the common case the first time anyone
    re-runs an existing player, and a gate that fires on the common case is one
    the operator learns to skip — the exact rot LESSONS records for
    "expected gate failure" lists.

    The unrecorded bucket is still NAMED rather than silently merged (it is a
    third answer: the machine is genuinely unknown). It just does not gate.
    """
    rep = audit.host_split_players([
        _rec("p", None, "t1"),         # legacy: machine unrecorded
        _rec("p", "desktopfp", "t2"),  # new run, one known machine
    ])
    assert rep.ok, "legacy + ONE known machine must not gate"
    assert "p" in rep.mixed_with_unrecorded
    assert not rep.problems


def test_legacy_plus_two_known_machines_still_gates():
    """The unrecorded bucket must not become an excuse that hides a real mix."""
    rep = audit.host_split_players([
        _rec("p", None, "t1"),
        _rec("p", "laptopfp", "t2"),
        _rec("p", "desktopfp", "t3"),
    ])
    assert not rep.ok


def test_all_legacy_records_do_not_trip_the_gate():
    """Today's published corpus is entirely pre-fingerprint.

    The gate must stay silent until someone actually mixes machines, or it is
    noise from the day it ships and gets ignored — the failure mode recorded in
    LESSONS as 'a documented expected-failure list rots into a blanket excuse'.
    """
    rep = audit.host_split_players([_rec("p", None, f"t{i}") for i in range(20)])
    assert rep.ok
    assert rep.n_unverifiable == 20


def test_report_counts_players_and_machines():
    rep = audit.host_split_players([
        _rec("a", "fp1"), _rec("b", "fp1"), _rec("c", "fp2"),
    ])
    assert rep.n_players == 3
    assert rep.n_hosts == 2


def test_problem_message_names_the_remedy():
    """A gate that says only 'wrong' makes the next person guess."""
    rep = audit.host_split_players([_rec("p", "fp1"), _rec("p", "fp2")])
    assert "_archive" in rep.problems[0]


def test_arena_id_is_carried_for_reporting():
    rep = audit.host_split_players([_rec("p", "fp1")], arena_id="grim-consistency-v1")
    assert rep.arena_id == "grim-consistency-v1"


def test_missing_provenance_block_does_not_crash():
    rep = audit.host_split_players([{"player_id": "p", "task_id": "t1"}])
    assert rep.ok
    assert rep.n_unverifiable == 1


def _rec_v(player_id, fingerprint, version, task_id="t1"):
    return {"player_id": player_id, "task_id": task_id, "provenance": {
        "host_fingerprint": {"fingerprint": fingerprint,
                             "fingerprint_version": version,
                             "label": f"box-{fingerprint}"}}}


def test_a_fingerprint_recipe_bump_is_not_a_machine_split():
    """Sonnet #3 (High), reproduced 2026-08-21.

    `FINGERPRINT_VERSION` exists precisely so a reader can tell 'different
    machine' from 'we changed how we hash'. The gate ignored it, so bumping the
    recipe would report every single-machine player as split across two machines —
    a false positive on a day when nothing about the hardware changed.
    """
    rep = audit.host_split_players([
        _rec_v("p", "aaa", 1, "t1"),
        _rec_v("p", "bbb", 2, "t2"),
    ])
    assert rep.ok, "a recipe bump must not read as two machines"
    assert "p" in rep.fingerprint_version_skew


def test_two_machines_within_one_recipe_version_still_gate():
    """The version-skew allowance must not become a hole."""
    rep = audit.host_split_players([
        _rec_v("p", "aaa", 1, "t1"),
        _rec_v("p", "bbb", 1, "t2"),
    ])
    assert not rep.ok
