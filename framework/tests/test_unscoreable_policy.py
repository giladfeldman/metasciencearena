"""What an unscoreable output is worth, and why it is not a flat zero.

WHY THIS TEST EXISTS
--------------------
Owner's decision, 2026-08-19 (TODO A5(b)). An output the arena's schema rejects
— unparseable, or missing a required field — used to be dropped from the mean
entirely. So a model that could not follow the output CONTRACT was *excused*
rather than penalised, and its published score described only the tasks it
happened to format correctly. `0.88 over 32 of 46 tasks` and `0.88 over 46` are
very different claims and looked identical.

A blanket zero is the obvious fix and it is wrong, for a reason visible in the
data rather than in principle: on `replication-target-lookup-v1` **all five**
players fail the same four tasks. Zeroing there publishes "every model is bad at
this" when the honest reading is that those four tasks are broken. So the rest of
the field is used as the control:

  * somebody else scored the task -> the failure is the player's: 0.0, counted.
  * nobody scored the task        -> the task is the suspect: excluded, reported.

Measured impact when introduced: 9 records became 0.0 across two arenas, moving
four players' means by -0.019 to -0.126 (nvidia-nemotron-lightning-power
0.660 -> 0.534 was the largest), and 4 tasks in replication-target-lookup-v1 were
flagged rather than zeroed.
"""
from __future__ import annotations

from framework.report import (
    UNSCOREABLE_ZERO_FLAG,
    _is_errored,
    _is_policy_zero,
    _is_scored,
    apply_unscoreable_policy,
)


def _scored(task: str, player: str, primary: float) -> dict:
    return {"task_id": task, "player_id": player,
            "score": {"primary": primary, "breakdown": {}}}


def _failed(task: str, player: str, reason: str = "output_schema_violation") -> dict:
    return {"task_id": task, "player_id": player,
            "score": {"primary": None, "breakdown": {"error": reason}}}


def test_a_failure_is_zeroed_when_another_player_managed_the_task():
    records, orphans = apply_unscoreable_policy([_scored("t1", "a", 0.9), _failed("t1", "b")])
    b = next(r for r in records if r["player_id"] == "b")
    assert b["score"]["primary"] == 0.0
    assert _is_scored(b), "a policy zero must COUNT — otherwise the policy does nothing"
    assert not _is_errored(b)
    assert _is_policy_zero(b)
    assert orphans == set()


def test_the_reason_survives_under_a_different_key():
    """`error` is renamed, not dropped — and the rename is load-bearing.

    `_is_errored` keys on `breakdown.error`. Leaving it would make the record
    both "scored 0.0" and "errored": counted in n_errored AND dropped from the
    mean by `_is_scored`, so the policy would silently have no effect at all.
    """
    records, _ = apply_unscoreable_policy([_scored("t1", "a", 0.9), _failed("t1", "b", "bad json")])
    bd = next(r for r in records if r["player_id"] == "b")["score"]["breakdown"]
    assert bd["unscoreable_reason"] == "bad json"
    assert "error" not in bd
    assert bd[UNSCOREABLE_ZERO_FLAG] is True


def test_a_task_nobody_could_score_is_reported_not_zeroed():
    """The replication-target-lookup-v1 case: the whole field fails one item."""
    records, orphans = apply_unscoreable_policy(
        [_failed("t2", p) for p in ("a", "b", "c", "d", "e")]
    )
    assert orphans == {"t2"}
    for r in records:
        assert r["score"]["primary"] is None
        assert not _is_policy_zero(r)
        assert _is_errored(r), "it stays an error; only the blame is withheld"


def test_a_real_zero_from_the_scorer_is_not_flagged():
    """The distinction is the entire point of recording the flag."""
    records, _ = apply_unscoreable_policy([_scored("t1", "a", 0.0), _scored("t1", "b", 0.5)])
    a = next(r for r in records if r["player_id"] == "a")
    assert a["score"]["primary"] == 0.0
    assert not _is_policy_zero(a), "answering wrongly is not the same as answering unusably"


def test_the_callers_records_are_never_mutated():
    """Records are read once from disk and shared between callers."""
    import copy
    original = [_scored("t1", "a", 0.9), _failed("t1", "b")]
    snapshot = copy.deepcopy(original)
    apply_unscoreable_policy(original)
    assert original == snapshot


def test_the_policy_is_idempotent():
    once, orph1 = apply_unscoreable_policy([_scored("t1", "a", 0.9), _failed("t1", "b")])
    twice, orph2 = apply_unscoreable_policy(once)
    assert twice == once and orph1 == orph2


def test_records_without_a_task_id_are_left_alone():
    """An unkeyed record must not be judged against a field it is not part of."""
    rec = {"player_id": "a", "score": {"primary": None, "breakdown": {"error": "x"}}}
    records, orphans = apply_unscoreable_policy([rec])
    assert orphans == set()
    assert records[0]["score"]["primary"] is None
