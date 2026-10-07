"""Ensemble analysis must not be able to manufacture headroom that isn't there.

The published claim this supports is "a combination of these tools does better
than the best one alone". That claim is only worth making if the arithmetic
cannot be inflated by an outage, by variance, or by a player that flags
everything — so those are what these tests pin.
"""
from __future__ import annotations

import json

import pytest

from framework import ensemble


def _write(tmp_path, records):
    arena = tmp_path / "test-arena-v1"
    (arena / "runs" / "v1").mkdir(parents=True, exist_ok=True)
    (arena / "runs" / "v1" / "r.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return arena


def _rec(player, task, primary, *, visibility="public", error=None):
    score = {"primary": primary}
    if error:
        score["breakdown"] = {"error": error}
    return {"player_id": player, "task_id": task, "score": score,
            "task_visibility": visibility}


def test_oracle_is_never_below_the_best_single_player(tmp_path):
    arena = _write(tmp_path, [
        _rec("a", "t1", 1.0), _rec("a", "t2", 0.0),
        _rec("b", "t1", 0.0), _rec("b", "t2", 1.0),
    ])
    res = ensemble.analyse(arena)
    assert res["best_single"]["mean"] == pytest.approx(0.5)
    # Perfectly complementary players: routing each task to the better one is 1.0.
    assert res["oracle_all"] == pytest.approx(1.0)
    assert res["headroom"] == pytest.approx(0.5)


def test_identical_players_produce_no_headroom(tmp_path):
    """Redundancy must show as zero gain, not as a spurious lift."""
    arena = _write(tmp_path, [
        _rec("a", "t1", 0.6), _rec("a", "t2", 0.4),
        _rec("b", "t1", 0.6), _rec("b", "t2", 0.4),
    ])
    res = ensemble.analyse(arena)
    assert res["headroom"] == pytest.approx(0.0)
    assert res["saturates_at_k"] == 1, "the second player adds nothing and must show it"


def test_errored_records_are_excluded_not_scored_as_zero(tmp_path):
    """An outage is an absence of evidence, not evidence of incapability.

    Counting a 429 or a crashed adapter as 0.0 would let infrastructure failure
    depress a player's single-score while leaving the oracle untouched, silently
    manufacturing headroom.
    """
    arena = _write(tmp_path, [
        _rec("a", "t1", 1.0), _rec("a", "t2", 1.0),
        _rec("b", "t1", 1.0), _rec("b", "t2", 0.0, error="HTTP 429"),
    ])
    scores = ensemble.load_public_scores(arena)
    assert "t2" not in scores["b"], "errored record must not enter the score table"
    # b is only comparable on t1, so the common task set is {t1} and there is no
    # headroom to claim.
    res = ensemble.analyse(arena)
    assert res["n_common_tasks"] == 1
    assert res["headroom"] == pytest.approx(0.0)


def test_null_primary_is_excluded(tmp_path):
    """`primary: null` means "excluded, unverifiable" in the contract."""
    arena = _write(tmp_path, [_rec("a", "t1", None), _rec("a", "t2", 1.0)])
    scores = ensemble.load_public_scores(arena)
    assert set(scores["a"]) == {"t2"}


def test_held_out_records_are_ignored(tmp_path):
    """Held-out records are redacted at write time; they carry nothing to ensemble."""
    arena = _write(tmp_path, [
        _rec("a", "t1", 1.0),
        _rec("a", "h1", 1.0, visibility="held_out"),
    ])
    scores = ensemble.load_public_scores(arena)
    assert set(scores["a"]) == {"t1"}


def test_trials_are_averaged_so_variance_cannot_buy_membership(tmp_path):
    """A player that sometimes gets lucky must be scored on its mean, not its max.

    Otherwise a high-variance player would join every greedy ensemble on the
    strength of its best trial, and the oracle would describe a run that never
    happened.
    """
    arena = _write(tmp_path, [
        _rec("lucky", "t1", 1.0), _rec("lucky", "t1", 0.0),  # mean 0.5
        _rec("steady", "t1", 0.75),
    ])
    scores = ensemble.load_public_scores(arena)
    assert scores["lucky"]["t1"] == pytest.approx(0.5)
    res = ensemble.analyse(arena)
    assert res["best_single"]["player_id"] == "steady"
    assert res["oracle_all"] == pytest.approx(0.75), "the lucky trial must not win the task"


def test_a_player_that_is_never_best_never_joins_the_greedy_ensemble(tmp_path):
    """The anti-spam property, stated as a test.

    A flag-everything player is punished by the arena's own primary metric, so it
    is never the max on any task and therefore contributes zero marginal gain.
    """
    arena = _write(tmp_path, [
        _rec("good", "t1", 0.9), _rec("good", "t2", 0.8),
        _rec("spam", "t1", 0.0), _rec("spam", "t2", 0.0),
    ])
    res = ensemble.analyse(arena)
    gains = {row["player_id"]: row["marginal_gain"] for row in res["greedy_curve"]}
    assert gains["good"] > 0
    assert gains["spam"] == pytest.approx(0.0)
    assert res["greedy_curve"][0]["player_id"] == "good"


def test_greedy_curve_is_monotonic(tmp_path):
    """Adding a player can never lower the oracle — a decrease means a bug."""
    arena = _write(tmp_path, [
        _rec("a", "t1", 0.2), _rec("a", "t2", 0.9),
        _rec("b", "t1", 0.7), _rec("b", "t2", 0.1),
        _rec("c", "t1", 0.5), _rec("c", "t2", 0.5),
    ])
    res = ensemble.analyse(arena)
    means = [row["oracle_mean"] for row in res["greedy_curve"]]
    assert means == sorted(means), f"greedy curve must not decrease: {means}"
    assert all(row["marginal_gain"] >= -1e-9 for row in res["greedy_curve"])


def test_archived_runs_are_not_counted(tmp_path):
    """`_archive/` and `_pilot_archive/` hold superseded evidence, not coverage."""
    arena = _write(tmp_path, [_rec("a", "t1", 1.0)])
    arch = arena / "runs" / "v1" / "_pilot_archive" / "2026-08-12"
    arch.mkdir(parents=True)
    (arch / "old.jsonl").write_text(json.dumps(_rec("ghost", "t1", 1.0)) + "\n", encoding="utf-8")
    scores = ensemble.load_public_scores(arena)
    assert "ghost" not in scores
