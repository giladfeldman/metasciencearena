"""Repeat trials of one task are ONE observation, not N.

Found 2026-08-19 while adding the Docling players. `framework/runner.py:268`
reads `n_trials = 1 if entry["deterministic"] else max(1, trials)`, and
`--trials` defaults to 3, so any player declared `deterministic: false` writes
three run RECORDS per task. `_summary` then computed the confidence interval as
`1.96 * sd / sqrt(n)` over those raw records.

The consequence is pure spurious precision. For a genuinely deterministic tool
the three trials are byte-identical, so the standard deviation is unchanged while
`n` triples -- the published interval comes out **sqrt(3) times too narrow**, and
`n_scored` reports a sample three times larger than the number of tasks actually
measured. Nothing crashes and no other test fails; the board simply looks more
certain than the measurement is. That is precisely the "a silently wrong value
reaches a user-visible artifact with no test failing" trigger from the portfolio
cross-model-review rule.

This was caught before it reached the board (the Docling players were re-run at
one trial after their determinism was measured), and no other player had repeated
task_ids at the time. The defect was latent, not live -- these tests keep it that
way.
"""
from __future__ import annotations

import math

from framework.report import _summary


def _rec(task_id: str, primary: float, player: str = "p") -> dict:
    return {
        "player_id": player,
        "player_version": "v1",
        "task_id": task_id,
        "latency_ms": 10,
        "score": {"primary": primary, "breakdown": {}},
    }


def _single_and_triple(values: dict[str, float]):
    single = [_rec(t, v) for t, v in values.items()]
    triple = [_rec(t, v) for t, v in values.items() for _ in range(3)]
    key = ("p", "v1")
    return _summary(single, single, key), _summary(triple, triple, key)


def test_identical_trials_do_not_narrow_the_confidence_interval():
    """Three byte-identical trials must produce the one-trial interval."""
    one, three = _single_and_triple({"t1": 0.2, "t2": 0.6, "t3": 1.0, "t4": 0.4})

    assert three["primary_mean"] == one["primary_mean"]
    assert math.isclose(three["primary_ci_half"], one["primary_ci_half"], rel_tol=1e-12), (
        "repeat trials of the same task were counted as independent observations: "
        f"3-trial CI {three['primary_ci_half']!r} vs 1-trial {one['primary_ci_half']!r} "
        f"(ratio {one['primary_ci_half'] / max(three['primary_ci_half'], 1e-12):.3f}, "
        "expected 1.0 — sqrt(3) ≈ 1.732 means the raw-record bug is back)"
    )


def test_n_scored_counts_tasks_measured_not_records_written():
    """`n_scored` is read as a sample size, so it must be one per task."""
    one, three = _single_and_triple({"t1": 0.2, "t2": 0.6, "t3": 1.0, "t4": 0.4})
    assert one["n_scored"] == 4
    assert three["n_scored"] == 4, (
        "n_scored reported run records rather than tasks, advertising a sample "
        "three times larger than what was measured"
    )


def test_varying_trials_average_within_a_task_before_the_mean():
    """A genuinely non-deterministic player collapses to its per-task mean.

    The per-task mean is the observation; the spread BETWEEN tasks is what the
    interval describes. Within-task jitter must not shrink it.
    """
    recs = [
        _rec("t1", 0.0), _rec("t1", 0.5), _rec("t1", 1.0),   # per-task mean 0.5
        _rec("t2", 0.4), _rec("t2", 0.5), _rec("t2", 0.6),   # per-task mean 0.5
    ]
    got = _summary(recs, recs, ("p", "v1"))
    assert math.isclose(got["primary_mean"], 0.5, rel_tol=1e-12)
    assert got["n_scored"] == 2
    # Both tasks have the same per-task mean, so there is no between-task spread.
    assert math.isclose(got["primary_ci_half"], 0.0, abs_tol=1e-12)


def test_ranking_also_uses_task_collapsed_means():
    """Rank must not shift because one player ran more trials than another."""
    a = [_rec("t1", 1.0, "a"), _rec("t2", 0.0, "a")]
    b = [_rec("t1", 0.9, "b") for _ in range(3)] + [_rec("t2", 0.1, "b") for _ in range(3)]
    allr = a + b
    sa = _summary(a, allr, ("a", "v1"))
    sb = _summary(b, allr, ("b", "v1"))
    assert math.isclose(sa["primary_mean"], 0.5, rel_tol=1e-12)
    assert math.isclose(sb["primary_mean"], 0.5, rel_tol=1e-12)
    assert sa["n_competitors"] == 2 and sb["n_competitors"] == 2
