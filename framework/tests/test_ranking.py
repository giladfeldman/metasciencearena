"""Calibration and behaviour tests for the paired-significance ranking.

A significance test is the kind of code that fails SILENTLY: a subtly wrong
p-value produces a plausible ranking, no exception, no red test. So this suite
does not check that the functions run — it checks the numbers against simulated
data whose answer is known by construction:

  * under a true null, the false-positive rate must land near alpha;
  * a real-but-tiny consistent advantage must be DETECTED even when the
    published error bars overlap heavily (the docpluck-vs-pdftotext shape);
  * Holm must match a hand-computed example;
  * the same input must produce the same ranks on every run.

Written 2026-08-15 alongside `framework/ranking.py`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from framework import ranking


def _rng(seed: int = 12345) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# Calibration: the property that makes a published p-value mean anything
# --------------------------------------------------------------------------

def test_false_positive_rate_matches_alpha_under_a_true_null():
    """Two players drawn from the SAME distribution must rarely be separated.

    This is the assertion that would catch a test that is simply too eager —
    the failure mode that publishes "tool A beats tool B" about real software
    when the data says nothing of the kind. 400 simulated arenas of 30 paired
    tasks each; at alpha = 0.05 the expected count of false positives is 20.

    The band is 3sd of Binomial(400, 0.05) (sd ~= 4.4), so a correctly
    calibrated test passes ~99.7% of the time and a test that is off by more
    than about half is caught. Deliberately not tighter: a flaky calibration
    test gets deleted, and then nothing checks calibration at all.
    """
    rng = _rng(20260815)
    n_sims, n_tasks, alpha = 400, 30, 0.05
    false_positives = 0
    for _ in range(n_sims):
        a = rng.normal(0.7, 0.2, n_tasks)
        b = rng.normal(0.7, 0.2, n_tasks)      # identical distribution
        p = ranking.paired_signflip_p(a - b, rng=rng, n_resamples=2000)
        false_positives += int(p < alpha)

    expected = n_sims * alpha
    sd = math.sqrt(n_sims * alpha * (1 - alpha))
    assert abs(false_positives - expected) <= 3 * sd, (
        f"false-positive rate {false_positives}/{n_sims} = "
        f"{false_positives / n_sims:.3f} is far from alpha={alpha}. The test is "
        f"{'too eager — it would publish differences that are not there'
           if false_positives > expected else 'too conservative — it would merge real differences'}."
    )


def test_a_tiny_but_consistent_advantage_is_detected_despite_overlapping_bars():
    """The case that rules out the "overlapping CIs means tied" shortcut.

    Modelled on the real one: `pdf-text-fidelity-v1`, docpluck-standard vs
    pdftotext-raw differ by 0.027 in mean with heavily overlapping 95% bars,
    yet docpluck wins on nearly every individual task (paired t = 11.3).

    Here: task difficulty varies a lot (sd 0.20) while player A is better than B
    by a steady 0.03 on every task. The across-task error bars overlap almost
    completely — and the paired test sees it immediately, because the shared
    task-difficulty term cancels.
    """
    rng = _rng(7)
    n = 40
    difficulty = rng.normal(0.70, 0.20, n)     # the shared, cancelling term
    a = difficulty + 0.03 + rng.normal(0, 0.01, n)
    b = difficulty + rng.normal(0, 0.01, n)

    half_a = 1.96 * a.std(ddof=1) / math.sqrt(n)
    half_b = 1.96 * b.std(ddof=1) / math.sqrt(n)
    assert abs(a.mean() - b.mean()) < half_a + half_b, (
        "the simulated bars do NOT overlap, so this test is not exercising the "
        "case it was written for"
    )

    p = ranking.paired_signflip_p(a - b, rng=rng, n_resamples=10_000)
    assert p < 0.001, (
        f"p={p:.4f}: a 0.03 advantage present on every one of {n} tasks was not "
        f"detected. An overlap-based rule would have called this a tie, and it "
        f"is the single most common real shape in this benchmark."
    )


def test_identical_players_are_never_separated():
    """The degenerate case that produced a NaN in the first implementation."""
    rng = _rng(3)
    d = np.zeros(25)
    assert ranking.paired_signflip_p(d, rng=rng, n_resamples=1000) == 1.0


def test_a_perfect_separation_yields_the_smallest_possible_p():
    """Every task favours A by the same amount: sd(d) == 0, mean(d) != 0.

    The studentized statistic is infinite here. It must not become NaN — the
    first version computed `np.sign(mean) * np.inf` on every row including the
    zero rows, and `0 * inf` is NaN, which compares False against everything and
    silently deflates the count.
    """
    rng = _rng(4)
    d = np.full(30, 0.05)
    p = ranking.paired_signflip_p(d, rng=rng, n_resamples=1000)
    assert p == pytest.approx(1 / 1001, rel=1e-6), (
        f"p={p} — a perfectly consistent difference should be as significant as "
        f"the resampling resolution allows"
    )


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------

def test_holm_matches_a_hand_computed_example():
    """m=4; adjusted_i = max over j<=i of (m-j)*p_(j), then made monotone."""
    p = [0.01, 0.02, 0.03, 0.04]
    #   sorted: 0.01*4=0.04 | 0.02*3=0.06 | 0.03*2=0.06 | 0.04*1=0.04 -> 0.06
    assert ranking.holm(p) == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_preserves_input_order_and_caps_at_one():
    # Sorted, the smallest p is multiplied by m=2 (0.001 -> 0.002); the largest
    # is multiplied by 1 and so is unchanged at 0.9. Returned in INPUT order.
    assert ranking.holm([0.9, 0.001]) == pytest.approx([0.9, 0.002])
    # The cap only bites when a multiplied value exceeds 1.
    assert ranking.holm([0.6, 0.5, 0.4]) == pytest.approx([1.0, 1.0, 1.0])
    assert ranking.holm([]) == []


def test_holm_is_what_prevents_a_false_separation_in_a_big_field():
    """With 10 players there are 45 pairs; uncorrected, ~2 look significant by
    chance alone at alpha=0.05. That is a published claim about a real tool."""
    raw = [0.04] + [0.6] * 44
    assert raw[0] < 0.05
    assert ranking.holm(raw)[0] > 0.05, (
        "a single p=0.04 among 45 comparisons survived correction — the field "
        "would publish a separation that is a multiplicity artifact"
    )


# --------------------------------------------------------------------------
# Rank assignment
# --------------------------------------------------------------------------

def _arena(scores: dict[str, list[float]], *, arena="test-v1", n=None):
    tasks = [f"t{i}" for i in range(len(next(iter(scores.values()))))]
    return ranking.rank_players(
        arena_id=arena, task_set_version="v1",
        scores={p: dict(zip(tasks, v)) for p, v in scores.items()},
        ranked=sorted(scores), common=tasks,
        n_resamples=n or 4000,
    )


def test_players_that_cannot_be_separated_share_a_rank():
    rng = _rng(11)
    base = rng.normal(0.7, 0.15, 30)
    out = _arena({
        "a": list(base + rng.normal(0, 0.02, 30)),
        "b": list(base + rng.normal(0, 0.02, 30)),
        "c": list(base + rng.normal(0, 0.02, 30)),
    })
    assert out["tested"] is True
    assert {r["rank"] for r in out["players"]} == {1}, (
        f"three indistinguishable players got ranks "
        f"{[r['rank'] for r in out['players']]} — the board would be asserting an "
        f"order the data does not support"
    )
    assert out["tie_groups"] == [["a", "b", "c"]]


def test_rank_is_one_plus_the_number_who_beat_you_and_needs_no_transitivity():
    """The reason adjacent-merge BANDING was rejected by two reviewers.

    Constructed so that A > C decisively, while B sits between them and is
    separated from neither. Banding would put A, B, C in one band and hide a
    real difference. Counting who beats you gives A rank 1, B rank 1 (nobody
    significantly beats B), C rank 2 — and C's `beaten_by` names A explicitly.
    """
    rng = _rng(5)
    difficulty = rng.normal(0.5, 0.10, 40)
    out = _arena({
        "a": list(difficulty + 0.30 + rng.normal(0, 0.01, 40)),
        "b": list(difficulty + 0.15 + rng.normal(0, 0.30, 40)),   # noisy middle
        "c": list(difficulty + rng.normal(0, 0.01, 40)),
    })
    by_id = {r["player_id"]: r for r in out["players"]}
    assert "a" in by_id["c"]["beaten_by"], "A>C was not detected at all"
    assert by_id["c"]["rank"] > by_id["a"]["rank"], (
        "C did not rank below A despite being significantly beaten by it — a "
        "banding rule would produce exactly this, and it hides a real result"
    )


def test_a_small_task_set_is_published_as_untested_not_as_a_tie():
    """Absence of power is not evidence of equivalence.

    Below MIN_TASKS_FOR_TESTING a paired test returns "not significant" for
    everything. Calling that a tie would assert equivalence the data cannot
    support, so the ordering is published with `tested: false` and a reason the
    UI can show.
    """
    out = _arena({"a": [1.0] * 8, "b": [0.0] * 8})
    assert out["tested"] is False
    assert out["method"] == "mean_order_untested"
    assert "at least" in out["untested_reason"]
    assert [r["player_id"] for r in out["players"]] == ["a", "b"]
    assert [r["rank"] for r in out["players"]] == [1, 2]
    assert all(r["tied_with"] == [] for r in out["players"])


def test_the_rank_interval_is_not_inverted():
    """The best player's bootstrap rank interval must contain 1, not N.

    The first implementation had the two broadcast operands the wrong way round
    and reported the best of six players as rank interval [6, 6] on real
    `pdf-text-fidelity-v1` data. Every other field looked right.
    """
    rng = _rng(9)
    difficulty = rng.normal(0.5, 0.05, 40)
    out = _arena({
        "best":   list(difficulty + 0.40 + rng.normal(0, 0.01, 40)),
        "middle": list(difficulty + 0.20 + rng.normal(0, 0.01, 40)),
        "worst":  list(difficulty + rng.normal(0, 0.01, 40)),
    })
    by_id = {r["player_id"]: r for r in out["players"]}
    assert by_id["best"]["rank_ci_low"] == 1 and by_id["best"]["rank_ci_high"] == 1
    assert by_id["worst"]["rank_ci_low"] == 3 and by_id["worst"]["rank_ci_high"] == 3


def test_ranks_are_reproducible_across_runs():
    """An unseeded permutation test publishes rank 2 today and 3 tomorrow."""
    rng = _rng(2)
    payload = {
        "a": list(rng.normal(0.7, 0.1, 30)),
        "b": list(rng.normal(0.68, 0.1, 30)),
        "c": list(rng.normal(0.5, 0.1, 30)),
    }
    first, second = _arena(payload), _arena(payload)
    assert first == second, "identical input produced a different ranking"
    assert first["seed"] == ranking.seed_for("test-v1", "v1")


def test_different_arenas_do_not_share_a_seed():
    assert ranking.seed_for("a-v1", "v1") != ranking.seed_for("b-v1", "v1")
    assert ranking.seed_for("a-v1", "v1") != ranking.seed_for("a-v1", "v2")


def test_a_ranked_player_missing_a_common_task_is_refused_not_nan_propagated():
    """`assertRankedSymmetry` should make this impossible; if it ever slips
    through, a NaN must not reach a published rank."""
    with pytest.raises(ValueError, match="missing common-set scores"):
        ranking.rank_players(
            arena_id="x-v1", task_set_version="v1",
            scores={"a": {"t0": 1.0, "t1": 1.0}, "b": {"t0": 0.5}},
            ranked=["a", "b"], common=["t0", "t1"],
        )
