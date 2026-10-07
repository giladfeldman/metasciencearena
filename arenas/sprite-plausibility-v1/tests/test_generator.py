"""Tests for the sprite-plausibility generator."""
import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("_sprite_plausibility_generator", ARENA_DIR / "generator.py")
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_sprite_plausibility_generator"] = generator
_SPEC.loader.exec_module(generator)

# Rounding tolerance: a clean stat is computed exactly then rounded to `decimals`,
# so its rounded values may sit just above the true bound by up to half a ULP.
_ROUND_SLACK = 0.51


def test_generate_is_deterministic():
    a = list(generator.generate("v1", seed=3))
    b = list(generator.generate("v1", seed=3))
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert [t["input"]["statistics"] for t in a] == [t["input"]["statistics"] for t in b]


def test_envelopes_have_required_fields_and_tags():
    tasks = list(generator.generate("v1", seed=0, split="revealed"))
    assert tasks
    for t in tasks:
        assert {"task_id", "arena_id", "task_set_version", "difficulty", "input"} <= t.keys()
        assert t["arena_id"] == "sprite-plausibility-v1"
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"
        assert "statistics" in t["input"]
        for s in t["input"]["statistics"]:
            assert {"stat_id", "label", "mean", "sd", "n",
                    "scale_min", "scale_max", "decimals"} <= s.keys()


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_t1_and_t2_have_no_issues():
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] in (1, 2):
            assert t["difficulty"]["n_issues"] == 0
            gt = generator.ground_truth(t["task_id"])
            assert gt["mistake_kinds"] == []
            assert all(not g["flagged"] for g in gt["records"])


def test_ground_truth_returns_records_and_mistake_kinds():
    tasks = list(generator.generate("v1", seed=0))
    gt = generator.ground_truth(tasks[0]["task_id"])
    assert "records" in gt
    assert "mistake_kinds" in gt and isinstance(gt["mistake_kinds"], list)
    assert all({"stat_id", "issue_kind", "flagged"} <= g.keys() for g in gt["records"])


def test_ground_truth_missing_task_raises_keyerror():
    import pytest
    list(generator.generate("v1", seed=0))
    with pytest.raises(KeyError):
        generator.ground_truth("sp-t1-0-sDOES_NOT_EXIST")


def test_revealed_set_covers_every_issue_kind():
    """The public benchmark must exercise the full array of injected issues."""
    all_kinds = set(generator.ISSUE_KINDS)
    seen = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        for g in gt["records"]:
            if g["flagged"]:
                seen.add(g["issue_kind"])
    assert seen == all_kinds


def test_splits_share_difficulty_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert [t["input"]["statistics"] for t in rev] != [t["input"]["statistics"] for t in priv]


def _stat_by_id(task):
    return {s["stat_id"]: s for s in task["input"]["statistics"]}


def test_gold_self_consistency_flagged_stats_violate_their_condition():
    """Every flagged stat must actually violate its stated impossibility condition,
    and every clean stat must satisfy mean in range AND sd^2 <= max_var."""
    seen_kinds = set()
    for seed in (0, 7, 12345):
        for t in generator.generate("v1", seed=seed):
            gt = generator.ground_truth(t["task_id"])
            stats = _stat_by_id(t)
            for g in gt["records"]:
                s = stats[g["stat_id"]]
                lo, hi, n = s["scale_min"], s["scale_max"], s["n"]
                mean, sd = s["mean"], s["sd"]
                max_var = (hi - mean) * (mean - lo)
                # Achievable samples use ddof=1, whose ceiling carries an n/(n-1)
                # Bessel inflation over the population bound.
                sample_max_var = max_var * n / (n - 1) if n > 1 else max_var
                if g["flagged"]:
                    seen_kinds.add(g["issue_kind"])
                    if g["issue_kind"] == "impossible_mean":
                        assert mean < lo or mean > hi, (
                            f"{g['stat_id']} flagged impossible_mean but mean {mean} in [{lo},{hi}]"
                        )
                    elif g["issue_kind"] == "impossible_sd":
                        # mean stays in range; sd^2 strictly exceeds the max variance.
                        assert lo <= mean <= hi
                        assert sd * sd > max_var, (
                            f"{g['stat_id']} flagged impossible_sd but sd^2 {sd*sd} <= max_var {max_var}"
                        )
                    else:
                        raise AssertionError(f"unexpected flagged kind {g['issue_kind']}")
                else:
                    # Clean / control: provably achievable. mean in range (+ rounding
                    # slack) and sd^2 <= max_var (+ small tolerance).
                    assert lo - _ROUND_SLACK <= mean <= hi + _ROUND_SLACK, (
                        f"{g['stat_id']} clean but mean {mean} outside [{lo},{hi}]"
                    )
                    tol = max(0.5, 0.02 * (sample_max_var if sample_max_var > 0 else 1.0))
                    assert sd * sd <= sample_max_var + tol, (
                        f"{g['stat_id']} clean but sd^2 {sd*sd} > sample_max_var {sample_max_var}"
                    )
    assert seen_kinds == set(generator.ISSUE_KINDS)


def test_clean_stats_never_trip_an_impossibility_condition():
    """A dedicated check that controls (T2, extreme-but-possible) are achievable."""
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] != 2:
            continue
        for s in t["input"]["statistics"]:
            lo, hi, n = s["scale_min"], s["scale_max"], s["n"]
            mean, sd = s["mean"], s["sd"]
            max_var = (hi - mean) * (mean - lo)
            sample_max_var = max_var * n / (n - 1) if n > 1 else max_var
            assert lo - _ROUND_SLACK <= mean <= hi + _ROUND_SLACK
            # Extreme controls sit below the achievable (sample) variance ceiling.
            assert sd * sd <= sample_max_var + 0.5
