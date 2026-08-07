"""Self-consistency + scorer tests for publication-bias-v1.

The arena's gold is COMPUTED: every constructed dataset must, when fed through an
independent Egger test, yield the verdict the construction claims. These tests prove
that — across both splits — and that the controls tier is never (constructively)
flagged and every biased tier contains a strongly-asymmetric set. They also cover the
scorer's correctness/calibration blend and both error categories.
"""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, ARENA_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


generator = _load("_pubbias_generator", "generator.py")
scorer = _load("_pubbias_scorer", "scorer.py")


# --------------------------------------------------------------------------- #
# Generator structure / determinism.
# --------------------------------------------------------------------------- #
def test_generate_is_deterministic():
    a = list(generator.generate("v1", seed=3))
    b = list(generator.generate("v1", seed=3))
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert [t["input"]["studies"] for t in a] == [t["input"]["studies"] for t in b]


def test_envelopes_have_required_fields_and_tags():
    tasks = list(generator.generate("v1", seed=0, split="revealed"))
    assert tasks
    for t in tasks:
        assert {"task_id", "arena_id", "task_set_version", "difficulty", "input"} <= t.keys()
        assert t["arena_id"] == "publication-bias-v1"
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"
        studies = t["input"]["studies"]
        assert len(studies) >= 10  # Egger needs enough studies
        assert t["input"]["k"] == len(studies)
        for s in studies:
            assert set(s.keys()) == {"yi", "sei"}
            assert s["sei"] > 0


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_both_labels_present_in_revealed():
    labels = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        labels.add(generator.ground_truth(t["task_id"])["label"])
    assert labels == {"unbiased", "biased"}


def test_ground_truth_missing_task_raises_keyerror():
    import pytest
    list(generator.generate("v1", seed=0))
    with pytest.raises(KeyError):
        generator.ground_truth("pubbias-tX-9-sDOES_NOT_EXIST")


def test_splits_share_tier_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert [t["input"]["studies"] for t in rev] != [t["input"]["studies"] for t in priv]


# --------------------------------------------------------------------------- #
# THE cross-validation invariant: computed gold is self-consistent under an
# independent Egger test, for EVERY task in BOTH splits and several seeds.
# --------------------------------------------------------------------------- #
def test_gold_is_self_consistent_egger_matches_construction():
    for split, seed in (("revealed", 0), ("private", 777), ("private", 5), ("private", 42)):
        for t in generator.generate("v1", seed=seed, split=split):
            gt = generator.ground_truth(t["task_id"])
            studies = t["input"]["studies"]
            yi = [s["yi"] for s in studies]
            sei = [s["sei"] for s in studies]
            verdict = generator.egger_verdict(yi, sei)
            assert verdict == gt["bias_detected"], (
                f"{t['task_id']}: constructed label={gt['label']} but Egger verdict={verdict} "
                f"(egger_p={generator.egger_p(yi, sei):.3e})"
            )


def test_biased_tasks_are_far_below_threshold():
    """Every biased dataset must be clearly asymmetric (Egger p < 0.05), not borderline."""
    for split, seed in (("revealed", 0), ("private", 5)):
        for t in generator.generate("v1", seed=seed, split=split):
            gt = generator.ground_truth(t["task_id"])
            if gt["label"] == "biased":
                p = gt["egger_p"]
                assert p < 0.05, f"{t['task_id']} biased but Egger p={p:.3e} not clearly significant"


def test_clean_tasks_are_well_above_threshold():
    """Every unbiased dataset must be comfortably symmetric (Egger p > 0.15)."""
    for split, seed in (("revealed", 0), ("private", 5)):
        for t in generator.generate("v1", seed=seed, split=split):
            gt = generator.ground_truth(t["task_id"])
            if gt["label"] == "unbiased":
                p = gt["egger_p"]
                assert p > 0.15, f"{t['task_id']} clean but Egger p={p:.3e} too close to threshold"


def test_controls_tier_is_never_flagged():
    """T2 is the false-alarm trap: heterogeneous-but-clean, must all be unbiased."""
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] == 2:
            gt = generator.ground_truth(t["task_id"])
            assert gt["bias_detected"] is False
            assert gt["label"] == "unbiased"


def test_each_biased_tier_has_a_flagged_task():
    """Tiers that carry a biased dataset (1,3,4,5,6) each contain >=1 biased task."""
    biased_by_tier = {}
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        biased_by_tier.setdefault(t["difficulty"]["tier"], []).append(gt["bias_detected"])
    for tier in (1, 3, 4, 5, 6):
        assert any(biased_by_tier[tier]), f"tier {tier} has no biased task"


# --------------------------------------------------------------------------- #
# Scorer.
# --------------------------------------------------------------------------- #
def _gt(tier, label):
    for t in generator.generate("v1", seed=0):
        g = generator.ground_truth(t["task_id"])
        if t["difficulty"]["tier"] == tier and g["label"] == label:
            return g
    raise AssertionError(f"no task with tier={tier} label={label}")


def test_perfect_confident_player_scores_one():
    g = _gt(3, "biased")
    out = {"bias_detected": True, "confidence": 1.0}
    s = scorer.score(out, g)
    assert s["primary"] == 1.0
    assert s["breakdown"]["correct"] == 1.0
    assert s["findings"] == []


def test_perfect_on_clean_scores_one():
    g = _gt(2, "unbiased")
    out = {"bias_detected": False, "confidence": 1.0}
    s = scorer.score(out, g)
    assert s["primary"] == 1.0
    assert s["breakdown"]["correct"] == 1.0


def test_bias_missed_finding():
    g = _gt(3, "biased")
    out = {"bias_detected": False, "confidence": 0.9}
    s = scorer.score(out, g)
    assert s["breakdown"]["correct"] == 0.0
    assert any(f["category"] == "bias_missed" for f in s["findings"])


def test_bias_false_alarm_finding():
    g = _gt(2, "unbiased")
    out = {"bias_detected": True, "confidence": 0.9}
    s = scorer.score(out, g)
    assert s["breakdown"]["correct"] == 0.0
    assert any(f["category"] == "bias_false_alarm" for f in s["findings"])


def test_calibration_penalises_overconfident_wrong():
    g = _gt(3, "biased")
    confident_wrong = scorer.score({"bias_detected": False, "confidence": 1.0}, g)
    humble_wrong = scorer.score({"bias_detected": False, "confidence": 0.0}, g)
    # Both wrong (correct=0) but the humble-when-wrong player calibrates better.
    assert humble_wrong["primary"] > confident_wrong["primary"]


def test_egger_p_abs_err_recorded_when_present():
    g = _gt(3, "biased")
    out = {"bias_detected": True, "confidence": 1.0, "egger_p": g["egger_p"] + 0.01}
    s = scorer.score(out, g)
    assert "egger_p_abs_err" in s["breakdown"]
    assert abs(s["breakdown"]["egger_p_abs_err"] - 0.01) < 1e-9


def test_primary_in_unit_interval_for_random_outputs():
    g = _gt(1, "biased")
    for bd in (True, False):
        for c in (0.0, 0.3, 0.7, 1.0):
            s = scorer.score({"bias_detected": bd, "confidence": c}, g)
            assert 0.0 <= s["primary"] <= 1.0
