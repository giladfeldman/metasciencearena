"""Self-consistency + scorer tests for p-curve-v1.

The arena's gold is COMPUTED: every constructed finding-set must, when fed through an
independent p-curve, yield the verdict the construction claims. These tests prove that
— across both splits and several seeds — and that the evidential tier sets are
strongly right-skewed (right_skew_p < .01) while the no-evidential sets are clearly
NOT right-skewed (right_skew_p > .5). They also cover the scorer's
correctness/calibration blend and both error categories.
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


generator = _load("_pcurve_generator", "generator.py")
scorer = _load("_pcurve_scorer", "scorer.py")


# --------------------------------------------------------------------------- #
# Generator structure / determinism.
# --------------------------------------------------------------------------- #
def test_generate_is_deterministic():
    a = list(generator.generate("v1", seed=3))
    b = list(generator.generate("v1", seed=3))
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert [t["input"]["findings"] for t in a] == [t["input"]["findings"] for t in b]


def test_envelopes_have_required_fields_and_tags():
    tasks = list(generator.generate("v1", seed=0, split="revealed"))
    assert tasks
    for t in tasks:
        assert {"task_id", "arena_id", "task_set_version", "difficulty", "input"} <= t.keys()
        assert t["arena_id"] == "p-curve-v1"
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"
        findings = t["input"]["findings"]
        assert len(findings) >= 5  # p-curve needs enough significant findings
        for f in findings:
            assert f["type"] in {"t", "F", "z", "chi2", "r"}
            # Every emitted finding must itself be significant (p < .05).
            assert generator.two_sided_p(f) < generator.SIG_ALPHA
            if f["type"] == "F":
                assert f["df1"] == 1
            if f["type"] == "r":
                assert "n" in f


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
    assert labels == {"evidential", "no-evidential"}


def test_ground_truth_missing_task_raises_keyerror():
    import pytest
    list(generator.generate("v1", seed=0))
    with pytest.raises(KeyError):
        generator.ground_truth("pcurve-tX-9-sDOES_NOT_EXIST")


def test_splits_share_tier_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=24680, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert [t["input"]["findings"] for t in rev] != [t["input"]["findings"] for t in priv]


# --------------------------------------------------------------------------- #
# THE cross-validation invariant: computed gold is self-consistent under an
# independent p-curve, for EVERY task in BOTH splits and several seeds.
# --------------------------------------------------------------------------- #
def test_gold_is_self_consistent_pcurve_matches_construction():
    for split, seed in (("revealed", 0), ("private", 777), ("private", 5), ("private", 42)):
        for t in generator.generate("v1", seed=seed, split=split):
            gt = generator.ground_truth(t["task_id"])
            verdict = generator.pcurve_verdict(t["input"]["findings"])
            assert verdict == gt["evidential_value"], (
                f"{t['task_id']}: constructed label={gt['label']} but p-curve verdict={verdict} "
                f"(right_skew_p={gt['right_skew_p']:.3e})"
            )


def test_evidential_tasks_are_far_below_threshold():
    """Every evidential set must be strongly right-skewed (right_skew_p < .01)."""
    for split, seed in (("revealed", 0), ("private", 5)):
        for t in generator.generate("v1", seed=seed, split=split):
            gt = generator.ground_truth(t["task_id"])
            if gt["label"] == "evidential":
                p = gt["right_skew_p"]
                assert p < 0.01, f"{t['task_id']} evidential but right_skew_p={p:.3e} not clearly significant"


def test_no_evidential_tasks_are_well_above_threshold():
    """Every no-evidential set must be comfortably NOT right-skewed (right_skew_p > .5)."""
    for split, seed in (("revealed", 0), ("private", 5)):
        for t in generator.generate("v1", seed=seed, split=split):
            gt = generator.ground_truth(t["task_id"])
            if gt["label"] == "no-evidential":
                p = gt["right_skew_p"]
                assert p > 0.5, f"{t['task_id']} no-evidential but right_skew_p={p:.3e} too close to threshold"


def test_controls_tier_separates_cleanly():
    """T2 is the controls tier: one clearly-evidential and one clearly-flat set."""
    labels = {}
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] == 2:
            gt = generator.ground_truth(t["task_id"])
            labels[gt["label"]] = gt["right_skew_p"]
    assert set(labels) == {"evidential", "no-evidential"}
    assert labels["evidential"] < 0.01
    assert labels["no-evidential"] > 0.5


def test_each_tier_has_an_evidential_and_a_no_evidential_task():
    by_tier = {}
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        by_tier.setdefault(t["difficulty"]["tier"], set()).add(gt["label"])
    for tier in (1, 2, 3, 4, 5, 6):
        assert by_tier[tier] == {"evidential", "no-evidential"}, f"tier {tier} missing a label"


def test_phacked_tiers_are_left_skewed_not_evidential():
    """T3/T5 carry the p-hacked no-evidential mechanism: left-skewed, never evidential."""
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] in (3, 5):
            gt = generator.ground_truth(t["task_id"])
            if gt["label"] == "no-evidential":
                assert gt["evidential_value"] is False
                assert gt["right_skew_p"] > 0.5


# --------------------------------------------------------------------------- #
# p-value computation sanity (the statistic-type plumbing).
# --------------------------------------------------------------------------- #
def test_two_sided_p_encodings_agree():
    """t, F(1,df), z, chi2(1), r encodings of the same effect give the same p-value."""
    df = 58
    t = 3.0
    p_t = generator.two_sided_p({"type": "t", "value": t, "df1": df})
    p_F = generator.two_sided_p({"type": "F", "value": t * t, "df1": 1, "df2": df})
    assert abs(p_t - p_F) < 1e-6
    # z and chi2 share the standardized scale.
    p_norm = generator.two_sided_p({"type": "z", "value": 1.96})
    assert abs(p_norm - 0.05) < 1e-3
    p_chi = generator.two_sided_p({"type": "chi2", "value": 1.96 ** 2, "df1": 1})
    assert abs(p_norm - p_chi) < 1e-6


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
    g = _gt(1, "evidential")
    out = {"evidential_value": True, "confidence": 1.0}
    s = scorer.score(out, g)
    assert s["primary"] == 1.0
    assert s["breakdown"]["correct"] == 1.0
    assert s["findings"] == []


def test_perfect_on_no_evidential_scores_one():
    g = _gt(2, "no-evidential")
    out = {"evidential_value": False, "confidence": 1.0}
    s = scorer.score(out, g)
    assert s["primary"] == 1.0
    assert s["breakdown"]["correct"] == 1.0


def test_evidential_missed_finding():
    g = _gt(1, "evidential")
    out = {"evidential_value": False, "confidence": 0.9}
    s = scorer.score(out, g)
    assert s["breakdown"]["correct"] == 0.0
    assert any(f["category"] == "evidential_missed" for f in s["findings"])


def test_evidential_false_alarm_finding():
    g = _gt(2, "no-evidential")
    out = {"evidential_value": True, "confidence": 0.9}
    s = scorer.score(out, g)
    assert s["breakdown"]["correct"] == 0.0
    assert any(f["category"] == "evidential_false_alarm" for f in s["findings"])


def test_calibration_penalises_overconfident_wrong():
    g = _gt(1, "evidential")
    confident_wrong = scorer.score({"evidential_value": False, "confidence": 1.0}, g)
    humble_wrong = scorer.score({"evidential_value": False, "confidence": 0.0}, g)
    assert humble_wrong["primary"] > confident_wrong["primary"]


def test_right_skew_p_abs_err_recorded_when_present():
    g = _gt(1, "evidential")
    out = {"evidential_value": True, "confidence": 1.0, "right_skew_p": g["right_skew_p"] + 0.01}
    s = scorer.score(out, g)
    assert "right_skew_p_abs_err" in s["breakdown"]
    assert abs(s["breakdown"]["right_skew_p_abs_err"] - 0.01) < 1e-9
    assert s["breakdown"]["right_skew_p_agree"] is True


def test_primary_in_unit_interval_for_random_outputs():
    g = _gt(1, "evidential")
    for ev in (True, False):
        for c in (0.0, 0.3, 0.7, 1.0):
            s = scorer.score({"evidential_value": ev, "confidence": c}, g)
            assert 0.0 <= s["primary"] <= 1.0
