"""Tests for the significance-language scorer."""
import importlib.util
import sys
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, ARENA_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


scorer = _load("_significance_language_scorer", "scorer.py")
generator = _load("_significance_language_generator_s", "generator.py")


def _oracle_output(gt, confidence=1.0):
    """A perfect player: reproduces every gold flag (span + category)."""
    return {"flags": [
        {"span": dict(f["span"]), "category": f["category"], "confidence": confidence}
        for f in gt["flags"]
    ]}


def _first_tier(seed, tier):
    return next(t for t in generator.generate("v1", seed=seed) if t["difficulty"]["tier"] == tier)


def test_perfect_player_scores_one_on_a_mistake_task():
    t3 = _first_tier(0, 3)
    gt = generator.ground_truth(t3["task_id"])
    out = _oracle_output(gt)
    s = scorer.score(out, gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["f1"] == 1.0
    assert s["breakdown"]["category_accuracy"] == 1.0
    assert s["findings"] == []


def test_perfect_player_on_multi_mistake_task_scores_one():
    t5 = _first_tier(0, 5)
    gt = generator.ground_truth(t5["task_id"])
    assert len(gt["flags"]) >= 2  # T5 carries multiple mistakes
    out = _oracle_output(gt)
    s = scorer.score(out, gt)
    assert s["primary"] == 1.0


def test_clean_correct_player_not_penalised_on_t2_trap():
    """The controls-only T2 trap: flagging NOTHING is the correct (perfect) answer."""
    t2 = _first_tier(0, 2)
    gt = generator.ground_truth(t2["task_id"])
    assert gt["flags"] == []
    out = {"flags": []}
    s = scorer.score(out, gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["fp"] == 0
    assert s["findings"] == []


def test_false_alarm_on_t2_trap_is_penalised():
    """Flagging a clean control on T2 is a flag_false_alarm and tanks precision."""
    t2 = _first_tier(0, 2)
    gt = generator.ground_truth(t2["task_id"])
    text = t2["input"]["text"]
    out = {"flags": [
        {"span": {"text": text[5:15], "char_start": 5, "char_end": 15},
         "category": "marginal_significance", "confidence": 0.9}
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fp"] == 1
    assert s["breakdown"]["precision"] < 1.0
    assert any(f["category"] == "flag_false_alarm" for f in s["findings"])


def test_missed_flag_is_penalised():
    t3 = _first_tier(0, 3)
    gt = generator.ground_truth(t3["task_id"])
    out = {"flags": []}  # player flags nothing on a mistake-bearing task
    s = scorer.score(out, gt)
    assert s["breakdown"]["fn"] >= 1
    assert s["breakdown"]["recall"] < 1.0
    assert any(f["category"] == "flag_missed" for f in s["findings"])


def test_category_mislabel_emitted_and_lowers_category_accuracy():
    t3 = _first_tier(0, 3)
    gt = generator.ground_truth(t3["task_id"])
    wrong = "spin_overclaim"
    # Pick a wrong-but-valid category different from the gold's.
    gold_cat = gt["flags"][0]["category"]
    if gold_cat == wrong:
        wrong = "causal_overclaim"
    out = {"flags": [
        {"span": dict(f["span"]), "category": wrong, "confidence": 1.0}
        for f in gt["flags"]
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["f1"] == 1.0           # localisation still perfect
    assert s["breakdown"]["category_accuracy"] < 1.0
    assert any(f["category"] == "category_mislabel" for f in s["findings"])


def test_findings_use_object_anchors():
    """Anchors must be objects or null (findings.schema.json), never strings."""
    t3 = _first_tier(0, 3)
    gt = generator.ground_truth(t3["task_id"])
    out = {"flags": []}  # produces flag_missed findings with anchors
    s = scorer.score(out, gt)
    assert s["findings"]
    for f in s["findings"]:
        if "anchor" in f:
            assert f["anchor"] is None or isinstance(f["anchor"], dict)


def test_each_error_category_can_be_emitted():
    """flag_missed, flag_false_alarm, category_mislabel each appear from a wrong player."""
    t3 = _first_tier(0, 3)
    gt = generator.ground_truth(t3["task_id"])
    text = t3["input"]["text"]
    gold = gt["flags"][0]
    wrong_cat = "spin_overclaim" if gold["category"] != "spin_overclaim" else "causal_overclaim"
    # A player that: mislabels the real flag's category AND adds a false alarm,
    # AND misses (we drop nothing here so use a separate miss scenario below).
    out = {"flags": [
        {"span": dict(gold["span"]), "category": wrong_cat, "confidence": 0.8},
        {"span": {"text": text[0:4], "char_start": 0, "char_end": 4},
         "category": "marginal_significance", "confidence": 0.7},
    ]}
    s = scorer.score(out, gt)
    cats = {f["category"] for f in s["findings"]}
    assert "category_mislabel" in cats
    assert "flag_false_alarm" in cats
    # And a miss when the player outputs nothing.
    s_miss = scorer.score({"flags": []}, gt)
    assert any(f["category"] == "flag_missed" for f in s_miss["findings"])


def test_overconfident_wrong_player_loses_calibration():
    """A player wrong with high confidence should be penalised on calibration."""
    t3 = _first_tier(0, 3)
    gt = generator.ground_truth(t3["task_id"])
    text = t3["input"]["text"]
    # One confident false alarm, no correct flags.
    out = {"flags": [
        {"span": {"text": text[0:5], "char_start": 0, "char_end": 5},
         "category": "marginal_significance", "confidence": 1.0}
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["calibration"] < 1.0
