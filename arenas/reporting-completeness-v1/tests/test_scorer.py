"""Tests for the reporting-completeness scorer."""
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


scorer = _load("_reporting_completeness_scorer", "scorer.py")
generator = _load("_reporting_completeness_generator_s", "generator.py")


def _oracle_output(gt, confidence=1.0):
    """A perfect player: flags every gold defect with the right span + category."""
    return {"flags": [
        {"span": dict(f["span"]), "category": f["category"], "confidence": confidence}
        for f in gt["flags"]
    ]}


def _first_tier(tier, seed=0):
    return next(t for t in generator.generate("v1", seed=seed) if t["difficulty"]["tier"] == tier)


def test_perfect_player_scores_one_on_a_defect_task():
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    assert gt["flags"]  # the task really has a defect
    s = scorer.score(_oracle_output(gt), gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["detection_f1"] == 1.0
    assert s["breakdown"]["fp"] == 0 and s["breakdown"]["fn"] == 0
    assert s["findings"] == []


def test_perfect_player_on_false_alarm_trap_scores_one():
    """T2 is the false-alarm trap: a clean-correct player (no flags) must score 1.0."""
    t2 = _first_tier(2)
    gt = generator.ground_truth(t2["task_id"])
    assert gt["flags"] == []
    s = scorer.score({"flags": []}, gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["fp"] == 0
    assert s["findings"] == []


def test_false_alarm_on_trap_is_penalised():
    """Flagging a complete report in the T2 trap must hurt precision + emit a finding."""
    t2 = _first_tier(2)
    gt = generator.ground_truth(t2["task_id"])
    text = t2["input"]["text"]
    # Player wrongly flags the first 20 chars of a complete report.
    out = {"flags": [{"span": {"text": text[:20], "char_start": 0, "char_end": 20},
                      "category": "missing_ci", "confidence": 0.9}]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fp"] == 1
    assert s["breakdown"]["precision"] < 1.0
    assert any(f["category"] == "flag_false_alarm" for f in s["findings"])


def test_missed_defect_is_penalised():
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    # Player flags nothing.
    s = scorer.score({"flags": []}, gt)
    assert s["breakdown"]["fn"] >= 1
    assert s["breakdown"]["recall"] < 1.0
    assert any(f["category"] == "flag_missed" for f in s["findings"])


def test_category_mislabel_emits_finding_and_lowers_score():
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    real = gt["flags"][0]
    wrong_cat = "imprecise_p" if real["category"] != "imprecise_p" else "missing_ci"
    out = {"flags": [{"span": dict(real["span"]), "category": wrong_cat, "confidence": 1.0}]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["category_mislabels"] == 1
    assert s["breakdown"]["detection_f1"] < 1.0
    assert any(f["category"] == "category_mislabel" for f in s["findings"])


def test_span_overlap_matches_by_containment():
    """A player span that CONTAINS the gold span (right category) is a true positive."""
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    text = t3["input"]["text"]
    g = gt["flags"][0]
    # Widen the span to the enclosing sentence (still contains the gold span).
    start = text.rfind(".", 0, g["span"]["char_start"]) + 1
    end = text.find(".", g["span"]["char_end"])
    end = end + 1 if end >= 0 else len(text)
    out = {"flags": [{"span": {"text": text[start:end], "char_start": start, "char_end": end},
                      "category": g["category"], "confidence": 1.0}]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["tp"] == 1
    assert s["breakdown"]["fp"] == 0


def test_calibration_penalises_overconfident_wrong_flag():
    """High confidence on a false alarm lowers calibration (and the composite)."""
    t1 = _first_tier(1)
    gt = generator.ground_truth(t1["task_id"])  # clean task, no gold flags
    text = t1["input"]["text"]
    out = {"flags": [{"span": {"text": text[:15], "char_start": 0, "char_end": 15},
                      "category": "missing_df", "confidence": 1.0}]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["calibration"] < 1.0
    assert s["primary"] < 1.0


def test_findings_use_object_anchors():
    """Anchors must be objects (findings.schema.json), so redaction strips cleanly."""
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    s = scorer.score({"flags": []}, gt)  # all misses
    assert s["findings"]
    for f in s["findings"]:
        if "anchor" in f and f["anchor"] is not None:
            assert isinstance(f["anchor"], dict)


def test_every_error_category_is_emittable():
    """flag_missed, flag_false_alarm, category_mislabel must each be reachable.

    Uses a T5 task (multiple gold defects) so we can mislabel one, leave another
    unflagged (a miss), and add a span with no gold overlap (a false alarm).
    """
    t5 = _first_tier(5)
    gt = generator.ground_truth(t5["task_id"])
    assert len(gt["flags"]) >= 2, "T5 must carry multiple defects for this test"
    text = t5["input"]["text"]
    real = gt["flags"][0]
    wrong_cat = "imprecise_p" if real["category"] != "imprecise_p" else "missing_ci"
    out = {"flags": [
        # category_mislabel: right span on the FIRST defect, wrong category.
        {"span": dict(real["span"]), "category": wrong_cat, "confidence": 0.8},
        # flag_false_alarm: a span with no overlapping gold defect.
        {"span": {"text": text[-12:], "char_start": len(text) - 12, "char_end": len(text)},
         "category": "missing_df", "confidence": 0.7},
        # (the remaining gold defect(s) go unflagged => flag_missed)
    ]}
    s = scorer.score(out, gt)
    cats = {f["category"] for f in s["findings"]}
    assert "category_mislabel" in cats
    assert "flag_false_alarm" in cats
    assert "flag_missed" in cats
