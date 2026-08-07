"""Tests for the zcurve-evidential scorer."""
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


scorer = _load("_zcurve_evidential_scorer", "scorer.py")
generator = _load("_zcurve_evidential_generator_s", "generator.py")


def _first_regime(regime):
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        if gt["regime"] == regime:
            return t, gt
    raise AssertionError(f"no task with regime {regime}")


def test_correct_verdict_scores_one():
    _, gt = _first_regime("evidential")
    out = {"has_evidential_value": True, "confidence": 1.0}
    s = scorer.score(out, gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["correct"] == 1
    assert s["breakdown"]["calibration"] == 1.0
    assert s["findings"] == []


def test_wrong_on_evidential_emits_misclassified_evidential():
    _, gt = _first_regime("evidential")
    out = {"has_evidential_value": False, "confidence": 0.9}
    s = scorer.score(out, gt)
    assert s["primary"] == 0.0
    cats = {f["category"] for f in s["findings"]}
    assert "misclassified_evidential" in cats


def test_wrong_on_non_evidential_emits_misclassified_non_evidential():
    _, gt = _first_regime("non_evidential")
    out = {"has_evidential_value": True, "confidence": 0.9}
    s = scorer.score(out, gt)
    assert s["primary"] == 0.0
    cats = {f["category"] for f in s["findings"]}
    assert "misclassified_non_evidential" in cats


def test_calibration_punishes_confident_wrong():
    _, gt = _first_regime("non_evidential")
    out = {"has_evidential_value": True, "confidence": 1.0}
    s = scorer.score(out, gt)
    assert s["breakdown"]["calibration"] == 0.0


def test_findings_use_object_anchors():
    _, gt = _first_regime("evidential")
    out = {"has_evidential_value": False, "confidence": 0.5}
    s = scorer.score(out, gt)
    assert s["findings"]
    for f in s["findings"]:
        if "anchor" in f:
            assert isinstance(f["anchor"], dict)


def test_perfect_player_scores_one_across_all_tasks():
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        out = {"has_evidential_value": gt["has_evidential_value"], "confidence": 1.0}
        s = scorer.score(out, gt)
        assert s["primary"] == 1.0
        assert 0.0 <= s["primary"] <= 1.0
