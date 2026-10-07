"""Tests for the power-reporting scorer."""
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


scorer = _load("_power_reporting_scorer", "scorer.py")
generator = _load("_power_reporting_generator_s", "generator.py")


def _oracle_output(gt, confidence=1.0):
    """A perfect player: matches detection, kind, and every reported field."""
    return {
        "has_power_analysis": gt["has_power_analysis"],
        "kind": gt["kind"],
        "fields": dict(gt["fields"]),
        "confidence": confidence,
    }


def _first(tier, *, mistake=None, clean=False):
    """Find the first task at `tier` with the given mistake (or a clean one)."""
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] != tier:
            continue
        gt = generator.ground_truth(t["task_id"])
        if clean and not gt["mistake_kinds"]:
            return t, gt
        if mistake is not None and mistake in gt["mistake_kinds"]:
            return t, gt
        if mistake is None and not clean:
            return t, gt
    raise AssertionError(f"no task at tier {tier} (mistake={mistake}, clean={clean})")


def test_perfect_player_scores_one_on_a_mistake_bearing_task():
    # T4 headline trap: a post-hoc analysis worded as a-priori. Oracle labels it
    # posthoc (the true kind) and extracts every field.
    _, gt = _first(4, mistake="posthoc_as_apriori")
    out = _oracle_output(gt)
    s = scorer.score(out, gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["detection"] == 1.0
    assert s["breakdown"]["kind"] == 1.0
    assert s["breakdown"]["field_f1"] == 1.0
    assert s["findings"] == []


def test_perfect_player_on_false_alarm_trap_scores_one():
    # T2 includes a no-power-analysis excerpt; a clean-correct player must score 1.
    _, gt = _first(2, mistake="no_power_analysis")
    out = _oracle_output(gt)  # has_power_analysis False, no fields
    s = scorer.score(out, gt)
    assert s["primary"] == 1.0
    assert not any(f["category"] == "power_false_alarm" for f in s["findings"])


def test_false_alarm_on_no_power_excerpt_is_penalised():
    _, gt = _first(2, mistake="no_power_analysis")
    # Player wrongly claims a power analysis is present.
    out = {"has_power_analysis": True, "kind": "apriori",
           "fields": {"alpha": ".05"}, "confidence": 0.95}
    s = scorer.score(out, gt)
    assert s["breakdown"]["detection"] == 0.0
    assert any(f["category"] == "power_false_alarm" for f in s["findings"])
    assert s["primary"] < 1.0


def test_missed_power_analysis_is_penalised():
    _, gt = _first(1, clean=True)  # a real, complete power analysis
    out = {"has_power_analysis": False, "kind": None, "fields": {}, "confidence": 0.9}
    s = scorer.score(out, gt)
    assert s["breakdown"]["detection"] == 0.0
    assert any(f["category"] == "power_missed" for f in s["findings"])


def test_kind_mislabel_is_penalised_on_the_trap():
    # The headline failure: calling the post-hoc-as-a-priori excerpt "apriori".
    _, gt = _first(4, mistake="posthoc_as_apriori")
    out = _oracle_output(gt)
    out["kind"] = "apriori"  # WRONG — true kind is posthoc
    s = scorer.score(out, gt)
    assert s["breakdown"]["detection"] == 1.0
    assert s["breakdown"]["kind"] == 0.0
    assert any(f["category"] == "kind_mislabel" for f in s["findings"])
    assert s["primary"] < 1.0


def test_field_wrong_is_emitted_and_lowers_field_f1():
    _, gt = _first(1, clean=True)
    out = _oracle_output(gt)
    # Corrupt one field's value.
    a_field = next(iter(gt["fields"]))
    out["fields"][a_field] = "TOTALLY WRONG VALUE"
    s = scorer.score(out, gt)
    assert s["breakdown"]["field_f1"] < 1.0
    assert any(f["category"] == "field_wrong" for f in s["findings"])


def test_hallucinated_field_is_field_wrong():
    # On a no-power excerpt (no gold fields), inventing a field is field_wrong.
    _, gt = _first(2, mistake="no_power_analysis")
    out = {"has_power_analysis": False, "kind": None,
           "fields": {"alpha": ".05"}, "confidence": 0.5}
    s = scorer.score(out, gt)
    assert any(f["category"] == "field_wrong" for f in s["findings"])


def test_calibration_punishes_confident_wrong_detection():
    _, gt = _first(1, clean=True)
    # Confidently wrong on detection -> calibration drops, composite drops.
    confident_wrong = {"has_power_analysis": False, "kind": None, "fields": {}, "confidence": 1.0}
    s = scorer.score(confident_wrong, gt)
    assert s["breakdown"]["calibration"] < 1.0


def test_findings_use_object_anchors():
    """Anchors must be objects (findings.schema.json), so redaction strips cleanly."""
    _, gt = _first(1, clean=True)
    out = {"has_power_analysis": False, "kind": None, "fields": {}, "confidence": 0.8}
    s = scorer.score(out, gt)
    assert s["findings"]
    for f in s["findings"]:
        if "anchor" in f:
            assert isinstance(f["anchor"], dict)


def test_primary_in_unit_interval_across_tiers():
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        s = scorer.score(_oracle_output(gt), gt)
        assert 0.0 <= s["primary"] <= 1.0
        assert s["primary"] == 1.0  # oracle is perfect everywhere
