"""Tests for the prereg-deviation scorer."""
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


scorer = _load("_prereg_deviation_scorer", "scorer.py")
generator = _load("_prereg_deviation_generator_s", "generator.py")


def _oracle_output(gt, confidence=1.0):
    """A perfect player: matches every gold dimension's deviation + kind."""
    return {"deviations": [
        {"dimension": g["dimension"], "deviation": g["deviation"],
         "deviation_kind": g["deviation_kind"], "confidence": confidence}
        for g in gt["dimensions"]
    ]}


def test_perfect_player_scores_one_on_a_deviation_task():
    # Use a T3 task (one real deviation) so detection metrics are non-trivial.
    t3 = next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 3)
    gt = generator.ground_truth(t3["task_id"])
    out = _oracle_output(gt)
    s = scorer.score(out, gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["detection_f1"] == 1.0
    assert s["breakdown"]["kind_accuracy"] == 1.0
    assert s["findings"] == []


def test_perfect_player_on_paraphrase_trap_scores_one():
    t2 = next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 2)
    gt = generator.ground_truth(t2["task_id"])
    out = _oracle_output(gt)  # all deviation=False
    s = scorer.score(out, gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["fp"] == 0


def test_false_alarm_on_paraphrase_is_penalised():
    t2 = next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 2)
    gt = generator.ground_truth(t2["task_id"])
    # Player wrongly flags the first dimension as a deviation.
    out = {"deviations": [
        {"dimension": g["dimension"],
         "deviation": (i == 0),
         "deviation_kind": "outcome_switch" if i == 0 else None,
         "confidence": 0.9}
        for i, g in enumerate(gt["dimensions"])
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fp"] == 1
    assert s["breakdown"]["precision"] < 1.0
    assert any(f["category"] == "deviation_false_alarm" for f in s["findings"])


def test_missed_deviation_is_penalised():
    t3 = next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 3)
    gt = generator.ground_truth(t3["task_id"])
    # Player says nothing deviates.
    out = {"deviations": [
        {"dimension": g["dimension"], "deviation": False, "deviation_kind": None, "confidence": 0.8}
        for g in gt["dimensions"]
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fn"] == 1
    assert s["breakdown"]["recall"] < 1.0
    assert any(f["category"] == "deviation_missed" for f in s["findings"])


def test_kind_mislabel_lowers_kind_accuracy():
    t3 = next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 3)
    gt = generator.ground_truth(t3["task_id"])
    out = {"deviations": [
        {"dimension": g["dimension"], "deviation": g["deviation"],
         "deviation_kind": ("WRONG_KIND" if g["deviation"] else None), "confidence": 1.0}
        for g in gt["dimensions"]
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["detection_f1"] == 1.0
    assert s["breakdown"]["kind_accuracy"] < 1.0
    assert any(f["category"] == "kind_mislabel" for f in s["findings"])


def test_unknown_dimension_flagged():
    t1 = next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 1)
    gt = generator.ground_truth(t1["task_id"])
    out = {"deviations": [{"dimension": "not_a_real_dimension", "deviation": True,
                           "deviation_kind": "x", "confidence": 0.5}]}
    s = scorer.score(out, gt)
    assert any(f["category"] == "unknown_dimension" for f in s["findings"])


def test_findings_use_object_anchors():
    """Anchors must be objects (findings.schema.json), so redaction strips cleanly."""
    t3 = next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 3)
    gt = generator.ground_truth(t3["task_id"])
    out = {"deviations": [
        {"dimension": g["dimension"], "deviation": False, "deviation_kind": None, "confidence": 0.8}
        for g in gt["dimensions"]
    ]}
    s = scorer.score(out, gt)
    for f in s["findings"]:
        if "anchor" in f:
            assert isinstance(f["anchor"], dict)


_NEW_KINDS = {
    "posthoc_results_leak",
    "unresolved_contingency",
    "nondirectional_as_confirmatory",
    "maineffect_to_interaction",
}


def _t3_tasks_by_new_kind():
    """Map each new deviation_kind -> the T3 task whose single deviation is that kind."""
    out = {}
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] != 3:
            continue
        gt = generator.ground_truth(t["task_id"])
        dev = [g for g in gt["dimensions"] if g["deviation"]]
        if len(dev) == 1 and dev[0]["deviation_kind"] in _NEW_KINDS:
            out[dev[0]["deviation_kind"]] = (t, gt)
    return out


def test_each_new_kind_is_detectable_by_an_oracle():
    """G4: an oracle that mirrors gold scores 1.0 on each new-kind T3 task."""
    by_kind = _t3_tasks_by_new_kind()
    assert set(by_kind) == _NEW_KINDS, f"new kinds missing a T3 task: {_NEW_KINDS - set(by_kind)}"
    for kind, (_t, gt) in by_kind.items():
        out = _oracle_output(gt)
        s = scorer.score(out, gt)
        assert s["primary"] == 1.0, f"{kind}: oracle did not score 1.0"
        assert s["breakdown"]["kind_accuracy"] == 1.0


def test_new_kind_deviation_is_missed_if_ignored():
    """G4 contrapositive: ignoring the deviating dimension costs recall.

    Proves the injected mistake carries real detectable signal — a player that
    leaves it unflagged is penalised, so the kind is not a degenerate no-op.
    """
    by_kind = _t3_tasks_by_new_kind()
    for kind, (_t, gt) in by_kind.items():
        blind = {"deviations": [
            {"dimension": g["dimension"], "deviation": False, "deviation_kind": None, "confidence": 0.8}
            for g in gt["dimensions"]
        ]}
        s = scorer.score(blind, gt)
        assert s["breakdown"]["fn"] == 1, f"{kind}: expected one missed deviation"
        assert s["breakdown"]["recall"] < 1.0
        assert any(f["category"] == "deviation_missed" for f in s["findings"])
