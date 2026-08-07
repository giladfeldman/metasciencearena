"""Tests for the grim-consistency scorer."""
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


scorer = _load("_grim_consistency_scorer", "scorer.py")
generator = _load("_grim_consistency_generator_s", "generator.py")


def _oracle_output(gt, confidence=1.0):
    """A perfect player: matches every gold statistic's flag + issue_kind."""
    return {"records": [
        {"stat_id": g["stat_id"], "flagged": g["flagged"],
         "issue_kind": g["issue_kind"], "confidence": confidence}
        for g in gt["records"]
    ]}


def _first_tier(tier):
    return next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == tier)


def test_perfect_player_scores_one_on_an_issue_task():
    # Use a T3 task (one real issue) so detection metrics are non-trivial.
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    out = _oracle_output(gt)
    s = scorer.score(out, gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["detection_f1"] == 1.0
    assert s["breakdown"]["kind_accuracy"] == 1.0
    assert s["findings"] == []


def test_perfect_player_on_controls_trap_scores_one():
    t2 = _first_tier(2)
    gt = generator.ground_truth(t2["task_id"])
    out = _oracle_output(gt)  # all flagged=False
    s = scorer.score(out, gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["fp"] == 0


def test_false_alarm_on_controls_is_penalised():
    t2 = _first_tier(2)
    gt = generator.ground_truth(t2["task_id"])
    # Player wrongly flags the first clean statistic as grim_inconsistent.
    recs = []
    for i, g in enumerate(gt["records"]):
        recs.append({
            "stat_id": g["stat_id"],
            "flagged": (i == 0),
            "issue_kind": "grim_inconsistent" if i == 0 else None,
            "confidence": 0.9,
        })
    s = scorer.score({"records": recs}, gt)
    assert s["breakdown"]["fp"] == 1
    assert s["breakdown"]["precision"] < 1.0
    assert any(f["category"] == "grim_false_alarm" for f in s["findings"])


def test_missed_issue_is_penalised():
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    # Player says nothing is flagged.
    out = {"records": [
        {"stat_id": g["stat_id"], "flagged": False, "issue_kind": None, "confidence": 0.8}
        for g in gt["records"]
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fn"] == 1
    assert s["breakdown"]["recall"] < 1.0
    assert any(f["category"] == "grim_missed" for f in s["findings"])


def test_kind_mislabel_lowers_kind_accuracy():
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    out = {"records": [
        {"stat_id": g["stat_id"], "flagged": g["flagged"],
         "issue_kind": ("WRONG_KIND" if g["flagged"] else None), "confidence": 1.0}
        for g in gt["records"]
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["detection_f1"] == 1.0
    assert s["breakdown"]["kind_accuracy"] < 1.0
    assert any(f["category"] == "kind_mislabel" for f in s["findings"])


def test_phantom_statistic_flag_is_false_alarm():
    t1 = _first_tier(1)
    gt = generator.ground_truth(t1["task_id"])
    out = {"records": [{"stat_id": "not_a_real_statistic", "flagged": True,
                        "issue_kind": "grim_inconsistent", "confidence": 0.5}]}
    s = scorer.score(out, gt)
    assert any(f["category"] == "grim_false_alarm" for f in s["findings"])


def test_each_error_category_emitted_by_a_wrong_player():
    """grim_missed, grim_false_alarm, kind_mislabel are all reachable."""
    # T5 has multiple flagged statistics; build a player that misses one, mislabels
    # another, and false-alarms a clean one — exercising all three categories.
    t5 = _first_tier(5)
    gt = generator.ground_truth(t5["task_id"])
    flagged = [g for g in gt["records"] if g["flagged"]]
    clean = [g for g in gt["records"] if not g["flagged"]]
    assert len(flagged) >= 2 and len(clean) >= 1

    recs = []
    for g in gt["records"]:
        if g is flagged[0]:                       # miss it
            recs.append({"stat_id": g["stat_id"], "flagged": False,
                         "issue_kind": None, "confidence": 0.5})
        elif g is flagged[1]:                     # mislabel it
            recs.append({"stat_id": g["stat_id"], "flagged": True,
                         "issue_kind": "WRONG_KIND", "confidence": 0.5})
        elif g is clean[0]:                       # false alarm
            recs.append({"stat_id": g["stat_id"], "flagged": True,
                         "issue_kind": "grim_inconsistent", "confidence": 0.5})
        else:
            recs.append({"stat_id": g["stat_id"], "flagged": g["flagged"],
                         "issue_kind": g["issue_kind"], "confidence": 0.5})
    s = scorer.score({"records": recs}, gt)
    cats = {f["category"] for f in s["findings"]}
    assert {"grim_missed", "grim_false_alarm", "kind_mislabel"} <= cats


def test_findings_use_object_anchors():
    """Anchors must be objects (findings.schema.json), so redaction strips cleanly."""
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    out = {"records": [
        {"stat_id": g["stat_id"], "flagged": False, "issue_kind": None, "confidence": 0.8}
        for g in gt["records"]
    ]}
    s = scorer.score(out, gt)
    assert s["findings"]
    for f in s["findings"]:
        if "anchor" in f:
            assert isinstance(f["anchor"], dict)


def test_calibration_rewards_confident_correct_player():
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    s = scorer.score(_oracle_output(gt, confidence=1.0), gt)
    assert s["breakdown"]["calibration"] == 1.0
