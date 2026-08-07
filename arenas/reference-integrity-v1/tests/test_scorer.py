"""Tests for the reference-integrity scorer."""
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


scorer = _load("_reference_integrity_scorer", "scorer.py")
generator = _load("_reference_integrity_generator_s", "generator.py")


def _oracle_output(gt, confidence=1.0):
    """A perfect player: matches every gold reference's flag + issue_kind."""
    return {"records": [
        {"reference_id": g["reference_id"], "flagged": g["flagged"],
         "issue_kind": g["issue_kind"], "confidence": confidence}
        for g in gt["references"]
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
    # Player wrongly flags the first clean reference as retracted.
    recs = []
    for i, g in enumerate(gt["references"]):
        recs.append({
            "reference_id": g["reference_id"],
            "flagged": (i == 0),
            "issue_kind": "retracted" if i == 0 else None,
            "confidence": 0.9,
        })
    s = scorer.score({"records": recs}, gt)
    assert s["breakdown"]["fp"] == 1
    assert s["breakdown"]["precision"] < 1.0
    assert any(f["category"] == "integrity_false_alarm" for f in s["findings"])


def test_missed_issue_is_penalised():
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    # Player says nothing is flagged.
    out = {"records": [
        {"reference_id": g["reference_id"], "flagged": False, "issue_kind": None, "confidence": 0.8}
        for g in gt["references"]
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fn"] == 1
    assert s["breakdown"]["recall"] < 1.0
    assert any(f["category"] == "integrity_missed" for f in s["findings"])


def test_kind_mislabel_lowers_kind_accuracy():
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    out = {"records": [
        {"reference_id": g["reference_id"], "flagged": g["flagged"],
         "issue_kind": ("WRONG_KIND" if g["flagged"] else None), "confidence": 1.0}
        for g in gt["references"]
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["detection_f1"] == 1.0
    assert s["breakdown"]["kind_accuracy"] < 1.0
    assert any(f["category"] == "kind_mislabel" for f in s["findings"])


def test_phantom_reference_flag_is_false_alarm():
    t1 = _first_tier(1)
    gt = generator.ground_truth(t1["task_id"])
    out = {"records": [{"reference_id": "not_a_real_reference", "flagged": True,
                        "issue_kind": "retracted", "confidence": 0.5}]}
    s = scorer.score(out, gt)
    assert any(f["category"] == "integrity_false_alarm" for f in s["findings"])


def test_each_error_category_emitted_by_a_wrong_player():
    """integrity_missed, integrity_false_alarm, kind_mislabel are all reachable."""
    # T5 has multiple flagged references; build a player that misses one, mislabels
    # another, and false-alarms a clean one — exercising all three categories.
    t5 = _first_tier(5)
    gt = generator.ground_truth(t5["task_id"])
    flagged = [g for g in gt["references"] if g["flagged"]]
    clean = [g for g in gt["references"] if not g["flagged"]]
    assert len(flagged) >= 2 and len(clean) >= 1

    recs = []
    for g in gt["references"]:
        if g is flagged[0]:                       # miss it
            recs.append({"reference_id": g["reference_id"], "flagged": False,
                         "issue_kind": None, "confidence": 0.5})
        elif g is flagged[1]:                     # mislabel it
            recs.append({"reference_id": g["reference_id"], "flagged": True,
                         "issue_kind": "WRONG_KIND", "confidence": 0.5})
        elif g is clean[0]:                       # false alarm
            recs.append({"reference_id": g["reference_id"], "flagged": True,
                         "issue_kind": "retracted", "confidence": 0.5})
        else:
            recs.append({"reference_id": g["reference_id"], "flagged": g["flagged"],
                         "issue_kind": g["issue_kind"], "confidence": 0.5})
    s = scorer.score({"records": recs}, gt)
    cats = {f["category"] for f in s["findings"]}
    assert {"integrity_missed", "integrity_false_alarm", "kind_mislabel"} <= cats


def test_findings_use_object_anchors():
    """Anchors must be objects (findings.schema.json), so redaction strips cleanly."""
    t3 = _first_tier(3)
    gt = generator.ground_truth(t3["task_id"])
    out = {"records": [
        {"reference_id": g["reference_id"], "flagged": False, "issue_kind": None, "confidence": 0.8}
        for g in gt["references"]
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


def _task_with_kind(kind):
    """Return the first task whose gold contains a flagged reference of `kind`."""
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        if any(g["flagged"] and g["issue_kind"] == kind for g in gt["references"]):
            return t, gt
    raise AssertionError(f"no task emits kind {kind!r}")


def test_perfect_player_scores_one_on_each_new_kind():
    """The scorer is kind-agnostic; confirm the 3 new kinds score cleanly."""
    for kind in ("invalid_doi", "predatory_source", "tortured_phrase"):
        _t, gt = _task_with_kind(kind)
        s = scorer.score(_oracle_output(gt), gt)
        assert s["breakdown"]["detection_f1"] == 1.0, kind
        assert s["breakdown"]["kind_accuracy"] == 1.0, kind
        assert s["findings"] == [], kind


def test_mislabelling_a_new_kind_is_caught():
    """Flagging an invalid_doi but calling it 'retracted' is a kind_mislabel."""
    _t, gt = _task_with_kind("invalid_doi")
    recs = []
    for g in gt["references"]:
        issue = g["issue_kind"]
        if g["flagged"] and issue == "invalid_doi":
            issue = "retracted"  # right flag, wrong kind
        recs.append({"reference_id": g["reference_id"], "flagged": g["flagged"],
                     "issue_kind": issue, "confidence": 1.0})
    s = scorer.score({"records": recs}, gt)
    assert s["breakdown"]["detection_f1"] == 1.0
    assert s["breakdown"]["kind_accuracy"] < 1.0
    assert any(f["category"] == "kind_mislabel" for f in s["findings"])
