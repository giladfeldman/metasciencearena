"""Tests for the open-practices-repro scorer."""
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


scorer = _load("_open_practices_repro_scorer", "scorer.py")
generator = _load("_open_practices_repro_generator_s", "generator.py")


def _oracle_output(gt, confidence=1.0):
    """A perfect player: one record per gold target, correct flag + kind."""
    return {"records": [
        {"target": g["target"],
         "flagged": g["issue_kind"] is not None,
         "issue_kind": g["issue_kind"],
         "confidence": confidence}
        for g in gt["records"]
    ]}


def _first_tier(tier):
    t = next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == tier)
    return t, generator.ground_truth(t["task_id"])


def test_perfect_player_scores_one_on_a_defect_task():
    # T3 has exactly one injected defect, so detection metrics are non-trivial.
    _, gt = _first_tier(3)
    s = scorer.score(_oracle_output(gt), gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["detection_f1"] == 1.0
    assert s["breakdown"]["kind_accuracy"] == 1.0
    assert s["findings"] == []


def test_perfect_player_on_false_alarm_trap_scores_one():
    """T2 trap: clean-but-suspicious files. A clean-correct player must score 1.0."""
    _, gt = _first_tier(2)
    s = scorer.score(_oracle_output(gt), gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["fp"] == 0


def test_false_alarm_on_trap_is_penalised():
    _, gt = _first_tier(2)
    # Player wrongly flags the first target as having an absolute_path defect.
    out = {"records": [
        {"target": g["target"],
         "flagged": (i == 0),
         "issue_kind": "absolute_path" if i == 0 else None,
         "confidence": 0.9}
        for i, g in enumerate(gt["records"])
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fp"] == 1
    assert s["breakdown"]["precision"] < 1.0
    assert any(f["category"] == "repro_issue_false_alarm" for f in s["findings"])


def test_missed_defect_is_penalised():
    _, gt = _first_tier(3)
    # Player flags nothing.
    out = {"records": [
        {"target": g["target"], "flagged": False, "issue_kind": None, "confidence": 0.8}
        for g in gt["records"]
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fn"] == 1
    assert s["breakdown"]["recall"] < 1.0
    assert any(f["category"] == "repro_issue_missed" for f in s["findings"])


def test_kind_mislabel_lowers_kind_accuracy():
    # Use T5 (multiple defects) so kind_total > 1 and mislabel is unambiguous.
    _, gt = _first_tier(5)
    out = {"records": [
        {"target": g["target"],
         "flagged": g["issue_kind"] is not None,
         "issue_kind": ("WRONG_KIND" if g["issue_kind"] is not None else None),
         "confidence": 1.0}
        for g in gt["records"]
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["detection_f1"] == 1.0
    assert s["breakdown"]["kind_accuracy"] < 1.0
    assert any(f["category"] == "kind_mislabel" for f in s["findings"])


def test_every_error_category_can_be_emitted():
    """A single wrong player should surface all three error_categories."""
    _, gt = _first_tier(5)  # has multiple defects + at least one clean target
    defects = [g for g in gt["records"] if g["issue_kind"] is not None]
    cleans = [g for g in gt["records"] if g["issue_kind"] is None]
    assert len(defects) >= 2 and cleans, "T5 should have >=2 defects and a clean target"
    mislabel_target = defects[0]["target"]   # kind_mislabel
    missed_target = defects[1]["target"]      # repro_issue_missed
    false_alarm_target = cleans[0]["target"]  # repro_issue_false_alarm
    out_records = []
    for g in gt["records"]:
        if g["target"] == mislabel_target:
            out_records.append({"target": g["target"], "flagged": True,
                                "issue_kind": "WRONG_KIND", "confidence": 0.7})
        elif g["target"] == missed_target:
            out_records.append({"target": g["target"], "flagged": False,
                                "issue_kind": None, "confidence": 0.7})
        elif g["target"] == false_alarm_target:
            out_records.append({"target": g["target"], "flagged": True,
                                "issue_kind": "absolute_path", "confidence": 0.7})
        else:
            out_records.append({"target": g["target"],
                                "flagged": g["issue_kind"] is not None,
                                "issue_kind": g["issue_kind"], "confidence": 0.7})
    s = scorer.score({"records": out_records}, gt)
    cats = {f["category"] for f in s["findings"]}
    assert {"repro_issue_missed", "repro_issue_false_alarm", "kind_mislabel"} <= cats


def test_broken_link_detection_on_repo_url():
    """A T3 broken_link task: the defect target is the repo_url."""
    bl = None
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        if any(r["issue_kind"] == "broken_link" for r in gt["records"]):
            bl = (t, gt)
            break
    assert bl is not None
    t, gt = bl
    s = scorer.score(_oracle_output(gt), gt)
    assert s["primary"] == 1.0


def test_unknown_flagged_target_is_a_false_alarm():
    _, gt = _first_tier(1)
    out = {"records": [{"target": "totally_made_up.R", "flagged": True,
                        "issue_kind": "absolute_path", "confidence": 0.5}]}
    s = scorer.score(out, gt)
    assert any(f["category"] == "repro_issue_false_alarm" for f in s["findings"])


def test_findings_use_object_anchors():
    """Anchors must be objects (findings.schema.json), so redaction strips cleanly."""
    _, gt = _first_tier(3)
    out = {"records": [
        {"target": g["target"], "flagged": False, "issue_kind": None, "confidence": 0.8}
        for g in gt["records"]
    ]}
    s = scorer.score(out, gt)
    for f in s["findings"]:
        if "anchor" in f:
            assert isinstance(f["anchor"], dict)


def test_calibration_penalises_overconfident_errors():
    """A confidently-wrong player should have calibration < 1."""
    _, gt = _first_tier(3)
    # Flag nothing but with high confidence -> the real defect is a confident miss.
    out = {"records": [
        {"target": g["target"], "flagged": False, "issue_kind": None, "confidence": 1.0}
        for g in gt["records"]
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["calibration"] < 1.0


def _task_with_kind(kind):
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        if any(r["issue_kind"] == kind for r in gt["records"]):
            return t, gt
    raise AssertionError(f"no task injects {kind!r}")


def test_perfect_player_scores_one_on_each_open_practices_kind():
    """A clean-correct player on a data/code/materials availability defect scores 1.0."""
    for kind in ("dead_data_link", "available_upon_request", "materials_claim_no_link"):
        _, gt = _task_with_kind(kind)
        s = scorer.score(_oracle_output(gt), gt)
        assert s["primary"] == 1.0, kind
        assert s["breakdown"]["detection_f1"] == 1.0
        assert s["findings"] == []


def test_false_alarm_on_clean_availability_statement_is_penalised():
    """Flagging a genuine-DOI data statement as dead_data_link is a false alarm."""
    # T3 dead_data_link task carries a CLEAN MATERIALS.md (real repo) as a control.
    t, gt = _task_with_kind("dead_data_link")
    clean_doc = next(
        g["target"] for g in gt["records"]
        if g["target"] == "MATERIALS.md" and g["issue_kind"] is None
    )
    out = {"records": [
        {"target": g["target"],
         "flagged": True if g["target"] == clean_doc else (g["issue_kind"] is not None),
         "issue_kind": "materials_claim_no_link" if g["target"] == clean_doc else g["issue_kind"],
         "confidence": 0.9}
        for g in gt["records"]
    ]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fp"] == 1
    assert any(f["category"] == "repro_issue_false_alarm" for f in s["findings"])
