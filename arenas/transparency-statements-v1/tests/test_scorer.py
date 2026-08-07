"""Tests for the transparency-statements scorer."""
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


scorer = _load("_transparency_statements_scorer", "scorer.py")
generator = _load("_transparency_statements_generator_s", "generator.py")


def _oracle_output(gt, confidence=1.0):
    """A perfect player: reproduces the gold field map exactly."""
    out = {"confidence": confidence}
    for field in ("coi", "funding"):
        out[field] = {"present": gt[field]["present"], "statement": gt[field]["statement"]}
    for field in ("data", "code", "materials"):
        out[field] = {"available": gt[field]["available"],
                      "on_request": gt[field]["on_request"],
                      "url": gt[field]["url"]}
    out["prereg"] = {"available": gt["prereg"]["available"], "url": gt["prereg"]["url"]}
    return out


def _first(tier):
    return next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == tier)


def test_perfect_player_scores_one_on_a_mistake_task():
    # T3 has exactly one injected mistake — detection is non-trivial.
    t3 = _first(3)
    gt = generator.ground_truth(t3["task_id"])
    s = scorer.score(_oracle_output(gt), gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["field_accuracy"] == 1.0
    assert s["breakdown"]["url_judgement"] == 1.0
    assert s["findings"] == []


def test_perfect_player_on_paraphrase_trap_scores_one():
    t2 = _first(2)
    gt = generator.ground_truth(t2["task_id"])
    s = scorer.score(_oracle_output(gt), gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["fp"] == 0
    assert s["findings"] == []


def test_false_positive_on_clean_field_is_penalised():
    """T2 trap: flagging an absent statement as present must hurt precision."""
    # Build a clean control where coi is genuinely absent (use a missing_coi task)
    t3 = _first(3)
    # find a task where coi is absent
    target = next(t for t in generator.generate("v1", seed=0)
                  if generator.ground_truth(t["task_id"])["coi"]["present"] is False)
    gt = generator.ground_truth(target["task_id"])
    out = _oracle_output(gt)
    out["coi"] = {"present": True, "statement": "The authors declare no competing interests."}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fp"] >= 1
    assert s["breakdown"]["precision"] < 1.0
    assert any(f["category"] == "statement_false_positive" for f in s["findings"])


def test_missed_statement_is_penalised():
    """Reporting a present funding statement as absent is a statement_missed."""
    t1 = _first(1)
    gt = generator.ground_truth(t1["task_id"])
    out = _oracle_output(gt)
    out["funding"] = {"present": False, "statement": ""}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fn"] >= 1
    assert s["breakdown"]["recall"] < 1.0
    assert any(f["category"] == "statement_missed" for f in s["findings"])


def test_url_misjudged_when_placeholder_called_real():
    """Treating a placeholder/broken URL as a real available repo is url_misjudged."""
    target = next(t for t in generator.generate("v1", seed=0)
                  if "placeholder_url" in generator.ground_truth(t["task_id"])["mistake_kinds"])
    gt = generator.ground_truth(target["task_id"])
    out = _oracle_output(gt)
    # find the offending field and wrongly mark it available
    for field in ("data", "code", "materials", "prereg"):
        if gt[field].get("available") is False and gt[field].get("url") is not None:
            out[field] = dict(out[field])
            out[field]["available"] = True
            break
    s = scorer.score(out, gt)
    assert s["breakdown"]["url_correct"] < s["breakdown"]["url_total"]
    assert s["breakdown"]["url_judgement"] < 1.0
    assert any(f["category"] == "url_misjudged" for f in s["findings"])


def test_on_request_treated_as_real_repo_is_url_misjudged():
    target = next(t for t in generator.generate("v1", seed=0)
                  if "data_on_request_not_real" in generator.ground_truth(t["task_id"])["mistake_kinds"])
    gt = generator.ground_truth(target["task_id"])
    out = _oracle_output(gt)
    # find an on-request open field (available False, url None) and call it available
    flipped = False
    for field in ("data", "code", "materials"):
        if gt[field].get("on_request") and gt[field].get("available") is False:
            out[field] = {"available": True, "on_request": False, "url": "https://osf.io/REAL/"}
            flipped = True
            break
    if not flipped:  # prereg-hosted on_request fallback
        out["prereg"] = {"available": True, "url": "https://osf.io/REAL/"}
    s = scorer.score(out, gt)
    assert any(f["category"] == "url_misjudged" for f in s["findings"])


def test_findings_use_object_anchors():
    """Anchors must be objects (findings.schema.json), so redaction strips cleanly."""
    t1 = _first(1)
    gt = generator.ground_truth(t1["task_id"])
    out = _oracle_output(gt)
    out["funding"] = {"present": False, "statement": ""}  # provoke a finding
    s = scorer.score(out, gt)
    assert s["findings"]
    for f in s["findings"]:
        if "anchor" in f:
            assert isinstance(f["anchor"], dict)


def test_calibration_component_present():
    t3 = _first(3)
    gt = generator.ground_truth(t3["task_id"])
    s = scorer.score(_oracle_output(gt, confidence=1.0), gt)
    assert "calibration" in s["breakdown"]
    # Perfect correct + full confidence => perfectly calibrated.
    assert s["breakdown"]["calibration"] == 1.0


def test_each_error_category_can_be_emitted():
    """statement_missed, statement_false_positive, url_misjudged each appear."""
    cats = set()
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        out = _oracle_output(gt)
        # corrupt every field one way to provoke all categories
        if gt["funding"]["present"]:
            out["funding"] = {"present": False, "statement": ""}        # statement_missed
        if gt["coi"]["present"] is False:
            out["coi"] = {"present": True, "statement": "x"}            # statement_false_positive
        for field in ("data", "code", "materials", "prereg"):
            if gt[field].get("available") is False and gt[field].get("url") is not None:
                out[field] = dict(out[field]); out[field]["available"] = True  # url_misjudged
        s = scorer.score(out, gt)
        cats.update(f["category"] for f in s["findings"])
    assert {"statement_missed", "statement_false_positive", "url_misjudged"} <= cats


def test_perfect_player_scores_one_on_every_task():
    """Gold is self-consistent across the whole broadened set: oracle => 1.0, no findings."""
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        s = scorer.score(_oracle_output(gt), gt)
        assert s["primary"] == 1.0, (t["task_id"], s["breakdown"])
        assert s["findings"] == [], (t["task_id"], s["findings"])


def _first_singleton(kind):
    return next(
        t for t in generator.generate("v1", seed=0)
        if generator.ground_truth(t["task_id"])["mistake_kinds"] == [kind]
    )


def test_false_open_claim_fooled_player_is_penalised():
    """Believing a bare 'openly available' (no-link) claim => statement_false_positive."""
    t = _first_singleton("false_open_claim")
    gt = generator.ground_truth(t["task_id"])
    out = _oracle_output(gt)
    for field in ("data", "code", "materials"):
        if (gt[field]["available"] is False and gt[field]["on_request"] is False
                and gt[field]["url"] is None):
            out[field] = {"available": True, "on_request": False, "url": None}
            break
    s = scorer.score(out, gt)
    assert s["breakdown"]["fp"] >= 1
    assert s["breakdown"]["field_accuracy"] < 1.0
    assert any(f["category"] == "statement_false_positive" for f in s["findings"])


def test_false_prereg_claim_fooled_player_is_penalised():
    """Believing a bare 'was preregistered' (no-link) claim => statement_false_positive."""
    t = _first_singleton("false_prereg_claim")
    gt = generator.ground_truth(t["task_id"])
    out = _oracle_output(gt)
    out["prereg"] = {"available": True, "url": None}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fp"] >= 1
    assert s["breakdown"]["field_accuracy"] < 1.0
    assert any(f["category"] == "statement_false_positive" for f in s["findings"])


def test_funding_on_request_fooled_player_is_penalised():
    """Counting a deferred funding line as a disclosure => statement_false_positive."""
    t = _first_singleton("funding_on_request")
    gt = generator.ground_truth(t["task_id"])
    out = _oracle_output(gt)
    out["funding"] = {"present": True, "statement": "Funding details are available on request."}
    s = scorer.score(out, gt)
    assert s["breakdown"]["fp"] >= 1
    assert s["breakdown"]["field_accuracy"] < 1.0
    assert any(f["category"] == "statement_false_positive" for f in s["findings"])
