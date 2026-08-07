"""Tests for the prereg-extraction scorer."""
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


scorer = _load("_prereg_extraction_scorer", "scorer.py")
generator = _load("_prereg_extraction_generator_s", "generator.py")


def _oracle_output(gt, confidence=1.0):
    """A perfect player: matches gold exactly."""
    return {
        "prereg_found": gt["prereg_found"],
        "platform": gt["platform"],
        "link": gt["link"],
        "fields": dict(gt["fields"]),
        "confidence": confidence,
    }


def _first_with(predicate):
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        if predicate(gt):
            return t, gt
    raise AssertionError("no matching task")


def test_perfect_player_scores_one_on_a_mistake_task():
    # A task with an injected mistake that still has an extractable prereg.
    _, gt = _first_with(lambda g: "prereg_plaintext" in g["mistake_kinds"])
    s = scorer.score(_oracle_output(gt), gt)
    assert s["primary"] == 1.0
    assert s["breakdown"]["detection"] == 1.0
    assert s["breakdown"]["platform_acc"] == 1.0
    assert s["breakdown"]["field_f1"] == 1.0
    assert s["findings"] == []


def test_perfect_player_on_decoy_trap_scores_one():
    # T2 decoy: clean control, gold prereg_found=False.
    t2 = next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 2)
    gt = generator.ground_truth(t2["task_id"])
    s = scorer.score(_oracle_output(gt), gt)
    assert s["primary"] == 1.0
    assert s["findings"] == []


def test_false_alarm_on_decoy_is_penalised():
    t2 = next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 2)
    gt = generator.ground_truth(t2["task_id"])
    # Player wrongly claims a prereg is present.
    out = {
        "prereg_found": True,
        "platform": "osf",
        "link": "https://osf.io/zzzzz/",
        "fields": {"hypotheses": "made up", "design": None, "sample_size": None, "analysis_plan": None},
        "confidence": 0.9,
    }
    s = scorer.score(out, gt)
    assert s["breakdown"]["detection"] == 0.0
    assert s["primary"] == 0.0
    assert any(f["category"] == "prereg_false_alarm" for f in s["findings"])


def test_missed_prereg_is_penalised():
    _, gt = _first_with(lambda g: g["prereg_found"] is True)
    out = {
        "prereg_found": False,
        "platform": None,
        "link": None,
        "fields": {"hypotheses": None, "design": None, "sample_size": None, "analysis_plan": None},
        "confidence": 0.8,
    }
    s = scorer.score(out, gt)
    assert s["breakdown"]["detection"] == 0.0
    assert any(f["category"] == "prereg_missed" for f in s["findings"])


def test_platform_mislabel_flagged():
    _, gt = _first_with(lambda g: g["platform"] == "osf")
    out = _oracle_output(gt)
    out["platform"] = "aspredicted"  # wrong platform
    s = scorer.score(out, gt)
    assert s["breakdown"]["platform_acc"] == 0.0
    assert any(f["category"] == "platform_mislabel" for f in s["findings"])


def test_field_wrong_flagged():
    _, gt = _first_with(lambda g: g["prereg_found"] is True)
    out = _oracle_output(gt)
    out["fields"] = {f: "completely unrelated filler text xyz" for f in out["fields"]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["field_f1"] < 1.0
    assert any(f["category"] == "field_wrong" for f in s["findings"])


def test_findings_use_object_anchors():
    """Anchors must be objects (findings.schema.json), so redaction strips cleanly."""
    _, gt = _first_with(lambda g: g["prereg_found"] is True)
    out = {
        "prereg_found": False,
        "platform": None,
        "link": None,
        "fields": {f: None for f in ("hypotheses", "design", "sample_size", "analysis_plan")},
        "confidence": 0.8,
    }
    s = scorer.score(out, gt)
    for f in s["findings"]:
        if "anchor" in f:
            assert isinstance(f["anchor"], dict)


def test_oracle_scores_one_on_every_task():
    """Self-consistency: a perfect player must score 1.0 with no findings on ALL
    tasks (incl. the new integrity abuses and their clean controls) — proof the
    broadened gold is not degenerate."""
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        s = scorer.score(_oracle_output(gt), gt)
        assert s["primary"] == 1.0, (t["task_id"], gt["mistake_kinds"], s["breakdown"])
        assert s["findings"] == [], (t["task_id"], gt["mistake_kinds"], s["findings"])


def test_embargoed_hallucinated_fields_are_field_wrong():
    """embargoed_at_publication: gold fields are null. Inventing field text must
    drop field_f1 and raise field_wrong."""
    _, gt = _first_with(lambda g: "embargoed_at_publication" in g["mistake_kinds"])
    assert gt["prereg_found"] is True
    out = _oracle_output(gt)
    out["fields"] = {f: "fabricated plan text" for f in out["fields"]}
    s = scorer.score(out, gt)
    assert s["breakdown"]["field_f1"] == 0.0
    assert any(f["category"] == "field_wrong" for f in s["findings"])


def test_withdrawn_flagged_as_present_is_false_alarm():
    """withdrawn_still_cited: gold found=False. Claiming a prereg is present must
    zero detection and raise prereg_false_alarm."""
    _, gt = _first_with(lambda g: "withdrawn_still_cited" in g["mistake_kinds"])
    assert gt["prereg_found"] is False
    out = {
        "prereg_found": True,
        "platform": "osf",
        "link": "https://osf.io/a1b2c/",
        "fields": {f: "x" for f in ("hypotheses", "design", "sample_size", "analysis_plan")},
        "confidence": 0.9,
    }
    s = scorer.score(out, gt)
    assert s["breakdown"]["detection"] == 0.0
    assert s["primary"] == 0.0
    assert any(f["category"] == "prereg_false_alarm" for f in s["findings"])


def test_viewonly_platform_mislabel_penalised():
    """viewonly_instead_of_doi: gold platform osf. Mislabelling it (e.g. as
    aspredicted because the link looks unusual) must zero platform_acc."""
    _, gt = _first_with(lambda g: "viewonly_instead_of_doi" in g["mistake_kinds"])
    assert gt["platform"] == "osf"
    out = _oracle_output(gt)
    out["platform"] = "aspredicted"
    s = scorer.score(out, gt)
    assert s["breakdown"]["platform_acc"] == 0.0
    assert any(f["category"] == "platform_mislabel" for f in s["findings"])


def test_all_error_categories_emittable():
    """Each declared error_category must be producible by some wrong output."""
    emitted = set()

    # prereg_missed + (platform unscorable): say nothing on a present prereg.
    _, gt_present = _first_with(lambda g: g["prereg_found"] is True and g["platform"] == "osf")
    s = scorer.score({"prereg_found": False, "platform": None, "link": None,
                      "fields": {f: None for f in ("hypotheses", "design", "sample_size", "analysis_plan")},
                      "confidence": 0.5}, gt_present)
    emitted.update(f["category"] for f in s["findings"])

    # prereg_false_alarm: flag a decoy.
    t2 = next(t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 2)
    gt2 = generator.ground_truth(t2["task_id"])
    s = scorer.score({"prereg_found": True, "platform": "osf", "link": "x",
                      "fields": {"hypotheses": "x", "design": None, "sample_size": None, "analysis_plan": None},
                      "confidence": 0.5}, gt2)
    emitted.update(f["category"] for f in s["findings"])

    # platform_mislabel + field_wrong on a present prereg.
    out = _oracle_output(gt_present)
    out["platform"] = "aspredicted"
    out["fields"] = {f: "irrelevant nonsense qqq" for f in out["fields"]}
    s = scorer.score(out, gt_present)
    emitted.update(f["category"] for f in s["findings"])

    assert {"prereg_missed", "prereg_false_alarm", "platform_mislabel", "field_wrong"} <= emitted
