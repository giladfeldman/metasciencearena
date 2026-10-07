"""Tests for the findings emitted by stats-extraction-v1's scorer.

One test per category fires; one perfection test guards against false-positives;
one schema-validation test ensures every emitted finding is contract-valid.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from framework.paths import schema_path

ARENA_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ARENA_DIR.parents[1]

# Load this arena's scorer.py under a unique module name so it doesn't collide
# with sibling arenas that also expose a top-level `scorer` module.
_SPEC = importlib.util.spec_from_file_location(
    "_stats_extraction_scorer", ARENA_DIR / "scorer.py"
)
scorer = importlib.util.module_from_spec(_SPEC)
sys.modules["_stats_extraction_scorer"] = scorer
_SPEC.loader.exec_module(scorer)

FINDINGS_SCHEMA_PATH = schema_path("findings.schema.json")


@pytest.fixture(scope="module")
def findings_validator() -> Draft202012Validator:
    with FINDINGS_SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return Draft202012Validator(json.load(f))


def _gt_item(*, kind="nhst_stat", value=2.34, char_start=0, char_end=4,
             truthful=True, deception_kind=None) -> dict:
    return {
        "kind": kind,
        "fields": {"test_type": "t", "value": value, "p": 0.04},
        "span": {"text": "x", "char_start": char_start, "char_end": char_end},
        "truthful": truthful,
        "deception_kind": deception_kind,
    }


def _ext(*, kind="nhst_stat", value=2.34, char_start=0, char_end=4,
         confidence=0.7, flagged=False) -> dict:
    return {
        "kind": kind,
        "fields": {"value": value, "p": 0.04},
        "span": {"char_start": char_start, "char_end": char_end},
        "confidence": confidence,
        "flagged_suspicious": flagged,
    }


def _categories(findings: list[dict]) -> list[str]:
    return [f["category"] for f in findings]


def test_perfect_extraction_emits_no_findings(findings_validator):
    gt = {"items": [_gt_item()]}
    out = {"extractions": [_ext()]}
    s = scorer.score(out, gt)
    assert s["findings"] == []
    findings_validator.validate(s["findings"])


def test_extraction_missed_fires(findings_validator):
    gt = {"items": [_gt_item()]}
    out = {"extractions": []}
    s = scorer.score(out, gt)
    assert "extraction_missed" in _categories(s["findings"])
    finding = next(f for f in s["findings"] if f["category"] == "extraction_missed")
    assert finding["correct_value"] == 2.34
    findings_validator.validate(s["findings"])


def test_extraction_extra_fires(findings_validator):
    gt = {"items": []}
    # Place the extra extraction at a span that doesn't overlap any gold (trivially true here).
    out = {"extractions": [_ext(value=9.99, char_start=100, char_end=104)]}
    s = scorer.score(out, gt)
    assert "extraction_extra" in _categories(s["findings"])
    findings_validator.validate(s["findings"])


def test_kind_mismatch_fires_when_unmatched_extraction_overlaps_wrong_kind(findings_validator):
    gt = {"items": [_gt_item(kind="nhst_stat", char_start=0, char_end=10)]}
    # Same span, wrong kind — won't match (matcher requires kind agreement)
    # but overlaps the gold span enough to fire kind_mismatch.
    out = {"extractions": [_ext(kind="effect_size", char_start=0, char_end=10)]}
    s = scorer.score(out, gt)
    cats = _categories(s["findings"])
    assert "kind_mismatch" in cats
    finding = next(f for f in s["findings"] if f["category"] == "kind_mismatch")
    assert finding["correct_value"] == "nhst_stat"
    assert finding["evidence"] == "effect_size"
    findings_validator.validate(s["findings"])


def test_value_off_fires_on_matched_pair_with_wrong_value(findings_validator):
    gt = {"items": [_gt_item(value=2.34)]}
    out = {"extractions": [_ext(value=5.55)]}
    s = scorer.score(out, gt)
    assert "value_off" in _categories(s["findings"])
    finding = next(f for f in s["findings"] if f["category"] == "value_off")
    assert finding["correct_value"] == 2.34
    assert finding["evidence"] == "5.55"
    findings_validator.validate(s["findings"])


def test_deception_missed_fires_when_player_does_not_flag(findings_validator):
    gt = {"items": [_gt_item(truthful=False, deception_kind="impossible_t_value")]}
    out = {"extractions": [_ext(flagged=False)]}
    s = scorer.score(out, gt)
    assert "deception_missed" in _categories(s["findings"])
    finding = next(f for f in s["findings"] if f["category"] == "deception_missed")
    assert finding["correct_value"] == "impossible_t_value"
    findings_validator.validate(s["findings"])


def test_deception_false_alarm_fires_on_truthful_flagged(findings_validator):
    gt = {"items": [_gt_item(truthful=True)]}
    out = {"extractions": [_ext(flagged=True)]}
    s = scorer.score(out, gt)
    assert "deception_false_alarm" in _categories(s["findings"])
    findings_validator.validate(s["findings"])


def test_calibration_overconfident_fires_on_high_confidence_wrong_value(findings_validator):
    gt = {"items": [_gt_item(value=2.34)]}
    # Matched-but-wrong (value differs) AND confidence=0.95
    out = {"extractions": [_ext(value=9.99, confidence=0.95)]}
    s = scorer.score(out, gt)
    cats = _categories(s["findings"])
    assert "calibration_overconfident" in cats
    assert "value_off" in cats
    findings_validator.validate(s["findings"])


def test_calibration_overconfident_fires_on_high_confidence_fp(findings_validator):
    gt = {"items": []}
    out = {"extractions": [_ext(value=9.99, confidence=0.95, char_start=100, char_end=104)]}
    s = scorer.score(out, gt)
    cats = _categories(s["findings"])
    assert "calibration_overconfident" in cats
    assert "extraction_extra" in cats
    findings_validator.validate(s["findings"])


def test_all_emitted_finding_categories_are_declared_in_arena_yaml():
    manifest = yaml.safe_load((ARENA_DIR / "arena.yaml").read_text(encoding="utf-8"))
    declared = {c["id"] for c in manifest["error_categories"]}
    # Run a "bad" scorer call covering several categories at once.
    gt = {"items": [
        _gt_item(value=2.34, char_start=0, char_end=4),
        _gt_item(truthful=False, deception_kind="bogus_p", char_start=10, char_end=14),
    ]}
    out = {"extractions": [
        _ext(value=9.99, confidence=0.95, char_start=0, char_end=4),
        _ext(flagged=False, char_start=10, char_end=14),
        _ext(value=1.0, confidence=0.95, char_start=200, char_end=204),
    ]}
    s = scorer.score(out, gt)
    for f in s["findings"]:
        assert f["category"] in declared, f"undeclared category: {f['category']}"
