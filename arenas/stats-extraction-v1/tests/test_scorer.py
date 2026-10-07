from pathlib import Path
import importlib.util
import sys

ARENA_DIR = Path(__file__).resolve().parents[1]

def _load(mod_name, filename):
    """Load an arena module under a UNIQUE name.

    A bare `import scorer` after `sys.path.insert(ARENA_DIR)` registers
    `sys.modules["scorer"]`, and all 19 arenas ship a `scorer.py`. The first
    arena imported wins, so every later arena's tests silently exercised the
    WRONG arena's code. 119 tests were also being dropped as duplicate modules
    before `consider_namespace_packages` was enabled (2026-08-07).
    """
    spec = importlib.util.spec_from_file_location(mod_name, ARENA_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod



scorer = _load("_stats_extraction_v1_scorer", "scorer.py")


def _gt(items):
    return {"items": items}


def _ext(span_start, span_end, kind, value, confidence=1.0, flagged=False):
    return {
        "span": {"text": "x", "char_start": span_start, "char_end": span_end},
        "kind": kind,
        "fields": {"value": value},
        "confidence": confidence,
        "flagged_suspicious": flagged,
    }


def test_perfect_match_scores_one():
    gt = _gt([{"kind": "nhst_stat", "fields": {"value": 2.31},
               "span": {"text": "2.31", "char_start": 10, "char_end": 14},
               "truthful": True, "deception_kind": None}])
    out = {"extractions": [_ext(10, 14, "nhst_stat", 2.31)], "player_strategy_notes": ""}
    s = scorer.score(out, gt)
    assert s["primary"] > 0.95
    assert s["breakdown"]["precision"] == 1.0
    assert s["breakdown"]["recall"] == 1.0


def test_missing_extraction_lowers_recall():
    gt = _gt([
        {"kind": "nhst_stat", "fields": {"value": 2.31},
         "span": {"text": "2.31", "char_start": 10, "char_end": 14}, "truthful": True, "deception_kind": None},
        {"kind": "effect_size", "fields": {"value": 0.5},
         "span": {"text": "0.50", "char_start": 30, "char_end": 34}, "truthful": True, "deception_kind": None},
    ])
    out = {"extractions": [_ext(10, 14, "nhst_stat", 2.31)], "player_strategy_notes": ""}
    s = scorer.score(out, gt)
    assert s["breakdown"]["recall"] == 0.5


# RETARGETED 2026-08-21. These two used to assert that `calibration` carried the
# deception signal: the scorer remapped a flagged deception's confidence to
# `1 - raw_conf`, so "low confidence + flagged" read as "expressed suspicion =
# correct". That only worked while extraction and verification were fused into
# one boolean, and it made `confidence` mean something different on a deception
# item than the prompt asks for ("your actual certainty").
#
# The two questions are now two heads, so the deception signal lives in
# `verification_score` and `confidence` means the same thing on every item. The
# INTENT of both tests is preserved — missing a deception must be punished,
# catching one must be rewarded — retargeted onto the head that now carries it.

def test_missing_a_deception_is_punished_by_the_verification_head():
    gt = _gt([{"kind": "nhst_stat", "fields": {"value": 2.31},
               "span": {"text": "2.31", "char_start": 10, "char_end": 14},
               "truthful": False, "deception_kind": "internal_inconsistency"}])
    out = {"extractions": [_ext(10, 14, "nhst_stat", 2.31, confidence=1.0, flagged=False)], "player_strategy_notes": ""}
    s = scorer.score(out, gt)
    assert s["breakdown"]["verification_score"] == 0.0
    assert s["primary"] == 0.0
    assert "deception_missed" in [f["category"] for f in s["findings"]]
    # The EXTRACTION was correct and confidently stated, and that is now honest
    # rather than punished: the player read the printed number right.
    assert s["breakdown"]["recall"] == 1.0


def test_catching_a_deception_is_rewarded_by_the_verification_head():
    gt = _gt([{"kind": "nhst_stat", "fields": {"value": 2.31},
               "span": {"text": "2.31", "char_start": 10, "char_end": 14},
               "truthful": False, "deception_kind": "internal_inconsistency"}])
    out = {"extractions": [_ext(10, 14, "nhst_stat", 2.31, confidence=0.2, flagged=True)], "player_strategy_notes": ""}
    s = scorer.score(out, gt)
    assert s["breakdown"]["verification_score"] == 1.0
    assert s["primary"] == 1.0
    assert not [f for f in s["findings"] if f["category"].startswith("deception")]


def test_missing_or_malformed_confidence_degrades_gracefully():
    # DR-0004: a player extraction without (or with a non-numeric) `confidence`
    # must NOT crash the scorer — the runner would otherwise record a full
    # task-error instead of a partial score. Default to 0.0.
    gt = _gt([{"kind": "nhst_stat", "fields": {"value": 2.31},
               "span": {"text": "2.31", "char_start": 10, "char_end": 14},
               "truthful": True, "deception_kind": None}])
    # matched extraction missing 'confidence' entirely
    matched = {"span": {"text": "x", "char_start": 10, "char_end": 14},
               "kind": "nhst_stat", "fields": {"value": 2.31}, "flagged_suspicious": False}
    # unmatched extraction with a non-numeric 'confidence'
    unmatched = {"span": {"text": "x", "char_start": 50, "char_end": 54},
                 "kind": "effect_size", "fields": {"value": 9.9},
                 "confidence": "high", "flagged_suspicious": False}
    out = {"extractions": [matched, unmatched], "player_strategy_notes": ""}
    s = scorer.score(out, gt)
    assert isinstance(s, dict) and "primary" in s
    assert "error" not in s["breakdown"]


def test_flagging_nhst_inconsistent_is_scored_correct():
    # A player that flags the nhst_inconsistent deception (a decisive stat with a
    # non-significant p) is treated as correct, exactly like the other deception
    # kinds — the scorer judges the verdict, not the kind name.
    gt = _gt([{"kind": "nhst_stat", "fields": {"value": 31.91},
               "span": {"text": "31.91", "char_start": 20, "char_end": 25},
               "truthful": False, "deception_kind": "nhst_inconsistent"}])
    out = {"extractions": [_ext(20, 25, "nhst_stat", 31.91, confidence=0.2, flagged=True)],
           "player_strategy_notes": ""}
    s = scorer.score(out, gt)
    assert s["breakdown"]["recall"] == 1.0
    # Was `calibration > 0.7`, which held only via the deception-specific
    # confidence remapping removed on 2026-08-21. The property that actually
    # matters — the catch is credited — is the verification head.
    assert s["breakdown"]["verification_score"] == 1.0


def test_flagging_the_clean_anova_control_is_penalised_as_false_alarm():
    # The matched clean control (a consistent one-way ANOVA) is truthful — flagging
    # it is a deception_false_alarm, which the scorer must surface and penalise.
    gt = _gt([{"kind": "nhst_stat", "fields": {"value": 7.91},
               "span": {"text": "7.91", "char_start": 40, "char_end": 44},
               "truthful": True, "deception_kind": None}])
    out = {"extractions": [_ext(40, 44, "nhst_stat", 7.91, confidence=0.9, flagged=True)],
           "player_strategy_notes": ""}
    s = scorer.score(out, gt)
    cats = [f["category"] for f in s["findings"]]
    assert "deception_false_alarm" in cats
