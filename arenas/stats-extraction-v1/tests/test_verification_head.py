"""RED-first tests for the verification head (2026-08-21).

Written against the UNFIXED scorer and watched fail before the fix, per the
portfolio rule that a regression test which has never failed is decoration.

The defect these pin: `stats-extraction-v1` told players to recompute p and flag
inconsistencies (`players/prompts/stats_extraction.txt:3,23-24`) while
`generator.py:236-238` drew p independently of the statistic, so 37 of 84
truthful `nhst_stat` gold items were genuine statcheck decision errors labelled
`truthful: true`. Flagging one was charged `deception_false_alarm` and, because
`confidence` is glued to the flag, took a task from primary 1.000 to 0.400.

Probed against live effectcheck 0.7.6 on 2026-08-21
(`docs/reports/evidence/2026-08-21-escimate-0.7.6-probe.md`):
`There was a correlation, r(129) = -0.82, p = 0.393.` returns
`decision_error=True, reason='reported_ns_computed_sig', p_computed=4.7e-33`.
The tool is right; the benchmark charged it.
"""
from pathlib import Path
import importlib.util
import sys

ARENA_DIR = Path(__file__).resolve().parents[1]


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, ARENA_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


scorer = _load("_stats_extraction_v1_scorer_vh", "scorer.py")
generator = _load("_stats_extraction_v1_generator_vh", "generator.py")


def _gold(**kw):
    item = {
        "kind": "nhst_stat",
        "span": {"text": "5.41", "char_start": 10, "char_end": 14},
        "fields": {"value": 5.41},
        "truthful": True,
        "deception_kind": None,
    }
    item.update(kw)
    return {"items": [item]}


def _ext(verdict=None, flagged=False, value=5.41, confidence=1.0):
    e = {
        "span": {"text": "5.41", "char_start": 10, "char_end": 14},
        "kind": "nhst_stat",
        "fields": {"value": value},
        "confidence": confidence,
        "flagged_suspicious": flagged,
    }
    if verdict is not None:
        e["verdict"] = verdict
    return {"extractions": [e]}


# --- 1. the headline defect --------------------------------------------------

def test_flagging_an_internally_inconsistent_truthful_item_is_not_a_false_alarm():
    """The item is truthfully transcribed AND statistically impossible. A checker
    that flags it is RIGHT. Before the fix this emitted `deception_false_alarm`
    and dropped primary from 1.000 to 0.400."""
    gt = _gold(truthful=True, checkable=True, expected_verdict="inconsistent")
    res = scorer.score(_ext(verdict="inconsistent", flagged=True, confidence=0.4), gt)
    cats = [f["category"] for f in res["findings"]]
    assert "deception_false_alarm" not in cats, cats
    assert res["primary"] == 1.0, res["breakdown"]


def test_calling_an_inconsistent_item_clean_is_a_miss():
    gt = _gold(truthful=True, checkable=True, expected_verdict="inconsistent")
    res = scorer.score(_ext(verdict="ok", flagged=False), gt)
    assert "deception_missed" in [f["category"] for f in res["findings"]]
    assert res["breakdown"]["verification_score"] == 0.0


def test_flagging_a_genuinely_clean_item_is_still_a_false_alarm():
    """The fix must not make flagging free."""
    gt = _gold(truthful=True, checkable=True, expected_verdict="ok")
    res = scorer.score(_ext(verdict="inconsistent", flagged=True), gt)
    assert "deception_false_alarm" in [f["category"] for f in res["findings"]]
    assert res["breakdown"]["verification_score"] == 0.0


# --- 2. abstention is not accusation ----------------------------------------

def test_unverifiable_is_neither_an_accusation_nor_a_miss():
    """`insufficient_data` means the tool cannot check this, and the old adapter
    scored it identically to a fabrication claim.

    AMENDED 2026-08-27. The claim this test is named for is unchanged and still
    asserted: declining is NOT charged as a false alarm. What changed is the two
    lines below it, which asserted that declining an answerable item left the
    denominator and returned the MAXIMUM verification score. Those two lines
    pinned a defect rather than a requirement — with them in place, answering
    "unverifiable" to every item scored 1.0000 and beat every real verifier
    (always-"ok" 0.8078, measured over the 36 revealed v3 tasks). A declined item
    now stays in the denominator as a non-hit. See
    `test_abstention_dominance.py` for the full measurement.
    """
    gt = _gold(truthful=True, checkable=True, expected_verdict="ok")
    res = scorer.score(_ext(verdict="unverifiable", flagged=True), gt)
    cats = [f["category"] for f in res["findings"]]
    assert "deception_false_alarm" not in cats, cats
    assert "abstain_on_checkable" in cats, cats
    # Counted, but as a non-hit rather than as an accusation.
    assert res["breakdown"]["verification_score"] == 0.0
    assert res["breakdown"]["n_verifiable"] == 1
    assert res["breakdown"]["verification_false_alarms"] == 0
    assert res["breakdown"]["verification_declined"] == 1


# --- 3. uncheckable gold leaves the verification denominator -----------------

def test_bare_effect_size_is_not_scored_for_verification():
    """A bare effect size with no test statistic has nothing to verify. Probed
    against live effectcheck 0.7.6 on 2026-08-21: all six bare forms the arena
    uses return ZERO rows. Scoring them as verification failures measures the
    tool outside its contract."""
    gt = {"items": [{
        "kind": "effect_size",
        "span": {"text": "2.64", "char_start": 0, "char_end": 4},
        "fields": {"value": 2.64},
        "truthful": True, "deception_kind": None,
        "checkable": False, "expected_verdict": "ok",
    }]}
    ext = {"extractions": [{
        "span": {"text": "2.64", "char_start": 0, "char_end": 4},
        "kind": "effect_size", "fields": {"value": 2.64},
        "confidence": 1.0, "flagged_suspicious": False, "verdict": "unverifiable",
    }]}
    res = scorer.score(ext, gt)
    assert res["breakdown"]["n_verifiable"] == 0
    assert res["breakdown"]["verification_score"] == 1.0
    assert res["primary"] == 1.0


# --- 4. calibration must leave the headline ---------------------------------

def test_calibration_is_reported_but_never_multiplied_into_primary():
    """ESCImate's confidence was `{0.4 x53, 1.0 x48}` — a two-point mass the
    ADAPTER invented (`0.4 if suspicious else 1.0`). ECE over it measured
    framework/player_adapter.py, not the tool, and it was one of three
    multiplied terms in the published score."""
    gt = _gold(truthful=True, checkable=True, expected_verdict="ok")
    res = scorer.score(_ext(verdict="ok", flagged=False, confidence=0.4), gt)
    assert "calibration" in res["breakdown"]          # still reported
    assert res["primary"] == 1.0                      # but not in the product
    assert res["breakdown"]["calibration"] < 1.0      # and genuinely imperfect


# --- 5. back-compat: a player that never learned `verdict` ------------------

def test_absent_verdict_falls_back_to_flagged_suspicious():
    gt = _gold(truthful=True, checkable=True, expected_verdict="inconsistent")
    res = scorer.score(_ext(verdict=None, flagged=True), gt)
    assert res["breakdown"]["verification_score"] == 1.0


# --- 6. the gold itself ------------------------------------------------------

def test_no_truthful_nhst_gold_item_is_silently_alpha_crossing():
    """Every truthful NHST item must declare its own verification status. This
    is the property the generator docstring at :10-12 CLAIMED and the code at
    :236-238 explicitly did not provide."""
    tasks = list(generator.generate("v2", 0, "revealed"))
    missing = []
    for t in tasks:
        gt = generator.ground_truth(t["task_id"])
        for it in gt["items"]:
            if it["kind"] != "nhst_stat":
                continue
            if "expected_verdict" not in it or "checkable" not in it:
                missing.append((t["task_id"], it["fields"].get("value")))
    assert not missing, f"{len(missing)} nhst gold items carry no verification label"


def test_expected_verdict_agrees_with_a_recomputation_of_the_gold():
    """The label must be DERIVED, not asserted. Recompute it independently."""
    tasks = list(generator.generate("v2", 0, "revealed"))
    wrong = []
    for t in tasks:
        gt = generator.ground_truth(t["task_id"])
        for it in gt["items"]:
            if it["kind"] != "nhst_stat" or not it.get("checkable"):
                continue
            f = it["fields"]
            tp = generator._two_sided_p(f.get("test_type"), f.get("value"),
                                        f.get("df1"), f.get("df2"))
            if tp is None:
                continue
            crosses = (f["p"] < 0.05) != (tp < 0.05)
            want = "inconsistent" if crosses else "ok"
            # a planted deception may carry a more specific verdict
            if it.get("truthful", True) and it["expected_verdict"] != want:
                wrong.append((t["task_id"], f.get("value"), f.get("p"), round(tp, 6),
                              it["expected_verdict"], want))
    assert not wrong, wrong[:8]
