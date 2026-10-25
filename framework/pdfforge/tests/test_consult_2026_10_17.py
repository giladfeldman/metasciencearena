"""Regression tests for the defects the 2026-10-17 tier-2 consult (Sonnet) found in the
PDF-universe answer keys (docs/consults/2026-10-17-pdf-universe-gold/round/01-sonnet.md).
Each was watched FAILING before its fix.
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

from framework.pdfforge import statgen as g
from framework.pdfforge.printing import print_record

REPO = Path(__file__).resolve().parents[3]
# stats-from-pdf-v1 is a private arena: the public mirror ships no scorer for it.
_needs_scorer = pytest.mark.skipif(not (REPO / "arenas/stats-from-pdf-v1/scorer.py").exists(),
                                   reason="stats-from-pdf-v1 is private, not in the public mirror")


def _scorer():
    spec = importlib.util.spec_from_file_location("sfp_sc", REPO / "arenas/stats-from-pdf-v1/scorer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


# Finding 3: statcheck's default pEqualAlphaSig = TRUE -- a printed p = .05 is significant.
def test_p_equal_alpha_counts_as_significant():
    # t(40) = 1.70 -> two-tailed p ~ .097: printed "p = .05" claims significance it does not
    # have, so with p = .05 significant this is a decision error.
    v = g.verdict("t", "1.70", 40, None, "=", ".05")
    assert v["consistent"] is False and v["decision_error"] is True


# Finding 2: a printed statistic whose rounding range straddles the .05 critical value is
# rule-dependent (strict point vs statcheck range); the gold must not take a side.
def test_boundary_straddling_result_is_ambiguous():
    v = g.verdict("t", "2.05", 27, None, "<", ".05")
    assert v.get("ambiguous") is True and v["consistent"] is None and v["decision_error"] is None


# Finding 1: a clean Welch record's effect size must agree with its printed t.
def test_welch_records_have_equal_groups_so_d_matches_t():
    import math
    from framework.pdfforge import content
    for s in range(30):
        rec = content._FAMILIES[1][1](__import__("numpy").random.default_rng(s), s)
        assert rec.design == "welch"
        n = rec.n // 2
        assert abs(rec.es_value / math.sqrt(2 / n) - rec.statistic) < 0.02 * max(1.0, abs(rec.statistic))


# Finding 7: a planted halved p must be visibly different from the clean print.
def test_r04_never_prints_the_clean_text():
    for s in range(200):
        clean = print_record(g.simulate_independent_t(s, 40, 40, 1.2), None)
        planted = print_record(g.simulate_independent_t(s, 40, 40, 1.2), "R04", random.Random(s))
        if planted.planted == "R04":
            assert planted.printed["text"] != clean.printed["text"]
            assert planted.verdict["consistent"] is False


# Finding 9: an "estimate outside its CI" plant must be visible in the PRINTED numbers.
def test_r06b_is_visible_after_rounding():
    for s in range(200):
        r = print_record(g.simulate_independent_t(s, 300, 300, 0.12), "R06b", random.Random(s))
        if r.planted == "R06b":
            lo, hi = (float(x) for x in r.printed["ci_txt"])
            assert not (lo <= float(r.printed["es_txt"]) <= hi)


# Finding 11: tied statistics must not swap verdicts between gold records.
@_needs_scorer
def test_tied_statistics_match_by_p_too():
    sc = _scorer()
    gold = {"gold_stats": [
        {"family": "t", "df1": 28, "df2": None, "printed": {"stat_txt": "2.31", "p_txt": ".028", "p_rel": "="},
         "verdict": {"consistent": True, "decision_error": False}},
        {"family": "t", "df1": 28, "df2": None, "printed": {"stat_txt": "2.31", "p_txt": ".040", "p_rel": "="},
         "verdict": {"consistent": False, "decision_error": False}}]}
    out = {"results": [{"test_type": "t", "df1": 28, "statistic": 2.31, "p_reported": 0.040, "verdict": "inconsistent"},
                       {"test_type": "t", "df1": 28, "statistic": 2.31, "p_reported": 0.028, "verdict": "consistent"}]}
    b = sc.score(out, gold)["breakdown"]
    assert b["false_alarms"] == 0 and b["verification_score"] == 1.0


# Finding 15: a non-string verdict must not crash scoring.
@_needs_scorer
def test_non_string_verdict_does_not_crash():
    sc = _scorer()
    gold = {"gold_stats": [{"family": "t", "df1": 28, "df2": None, "printed": {"stat_txt": "2.31", "p_txt": ".028", "p_rel": "="},
                            "verdict": {"consistent": True, "decision_error": False}}]}
    sc.score({"results": [{"test_type": "t", "df1": 28, "statistic": 2.31, "verdict": True}]}, gold)


# Finding 17: decision-error credit needs a decision_error verdict, not any inconsistency.
@_needs_scorer
def test_decision_error_credit_requires_a_decision_error_verdict():
    sc = _scorer()
    gold = {"gold_stats": [{"family": "F", "df1": 1, "df2": 70, "printed": {"stat_txt": "3.10", "p_txt": ".05", "p_rel": "<"},
                            "verdict": {"consistent": False, "decision_error": True}}]}
    b = sc.score({"results": [{"test_type": "F", "df1": 1, "df2": 70, "statistic": 3.10, "verdict": "inconsistent"}]}, gold)["breakdown"]
    assert b["decision_errors_caught"] == 0
    b = sc.score({"results": [{"test_type": "F", "df1": 1, "df2": 70, "statistic": 3.10, "verdict": "decision_error"}]}, gold)["breakdown"]
    assert b["decision_errors_caught"] == 1


# Finding 12: a document with no findable statistics still rewards silence and punishes invention.
@_needs_scorer
def test_no_findable_gold_scores_silence_and_invention():
    sc = _scorer()
    assert sc.score({"results": []}, {"gold_stats": []})["primary"] == 1.0
    assert sc.score({"results": [{"test_type": "t", "df1": 3, "statistic": 9.9}]}, {"gold_stats": []})["primary"] == 0.0
