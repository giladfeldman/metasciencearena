"""The answer key of the PDF universe is a computation; these tests check it is the right one.

Two-sided: every CLEAN printed result must check out as consistent (else the gold
would accuse correct reporting), and every planted error must be caught for the
reason it was planted (else the gold would credit a checker for nothing).
"""
from __future__ import annotations

import random

import pytest
from scipy import stats

from framework.pdfforge import statgen as g
from framework.pdfforge.printing import print_record

SIMS = {
    "t_ind": lambda s: g.simulate_independent_t(s, 40, 42, 0.45),
    "t_welch": lambda s: g.simulate_independent_t(s, 30, 55, 0.4, welch=True),
    "t_paired": lambda s: g.simulate_paired_t(s, 35, 0.4),
    "F_oneway": lambda s: g.simulate_oneway_F(s, 3, 30, 0.3),
    "F_inter": lambda s: g.simulate_interaction_F(s, 25, 0.35),
    "chi2_2x2": lambda s: g.simulate_chi2(s, 200, 2, 2, 0.4),
    "chi2_2x3": lambda s: g.simulate_chi2(s, 300, 2, 3, 0.3),
    "r": lambda s: g.simulate_r(s, 80, 0.25),
    "z": lambda s: g.simulate_z_proportions(s, 120, 130, .4, .3),
    "b": lambda s: g.simulate_regression_b(s, 150, 0.2),
    "OR": lambda s: g.simulate_odds_ratio(s, 300, 0.5),
    "H": lambda s: g.simulate_kruskal(s, 3, 25, 0.5),
}
SEEDS = range(40)


@pytest.mark.parametrize("name", sorted(SIMS))
def test_clean_prints_are_consistent(name):
    for s in SEEDS:
        r = print_record(SIMS[name](s))
        if r.verdict.get("ambiguous"):
            # allowed ONLY where the printed statistic's rounding interval straddles .05
            lo, hi = r.verdict["recomputed_p_range"]
            assert lo <= 0.05 < hi, (name, s, r.printed["text"], r.verdict)
            continue
        assert r.verdict["consistent"] is True, (name, s, r.printed["text"], r.verdict)
        assert r.verdict["decision_error"] is False
        assert r.verdict["flags"] == [], (name, s, r.printed["text"], r.verdict)


@pytest.mark.parametrize("name", sorted(SIMS))
def test_planted_errors_are_detected_for_their_reason(name):
    for s in SEEDS:
        rng = random.Random(s)
        r1 = print_record(SIMS[name](s), "R01", rng)
        if r1.printed.get("plant_unavailable"):
            # Refused only when the rounding interval straddles .05 -- and then printed clean.
            lo, hi = r1.verdict["recomputed_p_range"]
            assert lo <= 0.05 < hi and r1.planted is None and r1.verdict.get("decision_error") is not True
        else:
            assert r1.verdict["decision_error"] is True, (name, s, r1.printed["text"], r1.verdict)
        r2 = print_record(SIMS[name](s), "R02", rng)
        assert r2.printed.get("plant_unavailable") is None
        assert r2.verdict["consistent"] is False and r2.verdict["decision_error"] in (False, None), \
            (name, s, r2.printed["text"], r2.verdict)
        r5 = print_record(SIMS[name](s), "R05a", rng)
        assert r5.verdict["consistent"] is True or r5.verdict.get("ambiguous"), (name, s, r5.printed["text"], r5.verdict)
        rec = SIMS[name](s)
        if rec.family in ("t", "b", "r", "z", "OR"):
            r5c = print_record(rec, "R05c", rng)
            assert r5c.verdict["consistent"] is True or r5c.verdict.get("ambiguous"), (name, s, r5c.printed["text"], r5c.verdict)
        rec = SIMS[name](s)
        if rec.ci_lo is not None:
            assert "ci_reversed" in print_record(rec, "R06a", rng).verdict["flags"]
            assert "estimate_outside_ci" in print_record(SIMS[name](s), "R06b", rng).verdict["flags"]


def test_truth_matches_scipy_for_a_hand_case():
    r = g.simulate_independent_t(7, 20, 20, 0.5)
    assert r.p == pytest.approx(2 * stats.t.sf(abs(r.statistic), r.df1), rel=1e-9)
    assert r.ci_lo < r.es_value < r.ci_hi


def test_verdict_uses_statcheck_rounding():
    # t(30) = 2.0 stands for [1.95, 2.05): two-tailed p in (.0495, .0602), so p = .049 is
    # consistent only through rounding (statcheck's rule), while the printed 2.0 at face
    # value gives p = .055 -- a decision error. Rule-dependent: the gold takes no side.
    v = g.verdict("t", "2.0", 30, None, "=", ".049")
    assert v["ambiguous"] is True and v["consistent"] is None and v["decision_error"] is None
    assert v["range_rule"]["consistent"] is True
    # Unambiguous: the whole rounding interval is below .05.
    assert g.verdict("t", "2.31", 40, None, "=", ".026")["consistent"] is True
    # A gross decision error: F(1, 70) = 3.10 gives p = .083, printed p < .05.
    v = g.verdict("F", "3.10", 1, 70, "<", ".05")
    assert v["consistent"] is False and v["decision_error"] is True
