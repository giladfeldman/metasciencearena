"""framework.paperforge: numbers are computed, never written; prose is guarded; spans are exact."""
from __future__ import annotations

import random

import numpy as np
import pytest
from scipy import stats

from framework.paperforge import apa
from framework.paperforge.docbuild import Doc, References
from framework.paperforge.prose import ProseRequest, parse_blocks, violations, write_prose
from framework.paperforge.simulate import cronbach_alpha, simulate
from framework.paperforge.study import Effects, Plan, Study, power_analysis

TOPIC = {
    "id": "t", "field": "social psychology", "platform": "Prolific", "alt_platform": "an undergraduate participant pool",
    "treatment": {"label": "x", "description": "d"},
    "primary": {"name": "A Scale", "abbrev": "AS", "k": 6, "item": "i", "anchors": "1-7", "weak_item": 3},
    "secondary": {"name": "B Scale", "abbrev": "BS", "k": 5, "item": "i", "anchors": "1-7", "weak_item": 2},
}


def _study(**actual):
    pw = power_analysis(0.4, 0.8)
    plan = Plan(target_n=pw.target_n, recruited_n=pw.target_n)
    return Study(TOPIC, plan, plan.with_(**actual), Effects(0.4, 0.2, 0.2), pw)


def test_apa_formatting():
    assert apa.p_value(0.0234) == "p = .023"
    assert apa.p_value(0.0004) == "p < .001"
    assert apa.p_value(0.99996) == "p > .999"
    assert apa.t_test(2.314, 141.0, 0.022) == "t(141) = 2.31, p = .022"
    assert apa.t_test(2.314, 141.37, 0.022) == "t(141.37) = 2.31, p = .022"
    assert apa.bounded(0.734) == ".73" and apa.bounded(-0.25) == "-.25"
    assert apa.num(-0.001) == "0.00"  # no negative zero
    with pytest.raises(ValueError):
        apa.p_value(1.2)


def test_power_analysis_matches_gpower():
    # G*Power 3.1, t-test two independent means, d = 0.5, alpha .05 two-tailed, power .80 -> 64 per group.
    assert power_analysis(0.5, 0.80).n_per_group == 64
    # d = 0.4, power .80 -> 100 per group.
    assert power_analysis(0.4, 0.80).n_per_group == 100


def test_cronbach_alpha_known_value():
    items = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 5]], dtype=float)
    k = 3
    manual = k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum() / items.sum(axis=1).var(ddof=1))
    assert cronbach_alpha(items) == pytest.approx(manual)


def test_simulation_is_deterministic_and_follows_the_actual_plan():
    a, b = simulate(_study(), 11), simulate(_study(), 11)
    assert a == b
    strict = simulate(_study(exclusion="attn1"), 11)
    # Same participants, stricter rule: at least as many exclusions.
    assert strict["n_excl_attention"] >= a["n_excl_attention"]
    impute = simulate(_study(missing="mean_impute"), 11)
    assert impute["n_excl_missing"] == 0 and a["n_excl_missing"] == a["n_with_missing"]
    alt = simulate(_study(randomization="alternation"), 11)
    assert abs(alt["n_treat"] - alt["n_ctrl"]) <= alt["n_excl_attention"] + alt["n_excl_missing"] + 1
    one = simulate(_study(tails=1), 11)
    assert one["test_main"]["t"] == pytest.approx(a["test_main"]["t"])
    assert one["test_main"]["p"] == pytest.approx(a["test_main"]["p"] / 2)
    anc = simulate(_study(model="ancova"), 11)
    assert anc["test_main"]["kind"] == "ancova" and anc["test_main"]["df2"] == anc["n_final"] - 3


def test_reported_welch_statistics_are_internally_consistent():
    r = simulate(_study(), 5)["test_main"]
    t, dof = float(apa.num(r["t"])), float(apa.df(r["df"]))
    p_recomputed = 2 * stats.t.sf(abs(t), dof)
    printed = apa.p_value(r["p"])
    assert printed == apa.p_value(p_recomputed) or abs(p_recomputed - r["p"]) < 0.002


def test_prose_guard():
    assert violations("INTRODUCTION", "Prior work [[CITE]] shows gratitude matters.") == []
    for bad in ("In 2019 a study", "we excluded careless people", "Participants were randomly shown",
                "The primary aim", "This was preregistered", "Recruited on Prolific", "a two-tailed test",
                "scale items were averaged", "We hypothesised that"):
        assert violations("INTRODUCTION", bad), bad
    # Ordinary words that contain a forbidden stem are not rejections (measured false rejections).
    for ok in ("An alternative account", "exclusively online", "a detailed review", "a reliable pattern"):
        assert violations("INTRODUCTION", ok) == [], ok
    # The discussion may interpret hypotheses and significance, but never gets a number.
    assert violations("DISCUSSION", "Our hypothesis was supported by a significant effect.") == []
    assert violations("DISCUSSION", "an effect of 0.4") != []


def test_write_prose_retries_then_accepts_and_records_rejections():
    replies = iter(["===A===\nThere were 3 groups.\n===B===\nFine text.",
                    "===A===\nThere were three groups.\n===B===\nFine text."])
    entry = write_prose(ProseRequest("k", "brief", {"A": "x", "B": "y"}),
                        caller=lambda prompt, model: (next(replies), {"model_returned": ["m"]}))
    assert entry["attempts"] == 2 and entry["blocks"]["A"] == "There were three groups."
    assert "digit" in entry["rejections"][0][0]


def test_write_prose_drops_few_bad_sentences_as_last_resort():
    good = " ".join(f"Sentence number {w} is fine." for w in
                    ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"))
    reply = f"===A===\n{good} But we excluded people."
    entry = write_prose(ProseRequest("k", "b", {"A": "x"}), max_attempts=2,
                        caller=lambda p, m: (reply, {"model_returned": ["m"]}))
    assert "excluded" not in entry["blocks"]["A"]
    assert entry["dropped_sentences"] == [{"block": "A", "sentence": "But we excluded people."}]


def test_parse_blocks_requires_every_block():
    with pytest.raises(ValueError):
        parse_blocks("===A===\ntext", ["A", "B"])


def test_doc_spans_are_exact():
    d = Doc().heading("Title").para("Untagged.", ("Tagged one.", "x")).raw("Raw").para(("Tagged two.", "y"))
    text = d.render()
    for s in d.spans:
        assert text[s["start"]:s["end"]] == s["text"]
    assert [s["tag"] for s in d.spans] == ["x", "y"]


def test_references_are_synthetic_and_deterministic():
    def build():
        r = References(random.Random(3), ["gratitude", "wellbeing"])
        text = r.fill("As shown [[CITE]] [[CITE]], and later [[CITE]].")
        return text, r.render()
    a, b = build(), build()
    assert a == b
    assert "[[CITE]]" not in a[0] and a[0].count("(") == 2
    assert all("https://doi.org/10.5555/synth." in ref for ref in a[1])  # Crossref's example prefix


def test_a_sentence_can_be_evidence_for_several_dimensions():
    d = Doc().para(("t(10) = 2.00, p = .07.", ("measured", "model")))
    text = d.render()
    assert {s["tag"] for s in d.spans} == {"measured", "model"}
    assert all(text[s["start"]:s["end"]] == s["text"] for s in d.spans)


def test_discussion_may_say_significant_but_never_the_tails():
    assert violations("DISCUSSION", "The effect was significant.") == []
    assert violations("DISCUSSION", "the one-tailed test at an alpha of point oh five")
    assert violations("INTRODUCTION", "entries did not reflect genuine engagement")
    assert violations("INTRODUCTION", "every other participant joined the programme")
    assert violations("INTRODUCTION", "to encourage genuine engagement with the task") == []
