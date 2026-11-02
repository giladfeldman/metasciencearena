"""RegCheck -> prereg-deviation-fullpaper-v1 (three-way) mapping, end to end through the scorer."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from players.regcheck_shim.regcheck_to_runrecords import (
    FULLPAPER_DIMENSIONS_YAML,
    load_dimension_kinds,
    regcheck_items_to_threeway_output,
)

REPO = Path(__file__).resolve().parents[3]
ARENA = REPO / "arenas" / "prereg-deviation-fullpaper-v1"
GOLD = ARENA / "task_sets" / "v1" / "_ground_truth.json"
needs_gold = pytest.mark.skipif(not GOLD.is_file(), reason="committed gold is NEVER_PUBLISH (public-mirror checkout)")
#: The arena is a draft and not in the public mirror yet, while this test file is; there the
#: whole module skips with this reason instead of failing on an absent catalog.
pytestmark = pytest.mark.skipif(not FULLPAPER_DIMENSIONS_YAML.is_file(),
                                reason="prereg-deviation-fullpaper-v1 not in this checkout (public mirror without it)")


def _scorer():
    spec = importlib.util.spec_from_file_location("_pdfp_scorer_shim_t", ARENA / "scorer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_ANSWER = {"deviation": "yes", "no_deviation": "no", "insufficient_evidence": "missing"}


def _regcheck_items(gold: dict) -> list[dict]:
    """What a perfect RegCheck run returns: its own field names, quotes joined by blank lines."""
    return [{
        "dimension": g["dimension"],
        "deviation_judgement": _ANSWER[g["judgement"]],
        "paper_content_quotes": "\n\n".join([e["text"] for e in g["paper_evidence"][:1]]
                                            + ["A second, unrelated quote from the paper."]),
        "registration_content_quotes": "",
    } for g in gold["dimensions"]]


def test_each_regcheck_judgement_maps_to_its_three_way_class():
    items = [{"dimension": "hypotheses", "deviation_judgement": "Yes"},
             {"dimension": "sample_size", "deviation_judgement": "no"},
             {"dimension": "indices", "deviation_judgement": "missing"},
             {"dimension": "randomization", "deviation_judgement": "unclear"}]
    out = {d["dimension"]: d for d in regcheck_items_to_threeway_output(items)["deviations"]}
    assert out["hypotheses"]["judgement"] == "deviation" and out["hypotheses"]["deviation"] is True
    assert out["hypotheses"]["deviation_kind"] == "hypothesis_change"
    assert out["sample_size"]["judgement"] == "no_deviation" and out["sample_size"]["deviation"] is False
    assert out["indices"]["judgement"] == "insufficient_evidence" and out["indices"]["deviation"] is False
    assert "randomization" not in out  # an unknown answer is never guessed into a class
    assert all(d["disclosed"] is None for d in out.values())  # RegCheck does not code disclosure


def test_paper_quote_is_the_first_of_regcheck_s_joined_quotes():
    out = regcheck_items_to_threeway_output([{
        "dimension": "sample_size", "deviation_judgement": "no",
        "paper_content_quotes": "\n\nFirst quote here.\n\nSecond quote here.",
        "registration_content_quotes": None}])["deviations"][0]
    assert out["paper_quote"] == "First quote here." and out["prereg_quote"] is None


def test_catalog_kinds_are_the_full_paper_catalog():
    kinds = load_dimension_kinds(FULLPAPER_DIMENSIONS_YAML)
    assert len(kinds) == 11 and kinds["data_source"] == "data_source_change"


@needs_gold
def test_a_perfect_regcheck_run_scores_one_and_validates():
    scorer = _scorer()
    schema = json.loads((ARENA / "schemas" / "output.schema.json").read_text(encoding="utf-8"))
    committed = json.loads(GOLD.read_text(encoding="utf-8"))
    for tid, rec in committed.items():
        gold = rec["ground_truth"]
        out = regcheck_items_to_threeway_output(_regcheck_items(gold))
        Draft202012Validator(schema).validate(out)
        s = scorer.score(out, gold)
        assert s["breakdown"]["judgement_macro_f1"] == pytest.approx(1.0), tid
        assert s["breakdown"]["evidence_grounding"] == pytest.approx(1.0), tid
        assert s["breakdown"]["disclosure_accuracy"] in (None, 0.0), tid  # never claimed


@needs_gold
def test_a_two_way_collapse_would_lose_the_insufficient_evidence_class():
    """Control: the old v1 mapping (missing -> no deviation) must score worse here."""
    scorer = _scorer()
    gold = next(r["ground_truth"] for r in json.loads(GOLD.read_text(encoding="utf-8")).values()
                if any(g["judgement"] == "insufficient_evidence" for g in r["ground_truth"]["dimensions"]))
    items = _regcheck_items(gold)
    for it in items:
        if it["deviation_judgement"] == "missing":
            it["deviation_judgement"] = "no"
    assert scorer.score(regcheck_items_to_threeway_output(items), gold)["breakdown"]["judgement_macro_f1"] < 1.0


def test_regcheck_chunk_tags_are_not_part_of_the_quote():
    # Found by the 2026-10-30 fullpaper smoke run: RegCheck 50c8017 prefixes every
    # retrieved chunk with "[PAPER_0032, relevance_score=0.186] ". That tag is not paper
    # text, so a quote carrying it can never be found in the paper (RegCheck strips it
    # itself with _EVIDENCE_ID_RE in backend/services/comparisons.py).
    from players.regcheck_shim.regcheck_to_runrecords import (
        regcheck_items_to_threeway_output,
    )
    out = regcheck_items_to_threeway_output([{
        "dimension": "hypotheses", "deviation_judgement": "yes",
        "paper_content_quotes": "[PAPER_0032, relevance_score=0.186] We tested two hypotheses.\n\n[PAPER_0040] Other.",
        "registration_content_quotes": "[PREREG_0003, relevance_score=1.000] H1 predicts more.",
    }], {"hypotheses": "hypothesis_change"})
    d = out["deviations"][0]
    assert d["paper_quote"] == "We tested two hypotheses."
    assert d["prereg_quote"] == "H1 predicts more."
