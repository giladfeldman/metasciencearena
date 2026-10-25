"""Stage E panel tool: eligibility mirror, answer parsing, pre-registered sheet selection (fictional data)."""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
_spec = importlib.util.spec_from_file_location("ccf_stage_e_panel", ARENA / "tools" / "stage_e_panel.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

ABS = {"10.9/x": {"doi": "10.9/x", "abstract": "A" * 250}}


def _c(**kw):
    base = {"id": "c1", "referenceKey": "10.9/x", "doi": "10.9/x", "linkStatus": "matched",
            "sourceCount": 1, "sentence": "S (Okafor, 2031)."}
    base.update(kw)
    return base


def test_eligibility_mirrors_the_library():
    assert P.eligible(_c(), ABS)
    assert not P.eligible(_c(linkStatus="unmatched"), ABS)
    assert not P.eligible(_c(sourceCount=2), ABS)
    assert not P.eligible(_c(sentence=None), ABS)
    assert not P.eligible(_c(doi="10.9/other"), ABS)
    assert not P.eligible(_c(), {"10.9/x": {"doi": "10.9/x", "abstract": "A" * 199}})
    assert not P.eligible(_c(), {"10.9/x": {"doi": "10.9/y", "abstract": "A" * 250}})
    assert P.eligible(_c(doi="https://doi.org/10.9/X"), ABS)


def test_extract_json_object_like_the_library():
    assert P.extract_json_object('prose ```json\n{"items": []}\n``` tail') == {"items": []}
    assert P.extract_json_object('x {"items": [{"id": 1}]} y') == {"items": [{"id": 1}]}
    try:
        P.extract_json_object("no json")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_prompt_is_the_library_prompt_file():
    lib = ARENA / "prompts" / "v1.md"
    assert P.PROMPT == lib.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_largest_remainder_sums_to_total():
    a = P._largest_remainder({"psychology": 50, "economics": 7, "finance": 3}, 20)
    assert sum(a.values()) == 20 and a["psychology"] == 17


def _row(cid, unanimous, free):
    return {"citation_id": cid, "unanimous": unanimous, "free_model": {"verdict": free}}


def test_sheet_selection_follows_the_preregistration():
    rows = ([_row(f"n{i}", False, "silent") for i in range(5)]
            + [_row(f"u{i}", True, "consistent") for i in range(40)]
            + [_row(f"k{i}", True, "contradicts") for i in range(3)])
    disc = {r["citation_id"]: ("economics" if r["citation_id"].endswith("7") else "psychology") for r in rows}
    s1 = P.select_for_owner(rows, disc)
    s2 = P.select_for_owner(rows, disc)
    assert s1 == s2  # seeded, deterministic
    strata = Counter(x for v in s1.values() for x in v)
    assert strata["a_no_consensus"] == 5               # every non-consensus item
    assert strata["b_unanimous_audit"] == 5            # ceil(10% of 43)
    assert strata["c_free_contradicts"] == 3           # every free-model suggestion
    picked = [c for c, v in s1.items() if "c_free_consistent_sample" in v]
    assert len(picked) == 20 and not any("b_unanimous_audit" in s1[c] and len(s1[c]) > 1 for c in picked)
