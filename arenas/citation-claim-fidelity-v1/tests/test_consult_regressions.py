"""Regressions from the tier-3 consult of 2026-10-14 (Sonnet + Sol), arena side.

Each test was run against the pre-fix code (commit 69e703d) and failed there.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ARENA = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"ccf_reg_{name}", ARENA / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scorer = _load("scorer")
bars = _load("bars")
ABSTRACT = "Students in the restricted-sleep condition recalled fewer words than controls."
GT1 = {"abstract": ABSTRACT, "items": [{"id": 1, "gold": "silent", "type": "ORIG-silent"}]}

#: sha256 of the quote-parity fixture; citeclaim's tests/consult.test.ts pins the same value.
QUOTE_PARITY_SHA256 = "d1e4d73d74cb52269f79c0ac17299bf41512ffa820199f67b8b5165ec393e28c"


# ------------------------------------------------------------------ ids
@pytest.mark.parametrize("bad_id", [1.9, True, "1e0", "01", " 1"])
def test_non_canonical_ids_are_not_item_one(bad_id):
    out = {"items": [{"id": bad_id, "verdict": "silent", "evidence_quote": ""}]}
    assert scorer.score(out, GT1)["primary"] == 0.0


def test_zero_based_numbering_scores_nothing_instead_of_shifting():
    gt = {"abstract": ABSTRACT, "items": [{"id": 1, "gold": "silent", "type": "x"},
                                          {"id": 2, "gold": "silent", "type": "x"}]}
    out = {"items": [{"id": 0, "verdict": "silent"}, {"id": 1, "verdict": "silent"}]}
    assert scorer.score(out, gt)["primary"] == 0.0


def test_a_non_string_quote_is_no_verdict_not_a_crash():
    gt = {"abstract": ABSTRACT, "items": [{"id": 1, "gold": "contradicts", "type": "A4"},
                                          {"id": 2, "gold": "silent", "type": "x"}]}
    out = {"items": [{"id": 1, "verdict": "contradicts", "evidence_quote": ["recalled fewer words"]},
                     {"id": 2, "verdict": "silent", "evidence_quote": ""}]}
    s = scorer.score(out, gt)
    assert s["primary"] == 0.5  # the silent item still counts; only the bad item is lost


# ------------------------------------------------------------------ quote parity
def test_quote_parity_fixture_is_the_shared_copy():
    raw = (ARENA / "tests" / "quote_parity.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == QUOTE_PARITY_SHA256


def test_quote_guard_matches_the_shared_fixture():
    cases = json.loads((ARENA / "tests" / "quote_parity.json").read_text(encoding="utf-8"))["cases"]
    wrong = [(i, c["quote"]) for i, c in enumerate(cases)
             if scorer.quote_in_abstract(c["abstract"], c["quote"]) != c["expected"]]
    assert wrong == []


# ------------------------------------------------------------------ bars
def _items(n_clean=22, types=(("A1", 15), ("B3", 15)), verdict=lambda t: "silent"):
    items = [{"type": "ORIG-consistent", "gold": "consistent", "verdict": verdict("clean")} for _ in range(n_clean)]
    for t, n in types:
        gold = "not_consistent" if t in ("A1", "B3") else "contradicts"
        items += [{"type": t, "gold": gold, "verdict": verdict(t)} for _ in range(n)]
    return items


def test_an_all_silent_player_cannot_pass_when_contradiction_types_are_too_few():
    """Sol's counterexample: 22 clean + 15 A1 + 15 B3, all 'silent' -> passed:true."""
    report = bars.evaluate(_items(), grouping={"mean_difference": 0.0, "ci95_lower": 0.0, "n": 50})
    assert report["passed"] is False


def test_a_missing_grouping_result_fails_rather_than_skips():
    items = _items(types=(("A4", 20), ("A2", 20), ("C1", 20), ("C3", 20), ("A1", 20), ("B3", 20)),
                   verdict=lambda t: "consistent" if t == "clean" else "contradicts")
    for it in items:
        if it["verdict"] in ("consistent", "contradicts"):
            pass
    report = bars.evaluate(items)
    assert report["passed"] is False
    assert any(c["bar"].startswith("grouping") and c["pass"] is False for c in report["checks"])


def test_contradicts_on_an_abstract_silent_sentence_counts_as_a_false_suggestion():
    items = [{"type": "ORIG-silent", "gold": "silent", "verdict": "contradicts"} for _ in range(30)]
    report = bars.evaluate(items, grouping={"mean_difference": 0.0, "ci95_lower": 0.0, "n": 30})
    assert any(c["bar"].startswith("silent_false_alarm") and c["pass"] is False for c in report["checks"])


def test_coverage_is_required_within_each_gold_class():
    """A model that withholds verdicts only on clean items must not hide false alarms."""
    items = [{"type": "ORIG-consistent", "gold": "consistent", "verdict": None} for _ in range(10)]
    items += [{"type": "ORIG-consistent", "gold": "consistent", "verdict": "consistent"} for _ in range(10)]
    items += [{"type": "A4", "gold": "contradicts", "verdict": "contradicts"} for _ in range(100)]
    report = bars.evaluate(items, grouping={"mean_difference": 0.0, "ci95_lower": 0.0, "n": 30})
    assert any(c["bar"] == "coverage_consistent" and c["pass"] is False for c in report["checks"])


def test_false_alarm_upper_bound_accounts_for_items_sharing_an_abstract():
    """Clean items cluster by abstract; the bound must not be narrower than a
    cluster bootstrap's."""
    items = []
    for a in range(10):  # 10 abstracts, 4 clean items each; 2 abstracts carry all the alarms
        v = "contradicts" if a < 1 else "consistent"
        items += [{"type": "ORIG-consistent", "gold": "consistent", "verdict": v, "cluster": f"a{a}"}
                  for _ in range(4)]
    report = bars.evaluate(items, grouping={"mean_difference": 0.0, "ci95_lower": 0.0, "n": 30})
    ub = next(c for c in report["checks"] if c["bar"] == "clean_false_alarm_upper95")
    wilson_hi = bars.wilson(4, 40)[1]
    assert ub["value"] >= round(wilson_hi, 4)
    assert "cluster" in ub["detail"]


# ------------------------------------------------------------------ held-out gold hash
def test_held_out_gold_hash_cannot_be_recomputed_without_the_private_seed(tmp_path, monkeypatch):
    """Sol: instrument.gold_sha256 of an unsalted held-out gold (which contains the
    abstract) lets anyone with the public abstract confirm membership."""
    from framework.runner import _canonical_sha256
    generator = _load("generator")
    base = tmp_path / "ground_truth" / "citeclaim-2026-10"
    (base / "build_v1").mkdir(parents=True)
    task = {"task_key": "k1", "abstract_doi": "10.9/x", "title": "T", "year": 2020, "abstract": "An abstract. " * 30,
            "items": [{"item_id": "i", "sentence": "S.", "type": "ORIG-consistent", "gold": "consistent"}]}
    p = base / "build_v1" / "tasks_v1.jsonl"
    p.write_text(json.dumps(task) + "\n", encoding="utf-8")
    (base / "MANIFEST.json").write_text(json.dumps({"build_v1/tasks_v1.jsonl": {
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}}), encoding="utf-8")
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(tmp_path))
    hashes = []
    for seed in (11, 22):
        env = next(generator.generate("v1", seed, split="private"))
        gt = generator.ground_truth(env["task_id"])
        assert "salt" in gt
        hashes.append(_canonical_sha256(gt))
    unsalted = _canonical_sha256({"abstract": task["abstract"],
                                  "items": [{"id": 1, "gold": "consistent", "type": "ORIG-consistent", "traps": []}]})
    assert hashes[0] != hashes[1] and unsalted not in hashes
