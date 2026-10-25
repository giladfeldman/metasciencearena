"""Regressions from the tier-3 consult of 2026-10-15 (Sonnet + Sol) on task set v2.

Round: docs/consults/2026-10-15-citeclaim-v2-c1/. Each test was run against the
pre-fix code (commit c903843) and failed there.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]


def _load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, ARENA / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


analyze = _load("tools/analyze_run", "ccf_reg15_analyze")


def _revealed_run(tmp_path, *, duplicate=False, players=("p",), tamper_gold=False):
    """A run file over the revealed split, every item answered 'silent'."""
    from framework.runner import _canonical_sha256
    gen = analyze.generator
    recs = []
    for k, env in enumerate(gen.generate("v1", 0, split="revealed")):
        gt = gen.ground_truth(env["task_id"])
        out = {"items": [{"id": it["id"], "verdict": "silent", "evidence_quote": "", "reason": "r"}
                         for it in gt["items"]]}
        gold_hash = _canonical_sha256(gt) if not tamper_gold else "0" * 64
        recs.append({"task_id": env["task_id"], "run_id": f"r{k}", "player_id": players[k % len(players)],
                     "task_set_version": "v1", "output": out, "score": {"breakdown": {}},
                     "instrument": {"gold_sha256": gold_hash}})
    if duplicate:
        recs.append(dict(recs[0], run_id="dup"))
    p = tmp_path / "run.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return p


def _complete(run_file):
    items, meta = analyze.item_results(run_file, "revealed", 0)
    return analyze.summarize(items, meta, None)["complete"]


def test_a_clean_revealed_run_is_complete(tmp_path):
    """Control: the fixture itself is a complete run, so the failures below are the defects."""
    assert _complete(_revealed_run(tmp_path)) is True


def test_a_duplicated_record_makes_the_run_incomplete(tmp_path):
    """Sol #3 / Sonnet #5: a re-run appended to the file counted its items twice."""
    assert _complete(_revealed_run(tmp_path, duplicate=True)) is False


def test_records_from_two_players_make_the_run_incomplete(tmp_path):
    assert _complete(_revealed_run(tmp_path, players=("p", "q"))) is False


def test_a_record_whose_gold_hash_differs_from_todays_gold_is_not_scored(tmp_path):
    """Sonnet #4: verdicts were zipped onto regenerated gold without checking gold_sha256."""
    assert _complete(_revealed_run(tmp_path, tamper_gold=True)) is False


GOOD = {"player": "p", "task_set": "v2", "run_file_sha256": "abc", "n": 40, "n_expected": 40,
        "n_errored": 0, "n_missing_grouped": 0, "max_tasks": None, "mean_difference": 0.0,
        "ci95_lower": -0.02, "ci95_upper": 0.02, "n_tasks": 12}


def test_a_matching_complete_grouping_result_is_accepted():
    g, why = analyze.grouping_for_bars(dict(GOOD), player="p", task_set="v2", run_file_sha256="abc")
    assert g is not None and why is None


def test_grouping_results_that_do_not_belong_to_this_run_are_rejected():
    """Sol #4/#5, Sonnet #6: an errored, truncated or foreign probe satisfied the grouping bar."""
    for change in ({"n_errored": 3}, {"n": 39}, {"n_missing_grouped": 1}, {"max_tasks": 5},
                   {"player": "other"}, {"task_set": "v1"}, {"run_file_sha256": "zzz"}):
        g, why = analyze.grouping_for_bars({**GOOD, **change}, player="p", task_set="v2", run_file_sha256="abc")
        assert g is None and why, change
    legacy = {k: v for k, v in GOOD.items() if k not in ("player", "task_set", "run_file_sha256", "n_expected")}
    g, why = analyze.grouping_for_bars(legacy, player="p", task_set="v2", run_file_sha256="abc")
    assert g is None and why


def test_quota_probe_classifies_429s_from_structured_fields():
    """Sonnet #7 / Sol #8: a free-text 'daily' regex misread both directions."""
    probe = _load("tools/free_quota_probe", "ccf_reg15_probe")
    groq = ('{"error":{"message":"Rate limit reached for model `m` on tokens per %s (%s): Limit 200000, '
            'Used 199112, Requested 967.","type":"tokens","code":"rate_limit_exceeded"}}')
    assert probe.classify_429(groq % ("day", "TPD")) == "daily"
    assert probe.classify_429(groq % ("minute", "TPM")) == "minute"
    gem = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPer%sPerProjectPerModel-FreeTier", "quotaValue": "20"}]}]}}
    as_text = json.dumps(gem)
    assert probe.classify_429(as_text.replace("%s", "Day")) == "daily"
    assert probe.classify_429(as_text.replace("%s", "Minute")) == "minute"
    # Sol's counterexamples: prose mentioning "daily" is not a daily limit, and an
    # unrecognised daily message is "unknown" (the probe stops after 10), never "minute".
    assert probe.classify_429('{"error":{"message":"Per-minute limit reached; daily limit remains available"}}') == "unknown"
    assert probe.classify_429('{"error":{"message":"requests/day limit exceeded"}}') == "unknown"
    assert probe.classify_429("<html>busy</html>") == "unknown"


def _build(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(tmp_path))
    bt = _load("tools/build_tasks", "ccf_reg15_build")
    bt.BUILD = tmp_path / "b2"
    bt.BUILD.mkdir()
    seed = {"seed_id": "s1", "cited_doi": "10.1/x", "abstract": "We surveyed 312 adults; 45% agreed.",
            "sentence": "Smith [3] found that 45% of 312 adults agreed.", "marker": "3"}
    (bt.BUILD / "seeds.jsonl").write_text(json.dumps(seed) + "\n", encoding="utf-8")
    return bt


def _variant(bt, sentence):
    row = {"seed_id": "s1", "variants": [{"type": "C1", "sentence": sentence}]}
    (bt.BUILD / "perturbations_raw.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_a_c1_edit_must_change_only_numbers(tmp_path, monkeypatch):
    """Sol #6 / Sonnet #2: spelling a number out and flipping a verb passed as C1."""
    bt = _build(tmp_path, monkeypatch)
    _variant(bt, "Smith [3] found that forty-five percent of 312 adults disagreed.")
    assert bt.candidates_c1() == []
    _variant(bt, "Smith [3] found that 65% of 312 adults agreed.")
    assert len(bt.candidates_c1()) == 1


def test_a_cached_label_is_not_reused_for_a_changed_sentence(tmp_path, monkeypatch):
    """Sol #2 / Sonnet #3: labels keyed by item_id alone carried 'contradicts' to new text."""
    bt = _build(tmp_path, monkeypatch)
    _variant(bt, "Smith [3] found that 65% of 312 adults agreed.")
    item = bt.candidates_c1()[0]
    stale = {"item_id": item["item_id"], "label": "contradicts", "labeler": "sonnet",
             "sentence_sha256": bt._text_sha("Smith [3] found that 75% of 312 adults agreed.")}
    (bt.BUILD / "labels_items_sonnet.jsonl").write_text(json.dumps(stale) + "\n", encoding="utf-8")
    assert bt._validated_labels(bt.candidates_c1()) == {}
    fresh = dict(stale, sentence_sha256=bt._text_sha(item["sentence"]))
    (bt.BUILD / "labels_items_sonnet.jsonl").write_text(json.dumps(fresh) + "\n", encoding="utf-8")
    assert bt._validated_labels(bt.candidates_c1()) == {item["item_id"]: "contradicts"}
