"""Task set v3's real-data tiers (citeclaim Stage D) and the tier-3 consult of 2026-10-18.

Round: docs/consults/2026-10-18-citeclaim-stage-d-real-data/ (Sonnet + Sol). Every input
here is constructed; no custody row is read. Tests marked "consult" were run against the
pre-fix code (commit 820af38) and failed there.
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


scorer = _load("scorer", "ccf_rd_scorer")
bars = _load("bars", "ccf_rd_bars")
generator = _load("generator", "ccf_rd_generator")
analyze = _load("tools/analyze_run", "ccf_rd_analyze")
combine = _load("tools/combine_repeats", "ccf_rd_combine")

ZHANG = bars.load_tier_bars("rd-zhang2024")
SCIFACT = bars.load_tier_bars("rd-scifact-dev")


def _zhang_items(policy):
    """The tier's real shape: 121 Fully (citation studies), 107 Unsub (79 PubPeer, 21 study, 7 correction)."""
    rows = ([("not_contradicts", "citation-study")] * 121 + [("not_consistent", "PubPeer")] * 79
            + [("not_consistent", "citation-study")] * 21 + [("not_consistent", "Correction")] * 7)
    return [{"gold": g, "verdict": policy(g, src), "cluster": f"c{i}", "type": g, "cohort": src}
            for i, (g, src) in enumerate(rows)]


def _scifact_items(policy):
    rows = [("consistent", 138), ("contradicts", 71), ("silent", 130)]
    return [{"gold": g, "verdict": policy(g), "cluster": f"{g}{i}", "type": g}
            for g, n in rows for i in range(n)]


def test_not_contradicts_accepts_consistent_and_silent_only():
    assert scorer.is_correct("not_contradicts", "consistent")
    assert scorer.is_correct("not_contradicts", "silent")
    assert not scorer.is_correct("not_contradicts", "contradicts")
    assert not scorer.is_correct("not_contradicts", None)
    assert scorer._error_category("not_contradicts", "contradicts") == "false_alarm_on_clean"


def test_zhang_self_test():
    best = {"not_contradicts": "consistent", "not_consistent": "contradicts"}
    assert bars.evaluate_tier(_zhang_items(lambda g, s: best[g]), ZHANG)["passed"]
    assert not bars.evaluate_tier(_zhang_items(lambda g, s: "contradicts"), ZHANG)["passed"]
    assert not bars.evaluate_tier(_zhang_items(lambda g, s: "consistent"), ZHANG)["passed"]
    assert not bars.evaluate_tier(_zhang_items(lambda g, s: None), ZHANG)["passed"]


def test_zhang_limits_are_declared_not_hidden():
    """An all-silent player and a source-only policy PASS the Zhang gating bars (consult
    2026-10-18, Sonnet #2 / Sol #2). The tier makes no support-recognition or subtle-recall
    claim; this test pins that limit so a future reader cannot mistake a pass for more."""
    assert bars.evaluate_tier(_zhang_items(lambda g, s: "silent"), ZHANG)["passed"]
    oracle = lambda g, s: "consistent" if s == "citation-study" else "silent"  # noqa: E731
    assert bars.evaluate_tier(_zhang_items(oracle), ZHANG)["passed"]
    yaml_text = (ARENA / "arena.yaml").read_text(encoding="utf-8")
    assert "an all-'silent' player passes it" in yaml_text


def test_zhang_false_alarm_bar_is_the_stage_c_clean_bar():
    """Sonnet #5: 0.15/0.22 on real valid citations was too lax."""
    b = ZHANG["false_suggestion"]["substantiated_false_alarm"]
    assert (b["max_rate"], b["max_upper95"]) == (0.10, 0.15)
    pol = lambda g, s: "contradicts" if g == "not_contradicts" and s == "x" else "silent"  # noqa: E731
    items = _zhang_items(pol)
    for it in items[:13]:  # 13/121 = 0.107 > 0.10
        it["verdict"] = "contradicts"
    assert not bars.evaluate_tier(items, ZHANG)["passed"]


def test_scifact_self_test_and_is_not_gating():
    best = {"consistent": "consistent", "contradicts": "contradicts", "silent": "silent"}
    assert bars.evaluate_tier(_scifact_items(lambda g: best[g]), SCIFACT)["passed"]
    for v in ("contradicts", "consistent", "silent", None):
        assert not bars.evaluate_tier(_scifact_items(lambda g, v=v: v), SCIFACT)["passed"]
    assert SCIFACT["gating"] is False and ZHANG["gating"] is True


def test_consult_scifact_nei_false_reassurance_fails():
    """Sol #4: every NEI judged 'consistent' passed every SciFact bar."""
    def pol(g):
        return "consistent" if g == "silent" else ("consistent" if g == "consistent" else "contradicts")
    r = bars.evaluate_tier(_scifact_items(pol), SCIFACT)
    assert not r["passed"]
    assert "nei_false_reassurance_rate" in [c["bar"] for c in r["checks"] if not c["pass"]]


def test_consult_joint_cluster_bootstrap():
    """Sol #5: ten abstracts shared by both groups; resampling groups separately gave 0.20."""
    a, b = [], []
    for k in range(10):
        hi = k < 8
        a += [(f"s{k}", hi)] * 10
        b += [(f"s{k}", not hi)] * 10
    assert bars.cluster_bootstrap_diff_lower(a, b) <= 0.0


def test_consult_min_clusters_precondition():
    """Sol #6: zero errors among 120 items in two abstracts passed the false-alarm bar."""
    items = [{"gold": "not_contradicts", "verdict": "silent", "cluster": f"c{i % 2}"} for i in range(120)]
    items += [{"gold": "not_consistent", "verdict": "silent", "cluster": f"u{i}"} for i in range(107)]
    r = bars.evaluate_tier(items, ZHANG)
    fail = [c for c in r["checks"] if c["bar"] == "substantiated_false_alarm_rate"][0]
    assert not fail["pass"] and "precondition" in fail["detail"]


def test_consult_missing_required_bar_fails():
    """Sonnet #7: a mis-indented bar was skipped and the tier passed on coverage alone."""
    spec = dict(ZHANG)
    spec.pop("recall")
    best = {"not_contradicts": "consistent", "not_consistent": "contradicts"}
    r = bars.evaluate_tier(_zhang_items(lambda g, s: best[g]), spec)
    assert not r["passed"]
    assert any(c["bar"] == "unsubstantiated_detection" and not c["pass"] for c in r["checks"])
    assert not bars.evaluate_tier(_zhang_items(lambda g, s: best[g]), {"coverage_min": 0.9})["passed"]


def test_consult_one_instrument_per_run_file():
    """Sol #7: records of one player_id with different inference settings read complete."""
    base = {"call_config_sha256": None, "env_sha": "e", "instrument": {"scorer_sha256": "s"},
            "player_version": "m", "player_id": "p", "player_identity": {"inference_config_sha256": "a"}}
    other = dict(base, player_identity={"inference_config_sha256": "b"})
    assert analyze.instrument_key(base) == analyze.instrument_key(dict(base))
    assert analyze.instrument_key(base) != analyze.instrument_key(other)


def test_consult_repeats_are_joined():
    """Sonnet #9: nothing enforced 'every bar on each of 2 repeats'."""
    def doc(sha, passed=True, complete=True):
        return {"summary": {"meta": {"player_id": "p", "task_set": "v3", "tier": "rd-zhang2024",
                                     "run_file_sha256": sha}, "complete": complete,
                            "bars": {"passed": passed, "checks": []}}, "items": [{"task_id": "t1"}]}
    assert combine.combine([doc("a"), doc("b")])["passed"]
    assert not combine.combine([doc("a")])["passed"]
    assert not combine.combine([doc("a"), doc("a")])["passed"]
    assert not combine.combine([doc("a"), doc("b", passed=False)])["passed"]
    assert not combine.combine([doc("a"), doc("b", complete=False)])["passed"]


def test_v1_v2_gold_untouched_v3_gold_carries_tier():
    for v in ("v1", "v2"):
        for env in generator.generate(v, 0, split="revealed"):
            assert "tier" not in generator.ground_truth(env["task_id"])
    for env in generator.generate("v3", 0, split="revealed"):
        assert generator.ground_truth(env["task_id"])["tier"] == "synthetic"


def test_real_tier_report_suppresses_synthetic_only_summaries():
    """Sonnet #8: synthetic-tier summaries printed [0, 0] on a real tier."""
    best = {"not_contradicts": "consistent", "not_consistent": "contradicts"}
    items = _zhang_items(lambda g, s: best[g])
    for it in items:
        it.update(task_id=it["cluster"], correct=True, explanation=None)
    s = analyze.summarize(items, {"players": 1, "tasks_scored": 1, "tasks_expected": 1,
                                  "instrument_variants": 1}, None, "rd-zhang2024")
    assert s["by_cohort"] == {} and s["false_alarm_answered_only"] == {}
    assert "precision_at_prevalence" not in s
    assert s["tier_report"]["discrimination_pooled"]["difference"] == 1.0
    assert s["bars"]["passed"] is True
    json.dumps(s)
