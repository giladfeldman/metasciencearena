"""Grouping probe hardening from the tier-3 consult of 2026-10-15 (Sol #7, #8).

#8: the paired CI resampled TASKS, but tasks rotate sentences of one abstract, so
they are not independent. Sol's counterexample: two abstracts x four six-item tasks
-> lower bound -0.083 by task (passes the -0.10 bar), -0.167 by abstract (fails).
The bar now takes the wider of the two, as the false-alarm bars take
max(Wilson, abstract bootstrap). #7: an incomplete probe exited 0; it now exits 3,
and --resume reuses only answered pairs of a probe bound to the same run file.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ARENA = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ccf_gp_hardening", ARENA / "tools" / "grouping_probe.py")
gp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gp)

MIN_CI95_LOWER = -0.10  # arena.yaml pass_bars.grouping.min_ci95_lower


def _sol_counterexample():
    diffs, cluster = {}, {}
    for a, row in (("A", [-1, 0, 0, 0, 0, 0]), ("B", [1, 1, 1, 0, 0, 0])):
        for k in range(4):
            diffs[f"{a}{k}"] = [float(x) for x in row]
            cluster[f"{a}{k}"] = a
    return diffs, cluster


def test_the_task_level_bootstrap_alone_passes_the_counterexample():
    """Control: this is the defect -- by task, the CI bar passes."""
    diffs, _ = _sol_counterexample()
    mean, lo, _ = gp.paired_ci(diffs)
    assert round(mean, 4) == 0.1667 and lo >= MIN_CI95_LOWER


def test_the_conservative_bound_fails_the_counterexample():
    diffs, cluster = _sol_counterexample()
    mean, lo, _ = gp.conservative_ci(diffs, cluster)
    assert round(mean, 4) == 0.1667 and lo < MIN_CI95_LOWER


def test_when_every_task_is_its_own_abstract_nothing_changes():
    diffs = {f"t{k}": [float(k % 2), 0.0, -float(k % 3 == 0)] for k in range(12)}
    cluster = {t: t for t in diffs}
    assert gp.conservative_ci(diffs, cluster) == gp.paired_ci(diffs)


def _probe(tmp_path, **summary):
    s = {"player": "p", "task_set": "v2", "run_file_sha256": "abc", "max_tasks": None} | summary
    items = [{"task_id": "t", "id": 1, "single_verdict": "silent", "error": None},
             {"task_id": "t", "id": 2, "single_verdict": None, "error": "RuntimeError: 429"}]
    p = tmp_path / "probe.json"
    p.write_text(json.dumps({"summary": s, "items": items}), encoding="utf-8")
    return p


def test_resume_reuses_only_answered_pairs(tmp_path):
    prior = gp.load_prior(_probe(tmp_path), player="p", task_set="v2", run_file_sha256="abc", max_tasks=None)
    assert set(prior) == {("t", 1)}


def test_resume_refuses_a_probe_of_another_run_file(tmp_path):
    with pytest.raises(SystemExit):
        gp.load_prior(_probe(tmp_path), player="p", task_set="v2", run_file_sha256="other", max_tasks=None)


def test_a_probe_with_an_errored_pair_is_not_complete():
    ok = {"n_errored": 0, "n_missing_grouped": 0, "n_expected": 4, "n": 4}
    assert gp.probe_complete(ok) is True
    assert gp.probe_complete(ok | {"n_errored": 1, "n": 3}) is False
    assert gp.probe_complete(ok | {"n": 3}) is False
