"""Tests for the rate-limit-aware retry queue (`framework retry-failed`).

The retry loop re-plays only the tasks a flaky provider errored / missed and
merges fresh OK verdicts back in. We drive it with a stubbed `run_tournament`
(no real provider) and a tiny stubbed generator so no network / R / codex runs.
"""
from __future__ import annotations

import json
import types

import pytest

from framework import cli
from framework.paths import ARENAS_ROOT_ENV


def _rec(task_id, *, ok, split="revealed", visibility="public"):
    score = {"primary": 0.9, "breakdown": {}} if ok else {
        "primary": 0.0, "breakdown": {"error": "RuntimeError: 429 Too Many Requests"}}
    return {
        "player_id": "regcheck-groq", "task_id": task_id, "split": split,
        "task_visibility": visibility, "score": score,
    }


def test_record_is_ok_classifies_error_and_null():
    assert cli._record_is_ok(_rec("t1", ok=True))
    assert not cli._record_is_ok(_rec("t1", ok=False))
    assert not cli._record_is_ok({"score": {"primary": None, "breakdown": {}}})
    assert not cli._record_is_ok({"score": {"primary": 1.0, "breakdown": {"error": "x"}}})


def _write(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def test_retry_merges_ok_over_error_and_stops_when_complete(monkeypatch, tmp_path):
    arena = tmp_path / "arenas" / "prereg-deviation-v1"
    runs = arena / "runs" / "v1"
    target = runs / "regcheck-groq__revealed__c6.jsonl"
    # Start: t1 OK, t2 error, t3 missing entirely. Universe = {t1,t2,t3}.
    _write(target, [_rec("t1", ok=True), _rec("t2", ok=False)])

    # The arenas root is resolved per command via framework.paths, so the test
    # sets the documented env var instead of patching a module constant — which
    # also means it exercises the real resolution path rather than bypassing it.
    monkeypatch.setenv(ARENAS_ROOT_ENV, str(tmp_path / "arenas"))
    monkeypatch.setattr(cli, "_resolve_seed", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "_split_task_ids", lambda *a, **k: {"t1", "t2", "t3"})

    # Stub run_tournament: it "runs" the requested task_ids and writes OK records
    # for them into the temp output (simulating a successful retry round).
    def fake_run(*, output_path, only_tasks, **kw):
        _write(output_path, [_rec(t, ok=True) for t in sorted(only_tasks)])
        return len(only_tasks)

    monkeypatch.setattr(cli, "run_tournament", fake_run)

    args = types.SimpleNamespace(
        arena="prereg-deviation-v1", task_set="v1", player="regcheck-groq",
        tag="c6", split="revealed", seed=None, timeout=60, max_rounds=3, cooldown=0)
    rc = cli._cmd_retry_failed(args)
    assert rc == 0

    final = {json.loads(l)["task_id"]: json.loads(l) for l in target.read_text().splitlines()}
    assert set(final) == {"t1", "t2", "t3"}
    assert all(cli._record_is_ok(r) for r in final.values())  # t2 fixed, t3 filled


def test_recover_orphan_retry_temps_folds_ok_verdicts(tmp_path):
    # A previous round died after writing a temp but before merging. Recovery must
    # fold its OK verdicts into the target and delete the temp.
    runs = tmp_path / "runs"
    runs.mkdir()
    target = runs / "regcheck-openai__revealed__c6.jsonl"
    _write(target, [_rec("t1", ok=True), _rec("t2", ok=False)])
    # Orphan temp with a fresh OK t2 (fixes it) + a new t3.
    temp = runs / "regcheck-openai__revealed__c6.retry-r1.jsonl"
    _write(temp, [_rec("t2", ok=True), _rec("t3", ok=True)])

    cli._recover_orphan_retry_temps(target)

    assert not temp.exists()                         # temp consumed
    final = {json.loads(l)["task_id"]: json.loads(l) for l in target.read_text().splitlines()}
    assert set(final) == {"t1", "t2", "t3"}
    assert all(cli._record_is_ok(r) for r in final.values())


def test_retry_never_downgrades_ok_when_round_fails(monkeypatch, tmp_path):
    arena = tmp_path / "arenas" / "prereg-deviation-v1"
    target = arena / "runs" / "v1" / "regcheck-groq__revealed__c6.jsonl"
    _write(target, [_rec("t1", ok=True), _rec("t2", ok=False)])

    # The arenas root is resolved per command via framework.paths, so the test
    # sets the documented env var instead of patching a module constant — which
    # also means it exercises the real resolution path rather than bypassing it.
    monkeypatch.setenv(ARENAS_ROOT_ENV, str(tmp_path / "arenas"))
    monkeypatch.setattr(cli, "_resolve_seed", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "_split_task_ids", lambda *a, **k: {"t1", "t2"})

    # Retry keeps failing (still 429) — t2 stays error, t1 must remain OK.
    def fake_run_fail(*, output_path, only_tasks, **kw):
        _write(output_path, [_rec(t, ok=False) for t in sorted(only_tasks)])
        return len(only_tasks)

    monkeypatch.setattr(cli, "run_tournament", fake_run_fail)

    args = types.SimpleNamespace(
        arena="prereg-deviation-v1", task_set="v1", player="regcheck-groq",
        tag="c6", split="revealed", seed=None, timeout=60, max_rounds=2, cooldown=0)
    rc = cli._cmd_retry_failed(args)
    assert rc == 0

    final = {json.loads(l)["task_id"]: json.loads(l) for l in target.read_text().splitlines()}
    assert cli._record_is_ok(final["t1"])          # untouched, still OK
    assert not cli._record_is_ok(final["t2"])       # still error, not lost
