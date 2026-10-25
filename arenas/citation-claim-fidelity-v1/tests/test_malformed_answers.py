"""Owner decision 2026-10-15: an errored record whose call returned an answer is an answer.

Flash-Lite's v2 r1 run returned 28 bare JSON lists where {"items": [...]} is
required. The runner stores those as `output_schema_violation`; analyze_run counted
them as errored calls, so the run could never be complete. Now an errored record
whose raw provider response was stored (the runner keeps one for every call that got
an HTTP answer, none for a 429 / 5xx / timeout) is read through the scorer's guard
like any answer -- so what the citeclaim library would keep is kept and the rest is
no verdict (llm_malformed). The tier-3 consult (2026-10-15, Sol #2/#3) showed a
blanket "all items None" was broader than the library, and that an unparseable
answer reaches the record as a bare RuntimeError.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from framework import run_env

ARENA = Path(__file__).resolve().parents[1]


def _load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, ARENA / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


analyze = _load("tools/analyze_run", "ccf_malformed_analyze")


def _raw(text):
    return {"choices": [{"message": {"content": text}}]}


def _run(tmp_path, error: str | None, raw_text: str | None = None):
    """Revealed run, every item 'silent'; the first multi-item task gets `error` (and `raw_text`)."""
    from framework.runner import _canonical_sha256
    gen = analyze.generator
    envs = list(gen.generate("v1", 0, split="revealed"))
    first = next(e["task_id"] for e in envs if len(gen.ground_truth(e["task_id"])["items"]) >= 2)
    recs = []
    for k, env in enumerate(envs):
        gt = gen.ground_truth(env["task_id"])
        out = {"items": [{"id": it["id"], "verdict": "silent", "evidence_quote": "", "reason": "r"}
                         for it in gt["items"]]}
        score = {"breakdown": {}}
        if env["task_id"] == first and error is not None:
            out, score = {}, {"primary": 0.0, "breakdown": {"error": error}}
        recs.append({"task_id": env["task_id"], "run_id": f"r{k}", "player_id": "p",
                     "task_set_version": "v1", "output": out, "score": score,
                     "instrument": {"gold_sha256": _canonical_sha256(gt)}})
    p = tmp_path / "run.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    if raw_text is not None:
        rid = next(r["run_id"] for r in recs if r["task_id"] == first)
        run_env.append_raw(run_env.raw_sidecar_path(p, held_out=False, arena_id="x"),
                           {"run_id": rid, "raw_response": _raw(raw_text)})
    return p, first, gen.ground_truth(first)


def _analyse(run_file):
    items, meta = analyze.item_results(run_file, "revealed", 0)
    return items, meta, analyze.summarize(items, meta, None)


def _mine(items, tid):
    return [it["verdict"] for it in items if it["task_id"] == tid]


def test_a_bare_list_answer_is_no_verdict_on_every_item_and_the_run_is_complete(tmp_path):
    run, tid, gt = _run(tmp_path, "output_schema_violation",
                        json.dumps([{"id": 1, "verdict": "silent", "evidence_quote": "", "reason": "r"}]))
    items, meta, summary = _analyse(run)
    assert _mine(items, tid) == [None] * len(gt["items"])
    assert meta["answered_with_error"] == 1 and meta.get("errored", 0) == 0
    assert summary["complete"] is True


def test_a_schema_rejected_answer_keeps_the_items_the_library_would_keep(tmp_path):
    """Sol #2: item 2's null verdict fails the schema; item 1 is still a readable verdict."""
    answer = {"items": [{"id": 1, "verdict": "silent", "evidence_quote": "", "reason": "r"},
                        {"id": 2, "verdict": None, "evidence_quote": "", "reason": "r"}]}
    run, tid, gt = _run(tmp_path, "output_schema_violation", json.dumps(answer))
    items, _, summary = _analyse(run)
    assert _mine(items, tid)[:2] == ["silent", None]
    assert summary["complete"] is True


def test_an_unparseable_answer_reported_as_runtime_error_is_no_verdict_not_an_error(tmp_path):
    """Sol #3: prose instead of JSON reaches the record as a bare RuntimeError."""
    run, tid, gt = _run(tmp_path, "RuntimeError", "I cannot determine this from the abstract.")
    items, meta, summary = _analyse(run)
    assert _mine(items, tid) == [None] * len(gt["items"])
    assert meta["answered_with_error"] == 1 and summary["complete"] is True


def test_a_call_that_produced_no_answer_still_makes_the_run_incomplete(tmp_path):
    run, tid, _ = _run(tmp_path, "RuntimeError")
    items, meta, summary = _analyse(run)
    assert meta["errored"] == 1 and meta.get("answered_with_error", 0) == 0
    assert _mine(items, tid) == []
    assert summary["complete"] is False


def test_a_schema_violation_without_its_raw_answer_fails_closed(tmp_path):
    """No raw, no way to read the answer as the library would: incomplete, never guessed."""
    run, _, _ = _run(tmp_path, "output_schema_violation")
    _, meta, summary = _analyse(run)
    assert meta["errored"] == 1 and summary["complete"] is False


def test_a_transport_failure_with_a_stale_raw_is_still_an_errored_call(tmp_path):
    """Sol, consult 2026-10-19: a container call that raised kept the PREVIOUS task's raw,
    so raw presence alone made a timed-out or quota-failed call read as 'answered'."""
    for error in ("TransportTimeout", "TransportError"):
        run, tid, _ = _run(tmp_path, error, json.dumps({"items": []}))
        items, meta, summary = _analyse(run)
        assert meta["errored"] == 1 and meta.get("answered_with_error", 0) == 0, error
        assert _mine(items, tid) == [] and summary["complete"] is False, error
