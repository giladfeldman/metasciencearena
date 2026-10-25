"""A failed call must never carry the PREVIOUS task's raw response (Sol, consult 2026-10-19).

`ContainerizedAdapter.play_task` assigned `self._res` only when `run_in_container`
returned. When it RAISED (the docker clock, an unreadable result file), `_res` still
held the previous task's result, so `last_raw_response()` handed the runner the
previous task's answer, and the runner stored it under the NEW record's run_id.

That matters twice. The arena analysis and `retry-failed` both read "a raw is stored
for this run_id" as "the model answered" (cli._record_was_answered,
citation-claim-fidelity-v1/tools/analyze_run.py), and answer-preserving retry became
the default (d-24a295): a timed-out task would have been settled as answered and
never retried. Fixed at both ends: the adapter clears its telemetry before each call,
and a transport failure never counts as answered whatever raw sits beside it.
"""
from __future__ import annotations

import pytest

from framework import cli, container


def test_a_raising_container_call_exposes_no_previous_telemetry(monkeypatch):
    calls = iter([{"output": {"a": 1}, "raw_response": {"task": "first"}, "usage": {"x": 1}},
                  TimeoutError("docker run timed out")])

    def fake(entry, envelope, timeout_s, runner, host):
        nxt = next(calls)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(container, "run_in_container", fake)
    monkeypatch.setattr(container, "provider_host", lambda entry: "example.invalid")
    ad = container.ContainerizedAdapter({"player_id": "p"}, {"image": "x"})
    ad.play_task({"task_id": "t1"}, timeout_s=1)
    assert ad.last_raw_response() == {"task": "first"}  # control
    with pytest.raises(TimeoutError):
        ad.play_task({"task_id": "t2"}, timeout_s=1)
    assert ad.last_raw_response() is None
    assert ad.last_usage() is None


@pytest.mark.parametrize("error", [
    "TimeoutExpired: Command '['docker', 'run']' timed out after 420 seconds",
    "TransportError",
    "RuntimeError: p: OpenAI-compatible endpoint returned HTTP 429: quota",
])
def test_a_transport_failure_is_never_an_answer_even_with_a_raw(error):
    rec = {"run_id": "r1", "score": {"primary": 0.0, "breakdown": {"error": error}}}
    assert not cli._record_was_answered(rec, {"r1"})


def test_a_model_failure_with_its_raw_is_still_an_answer():
    rec = {"run_id": "r1", "score": {"primary": 0.0, "breakdown": {"error": "RuntimeError"}}}
    assert cli._record_was_answered(rec, {"r1"})  # control: the 2026-10-15 rule stands
