"""Run-summary outcome classification (cycle 9, 2026-08-04).

A CLI player wrapped in coreutils `timeout` reports a wall-clock kill as exit 124
(SIGTERM) or 137 (SIGKILL via -k). Neither message contains the word "Timeout",
so both were summarised as generic errors — hiding the distinction between "the
model failed" and "the wall was too tight". Real case: every gemma-4-31b task
came back `RuntimeError: timeout exited 124` and was reported as an error.
"""
from framework.runner import _record_outcome


def _rec(err=None):
    score = {"primary": 0.0, "breakdown": {"error": err}} if err else {"primary": 1.0, "breakdown": {}}
    return {"score": score}


def test_ok_record():
    assert _record_outcome(_rec()) == "ok"


def test_python_timeout_exception_is_a_timeout():
    assert _record_outcome(_rec("TimeoutExpired: command timed out")) == "timeout"


def test_coreutils_timeout_exit_124_is_a_timeout():
    assert _record_outcome(_rec("RuntimeError: timeout exited 124: \n> build")) == "timeout"


def test_coreutils_kill_exit_137_is_a_timeout():
    assert _record_outcome(_rec("RuntimeError: timeout exited 137")) == "timeout"


def test_a_real_error_is_not_a_timeout():
    assert _record_outcome(_rec("ValueError: output_schema_violation")) == "error"


def test_missing_binary_is_an_error_not_a_timeout():
    assert _record_outcome(_rec("RuntimeError: CLI binary not found on PATH: opencode")) == "error"
