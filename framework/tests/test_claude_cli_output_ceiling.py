"""The Claude CLI output ceiling is declared, sent, and recorded (d-60b28c).

Left unset, `claude --print` stops at 64000 output tokens, and publication-bias v2
answers stopped there: our ceiling, published as the model's failure. The ceiling is
now a registry key that must reach the child process as CLAUDE_CODE_MAX_OUTPUT_TOKENS
and the record as provenance.request_max_tokens.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from framework.player_adapter import SubprocessCliAdapter, build_adapter

_BASE = dict(player_id="p", player_version="claude-x", player_type="ai-model",
             confidence_strategy="native", deterministic=False)


def _adapter(tmp_path, **kw):
    tpl = tmp_path / "t.txt"
    tpl.write_text("{{INPUT_TEXT}}", encoding="utf-8")
    return SubprocessCliAdapter(**_BASE, cli_command=["claude", "--print", "--model", "claude-x"],
                                prompt_template_path=tpl, cli_json_envelope="claude", **kw)


def _fake_run(seen):
    def run(argv, **kwargs):
        seen["env"] = kwargs.get("env")
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
            {"result": '{"ok": true}', "stop_reason": "end_turn", "usage": {}}))
    return run


@pytest.fixture
def claude_on_path(monkeypatch):
    real = shutil.which
    monkeypatch.setattr(shutil, "which", lambda b: "claude" if b == "claude" else real(b))


def test_declared_ceiling_reaches_the_cli_env_and_the_record(tmp_path, monkeypatch, claude_on_path):
    seen = {}
    monkeypatch.setattr(subprocess, "run", _fake_run(seen))
    a = _adapter(tmp_path, cli_max_output_tokens=200000)
    a.play_task({"input": {"text": "x"}}, timeout_s=5)
    assert seen["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "200000"
    assert a.last_request_max_tokens() == 200000


def test_no_ceiling_declared_sends_and_records_none(tmp_path, monkeypatch, claude_on_path):
    # Two-sided control: nothing invented when nothing was set.
    seen = {}
    monkeypatch.setattr(subprocess, "run", _fake_run(seen))
    a = _adapter(tmp_path)
    a.play_task({"input": {"text": "x"}}, timeout_s=5)
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in (seen["env"] or {})
    assert a.last_request_max_tokens() is None


def test_registry_key_reaches_the_adapter(tmp_path):
    tpl = tmp_path / "t.txt"
    tpl.write_text("x", encoding="utf-8")
    a = build_adapter({**_BASE, "adapter_class": "SubprocessCliAdapter",
                       "cli_command": ["claude", "--print"], "prompt_template_path": str(tpl),
                       "cli_max_output_tokens": 200000})
    assert a.cli_max_output_tokens == 200000


@pytest.mark.parametrize("bad", [0, -1, "200000", True])
def test_bad_values_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        _adapter(tmp_path, cli_max_output_tokens=bad)


def test_only_the_claude_cli_takes_it(tmp_path):
    tpl = tmp_path / "t.txt"
    tpl.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        SubprocessCliAdapter(**_BASE, cli_command=["codex", "exec"], prompt_template_path=tpl,
                             cli_max_output_tokens=1000)


# --- d-daa31d: a non-zero exit keeps the CLI's own reason -------------------
# `claude --print --output-format json` reports its failures (usage limit, API
# error) in the stdout envelope and leaves stderr empty, so the 2026-10-08
# publication-bias re-run recorded 15 failures as a bare "claude exited 1: ".

def _failing_run(stdout, stderr=""):
    def run(argv, **kwargs):
        return SimpleNamespace(returncode=1, stderr=stderr, stdout=stdout)
    return run


def test_nonzero_exit_surfaces_the_envelope_error(tmp_path, monkeypatch, claude_on_path):
    env = {"type": "result", "subtype": "success", "is_error": True,
           "result": "Claude AI usage limit reached|1760000000"}
    monkeypatch.setattr(subprocess, "run", _failing_run(json.dumps(env)))
    with pytest.raises(RuntimeError) as exc:
        _adapter(tmp_path).play_task({"input": {"text": "x"}}, timeout_s=5)
    msg = str(exc.value)
    assert msg.startswith("claude exited 1: ")
    assert "usage limit reached" in msg
    assert "subtype=success" in msg and "is_error=True" in msg


def test_nonzero_exit_keeps_stderr_and_ignores_non_envelope_stdout(tmp_path, monkeypatch, claude_on_path):
    # Two-sided control: stderr still wins when present; prose stdout adds nothing.
    monkeypatch.setattr(subprocess, "run", _failing_run("not json", stderr="boom"))
    with pytest.raises(RuntimeError) as exc:
        _adapter(tmp_path).play_task({"input": {"text": "x"}}, timeout_s=5)
    assert str(exc.value) == "claude exited 1: boom"
