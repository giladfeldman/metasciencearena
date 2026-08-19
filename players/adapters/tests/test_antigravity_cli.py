import json
import subprocess
from pathlib import Path

import pytest

from players.adapters.antigravity_cli import AntigravityCliAdapter


def _adapter(tmp_path: Path, **kwargs) -> AntigravityCliAdapter:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return JSON for: {{INPUT_TEXT}}", encoding="utf-8")
    return AntigravityCliAdapter(
        player_id="agy-test",
        player_version="gemini-3.6-flash-low",
        player_type="ai-model",
        confidence_strategy="native",
        deterministic=False,
        prompt_template_path=prompt,
        agy_model="gemini-3.6-flash-low",
        **kwargs,
    )


def test_antigravity_cli_unwraps_json_response(monkeypatch, tmp_path):
    seen = {}

    def fake_which(name):
        return r"C:\Users\filin\AppData\Local\agy\bin\agy.exe"

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        body = {
            "status": "SUCCESS",
            "response": '{"label":"ok","confidence":0.8}\n',
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(body), stderr="")

    monkeypatch.setattr("players.adapters.antigravity_cli.shutil.which", fake_which)
    monkeypatch.setattr("players.adapters.antigravity_cli.subprocess.run", fake_run)

    out = _adapter(tmp_path).play_task({"input": {"text": "hello"}}, timeout_s=12)

    assert out == {"label": "ok", "confidence": 0.8}
    assert seen["cmd"][1:5] == ["-p", "Return JSON for: hello", "--output-format", "json"]
    assert seen["cmd"][-2:] == ["--model", "gemini-3.6-flash-low"]


def test_antigravity_cli_reports_failed_status(monkeypatch, tmp_path):
    monkeypatch.setattr("players.adapters.antigravity_cli.shutil.which", lambda _: "agy")
    monkeypatch.setattr(
        "players.adapters.antigravity_cli.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"status": "ERROR", "error": "quota exhausted"}),
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="quota exhausted"):
        _adapter(tmp_path).play_task({"input": {"text": "hello"}}, timeout_s=12)


def test_antigravity_cli_prefers_structured_output(monkeypatch, tmp_path):
    monkeypatch.setattr("players.adapters.antigravity_cli.shutil.which", lambda _: "agy")
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({
                "status": "SUCCESS",
                "response": "",
                "structured_output": {"records": []},
            }),
            stderr="",
        )

    monkeypatch.setattr("players.adapters.antigravity_cli.subprocess.run", fake_run)

    out = _adapter(tmp_path, agy_json_schema_path=schema).play_task(
        {"input": {"text": "hello"}},
        timeout_s=12,
    )

    assert out == {"records": []}
    assert seen["cmd"][-2:] == ["--json-schema", str(schema)]


def test_antigravity_cli_accepts_raw_fenced_json(monkeypatch, tmp_path):
    monkeypatch.setattr("players.adapters.antigravity_cli.shutil.which", lambda _: "agy")
    monkeypatch.setattr(
        "players.adapters.antigravity_cli.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            0,
            stdout='```json\n{"records":[]}\n```',
            stderr="",
        ),
    )

    out = _adapter(
        tmp_path,
        agy_json_schema='{"type":"object","properties":{"records":{"type":"array"}}}',
    ).play_task({"input": {"text": "hello"}}, timeout_s=12)

    assert out == {"records": []}


def test_antigravity_cli_resolved_tool_version(tmp_path):
    assert _adapter(tmp_path).resolved_tool_version() == "gemini-3.6-flash-low"
