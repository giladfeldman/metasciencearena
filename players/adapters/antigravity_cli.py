"""Google Antigravity CLI adapter for revealed-split model pilots.

Antigravity is Google's replacement CLI route for individual/free Gemini CLI
usage. It is an agent CLI, not a reusable API key. This adapter uses headless
print mode, unwraps Antigravity's JSON envelope, and parses the model response
as the arena output JSON.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from framework.player_adapter import (
    PlayerAdapter,
    _extract_json_value,
    register_adapter_class,
)


def _resolve_agy() -> str:
    found = shutil.which("agy")
    if found:
        return found
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "agy" / "bin" / "agy.exe"
    if local.exists():
        return str(local)
    raise RuntimeError(
        "Antigravity CLI binary not found. Run the official installer from "
        "https://antigravity.google/docs/cli/install, then open a new terminal "
        "or dot-source scripts/bootstrap_free_tier_env.ps1."
    )


class AntigravityCliAdapter(PlayerAdapter):
    """Run one task through `agy -p ... --output-format json`."""

    def __init__(
        self,
        *args,
        prompt_template_path: str | Path,
        agy_model: str | None = None,
        agy_effort: str | None = None,
        agy_agent: str | None = None,
        agy_print_timeout: str | None = None,
        agy_json_schema: str | None = None,
        agy_json_schema_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_template_path = Path(prompt_template_path)
        self.agy_model = (agy_model or "").strip()
        self.agy_effort = (agy_effort or "").strip()
        self.agy_agent = (agy_agent or "").strip()
        self.agy_print_timeout = (agy_print_timeout or "").strip()
        self.agy_json_schema = (agy_json_schema or "").strip()
        self.agy_json_schema_path = Path(agy_json_schema_path) if agy_json_schema_path else None
        self._agy: str | None = None
        self._template: str | None = None

    def prepare(self) -> None:
        self._agy = _resolve_agy()
        self._template = self.prompt_template_path.read_text(encoding="utf-8")

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._agy is None or self._template is None:
            self.prepare()

        payload = envelope["input"]
        if isinstance(payload, dict) and "text" in payload:
            input_str = payload["text"]
        else:
            input_str = json.dumps(payload, ensure_ascii=False)
        prompt = self._template.replace("{{INPUT_TEXT}}", input_str)

        cmd = [
            self._agy,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--print-timeout",
            self.agy_print_timeout or f"{timeout_s}s",
        ]
        if self.agy_model:
            cmd.extend(["--model", self.agy_model])
        if self.agy_effort:
            cmd.extend(["--effort", self.agy_effort])
        if self.agy_agent:
            cmd.extend(["--agent", self.agy_agent])
        if self.agy_json_schema:
            cmd.extend(["--json-schema", self.agy_json_schema])
        elif self.agy_json_schema_path:
            cmd.extend(["--json-schema", str(self.agy_json_schema_path)])

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr.strip() or proc.stdout.strip())[:500]
            raise RuntimeError(f"agy exited {proc.returncode}: {detail}")

        try:
            body = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return _extract_json_value(proc.stdout)

        if not isinstance(body, dict):
            raise RuntimeError(f"agy produced unexpected JSON envelope: {type(body).__name__}")
        if "status" not in body:
            return body

        if body.get("status") != "SUCCESS":
            raise RuntimeError(f"agy status={body.get('status')!r}: {body.get('error') or body}")
        structured = body.get("structured_output")
        if isinstance(structured, dict):
            return structured
        response = str(body.get("response") or "")
        if not response.strip():
            detail = (body.get("error") or proc.stderr.strip() or body) 
            raise RuntimeError(f"agy produced an empty response: {str(detail)[:500]}")
        return _extract_json_value(response)

    def resolved_tool_version(self) -> str | None:
        return self.agy_model or self.player_version


register_adapter_class("AntigravityCliAdapter", AntigravityCliAdapter)
