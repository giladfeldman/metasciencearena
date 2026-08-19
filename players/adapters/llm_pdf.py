"""LlmCliPdfAdapter — invoke a coding-CLI (claude / gemini / codex) to
extract clean text from a PDF, using the user's CLI subscription rather
than an API key.

The PDF is materialized to a temp file; the prompt template references
that file path with {{PDF_PATH}}. Both the Claude and Gemini CLIs support
reading local files via their built-in file-read tools (Claude's Read,
Gemini's @-mention or filesystem tool), so the adapter can stay vendor-
neutral. The CLI is given the templated prompt on stdin.

Output is parsed permissively: prefer fenced JSON, fall back to the
first {...} block, fall back to the raw string under `full_text` so the
scorer can still rank the player rather than hard-erroring on a chatty
response.

Configuration (in players/registry.yaml):
- cli_command: argv list, e.g. ["claude", "--print", "--model", "claude-haiku-4-5"]
- prompt_template_path: path to a .txt file with {{PDF_PATH}} and
  optionally {{N_PAGES}} placeholders.

Determinism: false (CLI subscriptions are non-deterministic by default).
The framework runs non-deterministic players for `trials` repetitions.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path  # noqa: F401  (kept for adapter parity)

from framework.player_adapter import PlayerAdapter, register_adapter_class

from framework import hermetic


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        s = s.rsplit("```", 1)[0]
    return s.strip()


def _coerce_output(raw: str) -> dict:
    """Permissively coerce an LLM response into the arena output shape."""
    candidate = _strip_fences(raw)
    try:
        data = json.loads(candidate)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {"full_text": raw.strip(), "pages": [raw.strip()], "footnotes": []}
        try:
            data = json.loads(m.group(0))
        except Exception:
            return {"full_text": raw.strip(), "pages": [raw.strip()], "footnotes": []}
    if not isinstance(data, dict):
        return {"full_text": str(data), "pages": [str(data)], "footnotes": []}
    full_text = str(data.get("full_text") or "")
    pages = data.get("pages")
    if not isinstance(pages, list):
        pages = [full_text]
    pages = [str(p) for p in pages]
    footnotes = data.get("footnotes")
    if not isinstance(footnotes, list):
        footnotes = []
    footnotes = [str(f) for f in footnotes]
    return {"full_text": full_text, "pages": pages, "footnotes": footnotes}


class LlmCliPdfAdapter(PlayerAdapter):
    cli_command: list[str]
    prompt_template_path: Path
    output_message_flag: str | None  # e.g. "-o" for codex; None for claude/gemini

    def __init__(self, *args, cli_command: list[str], prompt_template_path: str | Path,
                 output_message_flag: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_command = list(cli_command)
        self.prompt_template_path = Path(prompt_template_path)
        self.output_message_flag = output_message_flag
        self._template: str | None = None
        self._resolved: list[str] | None = None

    def prepare(self) -> None:
        first = self.cli_command[0]
        resolved = shutil.which(first)
        if resolved is None:
            raise RuntimeError(f"CLI binary not on PATH: {first}")
        self._resolved = [resolved] + self.cli_command[1:]
        self._template = self.prompt_template_path.read_text(encoding="utf-8")

    def _coerce_response(self, raw: str) -> dict:
        """Coerce raw CLI output into the arena output shape.

        Subclasses override this to change the output schema without needing
        to duplicate the PDF-materialization / CLI-invocation scaffolding.
        """
        return _coerce_output(raw)

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._resolved is None or self._template is None:
            self.prepare()
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        n_pages = envelope["input"].get("n_pages", 1)

        # Materialize the PDF inside the project workspace, not the global
        # tempdir. The Gemini CLI sandboxes file reads to the workspace dir
        # and rejects paths under %TEMP%; Claude CLI doesn't care; Codex
        # doesn't care. Single in-workspace location keeps all CLIs happy.
        tmp_dir = hermetic.hermetic_cwd(tag="pdf-workspace")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, dir=str(tmp_dir)
        ) as f:
            f.write(pdf_bytes)
            pdf_path = f.name
        # Backslashes confuse some CLIs' @-mention parsers and JSON-in-prompt
        # echoers. Use forward slashes everywhere — Windows handles them fine.
        pdf_path_fwd = pdf_path.replace("\\", "/")
        # If the CLI emits its final response only to a file (codex's
        # `-o <file>` / `--output-last-message`), give it a fresh path.
        out_file: str | None = None
        argv = list(self._resolved)
        if self.output_message_flag:
            out_file = str(tmp_dir / f"_lastmsg_{os.getpid()}_{id(envelope)}.txt")
            if os.path.exists(out_file):
                os.unlink(out_file)
            argv.extend([self.output_message_flag, out_file])

        # CONTAINMENT: the workspace above is also the cwd, and the only

        # tool this player gets is Read. Verified 2026-08-19 (claude-code

        # 2.1.224): Read returns a file inside that cwd and reports BLOCKED

        # for an absolute path into the repo. Hardening happens HERE, after

        # every adapter flag, so nothing is appended past the final flag.

        argv, spawn = hermetic.spawn_kwargs(

            argv, allow_tools=("Read",), cwd=tmp_dir)


        try:
            prompt = (self._template
                      .replace("{{PDF_PATH}}", pdf_path_fwd)
                      .replace("{{N_PAGES}}", str(n_pages)))
            proc = subprocess.run(
                argv,
                input=prompt,
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=timeout_s, check=False, **spawn,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{self.cli_command[0]} exited {proc.returncode}: "
                    f"{proc.stderr.strip()[:300]}"
                )
            # Prefer the captured last-message file when configured;
            # otherwise parse stdout. Falls back to stdout if the file
            # is empty/missing.
            raw = ""
            if out_file and os.path.exists(out_file):
                raw = Path(out_file).read_text(encoding="utf-8", errors="replace")
            if not raw.strip():
                raw = proc.stdout
            out = self._coerce_response(raw)
            out["player_strategy_notes"] = f"{self.cli_command[0]} CLI subscription"
            return out
        finally:
            try:
                os.unlink(pdf_path)
            except OSError:
                pass
            if out_file and os.path.exists(out_file):
                try:
                    os.unlink(out_file)
                except OSError:
                    pass


register_adapter_class("LlmCliPdfAdapter", LlmCliPdfAdapter)
