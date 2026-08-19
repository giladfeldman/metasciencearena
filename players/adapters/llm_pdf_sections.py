"""LlmCliPdfSectionsAdapter — invoke a CLI (claude/gemini/codex) on a PDF
and parse a JSON response into the pdf-section-structure-v1 output shape.

Inherits the file-materialization + permissive-output-parsing scaffolding
from llm_pdf.py; overrides _coerce_output to expect a {sections: [...]}
JSON object.
"""
from __future__ import annotations

import json
import re

from framework.player_adapter import register_adapter_class
from players.adapters.llm_pdf import LlmCliPdfAdapter, _strip_fences

from framework import hermetic

CANONICAL_LABELS = {
    "title", "abstract", "introduction", "methods", "results",
    "discussion", "conclusion", "references", "acknowledgments",
    "appendix", "other",
}


def _coerce_sections(raw: str) -> dict:
    candidate = _strip_fences(raw)
    try:
        data = json.loads(candidate)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {"sections": []}
        try:
            data = json.loads(m.group(0))
        except Exception:
            return {"sections": []}
    if not isinstance(data, dict):
        return {"sections": []}
    raw_secs = data.get("sections")
    if not isinstance(raw_secs, list):
        return {"sections": []}
    out_secs: list[dict] = []
    for i, s in enumerate(raw_secs):
        if not isinstance(s, dict):
            continue
        label_raw = (s.get("label") or "").strip().lower()
        if label_raw not in CANONICAL_LABELS:
            label_raw = "other"
        cs = s.get("char_start")
        ce = s.get("char_end")
        page = s.get("page")
        out_secs.append({
            "label": label_raw,
            "heading_text": (str(s.get("heading_text")) if s.get("heading_text") is not None else None),
            "section_index": int(s.get("section_index") or i),
            "char_start": int(cs) if isinstance(cs, (int, float)) else None,
            "char_end": int(ce) if isinstance(ce, (int, float)) else None,
            "page": int(page) if isinstance(page, (int, float)) else None,
        })
    return {"sections": out_secs}


class LlmCliPdfSectionsAdapter(LlmCliPdfAdapter):
    def _coerce_response(self, raw: str) -> dict:
        return _coerce_sections(raw)

    # Override play_task minimally: same flow as parent but use sections coercer
    # and emit the section output shape instead of pdf-text-fidelity-v1 shape.
    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        import base64
        import os
        import subprocess
        import tempfile
        from pathlib import Path

        if self._resolved is None or self._template is None:
            self.prepare()
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        n_pages = envelope["input"].get("n_pages", 1)
        tmp_dir = hermetic.hermetic_cwd(tag="pdf-workspace")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=str(tmp_dir)) as f:
            f.write(pdf_bytes)
            pdf_path = f.name
        pdf_path_fwd = pdf_path.replace("\\", "/")
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
                argv, input=prompt, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout_s, check=False, **spawn,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{self.cli_command[0]} exited {proc.returncode}: "
                    f"{proc.stderr.strip()[:300]}"
                )
            raw = ""
            if out_file and os.path.exists(out_file):
                raw = Path(out_file).read_text(encoding="utf-8", errors="replace")
            if not raw.strip():
                raw = proc.stdout
            out = self._coerce_response(raw)
            out["player_strategy_notes"] = f"{self.cli_command[0]} CLI subscription (sections)"
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


register_adapter_class("LlmCliPdfSectionsAdapter", LlmCliPdfSectionsAdapter)
