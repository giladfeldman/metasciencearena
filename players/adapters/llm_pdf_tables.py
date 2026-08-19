"""LlmCliPdfTablesAdapter — invoke a CLI (claude/gemini/codex) on a PDF
and parse a JSON response into the pdf-table-extraction-v1 output shape.

LLM adapters cap output to the first 8 tables per paper to bound cost;
this is a documented limitation, not silent truncation — players that
need to exercise > 8 tables/paper should be configured differently.
"""
from __future__ import annotations

import json
import re

from framework.player_adapter import register_adapter_class
from players.adapters.llm_pdf import LlmCliPdfAdapter, _strip_fences

from framework import hermetic

_MAX_TABLES_PER_PAPER = 8


def _coerce_tables(raw: str) -> dict:
    candidate = _strip_fences(raw)
    try:
        data = json.loads(candidate)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {"tables": []}
        try:
            data = json.loads(m.group(0))
        except Exception:
            return {"tables": []}
    if not isinstance(data, dict):
        return {"tables": []}
    raw_tables = data.get("tables")
    if not isinstance(raw_tables, list):
        return {"tables": []}
    out_tables: list[dict] = []
    for t in raw_tables[:_MAX_TABLES_PER_PAPER]:
        if not isinstance(t, dict):
            continue
        cells_in = t.get("cells")
        cells_out: list[dict] = []
        if isinstance(cells_in, list):
            for c in cells_in:
                if not isinstance(c, dict):
                    continue
                try:
                    cells_out.append({
                        "r": int(c.get("r") or 0),
                        "c": int(c.get("c") or 0),
                        "rowspan": max(1, int(c.get("rowspan") or 1)),
                        "colspan": max(1, int(c.get("colspan") or 1)),
                        "text": str(c.get("text") or ""),
                        "is_header": bool(c.get("is_header")),
                    })
                except (TypeError, ValueError):
                    continue
        out_tables.append({
            "label": (str(t.get("label")) if t.get("label") is not None else None),
            "page": int(t.get("page")) if isinstance(t.get("page"), (int, float)) else None,
            "caption": (str(t.get("caption")) if t.get("caption") is not None else None),
            "n_rows": int(t.get("n_rows") or 0),
            "n_cols": int(t.get("n_cols") or 0),
            "header_rows": int(t.get("header_rows") or 0),
            "cells": cells_out,
        })
    return {"tables": out_tables}


class LlmCliPdfTablesAdapter(LlmCliPdfAdapter):
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
            out = _coerce_tables(raw)
            out["player_strategy_notes"] = f"{self.cli_command[0]} CLI subscription (tables, capped at {_MAX_TABLES_PER_PAPER})"
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


register_adapter_class("LlmCliPdfTablesAdapter", LlmCliPdfTablesAdapter)
