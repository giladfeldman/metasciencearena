"""PdftotextSubprocessAdapter — `pdftotext` baseline.

Spawns `pdftotext` against a temporary file (xpdf's pdftotext on Windows
does not accept `-` as stdin; poppler does, but we standardize on the
temp-file path so the adapter works against either build). Emits the
text as `full_text` and `pages[0]`; per-page splitting is approximated
by splitting on form-feed (`\\f`), which pdftotext emits between pages
by default.
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile

from framework.player_adapter import PlayerAdapter, register_adapter_class


class PdftotextSubprocessAdapter(PlayerAdapter):
    pdftotext_binary: str

    def __init__(self, *args, pdftotext_binary: str = "pdftotext", **kwargs):
        super().__init__(*args, **kwargs)
        self.pdftotext_binary = pdftotext_binary
        self._resolved: str | None = None

    def prepare(self) -> None:
        resolved = shutil.which(self.pdftotext_binary)
        if resolved is None:
            raise RuntimeError(f"pdftotext binary not on PATH: {self.pdftotext_binary}")
        self._resolved = resolved

    def resolved_tool_version(self) -> str | None:
        from players.adapters._tool_version import pdftotext_version
        return pdftotext_version(self.pdftotext_binary)

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._resolved is None:
            self.prepare()
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f_in:
            f_in.write(pdf_bytes)
            in_path = f_in.name
        out_path = in_path + ".txt"
        try:
            proc = subprocess.run(
                [self._resolved, in_path, out_path],
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"pdftotext exited {proc.returncode}: {proc.stderr.decode('utf-8', errors='replace')[:200]}")
            with open(out_path, "rb") as f_out:
                text = f_out.read().decode("utf-8", errors="replace")
        finally:
            for p in (in_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        # Split on form-feed; pdftotext emits one between pages by default.
        pages = text.split("\f") if "\f" in text else [text]
        pages = [p.strip("\n") for p in pages if p.strip()]
        if not pages:
            pages = [""]
        full_text = "\n\n".join(pages)
        return {
            "full_text": full_text,
            "pages": pages,
            "footnotes": [],
            "player_strategy_notes": "pdftotext default mode",
        }


register_adapter_class("PdftotextSubprocessAdapter", PdftotextSubprocessAdapter)
