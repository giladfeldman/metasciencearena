"""AnystyleReferencesAdapter — run AnyStyle (a Ruby CLI reference-string
parser) over the bibliography text of a PDF, map its JSON to the arena
reference schema.

Flow in play_task:
  1. Decode the PDF, write it to a temp file.
  2. Run `pdftotext` to get the plain text.
  3. Slice out the bibliography region — the text after the last
     "References" / "Bibliography" heading — so AnyStyle only parses
     reference strings rather than the whole body.
  4. Write that region to a temp file and run
     `anystyle --format json parse <file>`.
  5. Map AnyStyle's JSON (a list of CSL-ish records) to the arena schema.

AnyStyle emits CSL-style records. Each record's fields are arrays of
strings (AnyStyle wraps even single values in a list):
  author          -> [{family, given}, ...]
  date / issued   -> ["2020"]
  title           -> ["A study"]
  container-title -> ["J Things"]
  volume          -> ["12"]
  issue / number  -> ["3"]
  pages           -> ["100-110"]
  doi             -> ["10.1/a"]
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
import tempfile
from pathlib import Path

from framework.player_adapter import PlayerAdapter, register_adapter_class

_BIB_HEADING = re.compile(r"^\s*(references|bibliography|works cited|literature cited)\s*$",
                          re.IGNORECASE | re.MULTILINE)


def _first(value):
    """AnyStyle wraps values in lists; pull the first scalar (or pass through)."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _str_or_none(value):
    v = _first(value)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _split_pages(pages):
    """Split a CSL `pages` value like '100-110' into (fpage, lpage)."""
    p = _str_or_none(pages)
    if not p:
        return None, None
    parts = re.split(r"[-–—]", p, maxsplit=1)
    fpage = parts[0].strip() or None
    lpage = parts[1].strip() if len(parts) > 1 else None
    return fpage, lpage


def _anystyle_json_to_references(anystyle_json: list) -> list[dict]:
    """Map AnyStyle's JSON output (list of CSL-ish records) to the arena schema."""
    out: list[dict] = []
    if not isinstance(anystyle_json, list):
        return out
    for n, rec in enumerate(anystyle_json, start=1):
        if not isinstance(rec, dict):
            continue
        authors: list[dict] = []
        raw_authors = rec.get("author")
        if isinstance(raw_authors, list):
            for a in raw_authors:
                if isinstance(a, dict):
                    surname = (a.get("family") or a.get("surname") or "").strip() or None
                    given = (a.get("given") or a.get("given_names") or "").strip() or None
                elif isinstance(a, str):
                    surname, given = a.strip() or None, None
                else:
                    continue
                if surname or given:
                    authors.append({"surname": surname, "given_names": given})

        year = None
        raw_date = _str_or_none(rec.get("date")) or _str_or_none(rec.get("issued"))
        if raw_date:
            m = re.search(r"\d{4}", raw_date)
            year = m.group(0) if m else raw_date

        fpage, lpage = _split_pages(rec.get("pages"))
        doi = _str_or_none(rec.get("doi"))

        raw_text = _str_or_none(rec.get("original")) or _str_or_none(rec.get("text")) or ""

        out.append({
            "reference_id": _str_or_none(rec.get("id")) or f"ref-{n}",
            "authors": authors,
            "year": year,
            "title": _str_or_none(rec.get("title")),
            "venue": _str_or_none(rec.get("container-title"))
                     or _str_or_none(rec.get("journal")),
            "volume": _str_or_none(rec.get("volume")),
            "issue": _str_or_none(rec.get("issue")) or _str_or_none(rec.get("number")),
            "fpage": fpage,
            "lpage": lpage,
            "doi": doi.lower() if doi else None,
            "pmid": _str_or_none(rec.get("pmid")),
            "raw_text": raw_text,
        })
    return out


def _extract_bibliography(full_text: str) -> str:
    """Return the text after the last References/Bibliography heading.

    If no such heading is found, return the whole text — AnyStyle can still
    parse, it will just see more noise.
    """
    matches = list(_BIB_HEADING.finditer(full_text))
    if not matches:
        return full_text
    return full_text[matches[-1].end():]


class AnystyleReferencesAdapter(PlayerAdapter):
    """Adapter wrapping the AnyStyle Ruby CLI. deterministic / tool."""

    cli_command: list[str]
    pdftotext_binary: str

    def __init__(self, *args, cli_command: list[str] | None = None,
                 pdftotext_binary: str = "pdftotext", **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_command = list(cli_command) if cli_command else ["anystyle"]
        self.pdftotext_binary = pdftotext_binary

    def prepare(self) -> None:
        import shutil
        for binary in (self.cli_command[0], self.pdftotext_binary):
            if shutil.which(binary) is None:
                raise RuntimeError(f"binary not found on PATH: {binary}")

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        tmp_dir = Path.cwd() / ".tmp_pdfs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = txt_path = bib_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False,
                                             dir=str(tmp_dir)) as f:
                f.write(pdf_bytes)
                pdf_path = f.name
            txt_path = pdf_path + ".txt"
            subprocess.run(
                [self.pdftotext_binary, "-q", pdf_path, txt_path],
                capture_output=True, timeout=timeout_s, check=True,
            )
            full_text = Path(txt_path).read_text(encoding="utf-8", errors="replace")
            bib_text = _extract_bibliography(full_text)

            bib_path = pdf_path + ".bib.txt"
            Path(bib_path).write_text(bib_text, encoding="utf-8")
            proc = subprocess.run(
                self.cli_command + ["--format", "json", "parse", bib_path],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout_s, check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"anystyle exited {proc.returncode}: {proc.stderr.strip()[:300]}")
            import json
            anystyle_json = json.loads(proc.stdout)
            references = _anystyle_json_to_references(anystyle_json)
            return {
                "references": references,
                "player_strategy_notes": "anystyle parse (bibliography region)",
            }
        finally:
            for path in (pdf_path, txt_path, bib_path):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass


register_adapter_class("AnystyleReferencesAdapter", AnystyleReferencesAdapter)
