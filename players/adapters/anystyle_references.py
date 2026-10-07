"""AnystyleReferencesAdapter — run AnyStyle (a Ruby CLI reference-string
parser) over the bibliography text of a PDF, map its JSON to the arena
reference schema.

Flow in play_task:
  1. Decode the PDF, write it to a temp file.
  2. Run `pdftotext` to get the plain text.
  3. Run `anystyle --format json find <text file>` -- AnyStyle's own document
     pipeline: its finder model locates the reference list and SEGMENTS it,
     then its parser labels each entry.
  4. Map AnyStyle's JSON (a list of CSL-ish records) to the arena schema.

Until 2026-09-28 step 3 was `anystyle parse` over the text after the last
"References" heading. `parse` takes ONE REFERENCE PER LINE, so every wrapped
entry was split at each line break: on ref-synth-08 (18 references) it returned
82 entries and scored 0.341, where `find` returns 16 and parses them cleanly.
That was our adapter misusing the tool, not the tool's quality, and it would
have been published as AnyStyle's score.

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


class AnystyleReferencesAdapter(PlayerAdapter):
    """Adapter wrapping the AnyStyle Ruby CLI. deterministic / tool."""

    cli_command: list[str]
    pdftotext_binary: str

    def __init__(self, *args, cli_command: list[str] | None = None,
                 pdftotext_binary: str = "pdftotext",
                 docker_container: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_command = list(cli_command) if cli_command else ["anystyle"]
        self.pdftotext_binary = pdftotext_binary
        # When set, `anystyle` runs inside this already-running container
        # (players/docker/anystyle). pdftotext still runs on the host, so only
        # the extracted plain TEXT is copied in, never the PDF.
        self.docker_container = docker_container

    def _in_container(self, *argv: str, timeout_s: int = 60):
        return subprocess.run(["docker", "exec", self.docker_container, *argv],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout_s, check=False)

    def prepare(self) -> None:
        import shutil
        binaries = [self.pdftotext_binary] + (["docker"] if self.docker_container
                                              else [self.cli_command[0]])
        for binary in binaries:
            if shutil.which(binary) is None:
                raise RuntimeError(f"binary not found on PATH: {binary}")
        if self.docker_container:
            probe = self._in_container("anystyle", "--version")
            if probe.returncode != 0:
                raise RuntimeError(
                    f"anystyle is not runnable in container {self.docker_container!r}: "
                    f"{(probe.stderr or probe.stdout).strip()[:300]}. Start it per "
                    f"players/docker/README.md")

    def resolved_tool_version(self) -> str | None:
        """`anystyle --version` from the binary that actually parses."""
        try:
            proc = (self._in_container("anystyle", "--version") if self.docker_container
                    else subprocess.run(self.cli_command + ["--version"], capture_output=True,
                                        text=True, timeout=60, check=False))
        except Exception:
            return None
        # "anystyle version 1.6.0 (cli 1.5.0, data 1.3.0)": the engine and the
        # CLI are released separately, so name both.
        out = proc.stdout or ""
        eng = re.search(r"version\s+(\d+\.\d+\.\d+)", out)
        cli = re.search(r"cli\s+(\d+\.\d+\.\d+)", out)
        if proc.returncode != 0 or not eng:
            return None
        return f"anystyle-{eng.group(1)}" + (f"-cli-{cli.group(1)}" if cli else "")

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
            bib_path = txt_path  # `find` reads the whole document text
            if self.docker_container:
                inside = f"/work/{Path(bib_path).name}"
                cp = subprocess.run(["docker", "cp", bib_path, f"{self.docker_container}:{inside}"],
                                    capture_output=True, text=True, timeout=timeout_s, check=False)
                if cp.returncode != 0:
                    raise RuntimeError(f"docker cp into {self.docker_container} failed: "
                                       f"{cp.stderr.strip()[:300]}")
                try:
                    proc = self._in_container("anystyle", "--verbose", "--format", "json", "find", inside,
                                              timeout_s=timeout_s)
                finally:
                    self._in_container("rm", "-f", inside)
            else:
                proc = subprocess.run(
                    self.cli_command + ["--verbose", "--format", "json", "find", bib_path],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=timeout_s, check=False,
                )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"anystyle exited {proc.returncode}: {proc.stderr.strip()[:300]}")
            import json
            if not proc.stdout.strip():
                # `find` prints NOTHING (not "[]") when its finder model locates
                # no reference list; --verbose makes it say so on stderr. Only
                # that statement makes an empty answer; silence alone raises.
                # Measured 2026-09-28: 5 of 29 held-out papers, e.g. one whose
                # text has a plain "REFERENCES" heading at line 955 of 63 KB.
                said = (proc.stderr + " " + proc.stdout).lower()
                if "no references found" not in said:
                    raise RuntimeError("anystyle returned no output and did not say why: "
                                       f"{proc.stderr.strip()[:300]!r}")
                return {"references": [],
                        "player_strategy_notes": "anystyle find: no references found"}
            anystyle_json = json.loads(proc.stdout)
            references = _anystyle_json_to_references(anystyle_json)
            return {
                "references": references,
                "player_strategy_notes": "anystyle find (pdftotext text)",
            }
        finally:
            for path in (pdf_path, txt_path, bib_path):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass


register_adapter_class("AnystyleReferencesAdapter", AnystyleReferencesAdapter)
