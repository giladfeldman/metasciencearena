"""JSONL storage for run records, validated against the contract schema."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from jsonschema import Draft202012Validator

from framework.paths import schema_path

logger = logging.getLogger(__name__)

# Package data, not a checkout path — see framework/paths.py.
RUN_RECORD_SCHEMA_PATH = schema_path("run_record.schema.json")


class RunRecordValidationError(Exception):
    pass


def _load_validator() -> Draft202012Validator:
    with RUN_RECORD_SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return Draft202012Validator(json.load(f))


class RunRecordWriter:
    """Context manager that writes validated run records to a JSONL file.

    By default opens in APPEND mode (``overwrite=False``) — the historical
    behavior. Re-running a tournament with the same output path then SILENTLY
    DOUBLES every record, and downstream aggregation double-counts (the
    2026-06-07 operational trap, DR-0013). Two mitigations:

      * ``overwrite=True`` opens in 'w' mode, so a re-run replaces rather than
        appends (the ``framework run --overwrite`` CLI flag wires this).
      * In append mode, opening on a NON-EMPTY file logs a WARNING naming the
        path + existing size, so the doubling is at least visible.
    """

    def __init__(self, path: Path, *, overwrite: bool = False):
        self.path = path
        self.overwrite = overwrite
        self._validator = _load_validator()
        self._fh = None

    def __enter__(self) -> "RunRecordWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.overwrite:
            self._fh = self.path.open("w", encoding="utf-8")
        else:
            if self.path.exists() and self.path.stat().st_size > 0:
                logger.warning(
                    "Appending run records to a non-empty file (%s, %d bytes) — "
                    "records will be ADDED, not replaced. Pass overwrite=True "
                    "(or `framework run --overwrite`) to replace instead.",
                    self.path, self.path.stat().st_size,
                )
            self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def append(self, record: dict) -> None:
        errors = sorted(self._validator.iter_errors(record), key=lambda e: e.path)
        if errors:
            raise RunRecordValidationError("; ".join(e.message for e in errors))
        if self._fh is None:
            raise RuntimeError("RunRecordWriter not opened (use 'with')")
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        # Flush after every record so a crash, kill, or full disk loses at most
        # the in-flight record rather than the entire 8 KB write buffer. Cheap
        # at ~tens-of-records/second; vital for long LLM player runs.
        self._fh.flush()


def _parse_line(line: str, path: Path, lineno: int) -> dict | None:
    """Parse one JSONL line, tolerating a single torn trailing record.

    `append` flushes per record, so a killed run can still leave ONE partially
    written line (observed 2026-08-03: a tournament interrupted mid-write left a
    fragment that made the whole 12-record file unparseable). A torn line is
    unrecoverable data, but it must not poison every valid record beside it —
    losing a whole tournament to one truncated write is far worse than dropping
    the fragment.

    Logged at WARNING rather than swallowed, so a corrupt file is visible rather
    than silently shortening a leaderboard.
    """
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        logger.warning("%s:%d: skipping unparseable run record (%s): %.80r",
                       path, lineno, e, line)
        return None


def read_records(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line:
                rec = _parse_line(line, path, lineno)
                if rec is not None:
                    out.append(rec)
    return out


def iter_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line:
                rec = _parse_line(line, path, lineno)
                if rec is not None:
                    yield rec
