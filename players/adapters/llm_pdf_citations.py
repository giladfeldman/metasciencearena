"""LlmPdfCitationsAdapter — invoke a coding-CLI (claude / gemini / codex) on a
PDF and parse a JSON response into the pdf-citation-matching-v1 output shape.

Mirrors llm_pdf_references.py: inherits the PDF-materialization + CLI-invocation
scaffolding from LlmCliPdfAdapter, overrides the output coercer to expect a
{"markers": [...], "consistency": {...}} JSON object, and emits the
citation-matching arena output shape. play_task is a thin super() delegation
that sets player_strategy_notes — the scaffolding is not duplicated.

The prompt template lives at players/prompts/pdf_citation_matching.txt (same
directory as the section / table / reference-parsing templates) and instructs
the model to read the PDF, link in-text markers to reference list entries, and
emit the arena {markers, consistency} schema.

player_type: "ai-model", deterministic: false (CLI subscriptions vary run to
run; the framework repeats non-deterministic players for `trials` reps).
"""
from __future__ import annotations

import json
import re

from framework.player_adapter import register_adapter_class
from players.adapters.llm_pdf import LlmCliPdfAdapter, _strip_fences


def _str_or_none(value):
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _empty_consistency() -> dict:
    return {
        "orphan_markers": [],
        "uncited_reference_ids": [],
        "duplicate_reference_groups": [],
    }


def _str_list(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if x is not None]


def _group_list(raw) -> list[list[str]]:
    if not isinstance(raw, list):
        return []
    out: list[list[str]] = []
    for grp in raw:
        if isinstance(grp, list):
            out.append([str(x) for x in grp if x is not None])
    return out


def _normalize_marker(rec: dict) -> dict:
    """Coerce one LLM-emitted marker object into the arena marker schema."""
    return {
        "marker_text": str(rec.get("marker_text") or ""),
        "char_start": _int_or_none(rec.get("char_start")),
        "char_end": _int_or_none(rec.get("char_end")),
        "reference_id": _str_or_none(rec.get("reference_id")),
    }


def _coerce_citations(raw: str) -> dict:
    """Permissively coerce an LLM response into {markers, consistency}.

    Strips ```json fences, falls back to the first {...} block. A missing
    `markers` defaults to []; a missing / malformed `consistency` defaults to
    the all-empty report. Garbage -> empty markers + empty consistency.
    """
    candidate = _strip_fences(raw)
    try:
        data = json.loads(candidate)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {"markers": [], "consistency": _empty_consistency()}
        try:
            data = json.loads(m.group(0))
        except Exception:
            return {"markers": [], "consistency": _empty_consistency()}
    if not isinstance(data, dict):
        return {"markers": [], "consistency": _empty_consistency()}

    raw_markers = data.get("markers")
    if isinstance(raw_markers, list):
        markers = [_normalize_marker(r) for r in raw_markers
                   if isinstance(r, dict)]
    else:
        markers = []

    raw_cons = data.get("consistency")
    if isinstance(raw_cons, dict):
        consistency = {
            "orphan_markers": _str_list(raw_cons.get("orphan_markers")),
            "uncited_reference_ids": _str_list(
                raw_cons.get("uncited_reference_ids")),
            "duplicate_reference_groups": _group_list(
                raw_cons.get("duplicate_reference_groups")),
        }
    else:
        consistency = _empty_consistency()

    return {"markers": markers, "consistency": consistency}


class LlmPdfCitationsAdapter(LlmCliPdfAdapter):
    """LLM-via-CLI citation matcher. ai-model / non-deterministic."""

    def _coerce_response(self, raw: str) -> dict:
        return _coerce_citations(raw)

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        out = super().play_task(envelope, timeout_s=timeout_s)
        out["player_strategy_notes"] = (
            f"{self.cli_command[0]} CLI subscription (citation matching)")
        return out


register_adapter_class("LlmPdfCitationsAdapter", LlmPdfCitationsAdapter)
