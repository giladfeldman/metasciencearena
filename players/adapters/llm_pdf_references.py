"""LlmPdfReferencesAdapter — invoke a coding-CLI (claude / gemini / codex) on a
PDF and parse a JSON response into the pdf-reference-parsing-v1 output shape.

Mirrors llm_pdf_sections.py: inherits the PDF-materialization + CLI-invocation
scaffolding from LlmCliPdfAdapter, overrides the output coercer to expect a
{"references": [...]} JSON object, and emits the reference-arena output shape.

The prompt template lives at players/prompts/pdf_reference_parsing.txt (same
directory as the section / table-extraction templates) and instructs the model
to read the PDF and emit the arena reference schema.

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


def _normalize_entry(rec: dict, n: int) -> dict:
    """Coerce one LLM-emitted reference object into the arena schema."""
    authors: list[dict] = []
    raw_authors = rec.get("authors")
    if isinstance(raw_authors, list):
        for a in raw_authors:
            if isinstance(a, dict):
                surname = _str_or_none(a.get("surname") or a.get("family"))
                given = _str_or_none(a.get("given_names") or a.get("given"))
            elif isinstance(a, str):
                surname, given = _str_or_none(a), None
            else:
                continue
            if surname or given:
                authors.append({"surname": surname, "given_names": given})
    doi = _str_or_none(rec.get("doi"))
    year = _str_or_none(rec.get("year"))
    return {
        "reference_id": _str_or_none(rec.get("reference_id")) or f"ref-{n}",
        "authors": authors,
        "year": year,
        "title": _str_or_none(rec.get("title")),
        "venue": _str_or_none(rec.get("venue")),
        "volume": _str_or_none(rec.get("volume")),
        "issue": _str_or_none(rec.get("issue")),
        "fpage": _str_or_none(rec.get("fpage")),
        "lpage": _str_or_none(rec.get("lpage")),
        "doi": doi.lower() if doi else None,
        "pmid": _str_or_none(rec.get("pmid")),
        "raw_text": _str_or_none(rec.get("raw_text")) or "",
    }


def _coerce_references(raw: str) -> dict:
    """Permissively coerce an LLM response into {"references": [...]}.

    Strips ```json fences, falls back to the first {...} block, and normalizes
    every entry to the arena reference schema. Garbage -> {"references": []}.
    """
    candidate = _strip_fences(raw)
    try:
        data = json.loads(candidate)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {"references": []}
        try:
            data = json.loads(m.group(0))
        except Exception:
            return {"references": []}
    if not isinstance(data, dict):
        return {"references": []}
    raw_refs = data.get("references")
    if not isinstance(raw_refs, list):
        return {"references": []}
    out_refs = [
        _normalize_entry(r, i + 1)
        for i, r in enumerate(raw_refs)
        if isinstance(r, dict)
    ]
    return {"references": out_refs}


class LlmPdfReferencesAdapter(LlmCliPdfAdapter):
    """LLM-via-CLI reference extractor. ai-model / non-deterministic."""

    def _coerce_response(self, raw: str) -> dict:
        return _coerce_references(raw)

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        out = super().play_task(envelope, timeout_s=timeout_s)
        out["player_strategy_notes"] = (
            f"{self.cli_command[0]} CLI subscription (references)")
        return out


register_adapter_class("LlmPdfReferencesAdapter", LlmPdfReferencesAdapter)
