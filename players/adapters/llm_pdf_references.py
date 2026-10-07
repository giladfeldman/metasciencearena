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


class UnparseableModelReply(RuntimeError):
    """The model answered, but not with the JSON object the prompt asks for.

    Raised, never coerced to an empty answer. Until 2026-09-28 garbage became
    {"references": []}, which the runner scored as "the model found nothing" --
    a truncated or chatty reply published as a zero for the model's skill,
    indistinguishable from a real miss. An error record says what happened.
    """


def _coerce_references(raw: str) -> dict:
    """Coerce an LLM response into {"references": [...]}.

    Strips ```json fences and falls back to the first {...} block; normalizes
    every entry to the arena reference schema. A reply with no parseable
    object carrying a `references` list raises UnparseableModelReply.
    """
    candidate = _strip_fences(raw)
    try:
        data = json.loads(candidate)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise UnparseableModelReply(f"no JSON object in reply ({len(raw)} chars): {raw[-200:]!r}")
        try:
            data = json.loads(m.group(0))
        except Exception as exc:
            raise UnparseableModelReply(
                f"reply JSON does not parse ({exc}); {len(raw)} chars, ends {raw[-200:]!r}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("references"), list):
        raise UnparseableModelReply("reply is not an object with a `references` list")
    raw_refs = data["references"]
    out_refs = [
        _normalize_entry(r, i + 1)
        for i, r in enumerate(raw_refs)
        if isinstance(r, dict)
    ]
    return {"references": out_refs}


from framework.player_adapter import (  # noqa: E402
    VersionAbsence, VERSION_ABSENT_BY_DESIGN, VERSION_ABSENT_PROBE_FAILED,
    VERSION_ABSENT_NEEDS_PREPARE)

class LlmPdfReferencesAdapter(LlmCliPdfAdapter):
    """LLM-via-CLI reference extractor. ai-model / non-deterministic."""

    def version_absence_reason(self) -> VersionAbsence:
        """The declared model id IS the version. Nothing is missing."""
        return VersionAbsence(
            VERSION_ABSENT_BY_DESIGN,
            "an LLM player's version is the model id already in player_version; "
            "there is no separate installed tool to detect",
        )

    def _coerce_response(self, raw: str) -> dict:
        return _coerce_references(raw)

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        out = super().play_task(envelope, timeout_s=timeout_s)
        out["player_strategy_notes"] = (
            f"{self.cli_command[0]} CLI subscription (references"
            + (", pdftotext input)" if self.document_to_text else ")"))
        return out


register_adapter_class("LlmPdfReferencesAdapter", LlmPdfReferencesAdapter)
