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
    """Coerce one LLM-emitted marker object into the arena marker schema.

    `reference_ids` (task set v2: every work a grouped or ranged citation
    cites) is kept when the model gives it; `reference_id` stays for v1.
    """
    out = {
        "marker_text": str(rec.get("marker_text") or ""),
        "char_start": _int_or_none(rec.get("char_start")),
        "char_end": _int_or_none(rec.get("char_end")),
        "reference_id": _str_or_none(rec.get("reference_id")),
    }
    ids = rec.get("reference_ids")
    if isinstance(ids, list):
        out["reference_ids"] = [str(x).strip() for x in ids if x is not None and str(x).strip()]
    return out


def _coerce_citations(raw: str) -> dict:
    """Coerce an LLM response into {references, markers, consistency}.

    Strips ```json fences, falls back to the first {...} block. A missing or
    malformed `consistency` defaults to the all-empty report and a missing
    `references` to []. A reply with no parseable object carrying a `markers`
    list RAISES UnparseableModelReply: until 2026-09-28 such garbage became an
    empty answer and was scored as "the model found no citations".
    """
    from players.adapters.llm_pdf_references import (UnparseableModelReply,
                                                      _normalize_entry)
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
    if not isinstance(data, dict) or not isinstance(data.get("markers"), list):
        raise UnparseableModelReply("reply is not an object with a `markers` list")

    markers = [_normalize_marker(r) for r in data["markers"] if isinstance(r, dict)]
    raw_refs = data.get("references")
    references = []
    if isinstance(raw_refs, list):
        for i, r in enumerate(raw_refs):
            if isinstance(r, dict):
                ref = _normalize_entry(r, i + 1)
                # The id is what the markers point at, so keep the model's own
                # label rather than the synthesized "ref-N" when it gave one.
                ref["reference_id"] = _str_or_none(r.get("reference_id")) or f"ref-{i + 1}"
                references.append(ref)

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

    return {"references": references, "markers": markers, "consistency": consistency}


from framework.player_adapter import (  # noqa: E402
    VersionAbsence, VERSION_ABSENT_BY_DESIGN, VERSION_ABSENT_PROBE_FAILED,
    VERSION_ABSENT_NEEDS_PREPARE)

class LlmPdfCitationsAdapter(LlmCliPdfAdapter):
    """LLM-via-CLI citation matcher. ai-model / non-deterministic."""

    def version_absence_reason(self) -> VersionAbsence:
        """The declared model id IS the version. Nothing is missing."""
        return VersionAbsence(
            VERSION_ABSENT_BY_DESIGN,
            "an LLM player's version is the model id already in player_version; "
            "there is no separate installed tool to detect",
        )

    def _coerce_response(self, raw: str) -> dict:
        return _coerce_citations(raw)

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        out = super().play_task(envelope, timeout_s=timeout_s)
        out["player_strategy_notes"] = (
            f"{self.cli_command[0]} CLI subscription (citation matching"
            + (", pdftotext input)" if self.document_to_text else ")"))
        return out


register_adapter_class("LlmPdfCitationsAdapter", LlmPdfCitationsAdapter)
