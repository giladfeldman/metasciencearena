"""DocpluckTablesAdapter — wrap docpluck.extract_pdf_structured for the
pdf-table-extraction-v1 arena.

Maps docpluck's `Table` records to the arena output schema:
  - drops `bbox` and `confidence` (would over-anchor scoring to docpluck's
    coordinate system).
  - `kind="isolated"` (Camelot detected a table but couldn't recover cells)
    is preserved as `cells=[]`, n_rows=0, n_cols=0 so the scorer treats
    it as a detection-level hit without cell credit.
  - `caption_status="uncaptioned_candidate"` (docpluck >= 2.4.145: a Camelot
    grid no caption claims, id `u<n>`, no label, no caption) IS emitted and
    scored like any other table. OWNER DECISION 2026-09-28 (Gilad), made on
    measured cost: docpluck labels these "unverified", and withholding them
    would have scored the tables docpluck claims — but this player is
    maintainer-built (maintainer_coi) and withholding FAVOURED it. Measured at
    2.4.146 on the 24 held-out papers: withheld .6480, emitted .5432 (22 of 24
    papers down; table_extra 3 -> 67, table_missed 59 -> 44). The arena
    scores what a tool returns in its table list; a consumer of
    `extract_pdf_structured()["tables"]` receives these candidates too.
    Their count is stated in `player_strategy_notes`. Note the 15 extra
    matches may partly be a matcher artefact: `_caption_sim(None, None)` is
    1.0, so a caption-less candidate pairs with any caption-less gold table
    (queued as d-24891c; affects every table player).
"""
from __future__ import annotations

import base64
import importlib
import re

from framework.player_adapter import PlayerAdapter, register_adapter_class


# docpluck's structured `caption` repeats the label verbatim at the head of the
# caption ("Table 1. <desc>") and carries the label separately in `label`. The
# arena gold follows the JATS convention — `label` separate, `caption` = the
# description only — so strip a leading "<label>." / generic "Table N." prefix.
_CAPTION_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:Table|Figure|Tab\.?|Fig\.?)\s+\d+[A-Za-z]?(?:\.\d+)?\s*[.:]\s+",
    re.IGNORECASE,
)


# docpluck 2.4.145 `Table.caption_status` value for a grid no caption claims.
# Read from docpluck's source (CHANGELOG 2.4.145), not from the request that
# reported it, which called it "uncaptioned".
_UNCAPTIONED_CANDIDATE = "uncaptioned_candidate"


def _strip_caption_label(caption: str | None, label: str | None) -> str | None:
    """Return the caption description with any redundant ``Table N.`` /
    ``Figure N.`` label prefix removed. Conservative: never strips a caption
    down to the empty string (a label-only caption is returned unchanged)."""
    if not caption:
        return caption
    if label:
        stripped = re.sub(rf"^\s*{re.escape(label)}\s*[.:]\s+", "", caption, count=1)
        if stripped != caption and stripped.strip():
            return stripped.strip()
    stripped = _CAPTION_LABEL_PREFIX_RE.sub("", caption, count=1)
    return stripped.strip() if stripped.strip() else caption


class DocpluckTablesAdapter(PlayerAdapter):
    def prepare(self) -> None:
        self._dp = importlib.import_module("docpluck")

    def resolved_tool_version(self) -> str | None:
        from players.adapters._tool_version import module_version
        return module_version("docpluck")

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if not hasattr(self, "_dp"):
            self.prepare()
        from players.adapters._timeout import run_with_timeout
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        return run_with_timeout(
            lambda: self._extract(pdf_bytes), timeout_s, label="docpluck tables")

    def _extract(self, pdf_bytes: bytes) -> dict:
        result = self._dp.extract_pdf_structured(pdf_bytes, thorough=False, table_text_mode="raw")
        # docpluck 2.4.x returns a dict, not a dataclass. Support both shapes
        # defensively so the adapter doesn't silently emit zero tables if the
        # library changes its return type.
        def _g(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        tables = _g(result, "tables", []) or []
        out_tables = []
        n_uncaptioned = 0
        for t in tables:
            if _g(t, "caption_status", None) == _UNCAPTIONED_CANDIDATE:
                n_uncaptioned += 1
            cells_out = []
            if _g(t, "kind", "") == "structured":
                for c in (_g(t, "cells", []) or []):
                    cells_out.append({
                        "r": int(_g(c, "r", 0) or 0),
                        "c": int(_g(c, "c", 0) or 0),
                        "rowspan": max(1, int(_g(c, "rowspan", 1) or 1)),
                        "colspan": max(1, int(_g(c, "colspan", 1) or 1)),
                        "text": str(_g(c, "text", "") or ""),
                        "is_header": bool(_g(c, "is_header", False)),
                    })
            _label = _g(t, "label", None) or None
            out_tables.append({
                "label": _label,
                "page": int(_g(t, "page", 0) or 0) or None,
                "caption": _strip_caption_label(_g(t, "caption", None) or None, _label),
                "n_rows": int(_g(t, "n_rows", 0) or 0),
                "n_cols": int(_g(t, "n_cols", 0) or 0),
                "header_rows": int(_g(t, "header_rows", 0) or 0),
                "cells": cells_out,
            })
        return {
            "tables": out_tables,
            "player_strategy_notes": (
                f"docpluck.extract_pdf_structured (table_extraction_version="
                f"{_g(result, 'table_extraction_version', '?')}; "
                f"uncaptioned_candidates_emitted={n_uncaptioned})"),
        }


register_adapter_class("DocpluckTablesAdapter", DocpluckTablesAdapter)
