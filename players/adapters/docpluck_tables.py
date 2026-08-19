"""DocpluckTablesAdapter — wrap docpluck.extract_pdf_structured for the
pdf-table-extraction-v1 arena.

Maps docpluck's `Table` records to the arena output schema:
  - drops `bbox` and `confidence` (would over-anchor scoring to docpluck's
    coordinate system).
  - `kind="isolated"` (Camelot detected a table but couldn't recover cells)
    is preserved as `cells=[]`, n_rows=0, n_cols=0 so the scorer treats
    it as a detection-level hit without cell credit.
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
        for t in tables:
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
            "player_strategy_notes": f"docpluck.extract_pdf_structured (table_extraction_version={_g(result, 'table_extraction_version', '?')})",
        }


register_adapter_class("DocpluckTablesAdapter", DocpluckTablesAdapter)
