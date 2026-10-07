"""LiteparseTablesHeuristicAdapter — bbox grid clustering on LiteParse text_items."""
from __future__ import annotations

import base64

from framework.player_adapter import PlayerAdapter, register_adapter_class

from players.adapters._liteparse_common import (
    LiteparseOcrConfigMixin,
    build_liteparse,
    iter_text_items,
    parse_pdf_bytes,
    strategy_suffix,
)
from players.adapters._liteparse_tables_heuristic import detect_tables_from_items


class LiteparseTablesHeuristicAdapter(LiteparseOcrConfigMixin, PlayerAdapter):

    def prepare(self) -> None:
        self._parser = build_liteparse(
            ocr_enabled=self.ocr_enabled,
            ocr_language=self.ocr_language,
            dpi=self.dpi,
            quiet=True,
        )

    def resolved_tool_version(self) -> str | None:
        from players.adapters._liteparse_common import liteparse_version
        return liteparse_version()

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if not hasattr(self, "_parser"):
            self.prepare()
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        result = parse_pdf_bytes(self._parser, pdf_bytes, timeout_s=timeout_s)
        items = list(iter_text_items(result))
        tables = detect_tables_from_items(items)
        return {
            "tables": tables,
            "player_strategy_notes": (
                "HEURISTIC: liteparse bbox row/column clustering; "
                f"n_items={len(items)} n_tables={len(tables)}; "
                + strategy_suffix(
                    ocr_enabled=self.ocr_enabled,
                    ocr_language=self.ocr_language,
                    dpi=self.dpi,
                )
            ),
        }


register_adapter_class("LiteparseTablesHeuristicAdapter", LiteparseTablesHeuristicAdapter)
