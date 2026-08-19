"""LiteparseSectionsHeuristicAdapter — IMRaD regex on LiteParse text."""
from __future__ import annotations

import base64

from framework.player_adapter import PlayerAdapter, register_adapter_class

from players.adapters._liteparse_common import (
    LiteparseOcrConfigMixin,
    build_liteparse,
    full_text_from_pages,
    pages_from_result,
    parse_pdf_bytes,
    strategy_suffix,
)
from players.adapters._liteparse_sections_heuristic import detect_sections_from_text


class LiteparseSectionsHeuristicAdapter(LiteparseOcrConfigMixin, PlayerAdapter):

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
        pages = pages_from_result(result)
        full_text = full_text_from_pages(pages)
        sections = detect_sections_from_text(full_text)
        return {
            "sections": sections,
            "full_text": full_text,
            "player_strategy_notes": (
                "HEURISTIC: liteparse text + IMRaD heading regex; "
                + strategy_suffix(
                    ocr_enabled=self.ocr_enabled,
                    ocr_language=self.ocr_language,
                    dpi=self.dpi,
                )
            ),
        }


register_adapter_class("LiteparseSectionsHeuristicAdapter", LiteparseSectionsHeuristicAdapter)
