"""LiteparseTextAdapter — LiteParse spatial text for pdf-text-fidelity-v1."""
from __future__ import annotations

import base64

from framework.player_adapter import PlayerAdapter, register_adapter_class

from players.adapters._liteparse_common import (
    build_liteparse,
    full_text_from_pages,
    pages_from_result,
    parse_pdf_bytes,
    strategy_suffix,
)


class LiteparseTextAdapter(PlayerAdapter):
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    dpi: float = 150
    strip_front_matter: bool = False

    def __init__(
        self,
        *args,
        ocr_enabled: bool = True,
        ocr_language: str = "eng",
        dpi: float = 150,
        strip_front_matter: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.dpi = dpi
        self.strip_front_matter = strip_front_matter
        self._parser = None

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
        if self._parser is None:
            self.prepare()
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        result = parse_pdf_bytes(self._parser, pdf_bytes, timeout_s=timeout_s)
        pages = pages_from_result(result)
        full_text = full_text_from_pages(pages)
        if self.strip_front_matter:
            full_text, pages = _strip_imrad_body(full_text, pages)
        extra = "strip_front_matter=true" if self.strip_front_matter else ""
        return {
            "full_text": full_text,
            "pages": pages,
            "footnotes": [],
            "player_strategy_notes": strategy_suffix(
                ocr_enabled=self.ocr_enabled,
                ocr_language=self.ocr_language,
                dpi=self.dpi,
                extra=extra,
            ),
        }


def _strip_imrad_body(full_text: str, pages: list[str]) -> tuple[str, list[str]]:
    """Truncate at References and drop lines before first IMRaD-like heading."""
    import re

    lines = full_text.split("\n")
    start_idx = 0
    heading_re = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
        r"(abstract|introduction|background|methods|materials and methods|results|discussion)\s*:?\s*$",
        re.IGNORECASE,
    )
    for i, line in enumerate(lines):
        if heading_re.match(line.strip()):
            start_idx = i
            break
    ref_re = re.compile(r"^\s*(?:\d+\.?\s+)?references\s*:?\s*$", re.IGNORECASE)
    end_idx = len(lines)
    for i, line in enumerate(lines[start_idx:], start=start_idx):
        if ref_re.match(line.strip()):
            end_idx = i
            break
    trimmed = "\n".join(lines[start_idx:end_idx]).strip()
    if not trimmed:
        return full_text, pages
    new_pages = [trimmed]
    return trimmed, new_pages


register_adapter_class("LiteparseTextAdapter", LiteparseTextAdapter)
