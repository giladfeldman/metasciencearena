"""DocpluckLibraryAdapter — direct in-process import of the docpluck library.

Two registry entries are expected, distinguished by their `docpluck_level`
field: `academic` (ASCIIfies Greek + flattens sub/superscripts; aimed at
the `ascii_greek` gold profile) and `standard` (preserves Greek; aimed at
`preserve_greek`). The pdf-text-fidelity-v1 scorer evaluates against both
gold profiles and reports the higher score, so this adapter does NOT need
to declare a target — it just emits its natural output.

Output schema (pdf-text-fidelity-v1):
    {full_text: str, pages: [str], footnotes: [str], player_strategy_notes: str}

docpluck DOES separate footnotes and running headers structurally — but only
when its layout-aware F0 step runs, which requires passing a pdfplumber
``LayoutDoc`` to ``normalize_text(..., layout=...)``. With layout, F0 strips
repeating running headers/footers and moves footnotes into an appendix after a
``\n\f\f\n`` marker, exposing them as ``report.footnote_texts`` (docpluck >=
2.4.83). We therefore:
  * extract the layout and pass it to ``normalize_text`` so the body excludes
    running headers + footnotes (matching the gold's ``full_text`` convention),
  * surface ``report.footnote_texts`` in the ``footnotes`` field.
Both steps degrade gracefully on older docpluck (no ``extract_pdf_layout`` /
no ``footnote_texts``) — the adapter then behaves like the pre-2.4.83 version
(flat ``full_text``, empty ``footnotes``).
"""
from __future__ import annotations

import base64

from framework.player_adapter import PlayerAdapter, register_adapter_class


class DocpluckLibraryAdapter(PlayerAdapter):
    docpluck_level: str  # "academic" | "standard" | "none"

    def __init__(self, *args, docpluck_level: str = "academic", **kwargs):
        super().__init__(*args, **kwargs)
        self.docpluck_level = docpluck_level

    def prepare(self) -> None:
        # Import lazily so registry loading doesn't fail when docpluck is absent.
        import importlib
        self._dp = importlib.import_module("docpluck")
        # Resolve the NormalizationLevel enum value once.
        if self.docpluck_level == "academic":
            self._level = self._dp.NormalizationLevel.academic
        elif self.docpluck_level == "standard":
            self._level = self._dp.NormalizationLevel.standard
        elif self.docpluck_level == "none":
            self._level = self._dp.NormalizationLevel.none
        else:
            raise ValueError(f"unknown docpluck_level: {self.docpluck_level}")

    def resolved_tool_version(self) -> str | None:
        from players.adapters._tool_version import module_version
        return module_version("docpluck")

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if not hasattr(self, "_dp"):
            self.prepare()
        from players.adapters._timeout import run_with_timeout
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        return run_with_timeout(
            lambda: self._extract(pdf_bytes), timeout_s, label="docpluck text")

    def _extract(self, pdf_bytes: bytes) -> dict:
        raw_text, _method = self._dp.extract_pdf(pdf_bytes)

        # Pass the pdfplumber layout so docpluck's F0 step strips running
        # headers/footers and separates footnotes out of the body. Resolve
        # extract_pdf_layout from the top level (docpluck >= 2.4.83) or the
        # submodule (older); fall back to the flat path if pdfplumber/layout
        # extraction is unavailable.
        _extract_layout = getattr(self._dp, "extract_pdf_layout", None)
        if _extract_layout is None:
            try:
                from docpluck.extract_layout import extract_pdf_layout as _extract_layout
            except Exception:
                _extract_layout = None
        layout = None
        if _extract_layout is not None:
            try:
                layout = _extract_layout(pdf_bytes)
            except Exception:
                layout = None

        if layout is not None:
            normalized, report = self._dp.normalize_text(
                raw_text, self._level, layout=layout
            )
        else:
            normalized, report = self._dp.normalize_text(raw_text, self._level)

        # Footnotes: prefer the first-class report.footnote_texts (docpluck >=
        # 2.4.83); fall back to parsing the "\n\f\f\n" appendix that F0 appends.
        footnotes = list(getattr(report, "footnote_texts", ()) or [])
        if "\n\f\f\n" in normalized:
            body, appendix = normalized.split("\n\f\f\n", 1)
            if not footnotes:
                footnotes = [s.strip() for s in appendix.split("\n\n") if s.strip()]
        else:
            body = normalized

        return {
            "full_text": body,
            "pages": [body],
            "footnotes": footnotes,
            "player_strategy_notes": (
                f"docpluck level={self.docpluck_level} "
                f"(F0_layout={'on' if layout is not None else 'off'}, "
                f"footnotes={len(footnotes)})"
            ),
        }


register_adapter_class("DocpluckLibraryAdapter", DocpluckLibraryAdapter)
