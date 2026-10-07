"""DoclingTextAdapter — Docling for pdf-text-fidelity-v1.

Two players share this class (`docling-text`, `docling-vlm-text`), selected by
`docling_vlm` in registry.yaml. See `docling_tables.py` for why one class serves
both pipelines.

Output is assembled from the DoclingDocument tree in reading order rather than
from `export_to_markdown()`. The Docling technical report is explicit that
markdown and HTML "cannot retain all meta information", and this arena needs
per-page attribution, which markdown cannot carry. `full_text` is the pages
joined by a blank line because that is how the arena schema DEFINES it — deriving
it from a second export would let the two renderings drift apart.

Page headers and footers are dropped: the arena gold is JATS body text, which has
no running heads or page numbers, so keeping them would be scored as inserted
text. Footnotes go to their own field, which the schema provides.

A known Docling weakness is relevant to this arena specifically: two-column
reading order is an open, unfixed defect upstream
(github.com/docling-project/docling/issues/2201 — it can jump to the right column
before finishing the left). This arena carries a `column_count` difficulty axis,
so the per-tier breakdown quantifies that defect rather than merely repeating the
bug report.
"""
from __future__ import annotations

import base64

from framework.player_adapter import PlayerAdapter, register_adapter_class

from players.adapters import _docling_common as dc


class DoclingTextAdapter(PlayerAdapter):
    docling_vlm: bool = False
    docling_table_mode: str = "accurate"
    docling_artifacts_path: str | None = None

    def __init__(
        self,
        *args,
        docling_vlm: bool = False,
        docling_table_mode: str = "accurate",
        docling_artifacts_path: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.docling_vlm = bool(docling_vlm)
        self.docling_table_mode = docling_table_mode
        self.docling_artifacts_path = docling_artifacts_path
        self._converter = None

    def prepare(self) -> None:
        self._converter = dc.build_converter(
            vlm=self.docling_vlm,
            table_mode=self.docling_table_mode,
            artifacts_path=self.docling_artifacts_path,
        )

    def resolved_tool_version(self) -> str | None:
        return dc.version_stack(self.docling_vlm, self.docling_table_mode)

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._converter is None:
            self.prepare()
        from players.adapters._timeout import run_with_timeout

        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        filename = envelope["input"].get("filename") or "task.pdf"
        return run_with_timeout(
            lambda: self._extract(pdf_bytes, filename),
            timeout_s,
            label="docling text",
        )

    def _extract(self, pdf_bytes: bytes, filename: str) -> dict:
        doc = dc.convert_bytes(self._converter, pdf_bytes, filename)
        out = dc.project_text(doc)
        out["player_strategy_notes"] = dc.strategy_notes(
            self.docling_vlm, self.docling_table_mode
        )
        return out


register_adapter_class("DoclingTextAdapter", DoclingTextAdapter)
