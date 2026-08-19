"""DoclingTablesAdapter — Docling for pdf-table-extraction-v1.

Two players share this class, distinguished by `docling_vlm` in registry.yaml:

  docling-tables      standard pipeline: layout detector + TableFormer(ACCURATE).
                      No generative model anywhere in the chain; table text is
                      matched back onto the PDF's own text cells.
  docling-vlm-tables  the opt-in VlmPipeline (granite-docling-258M). Local, no
                      egress, but GENERATIVE and therefore able to emit a cell
                      value that is not on the page.

Keeping them one class keeps the projection identical, so a score difference is
attributable to the pipeline rather than to two hand-written mappings drifting.

MEASURED BEHAVIOUR (2026-08-19, docling 2.120.3, 12 public tasks)
-----------------------------------------------------------------
The cell grid is excellent and the caption link is brittle. Eight of twelve
synthetic tasks score exactly 1.000 — spans, header rows, cell text and page all
correct, including the multi-row-header and two-table cases. Four score 0.000,
and every one of them for the same reason: Docling's layout model classified the
"Table 1. <caption>" line as `text` rather than `caption`, so it was never linked
to the table and `TableItem.caption_text()` returns "". The arena matches a
player table to a gold table by label or caption, so an unidentified table is an
unmatched table.

The caption text IS in the document — as an unlabelled paragraph. Recovering it
by grabbing the nearest preceding line would lift the score substantially and
would be dishonest: caption-to-table association is the capability under test,
and supplying it here would credit Docling with arena code. The zeros stand.
"""
from __future__ import annotations

import base64

from framework.player_adapter import PlayerAdapter, register_adapter_class

from players.adapters import _docling_common as dc


class DoclingTablesAdapter(PlayerAdapter):
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
            label="docling tables",
        )

    def _extract(self, pdf_bytes: bytes, filename: str) -> dict:
        doc = dc.convert_bytes(self._converter, pdf_bytes, filename)
        return {
            "tables": dc.project_tables(doc),
            "player_strategy_notes": dc.strategy_notes(
                self.docling_vlm, self.docling_table_mode
            ),
        }


register_adapter_class("DoclingTablesAdapter", DoclingTablesAdapter)
