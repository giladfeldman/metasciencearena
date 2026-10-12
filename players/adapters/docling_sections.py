"""DoclingSectionsAdapter — Docling for pdf-section-structure-v1.

Added 2026-10-07 with Docling 2.135.0. Docling's layout model labels each
heading `section_header` (and the document title `title`); this player turns
every heading Docling marked into one flat section, in reading order. Docling
does NOT say which canonical section a heading is -- that label comes from the
arena's own label map (players/adapters/_outline_sections.py), which is why the
registry entry carries `tool_native: false` and `wrapper: arena-label-map`, and
why the resolved version ends in `+arena-label-map`.

Same pipeline, same worker and same conversion as the Docling text and table
players: one `DocumentConverter` configuration across all three arenas.
"""
from __future__ import annotations

import base64

from framework.player_adapter import PlayerAdapter, register_adapter_class

from players.adapters import _docling_common as dc
from players.adapters._outline_sections import sections_from_outline

LABEL_STEP = "arena-label-map"


class DoclingSectionsAdapter(PlayerAdapter):
    def __init__(
        self,
        *args,
        docling_vlm: bool = False,
        docling_table_mode: str = "accurate",
        docling_artifacts_path: str | None = None,
        tool_env: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.docling_vlm = bool(docling_vlm)
        self.docling_table_mode = docling_table_mode
        self.docling_artifacts_path = docling_artifacts_path
        self.tool_env = tool_env
        self._worker = dc.out_of_process_worker(tool_env, docling_vlm, docling_table_mode)
        self._converter = None

    def prepare(self) -> None:
        if self._worker is not None:
            self._worker.start()
            return
        self._converter = dc.build_converter(
            vlm=self.docling_vlm,
            table_mode=self.docling_table_mode,
            artifacts_path=self.docling_artifacts_path,
        )

    def resolved_tool_version(self) -> str | None:
        try:
            if self._worker is not None:
                stack = self._worker.versions().get("resolved")
            else:
                stack = dc.version_stack(self.docling_vlm, self.docling_table_mode)
        except Exception:  # noqa: BLE001 - version detection never fails a task
            return None
        return f"{stack}+{LABEL_STEP}" if stack else None

    def cleanup(self) -> None:
        if self._worker is not None:
            self._worker.stop()

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._worker is not None:
            res = dc.convert_in_worker(self._worker, envelope, timeout_s)
            outline, notes = res["outline"], res["strategy_notes"]
        else:
            if self._converter is None:
                self.prepare()
            from players.adapters._timeout import run_with_timeout

            pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
            filename = envelope["input"].get("filename") or "task.pdf"
            doc = run_with_timeout(
                lambda: dc.convert_bytes(self._converter, pdf_bytes, filename),
                timeout_s, label="docling sections",
            )
            outline = dc.project_outline(doc)
            notes = dc.strategy_notes(self.docling_vlm, self.docling_table_mode)
        sections, full_text = sections_from_outline(outline)
        return {
            "sections": sections,
            "full_text": full_text,
            "player_strategy_notes": f"{notes}; headings = Docling section_header/title items; "
                                     f"labels = {LABEL_STEP} (arena code)",
        }


register_adapter_class("DoclingSectionsAdapter", DoclingSectionsAdapter)
