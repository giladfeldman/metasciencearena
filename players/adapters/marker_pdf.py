"""Marker (datalab-to/marker, marker-pdf 2.0.0) for the three PDF arenas.

Runs out of process in players/tool_envs/marker/ (see that worker for the fixed
configuration: fast mode, OCR disabled, no LLM). Code GPL-3.0; model weights
OpenRAIL-M, which permits research use.

  MarkerTextAdapter      markdown -> plain text (players/adapters/_markdown_projection.py)
  MarkerSectionsAdapter  markdown `#` headings -> flat sections; labels from the
                         arena label map (`tool_native: false`)
  MarkerTablesAdapter    Marker's JSON render: each Table block's own HTML, with
                         row/column spans, its page, and the Caption Marker
                         grouped with it in a TableGroup

WHY TABLES READ THE JSON AND NOT THE MARKDOWN. Measured 2026-10-07 on a
synthetic one-table PDF: in this configuration Marker's markdown printed the
caption and then NOTHING of the table -- the cells were absent from the text
entirely -- while the JSON held the Table block. Scoring tables off the markdown
would have measured the renderer, not the table recogniser.

The resolved version names the configuration as well as the release, because a
different `mode` or OCR setting is a different instrument (the Docling lesson):
`marker-pdf-<v>+surya-ocr-<v>+fast-noocr-<markdown|json>`.
"""
from __future__ import annotations

import base64

from framework.player_adapter import PlayerAdapter, register_adapter_class

from players.adapters._markdown_projection import marker_json_tables, md_outline, md_to_plain
from players.adapters._outline_sections import sections_from_outline
from players.adapters._tool_worker import ToolWorker, pdf_on_disk

CONFIG_TAG = "fast-noocr"
LABEL_STEP = "arena-label-map"


class _MarkerBase(PlayerAdapter):
    output_format = "markdown"
    version_suffix = ""

    def __init__(self, *args, tool_env: str | None = "marker", **kwargs):
        super().__init__(*args, **kwargs)
        if tool_env != "marker":
            raise ValueError(f"Marker players run only in tool_env 'marker', got {tool_env!r}")
        self.tool_env = tool_env
        # Weights are fetched once, before a tournament; never mid-run.
        self._worker = ToolWorker("marker", env={"HF_HUB_OFFLINE": "1"})

    def prepare(self) -> None:
        self._worker.start()

    def cleanup(self) -> None:
        self._worker.stop()

    def resolved_tool_version(self) -> str | None:
        try:
            v = self._worker.versions()
        except Exception:  # noqa: BLE001 - version detection never fails a task
            return None
        if not v.get("marker-pdf"):
            return None
        return (f"marker-pdf-{v['marker-pdf']}+surya-ocr-{v.get('surya-ocr')}"
                f"+{CONFIG_TAG}-{self.output_format}{self.version_suffix}")

    def version_absence_reason(self):
        from framework.player_adapter import VERSION_ABSENT_PROBE_FAILED, VersionAbsence

        return VersionAbsence(VERSION_ABSENT_PROBE_FAILED,
                              "the Marker venv did not report its version; is players/tool_envs/marker/.venv installed?")

    def _convert(self, envelope: dict, timeout_s: int) -> dict:
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        with pdf_on_disk(pdf_bytes, envelope["input"].get("filename") or "task.pdf") as path:
            return self._worker.call({"pdf_path": path, "output_format": self.output_format,
                                      "task_id": envelope.get("task_id")}, timeout_s)

    def _notes(self) -> str:
        return (f"{self.resolved_tool_version()}; ConfigParser(output_format={self.output_format!r}, "
                f"disable_ocr=True, mode='fast'); no LLM")


class MarkerTextAdapter(_MarkerBase):
    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        text = md_to_plain(self._convert(envelope, timeout_s)["markdown"]).strip()
        return {"full_text": text, "pages": [text], "footnotes": [],
                "player_strategy_notes": self._notes() + "; markdown -> plain text, nothing removed for scope"}


class MarkerSectionsAdapter(_MarkerBase):
    version_suffix = f"+{LABEL_STEP}"

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        sections, full_text = sections_from_outline(md_outline(self._convert(envelope, timeout_s)["markdown"]))
        return {"sections": sections, "full_text": full_text,
                "player_strategy_notes": self._notes() + f"; headings = markdown '#' lines; labels = {LABEL_STEP} (arena code)"}


class MarkerTablesAdapter(_MarkerBase):
    output_format = "json"

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        tables = marker_json_tables(self._convert(envelope, timeout_s)["json"])
        return {"tables": tables,
                "player_strategy_notes": self._notes() + "; Table blocks' HTML; caption only from Marker's TableGroup"}


register_adapter_class("MarkerTextAdapter", MarkerTextAdapter)
register_adapter_class("MarkerSectionsAdapter", MarkerSectionsAdapter)
register_adapter_class("MarkerTablesAdapter", MarkerTablesAdapter)
