"""DocpluckSectionsAdapter — wrap docpluck.extract_sections for the
pdf-section-structure-v1 arena.

Docpluck emits an 18-label taxonomy (docpluck/sections/taxonomy.py); we
collapse to the arena's canonical 11 via label_map.yaml (docpluck:
section, owned by the arena). char_start/char_end are passed through
unchanged because docpluck's SectionedDocument uses the same coordinate
system as its normalized text.

Output schema:
    {sections: [{label, heading_text, section_index, char_start, char_end, page}],
     full_text: str,
     player_strategy_notes: str}
"""
from __future__ import annotations

import base64
import importlib
from pathlib import Path

import yaml

from framework.player_adapter import PlayerAdapter, register_adapter_class

REPO_ROOT = Path(__file__).resolve().parents[2]
ARENA_LABEL_MAP_PATH = REPO_ROOT / "arenas" / "pdf-section-structure-v1" / "label_map.yaml"


def _load_docpluck_label_map() -> dict[str, str]:
    with ARENA_LABEL_MAP_PATH.open("r", encoding="utf-8") as f:
        full = yaml.safe_load(f) or {}
    return (full.get("docpluck") or {})


class DocpluckSectionsAdapter(PlayerAdapter):
    def prepare(self) -> None:
        self._dp = importlib.import_module("docpluck")
        self._label_map = _load_docpluck_label_map()

    def resolved_tool_version(self) -> str | None:
        from players.adapters._tool_version import module_version
        return module_version("docpluck")

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if not hasattr(self, "_dp"):
            self.prepare()
        from players.adapters._timeout import run_with_timeout
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        return run_with_timeout(
            lambda: self._extract(pdf_bytes), timeout_s, label="docpluck sections")

    def _extract(self, pdf_bytes: bytes) -> dict:
        doc = self._dp.extract_sections(file_bytes=pdf_bytes)
        out_sections = []
        for i, sec in enumerate(getattr(doc, "sections", []) or []):
            canonical = (sec.canonical_label.value
                         if hasattr(sec, "canonical_label") and hasattr(sec.canonical_label, "value")
                         else str(getattr(sec, "canonical_label", "")))
            raw_label = canonical or getattr(sec, "label", "")
            label = self._label_map.get(str(raw_label).lower(), "other")
            pages = list(getattr(sec, "pages", ()) or ())
            out_sections.append({
                "label": label,
                "heading_text": getattr(sec, "heading_text", None) or None,
                "section_index": i,
                "char_start": int(getattr(sec, "char_start", 0) or 0),
                "char_end": int(getattr(sec, "char_end", 0) or 0),
                "page": (pages[0] if pages else None),
            })
        return {
            "sections": out_sections,
            "full_text": getattr(doc, "normalized_text", "") or "",
            "player_strategy_notes": f"docpluck.extract_sections (sectioning_version={getattr(doc, 'sectioning_version', '?')})",
        }


register_adapter_class("DocpluckSectionsAdapter", DocpluckSectionsAdapter)
