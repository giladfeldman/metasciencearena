"""Outline blocks -> pdf-section-structure-v1 sections.

Shared by the Docling, Marker and MinerU section players. The split of labour,
which is what `tool_native: false` + `wrapper: arena-label-map` on those players
discloses:

* THE TOOL decides which lines are headings (Docling's layout model labels
  `section_header` / `title`; Marker and MinerU mark them `#`). Every heading the
  tool marked becomes one section, flat, in reading order -- the same flat shape
  GROBID's TEI <div>s give the GROBID sections player.
* THE ARENA decides which canonical label a heading gets, because none of these
  tools emit one. It uses the arena's OWN cross-vendor label map, through the
  scorer's own `canonicalize_player_label` -- the exact function the scorer
  already applies to any off-vocabulary label a player sends. Applying it here
  rather than leaving it to the scorer changes nothing about the label a heading
  ends up with; it only makes the output schema-valid, since the schema's
  `label` is an enum. A heading the map does not know ("Participants",
  "Experiment 1") is `other`, which is also what the gold calls such headings.
* A Docling `title` item is labelled `title`: that IS Docling's classification.

Text before the first heading belongs to no section and is not wrapped in an
invented one. Character spans index `full_text`, the joined block text; they are
informational (span IoU is not in the primary score).
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

from framework.scoring.text import normalize_heading

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORER_PATH = REPO_ROOT / "arenas" / "pdf-section-structure-v1" / "scorer.py"


@lru_cache(maxsize=1)
def _canonicalize():
    spec = importlib.util.spec_from_file_location("_sections_scorer_for_labels", SCORER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.canonicalize_player_label


def label_for_heading(heading: str) -> str:
    return _canonicalize()(normalize_heading(heading))


def sections_from_outline(blocks: list[dict]) -> tuple[list[dict], str]:
    chunks: list[str] = []
    pos = 0
    sections: list[dict] = []
    for b in blocks:
        text = (b.get("text") or "").strip()
        if not text:
            continue
        start = pos
        chunks.append(text)
        pos += len(text) + 2
        if b.get("kind") in ("heading", "title"):
            if sections:
                sections[-1]["char_end"] = start
            sections.append({
                "label": "title" if b["kind"] == "title" else label_for_heading(text),
                "heading_text": text,
                "section_index": len(sections),
                "char_start": start,
                "char_end": None,
                "page": None,
            })
    full_text = "\n\n".join(chunks)
    if sections:
        sections[-1]["char_end"] = len(full_text)
    return sections, full_text
