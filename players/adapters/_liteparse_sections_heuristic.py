"""IMRaD section detection from plain text (used by LiteParse sections adapter)."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from framework.scoring.text import normalize_heading

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_MAP_PATH = REPO_ROOT / "arenas" / "pdf-section-structure-v1" / "label_map.yaml"

# Heading line -> raw label key for label_map.yaml liteparse: block
_HEADING_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^abstract\s*$", re.I), "abstract"),
    (re.compile(r"^introduction\s*$", re.I), "introduction"),
    (re.compile(r"^(?:background|related work|literature review)\s*$", re.I), "introduction"),
    (re.compile(r"^(?:methods|methodology|materials and methods|methods and materials)\s*$", re.I), "methods"),
    (re.compile(r"^results\s*$", re.I), "results"),
    (re.compile(r"^discussion\s*$", re.I), "discussion"),
    (re.compile(r"^(?:conclusion|conclusions|summary)\s*$", re.I), "conclusion"),
    (re.compile(r"^references\s*$", re.I), "references"),
    (re.compile(r"^(?:acknowledg(?:e)?ments?|funding)\s*$", re.I), "acknowledgments"),
    (re.compile(r"^appendix\s*$", re.I), "appendix"),
    (re.compile(r"^title\s*$", re.I), "title"),
]

_LABEL_MAP_CACHE: dict[str, str] | None = None


def _load_liteparse_label_map() -> dict[str, str]:
    global _LABEL_MAP_CACHE
    if _LABEL_MAP_CACHE is None:
        with LABEL_MAP_PATH.open("r", encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
        _LABEL_MAP_CACHE = dict(full.get("liteparse") or {})
    return _LABEL_MAP_CACHE


def raw_label_for_heading_line(line: str) -> str | None:
    """Return liteparse raw label key if line looks like a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    norm = normalize_heading(stripped)
    for pat, raw in _HEADING_PATTERNS:
        if pat.match(norm) or pat.match(stripped):
            return raw
    # Numbered headings: "1. Introduction"
    m = re.match(
        r"^(?:\d+(?:\.\d+)*\.?\s+)?(.+?)\s*:?\s*$",
        stripped,
        re.IGNORECASE,
    )
    if m:
        inner = normalize_heading(m.group(1))
        for pat, raw in _HEADING_PATTERNS:
            if pat.match(inner):
                return raw
    return None


def canonical_label(raw: str) -> str:
    return _load_liteparse_label_map().get(raw, "other")


def detect_sections_from_text(full_text: str) -> list[dict]:
    """Return arena-shaped sections with char spans in full_text."""
    if not full_text:
        return []

    lines = full_text.split("\n")
    # Build line-start offsets
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    headings: list[tuple[int, int, str, str]] = []  # char_start, line_idx, raw, heading_text
    for i, line in enumerate(lines):
        raw = raw_label_for_heading_line(line)
        if raw:
            headings.append((offsets[i], i, raw, line.strip()))

    if not headings:
        return [{
            "label": "other",
            "heading_text": None,
            "section_index": 0,
            "char_start": 0,
            "char_end": len(full_text),
            "page": None,
        }]

    sections: list[dict] = []
    for idx, (char_start, _line_i, raw, heading_text) in enumerate(headings):
        char_end = headings[idx + 1][0] if idx + 1 < len(headings) else len(full_text)
        sections.append({
            "label": canonical_label(raw),
            "heading_text": heading_text,
            "section_index": idx,
            "char_start": char_start,
            "char_end": char_end,
            "page": None,
        })
    return sections
