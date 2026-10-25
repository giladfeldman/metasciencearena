"""Derive every arena's gold from one Paper and its placement record.

All keys an arena consumes start with `gold` (framework/holdout.py strips exactly
those from held-out tasks). Formats mirror the current task sets of the five PDF
arenas (contracts mapped 2026-10-17) so their scorers run unchanged:

  text      -> gold_preserve_greek / gold_ascii_greek three-set profiles (+ pages)
  sections  -> gold_sections + gold_full_text
  tables    -> gold_tables
  references-> gold_references (scoring "v2")
  citations -> gold_references + gold_markers + gold_consistency
  stats     -> gold_stats (new arena)

Nothing here reads the rendered PDF: page numbers come from the painter.
"""
from __future__ import annotations

import importlib.util
import re
import unicodedata
from pathlib import Path

from .model import Paper

_REPO = Path(__file__).resolve().parents[2]
_LIGS = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
         "ﬅ": "st", "ﬆ": "st"}


def _load_tf_normalize():
    p = _REPO / "arenas" / "pdf-text-fidelity-v1" / "_normalize.py"
    spec = importlib.util.spec_from_file_location("pdfu_tf_normalize", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_TFN = None


def _tfn():
    global _TFN
    if _TFN is None:
        _TFN = _load_tf_normalize()
    return _TFN


def visible(s: str) -> str:
    """What the reader sees, as plain text: ligature glyphs read as their letters,
    invisible marks removed. (Lying glyph maps are post-processing; runs hold the
    visible character already.)"""
    s = "".join(_LIGS.get(ch, ch) for ch in s)
    # Mathematical alphanumerics (U+1D400-1D7FF, catalog F13) read as their letters.
    s = "".join(unicodedata.normalize("NFKC", ch) if 0x1D400 <= ord(ch) <= 0x1D7FF else ch for ch in s)
    s = s.replace("­", "").replace("​", "").replace("‎", "").replace("﻿", "")
    return unicodedata.normalize("NFC", s)


def _frag_text(paper: Paper, placement: dict, kinds: set[str]) -> list[tuple[int, str, str]]:
    return [(f["page"], f["block_id"], visible(f["text"])) for f in placement["fragments"]
            if f.get("kind") in kinds]


BODY_KINDS = {"abstract", "para", "list_item", "box", "caption", "table_note"}


# ----------------------------------------------------------------------- text
def text_gold(paper: Paper, placement: dict) -> dict:
    body = _frag_text(paper, placement, BODY_KINDS)
    pages = {}
    for pg, _, t in body:
        pages.setdefault(pg, []).append(t)
    n_pages = placement["pages"]
    body_raw = "\n\n".join(t for _, _, t in body)
    furniture = []
    for f in placement["furniture"]:
        furniture += f["strings"]
    refs_raw = "\n\n".join(visible(r.raw_text) for r in paper.references)
    meta_raw = "\n\n".join([visible(", ".join(f"{g} {s}" for s, g in paper.authors))] +
                           [visible(a) for a in paper.affiliations] +
                           [visible(x) for x in paper.front_matter.get("masthead", [])] +
                           [visible("; ".join(paper.keywords)), paper.journal.get("doi", "")])
    headings = "\n\n".join(t for _, _, t in _frag_text(paper, placement, {"heading", "title"}))
    nonbody = "\n\n".join(t for _, _, t in _frag_text(paper, placement, {"nonbody"}))
    footnotes = [t for _, _, t in _frag_text(paper, placement, {"footnote"})]
    cells = "\n\n".join(t for _, _, t in _frag_text(paper, placement, {"table"}))

    def profile(ascii_greek: bool) -> dict:
        norm = lambda s: _tfn().normalize_for_gold(s, ascii_greek=ascii_greek)  # noqa: E731
        return {
            "must_include": norm(body_raw),
            "must_exclude": {"page_furniture": norm("\n\n".join(furniture)),
                             "reference": norm(refs_raw),
                             "article_meta": norm(meta_raw)},
            "optional": {"heading": norm(headings), "footnote": "\n\n".join(norm(f) for f in footnotes),
                         "table_cells": norm(cells), "back_matter": norm(nonbody)},
            "footnotes": [norm(f) for f in footnotes],
            "pages": [norm("\n\n".join(pages.get(p, []))) for p in range(1, n_pages + 1)],
        }

    return {"gold_preserve_greek": profile(False), "gold_ascii_greek": profile(True),
            "column_count": paper.layout.get("columns", 1)}


# ------------------------------------------------------------------- sections
_LABELS = {"Introduction": "introduction", "Method": "methods", "Methods": "methods",
           "Results": "results", "Discussion": "discussion", "General Discussion": "discussion",
           "Conclusion": "conclusion", "Conclusions": "conclusion"}


def sections_gold(paper: Paper) -> dict:
    chunks, sections = [], []
    abs_text = visible("".join(r.text for r in paper.abstract)).strip()
    sections.append({"label": "abstract", "heading_text": "Abstract"})
    chunks.append(abs_text)
    cur = None
    for b in paper.blocks:
        if b.kind == "heading":
            # the heading's MEANING decides its label (ALL CAPS, numbers, splits are print form)
            canon = b.layout.get("canonical") or visible(b.text).strip()
            name = canon
            lab = _LABELS.get(canon)
            if b.level == 1 or lab:
                cur = {"label": lab or "other", "heading_text": name, "_paras": []}
                sections.append(cur)
                chunks.append(None)
            continue
        if b.kind in ("para", "list_item", "box") and cur is not None:
            cur["_paras"].append(visible(b.text).strip())
    out_chunks = []
    for s, c in zip(sections, chunks):
        out_chunks.append(c if c is not None else "\n\n".join(s.pop("_paras")))
    keep = [(s, c) for s, c in zip(sections, out_chunks) if c]
    full, pos, out = [], 0, []
    for i, (s, c) in enumerate(keep):
        out.append({"label": s["label"], "heading_text": s["heading_text"], "section_index": i,
                    "char_start": pos, "char_end": pos + len(c)})
        full.append(c)
        pos += len(c) + 2
    return {"gold_sections": out, "gold_full_text": "\n\n".join(full),
            "section_count": len(out)}


# --------------------------------------------------------------------- tables
def tables_gold(paper: Paper, placement: dict) -> dict:
    page_of = {}
    for f in placement["fragments"]:
        page_of.setdefault(f["block_id"], f["page"])
    out = []
    tables = []
    for b in paper.blocks:
        if b.kind == "table" and b.table is not None:
            tables.append((b, b.table, f"{b.block_id}:caption"))
            if b.layout.get("side_by_side") is not None:
                tables.append((b, b.layout["side_by_side"], f"{b.block_id}:caption2"))
    for b, t, cap_key in tables:
        rows = t.header + t.body
        cells = []
        for r, row in enumerate(rows):
            c = 0
            for cell in row:
                cells.append({"r": r, "c": c, "rowspan": cell.rowspan, "colspan": cell.colspan,
                              "text": visible(cell.text), "is_header": cell.is_header})
                c += cell.colspan
        out.append({"label": t.label if t.caption_style != "untitled" and t.label else None,
                    "page": page_of.get(cap_key) or page_of.get(f"{b.block_id}:table"),
                    "caption": visible("".join(r.text for r in t.caption)).strip() or None,
                    "n_rows": len(rows), "n_cols": max(sum(c.colspan for c in row) for row in rows),
                    "header_rows": len(t.header), "cells": cells})
    return {"gold_tables": out, "n_tables": len(out)}


# ------------------------------------------------------------ refs + citations
def references_gold(paper: Paper) -> dict:
    refs = []
    for r in paper.references:
        refs.append({"reference_id": r.ref_id,
                     "authors": [{"surname": s, "given_names": g or None} for s, g in r.authors],
                     "year": re.sub(r"[a-h]$", "", r.year) if r.year[:4].isdigit() else r.year,
                     "title": r.title, "venue": r.venue, "volume": r.volume, "issue": r.issue,
                     "fpage": r.fpage, "lpage": r.lpage,
                     "doi": r.doi if r.doi and r.doi in r.raw_text else None,
                     "pmid": None, "raw_text": visible(r.raw_text)})
    return {"scoring": "v2", "gold_references": refs, "reference_count": len(refs)}


def citations_gold(paper: Paper) -> dict:
    g = references_gold(paper)
    ref_ids = {r["reference_id"] for r in g["gold_references"]}
    markers = [{"marker_text": m["marker_text"], "reference_ids": list(m["reference_ids"])}
               for m in getattr(paper, "citation_markers", [])]
    cited = {i for m in markers for i in m["reference_ids"]}
    orphan = [m["marker_text"] for m in markers if not set(m["reference_ids"]) <= ref_ids]
    return {**g, "gold_markers": markers,
            "gold_consistency": {"orphan_markers": orphan,
                                 "uncited_reference_ids": sorted(ref_ids - cited),
                                 "duplicate_reference_groups": getattr(paper, "duplicate_groups", [])}}


# ---------------------------------------------------------------------- stats
def stats_gold(paper: Paper, placement: dict) -> dict:
    pages_of_block: dict[str, list[int]] = {}
    for f in placement["fragments"]:
        pages_of_block.setdefault(f["block_id"].split(":")[0], []).append(f["page"])
    where: dict[str, dict] = {}
    for b in paper.blocks:
        runs = b.runs if b.kind != "table" else [r for row in b.table.header + b.table.body
                                                  for c in row for r in c.runs]
        for r in runs:
            if r.stat_id:
                w = where.setdefault(r.stat_id, {"blocks": [], "in_table": False, "pages": set()})
                if b.block_id not in w["blocks"]:
                    w["blocks"].append(b.block_id)
                w["in_table"] = w["in_table"] or b.kind == "table"
                w["pages"].update(pages_of_block.get(b.block_id, []))
    defects_by_target: dict[str, list[str]] = {}
    for d in paper.defects:
        if d.get("target"):
            defects_by_target.setdefault(d["target"], []).append(d["id"])
    out = []
    for sid, rec in paper.stats.items():
        w = where.get(sid, {"blocks": [], "in_table": False, "pages": set()})
        g = rec.as_gold()
        g["location"] = {"blocks": w["blocks"], "in_table": w["in_table"], "pages": sorted(w["pages"])}
        g["defects"] = sorted(set(defects_by_target.get(sid, [])))
        g["printed_visible"] = visible(rec.printed.get("text_visible", rec.printed["text"]))
        out.append(g)
    return {"gold_stats": out, "stat_count": len(out)}


def all_gold(paper: Paper, placement: dict) -> dict:
    return {"text": text_gold(paper, placement), "sections": sections_gold(paper),
            "tables": tables_gold(paper, placement), "references": references_gold(paper),
            "citations": citations_gold(paper), "stats": stats_gold(paper, placement),
            "defects": paper.defects, "paper_id": paper.paper_id}
