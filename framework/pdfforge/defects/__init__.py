"""Defect recipes: catalog rows (docs/superpowers/specs/2026-10-17-pdf-universe-defect-catalog.md)
turned into code that paints one failure mode into a Paper.

A recipe has up to two halves:
  apply(paper, rng)      -> changes the paper (content, runs, layout directives) and
                            records itself with paper.add_defect(id, target, ...)
  post(pdf_bytes, paper) -> rewrites the rendered PDF (font maps, page boxes)

Metadata on every recipe: family, title, corrupts (Tx Se Tb Rf Ci St), recoverable
("yes" | "glyph" | "no"; "glyph" = recoverable only from glyph/layout evidence, "no" =
known-impossible stratum, never in the main score), sources (catalog provenance).

`apply_defects(paper, ids, rng)` applies a list; `post_process(pdf, paper)` runs every
applied recipe's post step in order. A recipe that finds nothing to act on (e.g. no
negative statistic for a minus defect) records nothing and returns False, so the
corpus builder can pick another paper instead of shipping a no-op.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from ..model import Paper, Run


@dataclass
class Recipe:
    defect_id: str
    family: str
    title: str
    corrupts: str
    recoverable: str
    sources: str
    apply: Callable[[Paper, random.Random], bool]
    post: Callable[[bytes, Paper, dict], bytes] | None = None
    tags: list[str] = field(default_factory=list)
    # How the manifest proof judges this recipe (build.manifest_proof):
    #   "diff"     extraction differs from the clean twin's (layout, furniture, content)
    #   "visible"  extraction disagrees with what the page SHOWS (lying glyph maps)
    #   "notation" the printed variant survives extraction (the failure is the parser's)
    probe: str = "diff"


REGISTRY: dict[str, Recipe] = {}


def recipe(defect_id: str, family: str, title: str, corrupts: str, recoverable: str, sources: str,
           post=None, tags=None, probe: str | None = None):
    def deco(fn):
        REGISTRY[defect_id] = Recipe(defect_id, family, title, corrupts, recoverable, sources, fn,
                                     post, tags or [], probe or ("visible" if post and defect_id.startswith("F") else "diff"))
        return fn
    return deco


def apply_defects(paper: Paper, ids: list[str], rng: random.Random) -> list[str]:
    """Apply recipes in order; return the ids that actually changed the paper."""
    # Content first (statistics, tables, headings, references, citations), then layout and
    # furniture, then font lies last -- a lie must see the final printed characters.
    order = {f: k for k, f in enumerate("STRHBCLPF")}
    ids = sorted(ids, key=lambda i: (order.get(REGISTRY[i].family, 9), ids.index(i)))
    done = []
    for i in ids:
        r = REGISTRY[i]
        if r.apply(paper, rng):
            done.append(i)
    return done


def post_process(pdf: bytes, paper: Paper) -> bytes:
    applied = []
    for d in paper.defects:
        if d["id"] not in applied:
            applied.append(d["id"])
    ctx: dict = {}
    for i in applied:
        r = REGISTRY.get(i)
        if r and r.post:
            pdf = r.post(pdf, paper, ctx)
    return pdf


# ----------------------------------------------------------------- run helpers
def split_chars(runs: list[Run], pred: Callable[[str, int, Run], bool], **changes) -> list[Run]:
    """Split runs so each character matching pred becomes its own run with `changes`."""
    out: list[Run] = []
    for r in runs:
        if r.image or r.text == "\n":
            out.append(r)
            continue
        buf = ""
        for i, ch in enumerate(r.text):
            if pred(ch, i, r):
                if buf:
                    out.append(_clone(r, buf))
                    buf = ""
                out.append(_clone(r, ch, **changes))
            else:
                buf += ch
        if buf:
            out.append(_clone(r, buf))
    return out


def _clone(r: Run, text: str, **changes) -> Run:
    d = dict(text=text, style=r.style, font=r.font, stat_id=r.stat_id, cite_refs=r.cite_refs,
             note_id=r.note_id, image=r.image, gold_text=r.gold_text)
    d.update(changes)
    return Run(**d)


def stat_blocks(paper: Paper, in_table: bool | None = False):
    """(block, stat_id) for every statistic, body paragraphs first."""
    seen = []
    for b in paper.blocks:
        if b.kind == "table" and in_table is False:
            continue
        if b.kind != "table" and in_table is True:
            continue
        runs = b.runs if b.kind != "table" else [r for row in b.table.header + b.table.body for c in row for r in c.runs]
        for r in runs:
            if r.stat_id and (b, r.stat_id) not in seen:
                seen.append((b, r.stat_id))
    return seen


def map_stat_runs(paper: Paper, stat_id: str, fn: Callable[[list[Run]], list[Run]]) -> None:
    """Apply fn to the runs of every block holding stat_id (only its runs change)."""
    for b in paper.blocks:
        if b.kind == "table":
            for row in b.table.header + b.table.body:
                for c in row:
                    if any(r.stat_id == stat_id for r in c.runs):
                        c.runs = _map_segment(c.runs, stat_id, fn)
        elif any(r.stat_id == stat_id for r in b.runs):
            b.runs = _map_segment(b.runs, stat_id, fn)


def _map_segment(runs: list[Run], stat_id: str, fn) -> list[Run]:
    out, seg = [], []
    for r in runs + [None]:
        if r is not None and r.stat_id == stat_id:
            seg.append(r)
            continue
        if seg:
            out += fn(seg)
            seg = []
        if r is not None:
            out.append(r)
    return out


# import families so their recipes register
from . import fonts, layout, furniture, tables, notation, structure, refs  # noqa: E402,F401
