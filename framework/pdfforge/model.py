"""The in-memory paper every PDF-universe task is rendered from.

One `Paper` is the single source of truth: the renderer paints it, and every arena's
gold (text, sections, tables, references, citations, statistics) is DERIVED from it
(`gold.py`), never searched for in the rendered output. A defect changes the paper or
how it is painted and records itself in `paper.defects`, so the gold always knows which
defect touched which statistic, cell or reference.

Text is held as `Run`s: a string plus a style (regular, italic, bold, superscript,
subscript), an optional font role (math / a per-defect font, so a post-processor can
lie about exactly those glyphs and no others), and tags (the statistic or citation it
belongs to).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Run:
    text: str
    style: str = "r"                 # r | i | b | bi | sup | sub
    font: str | None = None          # None = body face; else a registered face name
    stat_id: str | None = None
    cite_refs: list[str] | None = None   # reference ids this run cites (a citation marker)
    note_id: str | None = None       # footnote marker
    image: str | None = None         # glyph drawn as an image, no character (catalog F16)
    gold_text: str | None = None     # what the reader SEES when `text` differs (lying glyph); None = text


@dataclass
class Cell:
    runs: list[Run]
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False

    @property
    def text(self) -> str:
        return "".join(r.gold_text if r.gold_text is not None else r.text for r in self.runs)


@dataclass
class Table:
    table_id: str
    label: str                       # "Table 1"
    caption: list[Run]
    header: list[list[Cell]]         # header rows (spans allowed)
    body: list[list[Cell]]
    note: list[Run] = field(default_factory=list)
    rotate: bool = False             # landscape table on a portrait page (L22)
    ruled: bool = True
    col_widths: list[float] | None = None
    caption_style: str = "apa"       # apa | pipe | label_alone | continued
    float_after: str | None = None   # block_id it is placed after (L13/L16 interleave)


@dataclass
class Figure:
    fig_id: str
    label: str
    caption: list[Run]
    ticks: list[str] = field(default_factory=list)    # axis tick labels painted in the chart (T19)


@dataclass
class Block:
    block_id: str
    kind: str                        # heading | para | table | figure | list_item | box
    runs: list[Run] = field(default_factory=list)
    level: int = 1                   # heading level
    number: str | None = None        # "2.1." heading number
    section_label: str | None = None  # gold label for a top-level heading (introduction, methods, ...)
    table: Table | None = None
    figure: Figure | None = None
    in_body: bool = True             # counts toward text gold must_include
    keep_with_next: bool = False
    layout: dict = field(default_factory=dict)   # painter directives (break_before, char_space, ...)

    @property
    def text(self) -> str:
        return "".join(r.gold_text if r.gold_text is not None else r.text for r in self.runs)


@dataclass
class Reference:
    ref_id: str
    authors: list[tuple[str, str]]   # (surname, given/initials)
    year: str                        # "2019", "2019a", "n.d.", "in press"
    title: str
    venue: str
    volume: str | None = None
    issue: str | None = None
    fpage: str | None = None
    lpage: str | None = None
    doi: str | None = None
    kind: str = "article"            # article | book | chapter | web | org
    raw_text: str = ""               # as printed in this paper's style (filled by refstyle)
    printed_runs: list[Run] = field(default_factory=list)


@dataclass
class Footnote:
    note_id: str
    marker: str
    runs: list[Run]


@dataclass
class Paper:
    paper_id: str
    seed: int
    split: str                       # revealed | private
    topic: str
    title: str
    authors: list[tuple[str, str]]
    affiliations: list[str]
    journal: dict                    # name, year, volume, issue, pages, doi
    abstract: list[Run]
    keywords: list[str]
    blocks: list[Block]
    references: list[Reference]
    cite_style: str                  # apa | harvard | aom | numeric | superscript
    ref_style: str                   # apa | harvard | aom | vancouver | ieee | nature
    footnotes: list[Footnote] = field(default_factory=list)
    stats: dict = field(default_factory=dict)        # stat_id -> StatRecord
    house_style: str = "plain"
    defects: list[dict] = field(default_factory=list)   # {"id", "target", "detail"}
    furniture: dict = field(default_factory=dict)       # painted outside the body: header, footer, stamp, ...
    front_matter: dict = field(default_factory=dict)    # masthead lines, correspondence, received dates
    layout: dict = field(default_factory=dict)          # columns, gutter, margins, page size

    def add_defect(self, defect_id: str, target: str | None = None, **detail) -> None:
        self.defects.append({"id": defect_id, "target": target, "detail": detail})

    def block(self, block_id: str) -> Block:
        for b in self.blocks:
            if b.block_id == block_id:
                return b
        raise KeyError(block_id)
