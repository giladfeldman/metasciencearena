"""Shared Docling plumbing: one converter builder, one projection.

Docling (https://docling.ai, IBM Research -> LF AI & Data, MIT) is a document
converter. Its DEFAULT pipeline contains no generative model: a C++ PDF parser
(docling-parse) feeds a DocLayNet-trained layout detector (RT-DETRv2) and
TableFormer, a vision transformer that predicts table structure. Table text is
matched back onto the PDF's own text cells rather than re-transcribed, so the
tool cannot invent a number that is not on the page. That makes it a LOCAL
SPECIALIST-MODEL tool -- the same egress category as GROBID and docpluck, and
categorically not an LLM player.

The OPT-IN `VlmPipeline` replaces that whole chain with a generative
vision-language model (granite-docling-258M). Still local, still no egress, but
now hallucination-capable. The two are wired as SEPARATE players precisely so
the board can show whether that distinction costs or buys accuracy.

WHY THE OPTIONS ARE HARD-CODED HERE AND NOT READ FROM registry.yaml
-------------------------------------------------------------------
`PdfPipelineOptions` can be pointed at a remote captioning/VLM API. The egress
gate classifies a player by its adapter class and declared endpoint, so a
registry-configurable pipeline would let one YAML edit ship held-out
(copyrighted) PDFs off-machine with the gate none the wiser -- exactly the
GROBID-endpoint hole recorded in `test_heldout_egress_gate.py`. The lesson there
was "the endpoint decides, not the class name"; the counterpart here is that a
local classification is only honest if the adapter cannot be reconfigured into a
remote one. So `enable_remote_services=False` is written in code and asserted by
`players/adapters/tests/test_docling_common.py`.

OCR IS OFF, DELIBERATELY
------------------------
`PdfPipelineOptions.ocr_options` defaults to `OcrAutoOptions()`, which selects
whichever OCR engine happens to be installed on the host. That makes the
instrument depend on the machine, which is the provenance defect this project
already hit with `resolved_tool_version`. Arena PDFs are born-digital with a
text layer, so OCR is disabled outright rather than left to auto-selection.

WHAT THE EGRESS GATE DOES *NOT* COVER
-------------------------------------
The gate classifies TASK egress -- whether a task's bytes leave the machine.
Nothing catches a background model download. Docling fetches weights from
HuggingFace on first use, so a run is only offline once they are cached;
`artifacts_path` pins them. Pre-fetch with `python -m docling.cli.models
download` before a tournament, or a run can silently pull new weights mid-flight
and the records either side are different instruments.
"""
from __future__ import annotations

import os
import re
from typing import Any

# Docling repeats the label at the head of the caption ("Table 1. <desc>"), while
# the arena gold follows the JATS convention: `label` separate, `caption` the
# description only. Same convention (and same conservative never-empty rule) as
# players/adapters/docpluck_tables.py.
_CAPTION_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:Table|Figure|Tab\.?|Fig\.?)\s+\d+[A-Za-z]?(?:\.\d+)?\s*[.:]?\s+",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(
    r"^\s*((?:Table|Figure|Tab\.?|Fig\.?)\s+\d+[A-Za-z]?(?:\.\d+)?)",
    re.IGNORECASE,
)


def strip_caption_label(caption: str | None) -> tuple[str | None, str | None]:
    """Split "Table 1. Descriptives" into ("Table 1", "Descriptives").

    Returns ``(label, caption)``. Never strips a caption to the empty string --
    a label-only caption keeps its text, matching the docpluck adapter.
    """
    if not caption:
        return None, caption
    text = caption.strip()
    label_m = _LABEL_RE.match(text)
    label = label_m.group(1).strip().rstrip(".") if label_m else None
    stripped = _CAPTION_LABEL_PREFIX_RE.sub("", text, count=1).strip()
    return label, (stripped or text)


def build_converter(
    *,
    vlm: bool = False,
    artifacts_path: str | None = None,
    table_mode: str = "accurate",
):
    """One `DocumentConverter`, configured for reproducible offline scoring.

    Imports are function-local so `players.adapters` still imports when docling
    is absent -- the runner logs a WARNING and skips the module rather than
    losing every other player (DR-0015).
    """
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    artifacts_path = artifacts_path or os.environ.get("DOCLING_ARTIFACTS_PATH") or None

    if vlm:
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.pipeline.vlm_pipeline import VlmPipeline

        opts = VlmPipelineOptions()
        opts.enable_remote_services = False
        if artifacts_path:
            opts.artifacts_path = artifacts_path
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline, pipeline_options=opts
                )
            }
        )

    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

    opts = PdfPipelineOptions()
    # Never negotiable: this is what makes the player LOCAL for the egress gate.
    opts.enable_remote_services = False
    opts.do_picture_description = False
    opts.do_picture_classification = False
    # Off by default already, pinned so a Docling default change cannot quietly
    # add a generative stage to a player we advertise as non-generative.
    opts.do_code_enrichment = False
    opts.do_formula_enrichment = False
    # See "OCR IS OFF, DELIBERATELY" above.
    opts.do_ocr = False
    opts.do_table_structure = True
    opts.table_structure_options.mode = (
        TableFormerMode.FAST if table_mode == "fast" else TableFormerMode.ACCURATE
    )
    if artifacts_path:
        opts.artifacts_path = artifacts_path
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def convert_bytes(converter, pdf_bytes: bytes, filename: str = "task.pdf"):
    """Convert in-memory PDF bytes to a `DoclingDocument`."""
    import io

    from docling.datamodel.base_models import DocumentStream

    stream = DocumentStream(name=filename, stream=io.BytesIO(pdf_bytes))
    return converter.convert(stream).document


def _page_of(item) -> int | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    page = getattr(prov[0], "page_no", None)
    return int(page) if isinstance(page, int) and page >= 1 else None


def project_tables(doc) -> list[dict]:
    """`DoclingDocument` -> arenas/pdf-table-extraction-v1 output schema.

    Field mapping (docling_core TableCell -> arena cell):
      start_row_offset_idx -> r          row_span      -> rowspan
      start_col_offset_idx -> c          col_span      -> colspan
      text                 -> text       column_header -> is_header

    `header_rows` is derived as the number of leading rows any column-header cell
    reaches, i.e. max(end_row_offset_idx) over header cells -- not a count of
    header cells, which would over-report a multi-column header row.

    bbox is deliberately dropped: scoring must not anchor to Docling's
    coordinate system (same reasoning as the docpluck adapter).
    """
    out: list[dict] = []
    for table in getattr(doc, "tables", None) or []:
        data = getattr(table, "data", None)
        cells_in = list(getattr(data, "table_cells", None) or []) if data else []
        cells: list[dict] = []
        header_row_idx: set[int] = set()
        for c in cells_in:
            is_header = bool(getattr(c, "column_header", False))
            if is_header:
                start = int(getattr(c, "start_row_offset_idx", 0) or 0)
                end = int(getattr(c, "end_row_offset_idx", start + 1) or start + 1)
                header_row_idx.update(range(start, max(end, start + 1)))
            cells.append({
                "r": max(0, int(getattr(c, "start_row_offset_idx", 0) or 0)),
                "c": max(0, int(getattr(c, "start_col_offset_idx", 0) or 0)),
                "rowspan": max(1, int(getattr(c, "row_span", 1) or 1)),
                "colspan": max(1, int(getattr(c, "col_span", 1) or 1)),
                "text": str(getattr(c, "text", "") or ""),
                "is_header": is_header,
            })
        # NOT wrapped in try/except. `caption_text` returns "" for a table with
        # no caption, which is the legitimate "absent" answer; an exception means
        # something is actually broken. Swallowing it would turn a crash into a
        # plausible low caption-recall score, which is indistinguishable from a
        # healthy run -- the failure mode the portfolio "No pretending" rule
        # exists to prevent. Let it surface as a task error instead.
        # (Sonnet cross-model review, 2026-08-19, finding 3.)
        raw_caption = table.caption_text(doc) or None
        label, caption = strip_caption_label(raw_caption)

        # LEADING header rows only. Deriving this as max(end_row_offset_idx) over
        # all header cells let a single repeated or misclassified mid-table
        # header cell report header_rows=7 on a table with a 1-row header
        # (reproduced; review finding 1). TableFormer tags `column_header` per
        # cell and nothing constrains those cells to the top of the table.
        header_rows = 0
        while header_rows in header_row_idx:
            header_rows += 1
        out.append({
            "label": label,
            "page": _page_of(table),
            "caption": caption,
            "n_rows": int(getattr(data, "num_rows", 0) or 0) if data else 0,
            "n_cols": int(getattr(data, "num_cols", 0) or 0) if data else 0,
            "header_rows": header_rows,
            "cells": cells,
        })
    return out


def project_text(doc) -> dict:
    """`DoclingDocument` -> arenas/pdf-text-fidelity-v1 output schema.

    Built from the document tree in reading order, NOT from
    `export_to_markdown()`: the Docling technical report states markdown and HTML
    "cannot retain all meta information", and we need per-page attribution. The
    arena schema DEFINES `full_text` as the pages joined by a blank line, so it
    is built that way rather than exported separately -- two independent
    renderings would drift.
    """
    from docling_core.types.doc.labels import DocItemLabel

    # WHAT COUNTS AS BODY TEXT
    # ------------------------
    # The arena gold is built by `_build_gold_full_text` in the generator, whose
    # contract is: abstract, then per section the paragraphs in order with
    # captions inline -- "Excludes: title, authors, headings, footnotes,
    # references."
    #
    # Every player must project onto that same definition or it is scored for
    # emitting text the gold never had. GROBID's adapter does this by taking only
    # TEI <p> elements (abstract + body paragraphs), which structurally drops
    # heads, title and the <back> bibliography. This is the Docling equivalent,
    # expressed in the labels Docling actually assigns -- a mapping decision, not
    # added capability.
    skip = {
        DocItemLabel.TITLE,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.DOCUMENT_INDEX,
        # Docling's own label for a bibliography entry. Where it applies this,
        # we honour it; where it does not, the references stay in the text and
        # cost us -- identifying the reference section is a GROBID capability
        # Docling does not claim, and faking it here would credit Docling with
        # arena code.
        DocItemLabel.REFERENCE,
    }

    pages: dict[int, list[str]] = {}
    footnotes: list[str] = []
    # An item can carry empty `prov` (page unresolved). Assigning those to page 1
    # put text that reads on page 9 at the FRONT of page 1, corrupting both
    # `pages[0]` and `full_text` -- and this arena scores primary directly off
    # `full_text`, so it moved the headline number (reproduced; review finding 2).
    # Carrying the last seen page forward keeps an unpaged item where it reads.
    current_page: int | None = None
    leading: list[str] = []
    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        if label == DocItemLabel.FOOTNOTE:
            footnotes.append(text)
            continue
        if label in skip:
            continue
        page = _page_of(item)
        if page is not None:
            current_page = page
        if current_page is None:
            # Nothing paged has been seen yet, so there is no page to inherit.
            leading.append(text)
            continue
        pages.setdefault(current_page, []).append(text)

    ordered = sorted(pages)
    if leading:
        if ordered:
            pages[ordered[0]] = leading + pages[ordered[0]]
        else:
            pages[1] = leading
            ordered = [1]
    page_strings = ["\n".join(pages[p]) for p in ordered]

    return {
        "full_text": "\n\n".join(page_strings),
        "pages": page_strings,
        "footnotes": footnotes,
    }


def version_stack(vlm: bool = False, table_mode: str = "accurate") -> str:
    """Identify the INSTRUMENT, not just the package.

    Docling swapped its default layout model between versions (+23.5% mAP), so
    `docling==X` alone does not say what ran. `resolved_tool_version` therefore
    reports the whole stack plus the pipeline configuration.
    """
    from players.adapters._tool_version import module_version

    # `module_version` imports the name first, so these must be MODULE names
    # (underscores), not distribution names (hyphens) -- the hyphenated form
    # silently returns None and would drop three quarters of the stack.
    parts = [module_version("docling") or "docling-?"]
    for mod in ("docling_core", "docling_ibm_models", "docling_parse"):
        v = module_version(mod)
        if v:
            parts.append(v)
    parts.append("vlm-granite-docling" if vlm else f"tableformer-{table_mode}")
    return "+".join(parts)


def strategy_notes(vlm: bool, table_mode: str) -> str:
    """Free-text provenance stored on every record."""
    pipeline = (
        "VlmPipeline/granite-docling-258M (LOCAL GENERATIVE)"
        if vlm
        else f"standard pipeline: layout + TableFormer({table_mode})"
    )
    # Thread count and device are part of the run configuration, not incidental:
    # torch results can move with either. Read the RESOLVED values rather than
    # guessing from the environment -- Docling's AcceleratorOptions default to
    # num_threads=4 regardless of OMP_NUM_THREADS, so reporting the env var
    # would have recorded "default" while 4 threads actually ran.
    try:
        from docling.datamodel.pipeline_options import AcceleratorOptions

        acc = AcceleratorOptions()
        accel = f"device={acc.device}; num_threads={acc.num_threads}"
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        accel = "device=?; num_threads=?"
    return (
        f"docling {version_stack(vlm, table_mode)}; {pipeline}; "
        f"ocr=disabled; enable_remote_services=False; {accel}"
    )
