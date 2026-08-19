"""Docling adapters: egress containment, projection fidelity, provenance.

The load-bearing test in this file is `test_converter_can_never_reach_a_remote_service`.
The egress gate classifies `DoclingTablesAdapter` / `DoclingTextAdapter` as LOCAL,
which is what permits them to run the held-out real-PMC corpus — copyrighted PDFs
that must not leave the machine. That classification is only true because
`_docling_common.build_converter` hard-codes `enable_remote_services=False` and
does not accept it from `registry.yaml`.

`test_heldout_egress_gate.py` records why a naming convention must never be the
safety mechanism (the five Grobid adapters had a configurable endpoint and no
cloud prefix, so one YAML edit would have shipped held-out PDFs off-machine with
no gate). The counterpart here: a local classification is only honest if the
adapter cannot be reconfigured into a remote one. This file is that assertion.
"""
from __future__ import annotations

import pytest

from players.adapters import _docling_common as dc

docling = pytest.importorskip("docling", reason="docling is an optional extra (pdf-docling)")


# --- caption / label splitting ----------------------------------------------

@pytest.mark.parametrize(
    "raw,label,caption",
    [
        ("Table 1. Descriptive statistics", "Table 1", "Descriptive statistics"),
        ("Table 3: Means and SDs", "Table 3", "Means and SDs"),
        ("TABLE 1. Results", "TABLE 1", "Results"),
        # A label-only caption is returned unchanged rather than emptied — same
        # conservative rule as players/adapters/docpluck_tables.py.
        ("Table 2", "Table 2", "Table 2"),
        ("Unlabelled caption", None, "Unlabelled caption"),
        (None, None, None),
        ("", None, ""),
    ],
)
def test_strip_caption_label(raw, label, caption):
    assert dc.strip_caption_label(raw) == (label, caption)


# --- egress containment ------------------------------------------------------

def test_converter_can_never_reach_a_remote_service():
    """The assertion the LOCAL egress classification rests on.

    To watch this fail, flip `opts.enable_remote_services` to True in
    `_docling_common.build_converter` and re-run: it goes red, as does the
    held-out egress reasoning that depends on it.
    """
    from docling.datamodel.base_models import InputFormat

    conv = dc.build_converter()
    opts = conv.format_to_options[InputFormat.PDF].pipeline_options
    assert opts.enable_remote_services is False, (
        "Docling would be permitted to call an external API — the held-out "
        "corpus is copyrighted and must not leave this machine"
    )
    assert opts.do_picture_description is False, (
        "picture description is the one default-path stage with a remote API "
        "option; it must stay off"
    )


def test_vlm_converter_is_also_contained():
    """The generative pipeline is local too — but only because we say so."""
    from docling.datamodel.base_models import InputFormat

    conv = dc.build_converter(vlm=True)
    opts = conv.format_to_options[InputFormat.PDF].pipeline_options
    assert opts.enable_remote_services is False


def test_pipeline_is_pinned_not_left_on_library_defaults():
    """Docling's defaults are not a contract; ours are.

    `ocr_options` defaults to `OcrAutoOptions()`, which picks whichever OCR
    engine happens to be installed — the instrument would then depend on the
    host machine. Enrichment stages are off by default upstream and pinned off
    here so a future Docling release cannot quietly add a generative stage to a
    player the board advertises as non-generative.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import TableFormerMode

    opts = dc.build_converter().format_to_options[InputFormat.PDF].pipeline_options
    assert opts.do_ocr is False
    assert opts.do_table_structure is True
    assert opts.table_structure_options.mode is TableFormerMode.ACCURATE
    assert opts.do_code_enrichment is False
    assert opts.do_formula_enrichment is False
    assert opts.do_picture_classification is False


def test_adapters_are_classified_local_by_the_egress_gate():
    from framework.runner import _LOCAL_ADAPTER_CLASSES, is_cloud_player

    for cls in ("DoclingTablesAdapter", "DoclingTextAdapter"):
        assert cls in _LOCAL_ADAPTER_CLASSES, "must be REVIEWED onto the allowlist"
        assert is_cloud_player({"player_id": "x", "adapter_class": cls}) is False


def test_pipeline_options_are_not_registry_configurable():
    """The egress-critical options must not be reachable from registry.yaml.

    `build_adapter` forwards only ADAPTER_EXTRA_KWARGS. If any of these names
    were ever added there, a one-line YAML edit could re-point a gate-approved
    local player at a remote API.
    """
    from framework.player_adapter import ADAPTER_EXTRA_KWARGS

    for forbidden in (
        "enable_remote_services",
        "do_picture_description",
        "picture_description_options",
        "vlm_options",
        "pipeline_cls",
    ):
        assert forbidden not in ADAPTER_EXTRA_KWARGS


# --- provenance ---------------------------------------------------------------

def test_version_stack_reports_the_whole_instrument():
    """`docling==X` alone does not identify what ran.

    Docling swapped its default layout model between versions (+23.5% mAP), so
    the record must name the stack and the pipeline, and must read installed
    DISTRIBUTION metadata rather than `__version__` (the liteparse 2.0.8 lesson —
    see test_tool_version_provenance.py).
    """
    std = dc.version_stack(vlm=False, table_mode="accurate")
    assert std.startswith("docling-")
    assert "docling_core-" in std
    assert "docling_ibm_models-" in std
    assert std.endswith("tableformer-accurate")

    vlm = dc.version_stack(vlm=True)
    assert vlm.endswith("vlm-granite-docling")
    assert vlm != std, "the two pipelines must never share a version string"


def test_strategy_notes_name_the_generative_pipeline_explicitly():
    """A reader of a run record must be able to tell the two apart."""
    assert "LOCAL GENERATIVE" in dc.strategy_notes(vlm=True, table_mode="accurate")
    assert "LOCAL GENERATIVE" not in dc.strategy_notes(vlm=False, table_mode="accurate")
    assert "ocr=disabled" in dc.strategy_notes(vlm=False, table_mode="accurate")


def test_registry_players_build_and_carry_their_pipeline_flag():
    import yaml

    import framework.runner  # noqa: F401 - self-registers adapter classes
    from framework.player_adapter import build_adapter
    from framework.registry import load_registry
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    entries = {e["player_id"]: e for e in load_registry(repo / "players" / "registry.yaml")}
    for pid, expect_vlm in (
        ("docling-tables", False),
        ("docling-text", False),
        ("docling-vlm-tables", True),
        ("docling-vlm-text", True),
    ):
        adapter = build_adapter(entries[pid])
        assert adapter.docling_vlm is expect_vlm, (
            f"{pid}: the registry's docling_vlm flag did not reach the adapter — "
            "see framework/tests/test_registry_keys_reach_adapters.py"
        )


# --- regressions from the Sonnet cross-model review, 2026-08-19 --------------
#
# Both were REPRODUCED against the real functions before being fixed; each test
# below was watched failing first. Finding 3 (a bare `except Exception` around
# `caption_text` that turned a crash into a plausible "no caption") was fixed by
# removing the swallow, so a genuine bug surfaces as a task error instead of a
# clean-looking low score.

class _Cell:
    """Minimal stand-in for docling_core TableCell."""

    def __init__(self, r, c, text, header=False, rowspan=1, colspan=1):
        self.start_row_offset_idx = r
        self.end_row_offset_idx = r + rowspan
        self.start_col_offset_idx = c
        self.end_col_offset_idx = c + colspan
        self.row_span = rowspan
        self.col_span = colspan
        self.text = text
        self.column_header = header


class _Data:
    def __init__(self, cells, n_rows, n_cols):
        self.table_cells = cells
        self.num_rows = n_rows
        self.num_cols = n_cols


class _Table:
    def __init__(self, data, caption="Table 1. X"):
        self.data = data
        self.prov = []
        self._caption = caption

    def caption_text(self, doc):
        return self._caption


class _Doc:
    def __init__(self, tables):
        self.tables = tables


def test_header_rows_is_not_inflated_by_a_mid_table_header_cell():
    """`header_rows` counted LEADING header rows, but was derived as
    max(end_row_offset_idx) over every header cell anywhere in the table.

    TableFormer tags `column_header` per cell from its OTSL `ched` token and
    nothing constrains those cells to the top of the table, so one repeated
    mid-table header (long tables, sectioned tables) or one misclassified cell
    set `header_rows` to that cell's row. Reproduced: a genuine 1-row header
    plus a stray header cell at row 6 of an 8-row table reported header_rows=7.

    That value is published as the `header_rows_wrong` finding
    (arenas/pdf-table-extraction-v1/scorer.py), so a correctly-parsed Docling
    table earned a false "structure wrong" citation in its feedback report.
    """
    cells = [_Cell(0, 0, "H1", header=True), _Cell(0, 1, "H2", header=True)]
    cells += [_Cell(r, c, f"v{r}{c}") for r in range(1, 8) for c in range(2)]
    cells.append(_Cell(6, 0, "REPEATED HEADER", header=True))

    out = dc.project_tables(_Doc([_Table(_Data(cells, 8, 2))]))
    assert out[0]["header_rows"] == 1


def test_header_rows_counts_a_multi_row_header():
    """The contiguous-from-row-0 rule must not under-count a real 2-row header."""
    cells = [
        _Cell(0, 0, "Predictor", header=True, rowspan=2),
        _Cell(0, 1, "Outcome A", header=True, colspan=2),
        _Cell(1, 1, "B", header=True),
        _Cell(1, 2, "SE", header=True),
        _Cell(2, 0, "X1"),
        _Cell(2, 1, "0.30"),
        _Cell(2, 2, "0.10"),
    ]
    out = dc.project_tables(_Doc([_Table(_Data(cells, 3, 3))]))
    assert out[0]["header_rows"] == 2


def test_header_rows_is_zero_when_nothing_is_a_header():
    cells = [_Cell(0, 0, "a"), _Cell(0, 1, "b"), _Cell(1, 0, "c"), _Cell(1, 1, "d")]
    out = dc.project_tables(_Doc([_Table(_Data(cells, 2, 2))]))
    assert out[0]["header_rows"] == 0


class _Prov:
    def __init__(self, page):
        self.page_no = page


class _Item:
    def __init__(self, text, page, label=None):
        from docling_core.types.doc.labels import DocItemLabel

        self.text = text
        self.label = label or DocItemLabel.TEXT
        self.prov = [_Prov(page)] if page else []


class _TextDoc:
    def __init__(self, items):
        self._items = items

    def iterate_items(self, **kwargs):
        return [(i, 0) for i in self._items]


def test_unpaged_text_stays_in_reading_order_instead_of_jumping_to_page_one():
    """An item with no page provenance was unconditionally PREPENDED to page 1.

    Reproduced: a paragraph appearing last in reading order was written into
    pages[0] ahead of the real page-1 text, giving
    'LATE UNPAGED PARAGRAPH\npage one body'. This one moves the headline score --
    pdf-text-fidelity-v1 computes primary directly from `full_text`
    (0.5*levenshtein + 0.5*token_f1) and also compares per page.

    The fix carries the last seen page forward, so an unpaged item lands where it
    actually reads.
    """
    doc = _TextDoc([
        _Item("page one body", 1),
        _Item("page two body", 2),
        _Item("LATE UNPAGED PARAGRAPH", None),
    ])
    out = dc.project_text(doc)
    assert out["pages"][0] == "page one body"
    assert out["pages"][1] == "page two body\nLATE UNPAGED PARAGRAPH"
    assert out["full_text"] == "page one body\n\npage two body\nLATE UNPAGED PARAGRAPH"


def test_unpaged_text_before_any_page_leads_the_first_page():
    """Items with no predecessor still have to go somewhere sensible: the front."""
    doc = _TextDoc([
        _Item("EARLY UNPAGED", None),
        _Item("page one body", 1),
    ])
    out = dc.project_text(doc)
    assert out["pages"][0] == "EARLY UNPAGED\npage one body"


def test_full_text_is_always_the_pages_joined_by_a_blank_line():
    """The schema DEFINES full_text this way; two renderings must not drift."""
    doc = _TextDoc([_Item("a", 1), _Item("b", 2), _Item("c", None)])
    out = dc.project_text(doc)
    assert out["full_text"] == "\n\n".join(out["pages"])


def test_a_caption_lookup_crash_is_not_silently_scored_as_no_caption():
    """A crash must surface as a task error, not as a plausible low score.

    The original code wrapped `caption_text` in a bare `except Exception` and
    returned None, which the scorer reads as a legitimate missing caption. A
    crashed module reporting a clean result is exactly what the portfolio
    "No pretending" rule forbids.
    """
    class _Exploding(_Table):
        def caption_text(self, doc):
            raise RuntimeError("broken caption ref")

    doc = _Doc([_Exploding(_Data([_Cell(0, 0, "x")], 1, 1))])
    with pytest.raises(RuntimeError):
        dc.project_tables(doc)
