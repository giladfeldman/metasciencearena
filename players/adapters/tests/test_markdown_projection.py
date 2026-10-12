"""Markdown/HTML -> arena shapes, for the Marker and MinerU players.

Every number these players publish passes through this projection, so a silent
error here moves a score with no test failing. The text cases are ESCImate's
mdplain self-test (tests/harness/extractor_pilot/mdplain.py, 2026-10-07), kept
verbatim, plus the trap that file exists for.
"""
from __future__ import annotations

import pytest

from players.adapters._markdown_projection import (
    html_table, marker_json_tables, md_outline, md_to_plain, mineru_middle_tables,
)
from players.adapters._outline_sections import SCORER_PATH, label_for_heading, sections_from_outline

_needs_sections_scorer = pytest.mark.skipif(
    not SCORER_PATH.is_file(),
    reason="pdf-section-structure-v1 is a PRIVATE arena; its scorer is not in the public mirror",
)


# ------------------------------------------------------------------- plain text

@pytest.mark.parametrize("md,plain", [
    ("found at *p* < .05 and **t**(48) = 2.31", "found at p < .05 and t(48) = 2.31"),
    (r"$\eta^2_p$ = .12, $\chi^2(1, N = 100)$ = 3.9", "η²p = .12, χ²(1, N = 100) = 3.9"),
    ("| M | SD |\n|---|---|\n| 2.1 | 0.4 |", "M\tSD\n2.1\t0.4"),
    ("a_b_c snake_case and 2*3*4", "a_b_c snake_case and 2*3*4"),
    ("t(48) = -2.31", "t(48) = -2.31"),
    ("F(1, 742) = 101.10, p < .001, η<sup>2</sup>p = 0.12, CI[0.08, 0.17]), and x > y",
     "F(1, 742) = 101.10, p < .001, η²p = 0.12, CI[0.08, 0.17]), and x > y"),
    ("η<sub>p</sub><sup>2</sup> = .05, p<.01 and p>.10", "ηp² = .05, p<.01 and p>.10"),
])
def test_md_to_plain_matches_the_escimate_self_test(md, plain):
    assert md_to_plain(md) == plain


def test_a_bare_less_than_is_never_treated_as_a_tag():
    """The trap: `<[^>]+>` deletes from the statistic's '<' to the next '>'."""
    out = md_to_plain("p < .001, η<sup>2</sup> and p <.05 <span id='x'></span>end")
    assert "p < .001" in out and "p <.05" in out and out.endswith("end")
    assert "span" not in out


def test_headings_and_marker_spans_are_demarked_not_dropped():
    out = md_to_plain('## <span id="page-3-0"></span>**Results**\n\nWe found *d* = 0.4.')
    assert out.splitlines()[0] == "Results"
    assert "We found d = 0.4." in out


def test_images_go_and_link_text_stays():
    assert md_to_plain("see ![fig](a.png) the [OSF page](https://osf.io/x)") == "see  the OSF page"


# ---------------------------------------------------------------------- outline

def test_outline_marks_only_what_the_tool_marked_as_a_heading():
    md = "# Title Here\n\nAbstract text *p* < .05.\n\n## 1. Introduction\n\nBody one.\nstill body.\n\n### Participants\n\nBody two."
    blocks = md_outline(md)
    assert [(b["kind"], b["text"]) for b in blocks] == [
        ("heading", "Title Here"), ("text", "Abstract text p < .05."),
        ("heading", "1. Introduction"), ("text", "Body one.\nstill body."),
        ("heading", "Participants"), ("text", "Body two."),
    ]
    assert [b.get("level") for b in blocks if b["kind"] == "heading"] == [1, 2, 3]


@pytest.mark.parametrize("heading,label", [
    ("1. Introduction", "introduction"), ("Materials and Methods", "methods"),
    ("RESULTS", "results"), ("General Discussion", "discussion"),
    ("References", "references"), ("Acknowledgements", "acknowledgments"),
    ("Participants", "other"), ("Experiment 1", "other"),
])
@_needs_sections_scorer
def test_labels_come_from_the_arena_map_through_the_scorer_function(heading, label):
    assert label_for_heading(heading) == label


@_needs_sections_scorer
def test_sections_are_flat_with_spans_into_full_text_and_no_invented_leading_section():
    blocks = [{"kind": "text", "text": "preamble"}, {"kind": "title", "text": "A Paper"},
              {"kind": "heading", "text": "Methods"}, {"kind": "text", "text": "We did x."},
              {"kind": "heading", "text": "Results"}, {"kind": "text", "text": "It worked."}]
    sections, full = sections_from_outline(blocks)
    assert [(s["label"], s["heading_text"], s["section_index"]) for s in sections] == [
        ("title", "A Paper", 0), ("methods", "Methods", 1), ("results", "Results", 2)]
    for s in sections:
        assert full[s["char_start"]:].startswith(s["heading_text"])
    assert sections[-1]["char_end"] == len(full)
    assert sections[0]["char_end"] == sections[1]["char_start"]


def test_no_headings_means_no_sections():
    assert sections_from_outline([{"kind": "text", "text": "only text"}])[0] == []


# ----------------------------------------------------------------------- tables

def test_html_grid_honours_rowspan_and_colspan():
    t = html_table("<table><thead><tr><th rowspan='2'>Var</th><th colspan='2'>Group A</th></tr>"
                   "<tr><th>M</th><th>SD</th></tr></thead>"
                   "<tbody><tr><td>Age</td><td>40.1</td><td>10.8</td></tr></tbody></table>",
                   caption="Table 2. Descriptives", page=4)
    assert (t["label"], t["caption"], t["page"]) == ("Table 2", "Descriptives", 4)
    assert (t["n_rows"], t["n_cols"], t["header_rows"]) == (3, 3, 2)
    pos = {(c["r"], c["c"]): c for c in t["cells"]}
    assert pos[(0, 0)]["rowspan"] == 2 and pos[(0, 1)]["colspan"] == 2
    assert pos[(1, 1)]["text"] == "M" and pos[(1, 2)]["text"] == "SD"
    assert pos[(2, 0)]["text"] == "Age" and not pos[(2, 0)]["is_header"]


def test_header_rows_count_leading_rows_only():
    t = html_table("<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr>"
                   "<tr><th>c</th><th>d</th></tr></table>")
    assert t["header_rows"] == 1


def test_an_empty_table_is_kept_as_a_detected_table_with_no_cells():
    t = html_table("<p></p>", caption="Table 1. X")
    assert (t["label"], t["n_rows"], t["n_cols"], t["cells"]) == ("Table 1", 0, 0, [])


def test_marker_caption_only_from_its_tablegroup():
    doc = {"block_type": "Document", "children": [{"block_type": "Page", "id": "/page/2/Page/1", "children": [
        {"block_type": "Caption", "id": "/page/2/Caption/0", "html": "<p>Table 9. Loose caption</p>"},
        {"block_type": "Table", "id": "/page/2/Table/1", "html": "<table><tr><td>1</td></tr></table>"},
        {"block_type": "TableGroup", "id": "/page/2/TableGroup/5", "children": [
            {"block_type": "Caption", "id": "/page/2/Caption/3", "html": "<p>Table 1. Linked</p>"},
            {"block_type": "Table", "id": "/page/2/Table/4", "html": "<table><tr><th>h</th></tr><tr><td>v</td></tr></table>"},
        ]},
    ]}]}
    loose, linked = marker_json_tables(doc)
    assert (loose["label"], loose["caption"], loose["page"]) == (None, None, 3)
    assert (linked["label"], linked["caption"], linked["page"], linked["header_rows"]) == ("Table 1", "Linked", 3, 1)


def test_mineru_caption_only_from_inside_the_table_block():
    pages = [{"page_idx": 0, "blocks": [
        {"type": "text", "content": [{"type": "text", "content": "Table 7. A paragraph, not linked"}]},
        {"type": "table", "content": [
            {"type": "table_body", "content": "<table><tr><td>a</td><td>b</td></tr></table>"}]},
        {"type": "table", "content": [
            {"type": "table_caption", "content": [{"type": "text", "content": "Table 1. Regression coefficients."}]},
            {"type": "table_body", "content": "<table><tbody><tr><td>P</td><td>A</td></tr><tr><td>X1</td><td>0.30</td></tr></tbody></table>"}]},
    ]}]
    unlinked, linked = mineru_middle_tables(pages)
    assert (unlinked["label"], unlinked["caption"]) == (None, None)
    assert (linked["label"], linked["caption"], linked["page"]) == ("Table 1", "Regression coefficients.", 1)
    assert (linked["n_rows"], linked["n_cols"], linked["header_rows"]) == (2, 2, 0)


# ------------------------------------- Sonnet cross-model review, 2026-10-07
# Each case below was reproduced on the code it was raised against, then fixed.

@pytest.mark.parametrize("md,plain", [
    ("r = .45** and r = .30**", "r = .45** and r = .30**"),   # significance stars are content
    ("- 2.31 was the t", "- 2.31 was the t"),                  # a leading minus is a sign
    ("+ .5 shift", "+ .5 shift"),
    ("x<y and y>z", "x<y and y>z"),                            # not a tag
    ("a <b>bold</b> c", "a bold c"),                           # a real tag still goes
    ("- item one", "item one"),                                # a real bullet still goes
])
def test_review_text_cases(md, plain):
    assert md_to_plain(md) == plain


def test_an_unclosed_cell_keeps_its_text():
    t = html_table("<table><tr><td>a<td>b</td></tr><tr><td>c</tr></table>")
    assert sorted((c["r"], c["c"], c["text"]) for c in t["cells"]) == [(0, 0, "a"), (0, 1, "b"), (1, 0, "c")]


def test_html_cells_use_the_same_superscript_convention_as_text():
    t = html_table("<table><tr><td>η<sup>2</sup>p</td></tr></table>")
    assert t["cells"][0]["text"] == md_to_plain("η<sup>2</sup>p") == "η²p"


def test_an_empty_row_still_occupies_its_place():
    t = html_table("<table><tr><td rowspan='3'>A</td><td>1</td></tr><tr></tr><tr><td>3</td></tr></table>")
    pos = {(c["r"], c["c"]): c["text"] for c in t["cells"]}
    assert pos == {(0, 0): "A", (0, 1): "1", (2, 1): "3"}
    assert t["n_rows"] == 3


def test_a_tablegroup_caption_goes_to_its_first_table_only():
    doc = {"block_type": "TableGroup", "id": "/page/0/TableGroup/1", "children": [
        {"block_type": "Caption", "html": "<p>Table 1. Two parts</p>"},
        {"block_type": "Table", "id": "/page/0/Table/2", "html": "<table><tr><td>a</td></tr></table>"},
        {"block_type": "Table", "id": "/page/0/Table/3", "html": "<table><tr><td>b</td></tr></table>"}]}
    first, second = marker_json_tables(doc)
    assert (first["label"], second["label"], second["caption"]) == ("Table 1", None, None)
