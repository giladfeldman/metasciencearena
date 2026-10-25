"""Catalog family P: page furniture painted outside the body (headers, footers, stamps,
watermarks, page-1 footnote block, masthead, line numbers). Every string the painter
draws is recorded in placement["furniture"] and becomes text gold must_exclude.
"""
from __future__ import annotations

import random

from ..model import Block, Cell, Paper, Run, Table
from . import recipe


@recipe("P01", "P", "Running header on every page, incl. 'Surname et al.'", "Tx Se Rf", "yes",
        "DP test_titlecase_etal_running_header; CL:1971")
def p01(paper: Paper, rng: random.Random) -> bool:
    paper.layout["header_left"] = paper.journal["name"].upper() if rng.random() < 0.5 else paper.journal["name"]
    paper.layout["header_right"] = f"{paper.authors[0][0]} et al."
    paper.add_defect("P01", None)
    return True


@recipe("P02", "P", "Publisher boilerplate footer that lands mid-sentence across page breaks", "Tx", "yes",
        "DP test_cambridge_footer_strip_real_pdf")
def p02(paper: Paper, rng: random.Random) -> bool:
    paper.layout["footer_text"] = f"https://doi.org/{paper.journal['doi']} Published online by Synthetic University Press"
    paper.add_defect("P02", None)
    return True


@recipe("P03", "P", "'Downloaded from ... by <institution> on <date>' stamp, horizontal and sideways", "Tx Rf", "yes",
        "EK 03 gallery #5, #13, #18, #21; CL:2079")
def p03(paper: Paper, rng: random.Random) -> bool:
    s = f"Downloaded from https://synth-journals.example/{paper.journal['doi'][-6:]} by Norwick University Library on 14 March 2025"
    if rng.random() < 0.5:
        paper.layout["stamp"] = s
    else:
        paper.layout["side_stamp"] = s
    paper.add_defect("P03", None, where="side" if "side_stamp" in paper.layout else "bottom")
    return True


@recipe("P04", "P", "Page-1 footnote block (affiliation, e-mail, ISSN, CONTACT)", "Tx Se", "yes",
        "DP test_elsevier_footer_strip; test_wrapped_contact_strip; test_body_affiliation_strip")
def p04(paper: Paper, rng: random.Random) -> bool:
    s, g = paper.authors[0]
    paper.layout["first_page_footnote"] = [
        f"CONTACT {g} {s}, {paper.affiliations[0]}",
        f"E-mail address: {s.lower().replace(' ', '')}@synth-univ.example",
        f"ISSN 0000-0000 (print) © {paper.journal['year']} Synthetic University Press",
    ]
    paper.layout["foot_reserve"] = max(paper.layout.get("foot_reserve", 48), 64)
    paper.add_defect("P04", None)
    return True


@recipe("P05", "P", "Masthead (journal, vol/issue, pages) between title and abstract", "Se", "yes",
        "DP test_sage_masthead_strip_real_pdf")
def p05(paper: Paper, rng: random.Random) -> bool:
    j = paper.journal
    paper.front_matter.setdefault("masthead", []).extend([
        f"{j['name']} {j['year']}, Vol. {j['volume']}({j['issue']}) {j['pages']}",
        f"https://doi.org/{j['doi']}",
        "Received 3 February; accepted 19 August"])
    paper.add_defect("P05", "masthead")
    return True


@recipe("P06", "P", "Page numbers plus a narrow integer column a page-number filter would delete", "Tb St", "yes",
        "DP test_page_number_strip_never_deletes_data; EM FINDINGS 2026-08-25")
def p06(paper: Paper, rng: random.Random) -> bool:
    paper.layout["page_numbers"] = "{page}"
    body = []
    for k in range(rng.randint(6, 10)):
        body.append([Cell([Run(f"Item {k + 1}")]), Cell([Run(str(rng.choice([0, 4, 7, 12, 40, 123, 999])))]),
                     Cell([Run(f"{rng.uniform(1, 7):.2f}")])])
    t = Table("tcounts", _next_label(paper), [Run("Item counts and mean ratings")],
              [[Cell([Run("Item")], is_header=True), Cell([Run("n", style="i")], is_header=True),
                Cell([Run("M", style="i")], is_header=True)]], body, col_widths=[2, 0.8, 0.8])
    _insert_after_first_results_para(paper, Block(block_id="bP06", kind="table", table=t))
    paper.add_defect("P06", "bP06")
    return True


@recipe("P07", "P", "Long table across pages repeating its header row", "Tb", "yes",
        "DP test_repeated_line_strip_is_page_scoped")
def p07(paper: Paper, rng: random.Random) -> bool:
    body = [[Cell([Run(f"Participant group {k + 1}")]), Cell([Run(f"{rng.uniform(1, 7):.2f}")]),
             Cell([Run(f"{rng.uniform(0.5, 2):.2f}")]), Cell([Run(str(rng.randint(20, 90)))])]
            for k in range(70)]
    t = Table("tlong", _next_label(paper), [Run("Descriptive statistics by participant group")],
              [[Cell([Run("Performance metric")], is_header=True), Cell([Run("M", style="i")], is_header=True),
                Cell([Run("SD", style="i")], is_header=True), Cell([Run("n", style="i")], is_header=True)]],
              body, col_widths=[2.2, 0.8, 0.8, 0.6])
    _insert_after_first_results_para(paper, Block(block_id="bP07", kind="table", table=t))
    paper.add_defect("P07", "bP07")
    return True


@recipe("P08", "P", "Diagonal watermark on every page", "Tx", "yes", "SA text-fidelity v2; DP test_watermark_arm")
def p08(paper: Paper, rng: random.Random) -> bool:
    paper.layout["watermark"] = rng.choice(["AUTHOR MANUSCRIPT", "PREPRINT", "DOWNLOADED"])
    paper.add_defect("P08", None)
    return True


@recipe("P09", "P", "PMC author-manuscript banner and 'Page N' label", "Rf Tx", "yes",
        "CL:1134; CG lesson 2026-03-14")
def p09(paper: Paper, rng: random.Random) -> bool:
    paper.layout["header_left"] = f"{paper.journal['name']}. Author manuscript; available in PMC {int(paper.journal['year']) + 1} January 01."
    paper.layout["header_on_first"] = True
    paper.layout["page_numbers"] = "Page {page}"
    paper.add_defect("P09", None)
    return True


@recipe("P10", "P", "Line-numbered review manuscript (numbers in the margin)", "Tx St", "yes",
        "ME v030c audit; EK 03 (not covered there)")
def p10(paper: Paper, rng: random.Random) -> bool:
    paper.layout["line_numbers"] = True
    paper.layout["margin"] = max(paper.layout.get("margin", 54), 64)
    paper.add_defect("P10", None)
    return True


def _next_label(paper: Paper) -> str:
    n = sum(1 for b in paper.blocks if b.kind == "table") + 1
    return f"Table {n}"


def _insert_after_first_results_para(paper: Paper, blk: Block) -> None:
    for i, b in enumerate(paper.blocks):
        if b.kind == "para" and any(r.stat_id for r in b.runs):
            paper.blocks.insert(i + 1, blk)
            return
    paper.blocks.append(blk)
