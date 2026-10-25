"""Catalog family H: section structure as printed (numbers, typography, impostor headings)."""
from __future__ import annotations

import random

from ..model import Block, Paper, Run
from . import recipe


def _headings(paper: Paper, level=None):
    return [b for b in paper.blocks if b.kind == "heading" and (level is None or b.level == level)]


@recipe("H01", "H", "Section number split from its title (5.4. | Discussion)", "Se", "yes",
        "DP test_orphan_multilevel_number; test_orphan_section_number; test_roman_numeral_section_promote")
def h01(paper: Paper, rng: random.Random) -> bool:
    hs = _headings(paper)
    if not hs:
        return False
    roman = rng.random() < 0.3
    ctr = [0, 0, 0]
    for b in hs:
        lv = min(b.level, 3) - 1
        ctr[lv] += 1
        for k in range(lv + 1, 3):
            ctr[k] = 0
        num = (["I", "II", "III", "IV", "V", "VI", "VII", "VIII"][ctr[0] - 1] + "." if roman and lv == 0
               else ".".join(str(x) for x in ctr[:lv + 1]) + ".")
        b.runs = [Run(num, style="b"), Run("\n")] + b.runs
    paper.add_defect("H01", None, roman=roman)
    return True


@recipe("H02", "H", "Heading typography: ALL CAPS, run-in headings, long wrapped headings", "Se", "yes",
        "DP test_all_caps_section_promote; test_sections_v161_text_annotator; test_wrapped_subsection_heading_rejoin")
def h02(paper: Paper, rng: random.Random) -> bool:
    for b in _headings(paper, 1):
        b.runs = [Run(b.text.upper(), style="b")]
    for b in _headings(paper, 2)[:2]:
        b.layout["run_in"] = True
    long = next((b for b in _headings(paper) if b.text.strip() == "Participants"), None)
    if long is not None:
        long.runs = [Run("Participants, Design, and the Procedure Used to Collect the Data in the Online Study", style="b")]
    paper.add_defect("H02", None)
    return True


@recipe("H03", "H", "Short title-case lines that are not headings (affiliation line, Key Points box)", "Se", "yes",
        "DP test_affiliation_heading_promote_guard; test_jama_open_cluster; test_abstract_affiliation_markers")
def h03(paper: Paper, rng: random.Random) -> bool:
    intro = next((i for i, b in enumerate(paper.blocks) if b.kind == "heading"), 0)
    box = Block("bH03a", "box", [Run("Key Points", style="b"), Run("\n"), Run("Question  Does the manipulation change choices?"),
                                 Run("\n"), Run("Findings  The effect was small"), Run("\n"),
                                 Run("Meaning  Context shapes judgment")])
    paper.blocks.insert(intro, box)
    para = next((i for i, b in enumerate(paper.blocks) if b.kind == "para" and i > intro + 2), None)
    if para is not None:
        paper.blocks.insert(para, Block("bH03b", "box", [Run(paper.affiliations[0])], in_body=False))
    paper.add_defect("H03", "bH03a")
    return True


@recipe("H04", "H", "Keywords sidebar right before the Introduction", "Se", "yes", "DP test_keywords_label_split_real_pdf")
def h04(paper: Paper, rng: random.Random) -> bool:
    intro = next((i for i, b in enumerate(paper.blocks) if b.kind == "heading"), 0)
    runs = [Run("KEYWORDS", style="b")]
    for k in paper.keywords:
        runs += [Run("\n"), Run(k)]
    paper.blocks.insert(intro, Block("bH04", "box", runs, in_body=False))
    paper.add_defect("H04", "bH04")
    return True


@recipe("H05", "H", "Multi-study paper (Study 1 / Study 2 / General Discussion)", "Se St", "yes",
        "SA TODO paperforge depth", probe="property")
def h05(paper: Paper, rng: random.Random) -> bool:
    if any(b.kind == "heading" and b.text.startswith("Study 2") for b in paper.blocks):
        paper.add_defect("H05", None)
        return True
    return False


@recipe("H06", "H", "Variant headings: 'Literature Review', stacked back matter, reference-list heading variants", "Se Rf", "yes",
        "CL:54; CG lesson 2026-03-14 (last References heading)")
def h06(paper: Paper, rng: random.Random) -> bool:
    th = next((b for b in paper.blocks if b.kind == "heading" and b.text == "Theoretical Background"), None)
    if th is not None:
        th.runs = [Run("Literature Review", style="b")]
        th.level, th.section_label = 1, "other"
        th.layout["canonical"] = "Literature Review"
    paper.front_matter["back_matter"] = [
        ("Notes", "We thank the participants and the anonymous reviewers."),
        ("Funding", "This research received no specific grant from any funding agency."),
        ("Author contributions", "All authors designed the study and wrote the manuscript."),
    ]
    paper.layout["refs_heading"] = rng.choice(["Reference List", "Literature", "References", None])
    paper.add_defect("H06", None, refs_heading=paper.layout["refs_heading"])
    return True
