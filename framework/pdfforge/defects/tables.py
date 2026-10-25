"""Catalog family T: tables as real papers print them, each carrying simulated statistics.

Every recipe builds or reshapes a table whose numeric cells come from `statgen`
records registered in paper.stats (so the stats gold knows they are in a table), or,
for decoy grids, from numbers that are explicitly NOT test statistics.
"""
from __future__ import annotations

import random

from .. import statgen as sg
from ..model import Block, Cell, Figure, Paper, Run, Table
from ..printing import print_record
from . import recipe


def _new_stat(paper: Paper, rec: sg.StatRecord, plant=None, rng=None, role="result") -> sg.StatRecord:
    n = len(paper.stats) + 1
    rec.stat_id = f"s{n:03d}"
    while rec.stat_id in paper.stats:
        n += 1
        rec.stat_id = f"s{n:03d}"
    print_record(rec, plant, rng or random.Random(n))
    rec.printed["role"] = role
    paper.stats[rec.stat_id] = rec
    return rec


def _h(text, style="r", **kw) -> Cell:
    if text == "ηp2":
        return Cell([Run("η"), Run("2", style="sup"), Run("p", style="sub")], is_header=True, **kw)
    return Cell([Run(text, style=style)], is_header=True, **kw)


def _ci(lo: float, hi: float) -> str:
    from ..printing import _num
    return f"[{_num(lo)}, {_num(hi)}]"


def _c(text, sid=None, style="r") -> Cell:
    return Cell([Run(text, style=style, stat_id=sid)])


def _p(rec) -> str:
    pr = rec.printed
    return ("< " if pr["p_rel"] == "<" else "") + pr["p_txt"]


def _df(x) -> str:
    return "—" if x is None else (str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.2f}")


def _label(paper: Paper) -> str:
    return f"Table {sum(1 for b in paper.blocks if b.kind == 'table') + 1}"


def _place(paper: Paper, blk: Block) -> None:
    """After the last results paragraph of the first study."""
    idx = None
    for i, b in enumerate(paper.blocks):
        if b.kind == "para" and any(r.stat_id for r in b.runs):
            idx = i
        elif idx is not None and b.kind == "heading" and b.level == 1:
            break
    paper.blocks.insert((idx + 1) if idx is not None else len(paper.blocks), blk)


def _bid(paper: Paper, tag: str) -> str:
    k = 1
    while any(b.block_id == f"b{tag}{k}" for b in paper.blocks):
        k += 1
    return f"b{tag}{k}"


@recipe("T01", "T", "Spanning super-header over F / p / ηp² sub-columns", "Tb St", "yes",
        "DP test_camelot_superheader_column_split; test_flatten_centred_group_headers")
def t01(paper: Paper, rng: random.Random) -> bool:
    effects = ["Condition", "Time", "Condition × Time"]
    hdr1 = [_h("Effect"), _h("Accuracy", colspan=3), _h("Confidence", colspan=3)]
    hdr2 = [_h(""), _h("F", "i"), _h("p", "i"), _h("ηp2"), _h("F", "i"), _h("p", "i"), _h("ηp2")]
    body = []
    for k, e in enumerate(effects):
        row = [_c(e)]
        for _ in range(2):
            rec = _new_stat(paper, sg.simulate_interaction_F(rng.randrange(10**6), rng.randint(25, 60),
                                                             rng.uniform(-0.3, 0.6)), rng=rng)
            row += [_c(rec.printed["stat_txt"], rec.stat_id), _c(_p(rec), rec.stat_id),
                    _c(rec.printed["es_txt"], rec.stat_id)]
        body.append(row)
    bid = _bid(paper, "T01")
    t = Table(bid, _label(paper), [Run("Mixed ANOVA results for accuracy and confidence")], [hdr1, hdr2], body,
              note=[Run("Note. All effects have df = 1 and the error df shown in the text.")],
              col_widths=[2.0, 0.8, 0.7, 0.7, 0.8, 0.7, 0.7])
    _place(paper, Block(bid, "table", table=t))
    paper.add_defect("T01", bid)
    return True


@recipe("T02", "T", "Tightly kerned numeric columns (cells almost touching)", "Tb St", "yes",
        "DP test_whitespace_char_fallback; test_fused_cell_gap_detector")
def t02(paper: Paper, rng: random.Random) -> bool:
    tbl = next((b for b in paper.blocks if b.kind == "table" and b.table is not None), None)
    if tbl is None:
        return False
    tbl.table.ruled = True
    tbl.layout["tight"] = True
    paper.add_defect("T02", tbl.block_id)
    return True


@recipe("T03", "T", "Regression table: raised significance stars, section rows, wrapped labels", "Tb", "yes",
        "DP L-062; test_section_row_label_no_merge; test_whitespace_row_cluster_drift")
def t03(paper: Paper, rng: random.Random) -> bool:
    hdr = [_h("Predictor"), _h("b", "i"), _h("SE", "i"), _h("t", "i"), _h("p", "i")]
    body = []
    for group in (f"Control (n = {rng.randint(150, 400)})", f"Treatment (n = {rng.randint(150, 400)})"):
        body.append([Cell([Run(group, style="i")], colspan=5)])
        for lab in ("Age", "Prior experience with the task in previous sessions", "Condition"):
            rec = _new_stat(paper, sg.simulate_regression_b(rng.randrange(10**6), rng.randint(150, 400),
                                                            rng.uniform(-0.3, 0.3)), rng=rng)
            stars = "***" if rec.p < .001 else "**" if rec.p < .01 else "*" if rec.p < .05 else ""
            est = Cell([Run(rec.printed["es_txt"] or f"{rec.es_value:.2f}", stat_id=rec.stat_id)] +
                       ([Run(stars, style="sup", stat_id=rec.stat_id)] if stars else []))
            body.append([_c(lab), est, _c(f"{rec.se:.2f}", rec.stat_id), _c(rec.printed["stat_txt"], rec.stat_id),
                         _c(_p(rec), rec.stat_id)])
    bid = _bid(paper, "T03")
    t = Table(bid, _label(paper), [Run("Regression of the outcome on participant characteristics by group")], [hdr],
              body, note=[Run("Note. *"), Run("p", style="i"), Run(" < .05. **"), Run("p", style="i"),
                          Run(" < .01. ***"), Run("p", style="i"), Run(" < .001.")],
              col_widths=[2.4, 0.7, 0.7, 0.7, 0.7])
    _place(paper, Block(bid, "table", table=t))
    paper.add_defect("T03", bid)
    return True


@recipe("T04", "T", "Wide table with sparse columns and indented sub-rows", "Tb", "yes",
        "DP test_lattice_region_not_clipped; test_region_covers_sparse_columns; test_camelot_bottom_extend")
def t04(paper: Paper, rng: random.Random) -> bool:
    hdr = [_h("Characteristic"), _h("Overall"), _h("Control"), _h("Treatment"), _h("Missing"), _h("Test"), _h("p", "i")]
    rows = []
    for lab, ind in [("Age, y", False), ("Mean (SD)", True), ("Range", True), ("Gender", False),
                     ("Women", True), ("Men", True), ("Education, y", False), ("Mean (SD)", True)]:
        if not ind:
            rows.append([_c(lab)] + [_c("") for _ in range(6)])
        else:
            vals = [f"{rng.uniform(20, 40):.1f} ({rng.uniform(5, 12):.1f})" for _ in range(3)]
            rows.append([_c("    " + lab)] + [_c(v) for v in vals] + [_c(str(rng.randint(0, 9)) if rng.random() < .3 else "—"),
                                                                    _c("—"), _c("—")])
    bid = _bid(paper, "T04")
    t = Table(bid, _label(paper), [Run("Sample characteristics")], [hdr], rows,
              col_widths=[2.2, 1.3, 1.3, 1.3, 0.7, 0.6, 0.5])
    _place(paper, Block(bid, "table", table=t))
    paper.add_defect("T04", bid)
    return True


@recipe("T05", "T", "Two ruled tables stacked on one page with no text between", "Tb St", "yes",
        "DP test_stacked_ruled_tables_keep_their_own_rows; EM NEWS:562, 578")
def t05(paper: Paper, rng: random.Random) -> bool:
    blocks = []
    for k in range(2):
        hdr = [_h("Comparison"), _h("t", "i"), _h("df", "i"), _h("p", "i"), _h("d", "i")]
        body = []
        for lab in ("Sleep vs Control", "Nap vs Control", "Sleep vs Nap"):
            rec = _new_stat(paper, sg.simulate_independent_t(rng.randrange(10**6), rng.randint(30, 60),
                                                             rng.randint(30, 60), rng.uniform(-0.4, 0.6)), rng=rng)
            body.append([_c(lab), _c(rec.printed["stat_txt"], rec.stat_id), _c(_df(rec.df1), rec.stat_id),
                         _c(_p(rec), rec.stat_id), _c(rec.printed["es_txt"], rec.stat_id)])
        bid = _bid(paper, "T05")
        lbl = f"Table {sum(1 for b in paper.blocks if b.kind == 'table') + 1 + k}"
        t = Table(bid, lbl, [Run(f"Pairwise comparisons, {'accuracy' if k == 0 else 'reaction time'}")],
                  [hdr], body, col_widths=[2.0, 0.7, 0.6, 0.7, 0.7])
        blocks.append(Block(bid, "table", table=t))
    for blk in reversed(blocks):
        _place(paper, blk)
    paper.add_defect("T05", blocks[0].block_id, second=blocks[1].block_id)
    return True


@recipe("T06", "T", "Caption shapes: 'Table 1 |' pipe, label alone, untitled", "Tb", "yes",
        "DP test_nature_pipe_captions; test_label_alone_caption_body")
def t06(paper: Paper, rng: random.Random) -> bool:
    tbls = [b for b in paper.blocks if b.kind == "table" and b.table is not None]
    if not tbls:
        return False
    b = rng.choice(tbls)
    b.table.caption_style = rng.choice(["pipe", "label_alone", "untitled"])
    if b.table.caption_style == "label_alone":
        b.table.caption = []
    paper.add_defect("T06", b.block_id, style=b.table.caption_style)
    return True


@recipe("T07", "T", "Uncaptioned numeric grid that holds no test statistics (decoy)", "St", "yes", "EM NEWS:515")
def t07(paper: Paper, rng: random.Random) -> bool:
    hdr = [_h("Item")] + [_h(str(k)) for k in range(1, 8)]
    body = [[_c(f"Q{k}")] + [_c(str(rng.randint(0, 60))) for _ in range(7)] for k in range(1, 7)]
    bid = _bid(paper, "T07")
    t = Table(bid, "", [], [hdr], body, caption_style="untitled", col_widths=[1.0] + [0.6] * 7)
    _place(paper, Block(bid, "table", table=t))
    paper.add_defect("T07", bid, decoy=True)
    return True


@recipe("T08", "T", "Narrow table whose header words split mid-word", "Tb St", "yes", "EM NEWS:1264")
def t08(paper: Paper, rng: random.Random) -> bool:
    hdr = [_h("Model"), _h("Odds Ratio or Coefficient"), _h("t for Treatment Group"), _h("p", "i")]
    body = []
    for k in range(1, 4):
        rec = _new_stat(paper, sg.simulate_regression_b(rng.randrange(10**6), rng.randint(100, 300),
                                                        rng.uniform(-0.3, 0.3)), rng=rng)
        body.append([_c(f"M{k}"), _c(rec.printed["es_txt"], rec.stat_id), _c(rec.printed["stat_txt"], rec.stat_id),
                     _c(_p(rec), rec.stat_id)])
    bid = _bid(paper, "T08")
    t = Table(bid, _label(paper), [Run("Supplementary model estimates")], [hdr], body, col_widths=[0.5, 0.45, 0.45, 0.4])
    t.layout_narrow = True  # type: ignore[attr-defined]
    _place(paper, Block(bid, "table", table=t, layout={"narrow": 0.45}))
    paper.add_defect("T08", bid)
    return True


@recipe("T09", "T", "Parallel models side by side (B SE t 95% CI per model)", "Tb St", "yes",
        "EM handoff of 2026-08-04:15, :40")
def t09(paper: Paper, rng: random.Random) -> bool:
    models = ["Model 1", "Model 2", "Model 3"]
    hdr1 = [_h("Predictor")] + [_h(m, colspan=3) for m in models]
    hdr2 = [_h("")] + [x for _ in models for x in (_h("B"), _h("t", "i"), _h("95% CI"))]
    body = []
    for lab in ("Condition", "Baseline"):
        row = [_c(lab)]
        for _ in models:
            rec = _new_stat(paper, sg.simulate_regression_b(rng.randrange(10**6), rng.randint(120, 400),
                                                            rng.uniform(-0.3, 0.3)), rng=rng)
            row += [_c(rec.printed["es_txt"], rec.stat_id), _c(rec.printed["stat_txt"], rec.stat_id),
                    _c(_ci(rec.ci_lo, rec.ci_hi), rec.stat_id)]
        body.append(row)
    bid = _bid(paper, "T09")
    t = Table(bid, _label(paper), [Run("Estimates from three nested models")], [hdr1, hdr2], body,
              col_widths=[1.2] + [0.6, 0.6, 1.1] * 3)
    _place(paper, Block(bid, "table", table=t))
    paper.add_defect("T09", bid)
    return True


@recipe("T10", "T", "ANOVA table with df in its own column, and a compact table without df", "St", "yes",
        "EM NEWS:627, 3191")
def t10(paper: Paper, rng: random.Random) -> bool:
    hdr = [_h("Source"), _h("df", "i"), _h("F", "i"), _h("p", "i"), _h("ηp2")]
    body = []
    for lab in ("Condition", "Load", "Condition × Load"):
        rec = _new_stat(paper, sg.simulate_interaction_F(rng.randrange(10**6), rng.randint(20, 60), rng.uniform(-0.3, 0.6)), rng=rng)
        body.append([_c(lab), _c(f"{_df(rec.df1)}, {_df(rec.df2)}", rec.stat_id), _c(rec.printed["stat_txt"], rec.stat_id),
                     _c(_p(rec), rec.stat_id), _c(rec.printed["es_txt"], rec.stat_id)])
    bid = _bid(paper, "T10")
    _place(paper, Block(bid, "table", table=Table(bid, _label(paper), [Run("Analysis of variance")], [hdr], body,
                                                 col_widths=[2.0, 0.8, 0.7, 0.7, 0.7])))
    # compact: M (SD) per group, t, p -- df only stated in the header
    n1, n2 = rng.randint(30, 70), rng.randint(30, 70)
    hdr2 = [_h("Measure"), _h("Control M (SD)"), _h("Treatment M (SD)"), _h(f"t({n1 + n2 - 2})"), _h("p", "i")]
    body2 = []
    for lab in ("Liking", "Intention"):
        rec = _new_stat(paper, sg.simulate_independent_t(rng.randrange(10**6), n1, n2, rng.uniform(-0.4, 0.6)), rng=rng)
        body2.append([_c(lab), _c(f"{rng.uniform(3, 5):.2f} ({rng.uniform(0.8, 1.6):.2f})"),
                      _c(f"{rng.uniform(3, 5):.2f} ({rng.uniform(0.8, 1.6):.2f})"),
                      _c(rec.printed["stat_txt"], rec.stat_id), _c(_p(rec), rec.stat_id)])
    bid2 = _bid(paper, "T10")
    _place(paper, Block(bid2, "table", table=Table(bid2, _label(paper), [Run("Group comparisons")], [hdr2], body2,
                                                   col_widths=[1.2, 1.2, 1.2, 0.7, 0.6])))
    paper.add_defect("T10", bid, compact=bid2)
    return True


@recipe("T11", "T", "Replication table: original study's results beside this study's", "St", "yes",
        "EM NEWS:2694, 2942")
def t11(paper: Paper, rng: random.Random) -> bool:
    hdr = [_h("Finding"), _h("Original effect"), _h("Replication effect")]
    body = []
    for lab in ("Anchoring", "Framing", "Sunk cost"):
        orig = _new_stat(paper, sg.simulate_independent_t(rng.randrange(10**6), rng.randint(20, 40), rng.randint(20, 40),
                                                          rng.uniform(0.3, 0.8)), rng=rng, role="original_study")
        rep = _new_stat(paper, sg.simulate_independent_t(rng.randrange(10**6), rng.randint(80, 200), rng.randint(80, 200),
                                                         rng.uniform(-0.1, 0.4)), rng=rng)
        body.append([_c(lab), Cell([Run(f"t({_df(orig.df1)}) = {orig.printed['stat_txt']}, d = {orig.printed['es_txt']}", stat_id=orig.stat_id)]),
                     Cell([Run(f"t({_df(rep.df1)}) = {rep.printed['stat_txt']}, d = {rep.printed['es_txt']}", stat_id=rep.stat_id)])])
    bid = _bid(paper, "T11")
    _place(paper, Block(bid, "table", table=Table(bid, _label(paper), [Run("Original and replication results")],
                                                 [hdr], body, col_widths=[1.0, 1.8, 1.8])))
    paper.add_defect("T11", bid)
    return True


@recipe("T13", "T", "Design (paired) stated only in the table note", "St", "yes", "EM NEWS:2430, 2813")
def t13(paper: Paper, rng: random.Random) -> bool:
    hdr = [_h("Measure"), _h("t", "i"), _h("df", "i"), _h("p", "i"), _h("dz")]
    body = []
    for lab in ("Arousal", "Valence"):
        rec = _new_stat(paper, sg.simulate_paired_t(rng.randrange(10**6), rng.randint(25, 90), rng.uniform(-0.4, 0.6)), rng=rng)
        body.append([_c(lab), _c(rec.printed["stat_txt"], rec.stat_id), _c(_df(rec.df1), rec.stat_id),
                     _c(_p(rec), rec.stat_id), _c(rec.printed["es_txt"], rec.stat_id)])
    bid = _bid(paper, "T13")
    _place(paper, Block(bid, "table", table=Table(bid, _label(paper), [Run("Change from session one to session two")],
                                                 [hdr], body, note=[Run("Note. Paired-samples t tests.")],
                                                 col_widths=[1.4, 0.7, 0.6, 0.7, 0.7])))
    paper.add_defect("T13", bid)
    return True


@recipe("T15", "T", "Table header states a 90% CI", "St", "yes", "EM NEWS:6-12")
def t15(paper: Paper, rng: random.Random) -> bool:
    hdr = [_h("Outcome"), _h("t", "i"), _h("df", "i"), _h("p", "i"), _h("d", "i"), _h("90% CI")]
    body = []
    for lab in ("Outcome A", "Outcome B"):
        rec = _new_stat(paper, sg.simulate_independent_t(rng.randrange(10**6), rng.randint(20, 60), rng.randint(20, 60),
                                                         rng.uniform(-0.3, 0.8), level=90), rng=rng)
        body.append([_c(lab), _c(rec.printed["stat_txt"], rec.stat_id), _c(_df(rec.df1), rec.stat_id), _c(_p(rec), rec.stat_id),
                     _c(rec.printed["es_txt"], rec.stat_id), _c(_ci(rec.ci_lo, rec.ci_hi), rec.stat_id)])
    bid = _bid(paper, "T15")
    _place(paper, Block(bid, "table", table=Table(bid, _label(paper), [Run("Equivalence-oriented comparisons")], [hdr], body,
                                                 col_widths=[1.2, 0.6, 0.5, 0.6, 0.6, 1.1])))
    paper.add_defect("T15", bid)
    return True


@recipe("T16", "T", "Correlation matrix, lower triangle with significance stars", "Tb St", "yes",
        "EM NEWS:2568; handoff of 2026-08-04:82")
def t16(paper: Paper, rng: random.Random) -> bool:
    names = ["1. Trust", "2. Reliance", "3. Expertise", "4. Effort"]
    n = rng.randint(120, 400)
    hdr = [_h("Variable")] + [_h(str(k)) for k in range(1, 5)]
    body = []
    for i, nm in enumerate(names):
        row = [_c(nm)]
        for j in range(4):
            if j < i:
                rec = _new_stat(paper, sg.simulate_r(rng.randrange(10**6), n, rng.uniform(-0.4, 0.5)), rng=rng)
                stars = "**" if rec.p < .01 else "*" if rec.p < .05 else ""
                row.append(Cell([Run(rec.printed["stat_txt"], stat_id=rec.stat_id)] +
                                ([Run(stars, style="sup", stat_id=rec.stat_id)] if stars else [])))
            else:
                row.append(_c("—" if i == j else ""))
        body.append(row)
    bid = _bid(paper, "T16")
    _place(paper, Block(bid, "table", table=Table(bid, _label(paper), [Run("Correlations among study variables")],
                                                 [hdr], body, note=[Run(f"Note. N = {n}. *"), Run("p", style="i"),
                                                                    Run(" < .05. **"), Run("p", style="i"), Run(" < .01.")],
                                                 col_widths=[1.4, 0.6, 0.6, 0.6, 0.6])))
    paper.add_defect("T16", bid, n=n)
    return True


@recipe("T17", "T", "df column of lone '1' cells (page-number filters delete them)", "Tb St", "yes",
        "EK 03 gallery #20; ME inbox 2026-10-08")
def t17(paper: Paper, rng: random.Random) -> bool:
    hdr = [_h("Effect"), _h("df1"), _h("df2"), _h("F", "i"), _h("p", "i")]
    body = []
    for lab in ("Condition", "Order", "Condition × Order"):
        rec = _new_stat(paper, sg.simulate_interaction_F(rng.randrange(10**6), rng.randint(25, 60), rng.uniform(-0.3, 0.6)), rng=rng)
        body.append([_c(lab), _c("1", rec.stat_id), _c(_df(rec.df2), rec.stat_id), _c(rec.printed["stat_txt"], rec.stat_id),
                     _c(_p(rec), rec.stat_id)])
    bid = _bid(paper, "T17")
    _place(paper, Block(bid, "table", table=Table(bid, _label(paper), [Run("Tests of the experimental effects")], [hdr], body,
                                                 col_widths=[1.8, 0.5, 0.6, 0.7, 0.7])))
    paper.add_defect("T17", bid)
    return True


@recipe("T18", "T", "European comma decimals in a table, comma-thousands integers in the text", "St Tb", "yes",
        "DP NORM:304; EM NEWS:1137")
def t18(paper: Paper, rng: random.Random) -> bool:
    hdr = [_h("Group"), _h("n", "i"), _h("t", "i"), _h("p", "i"), _h("d", "i")]
    body = []
    for lab in ("Gruppe A", "Gruppe B"):
        rec = _new_stat(paper, sg.simulate_independent_t(rng.randrange(10**6), rng.randint(30, 80), rng.randint(30, 80),
                                                         rng.uniform(-0.4, 0.6)), rng=rng)
        rec.printed["locale"] = "comma_decimal"
        eu = lambda s: s.replace(".", ",")  # noqa: E731
        body.append([_c(lab), _c(str(rec.n)), _c(eu(rec.printed["stat_txt"]), rec.stat_id), _c(eu(_p(rec)), rec.stat_id),
                     _c(eu(rec.printed["es_txt"]), rec.stat_id)])
    bid = _bid(paper, "T18")
    _place(paper, Block(bid, "table", table=Table(bid, _label(paper), [Run("Vergleiche zwischen den Gruppen")], [hdr], body,
                                                 col_widths=[1.4, 0.6, 0.7, 0.7, 0.7])))
    paper.add_defect("T18", bid)
    return True


@recipe("T19", "T", "Chart whose axis tick labels sit next to its caption", "Tx", "yes",
        "DP test_chart_data_trim_real_pdf; test_figure_caption_trim_real_pdf")
def t19(paper: Paper, rng: random.Random) -> bool:
    n = sum(1 for b in paper.blocks if b.kind == "figure") + 1
    fig = Figure(f"fig{n}", f"Figure {n}", [Run("Mean creativity ratings by condition")], ticks=["1", "2", "3", "4", "5", "6", "7"])
    bid = _bid(paper, "T19")
    _place(paper, Block(bid, "figure", figure=fig))
    paper.add_defect("T19", bid)
    return True


@recipe("T20", "T", "Author-year citations inside table cells and a numbered source note", "Ci", "yes",
        "CG handoff 06-26:10; CL:588")
def t20(paper: Paper, rng: random.Random) -> bool:
    book_refs = paper.references[:4]
    if len(book_refs) < 3:
        return False
    hdr = [_h("Study"), _h("Country"), _h("Design"), _h("N", "i")]
    body = []
    markers = getattr(paper, "citation_markers", [])
    for k, r in enumerate(book_refs[:3]):
        a = r.authors
        name = a[0][0] if len(a) == 1 else (f"{a[0][0]} and {a[1][0]}" if len(a) == 2 else f"{a[0][0]} et al.")
        txt = f"{name} ({r.year})"
        markers.append({"marker_text": txt, "reference_ids": [r.ref_id]})
        body.append([Cell([Run(txt, cite_refs=[r.ref_id])]), _c(rng.choice(["Brazil", "Norway", "Japan", "Kenya"])),
                     _c(rng.choice(["Survey experiment", "Field study", "Panel"])), _c(str(rng.randint(150, 4000)))])
    bid = _bid(paper, "T20")
    t = Table(bid, _label(paper), [Run("Prior studies of the effect")], [hdr], body, col_widths=[1.8, 0.9, 1.3, 0.6])
    _place(paper, Block(bid, "table", table=t))
    paper.add_defect("T20", bid)
    return True


@recipe("F26", "F", "Two captions side by side on one line (two half-width tables)", "Tb", "yes",
        "DP test_side_by_side_labels_with_space_glyphs")
def f26(paper: Paper, rng: random.Random) -> bool:
    halves = []
    for k in range(2):
        hdr = [_h("Group"), _h("M", "i"), _h("SD", "i")]
        body = [[_c(g), _c(f"{rng.uniform(2, 6):.2f}"), _c(f"{rng.uniform(0.6, 1.8):.2f}")] for g in ("Control", "Treatment")]
        halves.append((hdr, body))
    bid = _bid(paper, "F26")
    n = sum(1 for b in paper.blocks if b.kind == "table") + 1
    t1 = Table(bid + "a", f"Table {n}", [Run("Participant descriptives")], [halves[0][0]], halves[0][1])
    t2 = Table(bid + "b", f"Table {n + 1}", [Run("Adherence descriptives")], [halves[1][0]], halves[1][1])
    _place(paper, Block(bid, "table", table=t1, layout={"side_by_side": t2}))
    paper.add_defect("F26", bid)
    return True


@recipe("T12", "T", "Two near-identical rows that are distinct results (duplicate trap)", "St", "yes",
        "EM NEWS:2652, 3333")
def t12(paper: Paper, rng: random.Random) -> bool:
    n = rng.randint(300, 500)
    first = sg.simulate_oneway_F(rng.randrange(10**6), 3, n, 0.05)
    for _ in range(400):
        second = sg.simulate_oneway_F(rng.randrange(10**6), 3, n, 0.05)
        if abs(second.statistic - first.statistic) < 0.04 and f"{second.p:.3f}" != f"{first.p:.3f}":
            break
    else:
        return False
    hdr = [_h("Outcome"), _h("F", "i"), _h("df", "i"), _h("p", "i")]
    body = []
    for lab, rec in (("Immediate recall", first), ("Delayed recall", second)):
        rec = _new_stat(paper, rec, rng=rng)
        body.append([_c(lab), _c(rec.printed["stat_txt"], rec.stat_id), _c(f"{_df(rec.df1)}, {_df(rec.df2)}", rec.stat_id),
                     _c(_p(rec), rec.stat_id)])
    bid = _bid(paper, "T12")
    _place(paper, Block(bid, "table", table=Table(bid, _label(paper), [Run("Recall by condition")], [hdr], body,
                                                 col_widths=[1.6, 0.7, 0.8, 0.7])))
    paper.add_defect("T12", bid)
    return True


@recipe("T14", "T", "Trailing-zero p in a table cell (.040): printed precision must survive", "St", "yes",
        "EM TODO:1747; ME inbox 2026-10-12")
def t14(paper: Paper, rng: random.Random) -> bool:
    for _ in range(500):
        rec = sg.simulate_independent_t(rng.randrange(10**6), rng.randint(30, 60), rng.randint(30, 60), rng.uniform(0.2, 0.6))
        if 0.001 <= rec.p < 0.999 and f"{rec.p:.3f}".endswith("0"):
            break
    else:
        return False
    rec = _new_stat(paper, rec, rng=rng)
    hdr = [_h("Measure"), _h("t", "i"), _h("df", "i"), _h("p", "i")]
    bid = _bid(paper, "T14")
    body = [[_c("Persistence"), _c(rec.printed["stat_txt"], rec.stat_id), _c(_df(rec.df1), rec.stat_id), _c(_p(rec), rec.stat_id)]]
    _place(paper, Block(bid, "table", table=Table(bid, _label(paper), [Run("Persistence by condition")], [hdr], body,
                                                 col_widths=[1.4, 0.7, 0.6, 0.7])))
    paper.add_defect("T14", rec.stat_id, p_txt=rec.printed["p_txt"])
    return True
