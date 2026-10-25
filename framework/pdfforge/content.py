"""Build a clean synthetic paper (no defects yet) from a seed and a prose-bank entry.

Every number comes from `statgen` (simulated data); every prose sentence from the
prose bank (digit-free by construction); every citation from `refstyle`. Defects are
applied afterwards by `defects/` and change this paper in recorded ways.
"""
from __future__ import annotations

import random
import re

import numpy as np

from . import statgen as sg
from .model import Block, Cell, Footnote, Paper, Run, Table
from .printing import print_record
from .refstyle import STYLE_PAIRS, CitationBook

_SECTION_LABEL = {"Introduction": "introduction", "Method": "methods", "Methods": "methods",
                  "Results": "results", "Discussion": "discussion", "General Discussion": "discussion",
                  "Conclusion": "conclusion", "References": "references", "Abstract": "abstract"}
_UNIS = ["University of Norwick", "Halvern Institute of Technology", "Lindqvist University",
         "University of Castamere", "Ostrava College of Social Sciences", "Kelmar State University"]
_DEPTS = ["Department of Psychology", "School of Behavioural Science", "Department of Management",
          "Institute for Decision Research"]
_JOURNALS = ["Journal of Synthetic Behavioural Science", "Synthetic Psychology Reports",
             "Collected Studies in Judgment", "Review of Simulated Social Research"]

ITALIC_SYMBOLS = {"t", "F", "p", "d", "r", "M", "SD", "N", "n", "b", "SE", "z", "U", "W", "H", "g", "OR", "V"}

# (sentence template, simulator, family weight). {iv}/{dv} come from the topic words.
_FAMILIES = [
    ("Scores on {dv} were compared between the treatment and control conditions, {S}.",
     lambda g, s: sg.simulate_independent_t(s, int(g.integers(35, 120)), int(g.integers(35, 120)), float(g.uniform(-0.5, 0.7)))),
    ("Because variances differed, a Welch test compared the two conditions on {dv}, {S}.",
     # equal groups: Cohen's d (pooled SD) then agrees with the printed Welch t (consult finding 1)
     lambda g, s: (lambda n: sg.simulate_independent_t(s, n, n, float(g.uniform(-0.5, 0.6)), welch=True))(int(g.integers(30, 120)))),
    ("Ratings of {dv} were compared between the first and second session, {S}.",
     lambda g, s: sg.simulate_paired_t(s, int(g.integers(25, 120)), float(g.uniform(-0.5, 0.6)))),
    ("A one-way analysis compared {dv} across the three conditions, {S}.",
     lambda g, s: sg.simulate_oneway_F(s, 3, int(g.integers(25, 70)), float(g.uniform(0.0, 0.4)))),
    ("The interaction between condition and {iv} on {dv} was {SIGWORD}, {S}.",
     lambda g, s: sg.simulate_interaction_F(s, int(g.integers(20, 60)), float(g.uniform(0.0, 0.5)))),
    ("Choice of the target option was cross-tabulated with condition, {S}.",
     lambda g, s: sg.simulate_chi2(s, int(g.integers(120, 500)), 2, 2, float(g.uniform(0.0, 0.5)))),
    ("Responses across the three options were cross-tabulated with condition, {S}.",
     lambda g, s: sg.simulate_chi2(s, int(g.integers(150, 500)), 2, 3, float(g.uniform(0.0, 0.4)))),
    ("The correlation between {iv} and {dv} was estimated, {S}.",
     lambda g, s: sg.simulate_r(s, int(g.integers(60, 400)), float(g.uniform(-0.35, 0.35)))),
    ("In a regression controlling for age, the coefficient of {iv} on {dv} was estimated, {S}.",
     lambda g, s: sg.simulate_regression_b(s, int(g.integers(100, 500)), float(g.uniform(-0.3, 0.3)))),
    ("The odds of choosing the default were compared between conditions, {S}.",
     lambda g, s: sg.simulate_odds_ratio(s, int(g.integers(200, 800)), float(g.uniform(-0.6, 0.8)))),
    ("The proportions of participants who completed the task were compared, {S}.",
     lambda g, s: sg.simulate_z_proportions(s, int(g.integers(100, 300)), int(g.integers(100, 300)), 0.45, float(g.uniform(0.3, 0.45)))),
    ("A Kruskal-Wallis test compared the ordinal {dv} ratings across conditions, {S}.",
     lambda g, s: sg.simulate_kruskal(s, 3, int(g.integers(20, 60)), float(g.uniform(0.0, 0.7)))),
    ("A Mann-Whitney test compared {dv} between the two conditions, {S}.",
     lambda g, s: sg.simulate_mannwhitney(s, int(g.integers(20, 60)), int(g.integers(20, 60)), float(g.uniform(0.0, 0.7)))),
]


def stat_runs(rec: sg.StatRecord) -> list[Run]:
    """The printed result as styled runs: italic test symbols, superscript/subscript in
    effect-size names, plain digits. Concatenated text == what the reader sees."""
    return text_runs(rec.printed.get("text_visible") or rec.printed["text"], rec.stat_id)


def text_runs(text: str, sid: str | None) -> list[Run]:
    out: list[Run] = []
    # tokenise into symbols, greek labels and the rest
    for m in re.finditer(r"ηp2|χ2|ω2|φ|[A-Za-z]+|[^A-Za-zηχωφ]+", text):
        tok = m.group(0)
        if tok == "ηp2":
            out += [Run("η", stat_id=sid), Run("2", style="sup", stat_id=sid), Run("p", style="sub", stat_id=sid)]
        elif tok in ("χ2", "ω2"):
            out += [Run(tok[0], stat_id=sid), Run("2", style="sup", stat_id=sid)]
        elif tok in ITALIC_SYMBOLS or tok == "dz":
            if tok == "dz":
                out += [Run("d", style="i", stat_id=sid), Run("z", style="sub", stat_id=sid)]
            else:
                out.append(Run(tok, style="i", stat_id=sid))
        else:
            out.append(Run(tok, stat_id=sid))
    return out


def _fill_prose(text: str, book: CitationBook, rng: random.Random) -> list[list[Run]]:
    """Paragraphs of runs with [[CITE]] markers turned into citations."""
    paras = []
    for p in [x.strip() for x in text.split("\n\n") if x.strip()]:
        runs: list[Run] = []
        parts = re.split(r"((?:\[\[CITE\]\][\s,;]*)+)", p)
        for part in parts:
            if "[[CITE]]" in part:
                k = min(3, part.count("[[CITE]]") + (1 if rng.random() < 0.25 else 0))
                if runs and runs[-1].text and not runs[-1].text.endswith(" ") and book.cite_style != "superscript":
                    runs[-1].text += " "
                runs += book.cite(k)
                trail = part.split("]]")[-1]
                if trail.strip(" ,;"):
                    runs.append(Run(trail))
            elif part:
                if book.cite_style == "superscript" and runs and runs[-1].style == "sup":
                    part = part.lstrip()
                    if part and part[0].isalnum():
                        part = " " + part
                runs.append(Run(part))
        # collapse "( (" artefacts of a marker inside the model's own parentheses
        for r in runs:
            r.text = r.text.replace("((", "(").replace("))", ")")
        paras.append(runs)
    return paras


def build_paper(paper_id: str, seed: int, prose: dict, split: str, *, n_studies: int | None = None,
                n_stats: int | None = None, style_pair: tuple[str, str] | None = None,
                plant_mix: dict | None = None) -> Paper:
    rng = random.Random(seed)
    g = np.random.default_rng(seed)
    blocks_txt = prose["blocks"]
    topic = prose["topic"]
    words = [w for w in re.findall(r"[a-z]{5,}", topic) if w not in ("their", "after", "under", "toward")] or ["choice"]
    iv, dv = (words + ["judgment"])[0], (words + ["judgment", "behaviour"])[1 if len(words) > 1 else -1]
    cite_style, ref_style = style_pair or rng.choice(STYLE_PAIRS)
    book = CitationBook(rng, words + ["choice", "judgment", "behaviour"], cite_style, ref_style)

    authors = []
    for _ in range(rng.randint(2, 5)):
        from .refstyle import _surname  # noqa: PLC0415 - local helper
        authors.append((_surname(rng, rng.choice(["plain"] * 5 + ["particle", "diacritic"])),
                        rng.choice("ABCDEFGHJKLMNPRSTVW") + "."))
    affils = [f"{rng.choice(_DEPTS)}, {u}" for u in rng.sample(_UNIS, rng.randint(1, 3))]
    year = rng.randint(2019, 2025)
    p1 = rng.randint(100, 1800)
    journal = {"name": rng.choice(_JOURNALS), "year": str(year), "volume": str(rng.randint(5, 40)),
               "issue": str(rng.randint(1, 12)), "pages": f"{p1}–{p1 + rng.randint(9, 24)}",
               "doi": f"10.5555/pdfu.{rng.getrandbits(36):09x}"}

    blocks: list[Block] = []
    counter = {"b": 0}

    def add(kind, runs=None, **kw) -> Block:
        counter["b"] += 1
        b = Block(block_id=f"b{counter['b']:03d}", kind=kind, runs=runs or [], **kw)
        blocks.append(b)
        return b

    def heading(text, level=1, number=None):
        lab = _SECTION_LABEL.get(text) if level == 1 else None
        return add("heading", [Run(text, style="b")], level=level, number=number,
                   section_label=lab or ("other" if level == 1 else None), layout={"canonical": text})

    heading("Introduction")
    for runs in _fill_prose(blocks_txt["INTRO"], book, rng):
        add("para", runs)
    heading("Theoretical Background", level=2)
    for runs in _fill_prose(blocks_txt["THEORY"], book, rng):
        add("para", runs)

    # Each study needs its own procedure prose (PROCEDURE, MATERIALS); a third study would
    # repeat a paragraph verbatim, which extractors strip as a running line (catalog P07).
    n_studies = min(n_studies or rng.choice([1, 1, 2]), 2)
    stats: dict = {}
    sid = {"n": 0}
    plant_mix = plant_mix if plant_mix is not None else {"R01": 0.08, "R02": 0.10, "R03": 0.03,
                                                          "R04": 0.04, "R05a": 0.05, "R05c": 0.03,
                                                          "R06a": 0.02, "R06b": 0.02, "R08": 0.03}
    tables: list[Table] = []
    for st in range(1, n_studies + 1):
        pre = f"Study {st}: " if n_studies > 1 else ""
        if n_studies > 1:
            heading(f"Study {st}")
        heading("Method", level=2 if n_studies > 1 else 1)
        N = int(g.integers(80, 600))
        women = int(g.binomial(N, 0.55))
        age_m, age_sd = float(g.normal(29, 6)), float(abs(g.normal(8, 2)) + 2)
        heading("Participants", level=3 if n_studies > 1 else 2)
        add("para", [Run(f"A total of {N} adults took part ({women} women; age "), Run("M", style="i"),
                     Run(f" = {age_m:.2f}, "), Run("SD", style="i"), Run(f" = {age_sd:.2f}).")])
        heading("Procedure", level=3 if n_studies > 1 else 2)
        for runs in _fill_prose(blocks_txt["PROCEDURE"] if st == 1 else blocks_txt["MATERIALS"], book, rng):
            add("para", runs)
        heading("Results", level=2 if n_studies > 1 else 1)
        k = n_stats or rng.randint(4, 8)
        recs = []
        for j in range(k):
            tmpl, sim = _FAMILIES[int(g.integers(0, len(_FAMILIES)))]
            sid["n"] += 1
            rec = sim(g, seed * 1000 + sid["n"])
            rec.stat_id = f"s{sid['n']:03d}"
            plant = None
            u = rng.random()
            acc = 0.0
            for pid, w in plant_mix.items():
                acc += w
                if u < acc:
                    plant = pid
                    break
            print_record(rec, plant, rng)
            stats[rec.stat_id] = rec
            recs.append((tmpl, rec))
        for tmpl, rec in recs:
            sigword = "significant" if rec.p < 0.05 else "not significant"
            s = tmpl.format(dv=dv, iv=iv, IV=iv[0].upper() + iv[1:], SIGWORD=sigword, S="{S}")
            before, after = s.split("{S}")
            runs = [Run(before)] + stat_runs(rec) + [Run(after)]
            if rec.printed.get("note") == "one-tailed":
                runs[-1].text = " (one-tailed)" + runs[-1].text
            add("para", runs)
        # a results table for the t/F records of this study
        rows = [r for _, r in recs if r.family in ("t", "F")][:5]
        if rows:
            t = _results_table(f"t{len(tables) + 1}", f"Table {len(tables) + 1}", rows, pre)
            tables.append(t)
            add("table", table=t)

    heading("General Discussion" if n_studies > 1 else "Discussion")
    for runs in _fill_prose(blocks_txt["DISCUSSION"], book, rng):
        add("para", runs)
    heading("Limitations", level=2)
    for runs in _fill_prose(blocks_txt["LIMITATIONS"], book, rng):
        add("para", runs)
    heading("Conclusion")
    for runs in _fill_prose(blocks_txt["CONCLUSION"], book, rng):
        add("para", runs)

    refs = book.reference_list()
    abstract = [Run(blocks_txt["ABSTRACT_BACKGROUND"] + " ")]
    first = next(iter(stats.values()), None)
    if first is not None:
        abstract += [Run("The main comparison showed an effect, ")] + \
            [Run(r.text, style=r.style) for r in stat_runs(first)] + [Run(". ")]
    abstract += [Run(blocks_txt["ABSTRACT_CONCLUSION"])]
    fn = Footnote("n1", "1", [Run(blocks_txt["FOOTNOTE"])])
    # attach the footnote marker to the end of the first introduction paragraph
    first_para = next(b for b in blocks if b.kind == "para")
    first_para.runs.append(Run("1", style="sup", note_id="n1"))

    paper = Paper(paper_id=paper_id, seed=seed, split=split, topic=topic,
                  title=blocks_txt["TITLE"].strip().rstrip("."), authors=authors, affiliations=affils,
                  journal=journal, abstract=abstract,
                  keywords=[k.strip() for k in blocks_txt["KEYWORDS"].split(";") if k.strip()],
                  blocks=blocks, references=refs, cite_style=cite_style, ref_style=ref_style,
                  footnotes=[fn], stats=stats)
    paper.citation_markers = book.markers  # type: ignore[attr-defined]
    paper.citation_book = book  # type: ignore[attr-defined]  (recipes add citations through it)
    return paper


def _fmt_p(rec) -> str:
    return ("< " if rec.printed["p_rel"] == "<" else "") + rec.printed["p_txt"]


def _results_table(table_id: str, label: str, recs, pre: str) -> Table:
    hdr = [Cell([Run("Outcome")], is_header=True), Cell([Run("Test")], is_header=True),
           Cell([Run("df", style="i")], is_header=True), Cell([Run("Statistic")], is_header=True),
           Cell([Run("p", style="i")], is_header=True), Cell([Run("Effect size")], is_header=True),
           Cell([Run("95% CI")], is_header=True)]
    body = []
    for k, r in enumerate(recs, 1):
        pr = r.printed
        df_txt = (f"{_d(r.df1)}, {_d(r.df2)}" if r.df2 is not None else _d(r.df1))
        es = f"{pr['es_txt']}" if pr.get("es_txt") else "—"
        ci = f"[{_c(pr['ci'][0], r)}, {_c(pr['ci'][1], r)}]" if pr.get("ci") else "—"
        body.append([Cell([Run(f"Measure {k}")]), Cell([Run(r.family if r.family != "b" else "t", style="i")]),
                     Cell([Run(df_txt, stat_id=r.stat_id)]), Cell([Run(pr["stat_txt"], stat_id=r.stat_id)]),
                     Cell([Run(_fmt_p(r), stat_id=r.stat_id)]), Cell([Run(es, stat_id=r.stat_id)]),
                     Cell([Run(ci, stat_id=r.stat_id)])])
    cap = [Run(f"{pre}Inferential tests for the main outcomes")]
    note = [Run("Note. Effect sizes are Cohen's "), Run("d", style="i"),
            Run(" for t tests and partial eta squared for F tests.")]
    return Table(table_id=table_id, label=label, caption=cap, header=[hdr], body=body, note=note,
                 col_widths=[1.5, 0.6, 0.9, 1.0, 0.8, 0.9, 1.5])


def _d(x) -> str:
    if x is None:
        return "—"
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.2f}"


def _c(v, r) -> str:
    s = f"{v:.2f}"
    if r.es_name in ("r", "eta2p", "omega2", "phi", "V"):
        s = s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s
    return s
