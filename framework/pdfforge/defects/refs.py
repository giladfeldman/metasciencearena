"""Catalog families B (reference lists) and C (in-text citations and linking).

Recipes add references and citations through the paper's CitationBook (so the marker
gold and the reference list stay in step), or change how the list is printed
(layout directives the renderer reads: refs_layout, refs_reflow, refs_heading, ...).
"""
from __future__ import annotations

import random
import re

from ..model import Block, Cell, Paper, Reference, Run, Table
from ..refstyle import AUTHOR_YEAR, NUMERIC, make_reference, render_reference
from . import recipe


def _book(paper: Paper):
    return getattr(paper, "citation_book", None)


def _discussion_para(paper: Paper):
    seen = False
    for b in paper.blocks:
        if b.kind == "heading" and "Discussion" in (b.layout.get("canonical") or b.text):
            seen = True
        elif seen and b.kind == "para":
            return b
    return next((b for b in paper.blocks if b.kind == "para"), None)


def _add_ref(paper: Paper, rng: random.Random, **kw) -> Reference:
    book = _book(paper)
    idx = len(book.pool)
    words = ["choice", "judgment", "behaviour", "memory"]
    r = make_reference(rng, idx, words, **{k: v for k, v in kw.items() if k in ("flavour", "kind")})
    for k, v in kw.items():
        if k not in ("flavour", "kind"):
            setattr(r, k, v)
    book.pool.append(r)
    return r


def _cite_exact(paper: Paper, refs: list[Reference], text: str, block=None, prefix=" ", suffix="") -> None:
    """Append a citation with an exact printed form to a paragraph; record the marker."""
    book = _book(paper)
    for r in refs:
        if r.ref_id not in book.cited:
            book.cited.append(r.ref_id)
    book.markers.append({"marker_text": text, "reference_ids": [r.ref_id for r in refs]})
    b = block or _discussion_para(paper)
    b.runs += [Run(prefix), Run(text, cite_refs=[r.ref_id for r in refs]), Run(suffix)]


def _rebuild_reference_list(paper: Paper) -> None:
    paper.references = _book(paper).reference_list()


def _ay_name(r: Reference, amp=" & ") -> str:
    a = r.authors
    if len(a) == 1:
        return a[0][0]
    if len(a) == 2:
        return f"{a[0][0]}{amp}{a[1][0]}"
    return f"{a[0][0]} et al."


# ---------------------------------------------------------------- B: reference lists
@recipe("B01", "B", "Two-column reference list under a full-width table", "Rf", "yes",
        "DP test_o5_reference_inversion; CG handoff of 2026-06-07:278")
def b01(paper: Paper, rng: random.Random) -> bool:
    roles = ["Conceptualization", "Data curation", "Formal analysis", "Writing – original draft"]
    hdr = [Cell([Run("Role")], is_header=True)] + [Cell([Run(f"{s}")], is_header=True) for s, _ in paper.authors[:4]]
    body = [[Cell([Run(r)])] + [Cell([Run("✓" if rng.random() < .5 else "")]) for _ in paper.authors[:4]] for r in roles]
    paper.front_matter["credit_table"] = Table("tcredit", "", [Run("CRediT author statement")], [hdr], body,
                                               caption_style="untitled")
    paper.layout["refs_layout"] = "two_col_under_table"
    paper.add_defect("B01", None)
    return True


@recipe("B02", "B", "Long reference list crossing pages: running head and page number mid-entry", "Rf", "yes",
        "CL:1971, 910, 448; CG DOCPLUCK_FINDINGS_2026-08-04:76")
def b02(paper: Paper, rng: random.Random) -> bool:
    book = _book(paper)
    if book is None:
        return False
    para = _discussion_para(paper)
    for _ in range(4):
        para.runs += [Run(" Related work reached similar conclusions ")] + book.cite(3) + [Run(".")]
    _rebuild_reference_list(paper)
    paper.layout.setdefault("header_left", paper.journal["year"])
    paper.layout.setdefault("page_numbers", "{page}")
    paper.add_defect("B02", None, n_refs=len(paper.references))
    return True


@recipe("B03", "B", "Furniture and author bios around the reference list", "Rf", "yes", "CL:2079, 1134, 1327")
def b03(paper: Paper, rng: random.Random) -> bool:
    s, g = paper.authors[0]
    paper.front_matter["bios"] = [f"{g} {s} ({s.lower().replace(' ', '')}@synth-univ.example) is an associate professor at "
                                  f"{paper.affiliations[0].split(', ')[-1]}. Since {rng.randint(2010, 2020)}, their work has examined judgment."]
    paper.layout.setdefault("stamp", f"Downloaded from https://synth-journals.example by Norwick University Library on 14 March 2025")
    paper.add_defect("B03", None)
    return True


@recipe("B04", "B", "Reference entries reflowed into one paragraph (no hanging indent)", "Rf", "yes",
        "CL:1526, 1801, 1301")
def b04(paper: Paper, rng: random.Random) -> bool:
    paper.layout["refs_reflow"] = rng.choice(["one_paragraph", "pairs"])
    paper.layout["hanging_refs"] = False
    paper.add_defect("B04", None, mode=paper.layout["refs_reflow"])
    return True


@recipe("B05", "B", "DOI and URL wrapped mid-token; next entry glued after a URL", "Rf", "yes",
        "CL:1660; CG lesson 2026-03-14 (DOI broken across lines)")
def b05(paper: Paper, rng: random.Random) -> bool:
    url = _add_ref(paper, rng, kind="web")
    url.title = "Data and materials for the replication"
    url.venue = "Open Science Framework"
    url.doi = None
    _cite_exact(paper, [url], f"({_ay_name(url)}, {url.year})" if paper.cite_style in AUTHOR_YEAR else "")
    _rebuild_reference_list(paper)
    _fix_numeric_markers(paper, [url])
    url.raw_text = url.raw_text + " https://osf.io/tkm pc"
    n = 0
    for r in paper.references[: max(3, len(paper.references) // 3)]:
        if r.doi and r.doi in r.raw_text:
            i = r.raw_text.index(r.doi) + len(r.doi) // 2
            r.printed_runs = [Run(r.raw_text[:i]), Run("\n"), Run(r.raw_text[i:])]
            n += 1
    paper.add_defect("B05", None, wrapped=n)
    return n > 0


@recipe("B06", "B", "Organisation and full-given-name authors opening entries", "Rf Ci", "yes",
        "CL:1759, 1143, 1237")
def b06(paper: Paper, rng: random.Random) -> bool:
    if paper.cite_style not in AUTHOR_YEAR:
        return False
    org = _add_ref(paper, rng, flavour="org")
    full = _add_ref(paper, rng)
    full.authors = [(full.authors[0][0], "Martin L.")] + full.authors[1:]
    _cite_exact(paper, [org], f"({org.authors[0][0]}, {org.year})")
    _cite_exact(paper, [full], f"({_ay_name(full)}, {full.year})")
    _rebuild_reference_list(paper)
    paper.add_defect("B06", None, refs=[org.ref_id, full.ref_id])
    return True


@recipe("B07", "B", "Line wraps that leave digits at a line start in a numbered list", "Rf", "yes",
        "CL:1925, 1823, 1581")
def b07(paper: Paper, rng: random.Random) -> bool:
    if paper.cite_style not in NUMERIC:
        return False
    n = 0
    for r in paper.references[:6]:
        if r.lpage and r.lpage in r.raw_text:
            i = r.raw_text.rindex(r.lpage)
            r.printed_runs = [Run(r.raw_text[:i]), Run("\n"), Run(r.raw_text[i:])]
            n += 1
    if n:
        paper.add_defect("B07", None, wrapped=n)
    return n > 0


@recipe("B08", "B", "Numbering prefix variants ([1]. / 1.Author / 1Author) and a headerless list", "Rf", "yes",
        "CG lesson 2026-03-12; SA reference-parsing")
def b08(paper: Paper, rng: random.Random) -> bool:
    if paper.cite_style not in NUMERIC:
        return False
    form = rng.choice(["bracket_dot", "nospace", "glued"])
    for k, r in enumerate(paper.references, 1):
        body = re.sub(r"^(\[\d+\]|\d+\.)\s*", "", r.raw_text)
        r.raw_text = {"bracket_dot": f"[{k}]. {body}", "nospace": f"{k}.{body}", "glued": f"{k}{body}"}[form]
    paper.layout["refs_heading"] = None if rng.random() < 0.5 else "References"
    paper.add_defect("B08", None, form=form, headerless=paper.layout["refs_heading"] is None)
    return True


@recipe("B09", "B", "Titles ending in ? or !, internal periods, acronym-initial titles", "Rf", "yes", "CL:29, 1208, 1898")
def b09(paper: Paper, rng: random.Random) -> bool:
    titles = ["Is anchoring a matter of habit?", "TurkPrime. Com: A versatile crowdsourcing platform",
              "COVID-19 and trust in science: Evidence from two panels", "Replicate or perish!"]
    refs = []
    for t in titles:
        r = _add_ref(paper, rng)
        r.title = t
        refs.append(r)
    for r in refs:
        _cite_exact(paper, [r], f"({_ay_name(r)}, {r.year})" if paper.cite_style in AUTHOR_YEAR else "")
    _rebuild_reference_list(paper)
    _fix_numeric_markers(paper, refs)
    paper.add_defect("B09", None)
    return True


@recipe("B10", "B", "Date shapes: full dates, Jr., in press, n.d., Accessed..., arXiv ids", "Rf", "yes",
        "CL:541, 974, 518, 2347")
def b10(paper: Paper, rng: random.Random) -> bool:
    specs = [("in press", None), ("n.d.", None), (str(rng.randint(2018, 2024)), "arXiv:2103.04417")]
    refs = []
    for year, extra in specs:
        r = _add_ref(paper, rng)
        r.year = year
        if extra:
            r.venue, r.volume, r.issue, r.fpage, r.lpage, r.doi = extra, None, None, None, None, None
        refs.append(r)
    jr = _add_ref(paper, rng)
    jr.authors = [(jr.authors[0][0], "A. M., Jr.")] + jr.authors[1:]
    refs.append(jr)
    for r in refs:
        _cite_exact(paper, [r], f"({_ay_name(r)}, {r.year})" if paper.cite_style in AUTHOR_YEAR else "")
    _rebuild_reference_list(paper)
    _fix_numeric_markers(paper, refs)
    paper.add_defect("B10", None)
    return True


@recipe("B11", "B", "Particles, compound and diacritic surnames in cited works", "Rf Ci", "yes",
        "CL:704, 931, 1726, 691; CL:1955", probe="diff")
def b11(paper: Paper, rng: random.Random) -> bool:
    refs = [_add_ref(paper, rng, flavour=f) for f in ("particle", "particle", "diacritic", "compound", "camel")]
    for r in refs:
        _cite_exact(paper, [r], f"({_ay_name(r)}, {r.year})" if paper.cite_style in AUTHOR_YEAR else "")
    _rebuild_reference_list(paper)
    _fix_numeric_markers(paper, refs)
    paper.add_defect("B11", None)
    return True


@recipe("B12", "B", "IEEE / Vancouver list with abbreviated venues and mixed quote glyphs", "Rf", "yes",
        "CL:1030, 1741; CT:39")
def b12(paper: Paper, rng: random.Random) -> bool:
    if paper.ref_style not in ("ieee", "vancouver"):
        return False
    if paper.ref_style == "ieee":
        for r in paper.references[::2]:
            r.raw_text = r.raw_text.replace("“", "''").replace("”", "''")
    paper.add_defect("B12", None, style=paper.ref_style)
    return True


@recipe("B13", "B", "An error printed in the published list (extract as printed, do not fix)", "Rf", "yes", "CG TODO:123")
def b13(paper: Paper, rng: random.Random) -> bool:
    if not paper.references:
        return False
    r = rng.choice(paper.references)
    surname = r.authors[0][0]
    r.authors = [("Open", "S. C.")] + r.authors[1:]
    r.raw_text = r.raw_text.replace(surname, "Open", 1)
    paper.add_defect("B13", r.ref_id, printed_author="Open")
    return True


@recipe("B14", "B", "A second, short reference list in an appendix (main list must win)", "Rf", "yes",
        "CG lesson 2026-03-14; CT:31")
def b14(paper: Paper, rng: random.Random) -> bool:
    extra = [make_reference(rng, 900 + k, ["choice", "memory"]) for k in range(3)]
    paper.front_matter["appendix_refs"] = [render_reference(r, paper.ref_style, k + 1) for k, r in enumerate(extra)]
    paper.add_defect("B14", None)
    return True


def _fix_numeric_markers(paper: Paper, refs: list[Reference]) -> None:
    """In numeric styles a citation's printed form is its list number: fill those in."""
    if paper.cite_style not in NUMERIC:
        return
    book = _book(paper)
    for m in book.markers:
        if m["marker_text"] == "" and m["reference_ids"]:
            n = book.cited.index(m["reference_ids"][0]) + 1
            m["marker_text"] = str(n) if paper.cite_style == "superscript" else f"[{n}]"
    for b in paper.blocks:
        new = []
        for r in b.runs:
            if r.cite_refs and r.text == "":
                n = book.cited.index(r.cite_refs[0]) + 1
                if paper.cite_style == "superscript":
                    new.append(Run(str(n), style="sup", cite_refs=r.cite_refs))
                    continue
                r.text = f"[{n}]"
            if r.text == " " and new and new[-1].style == "sup":
                continue
            new.append(r)
        b.runs = new


# ---------------------------------------------------------------- C: citations
def _need_ay(paper):
    return paper.cite_style in AUTHOR_YEAR and _book(paper) is not None


@recipe("C01", "C", "Author-year paper with numbered equations and model columns [1] [2] [3]", "Ci", "yes",
        "CL:95; CG lesson 2026-09-25")
def c01(paper: Paper, rng: random.Random) -> bool:
    if not _need_ay(paper):
        return False
    meth = next((i for i, b in enumerate(paper.blocks) if b.kind == "para" and "took part" in b.text), None)
    if meth is None:
        return False
    eq = Block("bC01e", "para", [Run("The outcome was modelled as "), Run("y", style="i"), Run(" = "), Run("b", style="i"),
                                 Run("0", style="sub"), Run(" + "), Run("b", style="i"), Run("1", style="sub"),
                                 Run("x + e,          (1)")])
    paper.blocks.insert(meth + 1, eq)
    hdr = [Cell([Run("")], is_header=True)] + [Cell([Run(f"[{k}]")], is_header=True) for k in (1, 2, 3)]
    body = [[Cell([Run(lab)])] + [Cell([Run(f"{rng.uniform(-0.3, 0.5):.2f}")]) for _ in range(3)]
            for lab in ("Condition", "Baseline", "Age")]
    paper.blocks.insert(meth + 2, Block("bC01t", "table", table=Table("tC01", "Table 9", [Run("Coefficients from three specifications")],
                                                                    [hdr], body)))
    paper.add_defect("C01", "bC01t")
    return True


@recipe("C02", "C", "Superscript markers glued to digit-final words or set on the baseline", "Ci", "yes",
        "CL:1856; CG handoff of 2026-06-07:155")
def c02(paper: Paper, rng: random.Random) -> bool:
    if paper.cite_style != "superscript":
        return False
    n = 0
    for b in paper.blocks:
        for k, r in enumerate(b.runs):
            if r.cite_refs and r.style == "sup" and k > 0 and n < 4:
                prev = b.runs[k - 1]
                if n % 2 == 0:
                    prev.text = prev.text.rstrip() + " during COVID-19"
                else:
                    r.style = "r"
                    prev.text = prev.text.rstrip() + " "
                n += 1
    if n:
        paper.add_defect("C02", None, changed=n)
    return n > 0


@recipe("C03", "C", "Digit-glued decoys in a superscript-cited paper (beta1, BRCA1, isaric4c.net, r = .31)", "Ci", "yes",
        "CL:1705, 1778")
def c03(paper: Paper, rng: random.Random) -> bool:
    if paper.cite_style != "superscript":
        return False
    p = _discussion_para(paper)
    p.runs.append(Run(" Parameters beta1 and beta2 were estimated, the BRCA1 example is often cited, the tool is hosted at "
                      "isaric4c.net, and the correlation was r = .31 in pilot data."))
    paper.add_defect("C03", p.block_id)
    return True


@recipe("C04", "C", "Bracket and parenthesis decoys in a numeric paper ([0, 1], (1) ... (2))", "Ci", "yes",
        "CL:163, 1350")
def c04(paper: Paper, rng: random.Random) -> bool:
    if paper.cite_style != "numeric":
        return False
    p = _discussion_para(paper)
    p.runs.append(Run(" Probabilities were rescaled to [0, 1], and the procedure was as follows: (1) screen, (2) sort, (3) code."))
    paper.add_defect("C04", p.block_id)
    return True


@recipe("C05", "C", "Particle, diacritic and organisation authors cited, incl. sentence-initial", "Ci", "yes",
        "CL:775, 216, 1955, 1404")
def c05(paper: Paper, rng: random.Random) -> bool:
    if not _need_ay(paper):
        return False
    p = _discussion_para(paper)
    a = _add_ref(paper, rng, flavour="particle")
    o = _add_ref(paper, rng, flavour="org")
    d = _add_ref(paper, rng, flavour="diacritic")
    nm = a.authors[0][0]
    nm_cap = nm[0].upper() + nm[1:]
    _cite_exact(paper, [a], f"{nm_cap}{' et al.' if len(a.authors) > 2 else ''} ({a.year})", p, prefix=" As ", suffix=" argued, framing matters.")
    _cite_exact(paper, [o, d], f"({o.authors[0][0]}, {o.year}; {_ay_name(d)}, {d.year})", p, suffix=".")
    _rebuild_reference_list(paper)
    paper.add_defect("C05", p.block_id)
    return True


@recipe("C06", "C", "Several years for one author ((Bishop, 2019, 2020a, 2020b))", "Ci", "yes", "CL:2096, 300")
def c06(paper: Paper, rng: random.Random) -> bool:
    if not _need_ay(paper):
        return False
    base = _add_ref(paper, rng)
    base.authors = base.authors[:1]
    same = [base]
    for yr in ("2020a", "2020b"):
        r = _add_ref(paper, rng)
        r.authors = [base.authors[0]]
        r.year = yr
        same.append(r)
    base.year = "2019"
    _cite_exact(paper, same, f"({base.authors[0][0]}, 2019, 2020a, 2020b)", suffix=".")
    _rebuild_reference_list(paper)
    paper.add_defect("C06", None)
    return True


@recipe("C07", "C", "Citations with page and part locators ((X, 2009, p. 1397), (X 2012: 153))", "Ci", "yes",
        "CL:340, 1482")
def c07(paper: Paper, rng: random.Random) -> bool:
    if not _need_ay(paper):
        return False
    r1, r2 = _add_ref(paper, rng), _add_ref(paper, rng)
    _cite_exact(paper, [r1], f"({_ay_name(r1)}, {r1.year}, p. {rng.randint(10, 900)})", suffix=".")
    _cite_exact(paper, [r2], f"({r2.authors[0][0]} {r2.year}: {rng.randint(10, 400)})", suffix=".")
    _rebuild_reference_list(paper)
    paper.add_defect("C07", None)
    return True


@recipe("C08", "C", "Prose inside citation bundles (see for example ...; as cited in ...)", "Ci", "yes", "CL:2017, 1597")
def c08(paper: Paper, rng: random.Random) -> bool:
    if not _need_ay(paper):
        return False
    r1, r2, r3 = (_add_ref(paper, rng) for _ in range(3))
    _cite_exact(paper, [r1, r2], f"(see for example {_ay_name(r1)}, {r1.year}; {_ay_name(r2)}, {r2.year})", suffix=".")
    _cite_exact(paper, [r3], f"(as cited in {_ay_name(r3)}, {r3.year})", suffix=".")
    _rebuild_reference_list(paper)
    paper.add_defect("C08", None)
    return True


@recipe("C09", "C", "Narratives with connectors and possessives (Similarly, A, B, and C (2023); Barr's (2009))", "Ci", "yes",
        "CL:766, 885, 2070")
def c09(paper: Paper, rng: random.Random) -> bool:
    if not _need_ay(paper):
        return False
    r = _add_ref(paper, rng)
    while len(r.authors) < 3:
        r.authors.append(("Robbdal", "K."))
    names = ", ".join(s for s, _ in r.authors[:-1]) + f", and {r.authors[-1][0]}"
    p = _discussion_para(paper)
    _cite_exact(paper, [r], f"{names} ({r.year})", p, prefix=" Similarly, ", suffix=" found the same pattern.")
    q = _add_ref(paper, rng)
    q.authors = q.authors[:1]
    _cite_exact(paper, [q], f"{q.authors[0][0]}'s ({q.year})", p, prefix=" ", suffix=" account differs.")
    _rebuild_reference_list(paper)
    paper.add_defect("C09", p.block_id)
    return True


@recipe("C10", "C", "Unusual shapes: unparenthesised and year-elided citations", "Ci", "yes", "CL:630, 786, 2135")
def c10(paper: Paper, rng: random.Random) -> bool:
    if not _need_ay(paper):
        return False
    r = _add_ref(paper, rng)
    r.authors = r.authors[:2] if len(r.authors) >= 2 else r.authors + [("Feeldam", "S.")]
    p = _discussion_para(paper)
    _cite_exact(paper, [r], f"{r.authors[0][0]} & {r.authors[1][0]}, {r.year}", p, prefix=" ", suffix=" found that framing matters.")
    _cite_exact(paper, [r], f"({r.authors[0][0]} & {r.authors[1][0]}, p. {rng.randint(10, 600)})", p, suffix=".")
    _rebuild_reference_list(paper)
    paper.add_defect("C10", p.block_id)
    return True


@recipe("C11", "C", "A citation-shaped string inside a reference title (not an in-text citation)", "Ci", "yes",
        "CL:1940; CR:240")
def c11(paper: Paper, rng: random.Random) -> bool:
    if not paper.references:
        return False
    r = rng.choice(paper.references)
    r.title = r.title + ": An extension of Kogvane and Ritholm (2005) Study 2"
    _rebuild_reference_list(paper)
    paper.add_defect("C11", r.ref_id)
    return True


@recipe("C12", "C", "A month in parentheses that looks like a citation ((January 2023))", "Ci", "yes", "CL:2268")
def c12(paper: Paper, rng: random.Random) -> bool:
    p = _discussion_para(paper)
    p.runs.append(Run(" At the time of writing (January 2023), no direct replication had been published."))
    paper.add_defect("C12", p.block_id)
    return True


@recipe("C13", "C", "Linking traps: same first author and year, different co-authors; et al. vs two-author entry", "Ci", "yes",
        "CL:141, 23, 2088; CG TODO:2409")
def c13(paper: Paper, rng: random.Random) -> bool:
    if not _need_ay(paper):
        return False
    a = _add_ref(paper, rng)
    b_ = _add_ref(paper, rng)
    first = a.authors[0]
    a.authors = [first, ("Hegland", "M."), ("Fishburn", "B.")]
    b_.authors = [first, ("Hardwell", "T."), ("Austvik", "F."), ("Nordal", "E.")]
    b_.year = a.year
    _cite_exact(paper, [a], f"({first[0]}, Hegland, & Fishburn, {a.year})", suffix=".")
    _cite_exact(paper, [b_], f"({first[0]} et al., {a.year})", suffix=".")
    _rebuild_reference_list(paper)
    paper.add_defect("C13", None, refs=[a.ref_id, b_.ref_id])
    return True


@recipe("C14", "C", "A citation bundle split across a page break", "Ci", "yes", "CL:448; CG handoff of 2026-06-30:13")
def c14(paper: Paper, rng: random.Random) -> bool:
    if not _need_ay(paper):
        return False
    for b in paper.blocks:
        for k, r in enumerate(b.runs):
            if r.cite_refs and "; " in r.text:
                i = r.text.index("; ") + 2
                b.runs[k:k + 1] = [Run(r.text[:i], cite_refs=r.cite_refs), Run(r.text[i:], cite_refs=r.cite_refs)]
                b.layout.setdefault("splits", []).append({"at": k + 1, "kind": "page", "block": None})
                paper.add_defect("C14", b.block_id)
                return True
    return False
