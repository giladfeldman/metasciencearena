"""Catalog families S (notation) and R (planted reporting errors).

S: the same simulated values printed the way other journals and authors write them.
The values never change; only rec.printed["text_visible"] (what the page shows) does.
R: a planted reporting error re-prints the record via printing.print_record, which
recomputes the verdict a correct checker should reach.

Recipes act on statistics in BODY paragraphs (table cells are built from printed
fields elsewhere), and record each statistic they touch.
"""
from __future__ import annotations

import random
import re

from .. import statgen as sg
from ..content import text_runs
from ..model import Block, Paper, Run
from ..printing import print_record
from . import recipe, stat_blocks


def _body_sids(paper: Paper, rng: random.Random, pred=lambda rec: True, k=3,
               allow_table_copy: bool = False) -> list[str]:
    """Statistics printed in body paragraphs. Notation recipes may also take one that the
    results table restates (only the text copy changes -- a real prose/table mismatch);
    planted-error recipes may not, because the record has a single verdict."""
    s = []
    for b, sid in stat_blocks(paper, in_table=False):
        if sid not in s and pred(paper.stats[sid]) and (allow_table_copy or not _in_table(paper, sid)):
            s.append(sid)
    rng.shuffle(s)
    return s[:k]


def _in_table(paper: Paper, sid: str) -> bool:
    return any(b.kind == "table" and any(r.stat_id == sid for row in b.table.header + b.table.body for c in row for r in c.runs)
               for b in paper.blocks)


def set_visible(paper: Paper, sid: str, new_text: str) -> None:
    """Replace the painted form of statistic sid everywhere in body paragraphs."""
    rec = paper.stats[sid]
    rec.printed["text_visible"] = new_text
    new_runs = text_runs(new_text, sid)
    for b in paper.blocks:
        if b.kind == "table" or not any(r.stat_id == sid for r in b.runs):
            continue
        out, done = [], False
        for r in b.runs:
            if r.stat_id == sid:
                if not done:
                    out += [Run(x.text, style=x.style, stat_id=sid) for x in new_runs]
                    done = True
                continue
            out.append(r)
        b.runs = out


def _vis(rec) -> str:
    return rec.printed.get("text_visible") or rec.printed["text"]


def _new_sentence_stat(paper: Paper, rng: random.Random, rec: sg.StatRecord, sentence: str, plant=None) -> str:
    """Register a new statistic and add a results sentence carrying it."""
    n = len(paper.stats) + 1
    while f"s{n:03d}" in paper.stats:
        n += 1
    rec.stat_id = f"s{n:03d}"
    print_record(rec, plant, rng)
    paper.stats[rec.stat_id] = rec
    before, after = sentence.split("{S}")
    blk = Block(f"bN{n}", "para", [Run(before)] + text_runs(rec.printed["text"], rec.stat_id) + [Run(after)])
    last = max((i for i, b in enumerate(paper.blocks) if b.kind == "para" and any(r.stat_id for r in b.runs)),
               default=len(paper.blocks) - 1)
    paper.blocks.insert(last + 1, blk)
    return rec.stat_id


def _simple(defect_id: str, title: str, sources: str, pred, transform, k=3, corrupts="St"):
    @recipe(defect_id, "S", title, corrupts, "yes", sources, probe="notation")
    def fn(paper: Paper, rng: random.Random) -> bool:
        sids = _body_sids(paper, rng, pred, k, allow_table_copy=True)
        hit = 0
        for sid in sids:
            new = transform(_vis(paper.stats[sid]), paper.stats[sid], rng)
            if new and new != _vis(paper.stats[sid]):
                set_visible(paper, sid, new)
                paper.add_defect(defect_id, sid, printed=new)
                hit += 1
        return hit > 0
    return fn


# ---------------------------------------------------------------- S: notation
def _comma_decimal(t, rec, rng):
    t = re.sub(r"\[([^\]]+)\]", lambda m: "[" + m.group(1).replace(", ", "; ") + "]", t)
    t = re.sub(r"(?<=\d)\.(?=\d)", ",", t)
    return re.sub(r"(?<=[=<>] )\.(?=\d)", ",", t)


_simple("S01", "Decimal comma (t(40) = -2,48, p = 0,017 / p = ,025)", "EK 01 STATUS:17; EM NEWS:1454",
        lambda rec: True, _comma_decimal)
_simple("S03", "Unspaced CI (95%CI=[7.94,11.98])", "EM NEWS:1441",
        lambda rec: rec.ci_lo is not None,
        lambda t, rec, rng: re.sub(r"(\d+)% CI \[([^,\]]+), ([^\]]+)\]", r"\1%CI=[\2,\3]", t))
_simple("S04", "df written as a flattened subscript (t754 = -33.00, F1,200 = 5.32)", "EM test-parse:323",
        lambda rec: rec.family in ("t", "F", "r"),
        lambda t, rec, rng: re.sub(r"^([tFr])\(([^)]+)\)", lambda m: m.group(1) + m.group(2).replace(" ", ""), t))


def _es_variant(t, rec, rng):
    if "ηp2" in t:
        return t.replace("ηp2", rng.choice(["η^2p", "eta2_p", "ηp²", "ηG2"]))
    if re.search(r"\bd = ", t):
        return re.sub(r"\bd = ", rng.choice(["D = ", "Hedges' G = ", "Cohen's d = "]), t, count=1)
    return None


_simple("S05", "Effect-size spelling and order variants (η^2p, eta2_p, D, Hedges' G)", "EM NEWS:3350, 925, 3978",
        lambda rec: rec.es_name in ("eta2p", "d"), _es_variant)
_simple("S10", "Spacing variants (= 2.21p = .035; = - 3.79; t(28)=2.1,p=.045)", "EM test-parse:483, 645",
        lambda rec: True,
        lambda t, rec, rng: rng.choice([t.replace(", p ", "p ", 1), t.replace(" = -", " = - "),
                                        t.replace(" = ", "=").replace(", ", ",")]))


def _ci_delim(t, rec, rng):
    m = re.search(r"(\d+)% CI \[([^,\]]+), ([^\]]+)\]", t)
    if not m:
        return None
    lvl, lo, hi = m.groups()
    form = rng.choice([f"{lvl}% CI [{lo}; {hi}]", f"{lvl}% CI {lo}–{hi}", f"{lvl}% CI [{lo} to {hi}]",
                       f"{lvl}% CI: {lo}, {hi}", f"{lvl}% CI {{{lo}, {hi}}}"])
    return t[:m.start()] + form + t[m.end():]


_simple("S11", "CI delimiter variants ([a; b], a–b, [a to b], CI: a, b, braces)", "EM test-parse:238; DP test_a4_braces",
        lambda rec: rec.ci_lo is not None, _ci_delim)
_simple("S12", "Markdown italics reaching the text (*t*(17) = ...)", "EM NEWS:235; EK 01 STATUS:27",
        lambda rec: rec.family in ("t", "F"),
        lambda t, rec, rng: re.sub(r"^([tF])\(", r"*\1*(", t).replace(", p ", ", *p* ", 1))
_simple("S13", "Kruskal-Wallis H with a stray space (H (2))", "EK 03 gallery #6",
        lambda rec: rec.family == "H", lambda t, rec, rng: t.replace("H(", "H (", 1))
_simple("S14", "Unicode superscript in effect-size names (η²p)", "DP test_a5_*; test_a6_*",
        lambda rec: rec.es_name == "eta2p", lambda t, rec, rng: t.replace("ηp2", "η²p"))


def _p_forms(t, rec, rng):
    if rec.p < 0.001:
        return re.sub(r"p < \.001", rng.choice(["p = <.001", f"p = {rec.p:.2e}".replace("e-0", "e-"), "p < 10^-3"]), t)
    if rec.p > 0.05 and rng.random() < 0.6:
        return re.sub(r", p [=<>] [.\d]+", rng.choice([", ns", ", n.s."]), t)
    return None


_simple("S08", "p-value forms (p = <.001, e-notation, ns / n.s.)", "EM NEWS:3519, 2766, 4209; EK 01",
        lambda rec: rec.p < 0.001 or rec.p > 0.05, _p_forms)
_simple("S09", "JASP-style subscripted test labels (χ2Pearson(1))", "EM NEWS:3535, 1090",
        lambda rec: rec.family == "chi2", lambda t, rec, rng: t.replace("χ2(", "χ2Pearson(", 1))


@recipe("S02", "S", "Thousands separators in N and df; Greenhouse-Geisser fractional df that must stay", "St", "yes",
        "EM NEWS:1589, 3790, 1464; test-parse:577", probe="notation")
def s02(paper: Paper, rng: random.Random) -> bool:
    big = sg.simulate_independent_t(rng.randrange(10**6), rng.randint(1200, 1600), rng.randint(1200, 1600), rng.uniform(0.0, 0.2))
    sid = _new_sentence_stat(paper, rng, big, "In the full panel, the two cohorts were compared, {S}.")
    set_visible(paper, sid, re.sub(r"t\((\d)(\d{3})\)", r"t(\1,\2)", _vis(paper.stats[sid])))
    chi = sg.simulate_chi2(rng.randrange(10**6), rng.randint(1100, 3000), 2, 2, rng.uniform(0.0, 0.2))
    sid2 = _new_sentence_stat(paper, rng, chi, "Retention was cross-tabulated with cohort, {S}.")
    set_visible(paper, sid2, re.sub(r"N = (\d)(\d{3})", r"N = \1,\2", _vis(paper.stats[sid2])))
    # Greenhouse-Geisser: fractional df that a thousands rule must NOT touch
    k, n = 4, rng.randint(30, 60)
    base = sg.simulate_oneway_F(rng.randrange(10**6), k, n, 0.25)
    eps = round(rng.uniform(0.55, 0.85), 2)
    base.df1, base.df2 = base.df1 * eps, base.df2 * eps
    from scipy import stats as _st
    base.p = float(_st.f.sf(base.statistic, base.df1, base.df2))
    base.design = "rm_greenhouse_geisser"
    sid3 = _new_sentence_stat(paper, rng, base, "With a Greenhouse-Geisser correction, ratings changed across the four blocks, {S}.")
    for s_ in (sid, sid2, sid3):
        paper.add_defect("S02", s_)
    return True


@recipe("S06", "S", "Odd CI levels (90%, 99%)", "St", "yes", "ME lesson 2026-04-06 (99.9% CI)", probe="notation")
def s06(paper: Paper, rng: random.Random) -> bool:
    for lvl in (90, 99):
        rec = sg.simulate_independent_t(rng.randrange(10**6), rng.randint(30, 80), rng.randint(30, 80), rng.uniform(-0.3, 0.7), level=lvl)
        sid = _new_sentence_stat(paper, rng, rec, "A second comparison of the two conditions gave {S}.")
        paper.add_defect("S06", sid, level=lvl)
    return True


@recipe("S07", "S", "A Likert range '(1, 7)' right before an F test (df-as-CI trap)", "St", "yes",
        "ME lesson 2026-04-06 (47,885 false CIs)", probe="notation")
def s07(paper: Paper, rng: random.Random) -> bool:
    for b, sid in stat_blocks(paper, in_table=False):
        if paper.stats[sid].family == "F":
            i = next(k for k, r in enumerate(b.runs) if r.stat_id == sid)
            b.runs.insert(i, Run("ratings on a seven-point scale (1, 7) were compared, "))
            paper.add_defect("S07", sid)
            return True
    return False


@recipe("S15", "S", "Two results in one sentence (cross-binding trap)", "St", "yes", "EM NEWS:1680, 3035, 3614",
        probe="notation")
def s15(paper: Paper, rng: random.Random) -> bool:
    paras = [i for i, b in enumerate(paper.blocks) if b.kind == "para" and any(r.stat_id for r in b.runs)]
    for i, j in zip(paras, paras[1:]):
        if j == i + 1:
            a, b = paper.blocks[i], paper.blocks[j]
            sa = next(r.stat_id for r in a.runs if r.stat_id)
            sb = next(r.stat_id for r in b.runs if r.stat_id)
            ra = [r for r in a.runs if r.stat_id == sa]
            rb = [r for r in b.runs if r.stat_id == sb]
            a.runs = [Run("The two comparisons gave ")] + ra + [Run(" and ")] + rb + [Run(", respectively.")]
            paper.blocks.pop(j)
            paper.add_defect("S15", sa, partner=sb)
            return True
    return False


@recipe("S16", "S", "Decoy numbers next to results (Section 139.6, $20, rho = 0.7, 10,000 permutations)", "St", "yes",
        "EM NEWS:2969, 2742, 1782; EK 01", probe="notation")
def s16(paper: Paper, rng: random.Random) -> bool:
    paras = [b for b in paper.blocks if b.kind == "para" and any(r.stat_id for r in b.runs)]
    if not paras:
        return False
    decoys = ["Participants received a $20 gift card (see Section 139.6 of the protocol).",
              "The sensitivity analysis assumed a correlation of ρ = 0.7 between measures.",
              "Significance was assessed with 10,000 permutations."]
    for b, d in zip(rng.sample(paras, min(3, len(paras))), decoys):
        b.runs.append(Run(" " + d))
        paper.add_defect("S16", b.block_id, decoy=d)
    return True


@recipe("S17", "S", "Several different N = values in one paper", "St", "yes", "EM NEWS:2264", probe="notation")
def s17(paper: Paper, rng: random.Random) -> bool:
    b = next((b for b in paper.blocks if b.kind == "para" and "took part" in b.text), None)
    if b is None:
        return False
    inv = rng.randint(500, 900)
    b.runs.insert(0, Run(f"Of the "))
    b.runs.insert(1, Run("N", style="i"))
    b.runs.insert(2, Run(f" = {inv} people invited, {rng.randint(300, inv - 50)} opened the survey. "))
    paper.add_defect("S17", b.block_id, invited=inv)
    return True


@recipe("S18", "S", "A result quoted again later (restatement, not a new finding)", "St", "yes", "EM NEWS:482",
        probe="notation")
def s18(paper: Paper, rng: random.Random) -> bool:
    sids = _body_sids(paper, rng, k=1)
    disc = next((b for b in paper.blocks if b.kind == "para" and b.block_id > "b0" and
                 any(h.kind == "heading" and "Discussion" in h.text for h in paper.blocks[:paper.blocks.index(b)])), None)
    if not sids or disc is None:
        return False
    sid = sids[0]
    quoted = [Run(r.text, style=r.style, stat_id=sid) for r in text_runs(_vis(paper.stats[sid]), sid)]
    disc.runs += [Run(" As reported above (")] + quoted + [Run("), the effect was in the expected direction.")]
    paper.add_defect("S18", sid, quoted_in=disc.block_id)
    return True


@recipe("S19", "S", "Non-parametric family present (U, W, H) alongside parametric tests", "St", "yes",
        "EM 2026-10-07 comparison (26 unparsed)", probe="notation")
def s19(paper: Paper, rng: random.Random) -> bool:
    _new_sentence_stat(paper, rng, sg.simulate_mannwhitney(rng.randrange(10**6), 35, 38, 0.5),
                       "A Mann-Whitney test compared the skewed response times, {S}.")
    _new_sentence_stat(paper, rng, sg.simulate_wilcoxon(rng.randrange(10**6), 40, 0.4),
                       "A Wilcoxon signed-rank test compared the two sessions, {S}.")
    sid = _new_sentence_stat(paper, rng, sg.simulate_kruskal(rng.randrange(10**6), 3, 30, 0.6),
                             "A Kruskal-Wallis test compared the three groups, {S}.")
    paper.add_defect("S19", sid)
    return True


# ------------------------------------------------------- R: planted reporting errors
def _plant(defect_id: str, title: str, sources: str, plant: str, pred=lambda rec: True, k=2):
    @recipe(defect_id, "R", title, "St", "yes", sources, probe="notation")
    def fn(paper: Paper, rng: random.Random) -> bool:
        hit = 0
        for sid in _body_sids(paper, rng, pred, k * 3):
            rec = paper.stats[sid]
            print_record(rec, plant, rng)
            if rec.printed.get("plant_unavailable"):
                continue
            rec.printed.pop("text_visible", None)
            set_visible(paper, sid, rec.printed["text"])
            if rec.printed.get("note") == "one-tailed":
                for b in paper.blocks:
                    if any(r.stat_id == sid for r in b.runs):
                        last = max(i for i, r in enumerate(b.runs) if r.stat_id == sid)
                        b.runs.insert(last + 1, Run(" (one-tailed)"))
                        break
            paper.add_defect(defect_id, sid, verdict=rec.verdict)
            hit += 1
            if hit >= k:
                break
        return hit > 0
    return fn


_CHECKABLE = lambda rec: rec.family in ("t", "F", "chi2", "r", "z", "b", "OR", "H")  # noqa: E731
_plant("R01", "Gross decision error (printed p on the other side of .05)", "EK 01; EM NEWS:125", "R01", _CHECKABLE)
_plant("R02", "Inconsistent p, same side of .05", "EK 01", "R02", _CHECKABLE)
_plant("R03", "p printed as .000", "EM NEWS:137; EK 01", "R03", lambda rec: _CHECKABLE(rec) and rec.p < 0.0005)
_plant("R04", "p halved with no one-tailed statement", "EM NEWS:134", "R04", lambda rec: rec.family in ("t", "z", "r", "b"))
_plant("R05a", "Statistic rounded to one decimal (correct, looks off)", "EM NEWS:130", "R05a", _CHECKABLE)
_plant("R05c", "Declared one-tailed test (correct)", "EM NEWS:134", "R05c", lambda rec: rec.family in ("t", "z", "r", "b"))
_plant("R06a", "CI bounds reversed", "EM NEWS:960", "R06a", lambda rec: rec.ci_lo is not None)
_plant("R06b", "Estimate outside its own CI", "EM NEWS:1028; EK 01:26", "R06b", lambda rec: rec.ci_lo is not None)
_plant("R07a", "Bounded effect size or CI out of range (impossible value)", "EM NEWS:1558, 4260",
       "R07a", lambda rec: rec.es_name in ("eta2p", "V", "phi") or (rec.es_name == "r" and rec.ci_hi is not None))
@recipe("R08", "R", "Effect size inconsistent with the test", "St", "yes", "EM tests", probe="notation")
def r08(paper: Paper, rng: random.Random) -> bool:
    # d results are usually restated in the results table, so R08 adds its own result.
    rec = sg.simulate_independent_t(rng.randrange(10**6), rng.randint(30, 80), rng.randint(30, 80), rng.uniform(0.2, 0.7))
    sid = _new_sentence_stat(paper, rng, rec, "On the secondary measure, the conditions differed, {S}.", plant="R08")
    paper.add_defect("R08", sid, verdict=paper.stats[sid].verdict)
    return True


@recipe("R09", "R", "Author typo that must pass through unchanged (p < 05)", "St", "yes",
        "DP NORM:337; EM handoff of 2026-09-02", probe="notation")
def r09(paper: Paper, rng: random.Random) -> bool:
    for sid in _body_sids(paper, rng, lambda rec: rec.printed["p_rel"] == "<" or rec.p < 0.05, 3):
        t = _vis(paper.stats[sid])
        new = re.sub(r"p [<=] \.(\d+)", lambda m: "p < " + m.group(1), t, count=1)
        if new != t:
            set_visible(paper, sid, new)
            paper.stats[sid].verdict = {**paper.stats[sid].verdict, "author_typo": "p printed without its decimal point"}
            paper.add_defect("R09", sid, printed=new)
            return True
    return False


@recipe("R10", "R", "Values next to dashes and ranges (extractor-invented minus trap)", "St", "yes",
        "EM NEWS:953 (docpluck W0g)", probe="notation")
def r10(paper: Paper, rng: random.Random) -> bool:
    for sid in _body_sids(paper, rng, lambda rec: rec.family == "b", 1):
        for b in paper.blocks:
            if any(r.stat_id == sid for r in b.runs):
                i = next(k for k, r in enumerate(b.runs) if r.stat_id == sid)
                b.runs.insert(i, Run("on the 1–7 scale — "))
                paper.add_defect("R10", sid)
                return True
    return False
