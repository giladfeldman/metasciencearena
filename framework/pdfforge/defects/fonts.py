"""Catalog family F: the text layer says something other than what the page shows.

Pattern for a "lying glyph" recipe: the target characters (inside chosen statistics,
or document-wide) are drawn in a per-defect alias font (`render.register_alias`), and
the post step rewrites only that font's ToUnicode map. The page is pixel-identical to
a clean render; the extracted text is what real broken PDFs produce.
"""
from __future__ import annotations

import random
import re
import unicodedata

from ..model import Paper, Run
from ..render import register_alias
from . import _clone, map_stat_runs, recipe, split_chars, stat_blocks
from .post import rewrite_tounicode

MINUS = "−"


def _alias(defect_id: str) -> str:
    return register_alias(f"PF-Lie-{defect_id}")


def _true_minus(runs: list[Run]) -> list[Run]:
    out = []
    for r in runs:
        t = re.sub(r"-(?=[\d.])", MINUS, r.text) if not r.image else r.text
        out.append(_clone(r, t))
    return out


def _stats_with(paper: Paper, rng: random.Random, pred, k: int, in_table=False) -> list[str]:
    cands = []
    for b, sid in stat_blocks(paper, in_table=in_table):
        rec = paper.stats[sid]
        if pred(rec) and sid not in cands:
            cands.append(sid)
    rng.shuffle(cands)
    return cands[:k]


def _has_negative(rec) -> bool:
    return "-" in rec.printed["text"]


def _lie_recipe(defect_id: str, chars: str, mapping, *, prepare=None, pick=None, k=3,
                in_table=False, doc_wide=False):
    """Build apply/post for a lying-glyph defect on `chars`."""
    def apply(paper: Paper, rng: random.Random) -> bool:
        alias = _alias(defect_id)
        if doc_wide:
            hit = False
            for b in paper.blocks:
                if b.kind in ("para", "list_item"):
                    if prepare:
                        b.runs = prepare(b.runs)
                    if any(ch in r.text for r in b.runs for ch in chars):
                        b.runs = split_chars(b.runs, lambda ch, i, r: ch in chars and not r.font, font=alias)
                        hit = True
            if hit:
                paper.add_defect(defect_id, None, scope="document", chars=chars)
            return hit
        targets = _stats_with(paper, rng, pick or (lambda rec: True), k, in_table)
        hits = 0
        for sid in targets:
            def fn(runs, sid=sid):
                if prepare:
                    runs = prepare(runs)
                return split_chars(runs, lambda ch, i, r: ch in chars and not r.font, font=alias)
            map_stat_runs(paper, sid, fn)
            if any(ch in paper.stats[sid].printed["text"].replace("-", MINUS) for ch in chars):
                paper.add_defect(defect_id, sid, chars=chars)
                hits += 1
        return hits > 0

    def post(pdf: bytes, paper: Paper, ctx: dict) -> bytes:
        out, n = rewrite_tounicode(pdf, f"PF-Lie-{defect_id}", mapping)
        ctx.setdefault("post_changes", {})[defect_id] = n
        return out
    return apply, post


def _register_lie(defect_id, title, corrupts, recoverable, sources, chars, mapping, **kw):
    apply, post = _lie_recipe(defect_id, chars, mapping, **kw)
    recipe(defect_id, "F", title, corrupts, recoverable, sources, post=post)(apply)


# --- lying glyph maps -----------------------------------------------------------
_register_lie("F03", "Minus glyph mapped to digit 2", "St Tb", "glyph",
              "DP test_minus_sign_recovery_real_pdf", MINUS, {MINUS: "2"},
              prepare=_true_minus, pick=_has_negative)
_register_lie("F04", "'<' mapped to backslash", "St Tb Rf", "glyph",
              "DP test_lt_operator_recovery_real_pdf", "<", {"<": "\\"},
              pick=lambda rec: rec.printed["p_rel"] == "<")
_register_lie("F05", "'<' mapped to the letter b", "St", "glyph",
              "DP test_w0o_lt_as_b_glyph_recovery", "<", {"<": "b"},
              pick=lambda rec: rec.printed["p_rel"] == "<")
_register_lie("F07", "'=' mapped to 5 and minus to 2 (operator-only font)", "St", "glyph",
              "DP test_operator_glyph_evidence_real_pdf", "=" + MINUS, {"=": "5", MINUS: "2"},
              prepare=_true_minus)
_register_lie("F08", "'=' mapped to U+00BC throughout the paper", "St", "glyph",
              "EM handoff of 2026-06-29:122; EM 2026-10-07 comparison", "=", {"=": "¼"},
              doc_wide=True)
_register_lie("F09", "Minus glyph with no Unicode mapping (dropped / cid)", "St Tb", "glyph",
              "DP test_dropped_minus_layout_recovery; test_cid_minus_recovery", MINUS, {MINUS: None},
              prepare=_true_minus, pick=_has_negative)
_register_lie("F12", "Operator font with no ToUnicode at all (Linotype-Pi-like)", "St", "glyph",
              "DP test_linotype_pi_glyph_names; EM NEWS:279; ME inbox 2026-10-06", "=<" + MINUS, None,
              prepare=_true_minus)
_register_lie("F15", "Minus mapped to U+FFFD (undecodable glyph)", "St", "no",
              "DP test_fffd_passes_through; EM test-parse:557", MINUS, {MINUS: "�"},
              prepare=_true_minus, pick=_has_negative)
_register_lie("F17", "Comma inside a CI mapped to a period", "St", "glyph",
              "DP test_a4_ci_period_to_comma", ",", {",": "."},
              pick=lambda rec: rec.printed.get("ci") is not None)


# --- symbol letters ---------------------------------------------------------------
def _to_beta(paper: Paper, rng: random.Random) -> list[str]:
    """Regression results report a standardised beta: 'b = ...' -> 'β = ...'."""
    sids = []
    for sid, rec in paper.stats.items():
        if rec.family == "b":
            def fn(runs):
                out = []
                for r in runs:
                    if r.text == "b" and r.style == "i":
                        out.append(_clone(r, "β", style="r"))
                    else:
                        out.append(r)
                return out
            map_stat_runs(paper, sid, fn)
            sids.append(sid)
    return sids


@recipe("F10", "F", "β drawn in a symbol font mapped to Latin b", "St", "glyph",
        "DP nz:5714; OVERHAUL_REGISTER:109",
        post=lambda pdf, paper, ctx: rewrite_tounicode(pdf, "PF-Lie-F10", {"β": "b"})[0])
def f10(paper: Paper, rng: random.Random) -> bool:
    sids = _to_beta(paper, rng)
    alias = _alias("F10")
    for sid in sids:
        map_stat_runs(paper, sid, lambda runs: split_chars(runs, lambda ch, i, r: ch == "β", font=alias))
        paper.add_defect("F10", sid)
    return bool(sids)


@recipe("F11", "F", "Greek in a symbol font decoded to Private-Use-Area codepoints", "Tx St", "glyph",
        "DP test_pua_glyph_recovery_real_pdf; EK 03 gallery #7",
        post=lambda pdf, paper, ctx: rewrite_tounicode(pdf, "PF-Lie-F11", {"χ": "", "β": "", "η": ""})[0])
def f11(paper: Paper, rng: random.Random) -> bool:
    _to_beta(paper, rng)
    alias = _alias("F11")
    hit = False
    for sid, rec in paper.stats.items():
        if rec.family in ("chi2", "b") or rec.es_name == "eta2p":
            map_stat_runs(paper, sid, lambda runs: split_chars(runs, lambda ch, i, r: ch in "χβη", font=alias))
            paper.add_defect("F11", sid)
            hit = True
    return hit


_MATH_ITALIC = {"p": "\U0001D45D", "t": "\U0001D461", "F": "\U0001D439", "d": "\U0001D451",
                "r": "\U0001D45F", "η": "\U0001D702", "β": "\U0001D6FD", "χ": "\U0001D712"}


@recipe("F13", "F", "Statistic symbols as math-italic Unicode (𝑝, 𝑡, 𝜂)", "St", "yes",
        "DP NORM:18; test_mathitalic_greek_real_pdf; EK 03 #1")
def f13(paper: Paper, rng: random.Random) -> bool:
    hit = False
    for sid in list(paper.stats)[: max(2, len(paper.stats) // 2)]:
        def fn(runs):
            out = []
            for r in runs:
                if r.text in _MATH_ITALIC and r.style in ("i", "r") and len(r.text) == 1:
                    out.append(_clone(r, _MATH_ITALIC[r.text], style="r", gold_text=r.text))
                else:
                    out.append(r)
            return out
        map_stat_runs(paper, sid, fn)
        paper.add_defect("F13", sid)
        hit = True
    return hit


@recipe("F16", "F", "Minus drawn as an image (no character object)", "St", "no",
        "DP test_dropped_minus_layout (documented limit); DP L-010; EK 03 #10")
def f16(paper: Paper, rng: random.Random) -> bool:
    from ..glyphs import glyph_png
    png = glyph_png(MINUS)
    targets = _stats_with(paper, rng, _has_negative, 2)
    for sid in targets:
        def fn(runs):
            runs = _true_minus(runs)
            return split_chars(runs, lambda ch, i, r: ch == MINUS, image=png, gold_text=MINUS)
        map_stat_runs(paper, sid, fn)
        paper.add_defect("F16", sid)
    return bool(targets)


# --- invisible / spacing characters (content-level) ------------------------------
@recipe("F02", "F", "True minus U+2212 in statistics and CIs", "St Tb", "yes",
        "DP NORM:116; EM test-golden:116; EK 01")
def f02(paper: Paper, rng: random.Random) -> bool:
    hit = False
    for sid, rec in paper.stats.items():
        if _has_negative(rec):
            map_stat_runs(paper, sid, _true_minus)
            paper.add_defect("F02", sid)
            hit = True
    return hit


_LIG = [("ffi", "ﬃ"), ("ffl", "ﬄ"), ("ff", "ﬀ"), ("fi", "ﬁ"), ("fl", "ﬂ")]


@recipe("F01", "F", "Ligature glyphs (ﬁ ﬂ ﬀ ﬃ ﬄ) in the text layer", "Tx Tb Rf Ci", "yes",
        "DP NORM:75; CL:1503; EM test-parse:461; SA text-fidelity v2")
def f01(paper: Paper, rng: random.Random) -> bool:
    n = 0
    for b in paper.blocks:
        if b.kind not in ("para", "heading"):
            continue
        new = []
        for r in b.runs:
            t = r.text
            if not (r.stat_id or r.cite_refs or r.image):
                for a, lig in _LIG:
                    if a in t:
                        n += t.count(a)
                        t = t.replace(a, lig)
            new.append(_clone(r, t, gold_text=r.text if t != r.text else r.gold_text))
        b.runs = new
    for ref in paper.references:
        t = ref.raw_text
        for a, lig in _LIG:
            t = t.replace(a, lig)
        ref.printed_runs = [Run(t, gold_text=ref.raw_text)]
    if n:
        paper.add_defect("F01", None, ligatures=n)
    return n > 0


@recipe("F21", "F", "Soft hyphens U+00AD inside long words", "Tx Ci Rf", "yes",
        "DP test_normalize_soft_hyphen_dehyphenation; CL:1549")
def f21(paper: Paper, rng: random.Random) -> bool:
    n = 0
    for b in paper.blocks:
        if b.kind != "para":
            continue
        new = []
        for r in b.runs:
            if r.stat_id or r.cite_refs or r.image or r.font:
                new.append(r)
                continue

            def shy(m):
                nonlocal n
                w = m.group(0)
                n += 1
                k = len(w) // 2
                return w[:k] + "­" + w[k:]
            t = re.sub(r"[A-Za-z]{11,}", shy, r.text)
            new.append(_clone(r, t, gold_text=r.text if t != r.text else r.gold_text))
        b.runs = new
    if n:
        paper.add_defect("F21", None, soft_hyphens=n)
    return n > 0


@recipe("F22", "F", "Zero-width / bidi marks after citations", "Ci Tx", "yes",
        "DP test_normalize_bidi_marks_stripped")
def f22(paper: Paper, rng: random.Random) -> bool:
    n = 0
    for b in paper.blocks:
        for r in b.runs:
            if r.cite_refs and r.style != "sup":
                r.gold_text = r.gold_text or r.text
                r.text = r.text + rng.choice(["‎", "​", "﻿"])
                n += 1
    if n:
        paper.add_defect("F22", None, marks=n)
    return n > 0


@recipe("F23", "F", "NBSP / thin / narrow spaces inside statistics", "St", "yes", "DP NORM:139")
def f23(paper: Paper, rng: random.Random) -> bool:
    sids = list(paper.stats)[: max(2, len(paper.stats) // 2)]
    for sid in sids:
        sp = rng.choice([" ", " ", " "])
        map_stat_runs(paper, sid, lambda runs, sp=sp: [_clone(r, r.text.replace(" ", sp), gold_text=r.text)
                                                       if " " in r.text else r for r in runs])
        paper.add_defect("F23", sid, space=hex(ord(sp)))
    return bool(sids)


@recipe("F20", "F", "Decomposed diacritics (base letter + combining mark) in names", "Tx Rf Ci", "yes",
        "DP NORM:55; CG handoff 06-30:44")
def f20(paper: Paper, rng: random.Random) -> bool:
    n = 0
    for ref in paper.references:
        t = unicodedata.normalize("NFD", ref.raw_text)
        if t != ref.raw_text:
            ref.printed_runs = [Run(t, gold_text=ref.raw_text)]
            n += 1
    if n:
        paper.add_defect("F20", None, references=n)
    return n > 0


@recipe("F28", "F", "ηp² with a Greek rho as the subscript", "St", "yes", "EM NEWS:247", probe="notation")
def f28(paper: Paper, rng: random.Random) -> bool:
    hit = False
    for sid, rec in paper.stats.items():
        if rec.es_name == "eta2p":
            map_stat_runs(paper, sid, lambda runs: [_clone(r, "ρ", gold_text="ρ") if (r.text == "p" and r.style == "sub") else r for r in runs])
            paper.add_defect("F28", sid)
            hit = True
    return hit


@recipe("F29", "F", "χ² spellings: superscript ², chi^2, X2", "St", "yes",
        "EK 03 #19; EM NEWS:3415; EK 01 answer-key 4,12", probe="notation")
def f29(paper: Paper, rng: random.Random) -> bool:
    hit = False
    for sid, rec in paper.stats.items():
        if rec.family != "chi2":
            continue
        form = rng.choice(["sup2char", "caret", "X2"])

        def fn(runs, form=form):
            out = []
            i = 0
            while i < len(runs):
                r = runs[i]
                if r.text == "χ" and i + 1 < len(runs) and runs[i + 1].style == "sup":
                    rep = {"sup2char": "χ²", "caret": "chi^2", "X2": "X2"}[form]
                    out.append(_clone(r, rep))
                    i += 2
                    continue
                out.append(r)
                i += 1
            return out
        map_stat_runs(paper, sid, fn)
        paper.add_defect("F29", sid, form=form)
        hit = True
    return hit


@recipe("F31", "F", "OCR-like letter-for-digit in a printed statistic (author-level)", "St", "no",
        "EK 01 answer-key 32", probe="notation")
def f31(paper: Paper, rng: random.Random) -> bool:
    targets = _stats_with(paper, rng, lambda rec: rec.family == "t", 1)
    for sid in targets:
        def fn(runs):
            out = []
            for r in runs:
                if r.text.startswith("(") and any(c.isdigit() for c in r.text):
                    out.append(_clone(r, r.text.replace("1", "l", 1)))
                elif r.text.startswith(" = ") and "0" in r.text:
                    out.append(_clone(r, r.text.replace("0", "O", 1)))
                else:
                    out.append(r)
            return out
        map_stat_runs(paper, sid, fn)
        paper.add_defect("F31", sid)
    return bool(targets)


# --- more lying maps on body text --------------------------------------------------
def _alias_chars_in_block(paper, block, pred, alias):
    block.runs = split_chars(block.runs, pred, font=alias)


@recipe("F06", "F", "Multiplication sign drawn in a font that names it 'three'", "Tb Tx", "glyph",
        "DP test_times_glyph_recovery_real_pdf; ME w0k retrospective",
        post=lambda pdf, paper, ctx: rewrite_tounicode(pdf, "PF-Lie-F06", {"×": "3"})[0])
def f06(paper: Paper, rng: random.Random) -> bool:
    alias = _alias("F06")
    meth = next((b for b in paper.blocks if b.kind == "para" and "took part" in b.text), None)
    if meth is None:
        return False
    # the design sentence carries the ×; "Model 3" is a real 3 that must stay a 3 (decoy)
    meth.runs += [Run(" The design was a 2 × 2 factorial, and Model 3 of the analysis plan was used.")]
    _alias_chars_in_block(paper, meth, lambda ch, i, r: ch == "×", alias)
    paper.add_defect("F06", meth.block_id, decoy="Model 3")
    return True


@recipe("F18", "F", "Copyright sign under a mismatched encoding (© read as Ó)", "Tx Se", "yes",
        "DP test_sage_masthead_strip; test_elsevier_footer_strip",
        post=lambda pdf, paper, ctx: rewrite_tounicode(pdf, "PF-Lie-F18", {"©": "Ó"})[0])
def f18(paper: Paper, rng: random.Random) -> bool:
    _alias("F18")
    paper.front_matter.setdefault("masthead", []).append(f"© {paper.journal['year']} The Authors")
    paper.front_matter["masthead_font_alias"] = {"©": "PF-Lie-F18"}
    paper.add_defect("F18", "masthead")
    return True


@recipe("F19", "F", "A letter inside a word decoded to a control character", "Tx Se", "glyph",
        "DP NORM:242, 575-591 (a letter inside 'Schrittmacher' read as U+0002)",
        post=lambda pdf, paper, ctx: rewrite_tounicode(pdf, "PF-Lie-F19", {"t": "\u0002"})[0])
def f19(paper: Paper, rng: random.Random) -> bool:
    alias = _alias("F19")
    n = 0
    for b in paper.blocks:
        if b.kind != "para" or n >= 3:
            continue
        for k, r in enumerate(b.runs):
            m = re.search(r"\b\w*tt\w*\b", r.text) if not (r.stat_id or r.cite_refs or r.font) else None
            if m:
                cut = m.start() + m.group(0).index("tt") + 1
                b.runs[k:k + 1] = [_clone(r, r.text[:cut]), _clone(r, "t", font=alias), _clone(r, r.text[cut + 1:])]
                n += 1
                paper.add_defect("F19", b.block_id, word=m.group(0))
                break
    return n > 0


@recipe("F24", "F", "A sentence period decoded to a NUL byte", "Tx", "yes", "DP NORM:39",
        post=lambda pdf, paper, ctx: rewrite_tounicode(pdf, "PF-Lie-F24", {".": "\u0000"})[0])
def f24(paper: Paper, rng: random.Random) -> bool:
    alias = _alias("F24")
    b = next((b for b in paper.blocks if b.kind == "para" and b.text.endswith(".")), None)
    if b is None:
        return False
    last = b.runs[-1]
    if not last.text.endswith(".") or last.stat_id:
        return False
    b.runs[-1:] = [_clone(last, last.text[:-1]), _clone(last, ".", font=alias)]
    paper.add_defect("F24", b.block_id)
    return True


@recipe("F25", "F", "Word spaces drawn as glyphs with no Unicode (no space characters)", "Tx Tb", "yes",
        "DP L-007; LESSONS:9 (space ratio 0.005)",
        post=lambda pdf, paper, ctx: rewrite_tounicode(pdf, "PF-Lie-F25", {" ": None})[0])
def f25(paper: Paper, rng: random.Random) -> bool:
    alias = _alias("F25")
    paras = [b for b in paper.blocks if b.kind == "para" and not any(r.stat_id for r in b.runs)]
    if not paras:
        return False
    for b in rng.sample(paras, min(2, len(paras))):
        b.runs = split_chars(b.runs, lambda ch, i, r: ch == " " and not r.font and r.style != "sup", font=alias)
        paper.add_defect("F25", b.block_id)
    return True


@recipe("F30", "F", "Publisher wrong glyph in a table header", "Tb", "glyph", "EM TODO:527",
        post=lambda pdf, paper, ctx: rewrite_tounicode(pdf, "PF-Lie-F30", {"S": "\u2bd1"})[0])
def f30(paper: Paper, rng: random.Random) -> bool:
    alias = _alias("F30")
    for b in paper.blocks:
        if b.kind == "table" and b.table is not None:
            for c in b.table.header[0]:
                if c.text.startswith("S"):
                    c.runs = split_chars(c.runs, lambda ch, i, r: i == 0 and ch == "S", font=alias)
                    paper.add_defect("F30", b.block_id, header=c.text)
                    return True
    return False
