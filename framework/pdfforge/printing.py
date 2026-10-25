"""Print a StatRecord the way a paper would, optionally with a planted reporting error.

`print_record(rec, plant=None, rng=...)` fills `rec.printed` with the canonical APA
pieces (statistic, df, p relation and value, effect size, CI) and the assembled
string, then fills `rec.verdict` by checking those PRINTED numbers with
`statgen.verdict`. A planted error (catalog family R) changes only what is printed.

`plant` values (catalog ids):
  R01 gross decision error: printed p on the other side of .05
  R02 inconsistent p, same side of .05
  R03 p printed as ".000"
  R04 p halved with no one-tailed statement
  R05a statistic rounded to one decimal (correct, looks off)
  R05c one-tailed test declared in the text, p halved (correct)
  R06a CI bounds reversed
  R06b estimate outside its own CI
  R07a bounded effect size out of range (eta2p > 1 / V > 1 / r CI beyond 1)
  R08 effect size inconsistent with the test
"""
from __future__ import annotations

import math
import random

from .statgen import ALPHA, StatRecord, decimals_of, verdict

LABEL = {"t": "t", "b": "t", "F": "F", "chi2": "χ2", "r": "r", "z": "z", "U": "U", "W": "W",
         "H": "H", "OR": "z"}
ES_LABEL = {"d": "d", "dz": "dz", "g": "g", "eta2p": "ηp2", "omega2": "ω2", "r": "r", "phi": "φ",
            "V": "V", "r_rb": "r", "b": "b", "OR": "OR"}
BOUNDED = {"r", "eta2p", "omega2", "phi", "V", "r_rb"}


def _num(x: float, dp: int = 2) -> str:
    s = f"{x:.{dp}f}"
    return s[1:] if s.startswith("-") and float(s) == 0.0 else s


def _nolead(s: str) -> str:
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def _df(x) -> str:
    if x is None:
        return ""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.2f}"


def _p_parts(p: float) -> tuple[str, str]:
    if p < 0.001:
        return "<", ".001"
    s = f"{p:.3f}"
    if s == "1.000":
        return ">", ".999"
    return "=", _nolead(s)


def _es_txt(name: str, v: float) -> str:
    return _nolead(_num(v)) if name in BOUNDED else _num(v)


def print_record(rec: StatRecord, plant: str | None = None, rng: random.Random | None = None) -> StatRecord:
    rng = rng or random.Random(0)
    fam = rec.family
    stat_dp = 2
    stat_val = rec.statistic
    if fam in ("U", "W"):
        stat_txt = f"{stat_val:.1f}" if abs(stat_val - round(stat_val)) > 1e-9 else str(int(round(stat_val)))
    elif fam == "r":
        stat_txt = _nolead(_num(stat_val, 2))
    else:
        stat_txt = _num(stat_val, stat_dp)
    p_rel, p_txt = _p_parts(rec.p)
    es_name, es_val = rec.es_name, rec.es_value
    ci_lo, ci_hi = rec.ci_lo, rec.ci_hi
    tails = rec.tails
    note = None
    unavailable = False

    if plant == "R01":
        # The printed p claims the other side of .05; each candidate is checked with the
        # verdict so the planted record is a decision error by construction.
        cands = ([("<", ".05"), ("<", ".01"), ("=", ".032"), ("=", ".021"), ("=", ".012")]
                 if rec.p >= ALPHA else
                 [("=", ".062"), ("=", ".081"), ("=", ".14"), ("=", ".27"), ("=", ".45")])
        unavailable = True
        if fam in ("t", "b", "F", "chi2", "r", "z", "H", "OR"):
            for rel, txt in cands:
                if verdict(fam, stat_txt, rec.df1, rec.df2, rel, txt, tails)["decision_error"]:
                    p_rel, p_txt, unavailable = rel, txt, False
                    break
    elif plant == "R02":
        # A printed p that is provably inconsistent with the printed statistic yet on the
        # same side of .05, chosen by checking each candidate with the verdict itself.
        sig = rec.p < ALPHA
        pool = ([0.002, 0.004, 0.008, 0.012, 0.018, 0.024, 0.031, 0.038, 0.044] if sig
                else [0.062, 0.078, 0.094, 0.13, 0.19, 0.27, 0.36, 0.48, 0.61, 0.74, 0.88])
        rng.shuffle(pool)
        unavailable = True
        if fam in ("t", "b", "F", "chi2", "r", "z", "H", "OR"):
            for cand in pool:
                txt = _nolead(f"{cand:.3f}")
                v = verdict(fam, stat_txt, rec.df1, rec.df2, "=", txt, tails)
                if not v["consistent"] and not v["decision_error"]:
                    p_rel, p_txt, unavailable = "=", txt, False
                    break
    elif plant == "R03":
        p_rel, p_txt = "=", ".000"
    elif plant == "R04":
        p_rel, p_txt = _p_parts(rec.p / 2)
    elif plant == "R05a":
        stat_txt = _num(stat_val, 1)
    elif plant == "R05c":
        tails = 1
        p_rel, p_txt = _p_parts(rec.p / 2)
        note = "one-tailed"
    elif plant == "R06a" and ci_lo is not None:
        ci_lo, ci_hi = ci_hi, ci_lo
    elif plant == "R06b" and ci_lo is not None:
        w = ci_hi - ci_lo
        ci_lo, ci_hi = es_val + 0.1 * w, es_val + 1.1 * w
    elif plant == "R07a":
        if es_name in ("eta2p", "V", "phi"):
            es_val = 1.0 + rng.uniform(0.05, 0.6)
        elif ci_hi is not None and es_name == "r":
            ci_hi = 1.0 + rng.uniform(0.02, 0.2)
    elif plant == "R08" and es_val is not None:
        es_val = es_val * rng.choice([2.2, 0.35]) + (0.3 if abs(es_val) < 0.05 else 0)

    label = LABEL[fam]
    head = f"{label}({_df(rec.df1)}" + (f", {_df(rec.df2)}" if rec.df2 is not None else "") + ")" if rec.df1 is not None else label
    if fam == "chi2":
        head = f"χ2({_df(rec.df1)}, N = {rec.n})"
    if fam == "b":
        main = f"b = {_num(es_val)}, SE = {_num(rec.se)}, t({_df(rec.df1)}) = {stat_txt}"
    else:
        main = f"{head} = {stat_txt}"
    p_str = f"p {p_rel} {p_txt}"
    pieces = [main, p_str]
    es_str = ci_str = None
    if es_name and fam not in ("b", "r"):
        es_str = f"{ES_LABEL[es_name]} = {_es_txt(es_name, es_val)}"
        pieces.append(es_str)
    ci_txt = None
    if ci_lo is not None and rec.ci_level:
        fmt = (lambda v: _nolead(_num(v))) if es_name in BOUNDED else _num
        ci_txt = [fmt(ci_lo), fmt(ci_hi)]
        ci_str = f"{rec.ci_level}% CI [{ci_txt[0]}, {ci_txt[1]}]"
        pieces.append(ci_str)
    text = ", ".join(pieces)
    # A planted error must be VISIBLE in the printed numbers (consult 2026-10-17,
    # findings 7, 9, 10); if rounding hides it, refuse and let the builder pick another.
    es_print = _es_txt(es_name, es_val) if es_name and es_val is not None else None
    if plant == "R04" and (p_rel, p_txt) == _p_parts(rec.p):
        unavailable = True
    if plant == "R06a" and (ci_txt is None or ci_txt[0] == ci_txt[1]):
        unavailable = True
    if plant == "R06b":
        shown = float(es_print) if es_print is not None else (float(_num(es_val)) if es_val is not None else None)
        if ci_txt is None or shown is None or float(ci_txt[0]) <= shown <= float(ci_txt[1]):
            unavailable = True
    if unavailable:
        # This record cannot host the requested error (e.g. its rounding interval straddles
        # .05, so no printed p is a decision error). Printed clean; the builder picks another.
        return print_record(rec, None, rng)._mark_unavailable(plant)
    rec.planted = plant
    rec.printed = {"text": text, "stat_txt": stat_txt, "df1": rec.df1, "df2": rec.df2,
                   "p_rel": p_rel, "p_txt": p_txt, "es_name": es_name,
                   "es_txt": _es_txt(es_name, es_val) if es_name and es_val is not None else None,
                   "ci": [ci_lo, ci_hi] if ci_lo is not None else None, "ci_txt": ci_txt, "tails": tails,
                   "note": note}
    rec.verdict = _verdict(rec, stat_txt, p_rel, p_txt, tails, es_name, es_val, ci_lo, ci_hi)
    if plant == "R04" and rec.verdict.get("consistent") is not False:
        return print_record(rec, None, rng)._mark_unavailable(plant)
    return rec


def _verdict(rec, stat_txt, p_rel, p_txt, tails, es_name, es_val, ci_lo, ci_hi) -> dict:
    v: dict = {}
    if rec.family in ("t", "b", "F", "chi2", "r", "z", "H", "OR"):
        if p_txt == ".000":
            v.update({"consistent": None, "decision_error": False, "p_zero": True})
        else:
            # the df AS PRINTED (Welch df appears to 2 dp), consult finding 5
            pdf1 = float(_df(rec.df1)) if rec.df1 is not None else None
            pdf2 = float(_df(rec.df2)) if rec.df2 is not None else None
            v.update(verdict(rec.family, stat_txt, pdf1, pdf2, p_rel, p_txt, tails))
    else:
        v.update({"consistent": None, "decision_error": None, "uncheckable_family": rec.family})
    flags = []
    if ci_lo is not None and ci_hi is not None:
        # judged on the PRINTED (rounded) numbers, which is all a reader or tool can see
        fmt = (lambda x: float(_nolead(_num(x)))) if es_name in BOUNDED else (lambda x: float(_num(x)))
        lo_p, hi_p = fmt(ci_lo), fmt(ci_hi)
        es_p = fmt(es_val) if es_val is not None else None
        if lo_p > hi_p:
            flags.append("ci_reversed")
        if es_p is not None and not (min(lo_p, hi_p) <= es_p <= max(lo_p, hi_p)) \
                and rec.ci_of in ("es", "b", "or"):
            flags.append("estimate_outside_ci")
    if es_name in BOUNDED and es_val is not None and abs(es_val) > 1:
        flags.append("impossible_es")
    if es_name == "r" and ci_hi is not None and max(abs(ci_lo), abs(ci_hi)) > 1:
        flags.append("impossible_ci")
    if rec.planted == "R08":
        flags.append("es_inconsistent_with_test")
    if rec.planted == "R04":
        flags.append("p_halved_without_one_tailed_claim")
    v["flags"] = flags
    return v
