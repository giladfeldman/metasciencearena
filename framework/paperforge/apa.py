"""APA-7 number formatting. Every number a paperforge document prints passes through here.

The formatting is deliberately the plain APA style that statcheck-era corpora use
(`t(141) = 2.31, p = .023`), so a synthetic paper reads like the psychology articles
the tools under test are built for. Nothing here computes a statistic; it only prints
one that `simulate.py` computed.
"""
from __future__ import annotations

import math


def _strip_leading_zero(s: str) -> str:
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def num(x: float, dp: int = 2) -> str:
    """A plain number, `dp` decimals, with a true minus sign avoided (ASCII hyphen)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise ValueError("refusing to format a missing number")
    s = f"{x:.{dp}f}"
    # "-0.00" is a rounding artefact, not a negative number.
    return s[1:] if s.startswith("-") and float(s) == 0.0 else s


def bounded(x: float, dp: int = 2) -> str:
    """A statistic that cannot exceed 1 in magnitude (r, alpha, eta squared): no leading zero."""
    return _strip_leading_zero(num(x, dp))


def p_value(p: float) -> str:
    """`p = .023` / `p < .001` (APA 7: exact to three decimals, floor at .001)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p out of range: {p}")
    if p < 0.001:
        return "p < .001"
    s = f"{p:.3f}"
    if s == "1.000":
        return "p > .999"  # a p that rounds to 1.000 is never printed as such
    return "p = " + _strip_leading_zero(s)


def df(x: float) -> str:
    """Degrees of freedom: integer when integral (Student), two decimals otherwise (Welch)."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}"


def t_test(t: float, dof: float, p: float) -> str:
    return f"t({df(dof)}) = {num(t)}, {p_value(p)}"


def f_test(f: float, df1: float, df2: float, p: float) -> str:
    return f"F({df(df1)}, {df(df2)}) = {num(f)}, {p_value(p)}"


def ci(lo: float, hi: float, dp: int = 2, level: int = 95) -> str:
    return f"{level}% CI [{num(lo, dp)}, {num(hi, dp)}]"


def mean_sd(m: float, sd: float) -> str:
    return f"M = {num(m)}, SD = {num(sd)}"


def integer(n: int) -> str:
    """Counts. APA spells out numbers below ten at the start of a sentence; callers
    place counts mid-sentence, so digits throughout keep the text greppable."""
    return str(int(n))


def percent(x: float, dp: int = 1) -> str:
    return f"{100 * x:.{dp}f}%"
