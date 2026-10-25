"""Statistics for the PDF universe: every value computed from simulated data.

Each `StatRecord` is one reported result. `simulate_*` draws raw data with a seeded
numpy generator, runs the real test (scipy), and keeps the exact values. Nothing is
typed in by hand, so the answer key is the computation, not a claim about it.

A record carries three layers, kept apart on purpose:

- **truth**: the exact computed statistic, df, two-tailed p, effect size and CI;
- **printed**: what the paper says, built from truth by APA formatting, then by an
  optional planted reporting error (catalog family R) that changes what is printed
  but never the truth;
- **verdict**: what a correct checker should conclude about the printed result
  (consistent / inconsistent / decision error), computed from the printed numbers
  with statcheck's rounding rule (a printed statistic stands for every value that
  rounds to it).

Notation variants (catalog family S: decimal comma, U+2212, unspaced CIs, ...) are a
later, purely typographic step (`notation.py`); they never change any layer here.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import optimize, stats

ALPHA = 0.05


# --------------------------------------------------------------------------- records
@dataclass
class StatRecord:
    stat_id: str
    family: str                 # t | F | chi2 | r | z | U | W | H | b | OR
    design: str                 # independent | welch | paired | one_sample | oneway | interaction | ...
    statistic: float
    df1: float | None
    df2: float | None
    n: int
    p: float                    # exact two-tailed (or upper-tail for F/chi2/H) p
    es_name: str | None = None
    es_value: float | None = None
    ci_level: int | None = None
    ci_lo: float | None = None
    ci_hi: float | None = None
    ci_of: str | None = None    # which quantity the CI is for ("es" | "b" | "or")
    se: float | None = None
    tails: int = 2
    alt_es: dict = field(default_factory=dict)   # other effect sizes computed alongside (e.g. omega2)
    # printed layer (filled by `print_record`)
    printed: dict = field(default_factory=dict)
    planted: str | None = None  # catalog R-id or None
    verdict: dict = field(default_factory=dict)

    def _mark_unavailable(self, plant: str) -> "StatRecord":
        self.printed["plant_unavailable"] = plant
        return self

    def as_gold(self) -> dict:
        return asdict(self)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------- CI helpers (exact)
def _nct_ci(t: float, dof: float, level: float) -> tuple[float, float]:
    """CI for a noncentrality parameter by inverting the noncentral t (MBESS method)."""
    a = (1 - level) / 2

    def f(ncp, target):
        v = stats.nct.cdf(t, dof, ncp)
        if not np.isfinite(v):
            raise FloatingPointError(f"nct.cdf({t}, {dof}, {ncp}) is not finite")
        return v - target

    def solve(target):
        # cdf(t | ncp) falls as ncp rises; grow a bracket from t in unit steps so the
        # solver never evaluates far tails, where scipy's nct.cdf returns NaN.
        lo, hi = t - 1.0, t + 1.0
        for _ in range(40):
            if f(lo, target) > 0:
                break
            lo -= 1.0
        for _ in range(40):
            if f(hi, target) < 0:
                break
            hi += 1.0
        return optimize.brentq(f, lo, hi, args=(target,), xtol=1e-10)
    return solve(1 - a), solve(a)


def _fisher_ci(r: float, n: int, level: float) -> tuple[float, float]:
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    q = stats.norm.ppf(1 - (1 - level) / 2)
    return math.tanh(z - q * se), math.tanh(z + q * se)


# ------------------------------------------------------------------ simulators
def simulate_independent_t(seed: int, n1: int, n2: int, d: float, welch: bool = False,
                           level: int = 95) -> StatRecord:
    g = _rng(seed)
    a = g.normal(d, 1.0, n1)
    b = g.normal(0.0, 1.0 if not welch else 1.6, n2)
    res = stats.ttest_ind(a, b, equal_var=not welch)
    t = float(res.statistic)
    if welch:
        v1, v2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2
        dof = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    else:
        dof = n1 + n2 - 2
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    dval = (a.mean() - b.mean()) / sp
    k = math.sqrt(1 / n1 + 1 / n2)
    lo, hi = _nct_ci(dval / k, n1 + n2 - 2, level / 100)
    return StatRecord(stat_id="", family="t", design="welch" if welch else "independent",
                      statistic=t, df1=dof, df2=None, n=n1 + n2, p=float(res.pvalue),
                      es_name="d", es_value=float(dval), ci_level=level,
                      ci_lo=lo * k, ci_hi=hi * k, ci_of="es")


def simulate_paired_t(seed: int, n: int, dz: float, level: int = 95) -> StatRecord:
    g = _rng(seed)
    diff = g.normal(dz, 1.0, n)
    res = stats.ttest_1samp(diff, 0.0)
    t = float(res.statistic)
    dzv = diff.mean() / diff.std(ddof=1)
    lo, hi = _nct_ci(t, n - 1, level / 100)
    s = math.sqrt(n)
    return StatRecord(stat_id="", family="t", design="paired", statistic=t, df1=n - 1, df2=None,
                      n=n, p=float(res.pvalue), es_name="dz", es_value=float(dzv),
                      ci_level=level, ci_lo=lo / s, ci_hi=hi / s, ci_of="es")


def simulate_one_sample_t(seed: int, n: int, d: float) -> StatRecord:
    r = simulate_paired_t(seed, n, d)
    r.design, r.es_name = "one_sample", "d"
    return r


def simulate_oneway_F(seed: int, k: int, n_per: int, f_eff: float) -> StatRecord:
    g = _rng(seed)
    mus = np.linspace(-f_eff, f_eff, k)
    groups = [g.normal(m, 1.0, n_per) for m in mus]
    res = stats.f_oneway(*groups)
    df1, df2 = k - 1, k * n_per - k
    F = float(res.statistic)
    eta2 = F * df1 / (F * df1 + df2)
    omega2 = max(0.0, (F - 1) * df1 / (F * df1 + df2 + 1))
    rec = StatRecord(stat_id="", family="F", design="oneway", statistic=F, df1=df1, df2=df2,
                     n=k * n_per, p=float(res.pvalue), es_name="eta2p", es_value=eta2)
    rec.alt_es = {"omega2": omega2}
    return rec


def simulate_interaction_F(seed: int, n_per_cell: int, eff: float) -> StatRecord:
    """2x2 between-subjects interaction F(1, N-4) from OLS; partial eta squared."""
    g = _rng(seed)
    a = np.repeat([0, 1, 0, 1], n_per_cell)
    b = np.repeat([0, 0, 1, 1], n_per_cell)
    y = g.normal(0, 1, 4 * n_per_cell) + eff * (a * b - 0.25) * 2
    X = np.column_stack([np.ones_like(y), a, b, a * b])
    beta, res_ss, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(resid @ resid)
    X0 = X[:, :3]
    b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    r0 = y - X0 @ b0
    sse0 = float(r0 @ r0)
    df2 = len(y) - 4
    F = (sse0 - sse) / (sse / df2)
    p = float(stats.f.sf(F, 1, df2))
    eta2p = (sse0 - sse) / ((sse0 - sse) + sse)
    return StatRecord(stat_id="", family="F", design="interaction", statistic=float(F), df1=1,
                      df2=df2, n=len(y), p=p, es_name="eta2p", es_value=float(eta2p))


def simulate_chi2(seed: int, n: int, rows: int, cols: int, eff: float) -> StatRecord:
    g = _rng(seed)
    base = np.full((rows, cols), 1.0)
    base[0, 0] += eff * rows
    probs = (base / base.sum()).ravel()
    counts = g.multinomial(n, probs).reshape(rows, cols)
    counts = counts + (counts == 0)  # no empty cells
    chi2, p, dof, _ = stats.chi2_contingency(counts, correction=False)
    V = math.sqrt(chi2 / (counts.sum() * (min(rows, cols) - 1)))
    return StatRecord(stat_id="", family="chi2", design=f"{rows}x{cols}", statistic=float(chi2),
                      df1=dof, df2=None, n=int(counts.sum()), p=float(p),
                      es_name="phi" if (rows, cols) == (2, 2) else "V", es_value=V)


def simulate_r(seed: int, n: int, rho: float, level: int = 95) -> StatRecord:
    g = _rng(seed)
    x = g.normal(0, 1, n)
    y = rho * x + math.sqrt(1 - rho ** 2) * g.normal(0, 1, n)
    r, p = stats.pearsonr(x, y)
    lo, hi = _fisher_ci(float(r), n, level / 100)
    return StatRecord(stat_id="", family="r", design="pearson", statistic=float(r), df1=n - 2,
                      df2=None, n=n, p=float(p), es_name="r", es_value=float(r),
                      ci_level=level, ci_lo=lo, ci_hi=hi, ci_of="es")


def simulate_z_proportions(seed: int, n1: int, n2: int, p1: float, p2: float) -> StatRecord:
    g = _rng(seed)
    x1, x2 = g.binomial(n1, p1), g.binomial(n2, p2)
    pp = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (x1 / n1 - x2 / n2) / se
    p = float(2 * stats.norm.sf(abs(z)))
    return StatRecord(stat_id="", family="z", design="two_proportions", statistic=float(z),
                      df1=None, df2=None, n=n1 + n2, p=p)


def simulate_mannwhitney(seed: int, n1: int, n2: int, shift: float) -> StatRecord:
    g = _rng(seed)
    a, b = g.normal(shift, 1, n1), g.normal(0, 1, n2)
    res = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    U = float(res.statistic)
    rrb = 1 - 2 * U / (n1 * n2)
    return StatRecord(stat_id="", family="U", design="mann_whitney", statistic=U, df1=None,
                      df2=None, n=n1 + n2, p=float(res.pvalue), es_name="r_rb", es_value=-rrb)


def simulate_wilcoxon(seed: int, n: int, shift: float) -> StatRecord:
    g = _rng(seed)
    d = g.normal(shift, 1, n)
    res = stats.wilcoxon(d, method="approx")
    return StatRecord(stat_id="", family="W", design="wilcoxon", statistic=float(res.statistic),
                      df1=None, df2=None, n=n, p=float(res.pvalue))


def simulate_kruskal(seed: int, k: int, n_per: int, shift: float) -> StatRecord:
    g = _rng(seed)
    groups = [g.normal(m, 1, n_per) for m in np.linspace(0, shift, k)]
    res = stats.kruskal(*groups)
    return StatRecord(stat_id="", family="H", design="kruskal_wallis", statistic=float(res.statistic),
                      df1=k - 1, df2=None, n=k * n_per, p=float(res.pvalue))


def simulate_regression_b(seed: int, n: int, beta: float, level: int = 95) -> StatRecord:
    g = _rng(seed)
    x = g.normal(0, 1, n)
    z = g.normal(0, 1, n)
    y = beta * x + 0.3 * z + g.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x, z])
    bhat, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ bhat
    dof = n - X.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(cov[1, 1])
    t = bhat[1] / se
    q = stats.t.ppf(1 - (1 - level / 100) / 2, dof)
    return StatRecord(stat_id="", family="b", design="ols", statistic=float(t), df1=dof, df2=None,
                      n=n, p=float(2 * stats.t.sf(abs(t), dof)), es_name="b",
                      es_value=float(bhat[1]), se=se, ci_level=level,
                      ci_lo=float(bhat[1] - q * se), ci_hi=float(bhat[1] + q * se), ci_of="b")


def simulate_odds_ratio(seed: int, n: int, log_or: float, level: int = 95) -> StatRecord:
    """2x2 odds ratio with Woolf (log) CI; z test on log OR."""
    g = _rng(seed)
    p0 = 0.3
    p1 = 1 / (1 + math.exp(-(math.log(p0 / (1 - p0)) + log_or)))
    n1 = n // 2
    a = g.binomial(n1, p1); b = n1 - a
    c = g.binomial(n - n1, p0); d = (n - n1) - c
    a, b, c, d = [v + 0.5 if min(a, b, c, d) == 0 else v for v in (a, b, c, d)]
    lor = math.log((a * d) / (b * c))
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    q = stats.norm.ppf(1 - (1 - level / 100) / 2)
    z = lor / se
    return StatRecord(stat_id="", family="OR", design="2x2", statistic=float(z), df1=None, df2=None,
                      n=n, p=float(2 * stats.norm.sf(abs(z))), es_name="OR",
                      es_value=math.exp(lor), se=se, ci_level=level,
                      ci_lo=math.exp(lor - q * se), ci_hi=math.exp(lor + q * se), ci_of="or")


# ------------------------------------------------------------- p recomputation
def p_from(family: str, stat: float, df1, df2, tails: int = 2) -> float:
    """p implied by a (printed) statistic, the way statcheck recomputes it."""
    if family in ("t", "b"):
        p = 2 * stats.t.sf(abs(stat), df1)
    elif family == "F":
        p = stats.f.sf(stat, df1, df2)
    elif family in ("chi2", "H"):
        p = stats.chi2.sf(stat, df1)
    elif family == "r":
        if abs(stat) >= 1:
            return 0.0
        t = stat * math.sqrt(df1 / (1 - stat ** 2))
        p = 2 * stats.t.sf(abs(t), df1)
    elif family in ("z", "OR"):
        p = 2 * stats.norm.sf(abs(stat))
    else:
        raise ValueError(f"no recomputation for family {family!r}")
    if tails == 1 and family in ("t", "b", "r", "z", "OR"):
        p /= 2
    return float(p)


def decimals_of(s: str) -> int:
    s = s.strip().lstrip("-−")
    return len(s.split(".", 1)[1]) if "." in s else 0


def p_range_from_printed_stat(family: str, stat_txt: str, df1, df2, tails: int = 2) -> tuple[float, float]:
    """statcheck's rounding rule: the printed statistic stands for [x - h, x + h)."""
    x = float(stat_txt.replace("−", "-"))
    h = 0.5 * 10 ** (-decimals_of(stat_txt))
    lo_s, hi_s = abs(x) - h, abs(x) + h
    if family == "r":
        lo_s, hi_s = max(lo_s, 0.0), min(hi_s, 0.999999)
    ps = [p_from(family, s if family != "r" else s, df1, df2, tails) for s in (max(lo_s, 0.0), hi_s)]
    return min(ps), max(ps)


def verdict(family: str, stat_txt: str, df1, df2, p_rel: str, p_txt: str, tails: int = 2) -> dict:
    """Consistency of a printed result, statcheck-style, from the PRINTED numbers only.

    p_rel is "=", "<" or ">"; p_txt the printed p ("".023", ".001"). Returns
    {"consistent", "decision_error", "recomputed_p_range"}.
    """
    pmin, pmax = p_range_from_printed_stat(family, stat_txt, df1, df2, tails)
    rp = float(p_txt)
    if p_rel == "=":
        h = 0.5 * 10 ** (-decimals_of(p_txt))
        rep_lo, rep_hi = max(rp - h, 0.0), rp + h
    elif p_rel == "<":
        rep_lo, rep_hi = 0.0, rp
    else:
        rep_lo, rep_hi = rp, 1.0
    # Reported significance follows statcheck's default pEqualAlphaSig = TRUE: a printed
    # p = .05 (or p < .05) claims significance; p > .05 does not.
    reported_sig = rp <= ALPHA if p_rel in ("=", "<") else False

    def judge(lo: float, hi: float) -> tuple[bool, bool]:
        cons = not (hi < rep_lo or lo > rep_hi)
        sig_any = lo <= ALPHA
        nonsig_any = hi > ALPHA
        de = (not cons) and ((reported_sig and not sig_any) or ((not reported_sig) and not nonsig_any))
        return cons, de

    # statcheck's rule: the printed statistic stands for its whole rounding interval.
    consistent, decision_error = judge(pmin, pmax)
    # The DECISION can be rule-dependent (consult 2026-10-17, finding 2): when the rounding
    # interval straddles .05, the range rule finds no decision error, while a reader who
    # takes the printed statistic at face value puts it on the other side of .05 from the
    # printed p. The gold takes no side there; the record leaves the verification
    # denominator. (Consistency itself is statcheck's range rule; a point rule would call
    # almost every correctly rounded result inconsistent, which no checker does.)
    p_point = p_from(family, float(stat_txt.replace("−", "-")), df1, df2, tails)
    point_sig = p_point <= ALPHA
    out = {"recomputed_p_range": [pmin, pmax], "recomputed_p_point": p_point}
    if not decision_error and p_rel in ("=", "<") and reported_sig != point_sig and pmin <= ALPHA < pmax:
        # An inconsistency under the range rule is one under the point rule too, so only
        # the decision-error label is in doubt then; a range-consistent result is in doubt whole.
        out.update({"consistent": False if consistent is False else None, "decision_error": None,
                    "ambiguous": True,
                    "range_rule": {"consistent": consistent, "decision_error": decision_error},
                    "point_rule": {"decision_error": True}})
        return out
    out.update({"consistent": bool(consistent), "decision_error": bool(decision_error)})
    return out
