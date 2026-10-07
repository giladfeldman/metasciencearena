"""Generator for p-curve-v1 (computed-gold evidential-value arena).

Scientific task: given a SET of independent statistically-significant findings
(p < .05), each supplied as a test statistic, decide whether the set has
**evidential value** by the p-curve method (Simonsohn, Nelson & Simons, 2014,
*JEP:General* 143(2):534-547). p-curve tests the RIGHT-SKEW of the distribution of
significant p-values: when significant results reflect real, well-powered effects,
their p-values cluster near 0 (more p≈.01 than p≈.04 → right-skewed p-curve). When
they reflect selection over null effects (false positives) the curve is FLAT
(uniform on [0,.05]); intense p-hacking pushes mass just under .05 (LEFT-skewed).

This arena is genuinely DISTINCT from `zcurve-evidential-v1`. The z-curve arena fits
an EM mixture over the observed significant z-values to estimate the expected
discovery / replication rate. p-curve instead converts each significant result to its
**pp-value** (the conditional probability under the null of a p at least this small,
given p<.05 = p/.05), z-transforms it, and runs Stouffer's combination to test the
right-skew of the whole curve. Different statistic, different null, different verdict
rule — and the README spells this out.

This is a COMPUTED-GOLD arena. Each set is constructed with a KNOWN label, far from
the decision boundary, so the deterministic reference (the R p-curve player) agrees
with the constructed label regardless of tiny numerical differences. That agreement
is the cross-validation guarantee: the reference tool MUST score ~1.0.

  - evidential (evidential_value=true): findings drawn from a real non-null effect
    with decent power (true d≈0.5+, n≈50+) → significant results cluster near p≈0 →
    strong right skew → right_skew_p ≪ .01.
  - no-evidential (evidential_value=false): either TRUE-NULL significant results
    (false positives, p uniform on [0,.05] → flat curve) OR intense P-HACKING (results
    bunched just under .05 → left skew) → right_skew_p ≫ .5.

The gold label is known BY CONSTRUCTION, and the generator ALSO computes the full
p-curve in pure Python; the self-consistency test asserts the Python p-curve verdict
matches the constructed label for every task in both splits. Sets are kept well
separated from the verdict boundary.

Deterministic from (task_set_version, seed) via a sha256 `_seed_int`, mirroring
publication-bias-v1 and grim-consistency-v1. Both splits (revealed/private) run the
IDENTICAL tier matrix and construct the same sequence of labels; only the concrete
numbers differ by seed. Gold is cached in a module-level `_GROUND_TRUTH_CACHE` and
served by `ground_truth(task_id)` — registry-free; the secret is the private seed.
"""
from __future__ import annotations

import hashlib
import math
import random

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

# Closed set of constructed labels, in canonical cycle order. Both splits cover both
# labels (parity by construction).
LABELS = ["evidential", "no-evidential"]

# p-curve significance threshold (only p<.05 results enter the curve) and the
# verdict alpha for the full-curve right-skew test.
SIG_ALPHA = 0.05
SKEW_ALPHA = 0.05


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


# --------------------------------------------------------------------------- #
# Pure-stdlib statistical primitives (no SciPy dependency for the generator).
# --------------------------------------------------------------------------- #
def _norm_cdf(z: float) -> float:
    """Standard-normal CDF Phi(z)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (quantile). Acklam's rational approximation,
    accurate to ~1e-9 over the open interval, which is ample for p-curve."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via continued fraction (Numerical
    Recipes betacf / Lentz's method). Used for the t and F tail probabilities."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    tiny = 1e-30
    for i in range(0, 400):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        cd = c * d
        f *= cd
        if abs(1.0 - cd) < 1e-13:
            break
    return front * (f - 1.0)


def _t_sf(t: float, df: float) -> float:
    """Upper-tail survival of Student's t: P(T > t)."""
    x = df / (df + t * t)
    ib = 0.5 * _betai(df / 2.0, 0.5, x)
    return ib if t > 0 else 1.0 - ib


def _f_sf(f: float, df1: float, df2: float) -> float:
    """Upper-tail survival of the F distribution: P(F > f)."""
    if f <= 0.0:
        return 1.0
    x = df2 / (df2 + df1 * f)
    return _betai(df2 / 2.0, df1 / 2.0, x)


def _chi2_sf_1df(c: float) -> float:
    """Upper-tail survival of chi-square on 1 df: P(X > c) = 2*(1 - Phi(sqrt(c)))."""
    if c <= 0.0:
        return 1.0
    return 2.0 * (1.0 - _norm_cdf(math.sqrt(c)))


# --------------------------------------------------------------------------- #
# Exact two-sided p-value for each supported test statistic.
# p-curve only ever consumes F with df1 == 1.
# --------------------------------------------------------------------------- #
def two_sided_p(finding: dict) -> float:
    kind = finding["type"]
    v = float(finding["value"])
    if kind == "t":
        df = float(finding["df2"] if finding.get("df2") is not None else finding["df1"])
        return 2.0 * _t_sf(abs(v), df)
    if kind == "F":
        df1 = float(finding.get("df1", 1))
        df2 = float(finding["df2"])
        if abs(df1 - 1.0) > 1e-9:
            raise ValueError("p-curve only accepts F with df1 == 1")
        return _f_sf(v, 1.0, df2)
    if kind == "z":
        return 2.0 * (1.0 - _norm_cdf(abs(v)))
    if kind == "chi2":
        return _chi2_sf_1df(v)
    if kind == "r":
        n = float(finding["n"])
        df = n - 2.0
        t = v * math.sqrt(df / max(1.0 - v * v, 1e-12))
        return 2.0 * _t_sf(abs(t), df)
    raise ValueError(f"unknown finding type {kind!r}")


# --------------------------------------------------------------------------- #
# The p-curve (Simonsohn, Nelson & Simons, 2014).
#
# For each SIGNIFICANT finding (p<.05):
#   pp_i  = p_i / .05                         (right-skew pp-value: prob under the
#                                              null of a p this small, given p<.05)
#   z_i   = qnorm(pp_i)                        (z-transform)
# Right-skew (evidential-value) test, Stouffer's method:
#   Z          = sum(z_i) / sqrt(k)
#   right_skew_p = pnorm(Z)
# Evidential value is present when right_skew_p < .05.
#
# Optional flatness (inadequate-evidence) test: compare the observed curve against the
# pp-values expected under 33% power. The pp-value for the flatness test is the
# probability of observing a p at least this LARGE under 33% power, conditional on
# significance; combined the same Stouffer way. Reported but not used for the verdict.
# --------------------------------------------------------------------------- #
def _noncentral_t_crit(df: float) -> float:
    """Two-sided .05 critical |t| value on `df` df (the t such that 2*P(T>t)=.05)."""
    # P(T>t) = .025  ->  upper .975 quantile. Solve via bisection on _t_sf.
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_sf(mid, df) > 0.025:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pcurve(findings: list[dict]) -> dict:
    """Run the full p-curve. Returns a dict with the right-skew p-value, the flatness
    p-value, k (number of significant findings entering the curve) and the verdict."""
    ps = []
    for f in findings:
        p = two_sided_p(f)
        if p < SIG_ALPHA:
            # Guard the open interval for the z-transform.
            ps.append(min(max(p, 1e-12), SIG_ALPHA - 1e-12))

    k = len(ps)
    if k == 0:
        return {"k": 0, "right_skew_p": 1.0, "flatness_p": 1.0, "evidential_value": False}

    # Right-skew test (evidential value).
    z_skew = [_norm_ppf(p / SIG_ALPHA) for p in ps]
    Z_skew = sum(z_skew) / math.sqrt(k)
    right_skew_p = _norm_cdf(Z_skew)

    # Flatness test vs 33% power (optional diagnostic). The pp-value under 33% power
    # uses the noncentral-t cutoff; we approximate the per-study pp via the proportion
    # of the alternative below the observed p. For determinism + simplicity we use the
    # closed-form approximation from Simonsohn et al.'s supplement: pp_flat = (P(p_obs |
    # 33% power) within the significant region). We compute it through the standard
    # "prop" transform on the two-sided p relative to the 33%-power expected p.
    # A flat-or-left curve yields flatness_p that is NOT small; a right-skewed curve
    # yields a small flatness_p. It is reported only.
    z_flat = []
    for p in ps:
        pp_flat = _pp_flat_33(p)
        pp_flat = min(max(pp_flat, 1e-12), 1.0 - 1e-12)
        z_flat.append(_norm_ppf(pp_flat))
    Z_flat = sum(z_flat) / math.sqrt(k)
    flatness_p = _norm_cdf(Z_flat)

    return {
        "k": k,
        "right_skew_p": right_skew_p,
        "flatness_p": flatness_p,
        "evidential_value": right_skew_p < SKEW_ALPHA,
    }


def _pp_flat_33(p: float) -> float:
    """pp-value of a two-sided p under the 33%-power alternative, conditional on
    significance (Simonsohn et al. 2014 flatness test).

    Under H_alt with power=1/3, the test statistic is noncentral. The conditional
    probability that p' <= p given significance is, in the standardized-normal
    approximation p-curve uses, Phi( qnorm(p/2) - ncp ) scaled by the conditioning.
    We use the published approximation: take the z corresponding to the two-sided p,
    shift by the noncentrality giving 33% power at alpha=.05 (ncp = z_{.975} -
    z_{.333} ≈ 1.96 - (-0.43) is the design point), and renormalize over the
    significant region. The result is monotone in p (larger p -> larger pp_flat), so a
    right-skewed curve (small ps) gives small flatness_p and a flat/left curve gives a
    large flatness_p — exactly the intended diagnostic direction."""
    # z for the (one-sided, upper) observed p in the standardized p-curve scale.
    z_obs = _norm_ppf(1.0 - p / 2.0)
    z_crit = 1.959963984540054  # qnorm(.975)
    # Noncentrality giving 33% power at two-sided alpha=.05: power = 1 - Phi(z_crit-ncp)
    # => Phi(z_crit-ncp)=.667 => z_crit-ncp = qnorm(.667) => ncp = z_crit - qnorm(.667).
    ncp = z_crit - _norm_ppf(1.0 / 3.0)
    # Conditional CDF of the alternative below z_obs, given z>z_crit.
    num = _norm_cdf(z_obs - ncp) - _norm_cdf(z_crit - ncp)
    den = 1.0 - _norm_cdf(z_crit - ncp)
    pp = num / den
    return 1.0 - pp  # large observed p (small z_obs) -> large pp_flat


def pcurve_verdict(findings: list[dict]) -> bool:
    return pcurve(findings)["evidential_value"]


# --------------------------------------------------------------------------- #
# Finding-set construction. Each finding is emitted as a test statistic; we vary the
# statistic TYPE across tiers so the arena exercises t / F / z / chi2 / r. Only
# significant (p<.05) findings are kept (p-curve's domain).
# --------------------------------------------------------------------------- #
def _t_from_p_twosided(p: float, df: float) -> float:
    """Inverse of two_sided_p for a t statistic: |t| with P(2*sf)=p."""
    target = p / 2.0  # upper-tail prob
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _draw_evidential(k: int, kind: str, true_d: float, n: int, rng: random.Random):
    """Draw k SIGNIFICANT findings from a real, well-powered effect.

    We simulate a two-group difference: observe a t statistic ~ noncentral t with
    ncp = true_d * sqrt(n/2 per group... ) — but to stay pure-stdlib and deterministic
    we draw an observed Cohen's d ~ Normal(true_d, se_d) and convert to the requested
    statistic, keeping only significant (p<.05) draws. With true_d≈0.5+ and n≈50+ the
    bulk of significant results have small p (p≈.001-.02): a strongly right-skewed
    curve. Rejection-sample until k significant findings are collected."""
    df = 2 * n - 2
    se_d = math.sqrt(2.0 / n)  # SE of d for two equal groups of size n
    out: list[dict] = []
    guard = 0
    while len(out) < k:
        guard += 1
        if guard > 100000:
            raise RuntimeError("could not draw enough significant evidential findings")
        d_obs = rng.gauss(true_d, se_d)
        t = d_obs / se_d  # observed t for the two-group d
        if t <= 0:
            continue
        p = 2.0 * _t_sf(t, df)
        if p >= SIG_ALPHA:
            continue
        out.append(_emit_finding(kind, t, df, n, rng))
    return out


# No-evidential sets must land comfortably on the NOT-right-skewed side of the
# verdict. A finite flat/left draw can, by chance, scatter mostly small p-values and
# produce a spurious right skew near the boundary (the classic computed-gold failure).
# Rather than ship a borderline set we REJECTION-SAMPLE the whole set until its
# right_skew_p on the FINAL rounded findings is comfortably above this floor.
NULL_SKEW_FLOOR = 0.55


def _draw_null(k: int, kind: str, n: int, rng: random.Random):
    """Draw k SIGNIFICANT true-null findings (false positives).

    Under the null the two-sided p of a significant result is UNIFORM on (0,.05).
    We sample p ~ Uniform(eps,.05) and convert to the requested statistic — a FLAT
    p-curve (no right skew). Rejection-sampled so the rendered set's right_skew_p sits
    comfortably above NULL_SKEW_FLOOR (far from the .05 verdict boundary)."""
    df = 2 * n - 2
    for _attempt in range(4000):
        out = []
        for _ in range(k):
            p = rng.uniform(1e-4, SIG_ALPHA - 1e-4)
            t = _t_from_p_twosided(p, df)
            out.append(_emit_finding(kind, t, df, n, rng))
        if pcurve(out)["right_skew_p"] >= NULL_SKEW_FLOOR:
            return out
    raise RuntimeError("could not draw a comfortably-flat null set")


def _draw_phacked(k: int, kind: str, n: int, rng: random.Random):
    """Draw k SIGNIFICANT p-hacked findings: p bunched JUST under .05 (left skew).

    Intense p-hacking stops as soon as p crosses .05, so significant p-values pile up
    just below .05 (p≈.04-.0499) — a LEFT-skewed curve, the opposite of evidential
    value. We draw p ~ Uniform(.04,.0499). Left skew already gives right_skew_p≈1, but
    we rejection-sample for the same comfortable margin."""
    df = 2 * n - 2
    for _attempt in range(4000):
        out = []
        for _ in range(k):
            p = rng.uniform(0.040, 0.0499)
            t = _t_from_p_twosided(p, df)
            out.append(_emit_finding(kind, t, df, n, rng))
        if pcurve(out)["right_skew_p"] >= NULL_SKEW_FLOOR:
            return out
    raise RuntimeError("could not draw a comfortably-left-skewed p-hacked set")


def _emit_finding(kind: str, t: float, df: float, n: int, rng: random.Random) -> dict:
    """Render an observed t (on `df` df) as the requested statistic type, preserving
    the exact two-sided p-value so the p-curve is invariant to the chosen encoding."""
    if kind == "t":
        return {"type": "t", "value": round(t, 4), "df1": int(df)}
    if kind == "F":
        # F(1, df) = t^2, same two-sided p.
        return {"type": "F", "value": round(t * t, 4), "df1": 1, "df2": int(df)}
    if kind == "z":
        # Match the two-sided p exactly: z = qnorm(1 - p/2).
        p = 2.0 * _t_sf(t, df)
        z = _norm_ppf(1.0 - p / 2.0)
        return {"type": "z", "value": round(z, 4)}
    if kind == "chi2":
        # chi2(1) with the same two-sided p: c = z^2 where z matches p.
        p = 2.0 * _t_sf(t, df)
        z = _norm_ppf(1.0 - p / 2.0)
        return {"type": "chi2", "value": round(z * z, 4), "df1": 1}
    if kind == "r":
        # r from t on df = n-2: r = t / sqrt(t^2 + df), with n = df + 2.
        nn = int(df) + 2
        r = t / math.sqrt(t * t + df)
        return {"type": "r", "value": round(r, 4), "n": nn}
    raise ValueError(f"unknown emit kind {kind!r}")


def _assemble(task_id, tier, label, params, rng, split, visibility) -> tuple[dict, dict]:
    k = params["k"]
    kind = params["kind"]
    n = params["n"]

    if label == "evidential":
        findings = _draw_evidential(k, kind, params["true_d"], n, rng)
    elif label == "no-evidential":
        mech = params["mechanism"]  # "null" or "phacked"
        if mech == "null":
            findings = _draw_null(k, kind, n, rng)
        elif mech == "phacked":
            findings = _draw_phacked(k, kind, n, rng)
        else:  # pragma: no cover
            raise ValueError(f"unknown no-evidential mechanism {mech!r}")
    else:  # pragma: no cover - guarded by LABELS
        raise ValueError(f"unknown label {label!r}")

    res = pcurve(findings)
    evidential_value = label == "evidential"

    envelope = {
        "task_id": task_id,
        "arena_id": "p-curve-v1",
        "task_set_version": "v1",
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "k": k, "label": label},
        "input": {"findings": findings},
    }
    gold = {
        "evidential_value": evidential_value,
        "label": label,
        "right_skew_p": res["right_skew_p"],
        "flatness_p": res["flatness_p"],
        "computed_verdict": res["evidential_value"],
        "k": res["k"],
    }
    return envelope, gold


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    visibility = "public" if split == "revealed" else "held_out"

    def emit(tier, idx, label, params):
        tid = f"pcurve-t{tier}-{idx}-s{seed}"
        rng = random.Random(_seed_int(task_set_version, seed, tier, idx, label, params.get("mechanism", "")))
        env, gt = _assemble(tid, tier, label, params, rng, split, visibility)
        _GROUND_TRUTH_CACHE[tid] = gt
        return env

    # Strong, far-from-boundary presets per tier. Evidential sets use a real effect
    # with decent power (true_d>=0.5, n>=50) -> right_skew_p << .01. No-evidential sets
    # are either TRUE-NULL (flat) or P-HACKED (left skew) -> right_skew_p >> .5. We vary
    # the statistic TYPE across tiers so t / F / z / chi2 / r are all exercised.
    #
    # T1 small-k (k=5) baseline: evidential (t) vs true-null (t).
    yield emit(1, 0, "evidential",    {"k": 5,  "kind": "t",    "true_d": 0.6, "n": 60})
    yield emit(1, 1, "no-evidential", {"k": 5,  "kind": "t",    "true_d": 0.0, "n": 60, "mechanism": "null"})

    # T2 controls: an unmistakably-evidential set vs an unmistakably-FLAT (null) set,
    # k=12, F-statistics (df1=1) for the evidential, z for the null.
    yield emit(2, 0, "evidential",    {"k": 12, "kind": "F",    "true_d": 0.7, "n": 80})
    yield emit(2, 1, "no-evidential", {"k": 12, "kind": "z",    "true_d": 0.0, "n": 80, "mechanism": "null"})

    # T3 p-hacking mechanism (the hard no-evidential): evidential (z) vs P-HACKED (z),
    # k=15. Left-skewed hacked curve must NOT be read as evidential.
    yield emit(3, 0, "evidential",    {"k": 15, "kind": "z",    "true_d": 0.5, "n": 70})
    yield emit(3, 1, "no-evidential", {"k": 15, "kind": "z",    "true_d": 0.0, "n": 70, "mechanism": "phacked"})

    # T4 chi-square(1) findings, k=20: evidential vs true-null.
    yield emit(4, 0, "evidential",    {"k": 20, "kind": "chi2", "true_d": 0.5, "n": 65})
    yield emit(4, 1, "no-evidential", {"k": 20, "kind": "chi2", "true_d": 0.0, "n": 65, "mechanism": "null"})

    # T5 correlations r (n supplied), k=25: evidential vs P-HACKED.
    yield emit(5, 0, "evidential",    {"k": 25, "kind": "r",    "true_d": 0.55, "n": 90})
    yield emit(5, 1, "no-evidential", {"k": 25, "kind": "r",    "true_d": 0.0,  "n": 90, "mechanism": "phacked"})

    # T6 largest k=30, mixed-strength real effect (t): evidential vs true-null.
    yield emit(6, 0, "evidential",    {"k": 30, "kind": "t",    "true_d": 0.5, "n": 100})
    yield emit(6, 1, "no-evidential", {"k": 30, "kind": "t",    "true_d": 0.0, "n": 100, "mechanism": "null"})


def ground_truth(task_id: str) -> dict:
    """Return gold for a task, regenerated from seed via the in-process cache.

    The runner always calls generate() before ground_truth(); this arena needs no
    external gold registry because the secret is the private seed, not a stored
    answer key.
    """
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(
            f"No cached gold for {task_id!r}; call generate() for the matching "
            "split/seed before ground_truth()."
        )
    return _GROUND_TRUTH_CACHE[task_id]
