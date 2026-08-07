"""Generator for publication-bias-v1 (computed-gold meta-analysis arena).

Scientific task: given a meta-analytic dataset (a set of studies, each with an
effect-size estimate `yi` and its standard error `sei`), decide whether the set
shows small-study / publication bias — i.e. funnel-plot asymmetry, the hallmark of
selective reporting where small (high-`sei`) studies are over-represented among
significant, large-effect results.

This is a COMPUTED-GOLD arena. Each dataset is constructed with a KNOWN label, and
the construction is deliberately built FAR from the decision boundary so the
deterministic reference tool (metafor: Egger's regression test + trim-and-fill)
agrees with the constructed label regardless of tiny numerical differences. That
agreement is the cross-validation guarantee: the reference tool MUST score ~1.0.

  - clean (bias_detected=false): k studies drawn around a common true effect mu with
    symmetric sampling error and NO censoring. The funnel is symmetric, so Egger's
    p-value sits well above any sensible threshold.
  - biased (bias_detected=true): a STRONG small-study effect is induced by adding an
    explicit positive coupling between yi and sei (large/high-precision studies show
    a smaller effect; small/low-precision studies show an inflated effect) AND by
    censoring small-N non-significant studies. The asymmetry is strong (Egger
    p << 0.01).

The gold label is known BY CONSTRUCTION (we know which datasets we biased), but the
generator ALSO computes Egger's test in pure Python (the standard weighted
regression of yi on sei) and the self-consistency test asserts the Python Egger
verdict matches the constructed label for every task. Datasets are kept far from the
p ~ 0.05-0.10 grey zone.

Deterministic from (task_set_version, seed) via a sha256 `_seed_int`, mirroring
grim-consistency-v1. Both splits (revealed/private) run the IDENTICAL tier matrix and
construct the same sequence of clean/biased labels; only the concrete numbers differ
by seed. Gold is cached in a module-level `_GROUND_TRUTH_CACHE` and served by
`ground_truth(task_id)` — registry-free; the secret is the private seed.
"""
from __future__ import annotations

import hashlib
import math
import random

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

# The closed set of constructed labels, in canonical cycle order. Both splits cover
# both labels (parity by construction).
LABELS = ["unbiased", "biased"]

# Egger decision threshold used to derive the in-generator verdict (matches the
# metafor adapter's `bias_detected = egger_p < EGGER_ALPHA`).
EGGER_ALPHA = 0.10


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


# --------------------------------------------------------------------------- #
# Egger's regression test (the same check the gold is cross-validated against).
#
# Standard Egger test, weighted-regression / metafor `regtest(model="lm")` form:
# fit  yi ~ b0 + b1 * sei  by weighted least squares with weights 1/vi (vi=sei^2),
# and test H0: b1 = 0 (the small-study-effect slope). A significant slope == funnel
# asymmetry == small-study/publication bias. The p-value is two-sided from a
# t-distribution on k-2 degrees of freedom.
# --------------------------------------------------------------------------- #
def _t_sf(t: float, df: int) -> float:
    """Upper-tail survival function of Student's t (P(T > t)) via the regularized
    incomplete beta function. Pure-stdlib so the generator has no SciPy dependency.
    """
    x = df / (df + t * t)
    ib = _betai(df / 2.0, 0.5, x)
    p = 0.5 * ib
    return p if t > 0 else 1.0 - p


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via continued fraction (Numerical
    Recipes betacf)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    # Continued fraction (Lentz's method).
    f, c, d = 1.0, 1.0, 0.0
    tiny = 1e-30
    for i in range(0, 300):
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
        if abs(1.0 - cd) < 1e-12:
            break
    return front * (f - 1.0)


def egger_p(yi: list[float], sei: list[float]) -> float:
    """Two-sided p-value of the Egger small-study-effect slope (weighted lm of yi on
    sei with weights 1/sei^2)."""
    k = len(yi)
    w = [1.0 / (s * s) for s in sei]
    x = list(sei)
    y = list(yi)
    sw = sum(w)
    xbar = sum(wi * xi for wi, xi in zip(w, x)) / sw
    ybar = sum(wi * yj for wi, yj in zip(w, y)) / sw
    sxx = sum(wi * (xi - xbar) ** 2 for wi, xi in zip(w, x))
    sxy = sum(wi * (xi - xbar) * (yj - ybar) for wi, xi, yj in zip(w, x, y))
    b1 = sxy / sxx
    b0 = ybar - b1 * xbar
    resid = [yj - (b0 + b1 * xi) for xi, yj in zip(x, y)]
    rss = sum(wi * r * r for wi, r in zip(w, resid))
    df = k - 2
    sigma2 = rss / df
    se_b1 = math.sqrt(sigma2 / sxx)
    t = b1 / se_b1
    return 2.0 * _t_sf(abs(t), df)


def egger_verdict(yi: list[float], sei: list[float]) -> bool:
    return egger_p(yi, sei) < EGGER_ALPHA


# --------------------------------------------------------------------------- #
# Dataset construction.
# --------------------------------------------------------------------------- #
def _round_studies(yi, sei, decimals=4):
    return [{"yi": round(y, decimals), "sei": round(s, decimals)} for y, s in zip(yi, sei)]


def _draw_sei(k: int, sei_lo: float, sei_hi: float, rng: random.Random) -> list[float]:
    """Draw k standard errors spanning a wide precision range (so a funnel has a
    visible spread of study sizes). Sorted large->small is irrelevant; we just need a
    healthy spread between sei_lo (precise/large studies) and sei_hi (imprecise/small)."""
    return [rng.uniform(sei_lo, sei_hi) for _ in range(k)]


def _make_unbiased(k: int, mu: float, tau: float, sei_lo: float, sei_hi: float,
                   rng: random.Random) -> tuple[list[float], list[float]]:
    """Symmetric funnel: each study's yi = mu + between-study(tau) + sampling(sei).

    No coupling between yi and sei and no censoring => Egger slope ~ 0. tau injects
    between-study heterogeneity (a real meta-analysis trait) that must NOT be mistaken
    for asymmetry.

    Pure random draws can, by chance, produce a spurious yi~sei correlation that lands
    a clean set near the Egger boundary (the classic computed-gold failure). Rather
    than algebraically zeroing the slope (which leaves a degenerate fit that rounding
    re-perturbs into spurious significance), we REJECTION-SAMPLE: redraw the whole set
    until its Egger p on the FINAL rounded studies is comfortably above
    `clean_p_floor`. The accepted set is genuinely symmetric with realistic scatter,
    and because it is far from the boundary, rounding cannot flip the verdict. The loop
    converges quickly (a random symmetric funnel is asymmetric only by chance); the
    bound guards against an infinite loop on pathological parameters.
    """
    clean_p_floor = 0.25
    for _attempt in range(2000):
        sei = _draw_sei(k, sei_lo, sei_hi, rng)
        yi = []
        for s in sei:
            theta = mu + rng.gauss(0.0, tau)        # true effect for this study
            yi.append(theta + rng.gauss(0.0, s))    # observed = true + sampling error
        rounded = _round_studies(yi, sei)
        fy = [r["yi"] for r in rounded]
        fs = [r["sei"] for r in rounded]
        if egger_p(fy, fs) >= clean_p_floor:
            return yi, sei
    # Extremely unlikely; surface rather than ship a near-boundary clean set.
    raise RuntimeError("could not draw a comfortably-symmetric clean set")


def _make_biased(k: int, mu: float, tau: float, sei_lo: float, sei_hi: float,
                 slope: float, rng: random.Random) -> tuple[list[float], list[float]]:
    """Strong small-study effect: add an explicit positive coupling `slope` between
    yi and sei (small/imprecise studies report inflated effects), then CENSOR the
    most extreme small-study under-shooters so the funnel is markedly asymmetric.

    Rejection-sampled to guarantee Egger p is comfortably BELOW `biased_p_ceiling`
    (well under EGGER_ALPHA) on the FINAL rounded studies, so the construction never
    ships a borderline-biased set and rounding cannot flip the verdict.
    """
    biased_p_ceiling = 0.02
    for _attempt in range(2000):
        # Over-draw so that after censoring we still have k studies.
        over = k + max(6, k // 2)
        sei = _draw_sei(over, sei_lo, sei_hi, rng)
        yi = []
        for s in sei:
            theta = mu + rng.gauss(0.0, tau)
            # The coupling term scales with sei: imprecise studies are pushed up.
            yi.append(theta + slope * s + rng.gauss(0.0, s))
        # Censor the small-study (high-sei) results that fall BELOW the pooled-ish
        # centre: this is the selective-reporting mechanism (non-significant small
        # studies vanish), and it is exactly what makes the surviving funnel asymmetric.
        centre = mu + slope * (0.5 * (sei_lo + sei_hi))
        pairs = list(zip(yi, sei))
        # Rank candidates for censoring: high sei AND low yi are the prime omissions.
        censor_score = lambda p: (p[1] / sei_hi) - (p[0] - centre)
        pairs.sort(key=censor_score, reverse=True)
        n_drop = len(pairs) - k
        survivors = pairs[n_drop:]
        rng.shuffle(survivors)
        out_yi = [p[0] for p in survivors]
        out_sei = [p[1] for p in survivors]
        rounded = _round_studies(out_yi, out_sei)
        fy = [r["yi"] for r in rounded]
        fs = [r["sei"] for r in rounded]
        if egger_p(fy, fs) <= biased_p_ceiling:
            return out_yi, out_sei
    # Extremely unlikely; surface rather than ship a near-boundary biased set.
    raise RuntimeError("could not construct a strongly-asymmetric biased set")


def _assemble(task_id, tier, label, k, params, rng, split, visibility) -> tuple[dict, dict]:
    mu, tau, sei_lo, sei_hi, slope = (
        params["mu"], params["tau"], params["sei_lo"], params["sei_hi"], params["slope"],
    )
    if label == "unbiased":
        yi, sei = _make_unbiased(k, mu, tau, sei_lo, sei_hi, rng)
    elif label == "biased":
        yi, sei = _make_biased(k, mu, tau, sei_lo, sei_hi, slope, rng)
    else:  # pragma: no cover - guarded by LABELS
        raise ValueError(f"unknown label {label!r}")

    studies = _round_studies(yi, sei)
    # Compute the in-generator Egger verdict on the FINAL (rounded) studies so it
    # exactly matches what the player tools see.
    fy = [s["yi"] for s in studies]
    fs = [s["sei"] for s in studies]
    p_egger = egger_p(fy, fs)
    bias_detected = label == "biased"

    envelope = {
        "task_id": task_id,
        "arena_id": "publication-bias-v1",
        "task_set_version": "v1",
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "k": len(studies), "label": label},
        "input": {
            "k": len(studies),
            "studies": studies,
        },
    }
    gold = {
        "bias_detected": bias_detected,
        "label": label,
        "egger_p": p_egger,
        "egger_verdict": p_egger < EGGER_ALPHA,
        "k": len(studies),
    }
    return envelope, gold


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    visibility = "public" if split == "revealed" else "held_out"

    def emit(tier, idx, label, k, params):
        tid = f"pubbias-t{tier}-{idx}-s{seed}"
        rng = random.Random(_seed_int(task_set_version, seed, tier, idx, label))
        env, gt = _assemble(tid, tier, label, k, params, rng, split, visibility)
        _GROUND_TRUTH_CACHE[tid] = gt
        return env

    # Strong, far-from-boundary parameter presets per tier. `slope` is the small-study
    # coupling for biased datasets (large => unmistakable asymmetry); unbiased datasets
    # ignore it. sei spread is wide so the funnel has a real precision range.
    #
    # T1 baseline clean vs strong-biased (small k=10).
    yield emit(1, 0, "unbiased", 10, {"mu": 0.30, "tau": 0.00, "sei_lo": 0.05, "sei_hi": 0.40, "slope": 0.0})
    yield emit(1, 1, "biased",   10, {"mu": 0.30, "tau": 0.00, "sei_lo": 0.05, "sei_hi": 0.40, "slope": 2.5})

    # T2 controls / false-alarm trap: clean but NOISY — high between-study tau and a
    # wide precision range. Heterogeneity must NOT be read as asymmetry. (Two clean
    # datasets, no biased here.)
    yield emit(2, 0, "unbiased", 16, {"mu": 0.00, "tau": 0.25, "sei_lo": 0.06, "sei_hi": 0.45, "slope": 0.0})
    yield emit(2, 1, "unbiased", 24, {"mu": 0.50, "tau": 0.30, "sei_lo": 0.05, "sei_hi": 0.50, "slope": 0.0})

    # T3 larger k, null true effect — clean vs biased with k=20.
    yield emit(3, 0, "unbiased", 20, {"mu": 0.00, "tau": 0.05, "sei_lo": 0.05, "sei_hi": 0.40, "slope": 0.0})
    yield emit(3, 1, "biased",   20, {"mu": 0.00, "tau": 0.05, "sei_lo": 0.05, "sei_hi": 0.40, "slope": 2.0})

    # T4 heterogeneous biased vs heterogeneous clean (tau present in BOTH) — the test
    # must separate asymmetry from heterogeneity at k=24.
    yield emit(4, 0, "unbiased", 24, {"mu": 0.40, "tau": 0.20, "sei_lo": 0.05, "sei_hi": 0.45, "slope": 0.0})
    yield emit(4, 1, "biased",   24, {"mu": 0.40, "tau": 0.20, "sei_lo": 0.05, "sei_hi": 0.45, "slope": 2.2})

    # T5 large k=32, moderate effect — clean vs biased.
    yield emit(5, 0, "unbiased", 32, {"mu": 0.25, "tau": 0.10, "sei_lo": 0.04, "sei_hi": 0.42, "slope": 0.0})
    yield emit(5, 1, "biased",   32, {"mu": 0.25, "tau": 0.10, "sei_lo": 0.04, "sei_hi": 0.42, "slope": 1.8})

    # T6 largest k=40, strong-biased + a clean control at the same k.
    yield emit(6, 0, "unbiased", 40, {"mu": 0.20, "tau": 0.08, "sei_lo": 0.04, "sei_hi": 0.40, "slope": 0.0})
    yield emit(6, 1, "biased",   40, {"mu": 0.20, "tau": 0.08, "sei_lo": 0.04, "sei_hi": 0.40, "slope": 1.8})


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
