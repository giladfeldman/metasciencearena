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

THAT PARAGRAPH DESCRIBES TASK SET v1. Task set v2 (`_generate_v2`, below) scores
statistics rather than a verdict, and its gold is NOT regenerated from the seed: it
is computed in R by tools/build_gold_v2.py and read from a gold file, bound to each
dataset by sha256.
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
    # v2 is a different task (statistics, not a verdict) with its own datasets and
    # its own task_id prefix. It branches BEFORE any v1 rng call, so v1 generation
    # is byte-identical to what its fingerprint and published records were made from.
    if task_set_version == "v2":
        yield from _generate_v2(task_set_version, seed, split)
        return
    if task_set_version != "v1":
        # A new version needs its own branch; silently minting v2 datasets under a
        # v3 label is the failure this guards (review 2026-10-07).
        raise ValueError(f"publication-bias-v1 has no generator for task set {task_set_version!r}")
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


# --------------------------------------------------------------------------- #
# Task set v2 — score the STATISTICS, not a verdict (owner decision 2026-09-17).
#
# Gilad: "the verdict of how to translate these tests to a whether there is
# publication bias or not is a complex statistics issue I personally am not aware
# of a resolution to, so we should focus on whether they get the stats right."
#
# So v2 has no label to be far from. Datasets are drawn to span the range each
# method actually meets — no selection, p-value-based selection of the kind 3PSM
# and RoBMA model, and a small-study coupling of the kind Egger and PET model — at
# moderate strength, so the statistics land across their range instead of pinned at
# p = 0 (v1's biased sets had Egger p down to 1e-13, which makes a p-value a
# constant).
#
# The gold is NOT computed here. Trim-and-fill, the selection model and RoBMA exist
# only in R, so tools/build_gold_v2.py runs tools/pubbias_v2_stats.R over every
# dataset once and writes the result to a gold file; ground_truth() serves it after
# checking that the dataset it was computed on is byte-for-byte the one generated
# now (input_sha256). A stale gold file therefore fails loudly instead of scoring
# players against another dataset's answers.
# --------------------------------------------------------------------------- #
V2_TASK_PREFIX = "pubbias2"

# task_id -> why its gold is unavailable (the gold file was absent at generate time).
_V2_GOLD_MISSING: dict[str, str] = {}

# (tier, idx, construction, k, params). `construction` is a difficulty descriptor,
# never scored. `w` = probability a non-significant study is published.
V2_MATRIX = [
    (1, 0, "none",        10, {"mu": 0.30, "tau": 0.00, "sei_lo": 0.08, "sei_hi": 0.40}),
    (1, 1, "selection",   10, {"mu": 0.20, "tau": 0.05, "sei_lo": 0.08, "sei_hi": 0.40, "w": 0.30}),
    (2, 0, "none",        16, {"mu": 0.40, "tau": 0.25, "sei_lo": 0.06, "sei_hi": 0.45}),
    (2, 1, "selection",   16, {"mu": 0.25, "tau": 0.20, "sei_lo": 0.06, "sei_hi": 0.45, "w": 0.40}),
    (3, 0, "none",        20, {"mu": 0.00, "tau": 0.05, "sei_lo": 0.05, "sei_hi": 0.40}),
    (3, 1, "selection",   20, {"mu": 0.00, "tau": 0.05, "sei_lo": 0.05, "sei_hi": 0.40, "w": 0.20}),
    (4, 0, "none",        24, {"mu": 0.40, "tau": 0.20, "sei_lo": 0.05, "sei_hi": 0.45}),
    (4, 1, "small_study", 24, {"mu": 0.30, "tau": 0.15, "sei_lo": 0.05, "sei_hi": 0.45, "slope": 1.0}),
    (5, 0, "selection",   32, {"mu": 0.20, "tau": 0.10, "sei_lo": 0.04, "sei_hi": 0.42, "w": 0.50}),
    (5, 1, "small_study", 32, {"mu": 0.15, "tau": 0.10, "sei_lo": 0.04, "sei_hi": 0.42, "slope": 0.8}),
    (6, 0, "none",        40, {"mu": 0.20, "tau": 0.08, "sei_lo": 0.04, "sei_hi": 0.40}),
    (6, 1, "selection",   40, {"mu": 0.10, "tau": 0.08, "sei_lo": 0.04, "sei_hi": 0.40, "w": 0.10}),
]

V2_CONSTRUCTIONS = ["none", "selection", "small_study"]

# Every dataset must carry at least this many studies on EACH side of one-sided
# p = .025, or the three-parameter selection model has an empty interval and its
# weight is not identified. Enforced by redrawing, deterministically.
V2_MIN_PER_INTERVAL = 3


def _v2_split_tag(split: str, seed: int) -> str:
    """The split discriminator in a v2 task_id. For the private split it carries
    NOTHING derived from the seed.

    v1 wrote the raw seed into every task_id (`-s<seed>`), so a private record or
    log line published the secret. stats-extraction v2 replaced it with the first 8
    hex digits of sha256(seed) -- but the seeds are 10-digit integers, about 9e9
    candidates, so that tag is inverted by hashing every candidate (seconds on a
    GPU). Any deterministic function of a small secret is. A constant is not.

    Rotating the private seed inside v2 would therefore reuse task_ids for new
    datasets; don't -- rotate by minting a new task-set version. The gold file's
    input_sha256 binding refuses such a mix-up even if it happens.
    """
    if split == "revealed":
        return f"s{seed}"
    return "private"


def _one_sided_p(yi: float, sei: float) -> float:
    """One-sided p of z = yi/sei against H0: effect <= 0 (the p 3PSM and RoBMA select on)."""
    return 0.5 * math.erfc((yi / sei) / math.sqrt(2.0))


def _draw_v2(construction: str, k: int, p: dict, rng: random.Random) -> list[dict]:
    for _attempt in range(5000):
        yi: list[float] = []
        sei: list[float] = []
        while len(yi) < k:
            s = rng.uniform(p["sei_lo"], p["sei_hi"])
            theta = p["mu"] + rng.gauss(0.0, p["tau"])
            if construction == "small_study":
                theta += p["slope"] * s
            y = theta + rng.gauss(0.0, s)
            if construction == "selection" and _one_sided_p(y, s) >= 0.025:
                if rng.random() >= p["w"]:
                    continue  # a non-significant study that was never published
            yi.append(y)
            sei.append(s)
        studies = _round_studies(yi, sei)
        n_sig = sum(1 for st in studies if _one_sided_p(st["yi"], st["sei"]) < 0.025)
        # No tied sei (hence no tied vi) and no tied yi after rounding: Begg's test
        # is scored on its EXACT p-value, which does not exist with ties (R's
        # cor.test falls back to a normal approximation). Found by the Sonnet
        # consult 2026-09-30: 1 of the 12 revealed sets had a tied vi.
        no_ties = (len({st["sei"] for st in studies}) == k
                   and len({st["yi"] for st in studies}) == k)
        if n_sig >= V2_MIN_PER_INTERVAL and (k - n_sig) >= V2_MIN_PER_INTERVAL and no_ties:
            return studies
    raise RuntimeError(f"could not draw a v2 {construction} set with both p-intervals populated")


def v2_input_sha256(studies: list[dict]) -> str:
    """Hash of exactly what the player sees, used to bind gold to its dataset."""
    canon = "|".join(f"{s['yi']!r},{s['sei']!r}" for s in studies)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _v2_gold_path(task_set_version: str, split: str):
    from pathlib import Path
    base = Path(__file__).resolve().parent / "task_sets" / task_set_version
    # The private split's gold lives under the gitignored _held_out/ directory.
    return base / ("_r_gold.json" if split == "revealed" else "_held_out/_r_gold.json")


def _load_v2_gold(task_set_version: str, split: str) -> dict | None:
    import json
    path = _v2_gold_path(task_set_version, split)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _generate_v2(task_set_version: str, seed: int, split: str = "revealed", *, require_gold: bool = True):
    visibility = "public" if split == "revealed" else "held_out"
    gold_file = _load_v2_gold(task_set_version, split) if require_gold else None
    if require_gold and gold_file is None:
        # Envelopes are still generated -- parity, fingerprints and a CI machine
        # without the private gold need only the datasets -- but no gold is cached,
        # so ground_truth() raises for every task and nothing can be SCORED.
        missing = (
            f"No gold for publication-bias-v1 {task_set_version} split={split}: "
            f"{_v2_gold_path(task_set_version, split)} is missing. Build it with "
            "arenas/publication-bias-v1/tools/build_gold_v2.py (needs R with metafor "
            "and RoBMA/JAGS)."
        )
    for tier, idx, construction, k, params in V2_MATRIX:
        tid = f"{V2_TASK_PREFIX}-t{tier}-{idx}-{_v2_split_tag(split, seed)}"
        rng = random.Random(_seed_int(task_set_version, seed, tier, idx, construction))
        studies = _draw_v2(construction, k, params, rng)
        sha = v2_input_sha256(studies)
        envelope = {
            "task_id": tid,
            "arena_id": "publication-bias-v1",
            "task_set_version": task_set_version,
            "split": split,
            "visibility": visibility,
            "difficulty": {"tier": tier, "k": k, "construction": construction},
            "input": {"k": k, "studies": studies},
        }
        if require_gold and gold_file is None:
            _GROUND_TRUTH_CACHE.pop(tid, None)
            _V2_GOLD_MISSING[tid] = missing
        elif require_gold:
            _V2_GOLD_MISSING.pop(tid, None)
            entry = gold_file.get(tid)
            if entry is None:
                raise KeyError(f"{tid} has no entry in {_v2_gold_path(task_set_version, split)}")
            if entry.get("input_sha256") != sha:
                raise ValueError(
                    f"{tid}: gold was computed on a different dataset "
                    f"(gold input_sha256 {entry.get('input_sha256')!r} != generated {sha!r}). "
                    "Rebuild the gold; never score against it."
                )
            _GROUND_TRUTH_CACHE[tid] = {
                "gold_stats": entry["gold_stats"],
                "construction": construction,
                "k": k,
                "input_sha256": sha,
            }
        yield envelope


def ground_truth(task_id: str) -> dict:
    """Return gold for a task from the in-process cache filled by generate().

    v1: regenerated from the seed; no stored answer key. v2: read from the R-built
    gold file by `_generate_v2`, which refuses a dataset-hash mismatch; if that file
    was absent at generate time this raises FileNotFoundError naming the builder.
    """
    if task_id in _V2_GOLD_MISSING:
        raise FileNotFoundError(_V2_GOLD_MISSING[task_id])
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(
            f"No cached gold for {task_id!r}; call generate() for the matching "
            "split/seed before ground_truth()."
        )
    return _GROUND_TRUTH_CACHE[task_id]
