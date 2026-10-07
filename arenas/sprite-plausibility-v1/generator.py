"""Generator for sprite-plausibility-v1 (SPRITE-style summary-stat arena).

Builds tables of reported summary statistics (mean, SD, N on an integer response
scale) with KNOWN granularity/range plausibility defects injected per statistic,
mirroring the rsprite2 / SPRITE family of checks. All-procedural and deterministic
from (task_set_version, seed).

The arena KNOWS the ground truth because it COMPUTES it with rigorous *sufficient*
conditions — it never guesses:

  - CLEAN: draw N integers uniformly in [scale_min, scale_max], compute the real
    sample mean and sample SD (ddof=1), round both to `decimals`. This pair is
    achievable BY CONSTRUCTION, so it is provably plausible.
  - impossible_mean: place the reported mean strictly OUTSIDE [scale_min, scale_max]
    (a margin beyond a bound). No sample of in-range integers can have a mean outside
    its own range, so this is provably impossible.
  - impossible_sd: keep a valid in-range mean but set the SD so that sd^2 strictly
    exceeds the theoretical maximum variance for bounded data with that mean,
    max_var = (scale_max - mean) * (mean - scale_min). Reporting sd = sqrt(max_var*k)
    with k > 1 (rounded UP so the rounded sd^2 still strictly exceeds max_var) is
    provably impossible.

A clean stat NEVER trips either condition (asserted in tests). The exact NUMBERS
are seed-driven, so revealed and private content differ; the KIND assignment is
index-driven (cycling through ISSUE_KINDS, seed-independent), so both splits cover
every kind => framework/parity.py passes.

Gold is regenerated from the seed and served from the in-process cache (no external
registry): the secret is the private seed, not a stored answer key. The revealed
seed is committed in arena.yaml#benchmark_splits.
"""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import yaml

ARENA_DIR = Path(__file__).resolve().parent
CATALOGS_DIR = ARENA_DIR / "catalogs"

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

# The closed set of injected impossibility kinds, in canonical cycle order.
# Index-driven cycling through THIS list (not rng.choice) is what guarantees both
# splits cover every issue kind => parity passes.
ISSUE_KINDS = [
    "impossible_mean",
    "impossible_sd",
]

# Margin (in scale units) by which an impossible_mean is pushed outside a bound.
_MEAN_MARGIN = 0.5
# Variance-overshoot factor for a blatant impossible_sd (T3/T5/T6).
_SD_OVERSHOOT = 1.3
# Smaller overshoot for a SUBTLE impossible_sd (T4) — only barely over the bound.
_SD_OVERSHOOT_SUBTLE = 1.05


def _load_catalog() -> list[dict]:
    with (CATALOGS_DIR / "scales.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _round_half_up(x: float, decimals: int) -> float:
    """Round half away from zero to `decimals` places (deterministic, no banker's)."""
    factor = 10 ** decimals
    return math.floor(x * factor + 0.5) / factor if x >= 0 else -math.floor(-x * factor + 0.5) / factor


def _round_up(x: float, decimals: int) -> float:
    """Ceil to `decimals` places — used so a reported SD strictly overshoots the bound."""
    factor = 10 ** decimals
    return math.ceil(x * factor) / factor


def _rounds_unambiguously(x: float, decimals: int, tol: float = 0.04) -> bool:
    """True iff x is clear of a .5 rounding half-boundary at `decimals` precision.

    A clean/control mean drawn from a real integer sample is achievable, but if its
    true value lands exactly on a half-boundary (e.g. 4.625 at 2 dp) different
    rounding conventions disagree (half-up -> 4.63, half-even/tolerance-band -> 4.62)
    and the reference tool (rsprite2's internal GRIM) may then judge the reported,
    rounded value unachievable — a false positive on a genuinely-clean control.
    Requiring an unambiguous rounding makes the reported value achievable under ANY
    convention, so the gold and the tool agree.
    """
    scaled = abs(x) * (10 ** decimals)
    frac = scaled - math.floor(scaled)
    return abs(frac - 0.5) > tol


def _sample_mean_sd(values: list[int]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def _max_var(mean: float, scale_min: int, scale_max: int) -> float:
    """Theoretical maximum POPULATION variance for bounded data with this mean."""
    return (scale_max - mean) * (mean - scale_min)


def _max_sample_var(mean: float, scale_min: int, scale_max: int, n: int) -> float:
    """Maximum SAMPLE variance (ddof=1) attainable for n bounded integers with this
    mean. The sample variance carries an n/(n-1) Bessel inflation over the population
    bound, so a real two-point split can sit slightly above _max_var. Using THIS
    ceiling for the impossible_sd injection guarantees no achievable sample (of any
    composition) can ever reach the reported SD."""
    if n < 2:
        return _max_var(mean, scale_min, scale_max)
    return _max_var(mean, scale_min, scale_max) * n / (n - 1)


def _clean_stat(entry: dict, rng: random.Random) -> tuple[float, float]:
    """Draw N integers in range; return achievable (mean, sd) rounded to decimals.

    Redraws until both the mean and SD round unambiguously (clear of a .5 boundary),
    so the reported values are achievable under any rounding convention and the
    reference tool never false-flags a clean control.
    """
    lo, hi, n, dec = entry["scale_min"], entry["scale_max"], entry["typical_n"], entry["decimals"]
    mean = sd = 0.0
    for _ in range(60):
        values = [rng.randint(lo, hi) for _ in range(n)]
        mean, sd = _sample_mean_sd(values)
        if _rounds_unambiguously(mean, dec) and _rounds_unambiguously(sd, dec):
            break
    return _round_half_up(mean, dec), _round_half_up(sd, dec)


def _extreme_possible_stat(entry: dict, rng: random.Random) -> tuple[float, float]:
    """Build an extreme-but-ACHIEVABLE (mean, sd): a genuine two-point-ish sample
    pushed to the edges of what the range allows. Used for T2 controls.

    By constructing the sample explicitly we guarantee the pair is achievable; we
    only round to `decimals` for reporting, then we re-verify in tests that it never
    trips an impossibility condition.
    """
    lo, hi, n, dec = entry["scale_min"], entry["scale_max"], entry["typical_n"], entry["decimals"]
    flavor = rng.choice(["max_sd", "near_bound_mean"])
    mean = sd = 0.0
    # Vary the split each attempt until the reported (mean, sd) round unambiguously,
    # so the (genuinely achievable) extreme control is never a rounding-boundary case
    # the reference tool would dispute.
    for attempt in range(60):
        if flavor == "max_sd":
            # A lopsided low/high split: large spread (SD high relative to the mean) but
            # deliberately kept off the exact half/half edge so the sample SD stays
            # comfortably below the variance ceiling — still genuinely achievable.
            k = max(1, int(n * 0.30) + (attempt % 5))
            k = min(k, n - 1)
            values = [lo] * k + [hi] * (n - k)
        else:
            # A mean that hugs (but does not touch) a bound: most responses one tick in
            # from a bound, a handful spread out. Kept a tick inside so rounding the mean
            # to `decimals` doesn't distort the variance ceiling at the extreme edge.
            inner_lo = min(lo + 1, hi)
            inner_hi = max(hi - 1, lo)
            spread = max(2, n // 10 + (attempt % 5))
            spread = min(spread, n - 1)
            if rng.random() < 0.5:
                values = [inner_lo] * (n - spread) + [hi] * spread     # mean just above lo
            else:
                values = [inner_hi] * (n - spread) + [lo] * spread     # mean just below hi
        rng.shuffle(values)
        mean, sd = _sample_mean_sd(values)
        if _rounds_unambiguously(mean, dec) and _rounds_unambiguously(sd, dec):
            break
    return _round_half_up(mean, dec), _round_half_up(sd, dec)


def _impossible_mean_stat(entry: dict, rng: random.Random) -> tuple[float, float]:
    """Mean strictly outside [scale_min, scale_max]; SD a benign small value."""
    lo, hi, dec = entry["scale_min"], entry["scale_max"], entry["decimals"]
    if rng.random() < 0.5:
        mean = hi + _MEAN_MARGIN + rng.uniform(0, 1.0)   # above the top of the scale
    else:
        mean = lo - _MEAN_MARGIN - rng.uniform(0, 1.0)   # below the bottom of the scale
    sd = rng.uniform(0.3, max(0.5, (hi - lo) / 4))
    return _round_half_up(mean, dec), _round_half_up(sd, dec)


def _impossible_sd_stat(entry: dict, rng: random.Random, overshoot: float) -> tuple[float, float]:
    """In-range mean, but reported SD overshoots the max attainable for that mean.

    Returns (mean, sd) with sd rounded UP so that, even after rounding, sd^2 still
    strictly exceeds the SAMPLE variance ceiling n/(n-1) * (hi - mean)*(mean - lo)
    (which itself strictly exceeds the population bound max_var = (hi-mean)(mean-lo)
    quoted in the contract). Overshooting the sample ceiling guarantees no achievable
    sample of any composition can reach the reported SD.
    """
    lo, hi, n, dec = entry["scale_min"], entry["scale_max"], entry["typical_n"], entry["decimals"]
    # Pick an interior mean where max_var is comfortably positive (avoid the bounds).
    span = hi - lo
    mean = rng.uniform(lo + 0.35 * span, hi - 0.35 * span)
    mean = _round_half_up(mean, dec)
    sample_max_var = _max_sample_var(mean, lo, hi, n)
    sd = _round_up(math.sqrt(sample_max_var * overshoot), dec)
    # Defend against the (rare) case where rounding the SD down at coarse `decimals`
    # would land it back on/under the bound: bump by one ULP until strictly over.
    ulp = 10 ** (-dec)
    while sd * sd <= sample_max_var:
        sd = _round_up(sd + ulp, dec)
    return mean, sd


def _render_stat(entry: dict, stat_id: str, kind: str | None, rng: random.Random,
                 subtle: bool = False) -> tuple[dict, dict]:
    """Render one statistic dict + its gold record."""
    lo, hi, n, dec = entry["scale_min"], entry["scale_max"], entry["typical_n"], entry["decimals"]

    if kind is None or kind == "clean":
        mean, sd = _clean_stat(entry, rng)
    elif kind == "control":
        mean, sd = _extreme_possible_stat(entry, rng)
    elif kind == "impossible_mean":
        mean, sd = _impossible_mean_stat(entry, rng)
    elif kind == "impossible_sd":
        overshoot = _SD_OVERSHOOT_SUBTLE if subtle else _SD_OVERSHOOT
        mean, sd = _impossible_sd_stat(entry, rng, overshoot)
    else:
        raise ValueError(f"unknown issue kind {kind!r}")

    flagged_kind = kind if kind in ISSUE_KINDS else None
    stat = {
        "stat_id": stat_id,
        "label": entry["label"],
        "mean": mean,
        "sd": sd,
        "n": n,
        "scale_min": lo,
        "scale_max": hi,
        "decimals": dec,
    }
    gold = {
        "stat_id": stat_id,
        "issue_kind": flagged_kind,
        "flagged": flagged_kind is not None,
    }
    return stat, gold


def _assemble(task_id, tier, plan, rng, split, visibility, catalog,
              subtle: bool = False) -> tuple[dict, dict]:
    """`plan` is a list of kinds: None/"clean" => clean, "control" => extreme-but-
    possible, or an ISSUE_KINDS entry => injected impossibility."""
    statistics: list[dict] = []
    gold_records: list[dict] = []

    for slot, kind in enumerate(plan):
        entry = catalog[(_seed_int(task_id, slot) + slot) % len(catalog)]
        stat_id = f"{entry['id']}__s{slot}"
        stat, gold = _render_stat(entry, stat_id, kind, rng, subtle=subtle)
        statistics.append(stat)
        gold_records.append(gold)

    # Shuffle the visible ordering so position carries no signal.
    rng.shuffle(statistics)

    mistake_kinds = sorted({g["issue_kind"] for g in gold_records if g["flagged"]})
    envelope = {
        "task_id": task_id,
        "arena_id": "sprite-plausibility-v1",
        "task_set_version": "v1",
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "n_issues": sum(1 for g in gold_records if g["flagged"])},
        "input": {
            "statistics": statistics,
        },
    }
    gold = {"records": gold_records, "mistake_kinds": mistake_kinds}
    return envelope, gold


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    visibility = "public" if split == "revealed" else "held_out"
    catalog = _load_catalog()
    n_kinds = len(ISSUE_KINDS)
    base = 5  # statistics per task (clean tasks too)

    def emit(tier, idx, plan, subtle=False):
        tid = f"sp-t{tier}-{idx}-s{seed}"
        rng = random.Random(_seed_int(task_set_version, seed, tier, idx))
        env, gt = _assemble(tid, tier, plan, rng, split, visibility, catalog, subtle=subtle)
        _GROUND_TRUTH_CACHE[tid] = gt
        return env

    # T1: clean/simple — every reported (mean, SD) is a real achievable sample stat.
    for k in range(3):
        yield emit(1, k, [None] * base)

    # T2: controls-only / false-alarm trap. Extreme-but-POSSIBLE stats (SD near max,
    # mean near a bound) that a good player must NOT flag. n_issues == 0.
    for k in range(3):
        yield emit(2, k, ["control"] * base)

    # T3: exactly one injected issue, cycling through EVERY issue kind (so every kind
    # appears at least once in the revealed set), the rest clean.
    for i, kind in enumerate(ISSUE_KINDS):
        plan = [None] * base
        plan[i % base] = kind
        yield emit(3, i, plan)

    # T4: a single SUBTLE issue — an impossible_sd that only BARELY exceeds the
    # maximum attainable variance, amid clean stats. (Cycle both kinds; the mean kind
    # is rendered identically, but T4 stresses the hardest discrimination.)
    subtle_kinds = ["impossible_sd", "impossible_mean"]
    for i, kind in enumerate(subtle_kinds):
        plan = [None] * base
        plan[(i + 1) % base] = kind
        yield emit(4, i, plan, subtle=True)

    # T5: MULTIPLE co-occurring issues (deterministic cycling offset across kinds),
    # the remaining slots clean.
    for k in range(3):
        plan = [None] * base
        for o in range(2):
            slot = (k + o) % base
            plan[slot] = ISSUE_KINDS[(k * 2 + o) % n_kinds]
        yield emit(5, k, plan)

    # T6: full composition — every issue kind present at once across the slots.
    for k in range(2):
        plan = [None] * base
        for j in range(n_kinds):
            plan[(j + k) % base] = ISSUE_KINDS[(j + k) % n_kinds]
        yield emit(6, k, plan)


def ground_truth(task_id: str) -> dict:
    """Return gold for a task. Regenerated from seed via the in-process cache.

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
