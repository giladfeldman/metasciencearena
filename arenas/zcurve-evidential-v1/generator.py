"""Generator for zcurve-evidential-v1 (single-verdict / FIELD-MAP arena).

Given a SET of statistically-significant test results (their z-scores, all
|z|>1.96), the player must classify whether the set reflects genuine underlying
effects (evidential value) or is consistent with selection-for-significance over
null/weak effects (no evidential value / p-hacked). The reference tool is the R
package `zcurve` (z-curve 2.0), which estimates the expected discovery rate /
expected replicability from the distribution of significant z-scores.

KEY GOLD PRINCIPLE — the arena does NOT run zcurve to make gold. Gold = the
GENERATING REGIME the arena controlled:

  - "evidential": z-scores drawn from studies with REAL effects (high true
    power). We draw z = |Normal(mean=ncp, sd=1)| keeping only z>1.96, with the
    noncentrality `ncp` chosen to give high per-study true power (~0.6-0.9). Such
    a set genuinely has evidential value  ->  has_evidential_value=true.
  - "non_evidential": z-scores drawn from NULL effects selected for significance
    — a truncated standard normal restricted to z>1.96 with ncp~=0 (only the
    false positives that happened to clear significance). No evidential value
    ->  has_evidential_value=false.

Harder mixed tiers (low true power, small sets) still take their gold label from
the DOMINANT generating regime.

Dual-benchmark (revealed/private): both splits run the IDENTICAL tier matrix and
assign the regime DETERMINISTICALLY by index-cycling (NOT rng.choice), so every
split covers BOTH regimes equally — this is what makes framework/parity.py pass.
Only the concrete z DRAWS are seed-driven (a sha256 `_seed_int` seeding a
`random.Random`; Gaussians via random.gauss — numpy-free), so revealed and
private z-score content still differ.

Gold is regenerated from the seed and served from the in-process cache (no
external registry): the secret is the private seed, not a stored answer key. The
revealed seed is committed in arena.yaml#benchmark_splits.
"""
from __future__ import annotations

import hashlib
import random

ARENA_ID = "zcurve-evidential-v1"
TASK_SET_VERSION = "v1"

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

# The two generating regimes, cycled deterministically. ALWAYS placed in the
# top-level gold `mistake_kinds` list so framework/parity.py sees both categories
# in both splits.
REGIMES = ["evidential", "non_evidential"]

_Z_CRIT = 1.96  # the significance threshold; every emitted z exceeds it.


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _draw_significant_evidential(rng: random.Random, ncp: float) -> float:
    """One z from a REAL-effect study (noncentrality `ncp`), kept only if >1.96.

    z = |Normal(mean=ncp, sd=1)|, rejection-sampled until significant. With a
    sizeable ncp this clears 1.96 quickly; a small ncp models a low-power real
    study (still genuine signal, just weak).
    """
    for _ in range(10000):
        z = abs(rng.gauss(ncp, 1.0))
        if z > _Z_CRIT:
            return z
    # Degenerate fallback (ncp effectively 0): nudge just past the threshold.
    return _Z_CRIT + abs(rng.gauss(0.0, 0.01)) + 1e-6


def _draw_significant_null(rng: random.Random) -> float:
    """One z from a NULL study selected for significance: a half-standard-normal
    truncated to z>1.96 (a false positive that happened to clear the bar)."""
    for _ in range(10000):
        z = abs(rng.gauss(0.0, 1.0))
        if z > _Z_CRIT:
            return z
    return _Z_CRIT + abs(rng.gauss(0.0, 0.01)) + 1e-6


def _build_set(rng: random.Random, regime: str, n_studies: int,
               ncp: float, contamination: float) -> list[float]:
    """Build a set of n_studies significant z-scores for the given regime.

    `contamination` is the fraction of studies drawn from the OPPOSITE process,
    used to make subtle/mixed tiers. The gold label remains the dominant regime:
      - evidential set: `contamination` fraction are null false-positives mixed
        into otherwise real-effect studies (still dominated by real signal);
      - non_evidential set: `contamination` fraction are a few weak real effects
        mixed into otherwise pure selection-over-nulls (still dominated by
        selection-for-significance).
    """
    n_contam = round(contamination * n_studies)
    zs: list[float] = []
    for i in range(n_studies):
        is_contam = i < n_contam
        if regime == "evidential":
            if is_contam:
                zs.append(_draw_significant_null(rng))
            else:
                zs.append(_draw_significant_evidential(rng, ncp))
        else:  # non_evidential
            if is_contam:
                # A weak real effect smuggled into the null-selection set.
                zs.append(_draw_significant_evidential(rng, max(ncp, 1.0)))
            else:
                zs.append(_draw_significant_null(rng))
    rng.shuffle(zs)
    return [round(z, 4) for z in zs]


# Per-tier configuration. Each tier is the SAME binary task at a different
# difficulty (set size + effect strength + contamination set the difficulty).
#   regime          : the dominant generating regime  ->  the gold label
#   n_studies       : size of the significant set
#   ncp             : noncentrality of the real-effect studies (higher = higher
#                     true power = more clearly evidential)
#   contamination   : fraction drawn from the opposite process (mixing)
_TIERS = {
    # T1: clearly evidential — high power, large set.
    1: {"regime": "evidential",     "n_studies": 40, "ncp": 3.2, "contamination": 0.0},
    # T2: clearly non-evidential — pure nulls selected for significance, large set.
    2: {"regime": "non_evidential", "n_studies": 40, "ncp": 0.0, "contamination": 0.0},
    # T3: evidential, moderate true power.
    3: {"regime": "evidential",     "n_studies": 30, "ncp": 2.4, "contamination": 0.0},
    # T4: non-evidential, subtle — a few real effects mixed into nulls, but still
    #     dominated by selection-for-significance.
    4: {"regime": "non_evidential", "n_studies": 30, "ncp": 2.0, "contamination": 0.2},
    # T5: small-set evidential (harder — less data for any estimator).
    5: {"regime": "evidential",     "n_studies": 8,  "ncp": 2.6, "contamination": 0.0},
    # T6: small-set non-evidential (harder).
    6: {"regime": "non_evidential", "n_studies": 8,  "ncp": 0.0, "contamination": 0.0},
}

# How many tasks to emit per tier in the revealed/private suite. Total 18.
_PER_TIER = 3


def _make_task(task_set_version: str, seed: int, tier: int, idx: int,
               split: str, visibility: str) -> dict:
    cfg = _TIERS[tier]
    # DETERMINISTIC regime: every tier has a fixed dominant regime (index-driven,
    # NOT rng.choice). Both regimes appear across tiers, in BOTH splits.
    regime = cfg["regime"]
    rng = random.Random(_seed_int(task_set_version, seed, tier, idx, regime))

    n_studies = cfg["n_studies"]
    z_scores = _build_set(rng, regime, n_studies, cfg["ncp"], cfg["contamination"])

    tid = f"zc-t{tier}-{idx}-s{seed}"
    study_set_id = f"{tid}__set"
    has_evidential_value = regime == "evidential"

    envelope = {
        "task_id": tid,
        "arena_id": ARENA_ID,
        "task_set_version": task_set_version,
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "regime": regime},
        "input": {
            "study_set_id": study_set_id,
            "z_scores": z_scores,
            "n_studies": len(z_scores),
        },
    }
    gold = {
        "has_evidential_value": has_evidential_value,
        "regime": regime,
        # ALWAYS non-empty — both regimes must appear in both splits so parity
        # sees two categories. (Not a clean/empty convention here.)
        "mistake_kinds": [regime],
    }
    _GROUND_TRUTH_CACHE[tid] = gold
    return envelope


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    """Yield task envelopes for the requested split.

    Both splits run the identical 6-tier matrix; the dominant regime per tier is
    fixed (index-driven), so each split covers both regimes equally. Only the z
    draws are seed-driven, so revealed and private content differ.
    """
    visibility = "public" if split == "revealed" else "held_out"
    for tier in sorted(_TIERS):
        for idx in range(_PER_TIER):
            yield _make_task(task_set_version, seed, tier, idx, split, visibility)


def ground_truth(task_id: str) -> dict:
    """Return gold for a task. Regenerated from seed via the in-process cache.

    The runner always calls generate() before ground_truth(); this arena needs
    no external gold registry because the secret is the private seed, not a
    stored answer key. Raises KeyError if the task is unknown.
    """
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(
            f"No cached gold for {task_id!r}; call generate() for the matching "
            "split/seed before ground_truth()."
        )
    return _GROUND_TRUTH_CACHE[task_id]
