"""Generator for grim-consistency-v1 (metacheck-style arena).

Builds a table of reported means with KNOWN granularity defects injected per
statistic. GRIM (Granularity-Related Inconsistency of Means) tests whether a
reported mean could have arisen as the mean of N integer responses on a bounded
scale: the mean must equal total/N for some achievable integer total, so it sits
on a 1/N granularity grid. All-procedural and deterministic from
(task_set_version, seed).

This arena is GRIM-only (granularity of a reported CENTRAL TENDENCY). Mean
statistics use n_items=1 (single-item scores), so the mean has granularity 1/N —
exactly the convention `scrutiny::grim()` models. Percentage statistics report k
of N respondents, so they sit on a 100/N grid — `scrutiny::grim(percent = TRUE)`,
the same GRIM test applied to a proportion.

GRIMMER / SD-granularity is deliberately OUT OF SCOPE: SD range-plausibility is
covered by the separate sprite-plausibility-v1 arena, and a prior attempt to
inject it here was a GOLD BUG (see CHANGELOG "Fixed").

Injected kinds (all pure GRIM):
  - grim_inconsistent          off-granularity mean (blind +-k ulp perturbation)
  - grim_percent_inconsistent  percentage no integer count k/N rounds to

The arena KNOWS the ground truth because it CONSTRUCTS every statistic itself:
clean means are computed from real integer data (so they are provably consistent);
injected means are perturbed to values a correct GRIM check rejects. No external
`scrutiny` runtime is required — the gold is computed here and independently
re-verified in the tests.

Dual-benchmark (revealed/private): both splits run the IDENTICAL tier matrix and
assign the injected issue KIND deterministically (index-driven cycling through ALL
issue kinds, seed-independent), so every split covers the full array of injected
issues equally — this is what makes framework/parity.py pass. Only the concrete
numbers (which scale, which integer sample) are seed-driven, so revealed and
private content still differ.

Gold is regenerated from the seed and served from the in-process cache (no
external registry): the secret is the private seed, not a stored answer key. The
revealed seed is committed in arena.yaml#benchmark_splits.
"""
from __future__ import annotations

import hashlib
import random
import statistics
from pathlib import Path

import yaml

ARENA_DIR = Path(__file__).resolve().parent
CATALOGS_DIR = ARENA_DIR / "catalogs"

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

# The closed set of injected granularity-inconsistency kinds, in canonical cycle
# order. Index-driven cycling through THIS list (not rng.choice) is what
# guarantees both splits cover every issue kind => parity passes.
#
# This arena stays GRIM-only — every kind is a GRANULARITY impossibility of a
# reported CENTRAL TENDENCY (a mean, or a percentage-of-a-count, which lies on the
# same 1/N grid). GRIMMER / SD-granularity remains deliberately OUT OF SCOPE: SD
# range-plausibility belongs to sprite-plausibility-v1, and a prior attempt to
# inject "grimmer_inconsistent" here shipped a GOLD BUG (it set an SD above the
# max-variance range bound — a SPRITE impossibility, not a GRIMMER one — so
# scrutiny correctly declined to flag it and scored 0.61). See CHANGELOG "Fixed".
ISSUE_KINDS = [
    "grim_inconsistent",
    "grim_percent_inconsistent",
]


def _load_catalog() -> list[dict]:
    with (CATALOGS_DIR / "scales.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


# --------------------------------------------------------------------------- #
# GRIM mathematics (the same check the gold is verified against).
# --------------------------------------------------------------------------- #
def grim_consistent(mean: float, n: int, n_items: int, decimals: int,
                    scale_min: int, scale_max: int) -> bool:
    """True iff `mean` (reported to `decimals` dp) is GRIM-consistent.

    The reported mean is the mean of N INTEGER responses on a bounded scale. With
    n_items=1 each response is an integer in [scale_min, scale_max], so the mean
    equals total/N for some integer `total` (the sum of all N responses) and sits on
    a 1/N granularity grid. The reported value is GRIM-consistent iff some integer
    total in the achievable range rounds (to the reported number of decimals) to the
    reported mean — exactly what `scrutiny::grim()` checks.

    (n_items is kept in the signature for input-schema parity and is always 1 in
    this arena; it widens the per-response integer range but never the mean's
    denominator, which is N.)
    """
    a = n_items * scale_min  # min achievable response value
    b = n_items * scale_max  # max achievable response value
    lo = n * a               # total at the floor
    hi = n * b               # total at the ceiling
    for total in range(lo, hi + 1):
        if round(total / n, decimals) == round(mean, decimals):
            return True
    return False


def grim_percent_consistent(pct: float, n: int, decimals: int) -> bool:
    """True iff `pct` (a percentage of a count, reported to `decimals` dp) is GRIM-consistent.

    A reported percentage is k/n*100 for an integer count k in [0, n], so it sits on
    a 100/N granularity grid — the same GRIM logic as a mean, and exactly what
    `scrutiny::grim(percent = TRUE)` checks. Verified against scrutiny 0.6.1
    (2026-08-04): at n=63, "42.9" is consistent (27/63=42.857) and "43.0" is not.
    """
    for k in range(0, n + 1):
        if round(k / n * 100.0, decimals) == round(pct, decimals):
            return True
    return False


# --------------------------------------------------------------------------- #
# Clean-statistic construction (provably consistent by construction).
# --------------------------------------------------------------------------- #
def _make_clean_stat(entry: dict, rng: random.Random) -> dict:
    """Draw integer data and report its (rounded) mean => GRIM-clean by construction."""
    n = entry["typical_n"] + rng.choice([-6, -3, -1, 0, 1, 2, 4, 7])
    n = max(8, n)
    n_items = 1  # GRIM-only arena: single-item scores, mean granularity 1/N.
    smin, smax = entry["scale_min"], entry["scale_max"]
    decimals = entry["decimals"]
    a, b = n_items * smin, n_items * smax
    # Draw N integer respondent scores on [a, b]; the rounded mean is GRIM-consistent
    # by construction. (SD is reported as descriptive context only — it is never the
    # subject of an injected issue in this GRIM-only arena.)
    data = [rng.randint(a, b) for _ in range(n)]
    mean = round(sum(data) / n, decimals)
    sd = round(statistics.stdev(data), decimals) if n >= 2 else 0.0
    return {
        "mean": mean, "sd": sd, "n": n, "n_items": n_items,
        "scale_min": smin, "scale_max": smax, "decimals": decimals,
        "_a": a, "_b": b,
    }


def _smallest_offset(decimals: int) -> float:
    return 10.0 ** (-decimals)


def _make_clean_percent_stat(entry: dict, rng: random.Random) -> dict:
    """Draw an integer count k out of N and report k/N*100 => GRIM-consistent by construction.

    Percentages are reported to 1 dp (the overwhelmingly common convention in
    papers), which is what makes them GRIM-testable: at N=63 only 64 of the 1000
    one-dp values in [0,100] are achievable.
    """
    n = entry["typical_n"] + rng.choice([-6, -3, -1, 0, 1, 2, 4, 7])
    n = max(8, n)
    decimals = 1
    k = rng.randint(1, n - 1)  # avoid 0%/100%, which are trivially consistent
    pct = round(k / n * 100.0, decimals)
    return {
        "stat_type": "percent",
        "percent": pct, "n": n, "decimals": decimals,
        "_k": k,
    }


def _inject_grim_percent(base: dict) -> dict:
    """Perturb a clean percentage to a value NO integer count k/N rounds to."""
    n, decimals = base["n"], base["decimals"]
    step = _smallest_offset(decimals)
    for k in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5):
        cand = round(base["percent"] + k * step, decimals)
        if cand <= 0.0 or cand >= 100.0:
            continue
        if not grim_percent_consistent(cand, n, decimals):
            out = dict(base)
            out["percent"] = cand
            return out
    raise RuntimeError("could not find a GRIM-inconsistent percentage perturbation")


def _inject_grim(base: dict) -> dict:
    """Perturb a clean mean to a value that NO integer total rounds to.

    Add the smallest representable offset at the reported precision and verify with
    our own GRIM check that the result is inconsistent; if not, try -offset,
    +2*offset, ... until provably inconsistent. NEVER returns a consistent value.
    """
    n, n_items = base["n"], base["n_items"]
    decimals = base["decimals"]
    smin, smax = base["scale_min"], base["scale_max"]
    step = _smallest_offset(decimals)
    for k in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5):
        cand = round(base["mean"] + k * step, decimals)
        if cand < base["_a"] or cand > base["_b"]:
            continue
        if not grim_consistent(cand, n, n_items, decimals, smin, smax):
            out = dict(base)
            out["mean"] = cand
            return out
    raise RuntimeError("could not find a GRIM-inconsistent perturbation")


def _inject_grim_subtle(base: dict) -> dict:
    """A grim_inconsistent mean that is HARD to spot: many decimals, +1 ulp only.

    Same off-granularity construction as `_inject_grim` but constrained to the
    smallest possible single-step perturbation so the reported mean is visually
    indistinguishable from a consistent value — the subtlest GRIM error.
    """
    n, n_items = base["n"], base["n_items"]
    decimals = base["decimals"]
    smin, smax = base["scale_min"], base["scale_max"]
    step = _smallest_offset(decimals)
    for k in (1, -1):
        cand = round(base["mean"] + k * step, decimals)
        if cand < base["_a"] or cand > base["_b"]:
            continue
        if not grim_consistent(cand, n, n_items, decimals, smin, smax):
            out = dict(base)
            out["mean"] = cand
            return out
    # Fall back to the general injector if a single ulp happened to stay consistent.
    return _inject_grim(base)


# --------------------------------------------------------------------------- #
# REJECTED KIND — do not re-add: "grim_subgroup_n_mismatch"
#
# Cycle 7 (2026-08-04) prototyped a "wrong denominator" injection: take a mean
# that is achievable at the FULL sample n and print it against a smaller subgroup
# n, where it is impossible. The arithmetic works (verified against scrutiny
# 0.6.1: "4.24" is GRIM-consistent at n=80, inconsistent at n=27) and it models a
# real reporting error — but it is NOT a usable arena kind, because it is
# OBSERVATIONALLY IDENTICAL to a plain `grim_inconsistent` mean.
#
# The tempting discriminator — "the value is achievable at some LARGER n" — is
# true of ordinary off-grid means too: measured over the revealed split it
# labelled 9 of 18 injected means correctly and 9 incorrectly, i.e. chance. Since
# no player (deterministic or LLM) could infer the kind from the input, shipping
# it would deflate `kind_accuracy` for reasons unrelated to skill and would fail
# the cycle gate's "new mistakes are detectable" check.
#
# A future cycle could make it legitimate ONLY by putting the discriminating
# evidence in the input — e.g. rendering the subgroup row NEXT TO the total row it
# was copied from, so the duplication is visible. That is a different task shape
# (cross-row consistency), not a drop-in kind.
# --------------------------------------------------------------------------- #


# Kinds whose statistic is a reported PERCENTAGE rather than a mean. The clean
# look-alike control ("clean_percent") uses the same shape but is achievable.
_PERCENT_KINDS = {"grim_percent_inconsistent", "clean_percent"}


def _render_stat(entry: dict, stat_id: str, kind: str | None, rng: random.Random):
    """Render one statistic dict + its gold record for the given (clean) kind."""
    norm_kind = None if kind in (None, "clean") else kind

    # ---- percentage statistics (GRIM on a proportion: k/N*100 on a 100/N grid) ----
    if norm_kind in _PERCENT_KINDS:
        base = _make_clean_percent_stat(entry, rng)
        if norm_kind == "grim_percent_inconsistent":
            base = _inject_grim_percent(base)
        else:
            # clean_percent is a matched control: an odd-LOOKING but achievable
            # percentage. It is a clean statistic, so it carries no gold kind.
            norm_kind = None
        stat = {
            "stat_id": stat_id,
            "label": entry["percent_label"],
            "stat_type": "percent",
            "percent": base["percent"],
            "n": base["n"],
            "decimals": base["decimals"],
        }
        return stat, {"stat_id": stat_id, "issue_kind": norm_kind,
                      "flagged": norm_kind is not None}

    # ---- mean statistics ----
    base = _make_clean_stat(entry, rng)
    if norm_kind == "grim_inconsistent":
        base = _inject_grim(base)
    elif norm_kind == "grim_inconsistent_subtle":
        # Internal sub-variant: same gold kind, subtler perturbation (used by T4).
        base = _inject_grim_subtle(base)
        norm_kind = "grim_inconsistent"
    elif norm_kind is not None:
        raise ValueError(f"unknown issue kind {norm_kind!r}")

    stat = {
        "stat_id": stat_id,
        "label": entry["label"],
        "stat_type": "mean",
        "mean": base["mean"],
        "sd": base["sd"],
        "n": base["n"],
        "n_items": base["n_items"],
        "scale_min": base["scale_min"],
        "scale_max": base["scale_max"],
        "decimals": base["decimals"],
    }
    gold = {
        "stat_id": stat_id,
        "issue_kind": norm_kind,
        "flagged": norm_kind is not None,
    }
    return stat, gold


def _assemble(task_id, tier, plan, rng, split, visibility, catalog) -> tuple[dict, dict]:
    """`plan` is a list of issue kinds (None/"clean" => clean statistic)."""
    statistics_out: list[dict] = []
    gold_records: list[dict] = []

    for slot, kind in enumerate(plan):
        entry = catalog[(slot + _seed_int(task_id, slot)) % len(catalog)]
        stat_id = f"{entry['id']}__s{slot}"
        stat, gold = _render_stat(entry, stat_id, kind, rng)
        statistics_out.append(stat)
        gold_records.append(gold)

    # Shuffle the visible ordering so position carries no signal.
    rng.shuffle(statistics_out)

    mistake_kinds = sorted({g["issue_kind"] for g in gold_records if g["flagged"]})
    envelope = {
        "task_id": task_id,
        "arena_id": "grim-consistency-v1",
        "task_set_version": "v1",
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "n_issues": sum(1 for g in gold_records if g["flagged"])},
        "input": {
            "statistics": statistics_out,
        },
    }
    gold = {"records": gold_records, "mistake_kinds": mistake_kinds}
    return envelope, gold


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    visibility = "public" if split == "revealed" else "held_out"
    catalog = _load_catalog()
    base = 6  # statistics per task (clean tasks too)

    def emit(tier, idx, plan):
        tid = f"grim-t{tier}-{idx}-s{seed}"
        rng = random.Random(_seed_int(task_set_version, seed, tier, idx))
        env, gt = _assemble(tid, tier, plan, rng, split, visibility, catalog)
        _GROUND_TRUTH_CACHE[tid] = gt
        return env

    # T1: clean/simple — every reported mean is GRIM-consistent.
    for k in range(3):
        yield emit(1, k, [None] * base)

    # T2: controls-only / false-alarm trap. Statistics that LOOK odd (small N,
    # many decimals, single-item scales) but ARE consistent. A good player must
    # NOT flag any of them. n_issues == 0.
    #
    # The third T2 task is the PERCENTAGE trap: achievable-but-odd percentages
    # (e.g. 42.9% at N=63). A player that treats "% to 1 dp" as free-floating
    # rather than a k/N grid will either flag these (false alarm) or, worse, never
    # flag any percentage at all — T3/T5/T6 catch the latter.
    for k in range(2):
        yield emit(2, k, ["clean"] * base)
    yield emit(2, 2, ["clean_percent"] * base)

    # T3: exactly one injected GRIM issue, the rest clean. Cycling through EVERY
    # issue kind guarantees the revealed set exercises the full injected-issue
    # array (this is also what makes revealed/private parity hold by construction).
    for i, kind in enumerate(ISSUE_KINDS):
        plan = [None] * base
        plan[i % base] = kind
        yield emit(3, i, plan)

    # T4: a single SUBTLE GRIM issue amid otherwise consistent statistics, paired
    # with its confusable clean look-alike in the same task.
    #   T4-0: off-granularity mean perturbed by ONE ulp (visually indistinguishable).
    #   T4-1: an inconsistent percentage hidden among CLEAN percentages, so the
    #         player can pass neither by blanket-flagging nor by blanket-ignoring
    #         the percentage type — it must actually do the k/N arithmetic.
    #   T4-2: a mixed table (means + percentages) with the single issue on the
    #         percentage side, so type-switching within one task is exercised.
    plan = [None] * base
    plan[1 % base] = "grim_inconsistent_subtle"
    yield emit(4, 0, plan)

    plan = ["clean_percent"] * base
    plan[4 % base] = "grim_percent_inconsistent"
    yield emit(4, 1, plan)

    plan = [None if s % 2 == 0 else "clean_percent" for s in range(base)]
    plan[3 % base] = "grim_percent_inconsistent"
    yield emit(4, 2, plan)

    # T5: MULTIPLE co-occurring GRIM issues (three injected, the rest clean),
    # mixing both kinds so a player cannot specialise in one detector.
    for k in range(3):
        plan = [None] * base
        for o, kind in enumerate(("grim_inconsistent",
                                  "grim_percent_inconsistent",
                                  "grim_inconsistent")):
            plan[(k + o) % base] = kind
        yield emit(5, k, plan)

    # T6: full composition — the maximum density of GRIM issues (every slot
    # injected), cycling all kinds so the densest tier is also the most varied.
    for k in range(2):
        plan = [ISSUE_KINDS[(k + s) % len(ISSUE_KINDS)] for s in range(base)]
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
