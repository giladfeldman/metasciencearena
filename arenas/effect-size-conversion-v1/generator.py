"""Generator for effect-size-conversion-v1 (computed-gold conversion arena).

Scientific task: convert a single effect size from one metric to another across the
closed set {d, r, OR, eta2, f}. Deterministic tools (the R packages `effectsize`
and `esc`) do this exactly via well-known closed-form identities; LLMs routinely get
the constants subtly wrong (e.g. the d->OR scaling, the sign/shape of d<->r, or the
f<->eta2 relation). This arena pits a deterministic reference tool against a Claude
baseline.

COMPUTED-GOLD: each task's correct answer is computed here in pure Python with the
SAME formula family the reference R adapter uses, so the tool scores ~1.0 (the
cross-validation guarantee). The canonical formula set (documented in README.md and
matched function-for-function against `effectsize`):

  d  -> r    r = d / sqrt(d^2 + h),  h = (n1+n2-2)*(1/n1 + 1/n2); h = 4 when no group
             sizes are given (the equal-large-n limit). [effectsize::d_to_r]
  r  -> d    d = sqrt(h) * r / sqrt(1 - r^2).               [effectsize::r_to_d]
  d  -> OR   OR = exp(d * pi / sqrt(3))   (logistic/log-odds). [effectsize::d_to_oddsratio]
  OR -> d    d  = ln(OR) * sqrt(3) / pi.                    [effectsize::oddsratio_to_d]
  eta2 -> f  f    = sqrt(eta2 / (1 - eta2)).                [effectsize::eta2_to_f]
  f -> eta2  eta2 = f^2 / (1 + f^2).                        [effectsize::f_to_eta2]
  d  -> f    f = d / 2   (two equal groups: Cohen's f = d/2).
  f  -> d    d = 2 * f.

Every (from, to) ordered pair is one of these eight identities (the metric graph is a
small star around d), so every conversion is exact and deterministic. Values are kept
well away from degenerate boundaries (|r| < ~0.93, eta2 < ~0.85, OR within a sane
range) so rounding can never flip agreement.

Deterministic from (task_set_version, seed) via a sha256 `_seed_int`, mirroring
grim-consistency-v1 / publication-bias-v1. Both splits (revealed/private) run the
IDENTICAL tier matrix and the same sequence of (from,to) conversion KINDS; only the
concrete `value` (and group sizes) differ by seed. Gold is cached in a module-level
`_GROUND_TRUTH_CACHE` and served by `ground_truth(task_id)` — registry-free; the
secret is the private seed.
"""
from __future__ import annotations

import hashlib
import math
import random

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

PI = math.pi
SQRT3 = math.sqrt(3.0)

# The closed set of metrics. The conversion graph is a star around `d` plus the
# eta2<->f pair, so every ordered (from,to) pair below is a single closed-form step.
METRICS = ["d", "r", "OR", "eta2", "f"]

# The closed set of supported ordered conversions, in canonical cycle order. Both
# splits cover this whole list (parity by construction). `needs_context` flags the
# conversions that consume group sizes (n1, n2) from the envelope's `context`.
CONVERSIONS: list[tuple[str, str, bool]] = [
    ("d", "r", False),
    ("r", "d", False),
    ("d", "OR", False),
    ("OR", "d", False),
    ("eta2", "f", False),
    ("f", "eta2", False),
    ("d", "f", False),
    ("f", "d", False),
    ("d", "r", True),   # with group sizes (n1, n2)
    ("r", "d", True),   # with group sizes (n1, n2)
]


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


# --------------------------------------------------------------------------- #
# The canonical conversion identities (the same maths the gold is computed with
# AND the reference R adapter dispatches to). `context` carries optional group
# sizes {n1, n2}; when absent, the d<->r conversions use h = 4 (equal-large-n).
# --------------------------------------------------------------------------- #
def _rd_h(context: dict | None) -> float:
    """Cohen's h factor for d<->r. h = (n1+n2-2)*(1/n1 + 1/n2); h = 4 when no group
    sizes are supplied (the equal/large-sample limit; effectsize's default)."""
    if context and "n1" in context and "n2" in context:
        n1 = float(context["n1"])
        n2 = float(context["n2"])
        m = n1 + n2 - 2.0
        return m / n1 + m / n2
    return 4.0


def convert(value: float, frm: str, to: str, context: dict | None = None) -> float:
    """Return the exact converted effect size for one supported (frm,to) identity."""
    if frm == to:
        return value
    if frm == "d" and to == "r":
        h = _rd_h(context)
        return value / math.sqrt(value * value + h)
    if frm == "r" and to == "d":
        h = _rd_h(context)
        return math.sqrt(h) * value / math.sqrt(1.0 - value * value)
    if frm == "d" and to == "OR":
        return math.exp(value * PI / SQRT3)
    if frm == "OR" and to == "d":
        return math.log(value) * SQRT3 / PI
    if frm == "eta2" and to == "f":
        return math.sqrt(value / (1.0 - value))
    if frm == "f" and to == "eta2":
        return (value * value) / (1.0 + value * value)
    if frm == "d" and to == "f":
        return value / 2.0
    if frm == "f" and to == "d":
        return 2.0 * value
    raise ValueError(f"unsupported conversion {frm!r}->{to!r}")


# --------------------------------------------------------------------------- #
# Value sampling: draw a `from`-metric value in a safe, non-degenerate range so the
# conversion and its round-trip stay numerically clean.
# --------------------------------------------------------------------------- #
def _draw_value(metric: str, rng: random.Random, magnitude: str = "normal") -> float:
    """Draw a valid value for `metric`. `magnitude` widens (extreme) or tightens
    (small) the range; all ranges stay clear of the degenerate boundaries."""
    if metric == "d":
        lo, hi = (0.1, 1.2) if magnitude == "normal" else (
            (0.02, 0.12) if magnitude == "small" else (1.6, 2.6))
    elif metric == "r":
        lo, hi = (0.1, 0.6) if magnitude == "normal" else (
            (0.02, 0.1) if magnitude == "small" else (0.75, 0.92))
    elif metric == "OR":
        lo, hi = (1.3, 4.0) if magnitude == "normal" else (
            (1.05, 1.25) if magnitude == "small" else (6.0, 12.0))
    elif metric == "eta2":
        lo, hi = (0.02, 0.30) if magnitude == "normal" else (
            (0.005, 0.02) if magnitude == "small" else (0.40, 0.80))
    elif metric == "f":
        lo, hi = (0.1, 0.6) if magnitude == "normal" else (
            (0.02, 0.1) if magnitude == "small" else (0.9, 1.8))
    else:  # pragma: no cover - guarded by METRICS
        raise ValueError(f"unknown metric {metric!r}")
    return round(rng.uniform(lo, hi), 4)


def _draw_context(rng: random.Random) -> dict:
    """Draw unequal group sizes for context-requiring conversions."""
    n1 = rng.choice([18, 24, 30, 36, 45, 60])
    n2 = rng.choice([22, 28, 40, 50, 70, 90])
    return {"n1": n1, "n2": n2}


def _assemble(task_id, tier, conv, rng, split, visibility, magnitude="normal") -> tuple[dict, dict]:
    frm, to, needs_ctx = conv
    context = _draw_context(rng) if needs_ctx else None
    value = _draw_value(frm, rng, magnitude)
    gold_converted = round(convert(value, frm, to, context), 6)

    inp: dict = {"value": value, "from": frm, "to": to}
    if context is not None:
        inp["context"] = context

    envelope = {
        "task_id": task_id,
        "arena_id": "effect-size-conversion-v1",
        "task_set_version": "v1",
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "from": frm, "to": to, "needs_context": needs_ctx},
        "input": inp,
    }
    gold = {
        "converted": gold_converted,
        "from": frm,
        "to": to,
        "value": value,
        "context": context,
    }
    return envelope, gold


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    visibility = "public" if split == "revealed" else "held_out"

    def emit(tier, idx, conv, magnitude="normal"):
        tid = f"esconv-t{tier}-{idx}-s{seed}"
        rng = random.Random(_seed_int(task_set_version, seed, tier, idx, conv[0], conv[1], conv[2]))
        env, gt = _assemble(tid, tier, conv, rng, split, visibility, magnitude)
        _GROUND_TRUTH_CACHE[tid] = gt
        return env

    # T1: identity/simple — the canonical d->r conversion (the textbook starting point).
    yield emit(1, 0, ("d", "r", False))
    yield emit(1, 1, ("r", "d", False))

    # T2: round-trip controls — a value converted out and that MUST come back stable.
    # Emitted as the forward leg here; the self-consistency test verifies the round
    # trip closes. Two stable round-trip pairs (d<->OR, eta2<->f).
    yield emit(2, 0, ("d", "OR", False))
    yield emit(2, 1, ("eta2", "f", False))

    # T3: each pairwise conversion kind, exactly once (the full no-context array).
    for i, conv in enumerate([c for c in CONVERSIONS if not c[2]]):
        yield emit(3, i, conv)

    # T4: conversions that NEED context (group sizes) — d<->r with unequal n1, n2.
    for i, conv in enumerate([c for c in CONVERSIONS if c[2]]):
        yield emit(4, i, conv)

    # T5: extreme-but-valid magnitudes — large effects that stress the constants but
    # stay clear of the degenerate boundaries.
    yield emit(5, 0, ("d", "r", False), magnitude="extreme")
    yield emit(5, 1, ("OR", "d", False), magnitude="extreme")
    yield emit(5, 2, ("eta2", "f", False), magnitude="extreme")
    yield emit(5, 3, ("f", "eta2", False), magnitude="extreme")

    # T6: full mix — small magnitudes across several kinds (the subtle low end where
    # LLMs drift on the constants).
    yield emit(6, 0, ("d", "OR", False), magnitude="small")
    yield emit(6, 1, ("r", "d", False), magnitude="small")
    yield emit(6, 2, ("d", "f", False), magnitude="small")
    yield emit(6, 3, ("f", "d", False), magnitude="small")


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
