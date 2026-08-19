"""Which differences between players are real, and which are noise.

THE PROBLEM THIS SOLVES
-----------------------
The leaderboard printed a confident 1-2-3-4 ordering that its own statistics
could not support. Two measured facts, both from 2026-08-15:

  * `significance-language-v1` is a genuine FOUR-WAY TIE. Every adjacent pair
    overlaps and nothing survives multiplicity correction, yet the board ranked
    them 1 through 4 as if the order meant something.
  * The obvious fix — "if the error bars overlap, call it a tie" — is wrong in
    the other direction, and badly. Of 342 comparable pairs across all arenas it
    would merge 180, and **39 of those (22%) are cleanly separated** by a proper
    paired test. Worst case: `pdf-text-fidelity-v1`, docpluck-standard vs
    pdftotext-raw, differ by only 0.027 with heavily overlapping bars — and
    docpluck wins on nearly every individual task (paired t = 11.3).

Both errors have the same root. The published +/- is `1.96*sd/sqrt(n)` computed
ACROSS TASKS, so it describes how much task difficulty varies — not how
precisely the difference between two players is known. Non-overlapping intervals
do imply a difference, but overlapping ones imply nothing (Schenker & Gentleman,
2001, *The American Statistician* 55(3):182-186).

WHY A PAIRED TEST IS THE RIGHT INSTRUMENT HERE
----------------------------------------------
Every ranked player runs the IDENTICAL task set — that is what
`assertRankedSymmetry` enforces. So the comparison is paired, and the shared
task-difficulty variance, which is most of the variance, cancels. That is why
pdftotext vs docpluck can differ by 0.027 with wide error bars and still be
unambiguous: docpluck is better on almost every single task.

WHAT IS COMPUTED
----------------
  statistic     studentized mean of the per-task DIFFERENCE, over the ranked
                common set exactly as the caller supplies it.
  test          sign-flip permutation (an exact paired randomisation test),
                B = 10,000 by default, SEEDED from (arena, task_set_version) —
                unseeded, identical data would publish different ranks on
                different builds.
  multiplicity  Holm across all pairs within the arena.
  rank          1 + the number of players that SIGNIFICANTLY beat you.

THE RANK RULE IS NOT NEGOTIABLE, AND NOT THE OBVIOUS ONE
--------------------------------------------------------
The tempting display is "merge adjacent players whose difference is not
significant into a band". Two independent reviewers (Fable 5 and Codex)
rejected it for the same reason, and they are right: "tied with" is NOT
transitive. A can be tied with B, B tied with C, and A still significantly
better than C — and banding would hide that. Counting how many players beat you
needs no transitivity assumption and cannot hide a real A>C. It is also what
Chatbot Arena does.

Conservative is the correct direction of error. An over-merged tie is honest
about what we do not know; a false separation publishes a claim about a real
tool that the data does not support.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_RESAMPLES",
    "MIN_TASKS_FOR_TESTING",
    "PairVerdict",
    "holm",
    "paired_signflip_p",
    "rank_players",
    "seed_for",
]

DEFAULT_ALPHA = 0.05
DEFAULT_RESAMPLES = 10_000

#: Below this many shared tasks a paired test has no power worth reporting — it
#: would return "not significant" for everything and dress an absence of
#: evidence up as a finding of equivalence. Under it, the ordering is published
#: as UNTESTED (`method: "mean_order_untested"`) so the UI can say so, rather
#: than quietly labelling every player tied.
MIN_TASKS_FOR_TESTING = 15


def seed_for(arena_id: str, task_set_version: str) -> int:
    """A stable seed per (arena, task set).

    The permutation test is randomised, so an unseeded run can publish rank 2 on
    Monday and rank 3 on Tuesday from byte-identical inputs. Derived from the
    identifiers rather than stored, so a new arena needs no bookkeeping and a
    task-set rotation deliberately gets a fresh draw.
    """
    digest = hashlib.blake2b(f"{arena_id}\x00{task_set_version}".encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big") % (2**32)


def _studentized(d: np.ndarray) -> np.ndarray:
    """t-like statistic per row; rows are permutations, columns are tasks."""
    n = d.shape[-1]
    mean = d.mean(axis=-1)
    sd = d.std(axis=-1, ddof=1) if n > 1 else np.zeros_like(mean)
    # sd == 0 means every task gave the same difference. If that difference is
    # also 0 the players are identical (t = 0); otherwise the separation is
    # perfect and t is +/-inf, which compares correctly against any finite
    # observed value.
    #
    # Written with an explicit out= rather than np.where on both branches:
    # np.where evaluates BOTH arms, so `np.sign(0) * np.inf` was computed for
    # every all-zero row and produced `0 * inf = nan` plus a RuntimeWarning.
    # The nan then went into the comparison as a silent False.
    t = np.zeros_like(mean)
    ok = sd > 0
    np.divide(mean, sd / math.sqrt(n), out=t, where=ok)
    degenerate = ~ok & (mean != 0)
    t[degenerate] = np.sign(mean[degenerate]) * np.inf
    return t


def paired_signflip_p(
    diffs: list[float] | np.ndarray,
    *,
    rng: np.random.Generator,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> float:
    """Two-sided p for "the per-task differences are centred on zero".

    A sign-flip permutation test: under the null that the two players are
    exchangeable on each task, flipping the sign of any subset of the paired
    differences is equally likely. So the null distribution is generated by
    doing exactly that, rather than assumed to be normal.

    The `(1 + count) / (B + 1)` form is deliberate — it can never return 0, and
    a reported p of exactly zero would be a claim no finite resampling can make.
    """
    d = np.asarray(diffs, dtype=float)
    n = d.size
    if n == 0:
        return 1.0
    if np.allclose(d, 0.0):
        return 1.0
    observed = float(_studentized(d[None, :])[0])
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, n))
    null = _studentized(d[None, :] * signs)
    # 1e-12 slack: the observed assignment is itself one of the sign patterns,
    # and float noise must not exclude it from its own null distribution.
    at_least_as_extreme = int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    return (1.0 + at_least_as_extreme) / (n_resamples + 1.0)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Holm rather than Bonferroni because it is uniformly more powerful at the
    same family-wise error rate, and rather than FDR because the cost structure
    here is asymmetric: a false separation publishes a wrong claim about a named
    third-party tool, while a missed one only leaves two players sharing a rank.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        # Enforced monotone: an adjusted p may never fall below an earlier one.
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


@dataclass
class PairVerdict:
    a: str
    b: str
    mean_diff: float          # mean(a) - mean(b) over the common set
    p_raw: float
    p_holm: float
    significant: bool

    def as_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "mean_diff": self.mean_diff,
            "p_raw": self.p_raw,
            "p_holm": self.p_holm,
            "significant": self.significant,
        }


def _rank_intervals(
    matrix: np.ndarray, rng: np.random.Generator, n_resamples: int
) -> list[tuple[int, int]]:
    """Bootstrap 95% interval for each player's rank.

    Resamples TASKS (columns) with replacement — the tasks are the sampling
    unit, and resampling players would answer a different question. Reported as
    secondary detail beside the point rank: "rank 1-4" communicates the actual
    uncertainty far better than a bare 2 does.
    """
    n_players, n_tasks = matrix.shape
    if n_tasks == 0:
        return [(1, n_players)] * n_players
    idx = rng.integers(0, n_tasks, size=(n_resamples, n_tasks))
    # (B, players): mean over the resampled columns, for every player at once.
    means = matrix[:, idx].mean(axis=2).T
    # Competition rank on each draw: 1 + how many players scored strictly HIGHER.
    #
    # The axes are the whole content of this line, so they are spelled out.
    # `means[:, :, None]` is (B, i, 1) and `means[:, None, :]` is (B, 1, j), so
    # element [b, i, j] asks "did i beat j?". Summing over axis 1 (over i) gives,
    # for each j, how many players beat it. Writing the two operands the other
    # way round counts how many j BEATS — the exact inverse — and the first
    # version did: on pdf-text-fidelity-v1 the best of six players was reported
    # with a rank interval of [6, 6]. Plausible-looking output, silently
    # backwards, and no test would have felt wrong.
    ranks = 1 + (means[:, :, None] > means[:, None, :] + 1e-15).sum(axis=1)
    lo = np.percentile(ranks, 2.5, axis=0)
    hi = np.percentile(ranks, 97.5, axis=0)
    return [(int(math.floor(a)), int(math.ceil(b))) for a, b in zip(lo, hi)]


def rank_players(
    *,
    arena_id: str,
    task_set_version: str,
    scores: dict[str, dict[str, float]],
    ranked: list[str],
    common: list[str],
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> dict:
    """Ranks, tie groups and pairwise verdicts for one arena's ranked players.

    `ranked` and `common` are supplied by the CALLER and used verbatim. They are
    never re-derived here, and that is a correctness requirement rather than a
    convenience: the published buckets come from
    `leaderboard-app/scripts/lib/fairness-gates.mjs::computeBuckets`, which
    applies the best-effort rules, and an independently re-derived intersection
    silently answers a different question. A cross-model review made exactly
    that mistake on its first sweep of this data.
    """
    players = [p for p in ranked if p in scores]
    n_common = len(common)

    matrix = np.array(
        [[float(scores[p].get(t, math.nan)) for t in common] for p in players],
        dtype=float,
    ) if players and common else np.zeros((len(players), 0))

    # A NaN means a ranked player is missing a common task, which
    # assertRankedSymmetry is supposed to make impossible. Refuse rather than
    # nan-propagate into a published rank.
    if matrix.size and not np.isfinite(matrix).all():
        missing = [
            (players[i], common[j])
            for i, j in zip(*np.where(~np.isfinite(matrix)))
        ]
        raise ValueError(
            f"{arena_id}/{task_set_version}: ranked players are missing common-set "
            f"scores, so the pairing is broken: {missing[:5]}"
        )

    means = matrix.mean(axis=1) if matrix.size else np.zeros(len(players))
    mean_by = dict(zip(players, (float(m) for m in means)))

    rng = np.random.default_rng(seed_for(arena_id, task_set_version))
    testable = n_common >= MIN_TASKS_FOR_TESTING and len(players) >= 2

    verdicts: list[PairVerdict] = []
    if testable:
        raw: list[float] = []
        pairs: list[tuple[int, int]] = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                raw.append(paired_signflip_p(
                    matrix[i] - matrix[j], rng=rng, n_resamples=n_resamples
                ))
                pairs.append((i, j))
        adjusted = holm(raw)
        for (i, j), p_raw, p_adj in zip(pairs, raw, adjusted):
            verdicts.append(PairVerdict(
                a=players[i], b=players[j],
                mean_diff=float(means[i] - means[j]),
                p_raw=float(p_raw), p_holm=float(p_adj),
                significant=bool(p_adj < alpha),
            ))

    # rank = 1 + the number of players that significantly BEAT you.
    beaten_by: dict[str, list[str]] = {p: [] for p in players}
    for v in verdicts:
        if not v.significant:
            continue
        loser, winner = (v.b, v.a) if v.mean_diff > 0 else (v.a, v.b)
        beaten_by[loser].append(winner)

    if testable:
        ranks = {p: 1 + len(beaten_by[p]) for p in players}
    else:
        # No power to test: publish the mean ordering, labelled untested, rather
        # than declaring everything tied — which would assert equivalence that
        # the data does not support either.
        order = sorted(players, key=lambda p: -mean_by[p])
        ranks = {p: 1 + sum(1 for q in order if mean_by[q] > mean_by[p] + 1e-15)
                 for p in players}

    intervals = (
        _rank_intervals(matrix, rng, n_resamples)
        if testable and matrix.size else [(ranks[p], ranks[p]) for p in players]
    )

    significant_pairs = {
        frozenset((v.a, v.b)) for v in verdicts if v.significant
    }
    rows = []
    for p, (lo, hi) in zip(players, intervals):
        tied = sorted(
            q for q in players
            if q != p and frozenset((p, q)) not in significant_pairs
        )
        rows.append({
            "player_id": p,
            "primary_mean": mean_by[p],
            "rank": ranks[p],
            "beaten_by": sorted(beaten_by[p]),
            "tied_with": tied if testable else [],
            "rank_ci_low": lo,
            "rank_ci_high": hi,
        })
    rows.sort(key=lambda r: (r["rank"], -r["primary_mean"], r["player_id"]))

    groups: dict[int, list[str]] = {}
    for r in rows:
        groups.setdefault(r["rank"], []).append(r["player_id"])

    return {
        "arena_id": arena_id,
        "task_set_version": task_set_version,
        "method": "paired_signflip_holm" if testable else "mean_order_untested",
        "tested": testable,
        "untested_reason": None if testable else (
            f"only {n_common} shared task(s); a paired test needs at least "
            f"{MIN_TASKS_FOR_TESTING} to have power worth publishing"
            if len(players) >= 2 else "fewer than two ranked players"
        ),
        "alpha": alpha,
        "n_resamples": n_resamples if testable else 0,
        "seed": seed_for(arena_id, task_set_version),
        "n_common": n_common,
        "n_players": len(players),
        "players": rows,
        "pairs": [v.as_dict() for v in verdicts],
        "tie_groups": [sorted(v) for _, v in sorted(groups.items())],
    }
