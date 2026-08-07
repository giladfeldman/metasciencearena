"""Leaderboard rendering from run-record JSONL.

Intersection-aware ranking (2026-06-13 global-fairness handoff §A). A ranked
head-to-head is only fair when every compared player is scored over the SAME set
of task_ids. ``aggregate`` therefore never means over "whatever each player
happened to run": it derives a **common task set** and reports, per player,
``n_ranked`` (scored on the common set), ``n_ran_total`` (everything it ran), and
``n_missing`` (tasks in the co-ranked universe the player skipped). A player that
skipped any co-ranked task is flagged ``bucket="partial"`` so the site can move it
out of the headline ranked table; a player whose task set is *disjoint* from the
rest is never averaged head-to-head at all (``disjoint=True``).

This is a framework-level invariant, so it protects EVERY arena and every player
at once — not just the PDF ones the handoff was written against.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict


def _player_task_sets(records: list[dict]) -> dict[str, set[str]]:
    """task_ids each player ATTEMPTED (scored, excluded, or errored alike).

    Symmetry is about which tasks a player was asked to do, so an errored or
    unverifiable attempt still counts toward the attempted set.
    """
    attempted: dict[str, set[str]] = defaultdict(set)
    for r in records:
        tid = r.get("task_id")
        if tid is not None:
            attempted[r["player_id"]].add(tid)
    return attempted


def _derive_common(attempted: dict[str, set[str]]) -> tuple[set[str], set[str], set[str]]:
    """Return (common, co_ranked_players, ranking_universe).

    * ``co_ranked`` — players that share at least one task with the rest. A
      player whose task set is disjoint from every other player's is dropped: it
      cannot enter a head-to-head mean.
    * ``common`` — the intersection of the co-ranked players' task sets (the set
      every co-ranked player can be compared on).
    * ``ranking_universe`` — the union of the co-ranked players' task sets (used
      to compute ``n_missing``: a co-ranked player that ran fewer than the whole
      universe is incomplete/partial).
    """
    players = sorted(attempted)
    if not players:
        return set(), set(), set()
    if len(players) == 1:
        only = attempted[players[0]]
        return set(only), set(players), set(only)

    co_ranked: set[str] = set()
    for pid in players:
        others: set[str] = set()
        for q in players:
            if q != pid:
                others |= attempted[q]
        if attempted[pid] & others:
            co_ranked.add(pid)
    if not co_ranked:
        # Everyone pairwise-disjoint (degenerate): nothing is comparable.
        return set(), set(), set()
    co_sets = [attempted[pid] for pid in co_ranked]
    common = set.intersection(*co_sets) if co_sets else set()
    universe = set.union(*co_sets) if co_sets else set()
    return common, co_ranked, universe


def aggregate(records: list[dict], *, rank_task_ids: set[str] | list[str] | None = None) -> list[dict]:
    """Aggregate run records into per-player rows, ranked over a COMMON task set.

    ``rank_task_ids`` pins the comparison set explicitly; when ``None`` it is
    derived as the intersection of the co-ranked players' task sets (see
    ``_derive_common``). Only records whose ``task_id`` is in the common set enter
    a player's ranked mean.

    Three record buckets (unchanged from the original contract):
      * scored   — ``score.primary`` is a number; included in the mean (if its
                   task is in the common set).
      * excluded — ``score.primary`` is None (task marked unverifiable); reported,
                   never in the mean.
      * errored  — ``score.breakdown.error`` is set (adapter/infra failure);
                   reported, never in the mean — a crash is not a measurement of
                   skill (handoff §F).

    Per-player row fields:
      * ``primary_mean`` / ``primary_ci_half`` — over the ranked (common-set)
        records only.
      * ``n_ranked``    — scored records on the common set (drives the mean).
      * ``n_ran_total`` — scored records the player produced anywhere.
      * ``n_missing``   — tasks in the co-ranked universe the player did NOT run.
      * ``n_common``    — size of the common task set.
      * ``bucket``      — "ranked" (shares the common set with the field and has
                          ≥1 ranked score) or "partial" (disjoint, or produced no
                          score on the common set). A ranked player MAY still have
                          ``n_missing>0`` — it is compared fairly over the common
                          set, and ``n_missing`` separately flags the harder
                          (e.g. held-out) tasks it skipped.
      * ``disjoint``    — task set shares nothing with any other player.
      * ``n_trials``    — alias of ``n_ranked`` (kept for back-compat callers).
    """
    attempted = _player_task_sets(records)
    if rank_task_ids is not None:
        common = set(rank_task_ids)
        co_ranked = set(attempted)
        universe = set(common)
        for s in attempted.values():
            universe |= s
    else:
        common, co_ranked, universe = _derive_common(attempted)

    grouped: dict[str, list[float]] = defaultdict(list)   # common-set primaries
    all_scored: dict[str, int] = defaultdict(int)         # scored anywhere
    excluded: dict[str, int] = defaultdict(int)
    errored: dict[str, int] = defaultdict(int)
    latencies: dict[str, list[float]] = defaultdict(list)
    versions: dict[str, str] = {}
    resolved: dict[str, str | None] = {}
    for r in records:
        pid = r["player_id"]
        versions[pid] = r.get("player_version", "")
        if r.get("resolved_tool_version") and not resolved.get(pid):
            resolved[pid] = r.get("resolved_tool_version")
        breakdown = r["score"].get("breakdown") or {}
        if isinstance(breakdown, dict) and breakdown.get("error"):
            errored[pid] += 1
            continue
        primary = r["score"].get("primary")
        if primary is None:
            excluded[pid] += 1
            continue
        all_scored[pid] += 1
        if r.get("task_id") in common:
            grouped[pid].append(float(primary))
            lat = r.get("latency_ms")
            if lat is not None:
                latencies[pid].append(float(lat))

    all_players = set(attempted) | set(versions)
    rows = []
    for player_id in sorted(all_players):
        scores = grouped.get(player_id, [])
        n = len(scores)
        mean = (sum(scores) / n) if n else 0.0
        if n >= 2:
            sd = statistics.stdev(scores)
            ci_half = 1.96 * sd / math.sqrt(n)
        else:
            ci_half = 0.0
        lat_list = latencies.get(player_id, [])
        latency_mean = (sum(lat_list) / len(lat_list)) if lat_list else None
        is_disjoint = player_id not in co_ranked
        n_missing = len(universe - attempted.get(player_id, set()))
        # A player is co-ranked as long as it shares the common set with the
        # field and produced ≥1 score there; n_missing>0 (e.g. it skipped the
        # held-out tasks) does NOT exclude it — that's reported, not penalized.
        bucket = "ranked" if (not is_disjoint and n >= 1) else "partial"
        rows.append({
            "player_id": player_id,
            "player_version": versions.get(player_id, ""),
            "resolved_tool_version": resolved.get(player_id),
            "n_trials": n,            # back-compat alias of n_ranked
            "n_ranked": n,
            "n_ran_total": all_scored.get(player_id, 0),
            "n_missing": n_missing,
            "n_common": len(common),
            "bucket": bucket,
            "disjoint": is_disjoint,
            "n_excluded": excluded.get(player_id, 0),
            "n_errored": errored.get(player_id, 0),
            "primary_mean": mean,
            "primary_ci_half": ci_half,
            "latency_ms_mean": latency_mean,
        })
    return rows


def render_leaderboard(records: list[dict]) -> str:
    rows = aggregate(records)
    lines = [
        "| Player | Version | bucket | n_ranked | n_missing | n_excluded | n_errored | Primary (mean ± 95% CI) | Latency |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["n_ranked"] >= 2:
            score_cell = f"{r['primary_mean']:.3f} ± {r['primary_ci_half']:.3f}"
        elif r["n_ranked"] == 1:
            score_cell = f"{r['primary_mean']:.3f}"
        else:
            score_cell = "—"
        lat = r.get("latency_ms_mean")
        latency_cell = f"{int(round(lat))}ms" if lat is not None else "—"
        lines.append(
            f"| {r['player_id']} | {r['player_version']} | {r['bucket']} | {r['n_ranked']} | "
            f"{r['n_missing']} | {r['n_excluded']} | {r.get('n_errored', 0)} | {score_cell} | {latency_cell} |"
        )
    return "\n".join(lines) + "\n"
