"""Ensemble analysis: what a COMBINATION of players achieves, not just the best one.

Numbers are returned at full float precision; rounding happens only where they are
printed. An earlier version rounded inside `analyse()`, which made this module
disagree with the JavaScript build implementation in the 8th decimal place and
would have published one number while the CLI reported another.

Motivation. The Dawes Institute planted-error benchmark (2026-08) reported that
the best single AI reviewer caught 71/100 errors while the union of all 14 caught
93/100 — and that five systems reached 90 while the remaining six added nothing.
A separate arXiv paper found union-of-6 recall 83.3% vs 71.6% best-single. Both
say the same thing: a leaderboard that only ranks individuals under-describes
what the field can actually do, and hides which tools are complementary versus
redundant.

Meta Science Arena publishes only single-player rows. This module computes the
combination view from run records we already have.

WHAT THIS MEASURES, PRECISELY
-----------------------------
`oracle_best_of` is **best-of-N at TASK granularity**: for each task, take the
highest primary score any member scored, then average over tasks. It answers
"if you could route each task to whichever of these players handles it best, how
well would you do?"

It is deliberately NOT called "union recall", because it is not the same quantity
Dawes reports. Theirs is item-level: of 100 planted errors, how many did at least
one system flag. Ours is task-level, because `score.findings[]` records a player's
MISTAKES rather than its catches, and turning that into an item-level union needs
each arena to declare which of its `error_categories` mean "missed it" versus
"false alarm". That field does not exist yet (see POLARITY below), and inferring
it from category-id spelling would be exactly the naming-convention classification
mistake the egress gate was burned by. Task-level is the honest thing computable
today; it is a lower bound on the item-level union.

WHY THIS IS NOT TRIVIALLY GAMEABLE
----------------------------------
A player that flags everything cannot inflate best-of-N, because it only
contributes on a task where its own primary score is highest — and every arena's
primary already penalises false positives within its own metric (GRIM's composite
multiplies by a false-alarm-sensitive term, extraction arenas use F1, and so on).
The precision guard is inherited from the scorer rather than bolted on here. That
is a real property, not an assumption: `groq-llama-3-1-8b-instant` scored 0.000
on GRIM precisely by over-flagging clean controls, so it can never join a greedy
ensemble on that arena.

HELD-OUT
--------
Public/revealed tasks only. Held-out records are redacted at write time
(`framework/holdout.py`): `output` is emptied, `breakdown` is emptied, findings
are stripped to `{category, count}`. A held-out ensemble would therefore have to
be computed inside the runner before redaction, which is a separate change.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

#: Categories whose meaning is "the player missed something that was there".
#: Populated per-arena from arena.yaml#error_categories[].polarity once that
#: field exists; until then item-level union is not computed at all rather than
#: guessed. See the module docstring.
POLARITY_FIELD = "polarity"


def load_public_scores(arena_dir: Path, task_set_version: str | None = None):
    """{player_id: {task_id: primary}} over scored public records.

    Excludes:
      - held-out records (nothing to ensemble; fields are redacted),
      - errored records (`score.breakdown.error`), which are an absence of
        evidence rather than evidence of failure — `aggregate()` excludes them
        from the mean too, and counting them as 0.0 here would let an outage
        masquerade as incapability,
      - `primary: null`, which the contract defines as "excluded, unverifiable".

    Multiple trials of a non-deterministic player collapse to their MEAN, so a
    player cannot buy ensemble membership with variance alone.
    """
    trials: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(arena_dir.glob("runs/*/*.jsonl")):
        if "_archive" in path.parts or "_pilot_archive" in path.parts:
            continue
        if task_set_version and f"/runs/{task_set_version}/" not in path.as_posix():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("task_visibility") != "public":
                continue
            score = rec.get("score") or {}
            if (score.get("breakdown") or {}).get("error"):
                continue
            primary = score.get("primary")
            if not isinstance(primary, (int, float)):
                continue
            trials[rec["player_id"]][rec["task_id"]].append(float(primary))
    return {
        player: {task: statistics.fmean(vals) for task, vals in tasks.items()}
        for player, tasks in trials.items()
    }


def common_tasks(scores: dict[str, dict[str, float]], players) -> set[str]:
    """Tasks every named player actually scored.

    The ensemble must be computed over a shared task set for the same reason the
    leaderboard ranks over one: otherwise a player that attempted only the easy
    tasks would appear to lift the combination.
    """
    sets = [set(scores.get(p, {})) for p in players]
    return set.intersection(*sets) if sets and all(sets) else set()


def oracle_best_of(scores, players, tasks=None) -> float | None:
    """Mean over tasks of the best score any member achieved on that task."""
    tasks = tasks if tasks is not None else common_tasks(scores, players)
    if not tasks or not players:
        return None
    return statistics.fmean(
        max(scores[p][t] for p in players if t in scores.get(p, {})) for t in tasks
    )


def greedy_order(scores, players, tasks=None):
    """Add players best-first, each time taking whoever lifts the oracle most.

    Reproduces the shape of the Dawes saturation curve: the point where marginal
    contribution hits zero is where the remaining players are redundant.
    """
    tasks = tasks if tasks is not None else common_tasks(scores, players)
    remaining, chosen, curve = list(players), [], []
    prev = 0.0
    while remaining:
        best, best_val = None, None
        for cand in remaining:
            val = oracle_best_of(scores, chosen + [cand], tasks)
            if val is not None and (best_val is None or val > best_val):
                best, best_val = cand, val
        if best is None:
            break
        chosen.append(best)
        remaining.remove(best)
        curve.append({
            "k": len(chosen),
            "player_id": best,
            "oracle_mean": best_val,
            "marginal_gain": best_val - prev,
        })
        prev = best_val
    return curve


def redundancy(scores, players, tasks=None):
    """For each pair, the share of tasks on which A is strictly better than B.

    A player that is never strictly better than some other player contributes
    nothing an ensemble containing that other player does not already have.
    """
    tasks = tasks if tasks is not None else common_tasks(scores, players)
    out = {}
    for a in players:
        for b in players:
            if a == b:
                continue
            wins = sum(1 for t in tasks if scores[a].get(t, 0) > scores[b].get(t, 0))
            out[f"{a}|{b}"] = (wins / len(tasks)) if tasks else 0.0
    return out


def analyse(arena_dir: Path, task_set_version: str | None = None) -> dict:
    scores = load_public_scores(arena_dir, task_set_version)
    players = sorted(scores)
    tasks = common_tasks(scores, players)
    if not players or not tasks:
        return {
            "arena_id": arena_dir.name,
            "n_players": len(players),
            "n_common_tasks": len(tasks),
            "note": "no shared public task set — nothing to ensemble",
        }
    singles = {p: statistics.fmean(scores[p][t] for t in tasks) for p in players}
    best_player = max(singles, key=singles.get)
    curve = greedy_order(scores, players, tasks)
    # Saturation = how many players actually CONTRIBUTE, i.e. the largest k whose
    # marginal gain is non-zero. Not "the k of the first player that adds
    # nothing", which is one too many: with two identical players the second
    # contributes zero, so the ensemble saturates at 1, not 2. This is the number
    # Dawes reports as "five systems reach 90, the remaining six add nothing".
    contributing = [row["k"] for row in curve if row["marginal_gain"] > 1e-9]
    saturation = max(contributing) if contributing else 0
    return {
        "arena_id": arena_dir.name,
        "n_players": len(players),
        "n_common_tasks": len(tasks),
        "best_single": {"player_id": best_player, "mean": singles[best_player]},
        "oracle_all": oracle_best_of(scores, players, tasks),
        "headroom": oracle_best_of(scores, players, tasks) - singles[best_player],
        "greedy_curve": curve,
        "saturates_at_k": saturation,
        "singles": {p: v for p, v in sorted(singles.items(), key=lambda kv: -kv[1])},
        "redundancy_pairwise": redundancy(scores, players, tasks),
        "scope": "public/revealed tasks only; task-level best-of-N, not item-level union",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ensemble analysis over run records.")
    ap.add_argument("--arena", help="arena id; omit for all arenas")
    ap.add_argument("--task-set", default=None)
    ap.add_argument("--arenas-root", default="arenas")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    root = Path(args.arenas_root)
    dirs = [root / args.arena] if args.arena else sorted(
        d for d in root.iterdir() if (d / "arena.yaml").exists()
    )
    results = [analyse(d, args.task_set) for d in dirs if (d / "arena.yaml").exists()]

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"{'arena':32s} {'n':>2s} {'tasks':>5s} {'best single':>11s} {'oracle':>7s} {'gain':>7s} {'sat':>4s}")
    for r in results:
        if "note" in r:
            print(f"{r['arena_id']:32s} {r['n_players']:2d} {r['n_common_tasks']:5d}   {r['note']}")
            continue
        print(f"{r['arena_id']:32s} {r['n_players']:2d} {r['n_common_tasks']:5d} "
              f"{r['best_single']['mean']:11.3f} {r['oracle_all']:7.3f} "
              f"{r['headroom']:+7.3f} {r['saturates_at_k']:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
