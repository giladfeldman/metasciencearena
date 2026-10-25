"""Data tiers: one task set holding several populations of tasks, scored apart.

citation-claim-fidelity-v1 task set v4 holds four populations (owner 2026-10-19). One mean
over all of a player's records would average different questions, and on rd-zhang2024 the
scorer's number is constraint agreement (an all-'silent' player scores 1.0; consult finding S3,
docs/design/2026-10-citeclaim-stage-d-real-data.md). So scores are per tier and never pooled, a
player counts on a tier only if it ran every task of that tier outside the optional "revealed"
practice split,
and a tier whose arena.yaml ``data_tiers.<id>.score_label`` is null gets no score at all.

Source: ``arenas/<id>/task_sets/<v>/tiers.jsonl`` (``{task_id, tier, split}`` per task). Run
records carry no tier. No file means the task set is one population.

Mirror of ``leaderboard-app/scripts/lib/data-tiers.mjs::planDataTiers``; both are pinned against
``framework/contract/fixtures/data_tier_cases.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

from framework.report import _is_scored, apply_unscoreable_policy, collapse_trials


def read_tier_index(arena_dir: Path, task_set_version: str) -> dict[str, dict] | None:
    path = Path(arena_dir) / "task_sets" / task_set_version / "tiers.jsonl"
    if not path.exists():
        return None
    return parse_tier_index(path.read_text(encoding="utf-8"), str(path))


def parse_tier_index(text: str, where: str = "tiers.jsonl") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("task_id") or not rec.get("tier") or not rec.get("split"):
            raise ValueError(f"{where}:{i}: tier record without task_id/tier/split")
        if rec["task_id"] in out:
            raise ValueError(f"{where}: duplicate task_id {rec['task_id']}")
        out[rec["task_id"]] = {"tier": rec["tier"], "split": rec["split"]}
    return out


def plan_data_tiers(*, tier_index: dict[str, dict] | None, declarations: dict | None,
                    records: list[dict]) -> dict | None:
    """Which tiers exist, who is complete on each, and what each complete player scored.

    Same output as the Node mirror: ``{tiers: [{id, label, score_label, note, n_tasks,
    n_tasks_by_split, task_ids, players: [{player_id, n_attempted, complete, n_scored?,
    mean?}]}], unknown_task_ids, undeclared_tiers}``. ``mean``/``n_scored`` exist only for a
    complete player on a tier with a score_label.
    """
    if tier_index is None:
        return None
    decl = declarations or {}
    tasks_by_tier: dict[str, dict[str, str]] = {}
    for tid, hit in tier_index.items():
        tasks_by_tier.setdefault(hit["tier"], {})[tid] = hit["split"]
    undeclared = sorted(t for t in tasks_by_tier if t not in decl)
    unknown: set[str] = set()
    recs_by_tier: dict[str, list[dict]] = {}
    for r in records:
        hit = tier_index.get(r.get("task_id"))
        if hit is None:
            unknown.add(r.get("task_id"))
            continue
        recs_by_tier.setdefault(hit["tier"], []).append(r)

    tiers = []
    for tier_id, d in decl.items():
        tasks = tasks_by_tier.get(tier_id)
        if not tasks:
            continue
        d = d or {}
        score_label = d.get("score_label")
        by_split: dict[str, int] = {}
        for split in tasks.values():
            by_split[split] = by_split.get(split, 0) + 1
        tier_recs = recs_by_tier.get(tier_id, [])

        attempted: dict[str, set[str]] = {}
        for r in tier_recs:
            attempted.setdefault(r["player_id"], set()).add(r["task_id"])
        # Complete = ran every task outside the "revealed" split (optional practice examples),
        # or every task when the tier has only revealed ones. Mirrors the Node rule.
        need = [t for t, s in tasks.items() if s != "revealed"] or list(tasks)
        complete = set()
        for pid, ran in attempted.items():
            if all(t in ran for t in need):
                complete.add(pid)

        scored_by: dict[str, list[dict]] = {}
        if score_label is not None:
            interpreted, _ = apply_unscoreable_policy([r for r in tier_recs if r["player_id"] in complete])
            for r in interpreted:
                if _is_scored(r):
                    scored_by.setdefault(r["player_id"], []).append(r)

        players = []
        for pid in sorted(attempted):
            row = {"player_id": pid, "n_attempted": len(attempted[pid]), "complete": pid in complete}
            if score_label is not None and pid in complete:
                ps = collapse_trials(scored_by.get(pid, []))
                row["n_scored"] = len(ps)
                row["mean"] = (sum(ps) / len(ps)) if ps else None
            players.append(row)

        tiers.append({
            "id": tier_id,
            "label": d.get("label") or tier_id,
            "score_label": score_label,
            "note": d.get("note"),
            "n_tasks": len(tasks),
            "n_tasks_by_split": by_split,
            "task_ids": sorted(tasks),
            "players": players,
        })
    return {"tiers": tiers, "unknown_task_ids": sorted(unknown), "undeclared_tiers": undeclared}


def public_tier_plan(plan: dict | None) -> dict | None:
    """The plan without per-tier task id lists: what the board publishes and the fixture pins."""
    if plan is None:
        return None
    return {
        "tiers": [{k: v for k, v in t.items() if k != "task_ids"} for t in plan["tiers"]],
        "unknown_task_ids": plan["unknown_task_ids"],
        "undeclared_tiers": plan["undeclared_tiers"],
    }
