"""CLI entry point: `python -m framework <command>`."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from framework.discovery import discover_arenas, load_arena
from framework.paths import (
    ARENAS_ROOT_ENV,
    REGISTRY_PATH_ENV,
    RootNotFoundError,
    arenas_root,
    registry_path,
)
from framework.leaderboard import render_leaderboard
from framework.registry import load_registry
from framework.report import generate_report
from framework.runner import run_tournament
from framework.storage import read_records

# Roots are resolved PER COMMAND, not at import. At import time this module may
# be running from a wheel with no arenas beside it, and `--help` must still work.
# `framework.paths` raises rather than returning an empty root — see the module
# docstring there for why a silently-empty root is the dangerous failure.


def _cmd_arenas_list(_args):
    # No `if not root.exists(): return 0` guard any more. That turned a
    # misconfigured root into a clean exit-0 with no output — a broken install
    # reading exactly like a repo with no arenas. `arenas_root()` raises instead.
    root = arenas_root()
    for arena in discover_arenas(root):
        print(f"{arena['arena_id']:40s}  {arena['root'].relative_to(root.parent)}")
    return 0


def _cmd_players_list(_args):
    for entry in load_registry(registry_path()):
        print(f"{entry['player_id']:25s}  {entry['adapter_class']:30s}  conf={entry['confidence_strategy']}  det={entry['deterministic']}")
    return 0


def _resolve_seed(arena_dir, task_set_version, split, explicit_seed):
    """Explicit --seed wins; else use the split's seed from benchmark_splits; else 0."""
    if explicit_seed is not None:
        return explicit_seed
    try:
        manifest = load_arena(arena_dir)["manifest"]
    except Exception:
        return 0
    if not manifest.get("benchmark_splits"):
        return 0
    from framework.parity import resolve_seed
    seed, note = resolve_seed(manifest, arena_dir, task_set_version, split)
    if note:
        print(f"WARNING: {note}")
    return seed


def _cmd_run(args):
    arena_dir = arenas_root() / args.arena
    seed = _resolve_seed(arena_dir, args.task_set, args.split, args.seed)
    # Partition run records by split so revealed and private never share a file.
    out = arena_dir / "runs" / args.task_set / f"{'_'.join(args.players)}__{args.split}__{args.tag}.jsonl"

    # SAFETY DEFAULT (2026-08-04): `--split revealed` implies --public-only.
    #
    # `--split` selects the SEED; it never filtered by visibility. Arenas whose
    # generator emits both visibilities from one call (every PDF arena) therefore
    # played their HELD-OUT real papers during a `--split revealed` run. With an
    # LLM-CLI player that means shipping held-out PDFs — potentially copyrighted
    # APA papers (DATA_HANDLING.md) — to a third-party provider, from a command
    # that reads as "run the public split". That happened for real: a cycle-10
    # run sent 9 held-out PMC papers before it was caught.
    #
    # Egress is irreversible, so the default is now the safe one. Pass
    # --include-held-out to deliberately play held-out tasks in a revealed run.
    public_only = args.public_only
    if args.split == "revealed" and not args.held_out_only and not args.include_held_out:
        public_only = True

    n = run_tournament(
        arena_dir=arena_dir,
        task_set_version=args.task_set,
        registry_path=registry_path(),
        player_ids=args.players,
        output_path=out,
        trials=args.trials,
        timeout_s=args.timeout,
        seed=seed,
        public_only=public_only,
        held_out_only=args.held_out_only,
        max_tasks=args.max_tasks,
        split=args.split,
        overwrite=args.overwrite,
    )
    print(f"wrote {n} run records to {out} (split={args.split}, seed={seed})")
    return 0


def _record_is_ok(rec: dict) -> bool:
    """A record counts as a real verdict iff it carries no error breakdown and a
    non-null primary score (429 / timeout / adapter failures all fail this)."""
    score = rec.get("score") or {}
    if (score.get("breakdown") or {}).get("error"):
        return False
    return score.get("primary") is not None


def _cmd_retry_failed(args):
    """Rate-limit-aware retry loop for a single flaky (LLM/HTTP) player.

    Reads the player's run file, finds the tasks that errored (429 / timeout) or
    are missing versus the split's full task set, and re-plays ONLY those — merging
    fresh OK verdicts back in (an OK result replaces a prior error; other tasks are
    left untouched). Between rounds it sleeps ``--cooldown`` seconds so a per-day /
    per-minute token cap has time to recover. This is the durable "retry queue"
    for best-effort free-tier / slow providers: run it again tomorrow and it picks
    up exactly the tasks still outstanding. Records for tasks that never succeed
    stay as honest error records (they are excluded from scoring downstream)."""
    import time as _time

    arena_dir = arenas_root() / args.arena
    [player] = [args.player]
    seed = _resolve_seed(arena_dir, args.task_set, args.split, args.seed)
    target = arena_dir / "runs" / args.task_set / f"{player}__{args.split}__{args.tag}.jsonl"

    # Full task-id universe for this split (from the generator), so we can also
    # retry tasks that produced NO record at all.
    universe = _split_task_ids(arena_dir, args.task_set, args.split, seed)

    # Crash recovery: a previous invocation killed mid-round leaves an orphaned
    # `<target>.retry-r*.jsonl` temp file whose OK verdicts were never merged. Fold
    # those in first so no completed work is lost, then remove the temps.
    _recover_orphan_retry_temps(target)

    for round_i in range(1, args.max_rounds + 1):
        existing = {r["task_id"]: r for r in read_records(target)} if target.exists() else {}
        ok_ids = {t for t, r in existing.items() if _record_is_ok(r)}
        todo = sorted((universe - ok_ids)) if universe else sorted(
            t for t, r in existing.items() if not _record_is_ok(r))
        if not todo:
            print(f"[retry] {player}: all {len(ok_ids)} tasks OK — nothing to retry.")
            return 0
        print(f"[retry] round {round_i}/{args.max_rounds}: {len(ok_ids)} OK, "
              f"{len(todo)} to (re)try -> {todo[:6]}{'…' if len(todo) > 6 else ''}")

        tmp = target.with_suffix(f".retry-r{round_i}.jsonl")
        run_tournament(
            arena_dir=arena_dir, task_set_version=args.task_set, registry_path=registry_path(),
            player_ids=[player], output_path=tmp, trials=1, timeout_s=args.timeout,
            seed=seed, split=args.split, overwrite=True, only_tasks=set(todo),
        )
        fresh = {r["task_id"]: r for r in read_records(tmp)} if tmp.exists() else {}
        tmp.unlink(missing_ok=True)

        # Merge: a fresh OK verdict replaces whatever was there; a fresh error only
        # fills a previously-missing slot (never downgrades an existing OK).
        merged = dict(existing)
        gained = 0
        for tid, rec in fresh.items():
            if _record_is_ok(rec):
                if not _record_is_ok(existing.get(tid, {})):
                    gained += 1
                merged[tid] = rec
            elif tid not in merged:
                merged[tid] = rec
        # Persist merged file (stable task order).
        order = [t for t in universe if t in merged] if universe else sorted(merged)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for tid in order:
                fh.write(json.dumps(merged[tid], ensure_ascii=False) + "\n")

        now_ok = sum(1 for r in merged.values() if _record_is_ok(r))
        print(f"[retry] round {round_i}: +{gained} newly OK -> {now_ok}/"
              f"{len(universe) if universe else len(merged)} OK")
        if now_ok >= (len(universe) if universe else len(merged)):
            print(f"[retry] {player}: complete.")
            return 0
        if round_i < args.max_rounds:
            print(f"[retry] cooling down {args.cooldown}s before next round…")
            _time.sleep(args.cooldown)

    print(f"[retry] {player}: stopped after {args.max_rounds} rounds (still some "
          f"outstanding — run again later; the queue persists in the file).")
    return 0


def _recover_orphan_retry_temps(target: Path) -> None:
    """Merge OK verdicts from any orphaned ``<target>.retry-r*.jsonl`` temp files
    (left by a retry round that was killed before it merged) into ``target``, then
    delete them. An OK verdict replaces a prior error / fills a gap; nothing is
    downgraded. Best-effort: a malformed temp is skipped, not fatal."""
    temps = sorted(target.parent.glob(f"{target.stem}.retry-r*.jsonl"))
    if not temps:
        return
    existing = {r["task_id"]: r for r in read_records(target)} if target.exists() else {}
    gained = 0
    for tmp in temps:
        try:
            fresh = {r["task_id"]: r for r in read_records(tmp)}
        except Exception:  # noqa: BLE001 - a corrupt temp must not abort recovery
            tmp.unlink(missing_ok=True)
            continue
        for tid, rec in fresh.items():
            if _record_is_ok(rec) and not _record_is_ok(existing.get(tid, {})):
                existing[tid] = rec
                gained += 1
            elif tid not in existing:
                existing[tid] = rec
        tmp.unlink(missing_ok=True)
    if existing:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for tid in sorted(existing):
                fh.write(json.dumps(existing[tid], ensure_ascii=False) + "\n")
    if gained:
        print(f"[retry] recovered {gained} verdict(s) from {len(temps)} orphaned temp file(s).")


def _split_task_ids(arena_dir: Path, task_set_version: str, split: str, seed: int) -> set[str]:
    """Return the full set of task_ids the generator emits for one split (revealed
    → public visibility only). Empty set if the generator can't be introspected."""
    import importlib.util
    import inspect
    import sys
    try:
        spec = importlib.util.spec_from_file_location(
            f"_arena_gen_retry_{arena_dir.name}", arena_dir / "generator.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        gen_kwargs = {}
        if "split" in inspect.signature(mod.generate).parameters:
            gen_kwargs["split"] = split
        ids = set()
        for env in mod.generate(task_set_version, seed, **gen_kwargs):
            if split == "revealed" and env.get("visibility") not in ("public", None):
                continue
            if split == "private" and env.get("visibility") != "held_out":
                continue
            ids.add(env["task_id"])
        return ids
    except Exception as exc:  # noqa: BLE001 - introspection is best-effort
        import logging
        logging.getLogger(__name__).warning("could not enumerate tasks for %s: %s",
                                             arena_dir.name, exc)
        return set()


def _cmd_leaderboard(args):
    arena_dir = arenas_root() / args.arena
    runs_dir = arena_dir / "runs" / args.task_set
    if not runs_dir.exists():
        print(f"no runs at {runs_dir}")
        return 1
    records: list[dict] = []
    for jsonl in sorted(runs_dir.glob("*.jsonl")):
        records.extend(read_records(jsonl))
    print(render_leaderboard(records))
    return 0


def _cmd_report(args):
    """Generate a tool-feedback report bundle for one focal player.

    Optionally rebuilds the envelope/ground-truth caches from the arena's
    generator so per-task drilldowns can include difficulty axes, reproduce
    envelopes, and gold answers (public tasks only). Without --rebuild-tasks
    the report engine still produces summary + ranking + categorical histogram.
    """
    arena_dir = arenas_root() / args.arena
    envelopes_by_task = None
    ground_truth_lookup = None
    if args.rebuild_tasks:
        import importlib.util
        import inspect
        import sys
        spec = importlib.util.spec_from_file_location(
            f"_arena_generator_{args.arena}",
            arena_dir / "generator.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        gen_kwargs = {}
        if "split" in inspect.signature(mod.generate).parameters:
            gen_kwargs["split"] = args.split
        envelopes_by_task = {env["task_id"]: env for env in mod.generate(args.task_set, args.seed, **gen_kwargs)}
        ground_truth_lookup = mod.ground_truth

    bundle = generate_report(
        arena_dir=arena_dir,
        task_set_version=args.task_set,
        player_id=args.player,
        player_version=args.player_version,
        envelopes_by_task=envelopes_by_task,
        ground_truth_lookup=ground_truth_lookup,
    )
    print(f"wrote report bundle to {bundle}")
    return 0


def _cmd_audit(args):
    """Leaderboard fairness audits: tool-version drift (Finding 1) + task-set
    symmetry (Finding 2). Exit non-zero if any check fails (CI/cron gate)."""
    from framework.audit import (
        current_input_hashes,
        empty_run_files,
        load_arena_records,
        orphaned_retry_temps,
        per_split_symmetry_ok,
        stale_input_records,
        task_set_symmetry,
        version_drift_from_registry,
    )

    failed = False

    if args.versions:
        rows = version_drift_from_registry(load_registry(registry_path()))
        print("== tool-version drift (declared vs installed) ==")
        for row in rows:
            mark = "  DRIFT" if row.drift else ""
            print(f"  {row.player_id:32s} declared={row.declared:36s} "
                  f"installed={row.resolved}{mark}")
            if row.drift:
                failed = True
        if not any(r.drift for r in rows):
            print("  (all declared versions match the installed tool)")

    if args.symmetry:
        # Players flagged best_effort in the registry are documented partial
        # (free-tier / rate-limited / slow LLM providers) — see per_split_symmetry_ok.
        best_effort = {e["player_id"] for e in load_registry(registry_path())
                       if e.get("best_effort")}
        arena_dir = arenas_root() / args.arena if args.arena else None
        arenas = [arena_dir] if arena_dir else [
            a["root"] for a in discover_arenas(arenas_root())
        ]
        for adir in arenas:
            records = load_arena_records(adir, args.task_set)
            if not records:
                continue
            for vis, label in ((None, "all"), ("public", "revealed"),
                               ("held_out", "private")):
                rep = task_set_symmetry(records, arena_id=adir.name, visibility=vis)
                if vis is not None and not rep.per_player:
                    continue
                # Annotate best-effort players in the print so a tolerated partial
                # (free-tier throttle / slow provider) is not read as a real failure.
                be_here = sorted(best_effort & set(rep.per_player))
                suffix = f"  [best-effort (partial OK): {', '.join(be_here)}]" if be_here else ""
                # The PER-SPLIT report must exclude best-effort players exactly as
                # the gate does, or it prints "[revealed] SYMMETRY FAIL" for an
                # arena the gate passes — which reads as a real failure and sends
                # someone chasing a non-bug. (Observed 2026-08-04 on
                # prereg-deviation-v1: the three throttled regcheck providers made
                # the report cry foul while [GATE] said PASS one line below.)
                # The pooled "[all]" scope keeps every player, since it is
                # explicitly a diagnostic view and never gates.
                if vis is not None and be_here:
                    primary = [r for r in records
                               if r.get("player_id") not in best_effort]
                    gate_rep = task_set_symmetry(primary, arena_id=adir.name,
                                                 visibility=vis)
                    if gate_rep.per_player:
                        rep = gate_rep
                print(f"\n[{label}] {rep.summary()}{suffix}")
            # Only the PER-SPLIT checks gate the exit code (see per_split_symmetry_ok).
            # The pooled "[all]" scope is printed above for visibility but does not
            # gate: it mixes revealed+private task_ids, so a player that legitimately
            # runs BOTH splits (escimate on stats-extraction, the PDF parsers) looks
            # asymmetric versus revealed-only AI players — a false positive. Likewise
            # a best-effort provider short of the full task set is a tolerated partial.
            gate_ok = per_split_symmetry_ok(records, arena_id=adir.name,
                                            best_effort=best_effort)
            print(f"[GATE] {adir.name}: {'PASS' if gate_ok else 'FAIL'}")
            if not gate_ok:
                failed = True

    if args.fresh:
        # Symmetry asks WHICH task_ids ran; it cannot see that a kept task_id's
        # CONTENT changed underneath (in-place broadening does exactly that), so a
        # published score can describe text that no longer exists. Compare each
        # record's stored input_hash against the task the generator emits today.
        print("\n== stale-input records (score vs task content that has changed) ==")
        arena_dir = arenas_root() / args.arena if args.arena else None
        arenas = [arena_dir] if arena_dir else [
            a["root"] for a in discover_arenas(arenas_root())
        ]
        n_unchecked = 0
        for adir in arenas:
            records = load_arena_records(adir, args.task_set)
            if not records:
                continue
            hashes = current_input_hashes(adir, args.task_set)
            if not hashes:
                # Could not regenerate in-process — report it rather than pass silently.
                n_unchecked += 1
                print(f"  [SKIP] {adir.name}: generator not runnable in-process "
                      f"(not checked — this is NOT an all-clear)")
                continue
            rep = stale_input_records(records, hashes, arena_id=adir.name)
            if rep.ok:
                print(f"  [FRESH] {adir.name}: {rep.n_checked} record(s) match current tasks")
            else:
                print(f"  [STALE] {adir.name}:")
                for p in rep.problems:
                    print(f"    {p}")
                failed = True
        if n_unchecked:
            print(f"  ({n_unchecked} arena(s) skipped — see [SKIP] above)")

        # Surface unmerged retry temps. These are recoverable (the next
        # `retry-failed` folds them in) but nothing announced them, so real work
        # can sit unnoticed — see orphaned_retry_temps() for the cycle-8 incident.
        # Zero-byte run files: a run killed before its first task still leaves a
        # file named after a real player, which the build can publish as a
        # phantom zero-record entry. Blocking, because it corrupts the roster.
        empty_rows = []
        for adir in arenas:
            empty_rows += [(adir.name, p) for p in empty_run_files(adir, args.task_set)]
        if empty_rows:
            print("\n== empty (0-byte) run files ==")
            for aname, path in empty_rows:
                print(f"  [EMPTY] {aname}: {path.name} — a killed run wrote nothing; "
                      f"delete it or re-run that player")
            failed = True

        orphan_rows = []
        for adir in arenas:
            orphan_rows += [(adir.name, t, n)
                            for t, n in orphaned_retry_temps(adir, args.task_set)]
        if orphan_rows:
            print("\n== orphaned retry temps holding UNMERGED verdicts ==")
            for aname, temp, n in orphan_rows:
                print(f"  [ORPHAN] {aname}: {temp.name} has {n} unmerged OK verdict(s) — "
                      f"re-run `framework retry-failed` for that player to fold them in")

    return 1 if failed else 0


def _cmd_export_kaggle(args):
    # Imported lazily: the exporter pulls in the runner's arena-module loader,
    # and `framework --help` should not pay for that.
    from framework.export_kaggle import export_arena
    n = export_arena(args.arena, args.task_set, args.out, prompt_template=args.prompt)
    print(f"exported {n} public task(s) for {args.arena} to {args.out}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="framework")
    # Data roots are INPUTS. Installed off-repo there is no `arenas/` beside the
    # package, so a third party must be able to point at their own. Setting the
    # env var (rather than threading the value through every command) also means
    # subprocess players inherit the same root.
    parser.add_argument(
        "--arenas-root", type=Path, default=None, metavar="DIR",
        help=f"Directory of arena subdirectories. Overrides ${ARENAS_ROOT_ENV}.",
    )
    parser.add_argument(
        "--registry", type=Path, default=None, metavar="FILE",
        help=f"players/registry.yaml to use. Overrides ${REGISTRY_PATH_ENV}.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub_arenas = sub.add_parser("arenas").add_subparsers(dest="sub", required=True)
    sub_arenas.add_parser("list").set_defaults(func=_cmd_arenas_list)

    sub_players = sub.add_parser("players").add_subparsers(dest="sub", required=True)
    sub_players.add_parser("list").set_defaults(func=_cmd_players_list)

    p_run = sub.add_parser("run")
    p_run.add_argument("--arena", required=True)
    p_run.add_argument("--task-set", required=True)
    p_run.add_argument("--players", nargs="+", required=True)
    p_run.add_argument("--trials", type=int, default=3)
    p_run.add_argument("--timeout", type=int, default=120)
    p_run.add_argument("--tag", default="run")
    p_run.add_argument("--public-only", action="store_true",
                       help="Skip tasks whose envelope.visibility != 'public'. "
                            "IMPLIED by --split revealed (see --include-held-out).")
    p_run.add_argument("--include-held-out", action="store_true",
                       help="Opt OUT of the safe default: play held-out tasks during a "
                            "--split revealed run. Arenas whose generator emits both "
                            "visibilities (the PDF arenas) will then send held-out real "
                            "papers to the player — for a cloud LLM player that is "
                            "third-party egress of possibly copyrighted PDFs "
                            "(see DATA_HANDLING.md). Only pass this deliberately.")
    p_run.add_argument("--held-out-only", action="store_true",
                       help="Mirror of --public-only: keep ONLY held-out tasks. Use for "
                            "private/held-out runs of arenas whose generator emits both "
                            "visibilities every split (PDF arenas), so the private file "
                            "doesn't overlap the revealed public tasks.")
    p_run.add_argument("--max-tasks", type=int, default=None,
                       help="Stop after N envelopes (after public-only filter).")
    p_run.add_argument("--split", choices=["revealed", "private"], default="revealed",
                       help="Which benchmark suite to run. 'revealed' = published Open "
                            "Benchmark; 'private' = official held-out suite (redacted).")
    p_run.add_argument("--seed", type=int, default=None,
                       help="Override the generator seed. Default: the split's seed from "
                            "arena.yaml#benchmark_splits (revealed=committed, private=secret).")
    p_run.add_argument("--overwrite", action="store_true",
                       help="Replace the output JSONL instead of appending. Without this, "
                            "re-running with the same tag/split DOUBLES records (DR-0013).")
    p_run.set_defaults(func=_cmd_run)

    # retry-failed: rate-limit-aware retry queue for one flaky provider/player.
    p_retry = sub.add_parser(
        "retry-failed",
        help="Re-play only the tasks a flaky (free-tier / slow) player errored or "
             "missed, merging fresh OK verdicts back in; sleeps between rounds so "
             "token caps recover. Idempotent — safe to run again later.")
    p_retry.add_argument("--arena", required=True)
    p_retry.add_argument("--task-set", required=True)
    p_retry.add_argument("--player", required=True, help="Single player_id to backfill.")
    p_retry.add_argument("--tag", default="run", help="Run tag (must match the target file).")
    p_retry.add_argument("--split", choices=["revealed", "private"], default="revealed")
    p_retry.add_argument("--seed", type=int, default=None)
    p_retry.add_argument("--timeout", type=int, default=300)
    p_retry.add_argument("--max-rounds", type=int, default=6,
                         help="Max retry rounds this invocation (default 6).")
    p_retry.add_argument("--cooldown", type=int, default=150,
                         help="Seconds to sleep between rounds so a per-minute/day "
                              "token cap recovers (default 150).")
    p_retry.set_defaults(func=_cmd_retry_failed)

    p_lb = sub.add_parser("leaderboard")
    p_lb.add_argument("--arena", required=True)
    p_lb.add_argument("--task-set", required=True)
    p_lb.set_defaults(func=_cmd_leaderboard)

    p_rep = sub.add_parser("report", help="Generate a tool-feedback report bundle.")
    p_rep.add_argument("--arena", required=True)
    p_rep.add_argument("--task-set", required=True)
    p_rep.add_argument("--player", required=True, help="Focal player_id.")
    p_rep.add_argument("--player-version", required=True)
    p_rep.add_argument("--rebuild-tasks", action="store_true",
                       help="Re-run the generator to populate envelope/ground-truth caches "
                            "(needed for difficulty breakdown, reproduce envelopes, and gold in drilldown).")
    p_rep.add_argument("--seed", type=int, default=0,
                       help="Generator seed when --rebuild-tasks (must match the seed used at run time).")
    p_rep.add_argument("--split", choices=["revealed", "private"], default="revealed",
                       help="Which benchmark suite to rebuild tasks for (when --rebuild-tasks).")
    p_rep.set_defaults(func=_cmd_report)

    p_audit = sub.add_parser(
        "audit",
        help="Leaderboard fairness audits (version drift + task-set symmetry + input freshness).")
    p_audit.add_argument("--arena", default=None,
                         help="Limit symmetry check to one arena (default: all arenas).")
    p_audit.add_argument("--task-set", default="v1")
    p_audit.add_argument("--versions", action="store_true",
                         help="Check declared vs installed tool versions (Finding 1).")
    p_audit.add_argument("--symmetry", action="store_true",
                         help="Check players ran identical task sets (Finding 2).")
    p_audit.add_argument("--fresh", action="store_true",
                         help="Check each record's input_hash still matches the task the "
                              "generator emits today (catches scores left stale by an "
                              "in-place broaden, which symmetry cannot see).")
    p_audit.set_defaults(func=_cmd_audit)

    p_kaggle = sub.add_parser(
        "export-kaggle",
        help="Emit a Kaggle Benchmarks @kbench.task module for an arena's PUBLIC split.",
    )
    p_kaggle.add_argument("--arena", required=True)
    p_kaggle.add_argument("--task-set", default="v1")
    p_kaggle.add_argument("--out", type=Path, default=Path("build/kaggle"))
    p_kaggle.add_argument("--prompt", type=Path, default=None,
                          help="Prompt template containing {{INPUT_TEXT}}.")
    p_kaggle.set_defaults(func=_cmd_export_kaggle)

    args = parser.parse_args(argv)
    if args.arenas_root is not None:
        os.environ[ARENAS_ROOT_ENV] = str(args.arenas_root)
    if args.registry is not None:
        os.environ[REGISTRY_PATH_ENV] = str(args.registry)
    # `audit` with no flags runs every check.
    if args.cmd == "audit" and not (args.versions or args.symmetry or args.fresh):
        args.versions = args.symmetry = args.fresh = True
    return args.func(args)


def _console_main() -> int:
    """`metasciencearena` console-script entry point.

    A missing data root is a configuration error, not a crash: print the
    explanation `framework.paths` already wrote and exit 2, rather than dumping
    a traceback at someone who simply has no `arenas/` beside the wheel.
    """
    import sys
    try:
        return main(sys.argv[1:])
    except RootNotFoundError as exc:
        print(f"framework: {exc}", file=sys.stderr)
        return 2
