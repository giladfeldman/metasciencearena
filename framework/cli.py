"""CLI entry point: `python -m framework <command>`."""
from __future__ import annotations

import argparse
import json
import os
import re
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
from framework.report import apply_unscoreable_policy, generate_report
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


def _cmd_arenas_versions(args):
    """Print one arena's task-set version ladder: status, why, and comparability.

    Answers the question a reader of a changed leaderboard actually has -- "which
    task set produced this number, and can I compare it to the last one?" --
    without opening arena.yaml.
    """
    import yaml

    from framework.discovery import version_ladder

    root = arenas_root()
    names = [args.arena] if args.arena else sorted(d.name for d in root.iterdir() if (d / "arena.yaml").is_file())
    for name in names:
        path = root / name / "arena.yaml"
        if not path.is_file():
            print(f"{name}: no arena.yaml")
            continue
        manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        ladder = version_ladder(manifest)
        declared = any(v.get("status") for v in (manifest.get("task_set_versions") or []))
        note = "" if declared else "   (status inferred: this arena declares none)"
        print(f"{name}{note}")
        for v in ladder:
            runs = root / name / "runs" / v["version"]
            has_runs = "runs" if runs.is_dir() else "no runs"
            comparable = v.get("comparable_with_previous")
            comp = {True: "comparable", False: "NOT comparable", None: "comparability undeclared"}[comparable]
            # Only a FROZEN version needs a fingerprint; flagging a current one as
            # "UNFINGERPRINTED" would be noise that trains the reader to ignore the word.
            if v["status"] == "current":
                fp = "-"
            else:
                fp = "fingerprinted" if v.get("generator_fingerprint") else "UNFINGERPRINTED"
            print(f"  {v['version']:<5s} {v['status']:<11s} {v.get('n_tasks', '?'):>4} tasks  "
                  f"{v.get('released_on', '?'):<12s} {has_runs:<8s} {fp:<15s} {comp}")
            if v.get("rationale"):
                print(f"        why: {' '.join(str(v['rationale']).split())}")
        print()
    return 0


def _cmd_players_list(_args):
    for entry in load_registry(registry_path()):
        print(f"{entry['player_id']:25s}  {entry['adapter_class']:30s}  conf={entry['confidence_strategy']}  det={entry['deterministic']}")
    return 0


def _resolve_seed(arena_dir, task_set_version, split, explicit_seed):
    """Explicit --seed wins; else use the split's seed from benchmark_splits; else 0.

    A PRIVATE run refuses the dev-fallback seed outright. `parity.resolve_seed`
    falls back to `revealed_seed + _DEV_PRIVATE_OFFSET` when the secret file is
    missing — fine for a local parity check, fatal for a run that WRITES records,
    because that seed is publicly computable (the revealed seed is committed in
    `arena.yaml` and the offset is a source constant). The old behaviour printed a
    warning and carried on, so a 120-call tournament could publish a "held-out"
    score for a split anyone with repo access could regenerate, with run records
    indistinguishable from official ones.

    Both callers (`_cmd_run`, `_cmd_retry_failed`) write records, so the gate lives
    here rather than in each of them. `--seed` still wins: an explicit seed is a
    deliberate, recorded choice.
    """
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
    if note and split == "private":
        raise SystemExit(
            f"REFUSING to run the private split of {arena_dir.name} without its secret seed.\n"
            f"  {note}\n"
            f"  Expected: {arena_dir / 'task_sets' / task_set_version / '.private_seed'}\n"
            "  That fallback seed is derivable from committed data, so records written with\n"
            "  it would be a private split in name only. Restore the secret (see\n"
            "  ~/sciencearena_private_seeds.backup.txt), or pass --seed to choose one\n"
            "  deliberately."
        )
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
    # NEVER print a private seed. The revealed seed is committed in arena.yaml, so
    # showing it aids reproduction; the private one IS the secret, and printing it
    # puts it in terminal scrollback, CI logs and session transcripts. A v2 private
    # run announced the freshly-rotated seed on its own first line (2026-08-09).
    shown = seed if args.split == "revealed" else "<redacted:private>"
    print(f"wrote {n} run records to {out} (split={args.split}, seed={shown})")
    return 0


def _record_ceiling(rec: dict) -> int | None:
    """The output ceiling this record says was SENT, or None if it does not say.

    `provenance.request_max_tokens` is the schema-v9 field and the authority;
    `response_meta.max_tokens` is what the local llama.cpp adapter writes and is
    the only source on pre-v9 records that have one at all. Measured 2026-09-11:
    of 11,709 records, the OpenAI-compatible path wrote NEITHER, so a `None` here
    is the common case for a hosted player and must never be read as "zero".
    """
    prov = rec.get("provenance") or {}
    for value in (prov.get("request_max_tokens"), (rec.get("response_meta") or {}).get("max_tokens")):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _record_is_truncated(rec: dict) -> bool:
    """Was this answer cut off at OUR output ceiling rather than by the model?

    `finish_reason == "length"` means the completion hit max_tokens. The record then
    carries a real-looking score computed from a half-finished answer -- usually 0.0,
    because truncated JSON does not parse -- and is INDISTINGUISHABLE from a model
    that cannot do the task. Measured 2026-08-20: 3 of 36 stats-extraction records,
    against a comparison player in the same arena that ran with no cap at all.
    """
    return ((rec.get("response_meta") or {}).get("finish_reason") == "length")


def _record_is_ok(rec: dict, *, include_truncated: bool = False) -> bool:
    """A record counts as a real verdict iff it carries no error breakdown and a
    non-null primary score (429 / timeout / adapter failures all fail this).

    ``include_truncated`` additionally demotes an answer cut off at our own token
    ceiling. It is OFF by default because the two callers want opposite things: a
    rate-limit backfill must not churn on a task that merely produced a long answer,
    while a ceiling-raise retry exists precisely to redo those. Turning it on without
    also raising the ceiling just burns rounds re-truncating the same task, which
    `_cmd_retry_failed` detects and stops.
    """
    score = rec.get("score") or {}
    if (score.get("breakdown") or {}).get("error"):
        return False
    if include_truncated and _record_is_truncated(rec):
        return False
    return score.get("primary") is not None


def _record_has_a_verdict(rec: dict) -> bool:
    """The SCORER ran and reached a conclusion about this task -- including the
    conclusion that it must be EXCLUDED from scoring.

    Deliberately weaker than :func:`_record_is_ok`, which additionally requires a
    numeric ``primary``. The difference is the whole point.

    MEASURED 2026-09-22, and this is a real published defect rather than a tidy-up.
    ``rtl-v1-0017`` errored with a 429 on the first pass of a SciMeto run. Its true
    verdict is ``excluded`` (``label_status == 'unverifiable'``) -- the same verdict
    the June 1.71.3 run recorded for it. On every retry the task replayed
    SUCCESSFULLY (verified in isolation through the real adapter: 91 matches), the
    scorer excluded it, and the merge below then DISCARDED that fresh record because
    an exclusion carries ``primary: None`` and so is not "OK". Fourteen retry rounds
    across four invocations changed nothing, and the run file kept publishing a 0.0
    that was manufactured by OUR OWN rate limiting -- the shape this project already
    has rules against (a timeout below the model's own ceiling scores our clock).

    A record with no error breakdown is strictly better evidence than one with an
    error breakdown, whether or not it yields a number. That is what this predicate
    is for, and it is used for exactly two things: deciding what still needs
    retrying, and letting a fresh verdict displace a stale transport failure. The
    "+N newly OK" counter keeps using :func:`_record_is_ok`, because an exclusion is
    not a score.
    """
    return not ((rec.get("score") or {}).get("breakdown") or {}).get("error")


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

    # Tasks that came back truncated AGAIN after a retry at a raised ceiling. The
    # ceiling was not the binding constraint for these -- the model spiralled without
    # ever reaching an answer -- so every further round costs another full-length
    # generation (~30 min at 12288 tokens on this machine) to reproduce the same
    # record. Give up PER TASK, not per round: one task that keeps spiralling must not
    # drag the others through pointless rounds, and a round that gained something
    # elsewhere must not mask it.
    hopeless: set[str] = set()

    for round_i in range(1, args.max_rounds + 1):
        existing = {r["task_id"]: r for r in read_records(target)} if target.exists() else {}
        inc_tr = bool(getattr(args, "include_truncated", False))
        ok_ids = {t for t, r in existing.items() if _record_is_ok(r, include_truncated=inc_tr)}
        # WHAT STILL NEEDS RETRYING is "has no verdict", not "has no score". A task
        # the scorer EXCLUDED has been answered; re-playing it buys nothing and it
        # can never become OK, so keying the queue on _record_is_ok made every round
        # re-play the same excluded tasks for ever and report "+0 newly OK" -- a
        # stall that reads exactly like a broken retry. Measured 2026-09-22: three
        # permanently-excluded tasks were re-played in all 14 rounds of one queue.
        settled = {t for t, r in existing.items()
                   if _record_is_ok(r, include_truncated=inc_tr) or _record_has_a_verdict(r)}
        if inc_tr:
            # A truncated answer carries no error breakdown, so it would count as
            # settled here. The whole point of --include-truncated is to redo it.
            settled -= {t for t, r in existing.items() if _record_is_truncated(r)}
        todo = sorted((universe - settled) - hopeless) if universe else sorted(
            t for t in existing if t not in settled and t not in hopeless)
        if not todo:
            print(f"[retry] {player}: all {len(ok_ids)} tasks OK — nothing to retry.")
            return 0
        print(f"[retry] round {round_i}/{args.max_rounds}: {len(ok_ids)} OK, "
              f"{len(todo)} to (re)try -> {todo[:6]}{'…' if len(todo) > 6 else ''}")

        tmp = target.with_suffix(f".retry-r{round_i}.jsonl")
        # Mirror _cmd_run's safety default: `--split revealed` implies
        # --public-only. Without it this passed public_only=False, the held-out
        # egress gate saw will_play_held_out=True, and a REVEALED backfill of any
        # cloud player died with "refusing to send HELD-OUT tasks" — for tasks that
        # are public by definition. The two commands must agree on the visibility
        # filter, or `retry-failed` cannot finish what `run` started.
        run_tournament(
            arena_dir=arena_dir, task_set_version=args.task_set, registry_path=registry_path(),
            player_ids=[player], output_path=tmp, trials=1, timeout_s=args.timeout,
            timeout_fallback_s=300,
            seed=seed, split=args.split, overwrite=True, only_tasks=set(todo),
            public_only=(args.split == "revealed"),
        )
        fresh = {r["task_id"]: r for r in read_records(tmp)} if tmp.exists() else {}
        tmp.unlink(missing_ok=True)

        # Merge: a fresh OK verdict replaces whatever was there; a fresh error only
        # fills a previously-missing slot (never downgrades an existing OK).
        merged = dict(existing)
        gained = 0
        for tid, rec in fresh.items():
            if _record_is_ok(rec, include_truncated=inc_tr):
                if not _record_is_ok(existing.get(tid, {}), include_truncated=inc_tr):
                    gained += 1
                merged[tid] = rec
            elif tid not in merged:
                merged[tid] = rec
            elif _record_has_a_verdict(rec) and not _record_has_a_verdict(merged[tid]):
                # A FRESH VERDICT DISPLACES A STALE TRANSPORT ERROR, even when that
                # verdict is an exclusion and so carries no number. Without this the
                # 429 record stays for ever and the file publishes a 0.0 that
                # measures our own rate limiting rather than the player. See
                # _record_has_a_verdict for the measurement.
                merged[tid] = rec
            elif _record_error_has_no_reason(merged[tid]) and not _record_error_has_no_reason(rec):
                # A REASONED FAILURE DISPLACES A REASONLESS ONE (d-94418a). Records
                # from before d-daa31d say only "claude exited 1:"; keeping them
                # over a retry that recorded the cause discards the one fact a
                # maintainer needs. Narrow on purpose: error-vs-error swaps beyond
                # this could trade a counted failure for an excluded one.
                merged[tid] = rec
            elif (
                inc_tr
                and _record_is_truncated(merged[tid])
                and _record_ceiling(rec) is not None
                and _record_ceiling(rec) > (_record_ceiling(merged[tid]) or 0)
            ):
                # A TRUNCATION PRODUCED UNDER A LOWER CEILING IS NOT EVIDENCE.
                # Without this branch the fresh attempt was discarded and the stale
                # record kept -- so a task retried at a CORRECTED ceiling still
                # published the attempt made under the wrong one, and carried no
                # trace of having been retried at all. Found by both seats of the
                # 2026-09-11 consult round; the Cohere grant's 5 truncations were
                # all in exactly this state, published at 4000 against a measured
                # provider maximum of 4096.
                #
                # Deliberately NARROW, because the surrounding rule ("a fresh error
                # never downgrades an existing OK") is correct and load-bearing:
                # this fires only when the kept record is TRUNCATED and the fresh
                # attempt used a STRICTLY HIGHER ceiling. Equal ceilings do not
                # qualify -- that is a re-roll of a non-deterministic player, which
                # buys a different sample rather than better evidence.
                merged[tid] = rec
        # Persist merged file (stable task order).
        order = [t for t in universe if t in merged] if universe else sorted(merged)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for tid in order:
                fh.write(json.dumps(merged[tid], ensure_ascii=False) + "\n")

        # Anything we just retried that is STILL truncated will not improve by being
        # retried again at the same ceiling. Retire it here, with its evidence.
        if inc_tr:
            # Judge the FRESH attempt, not the merged record. A fresh ERROR never
            # replaces an existing one, so `merged[t]` is still the old truncated
            # record and reading it reports "truncated again" for a retry that in fact
            # failed some other way. Measured 2026-08-21: at a 12288 ceiling this task
            # did not truncate at all, it REFUSED -- "I'm sorry, but I can't provide the
            # answer you're looking for" -- and the first version of this code called
            # that a truncation.
            newly_hopeless = {t for t in todo if t in fresh}
            if newly_hopeless:
                hopeless |= newly_hopeless
                for t in sorted(newly_hopeless)[:6]:
                    r = fresh[t]
                    if _record_is_truncated(r):
                        why = ("truncated AGAIN at the raised ceiling - the ceiling is "
                               "not the binding constraint; the model ran to the limit "
                               "without reaching an answer")
                    else:
                        err = ((r.get("score") or {}).get("breakdown") or {}).get("error")
                        why = (f"retry produced a DIFFERENT failure: {str(err)[:120]}"
                               if err else "retry produced no usable verdict")
                    print(f"[retry] {t}: {why}")
                print(f"[retry] retiring {len(newly_hopeless)} task(s) from this queue "
                      "rather than re-generating the same outcome.")

        now_ok = sum(1 for r in merged.values() if _record_is_ok(r, include_truncated=inc_tr))
        print(f"[retry] round {round_i}: +{gained} newly OK -> {now_ok}/"
              f"{len(universe) if universe else len(merged)} OK")
        total = len(universe) if universe else len(merged)
        if now_ok + len(hopeless) >= total:
            if hopeless:
                print(f"[retry] {player}: {now_ok}/{total} OK; {len(hopeless)} retired as "
                      f"ceiling-independent. Nothing further this queue can do.")
                return 1
            print(f"[retry] {player}: complete.")
            return 0
        # STILL TRUNCATED AFTER A RETRY = the ceiling is the binding constraint, and
        # another round re-truncates the same task at the same cap. Retrying is only
        # a remedy when something CHANGED; here nothing did, so say so and stop
        # rather than burn --max-rounds pretending to make progress.
        if inc_tr and gained == 0 and not todo:
            still = sorted(t for t, r in merged.items() if _record_is_truncated(r))
            if still:
                shown = ", ".join(still[:6]) + ("..." if len(still) > 6 else "")
                print(f"[retry] {player}: {len(still)} task(s) STILL truncated at "
                      f"the current ceiling -> {shown}")
                print("[retry] raise `openai_extra_body.max_tokens` for this "
                      "player in registry.yaml (bounded by the server's -c "
                      "context minus the prompt) and re-run. Stopping instead "
                      "of re-truncating.")
                return 1
        if round_i < args.max_rounds:
            print(f"[retry] cooling down {args.cooldown}s before next round…")
            _time.sleep(args.cooldown)

    print(f"[retry] {player}: stopped after {args.max_rounds} rounds (still some "
          f"outstanding — run again later; the queue persists in the file).")
    return 0


_REASONLESS_EXIT = re.compile(r"exited -?\d+:\s*$")


def _record_error_has_no_reason(rec: dict) -> bool:
    """True for a record whose error is a bare ``<cli> exited N:`` with nothing after."""
    err = ((rec.get("score") or {}).get("breakdown") or {}).get("error")
    return isinstance(err, str) and bool(_REASONLESS_EXIT.search(err))


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
            elif (_record_error_has_no_reason(existing[tid])
                  and not _record_is_ok(rec) and not _record_error_has_no_reason(rec)):
                existing[tid] = rec  # d-94418a: keep the failure that recorded its cause
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
    # What an unscoreable output is worth (TODO A5(b), decided 2026-08-19).
    # Applied HERE rather than inside `aggregate`, mirroring report.py::
    # build_report and lib/compute.ts::aggregateAllPlayers: the rule needs the
    # whole field, and `aggregate` stays a pure function over whatever records it
    # is handed. Without this the console printed 14 errored records corpus-wide
    # where the published reports printed 1 — the policy renames 13 of them to
    # counted zeros — so a reader comparing the two surfaces saw a player fail
    # tasks the site said it merely got wrong.
    records, orphan_tasks = apply_unscoreable_policy(records)
    print(render_leaderboard(records, orphan_tasks=orphan_tasks))
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
        formula_provenance_problems,
        frozen_drift,
        host_split_players,
        resolve_task_set_version,
        load_arena_records,
        orphaned_retry_temps,
        per_split_symmetry_ok,
        stale_input_records,
        task_set_symmetry,
        version_drift_from_registry,
        version_drift_gates,
        version_drift_summary,
        write_generator_fingerprints,
    )
    from framework import identity as identity_mod
    from framework.registry import undeclared_instruments

    failed = False

    if args.identity:
        # THE DECLARED INSTRUMENT. `player_version` labels weights and records no
        # inference configuration, so the same GGUF with thinking on and thinking
        # off declares one identical string while differing by +0.24 to +0.28
        # mean score. The instrument is declared instead, and this is where the
        # declaration is checked against the configs it claims to name.
        entries = load_registry(registry_path())
        print("== declared instrument identity ==")
        problems = identity_mod.audit_registry(entries)
        # A count first, so a clean run cannot be confused with a check that
        # examined nothing -- `audit_registry([])` also returns [].
        declared = [e for e in entries if e.get("instrument")]
        print(f"  {len(declared)} of {len(entries)} entries declare an instrument, "
              f"{len({e['instrument'] for e in declared})} distinct")
        undeclared = undeclared_instruments(entries)
        if undeclared:
            # NOT a gate here, and that is deliberate: a registry mid-edit may
            # carry players that have not published yet. The gate is at
            # publication (build-data.mjs §I), where the cost is actually paid.
            print(f"  {len(undeclared)} entry(ies) declare NO instrument, so any row they "
                  f"publish would fall back to the normalised player_version: "
                  f"{', '.join(undeclared[:10])}"
                  + (" ..." if len(undeclared) > 10 else ""))
        for prob in problems:
            print(f"  [PROBLEM] {prob}")
        if problems:
            failed = True
        else:
            print("  OK: no instrument covers two configurations, no configuration "
                  "is split across two names, every name is slug-safe.")

        # THE RECORDS, NOT JUST THE REGISTRY. The registry is a CLAIM about
        # which settings each instrument runs with; the `player_identity` block
        # a schema-v10 record carries is the EVIDENCE. build-data.mjs runs the
        # Node twin of this check (auditRecords) as a build gate. Until this
        # block existed the Python half had no non-test call site at all, so
        # "both engines implement it" was true of the Node engine only.
        #
        # Counts are printed before the verdict because TODAY almost every
        # record predates v10 and carries no block: a clean result over zero
        # stamped records is a real zero only if the reader can see it was zero.
        arena_dir = arenas_root() / args.arena if args.arena else None
        id_arenas = [arena_dir] if arena_dir else [
            a["root"] for a in discover_arenas(arenas_root())
        ]
        n_records = n_stamped = 0
        record_problems: list[str] = []
        for adir in id_arenas:
            ts = resolve_task_set_version(adir, args.task_set)
            recs = load_arena_records(adir, ts)
            n_records += len(recs)
            n_stamped += sum(1 for r in recs if r.get("player_identity"))
            record_problems += [f"{adir.name}: {p}"
                                for p in identity_mod.audit_records(recs, entries)]
        print(f"  records: {n_stamped} of {n_records} across {len(id_arenas)} arena(s) "
              "carry player_identity (the rest predate schema v10 -- unrecorded, "
              "never inferred)")
        for prob in record_problems:
            print(f"  [PROBLEM] {prob}")
        if record_problems:
            failed = True
        elif n_stamped:
            print("  OK: every stamped record agrees with the registry, and no "
                  "instrument name covers two inference configurations.")

    if args.versions:
        rows = version_drift_from_registry(load_registry(registry_path()))
        print("== tool-version drift (declared vs installed) ==")
        for row in rows:
            if row.verdict == "UNCHECKED":
                # The KIND is printed beside the prose, because the prose is for a
                # human and the kind is the thing that separates "no version
                # concept by design" from "the service is down". All 98 of these
                # used to read `adapter reported no installed version`.
                mark = "  PROBLEM" if row.absence_is_problem else ""
                print(f"  [UNCHECKED:{row.absence_kind or 'unclassified'}] "
                      f"{row.player_id:32s} declared={row.declared:36s} "
                      f"{row.reason}{mark}")
                continue
            mark = "  DRIFT" if row.drift else ""
            print(f"  [{row.verdict:<9s}] {row.player_id:32s} declared={row.declared:36s} "
                  f"installed={row.resolved}{mark}")
        # ONE function owns what gates, so the CLI and any other caller cannot
        # disagree about it. `probe_failed` is reported loudly by the summary
        # below and deliberately does NOT gate -- see version_drift_gates().
        if version_drift_gates(rows):
            failed = True
        for line in version_drift_summary(rows):
            print(line)

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
            ts = resolve_task_set_version(adir, args.task_set)
            records = load_arena_records(adir, ts)
            if not records:
                # Same rule as --fresh: an arena with nothing to check must SAY so.
                # `continue` alone let an arena disappear from the symmetry report
                # and still exit 0, which reads as a pass.
                print(f"  [SKIP] {adir.name}: no run records under runs/{ts}/ "
                      f"(not checked — this is NOT an all-clear)")
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
                # `vis is None` IS the pooled "[all]" scope, which never gates.
                print(f"\n[{label}] {rep.summary(gating=vis is not None)}{suffix}")
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
            # Per-arena, not the global --task-set default: auditing every arena at
            # a hardcoded "v1" made stats-extraction-v1 vanish from this report the
            # moment it moved to v2 and archived its v1 runs.
            ts = resolve_task_set_version(adir, args.task_set)
            records = load_arena_records(adir, ts)
            if not records:
                # NEVER silent. An arena with no records at the version we scanned
                # is an unchecked arena, and an unchecked arena that prints nothing
                # is indistinguishable from a clean one.
                n_unchecked += 1
                print(f"  [SKIP] {adir.name}: no run records under runs/{ts}/ "
                      f"(not checked — this is NOT an all-clear)")
                continue
            hashes = current_input_hashes(adir, ts)
            if not hashes:
                # Could not regenerate in-process — report it rather than pass silently.
                n_unchecked += 1
                print(f"  [SKIP] {adir.name}: generator not runnable in-process "
                      f"(not checked — this is NOT an all-clear)")
                continue
            rep = stale_input_records(records, hashes, arena_id=adir.name)
            # An unchecked record is not a clean one. Before this, `[FRESH] N
            # record(s)` looked identical whether N was 220 of 220 or 148 of 220.
            if rep.n_ignored:
                n_unchecked += 1
                print(f"  [PARTIAL] {adir.name}: {rep.n_ignored} of "
                      f"{rep.n_checked + rep.n_ignored} record(s) could NOT be checked "
                      f"(task_id not in the current task set — missing secret seed, or "
                      f"records from a retired split)")
            if rep.n_unverifiable:
                # Not a defect and not an all-clear: held-out records have their
                # input_hash redacted on purpose, so freshness is unknowable for
                # them. Say that, rather than calling them fresh OR stale.
                print(f"  [HELD-OUT] {adir.name}: {rep.n_unverifiable} held-out record(s) "
                      f"cannot be freshness-checked (input_hash is redacted by design)")
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
            empty_rows += [(adir.name, p)
                           for p in empty_run_files(adir, resolve_task_set_version(adir, args.task_set))]
        if empty_rows:
            print("\n== empty (0-byte) run files ==")
            for aname, path in empty_rows:
                print(f"  [EMPTY] {aname}: {path.name} — a killed run wrote nothing; "
                      f"delete it or re-run that player")
            failed = True

        orphan_rows = []
        for adir in arenas:
            orphan_rows += [(adir.name, t, n)
                            for t, n in orphaned_retry_temps(adir, resolve_task_set_version(adir, args.task_set))]
        if orphan_rows:
            print("\n== orphaned retry temps holding UNMERGED verdicts ==")
            for aname, temp, n in orphan_rows:
                print(f"  [ORPHAN] {aname}: {temp.name} has {n} unmerged OK verdict(s) — "
                      f"re-run `framework retry-failed` for that player to fold them in")

    # ALWAYS ON, not behind a flag, and free until something is actually frozen:
    # the loop below only visits versions marked `frozen`/`superseded`, of which
    # there are none on an arena that has never versioned. A frozen task set is a
    # PROMISE that the generator still emits exactly the tasks the published
    # scores were computed on, and nothing else in the repo can check it --
    # pinning a constant and merely reordering two `rng` calls each silently
    # rewrote a frozen task set before, caught only by diffing the oldest gold.
    frozen_arenas = ([arenas_root() / args.arena] if args.arena
                     else [a["root"] for a in discover_arenas(arenas_root())])
    if getattr(args, "write_fingerprints", False):
        print("\n== baselining generator fingerprints for frozen task sets ==")
        for adir in frozen_arenas:
            res = write_generator_fingerprints(adir)
            for version, fp in res.written:
                print(f"  [WROTE] {adir.name} {version}: {fp}")
            for version, why in res.refused:
                # A refusal that printed nothing read as "this arena needed
                # nothing". Baselining a half-computed hash is worse than not
                # baselining at all: it records a guard that can never fail.
                print(f"  [REFUSED] {adir.name} {version}: NOT baselined — a split "
                      f"contributed nothing, so the fingerprint would cover only "
                      f"part of the task set and could never catch drift in the "
                      f"rest.")
                print(f"      why: {why}")
                print(f"      Fix the machine (obtain the seed / restore the corpus) "
                      f"and re-run. Baseline only from a machine that can generate "
                      f"the WHOLE task set.")
    frozen_rows = [r for adir in frozen_arenas for r in frozen_drift(adir)]
    if frozen_rows:
        print("\n== task-set generator drift ==")
        for r in frozen_rows:
            if r.verdict == "OK":
                print(f"  [OK] {r.arena_id} {r.version} ({r.status}): "
                      f"{r.n_tasks} tasks reproduce their fingerprint")
            elif r.verdict == "DRIFT":
                print(f"  [DRIFT] {r.arena_id} {r.version} ({r.status}): the generator no "
                      f"longer reproduces this task set.")
                print(f"      recorded: {r.expected}")
                print(f"      current : {r.actual}  ({r.n_tasks} tasks)")
                # THE FALSE-ALARM CHECK COMES FIRST, DELIBERATELY. The most likely
                # cause of a DRIFT on a fresh checkout is a missing secret, not a
                # changed generator, and a guard whose failure mode is a confident
                # false alarm is one people learn to ignore. Say how to rule that
                # out BEFORE sending anyone to investigate the generator.
                print(f"      FIRST: do you have the private seed? This fingerprint covers "
                      f"BOTH splits and resolves the private seed from the gitignored "
                      f"`.private_seed` (or SCIENCEARENA_PRIVATE_SEED). On a machine "
                      f"without it that split contributes NOTHING, so the fingerprint is "
                      f"computed over the REVEALED tasks only and cannot match one recorded "
                      f"on a machine that had the seed. Obtain the seed and re-run before "
                      f"investigating anything else.")
                print(f"      IF THE SEED IS PRESENT: published scores for {r.version} were "
                      f"computed on the RECORDED tasks. Revert the generator change, or "
                      f"branch it behind the task_set_version instead of editing shared "
                      f"code.")
                if r.reason:
                    # MEASURED, not guessed. A DRIFT is computed over whichever
                    # splits produced tasks, so one split failing silently makes
                    # a half-task-set fingerprint that cannot match a full one --
                    # reported here as drift, with no hint that half the input
                    # went missing. An exception name on this line means the
                    # generator broke, not that it changed.
                    print(f"      split notes: {r.reason}")
                    print(f"      (a split emitting nothing is NORMAL for an arena with no "
                          f"private split -- replication-target-lookup-v1 is one. An "
                          f"exception name here is NOT normal.)")
                failed = True
            elif r.verdict == "UNFINGERPRINTED":
                # NOT a pass. Named as its own state rather than folded into OK,
                # because "we never recorded what this was" and "it still matches"
                # are different facts and only one of them is reassuring.
                # Say "live" for a current set. This branch used to hardcode
                # "frozen but unguarded", which was true while only frozen sets
                # reached it and became a FALSE STATEMENT IN OUTPUT once current
                # ones did (2026-08-26). A guard that misdescribes what it checked
                # is the same defect class as one that does not check.
                kind = "LIVE and" if r.status == "current" else "frozen but"
                print(f"  [UNFINGERPRINTED] {r.arena_id} {r.version} ({r.status}): {kind} "
                      f"unguarded ({r.n_tasks} tasks). Baseline it with "
                      f"`framework audit --write-fingerprints`.")
            elif r.verdict == "PARTIAL-GUARD":
                # A MATCH THAT PROVES NOTHING. Gates, because the recorded value
                # covers only part of the task set and every future audit on this
                # machine will keep saying OK.
                print(f"  [PARTIAL-GUARD] {r.arena_id} {r.version} ({r.status}): the "
                      f"recorded fingerprint MATCHES, but it was recomputed over "
                      f"only part of the task set ({r.n_tasks} tasks), so the rest "
                      f"is guarded by nothing.")
                print(f"      why: {r.reason}")
                print(f"      Either this machine cannot generate the whole task set "
                      f"(obtain the seed / restore the corpus and re-run), or the "
                      f"recorded fingerprint was itself baselined on a machine that "
                      f"could not — in which case it is a rubber stamp and must be "
                      f"re-baselined from a complete machine, with the reason in the "
                      f"arena CHANGELOG.")
                failed = True
            elif r.verdict == "ASSET-ABSENT":
                # DOES NOT GATE. Every fresh clone is in this state: the held-out
                # corpora and the private seed are gitignored by design. Nothing
                # in the code is wrong and re-running changes nothing -- so the
                # [GUARD-BROKEN] advice below ("RE-RUN FIRST") would be actively
                # misleading here. Still NOT an all-clear: say plainly that
                # immutability was not verified on this machine.
                print(f"  [ASSET-ABSENT] {r.arena_id} {r.version} ({r.status}): "
                      f"immutability NOT verified here — a local-only asset this "
                      f"machine does not hold is required to regenerate the task set. "
                      f"Not gated: a fresh clone is expected to be in this state.")
                print(f"      why: {r.reason or 'no reason recorded'}")
            elif r.verdict == "GUARD-BROKEN":
                # GATES, unlike UNCHECKABLE below. A fingerprint is recorded for
                # this version, so it WAS computable when it was baselined; not
                # being computable now is a guard that has stopped working on a
                # task set every published score for it was computed against.
                # Exiting 0 here is what let a real drift read as a clean run.
                print(f"  [GUARD-BROKEN] {r.arena_id} {r.version} ({r.status}): a "
                      f"generator_fingerprint is recorded, but it could NOT be "
                      f"recomputed, so drift was never compared.")
                print(f"      recorded: {r.expected}")
                print(f"      why     : {r.reason or 'no reason recorded'}")
                print(f"      This is NOT an all-clear and NOT the same as an unguarded "
                      f"task set. RE-RUN FIRST: a generation failure can be transient "
                      f"(a locked file, a busy machine), and a re-run that reports [OK] "
                      f"or [DRIFT] is the answer. If it persists, fix the generator or "
                      f"the environment -- do NOT re-baseline the fingerprint, which "
                      f"would replace the guard with whatever the broken state produces.")
                failed = True
            else:
                print(f"  [UNCHECKABLE] {r.arena_id} {r.version} ({r.status}): never "
                      f"fingerprinted AND cannot be generated in-process, so immutability "
                      f"is unguarded and unverifiable (not gated -- this was already true)")
                print(f"      why: {r.reason or 'no reason recorded'}")

    # ALWAYS ON. `runner._resolve_formula_id` stamps three distinct placeholders
    # when it cannot name the formula that produced a score. Keeping them distinct
    # was right and bought nothing on its own, because NOTHING READ THEM — an
    # honest label nobody consumes is a silent fallback with better manners.
    #
    # Starts from an empty set: measured 2026-08-27, 8784 records carry 0 instrument
    # blocks, so this is a no-op today and fires only on a real future failure.
    print("\n== scoring-formula provenance ==")
    formula_arenas = ([arenas_root() / args.arena] if args.arena
                      else [a["root"] for a in discover_arenas(arenas_root())])
    n_formula_problems = 0
    for adir in formula_arenas:
        recs = load_arena_records(adir, resolve_task_set_version(adir, args.task_set))
        if not recs:
            continue
        for prob in formula_provenance_problems(recs, arena_id=adir.name):
            print(f"  [FORMULA-PLACEHOLDER] {prob.arena_id}: {prob.n_records} record(s) "
                  f"carry `{prob.formula_id}`")
            print(f"      {prob.detail}")
            n_formula_problems += prob.n_records
            if prob.gates:
                failed = True
    if not n_formula_problems:
        # Say the all-clear only when records were actually inspected — an arena
        # with no records contributes nothing and must not read as a pass.
        print("  (no record names a placeholder instead of a formula)")

    # ALWAYS ON, not behind a flag. Both leaderboard engines group on player_id
    # alone, so records from two machines under one player_id publish as a single
    # row — one machine's label over the mean of both. Cheap (a pure pass over
    # already-loaded records) and silent by construction until someone actually
    # mixes machines, so it costs nothing to leave running.
    print("\n== one player, one machine ==")
    host_arenas = ([arenas_root() / args.arena] if args.arena
                   else [a["root"] for a in discover_arenas(arenas_root())])
    n_legacy = 0
    n_checked_arenas = 0
    for adir in host_arenas:
        records = load_arena_records(adir, resolve_task_set_version(adir, args.task_set))
        if not records:
            continue
        n_checked_arenas += 1
        rep = host_split_players(records, arena_id=adir.name)
        n_legacy += rep.n_unverifiable
        if rep.fingerprint_version_skew:
            # A recipe change, not a machine change. Digests either side are not
            # comparable, so this is reported and never gated.
            print(f"  [RECIPE] {adir.name}: {', '.join(rep.fingerprint_version_skew)} "
                  f"— records span more than one host-fingerprint version; digests "
                  f"either side of the bump are not comparable")
        if rep.mixed_with_unrecorded:
            # Reported, deliberately not gating: one KNOWN machine plus records
            # that predate the fingerprint. Probably the same box, but the old
            # records cannot say so, and pretending otherwise would be the false
            # all-clear this project keeps having to name rather than guess.
            print(f"  [MIXED] {adir.name}: {', '.join(rep.mixed_with_unrecorded)} "
                  f"— one known machine plus pre-fingerprint records (machine unverifiable; "
                  f"archive the old records if you need a clean comparison)")
        if not rep.ok:
            print(f"  [SPLIT] {adir.name}:")
            for p in rep.problems:
                print(f"    {p}")
            failed = True
    if n_legacy:
        # Not a defect and not an all-clear — the machine is genuinely unknown for
        # these. Named rather than folded into either bucket, same posture as the
        # held-out freshness case above.
        print(f"  [PRE-FINGERPRINT] {n_legacy} record(s) across {n_checked_arenas} arena(s) "
              f"predate provenance.host_fingerprint — machine unrecorded, not verifiable")

    return 1 if failed else 0


def _cmd_export_kaggle(args):
    # Imported lazily: the exporter pulls in the runner's arena-module loader,
    # and `framework --help` should not pay for that.
    from framework.export_kaggle import export_arena
    n = export_arena(args.arena, args.task_set, args.out, prompt_template=args.prompt)
    print(f"exported {n} public task(s) for {args.arena} to {args.out}")
    return 0


def _cmd_export_inspect(args):
    # Lazily imported for the same reason as export-kaggle.
    from framework.export_inspect import export_arena
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
    p_versions = sub_arenas.add_parser(
        "versions",
        help="Show an arena's task-set version ladder: status, rationale, comparability.")
    p_versions.add_argument("--arena", default=None, help="One arena (default: all).")
    p_versions.set_defaults(func=_cmd_arenas_versions)

    sub_players = sub.add_parser("players").add_subparsers(dest="sub", required=True)
    sub_players.add_parser("list").set_defaults(func=_cmd_players_list)

    p_run = sub.add_parser("run")
    p_run.add_argument("--arena", required=True)
    p_run.add_argument("--task-set", required=True)
    p_run.add_argument("--players", nargs="+", required=True)
    # DEFAULT 1, NOT 3. Measured 2026-08-28: 3,585 of the 3,605 published records
    # carry provenance.trials = 1 and NOT ONE carries any other value (the other
    # 20 predate the field). So a default of 3 matched nothing that has ever been
    # published, and it is a live trap rather than a preference: a deterministic
    # player collapses trials to 1 on its own, so a mixed run silently pairs
    # trials=1 tools against trials=3 models, and the ranking then partly measures
    # the run configuration. It cost a real run on 2026-08-28 -- the three Claude
    # players were relaunched after test_one_arena_and_task_set_never_mixes_trial_counts
    # caught "trials=1: escimate, statcheck; trials=3: claude-opus-4-8-stats".
    # Trials > 1 also narrowed the published CI by sqrt(3) in all three report
    # engines until 2026-08-19. Pass --trials N deliberately if you want variance.
    p_run.add_argument("--trials", type=int, default=1)
    # Default None = size each player's clock from its declared max_tokens
    # (framework/timeouts.py; 120 s for players that declare none). d-545c62.
    p_run.add_argument("--timeout", type=int, default=None)
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
    # Default None = sized from the declared max_tokens, 300 s floor. The old fixed
    # 300 was SHORTER than most run timeouts, so a retry re-timed-out (run_health).
    p_retry.add_argument("--timeout", type=int, default=None)
    p_retry.add_argument("--max-rounds", type=int, default=6,
                         help="Max retry rounds this invocation (default 6).")
    p_retry.add_argument("--include-truncated", action="store_true",
                         help="Also re-play tasks whose answer was cut off at OUR "
                              "output ceiling (finish_reason='length'). Such a record "
                              "carries a real-looking score computed from half an "
                              "answer and is indistinguishable from incapability. "
                              "ONLY useful once max_tokens has actually been raised "
                              "for the player -- otherwise the retry re-truncates, "
                              "which this detects and stops on.")
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
    # Default None = resolve per arena from arena.yaml#task_set_versions (newest).
    # A literal "v1" default silently audited the wrong version for any arena that
    # had moved on, and an arena whose runs/v1/ was empty simply vanished from the
    # report. Pass --task-set explicitly to audit a specific (e.g. archived) version.
    p_audit.add_argument("--task-set", default=None)
    p_audit.add_argument("--versions", action="store_true",
                         help="Check declared vs installed tool versions (Finding 1).")
    p_audit.add_argument("--symmetry", action="store_true",
                         help="Check players ran identical task sets (Finding 2).")
    p_audit.add_argument("--fresh", action="store_true",
                         help="Check each record's input_hash still matches the task the "
                              "generator emits today (catches scores left stale by an "
                              "in-place broaden, which symmetry cannot see).")
    p_audit.add_argument("--identity", action="store_true",
                         help="Check the DECLARED instrument on every registry entry: that "
                              "one instrument never covers two inference configurations, "
                              "that one configuration is never split across two names, and "
                              "that every name is slug-safe. Mirrored in "
                              "leaderboard-app/scripts/lib/identity.mjs, which runs the same "
                              "rules as a build gate.")
    p_audit.add_argument("--write-fingerprints", action="store_true",
                         help="Baseline `generator_fingerprint` into arena.yaml for every "
                              "FROZEN task-set version that lacks one. Never overwrites an "
                              "existing fingerprint — that would make the drift gate a "
                              "rubber stamp.")
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

    p_inspect = sub.add_parser(
        "export-inspect",
        help="Emit an Inspect AI task module for an arena's PUBLIC split "
             "(the format OpenBench and Epoch AI ingest).",
    )
    p_inspect.add_argument("--arena", required=True)
    p_inspect.add_argument("--task-set", default="v1")
    p_inspect.add_argument("--out", type=Path, default=Path("build/inspect"))
    p_inspect.add_argument("--prompt", type=Path, default=None,
                           help="Prompt template containing {{INPUT_TEXT}}.")
    p_inspect.set_defaults(func=_cmd_export_inspect)

    args = parser.parse_args(argv)
    if args.arenas_root is not None:
        os.environ[ARENAS_ROOT_ENV] = str(args.arenas_root)
    if args.registry is not None:
        os.environ[REGISTRY_PATH_ENV] = str(args.registry)
    # `audit` with no flags runs every check.
    if args.cmd == "audit" and not (args.versions or args.symmetry or args.fresh
                                    or args.identity):
        args.versions = args.symmetry = args.fresh = args.identity = True
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


# `python -m framework.cli …` used to import this module and exit 0 having run
# NOTHING — argparse never fired, no gate executed, and the shell saw success.
# On a project whose one-pager cites "framework audit exits 0" as evidence, a
# plausible invocation that silently no-ops is the exact false-green class
# LESSONS.md keeps recording. (Fable 5 cross-review, 2026-08-15: the reviewer hit
# it and nearly recorded the audit as vacuously passing.)
#
# `python -m framework …` (framework/__main__.py) and the `metasciencearena`
# console script remain the documented entry points; this makes the third form
# behave like them instead of lying.
if __name__ == "__main__":
    import sys

    sys.exit(_console_main())
