"""Tournament runner: tasks × players × trials → run records."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import logging
import os
import shlex
import sys
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator

from framework.discovery import load_arena
from framework.holdout import redact_held_out_record
from framework.player_adapter import build_adapter, resolve_rscript_binary
from framework.registry import load_registry
from framework.storage import RunRecordWriter

logger = logging.getLogger(__name__)


#: Entry-point group third-party adapter packages advertise themselves under.
#: A distribution adds, in its own pyproject.toml::
#:
#:     [project.entry-points."metasciencearena.adapters"]
#:     my_tool = "my_pkg.adapters"
#:
#: and every module in that package is imported here, self-registering its
#: adapter classes exactly as the in-repo ones do.
ADAPTER_ENTRY_POINT_GROUP = "metasciencearena.adapters"


def _import_adapter_package(pkg, label: str) -> None:
    """Import every non-test module in an adapter package.

    A single adapter module that fails to import (e.g. an optional third-party
    dep is absent) must NOT abort loading the others — it is logged at WARNING
    and skipped, so the tournament can still run every player whose adapter did
    load (DR-0015; previously the first bad import killed all registration).
    """
    import importlib
    import pkgutil
    for _, name, _ in pkgutil.iter_modules(pkg.__path__):
        if name.startswith("test_"):
            continue
        try:
            importlib.import_module(f"{pkg.__name__}.{name}")
        except Exception as exc:  # noqa: BLE001 - one bad adapter must not block the rest
            logger.warning("Failed to import adapter module %s.%s: %s", label, name, exc)


def _autoload_adapter_modules() -> None:
    """Load adapter classes from entry points, then from the in-repo package.

    WHY AN ENTRY POINT
    ------------------
    This module used to `import players.adapters` unconditionally — a library
    reaching up into the application that happens to sit above it in one
    checkout. Installed as a package that import simply fails, silently, and
    every player is then "unknown adapter". Entry points invert the dependency:
    the adapter provider declares itself and the framework discovers it, so a
    third party can register adapters without a `players/` directory existing.

    The in-repo `players.adapters` import is KEPT as a second source, because
    this repo's own adapters are not an installed distribution. It is attempted
    after the entry points so a published adapter package cannot be shadowed.
    """
    import importlib
    from importlib.metadata import entry_points

    loaded: set[str] = set()
    for ep in entry_points(group=ADAPTER_ENTRY_POINT_GROUP):
        try:
            pkg = ep.load()
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not block the rest
            logger.warning("Failed to load adapter entry point %r: %s", ep.name, exc)
            continue
        loaded.add(getattr(pkg, "__name__", ep.name))
        _import_adapter_package(pkg, ep.name)

    # In-repo adapters. Absent under a bare `pip install`, which is correct:
    # this repo's players are not part of the published package.
    try:
        import players.adapters as pkg
    except ImportError:
        return
    if pkg.__name__ not in loaded:
        _import_adapter_package(pkg, "players.adapters")


_autoload_adapter_modules()


def _import_arena_module(arena_dir: Path, module_name: str):
    """Import generator.py or scorer.py from an arena directory by absolute path."""
    path = arena_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_arena_{module_name}_{arena_dir.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hash_input(envelope: dict) -> str:
    payload = json.dumps(envelope, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_SECRET_MARKERS = ("api_key", "apikey", "api-key", "secret", "token", "sk-")


def _sanitize_command(parts) -> str:
    """Render an adapter command without secrets.

    Redacts a token when it (a) itself contains a secret marker, OR (b) follows a
    flag whose name contains a secret marker (e.g. `--api-key REALKEY` -> the
    value token is redacted even though it carries no marker of its own).
    """
    if not parts:
        return ""
    safe = []
    prev_is_secret_flag = False
    for tok in parts:
        t = str(tok)
        tl = t.lower()
        if prev_is_secret_flag or any(k in tl for k in _SECRET_MARKERS):
            safe.append("<redacted>")
        else:
            safe.append(t)
        # A flag like --api-key / --secret / -t/--token marks the NEXT token as a
        # value to redact. Only flag-shaped tokens (leading dash) arm this.
        prev_is_secret_flag = t.startswith("-") and any(k in tl for k in _SECRET_MARKERS)
    return " ".join(shlex.quote(s) for s in safe)


def _adapter_command(entry: dict) -> str:
    """Best-effort exact invocation string for the test card, sanitized.

    player_id is appended as a comment tag so each record is unambiguously
    traceable to the player that produced it even when multiple players share
    the same script path.
    """
    pid = entry.get("player_id", "")
    if entry.get("cli_command"):
        base = _sanitize_command(entry["cli_command"])
        return f"{base}  # {pid}" if pid else base
    if entry.get("r_script"):
        base = _sanitize_command([resolve_rscript_binary(entry.get("rscript_binary")), entry["r_script"]])
        return f"{base}  # {pid}" if pid else base
    if entry.get("endpoint"):
        return f"{entry.get('adapter_class', 'HttpAdapter')} -> {entry['endpoint']}  # {pid}" if pid else f"{entry.get('adapter_class', 'HttpAdapter')} -> {entry['endpoint']}"
    return entry.get("adapter_class", "")


def run_tournament(
    *,
    arena_dir: Path,
    task_set_version: str,
    registry_path: Path,
    player_ids: list[str],
    output_path: Path,
    trials: int = 3,
    timeout_s: int = 120,
    seed: int = 0,
    public_only: bool = False,
    held_out_only: bool = False,
    max_tasks: int | None = None,
    split: str = "revealed",
    overwrite: bool = False,
    only_tasks: set[str] | None = None,
) -> int:
    """Run a tournament. Returns the number of run records written.

    `split` ("revealed" | "private") selects which benchmark suite to generate.
    It is forwarded to the arena generator only when the generator's signature
    accepts a `split` parameter, so legacy single-suite arenas keep working
    unchanged. The recorded `split` is taken from each envelope (the source of
    truth), and private-split tasks (visibility=held_out) keep findings redaction.

    `only_tasks`, when given, restricts the run to exactly those task_ids (used by
    the rate-limit-aware retry loop to re-play just the tasks a rate-limited /
    slow provider failed on). It composes with `overwrite`: the caller runs the
    subset into a temp file and merges it back, so the retry never clobbers the
    already-good records for other tasks.
    """
    arena = load_arena(arena_dir)
    output_validator = Draft202012Validator(json.loads(arena["output_schema_path"].read_text(encoding="utf-8")))
    generator_mod = _import_arena_module(arena_dir, "generator")
    scorer_mod = _import_arena_module(arena_dir, "scorer")

    gen_kwargs = {}
    try:
        if "split" in inspect.signature(generator_mod.generate).parameters:
            gen_kwargs["split"] = split
    except (TypeError, ValueError):
        pass

    all_players = {p["player_id"]: p for p in load_registry(registry_path)}
    selected = []
    for pid in player_ids:
        if pid not in all_players:
            raise ValueError(f"Unknown player_id: {pid}")
        selected.append(all_players[pid])

    adapters = [(entry, build_adapter(entry)) for entry in selected]
    for _, adapter in adapters:
        adapter.prepare()

    # Refuse held-out egress to cloud players BEFORE the first task runs — after
    # is too late, the data is already gone. `public_only` is the only thing that
    # guarantees no held-out envelope reaches a player.
    assert_heldout_egress_allowed(selected, will_play_held_out=not public_only)

    written = 0
    tasks_seen = 0
    # Per-(player, outcome) tally for the end-of-run summary. Outcome is one of
    # "ok" | "timeout" | "error" — derived from the persisted score.breakdown so
    # an operator can tell a healthy run from a half-failed one at a glance
    # (DR-0015), instead of having to parse the JSONL afterwards.
    outcomes: Counter = Counter()
    logger.info("Running %s (split=%s, seed=%s) with %d player(s)",
                arena["arena_id"], split, seed, len(adapters))
    try:
        with RunRecordWriter(output_path, overwrite=overwrite) as writer:
            for envelope in generator_mod.generate(task_set_version, seed, **gen_kwargs):
                if public_only and envelope.get("visibility") != "public":
                    continue
                # held_out_only is the mirror of public_only: it keeps ONLY the
                # held-out (private) tasks. Generators that emit BOTH visibilities
                # for every split (e.g. the PDF arenas) need this so a `--split
                # private` run produces a clean held-out-only file that does not
                # overlap (and double-count) the public synthetic tasks already
                # in the revealed file.
                if held_out_only and envelope.get("visibility", "held_out") != "held_out":
                    continue
                if only_tasks is not None and envelope["task_id"] not in only_tasks:
                    continue
                if max_tasks is not None and tasks_seen >= max_tasks:
                    break
                tasks_seen += 1
                gt = generator_mod.ground_truth(envelope["task_id"])
                for entry, adapter in adapters:
                    n_trials = 1 if entry["deterministic"] else max(1, trials)
                    for _ in range(n_trials):
                        record = _play_one(envelope, gt, entry, adapter, scorer_mod, output_validator, timeout_s,
                                           arena["arena_id"], trials=n_trials, seed=seed, split=split)
                        outcome = _record_outcome(record)
                        outcomes[(entry["player_id"], outcome)] += 1
                        if outcome != "ok":
                            err = (record.get("score", {}).get("breakdown") or {}).get("error", "")
                            logger.warning("%s on %s/%s: %s", outcome, entry["player_id"],
                                            envelope["task_id"], err)
                        writer.append(record)
                        written += 1
    finally:
        for _, adapter in adapters:
            try:
                adapter.cleanup()
            except Exception as exc:  # noqa: BLE001 - cleanup must never mask the run result
                logger.warning("Adapter cleanup failed for %s: %s",
                               getattr(adapter, "player_id", adapter.__class__.__name__), exc)

    _log_run_summary(arena["arena_id"], outcomes, written)
    return written


# A CLI player wrapped in coreutils `timeout` (see the free-tier entries in
# players/registry.yaml) surfaces a wall-clock kill as exit 124 — SIGTERM'd — or
# 137 (128+9) when `-k` escalates to SIGKILL. Neither string contains "Timeout",
# so both used to be summarised as generic errors, hiding the one thing the
# operator needs to know: the model did not fail, the wall was too tight.
_TIMEOUT_MARKERS = ("timeout", "exited 124", "exited 137")


# --------------------------------------------------------------------------- #
# Held-out egress gate.
#
# Sending a held-out task to a third-party provider is IRREVERSIBLE, so it must
# never happen as a side effect of a command that does not say so. On 2026-08-04
# a `--split revealed` run of a cloud LLM player transmitted 9 held-out real
# papers before anyone noticed, purely because --split selects the seed and does
# not filter visibility.
#
# Two independent guards now stand in the way: the CLI makes `--split revealed`
# imply --public-only, and this gate refuses the combination outright unless the
# operator sets SCIENCEARENA_ALLOW_HELDOUT_EGRESS=1. Belt and braces on purpose —
# the CLI default protects the common case, this protects every OTHER caller
# (scripts, notebooks, the retry queue, a future API) that bypasses the CLI.
# --------------------------------------------------------------------------- #
_EGRESS_ALLOW_ENV = "SCIENCEARENA_ALLOW_HELDOUT_EGRESS"

# Adapters that hand the task to someone else's machine. Matched by CLASS NAME so
# a new adapter is opt-IN to egress rather than silently exempt: anything whose
# name starts with one of these prefixes counts.
_CLOUD_ADAPTER_PREFIXES = ("LlmCli", "LlmPdf", "SubprocessCli", "Http",
                           "RegcheckShim", "WatsonShim", "Scimeto")

# CLI players that run entirely locally despite using SubprocessCliAdapter.
# Keep this list short and explicit; when in doubt, treat a player as cloud.
_LOCAL_SUBPROCESS_PLAYERS: frozenset[str] = frozenset()


def is_cloud_player(entry: dict) -> bool:
    """True when running this player transmits the task off-machine.

    Deliberately conservative: unknown adapters are treated as cloud, because the
    cost of a false positive is one env var and the cost of a false negative is
    irreversible disclosure.
    """
    if entry.get("player_id") in _LOCAL_SUBPROCESS_PLAYERS:
        return False
    cls = str(entry.get("adapter_class", ""))
    return cls.startswith(_CLOUD_ADAPTER_PREFIXES)


def assert_heldout_egress_allowed(entries: list[dict], *, will_play_held_out: bool) -> None:
    """Refuse to send held-out tasks to a third-party player without consent.

    Raises RuntimeError naming the offending players and the exact opt-in, so the
    operator makes the disclosure decision explicitly rather than discovering it
    in a run record afterwards.
    """
    if not will_play_held_out:
        return
    cloud = sorted({e["player_id"] for e in entries if is_cloud_player(e)})
    if not cloud:
        return
    if os.environ.get(_EGRESS_ALLOW_ENV, "").strip() in {"1", "true", "TRUE", "yes"}:
        logger.warning(
            "HELD-OUT EGRESS ALLOWED by %s: sending held-out task inputs to "
            "third-party provider(s) via %s", _EGRESS_ALLOW_ENV, ", ".join(cloud))
        return
    raise RuntimeError(
        "refusing to send HELD-OUT tasks to third-party player(s): "
        + ", ".join(cloud)
        + ".\nThis run would transmit held-out task inputs (real papers, some "
          "copyrighted — see DATA_HANDLING.md) off-machine, and that cannot be "
          "undone.\nIf that is what you intend, re-run with "
        + f"{_EGRESS_ALLOW_ENV}=1 set in the environment.\n"
          "To keep the run local instead, use --public-only, or run only "
          "local tool players (docpluck, liteparse, GROBID, statcheck, the R tools)."
    )


def _record_outcome(record: dict) -> str:
    """Classify a persisted record as 'ok' | 'timeout' | 'error' for the summary."""
    breakdown = (record.get("score") or {}).get("breakdown") or {}
    err = breakdown.get("error") if isinstance(breakdown, dict) else None
    if not err:
        return "ok"
    low = str(err).lower()
    return "timeout" if any(m in low for m in _TIMEOUT_MARKERS) else "error"


def _log_run_summary(arena_id: str, outcomes: "Counter", written: int) -> None:
    """Emit one end-of-run line per player: n_ok / n_error / n_timeout (DR-0015)."""
    players = sorted({pid for (pid, _) in outcomes})
    total_bad = sum(c for (_, o), c in outcomes.items() if o != "ok")
    level = logging.WARNING if total_bad else logging.INFO
    logger.log(level, "Run complete: %s — %d records across %d player(s), %d non-ok.",
               arena_id, written, len(players), total_bad)
    for pid in players:
        n_ok = outcomes.get((pid, "ok"), 0)
        n_err = outcomes.get((pid, "error"), 0)
        n_to = outcomes.get((pid, "timeout"), 0)
        logger.log(logging.WARNING if (n_err or n_to) else logging.INFO,
                   "  %s: ok=%d error=%d timeout=%d", pid, n_ok, n_err, n_to)


_HELD_OUT_FINDING_KEEP = {"category", "count"}


def _redact_findings_for_held_out(findings):
    """Strip content-bearing fields from findings before persistence.

    Held-out tasks: only `category` and (if present) `count` survive. This is
    enforced at write time, not at render time, so a leak of the JSONL file
    cannot reveal held-out gold. See spec §4.4 + Decisions Log D4.
    """
    if not isinstance(findings, list):
        return findings
    redacted = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        kept = {k: v for k, v in f.items() if k in _HELD_OUT_FINDING_KEEP}
        if "category" in kept:
            redacted.append(kept)
    return redacted


def _play_one(envelope, ground_truth, entry, adapter, scorer_mod, output_validator, timeout_s,
              arena_id, *, trials, seed, split) -> dict:
    started = time.monotonic()
    visibility = envelope.get("visibility", "held_out")
    try:
        output = adapter.play_task(envelope, timeout_s=timeout_s)
        out_errors = sorted(output_validator.iter_errors(output), key=lambda e: e.path)
        if out_errors:
            score = {"primary": 0.0, "breakdown": {"error": "output_schema_violation: " + "; ".join(e.message for e in out_errors)}}
            output_to_store = {}
        else:
            score = scorer_mod.score(output, ground_truth)
            output_to_store = output
    except Exception as exc:
        score = {"primary": 0.0, "breakdown": {"error": f"{type(exc).__name__}: {exc}"}}
        output_to_store = {}
        # Persisted record carries only the short `Type: message`; the full stack
        # goes to the DEBUG log so an operator can diagnose without bloating the
        # JSONL (the previously-unused `traceback` import, DR-0015).
        logger.debug("adapter %s failed on %s:\n%s", entry.get("player_id"),
                     envelope.get("task_id"), traceback.format_exc())
    latency_ms = int((time.monotonic() - started) * 1000)

    # Runtime tool-version detection: stamp the version the score was ACTUALLY
    # produced with (e.g. docpluck.__version__), independent of the static label
    # in registry.yaml. Best-effort — must never fail a task.
    try:
        resolved_tool_version = adapter.resolved_tool_version()
    except Exception:
        resolved_tool_version = None

    # Held-out redaction: strip findings content before persistence.
    if visibility == "held_out" and isinstance(score, dict) and "findings" in score:
        score = dict(score)  # shallow copy so we don't mutate the scorer's return value
        score["findings"] = _redact_findings_for_held_out(score["findings"])

    record = {
        "run_id": str(uuid.uuid4()),
        "arena_id": arena_id,
        "task_set_version": envelope["task_set_version"],
        "task_id": envelope["task_id"],
        "player_id": entry["player_id"],
        "player_version": entry["player_version"],
        "resolved_tool_version": resolved_tool_version,
        "player_type": entry["player_type"],
        "input_hash": _hash_input(envelope),
        "output": output_to_store,
        "score": score,
        "task_visibility": visibility,
        "timestamp_utc": _utc_now_iso(),
        "latency_ms": latency_ms,
    }
    record["provenance"] = {
        "tested_at_utc": _utc_now_iso(),
        "host": os.environ.get("SCIENCEARENA_HOST", "win11-local"),
        "adapter_class": entry.get("adapter_class", ""),
        "command": _adapter_command(entry),
        "tool_version_detail": entry["player_version"],
        "resolved_tool_version": resolved_tool_version,
        "trials": trials,
        "seed": seed,
        "split": envelope.get("split", split),
    }
    # Record which benchmark suite this task belongs to (source of truth is the
    # envelope). Omitted for legacy single-suite arenas that don't tag `split`.
    split_val = envelope.get("split")
    if split_val is not None:
        record["split"] = split_val

    # Held-out contamination boundary (DR-0007): for a held-out task the run
    # record is written WITHOUT the player output (reconstructs the gold doc),
    # the input_hash (membership oracle), or the score.breakdown (per-task gold
    # metadata) — score.primary + the already-stripped {category,count} findings
    # survive so aggregate ranking and the category-level report still work.
    # Enforced at write time so a leak of the tracked JSONL reveals no gold.
    return redact_held_out_record(record)
