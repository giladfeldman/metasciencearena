"""Tournament runner: tasks × players × trials → run records."""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import ipaddress
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
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator

from framework.discovery import load_arena
from framework.holdout import redact_held_out_record
from framework.paths import schema_path
from framework.player_adapter import build_adapter, resolve_rscript_binary
from framework.registry import load_registry
from framework.storage import RunRecordWriter
from framework import hermetic

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
    # See framework/cli.py: a private seed must never reach a log line.
    logger.info("Running %s (split=%s, seed=%s) with %d player(s)",
                arena["arena_id"], split,
                seed if split == "revealed" else "<redacted:private>",
                len(adapters))
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
                # Validate the TASK, not just the player's answer. Until 2026-08-12
                # only the output was checked, so a generator could hand players a
                # malformed envelope and every resulting score would look fine.
                _validate_envelope(envelope, arena["arena_id"])
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
                           "RegcheckShim", "WatsonShim", "Scimeto",
                           "OpenAIChatCompletions", "AntigravityCli",
                           "ModelProxy")

# Adapter classes REVIEWED and confirmed to keep the task on this machine when no
# endpoint is declared. This is an allowlist, not a fallback: any registered class
# absent from it AND from _CLOUD_ADAPTER_PREFIXES is treated as cloud, so adding an
# adapter forces a deliberate decision instead of inheriting a silent default.
# Pinned by test_every_registered_adapter_class_is_explicitly_classified.
#
# A declared `endpoint` still outranks this list — the Grobid adapters are here
# because their no-endpoint default is `http://localhost:8070`, but an entry that
# points one at a remote host is classified from that endpoint and stays cloud.
_LOCAL_ADAPTER_CLASSES: frozenset[str] = frozenset({
    # pure-Python / library parsers
    "DocpluckLibraryAdapter", "DocpluckSectionsAdapter", "DocpluckTablesAdapter",
    "LiteparseTextAdapter", "LiteparseSectionsHeuristicAdapter",
    "LiteparseTablesHeuristicAdapter",
    # Docling (2026-08-19). REVIEWED, not assumed. Evidence: the adapters declare
    # no `endpoint`; `_docling_common.build_converter` hard-codes
    # `enable_remote_services=False` and `do_picture_description=False` and does
    # not read either from registry.yaml, so the one switch that could reach a
    # remote API cannot be flipped from config; model weights are local and
    # pre-fetched. Asserted by players/adapters/tests/test_docling_common.py.
    # A remote-VLM Docling variant, if ever added, MUST declare an `endpoint:` so
    # the gate classifies it from the URL (the ModelProxyAdapter precedent) --
    # it must not reuse these class names.
    "DoclingTablesAdapter", "DoclingTextAdapter",
    # local subprocess tools
    "AnystyleReferencesAdapter", "CermineReferencesAdapter",
    "PdftotextSubprocessAdapter", "RCliAdapter",
    # local GROBID server (endpoint, when declared, decides instead)
    "GrobidTextAdapter", "GrobidSectionsAdapter", "GrobidTablesAdapter",
    "GrobidReferencesAdapter", "GrobidCitationsAdapter",
    # no I/O at all
    "FredFridaOnlyAdapter", "XlatFixtureAdapter",
    "StubPassAdapter", "StubFailAdapter",
})

# CLI players that run entirely locally despite using SubprocessCliAdapter.
# Keep this list short and explicit; when in doubt, treat a player as cloud.
_LOCAL_SUBPROCESS_PLAYERS: frozenset[str] = frozenset()


def _is_loopback_endpoint(endpoint: str) -> bool:
    """True when `endpoint`'s host is unambiguously this machine.

    Parsed with urlsplit and matched on the HOST ONLY — never a substring test.
    `http://127.0.0.1.attacker.com/x` and `http://evil.com#127.0.0.1` both contain
    a loopback-looking string and are both remote; treating either as local would
    turn this gate into a hole.
    """
    try:
        host = urlsplit(endpoint).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_cloud_player(entry: dict) -> bool:
    """True when running this player transmits the task off-machine.

    Deliberately conservative: unknown adapters are treated as cloud, because the
    cost of a false positive is one env var and the cost of a false negative is
    irreversible disclosure.

    A DECLARED ENDPOINT DECIDES, whatever the adapter class is called. The endpoint
    is direct evidence of where the bytes go; the class name is a naming convention,
    and this function used to trust the convention alone.

    That cut both ways, and both showed up on 2026-08-09:

    * False positive — `escimate` is an `HttpAdapter` on `http://127.0.0.1:9422`
      started by a local `start_command`. Nothing leaves the machine, yet the gate
      refused every private-split run it appeared in, blocking the only
      non-statcheck player the stats-extraction-v1 private split has ever had.
    * False NEGATIVE, the dangerous one — every adapter class outside
      `_CLOUD_ADAPTER_PREFIXES` returned False without the endpoint ever being
      looked at, flatly contradicting the "unknown adapters are treated as cloud"
      promise below. The five `Grobid*Adapter`s each take
      `endpoint: str = "http://localhost:8070"` and match no prefix, so pointing one
      at a public GROBID server — they exist — would have shipped held-out real
      papers, including copyrighted APA PDFs, off-machine with no gate and no
      opt-in. Dormant only because every shipped entry uses the localhost default.

    With no endpoint there is no evidence, so an adapter is cloud unless its class
    was reviewed onto `_LOCAL_ADAPTER_CLASSES`. That sentence used to be a comment
    the code contradicted: the fallback was `cls.startswith(_CLOUD_ADAPTER_PREFIXES)`,
    which makes an UNRECOGNISED class local — egress opt-OUT for anything nobody
    remembered to classify. `ModelProxyAdapter` was the live instance (2026-08-12):
    it POSTs every task to the Kaggle proxy, declares no endpoint, matched no
    prefix, and so bypassed this gate entirely. Dormant only because its registry
    entries were still commented out.
    """
    if entry.get("player_id") in _LOCAL_SUBPROCESS_PLAYERS:
        return False
    endpoint = entry.get("endpoint")
    if endpoint:
        return not _is_loopback_endpoint(str(endpoint))
    cls = str(entry.get("adapter_class", ""))
    if cls.startswith(_CLOUD_ADAPTER_PREFIXES):
        return True
    return cls not in _LOCAL_ADAPTER_CLASSES


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


@functools.lru_cache(maxsize=1)
def _envelope_validator() -> Draft202012Validator:
    """The task-envelope contract, loaded once.

    Same gap as the findings schema: `task_envelope.schema.json` declares
    `additionalProperties: false` and five required fields, and until 2026-08-12
    nothing loaded it — the runner validated only the player's OUTPUT, never the
    task it handed over. A generator emitting a malformed or misspelled envelope
    key would have produced records that scored fine and meant nothing.

    All 22 arenas' generated envelopes were verified against it before this was
    switched on (the PDF arenas need PYTHONPATH=repo root to import at all, which
    is why an earlier partial check reported a false all-clear on 17 of 22).
    """
    return Draft202012Validator(
        json.loads(schema_path("task_envelope.schema.json").read_text(encoding="utf-8"))
    )


def _validate_envelope(envelope: dict, arena_id: str) -> None:
    """Fail loudly when a generator emits an envelope that breaks the contract."""
    errors = sorted(_envelope_validator().iter_errors(envelope), key=lambda e: list(e.path))
    if errors:
        detail = "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:3])
        raise ValueError(
            f"task_envelope_schema_violation in {arena_id}/"
            f"{envelope.get('task_id', '<no task_id>')}: {detail}"
        )


@functools.lru_cache(maxsize=1)
def _findings_validator() -> Draft202012Validator:
    """The findings contract, loaded once.

    `contract/schemas/findings.schema.json` existed since the contract was written
    but was named only inside a `description` string on run_record.schema.json,
    whose `findings.items` is a bare `{"type": "object"}`. So nothing ever loaded
    it, and it silently became documentation rather than a contract: one arena
    emitted an undeclared `detail` field (which then became the ONLY field the
    task page rendered), and another emitted a numeric `evidence` against a
    string-typed declaration. Both were reconciled 2026-08-12 and all 2599 existing
    findings arrays validate, so this can now be enforced without a migration.
    """
    return Draft202012Validator(
        json.loads(schema_path("findings.schema.json").read_text(encoding="utf-8"))
    )


def _validate_findings(score: dict, arena_id: str, task_id: str | None) -> None:
    """Fail loudly when a scorer emits findings that break the contract.

    Raised rather than logged: a malformed finding is a scorer defect, and the
    whole point of findings is that a reader can trust what they say went wrong.
    The caller turns this into an errored record with a named cause.
    """
    findings = (score or {}).get("findings")
    if findings is None:
        return
    errors = sorted(_findings_validator().iter_errors(findings), key=lambda e: list(e.path))
    if errors:
        detail = "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:3])
        raise ValueError(
            f"findings_schema_violation in {arena_id}/{task_id}: {detail}"
        )



#: Bumped whenever the run-record shape changes. Without it, a reader three years
#: from now cannot tell a field's ABSENCE ("that run predates the field") from its
#: omission ("not applicable to that run") — Codex and Sonnet both flagged this
#: as the first thing a replication attempt would demand.
RECORD_SCHEMA_VERSION = 3


@functools.lru_cache(maxsize=1)
def _code_version() -> str | None:
    """git SHA of THIS repo, so a score is attributable to our own code.

    `tool_version_detail` pins the third-party tool; nothing pinned the framework
    and scorer that produced the number. After a scoring fix lands, an old record
    is otherwise indistinguishable from a new one.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=15,
                             cwd=Path(__file__).resolve().parents[1])
        sha = out.stdout.strip()
        return sha or None
    except Exception:
        return None


@functools.lru_cache(maxsize=64)
def _prompt_template_sha(path: str) -> str | None:
    """Hash the prompt template actually used.

    `task_id` pins the TASK; it does not pin the instructions wrapped around it,
    and those get edited. Two records with the same task_id and different template
    hashes are not comparable, and today nothing would show that.

    LINE ENDINGS ARE NORMALISED before hashing, and that is not cosmetic. Until
    2026-08-19 this hashed the raw working-tree bytes, so on Windows (autocrlf)
    it recorded the CRLF hash while the committed blob - what CI, the public
    mirror and every other reader checks out - is LF. Measured on
    `power_reporting.txt`: published records claim `c02b588da306ed86`, the file
    in git hashes to `3d89484f4959b236`, and CI reported every published value as
    naming a template that does not exist.

    A provenance hash that only reproduces on the machine that wrote it is not
    provenance. Normalising makes the value identify the template's CONTENT,
    which is what the field is supposed to pin, on any platform.
    """
    try:
        raw = Path(path).read_bytes()
    except Exception:
        return None
    return hashlib.sha256(normalise_newlines(raw)).hexdigest()[:16]


def normalise_newlines(raw: bytes) -> bytes:
    """CRLF -> LF, so a hash of file content is identical on every platform."""
    return raw.replace(b"\r\n", b"\n")
def _build_response_meta(adapter, entry: dict, visibility: str) -> dict | None:
    """Provider metadata that says whether a score can be TRUSTED.

    Both review models independently ranked `finish_reason` first: a completion cut
    off at the token limit fails JSON parsing, scores 0.0, and is indistinguishable
    from genuine incapability — publishing OUR max_tokens as THEIR failure. Sonnet
    rated `served_model` equally or more serious, because a silently rerouted model
    makes the whole record evidence about the wrong artifact.

    HELD-OUT: everything here is safe to keep EXCEPT `provider_request_id`. It
    leaks no content itself, but it is a pointer into the provider's own logs,
    which do retain the full document — so anyone with provider log access could
    correlate a held-out record to its source paper (Sonnet). Stripped there.
    """
    try:
        meta = adapter.last_response_meta()
    except Exception:
        meta = None
    if not meta:
        return None
    meta = dict(meta)

    # Intent vs evidence, never merged: `player_version` is what we ASKED for,
    # `served_model` is what the provider says it actually ran. This project has
    # published a version it was not running; a queryable flag makes that drift
    # visible across every record instead of needing a manual diff.
    served = meta.get("served_model")
    asked = (entry.get("openai_model") or entry.get("player_version") or "").strip()
    if served and asked:
        meta["model_mismatch"] = served.strip() != asked

    if visibility == "held_out":
        meta.pop("provider_request_id", None)
    return meta or None


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
            _validate_findings(score, arena_id, envelope.get("task_id"))
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

    # Token usage, when the player is a metered API. Best-effort like the version
    # probe: telemetry must never fail a task.
    try:
        usage = adapter.last_usage()
    except Exception:
        usage = None
    # HELD-OUT: strip it. `prompt_tokens` is a near-exact proxy for the length of
    # the input document, and the held-out corpus is real (sometimes copyrighted)
    # papers whose `input_hash` this project already redacts on the same
    # reasoning. `latency_ms` survives redaction and is also size-correlated, but
    # it is noisy enough to carry little; a token count is not. Cost for held-out
    # runs can be recovered in aggregate later if it is ever needed.
    if visibility == "held_out":
        usage = None

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
    if usage:
        record["usage"] = usage
    record["schema_version"] = RECORD_SCHEMA_VERSION
    response_meta = _build_response_meta(adapter, entry, visibility)
    if response_meta:
        record["response_meta"] = response_meta
    record["provenance"] = {
        "tested_at_utc": _utc_now_iso(),
        "host": os.environ.get("SCIENCEARENA_HOST", "win11-local"),
        "adapter_class": entry.get("adapter_class", ""),
        "command": _adapter_command(entry),
        "tool_version_detail": entry["player_version"],
        # Our OWN code, not just the third party's. A scoring fix makes old and new
        # records incomparable and nothing recorded which side a number came from.
        "code_version": _code_version(),
        "prompt_template_sha256": (
            _prompt_template_sha(entry["prompt_template_path"])
            if entry.get("prompt_template_path") else None
        ),
        # The direct companion to finish_reason: without the limit we sent, nobody
        # can confirm whether a truncation was our configuration error.
        "request_temperature": entry.get("openai_temperature"),
        "request_timeout_s": timeout_s,
        "resolved_tool_version": resolved_tool_version,
        "trials": trials,
        "seed": seed,
        "split": envelope.get("split", split),
        # HOW THE PLAYER WAS CONTAINED (CC6). An agentic CLI player is not the
        # model — it is `<model> via <cli>@<version>`, a different instrument
        # whose harness auto-updates underneath us. This block records the CLI
        # version, the tool allowlist and a fingerprint of the containment, so a
        # score can be attributed to the thing that actually produced it and a
        # containment change mints a new instrument instead of silently
        # appending to an old one.
        #
        # `None` for non-CLI players (HTTP/API, R, library) — they have no
        # ambient environment to inherit. Its ABSENCE on a CLI record is the
        # signal that the record predates containment; see
        # `hermetic.containment_state`. Records are never backfilled.
        "containment": hermetic.record_containment(entry.get("cli_command")),
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
