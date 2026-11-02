"""One recorder for every subscription-CLI model call (``claude``, ``codex``, and the launch
facts of any other agentic CLI).

Why this exists (measured 2026-11-01, docs/design/2026-11-run-provenance-completeness.md):

* no CLI player ever passed an effort level, and nothing recorded which one a call used;
* Claude Code ignores a misspelled ``--effort`` with only a stderr warning, and drops
  ``--effort`` entirely for models without effort control (Haiku 4.5);
* every player launched from the desktop app inherited ~120 of its environment variables,
  including the owner's e-mail, organisation id, a retry count of 100 and a compaction switch;
* Codex injected a 21,000-character block of the owner's local skill descriptions;
* the CLIs' own session logs, the only record of what was actually sent, are deleted after 30
  days (Claude Code) or compressed (Codex).

So every call goes through two steps:

``prepare(argv, effort=..., purpose=...)``
    validates the effort against what the CLI accepts (Codex: its own model catalog), appends
    it, gives a Claude call a session id of our choosing (``--session-id``), hardens the argv
    and sets an ALLOWLISTED environment (``framework.hermetic``), and freezes the launch spec
    BEFORE the process starts.

``finish(plan, returncode=..., stdout=..., stderr=...)``
    reads the session log the CLI wrote (``framework.cli_session_log``, the same reader the
    recovery of past runs uses, so past and future can never disagree), verifies that the
    effort the CLI sent is the effort we asked for and that only the requested model was
    billed, copies the log verbatim to article custody, appends the call to the custody call
    ledger, and returns the block that goes on the run record (``model_call``).

A call whose effort is not stated is refused (``UnrecordableCall``), never run with a default.
A model with no effort control is declared with ``effort=NOT_APPLICABLE`` and a reason; that is
an answer, not a gap.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from framework import cli_session_log as L
from framework import hermetic

RECORDER_VERSION = 1

#: Declares that the requested model has no effort control in this CLI. Needs a reason.
NOT_APPLICABLE = "not-applicable"

#: Claude Code 2.1.316 lists these in its own warning for an unknown value (measured 2026-11-01).
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")

#: Fallback when Codex's model catalog cannot be read; the catalog is preferred (per model).
CODEX_EFFORTS_FALLBACK = ("low", "medium", "high", "xhigh", "max")

SESSION_LOG_BINARIES = ("claude", "codex")


class UnrecordableCall(RuntimeError):
    """The call would run with a setting nobody could later state. Refused before launch."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _flag_value(argv: Sequence[str], *names: str) -> str | None:
    for i, tok in enumerate(argv):
        for n in names:
            if tok == n and i + 1 < len(argv):
                return str(argv[i + 1])
            if tok.startswith(n + "="):
                return tok.split("=", 1)[1]
    return None


def _codex_config_value(argv: Sequence[str], key: str) -> str | None:
    """Value of ``-c key=value`` (quotes stripped) in a Codex argv."""
    for i, tok in enumerate(argv):
        if tok in ("-c", "--config") and i + 1 < len(argv):
            k, _, v = str(argv[i + 1]).partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    return None


def codex_supported_efforts(model: str | None) -> tuple[tuple[str, ...], str | None, str]:
    """(supported levels, catalog default, source) for a Codex model, from Codex's own catalog."""
    try:
        with open(L.codex_home() / "models_cache.json", encoding="utf-8") as fh:
            cat = json.load(fh)
        for m in cat.get("models") or []:
            if isinstance(m, dict) and m.get("slug") == model:
                levels = tuple(x.get("effort") for x in m.get("supported_reasoning_levels") or []
                               if isinstance(x, dict) and x.get("effort"))
                if levels:
                    return levels, m.get("default_reasoning_level"), "codex models_cache.json"
    except (OSError, ValueError):
        pass
    return CODEX_EFFORTS_FALLBACK, None, "fallback list (catalog unreadable or model not in it)"


def _after(argv: list[str], binary: str) -> int:
    """Insertion point for our flags: right after the binary (and Codex's ``exec`` subcommand),
    so a wrapper such as ``timeout`` in front keeps working."""
    for i, tok in enumerate(argv):
        if hermetic.normalise_binary(tok) == binary:
            return i + 2 if binary == "codex" and i + 1 < len(argv) and argv[i + 1] == "exec" else i + 1
    raise UnrecordableCall(f"binary {binary!r} not found in argv")


def resolve_effort(binary: str | None, argv: Sequence[str], effort: str | None, effort_reason: str | None,
                   purpose: str) -> tuple[str | None, str | None, str | None]:
    """``(effort, effort already in argv, what it was validated against)``, or ``UnrecordableCall``.

    Unstated, misspelt, unsupported, contradicted, or NOT_APPLICABLE without a reason: refused.
    """
    model = _flag_value(argv, "--model", "-m")
    if binary == "claude":
        in_argv = _flag_value(argv, "--effort")
    elif binary == "codex":
        in_argv = _codex_config_value(argv, "model_reasoning_effort")
    else:
        in_argv = None
    if effort is None and in_argv is None and binary in SESSION_LOG_BINARIES:
        raise UnrecordableCall(
            f"{binary} call for {purpose!r} states no effort; pass effort=<level> or "
            f"effort=NOT_APPLICABLE with a reason. A default nobody wrote down is not reproducible.")
    if effort is not None and in_argv is not None and effort != in_argv:
        raise UnrecordableCall(f"effort {effort!r} contradicts {in_argv!r} already in argv")
    effort = effort if effort is not None else in_argv
    if effort == NOT_APPLICABLE and not effort_reason:
        raise UnrecordableCall("effort=NOT_APPLICABLE needs effort_reason")
    source = None
    if binary == "claude" and effort != NOT_APPLICABLE:
        if effort not in CLAUDE_EFFORTS:
            # Claude Code would warn on stderr and silently use its default.
            raise UnrecordableCall(f"Claude effort {effort!r} is not one of {CLAUDE_EFFORTS}")
        source = "claude-code accepted values"
    elif binary == "codex" and effort != NOT_APPLICABLE:
        levels, _default, source = codex_supported_efforts(model)
        if effort not in levels:
            raise UnrecordableCall(f"Codex effort {effort!r} is not supported for {model!r}: {levels}")
    return effort, in_argv, source


@dataclass
class CallPlan:
    binary: str
    argv: list[str]
    spawn: dict
    purpose: str
    model: str | None
    effort: str
    effort_reason: str | None
    session_id: str | None
    launch: dict
    started_utc: str = field(default_factory=_utc_now)


def prepare(argv: Sequence[str], *, effort: str | None, purpose: str,
            cwd: str | Path | None = None, allow_tools: Sequence[str] = (),
            effort_reason: str | None = None, extra_env: dict[str, str] | None = None,
            redact: Sequence[str] = (), allow_alias: bool = False) -> CallPlan:
    """Validate, harden and freeze one CLI call. Raises ``UnrecordableCall`` instead of guessing.

    ``allow_alias``: tooling that deliberately follows the newest model (consult and gold seats,
    task builders) may name an alias such as ``opus``; the id actually served is read from the
    session log. A scored player may not: an alias is a different model next month.
    """
    argv = [str(a) for a in argv]
    binary = hermetic.effective_binary(argv)
    model = _flag_value(argv, "--model", "-m")

    if binary == "claude" and (_flag_value(argv, "--session-id", "--resume") or "--continue" in argv):
        raise UnrecordableCall("the recorder chooses the Claude session id; argv must not carry one")
    effort, in_argv, effort_source = resolve_effort(binary, argv, effort, effort_reason, purpose)
    if not allow_alias and binary == "claude" and not (model or "").startswith("claude-"):
        raise UnrecordableCall(f"Claude model {model!r} is an alias or the CLI default; name the full model id")
    if not allow_alias and binary == "codex" and not model:
        raise UnrecordableCall("Codex call names no model, so Codex would pick its configured default")

    if binary == "claude" and effort != NOT_APPLICABLE and in_argv is None:
        argv[_after(argv, binary):_after(argv, binary)] = ["--effort", effort]
    elif binary == "codex" and effort != NOT_APPLICABLE and in_argv is None:
        argv[_after(argv, binary):_after(argv, binary)] = ["-c", f"model_reasoning_effort={effort}"]

    session_id = None
    if binary == "claude":
        session_id = str(uuid.uuid4())
        argv[_after(argv, binary):_after(argv, binary)] = ["--session-id", session_id]

    if binary == "codex" and cwd is None:
        # One folder per call, so the call's thread can be found in Codex's store by folder.
        cwd = hermetic.hermetic_cwd(tag=f"{re.sub(r'[^a-z0-9-]', '-', purpose.lower())[:40]}-{uuid.uuid4().hex[:12]}")

    hardened, spawn = hermetic.spawn_kwargs(argv, allow_tools=allow_tools, cwd=cwd, via_recorder=True)
    if extra_env:
        # Caller-set knobs (e.g. CLAUDE_CODE_MAX_OUTPUT_TOKENS) are recorded with their values.
        spawn["env"] = {**(spawn.get("env") or dict(os.environ)), **extra_env}
    env = spawn.get("env") or {}
    launch = {
        "recorder_version": RECORDER_VERSION,
        "binary": binary,
        "cli_version": hermetic.cli_version(binary) if binary else None,
        # A prompt passed as an argument (agy, cursor-agent) is replaced by its digest.
        "argv": [f"<redacted sha256={hashlib.sha256(a.encode('utf-8')).hexdigest()[:16]} chars={len(a)}>"
                 if a in set(redact) else a for a in hardened],
        "cwd_purpose": purpose,
        "env_var_names": sorted(env),
        "env_pinned": {k: env[k] for k in sorted({*hermetic.PINNED_ENV, *(extra_env or {})}) if k in env},
        "model_requested": model,
        "model_alias_allowed": allow_alias,
        "effort_requested": effort,
        "effort_reason": effort_reason,
        "effort_validated_against": effort_source,
        "allow_tools": list(allow_tools),
        "containment_profile": hermetic.PROFILE_VERSION,
    }
    return CallPlan(binary=binary, argv=hardened, spawn=spawn, purpose=purpose, model=model,
                    effort=effort, effort_reason=effort_reason, session_id=session_id, launch=launch)


#: Codex thread ids already attributed to a call in this process: one thread, one call (review
#: 2026-11-01: a caller-supplied folder could match the previous call's thread).
_CLAIMED_THREADS: set[str] = set()

_VERSION_TOKEN = re.compile(r"\d+(?:\.\d+)+(?:[-+][0-9A-Za-z.]+)?")


def _version_token(v) -> str | None:
    m = _VERSION_TOKEN.search(str(v or ""))
    return m.group(0) if m else None


def _find_codex_thread(plan: CallPlan, ended: dt.datetime) -> tuple[dict | None, list[dict], str]:
    """The Codex thread this call wrote, by its own folder and time window; never one already claimed."""
    want = os.path.normcase(os.path.normpath(str(plan.spawn.get("cwd") or "")))
    started = dt.datetime.fromisoformat(plan.started_utc.replace("Z", "+00:00"))
    lo = (started - dt.timedelta(seconds=5)).timestamp() * 1000
    hi = (ended + dt.timedelta(seconds=5)).timestamp() * 1000
    hits = []
    for t in L.codex_threads(cwd_like=(f"%{Path(want).name}%",)):
        cwd = os.path.normcase(os.path.normpath(str(t.get("cwd") or "").replace("\\\\?\\", "")))
        if cwd == want and lo <= (t.get("created_at_ms") or 0) <= hi and t.get("id") not in _CLAIMED_THREADS:
            hits.append(t)
    if len(hits) != 1:
        return None, [], "not-found" if not hits else f"ambiguous ({len(hits)} threads)"
    _CLAIMED_THREADS.add(hits[0]["id"])
    turns = L.codex_thread_turns([hits[0]["id"]]).get(hits[0]["id"], [])
    return hits[0], turns, "found"


CUSTODY_SUBDIR = Path("ground_truth") / "sciencearena-raw" / "cli-session-logs"


def _custody_root() -> Path | None:
    """Where call logs and the call ledger go. ``SCIENCEARENA_CALL_CUSTODY_DIR`` overrides it
    (the test suite points it at a temp folder so no test row reaches custody)."""
    override = os.environ.get("SCIENCEARENA_CALL_CUSTODY_DIR")
    if override:
        return Path(override)
    art = os.environ.get("ARTICLE_REPOSITORY")
    return Path(art) / CUSTODY_SUBDIR if art else None


def _custody_dir() -> Path | None:
    root = _custody_root()
    if root is None:
        return None
    return root / "calls" / dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _copy_log(src: str | os.PathLike, binary: str, session: str) -> dict:
    dest_dir = _custody_dir()
    if dest_dir is None:
        return {"copied": False, "reason": "ARTICLE_REPOSITORY unset; the CLI keeps its own copy for a limited time"}
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".jsonl.zst" if str(src).endswith(".zst") else ".jsonl"
    dest = dest_dir / f"{binary}-{session}{suffix}"
    shutil.copyfile(src, dest)
    return {"copied": True, "sha256": L.file_sha256(dest),
            "custody_path": (CUSTODY_SUBDIR / dest.relative_to(_custody_root())).as_posix()}


def finish(plan: CallPlan, *, returncode: int | None, stdout: str | None, stderr: str | None) -> dict:
    """The ``model_call`` block for the run record (see module docstring)."""
    ended = dt.datetime.now(dt.timezone.utc)
    facts, log_state, log_copy = None, "not-applicable", None
    if plan.binary == "claude" and plan.session_id:
        path = L.find_claude_transcript(plan.session_id)
        if path:
            facts, log_state = L.extract_claude(path), "found"
            log_copy = _copy_log(path, "claude", plan.session_id)
        else:
            log_state = "not-found"
    elif plan.binary == "codex":
        thread, turns, log_state = _find_codex_thread(plan, ended)
        if thread:
            roll = L.codex_rollout_file(thread["rollout_path"])
            db = L.extract_codex_db(thread, turns)
            if roll:
                facts = L.extract_codex(roll)
                facts["turns"], facts["reasoning_items"] = db["turns"], db["reasoning_items"]
                log_copy = _copy_log(roll, "codex", thread["id"])
            else:
                # The rollout (the request log) is gone: the thread table holds a stored setting,
                # not what was sent. Recorded, but never as a confirmation.
                facts, log_state = db, "thread-table-only"

    envelope = _claude_envelope(stdout) if plan.binary == "claude" else None

    checks: dict = {"session_log": log_state}
    if facts:
        sent = facts.get("effort_sent")
        if plan.effort == NOT_APPLICABLE:
            checks["effort_applied"] = "not-applicable" if sent in (L.ABSENT, L.NOT_IN_LOG) else f"unexpected: {sent}"
        else:
            checks["effort_applied"] = sent == plan.effort
        billed = facts.get("models_billed")
        # An alias ("opus") is resolved by the CLI; the request it logged names the real id.
        asked = facts.get("model_requested") if facts.get("model_requested") not in (None, L.NOT_IN_LOG) else plan.model
        if isinstance(billed, list) and asked:
            checks["only_requested_model_billed"] = billed == [] or set(billed) <= {asked}
        # Which model ANSWERED, for both CLIs: Codex has no bill, and can reroute a turn (go review
        # 2026-11-01). Codex logs our own --model back, so it is held to the launch; Claude resolves an
        # alias, so its logged request is the reference. No served model logged is not a pass.
        served = facts.get("models_served")
        want = plan.model if plan.binary == "codex" and plan.model else asked
        if isinstance(served, list) and want:
            checks["only_requested_model_served"] = bool(served) and set(served) <= {want}
        checks["tools_called"] = len(facts.get("tools_called") or [])
        if log_state == "thread-table-only" and checks.get("effort_applied") is True:
            checks["effort_applied"] = "unverified"
        hv = _version_token(facts.get("harness_version"))
        lv = _version_token(plan.launch.get("cli_version"))
        if hv and hv != lv:
            # Exact, not a substring (2.1.3 is not 2.1.316); an unknown launch version is a change too.
            checks["cli_version_changed"] = {"launch": lv, "session_log": hv}
        checks["context_injections"] = facts.get("context_injections")
    if stderr and "Unknown --effort" in stderr:
        checks["effort_applied"] = False

    block = {
        "recorder_version": RECORDER_VERSION,
        "launch": plan.launch,
        "started_utc": plan.started_utc,
        "ended_utc": ended.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "returncode": returncode,
        "stderr_sha256": hashlib.sha256((stderr or "").encode("utf-8")).hexdigest() if stderr else None,
        "stderr_chars": len(stderr or ""),
        "session_id": plan.session_id or (facts or {}).get("session_id"),
        "session_log": log_copy,
        "envelope": envelope,
        "facts": facts,
        "checks": checks,
    }
    _ledger(block)
    # The CLI's raw output, for the run's raw sidecar (public) or custody (held-out). Kept off
    # the ledger and stripped by public_block(); the runner moves it to the raw store.
    block["_raw"] = {"stdout": stdout, "stderr": stderr}
    return block


def _claude_envelope(stdout: str | None) -> dict | None:
    """The CLI's result envelope minus the answer text: ``--output-format json`` (one object) or
    ``stream-json`` (the last ``{"type": "result"}`` line)."""
    if not stdout:
        return None
    lines = [ln for ln in stdout.splitlines() if ln.lstrip().startswith("{")]
    cands = [stdout] + list(reversed(lines))
    for c in cands:
        try:
            env = json.loads(c)
        except ValueError:
            continue
        if isinstance(env, dict) and (env.get("type") == "result" or "session_id" in env and "usage" in env):
            return {k: v for k, v in env.items() if k != "result"}
    return None


def _ledger(block: dict) -> None:
    """Append every call, from every caller, to the custody call ledger (best effort)."""
    d = _custody_dir()
    if d is None:
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "calls.jsonl", "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(block, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    except OSError:
        pass


_REDACTED_ARG = re.compile(r"^<redacted sha256=[0-9a-f]+ chars=\d+>$")


def public_block(block: dict | None, *, held_out: bool) -> dict | None:
    """The part of a ``model_call`` block that may be written to a tracked run record.

    Idempotent, and an aggregate (several calls for one record) is reduced block by block.
    """
    if not block:
        return block
    out = dict(block)
    out.pop("_raw", None)
    if isinstance(out.get("blocks"), list):
        out["blocks"] = [public_block(b, held_out=held_out) for b in out["blocks"]]
        return out
    if out.get("facts"):
        out["facts"] = L.public_view(out["facts"], held_out=held_out)
    if held_out:
        # checks.context_injections is a copy of facts.context_injections: strip it the same way.
        ci = (out.get("checks") or {}).get("context_injections")
        if ci:
            def _strip(c):
                return {k: v for k, v in c.items() if k != "host_skills_text"} if isinstance(c, dict) else c
            out["checks"] = {**out["checks"],
                             "context_injections": [_strip(c) for c in ci] if isinstance(ci, list) else _strip(ci)}
        for k in ("session_id", "session_log", "envelope", "stderr_sha256", "stderr_chars",
                  "started_utc", "ended_utc"):
            out.pop(k, None)
        launch = dict(out.get("launch") or {})
        if isinstance(launch.get("argv"), list):
            # A prompt passed as an argument is recorded as digest + length; the length of a
            # held-out document is itself a measurement of it.
            argv = ["<redacted>" if isinstance(a, str) and _REDACTED_ARG.match(a) else a for a in launch["argv"]]
            # The session id names the custody transcript; it is withheld with the other ids.
            for i in range(len(argv) - 1):
                if argv[i] == "--session-id":
                    argv[i + 1] = "<withheld>"
            launch["argv"] = argv
        out["launch"] = launch
        out["withheld"] = "held-out: ids, timings, usage and the CLI envelope are kept in article custody"
    return out


def aggregate_blocks(blocks: list[dict]) -> dict:
    """One record that made several CLI calls (RegCheck): every block kept, plus a summary."""
    efforts = sorted({str(b["launch"].get("effort_requested")) for b in blocks})
    applied = [b["checks"].get("effort_applied") for b in blocks]
    return {"recorder_version": RECORDER_VERSION, "calls": len(blocks), "efforts_requested": efforts,
            "effort_applied_all": all(a is True or a == "not-applicable" for a in applied) if applied else None,
            "blocks": blocks}


def run(argv: Sequence[str], *, effort: str | None, purpose: str, input: str | None = None,
        timeout_s: float | None = None, cwd: str | Path | None = None, allow_tools: Sequence[str] = (),
        effort_reason: str | None = None, extra_env: dict[str, str] | None = None, sink=None,
        redact: Sequence[str] = (), allow_alias: bool = False):
    """prepare + subprocess.run + finish in one step. Returns ``(CompletedProcess, block)``.

    The block is produced on every exit path; a timeout re-raises after it is handed to
    ``sink`` (a callable), so a caller that loses the exception still keeps the record.
    """
    import subprocess
    plan = prepare(argv, effort=effort, purpose=purpose, cwd=cwd, allow_tools=allow_tools,
                   effort_reason=effort_reason, extra_env=extra_env, redact=redact, allow_alias=allow_alias)
    proc = None
    try:
        proc = subprocess.run(plan.argv, input=input, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout_s, check=False, **plan.spawn)
    finally:
        block = finish(plan, returncode=proc.returncode if proc else None,
                       stdout=proc.stdout if proc else None, stderr=proc.stderr if proc else None)
        if sink is not None:
            sink(block)
    return proc, block
