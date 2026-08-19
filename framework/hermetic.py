"""The single boundary between an agentic CLI player and the benchmark it is scored in.

WHY THIS MODULE EXISTS
----------------------
Shelling out to ``claude --print`` (or ``codex exec``, or ``cursor-agent -p``)
does not give you a model. It gives you *the model plus that CLI's entire ambient
environment* — and by default that environment includes file-system tools pointed
at the current working directory, which for this project is the repo containing
the answer key.

Measured 2026-08-16, not inferred: a player invoked exactly as
``SubprocessCliAdapter`` invoked it read ``scorer.py``, read the committed
``_ground_truth.json``, and returned a planted unguessable nonce verbatim.
Full write-up: ``docs/design/cli-player-contamination.md``.

This module is the one place that decides what a player subprocess inherits, in
the same spirit as ``framework/holdout.py``: fix the class at a single seam,
then make every writer go through it. Adapters must not assemble their own
sanitisation — a second copy is a second thing to forget.

THE FOUR CHANNELS, AND HOW EACH IS CLOSED
-----------------------------------------
============================  =========================================================
channel                       closed by
============================  =========================================================
tools pointed at the repo     ``--tools ""`` — an EMPTY ALLOWLIST. A denylist has to
                              name every tool, so the next tool the CLI ships is
                              allowed by default. Verified: returns BLOCKED against
                              the nonce.
cwd = the repo                ``hermetic_cwd()`` — a scratch directory outside the
                              repo. This also removes the ``CLAUDE.md`` hierarchy and
                              the cwd-keyed project memory (this project's own notes
                              about the thing being measured).
user settings + memory        ``--setting-sources ""``. Verified two-sided: the user
                              memory probe answers YES without the flag and NO with
                              it, and ``settings.json``'s permission warnings stop
                              appearing on stderr. NOTE an empty ``CLAUDE_CONFIG_DIR``
                              was tried first and is WRONG — it also removes the
                              OAuth credentials, so the CLI exits "Not logged in".
env carrying our own secrets  ``hermetic_env()``. ``framework/parity.py`` reads the
                              PRIVATE SEED from ``SCIENCEARENA_PRIVATE_SEED``, so an
                              inherited environment hands a player the seed that
                              generates the private split. Found 2026-08-19 while
                              wiring this module; it is not in the original defect
                              report.
============================  =========================================================

FAIL LOUD, NOT OPEN
-------------------
Every binary is treated as an agent until it appears on ``INERT_LOCAL_TOOLS``.
An unrecognised binary raises ``UnhardenedCliError`` rather than running
unhardened. That is deliberate and is the whole point: the defect this module
exists for was created by a player being wired up without anyone asking what the
subprocess inherits, while every test stayed green.

WHAT IS VERIFIED AND WHAT IS NOT
--------------------------------
``claude`` is verified end-to-end by nonce probe on claude-code 2.1.224
(2026-08-19), in both directions. ``codex`` and ``cursor-agent`` profiles are
built from their documented flags but their tool containment is **UNVERIFIED by
nonce** — Codex was out of quota at the time. Each profile carries its own
``verified`` field, ``framework audit`` reports it, and
``player_instrument_label()`` marks an unverified profile in the published label
rather than letting it read as hermetic.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

__all__ = [
    "UnhardenedCliError",
    "HermeticProfile",
    "HERMETIC_PROFILES",
    "INERT_LOCAL_TOOLS",
    "PROFILE_VERSION",
    "is_agentic_cli",
    "normalise_binary",
    "effective_binary",
    "WRAPPER_TOOLS",
    "harden_argv",
    "hermetic_cwd",
    "hermetic_env",
    "spawn_kwargs",
    "cli_version",
    "config_fingerprint",
    "player_instrument_label",
    "takes_stdin_prompt",
    "needs_fresh_session",
    "prompt_goes_in_a_file",
]

#: Bumped whenever the containment changes in a way that makes scores from
#: before and after different instruments. It is folded into
#: ``config_fingerprint()``, which is folded into the published player version,
#: so a change mints a NEW player rather than silently appending rows to an old
#: one (Fable 5's point during the 2026-08-16 consult).
PROFILE_VERSION = "hermetic-1"


class UnhardenedCliError(RuntimeError):
    """Raised when a CLI would be spawned without a verified containment profile."""


@dataclass(frozen=True)
class HermeticProfile:
    """How one CLI family is contained.

    ``tools_flag``/``empty_tools_value``
        How this CLI expresses "no tools". ``None`` means the CLI offers no tool
        restriction at all and containment rests entirely on ``cwd``.
    ``trailing_flags``
        Appended LAST and deliberately boolean-only. ``--tools`` is variadic, so
        a positional argument following it is swallowed as a tool name; ending on
        a boolean flag keeps a trailing positional safe. (This cost a probe:
        ``claude --print --tools "" "<prompt>"`` dies with "Input must be
        provided either through stdin or as a prompt argument".)
    ``verified``
        True only when a live nonce probe confirmed containment for this binary.
    """

    binary: str
    tools_flag: str | None
    empty_tools_value: str | None
    isolation_flags: tuple[str, ...] = ()
    trailing_flags: tuple[str, ...] = ()
    cwd_flag: str | None = None
    verified: bool = False
    verified_note: str = ""
    version_argv: tuple[str, ...] = ("--version",)
    #: True when a nonce probe MEASURED this CLI reading outside its workspace
    #: despite every flag it offers. Such a CLI cannot be contained in-process;
    #: only an OS-level boundary (container / VM) works. Its records are labelled
    #: `uncontrolled` no matter what flags were passed.
    flags_insufficient: bool = False
    stdin_prompt: bool = True
    #: True when the CLI keeps a conversation across invocations in the same
    #: workspace, so consecutive tasks are NOT independent. Such a CLI needs a
    #: fresh workspace per task or the benchmark measures a conversation
    #: instead of a set of items.
    session_persists: bool = False
    #: True when a multi-line prompt cannot be passed on argv and must be handed
    #: over as a file in the workspace instead. Set from measurement, never from
    #: documentation.
    prompt_via_file: bool = False


HERMETIC_PROFILES: dict[str, HermeticProfile] = {
    # Verified end-to-end 2026-08-19 on claude-code 2.1.224, both directions:
    #   repo nonce with --tools ""            -> not returned
    #   repo nonce with --tools Read, cwd=scratch -> BLOCKED
    #   file inside cwd with --tools Read     -> returned (so Read still works)
    #   user memory probe, with/without --setting-sources "" -> NO / YES
    "claude": HermeticProfile(
        binary="claude",
        tools_flag="--tools",
        empty_tools_value="",
        # Drops user + project settings.json AND the CLAUDE.md memory hierarchy,
        # while leaving OAuth intact. `--setting-sources ""` is passed as a
        # single empty value: the flag takes a comma-separated list.
        isolation_flags=("--setting-sources", ""),
        # Boolean, so it is safe as the final token. Also stops any MCP server
        # configured globally from attaching to a benchmark player.
        trailing_flags=("--strict-mcp-config",),
        verified=True,
        verified_note="nonce probe, claude-code 2.1.224, 2026-08-19",
    ),
    # `--ignore-user-config` is Codex's analogue of `--setting-sources ""`:
    # "Do not load $CODEX_HOME/config.toml; auth still uses CODEX_HOME".
    # Codex exposes no tool allowlist, so containment is cwd + sandbox only.
    # UNVERIFIED: Codex was out of quota when this was written, so no nonce
    # probe has been run against it. Do not describe a Codex run as hermetic
    # until one has.
    "codex": HermeticProfile(
        binary="codex",
        tools_flag=None,
        empty_tools_value=None,
        isolation_flags=("--ignore-user-config", "--skip-git-repo-check",
                         "--sandbox", "read-only"),
        cwd_flag="--cd",
        verified=False,
        verified_note="no nonce probe run (Codex out of quota 2026-08-16..20)",
    ),
    # Cursor CLI. MEASURED LEAKY 2026-08-19 on cursor-agent 2026.08.11-e8db854.
    # With `--mode ask` (its read-only Q&A mode), cwd set to a scratch directory
    # AND `--workspace <scratch>`, it still read an ABSOLUTE path into the repo
    # and returned the planted nonce. Its own `--sandbox enabled` refuses to
    # start on this host: "Sandbox mode is enabled but not available on this
    # system. Sandbox requires macOS or Linux."
    #
    # So on Windows there is NO flag combination that contains this CLI. A
    # cursor-agent player is containable only by an OS-level boundary — a Linux
    # container with the repo simply absent from the filesystem. Until that
    # exists, every cursor-agent record is labelled `uncontrolled`, which is
    # what `flags_insufficient` forces.
    #
    # It also does not read the prompt from stdin ("Error: No prompt provided
    # for print mode"), unlike every other CLI here — the prompt goes on argv.
    "cursor-agent": HermeticProfile(
        binary="cursor-agent",
        tools_flag=None,
        empty_tools_value=None,
        isolation_flags=("--mode", "ask", "--trust"),
        cwd_flag="--workspace",
        verified=False,
        verified_note=("MEASURED LEAKY on Windows: returned a repo nonce under "
                       "--mode ask + --workspace; --sandbox requires macOS/Linux "
                       "(probe 2026-08-19, cursor-agent 2026.08.11-e8db854)"),
        flags_insufficient=True,
        stdin_prompt=False,
        # MEASURED 2026-08-19, and it invalidated an entire pilot run before it
        # was caught. Driving 8 tasks through one workspace returned the
        # IDENTICAL value (0.242535625) for all eight — across d->r, r->d,
        # d->OR, eta2->f and OR->d — because that was the answer to the first
        # question ever asked in that directory. Prompt tokens grew 62k -> 114k
        # within a single run, which is the accumulating conversation.
        #
        # Two things follow, and the second is the serious one:
        #   1. answers are stale, so the score is an artifact, not a measurement;
        #   2. task N can SEE tasks 1..N-1 — cross-item contamination inside the
        #      benchmark, independent of the repo-access channel.
        # A frozen wrong answer is the good case here: it is obvious. A
        # conversation that merely biases later answers would have published.
        session_persists=True,
        # MEASURED 2026-08-19 with a nonce at the END of the prompt, the only
        # probe that separates "truncated" from "ignored the instruction":
        #   single-line argv -> returned BANANA-7731
        #   multi-line  argv -> "I'll look in the workspace for the token...
        #                        Nothing follows"
        # `cursor-agent.CMD` is a batch shim that re-launches
        # `powershell.exe -File cursor-agent.ps1 %*`, and a newline does not
        # survive that double hop. EVERY arena prompt template here is
        # multi-line, so a cursor player would have been scored on its FIRST
        # LINE ONLY — and note the model did not report a problem, it invented
        # an approach. A silently truncated prompt that still returns
        # well-formed JSON is indistinguishable from a real answer.
        prompt_via_file=True,
        version_argv=("--version",),
    ),
    # opencode gateway (free-tier routing to OpenRouter / NVIDIA / Zen). Reached
    # through a `timeout` wrapper in registry.yaml, which is why `harden_argv`
    # resolves the effective binary rather than argv[0]. UNVERIFIED: no nonce
    # probe has been run against it, so its records label as
    # hermetic-unverified, never hermetic.
    "opencode": HermeticProfile(
        binary="opencode",
        tools_flag=None,
        empty_tools_value=None,
        verified=False,
        verified_note="pending nonce probe",
    ),
    # Google Antigravity CLI. Headless `--print` with an explicit agent; no
    # documented tool allowlist, so containment is cwd only.
    "agy": HermeticProfile(
        binary="agy",
        tools_flag=None,
        empty_tools_value=None,
        verified=False,
        verified_note="pending nonce probe",
    ),
    # Legacy Gemini CLI. Retained because commented-out registry entries still
    # name it; it is not currently an active player and is unverified.
    "gemini": HermeticProfile(
        binary="gemini",
        tools_flag=None,
        empty_tools_value=None,
        verified=False,
        verified_note="legacy route, superseded by agy; unverified",
    ),
}

#: Binaries that are plain local tools, not agents: they take an input, produce
#: an output, and have no capability to look around. They need no containment,
#: and pretending otherwise would break them.
#:
#: This is an ALLOWLIST on purpose. Anything not named here is treated as an
#: agent, because the cost of wrongly trusting an agent is publishing a
#: contaminated score and the cost of wrongly distrusting a tool is one line
#: added to this list.
INERT_LOCAL_TOOLS: frozenset[str] = frozenset({
    "anystyle", "pdftotext", "pdftoppm", "pdfinfo",
    "rscript", "r", "java", "python", "python3", "node",
    "git", "timeout", "grobid", "cermine", "docpluck", "liteparse",
})


#: Commands that WRAP another command rather than doing work themselves.
#: `players/registry.yaml` really does ship
#: `["timeout", "-k", "20", "240", "opencode", "run", ...]`, and the first
#: version of this module matched on argv[0] only — so `timeout` (an inert local
#: tool) hid an agentic CLI behind it and those players were never hardened.
#: A wrapper must never be able to launder an agent.
WRAPPER_TOOLS: frozenset[str] = frozenset({"timeout", "env", "nice", "stdbuf", "chrt"})


def effective_binary(argv: Sequence[str]) -> str:
    """The binary that actually runs, looking through wrappers.

    ``["timeout", "-k", "20", "240", "opencode", "run"]`` -> ``opencode``.
    Skips the wrapper's own flags and numeric arguments (durations), stopping at
    the first token that is neither.
    """
    tokens = [str(t) for t in argv]
    idx = 0
    seen = 0
    while idx < len(tokens) and normalise_binary(tokens[idx]) in WRAPPER_TOOLS and seen < 4:
        seen += 1
        idx += 1
        while idx < len(tokens):
            tok = tokens[idx]
            if tok.startswith("-"):
                idx += 1
                continue
            # A bare number is a wrapper argument (timeout's duration, nice's
            # increment), not the command being wrapped.
            if tok.replace(".", "", 1).rstrip("smhd").isdigit():
                idx += 1
                continue
            break
    return normalise_binary(tokens[idx]) if idx < len(tokens) else ""


def normalise_binary(binary: str) -> str:
    """`C:/x/claude.cmd` -> `claude`. Windows shims and paths must not defeat matching."""
    name = Path(str(binary)).name.lower()
    for suffix in (".cmd", ".bat", ".exe", ".ps1"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def is_agentic_cli(binary: str) -> bool:
    """True unless `binary` is a vetted inert local tool.

    Note the direction: this answers "must I contain this?", not "is this
    definitely an AI agent?". An unknown binary answers True, which routes it to
    ``harden_argv``'s refusal instead of to an open subprocess.
    """
    return normalise_binary(binary) not in INERT_LOCAL_TOOLS


def harden_argv(argv: Sequence[str], *, allow_tools: Sequence[str] = ()) -> list[str]:
    """Return `argv` with this CLI's containment flags appended.

    ``allow_tools`` is the *only* dial a caller gets, and it exists for one real
    case: the PDF adapters hand the model a file path and need ``Read`` to open
    it. Everything else passes ``()`` — no tools at all.

    Raises ``UnhardenedCliError`` for a binary that is neither an inert local
    tool nor a profiled agent.
    """
    argv = list(argv)
    if not argv:
        raise UnhardenedCliError("empty argv")
    # Look THROUGH wrappers: `timeout -k 20 240 opencode run ...` is an
    # opencode player, and matching on argv[0] alone let `timeout` launder it.
    binary = effective_binary(argv)

    if not is_agentic_cli(binary):
        return argv

    profile = HERMETIC_PROFILES.get(binary)
    if profile is None:
        raise UnhardenedCliError(
            f"{binary!r} has no hermetic profile. Every agentic CLI player must "
            f"declare how it is contained before it can produce a published "
            f"score — add a HermeticProfile in framework/hermetic.py and verify "
            f"it with a nonce probe. Known profiles: {sorted(HERMETIC_PROFILES)}. "
            f"If this is an inert local tool, add it to INERT_LOCAL_TOOLS instead."
        )

    hardened = list(argv)
    if profile.tools_flag and profile.tools_flag not in hardened:
        value = ",".join(allow_tools) if allow_tools else profile.empty_tools_value
        hardened += [profile.tools_flag, value or ""]
    if profile.isolation_flags and profile.isolation_flags[0] not in hardened:
        hardened += list(profile.isolation_flags)
    # Trailing flags are boolean and go LAST so a positional argument appended by
    # a caller cannot be eaten by a preceding variadic flag.
    for flag in profile.trailing_flags:
        if flag not in hardened:
            hardened.append(flag)
    return hardened


def prompt_goes_in_a_file(argv: Sequence[str]) -> bool:
    """Must this CLI receive its prompt as a workspace file rather than on argv?"""
    profile = HERMETIC_PROFILES.get(effective_binary(argv))
    return bool(profile and profile.prompt_via_file)


def needs_fresh_session(argv: Sequence[str]) -> bool:
    """Does this CLI need a brand-new workspace for every task?

    True only for CLIs measured to carry a conversation between invocations.
    For those, a per-run working directory is not enough: task independence is
    an assumption the benchmark's whole design rests on.
    """
    profile = HERMETIC_PROFILES.get(effective_binary(argv))
    return bool(profile and profile.session_persists)


def takes_stdin_prompt(argv: Sequence[str]) -> bool:
    """Does this CLI accept its prompt on stdin?

    True for everything except `cursor-agent`, which has no stdin path at all
    ("Error: No prompt provided for print mode", measured 2026-08-19) and needs
    the prompt as a trailing positional argument. A property of the CLI, so it
    lives on the profile rather than being duplicated per player.
    """
    profile = HERMETIC_PROFILES.get(effective_binary(argv))
    return True if profile is None else profile.stdin_prompt


def hermetic_cwd(*, tag: str = "player") -> Path:
    """A scratch working directory outside the repo, created on demand.

    Placed under the OS temp dir rather than anywhere beneath the repo, because
    the point is that no ancestor of the cwd is the project: ``CLAUDE.md`` files,
    ``.claude/`` settings and cwd-keyed project memory are all resolved by
    walking UP from the cwd.
    """
    base = Path(tempfile.gettempdir()) / "sciencearena-hermetic" / tag
    base.mkdir(parents=True, exist_ok=True)
    return base


#: Environment variables stripped from every player subprocess.
#:
#: ``SCIENCEARENA_PRIVATE_SEED`` is the sharp one: ``framework/parity.py`` reads
#: the private-split seed from the environment, so an inherited environment
#: hands the player the seed that generates the private tasks. The rest are
#: stripped because they point the child at our paths and data.
#:
#: ``ANTHROPIC_API_KEY`` is stripped for a different reason: this portfolio runs
#: Claude strictly on the Max subscription, and a stray API key in the
#: environment silently reroutes a CLI player to metered billing.
STRIPPED_ENV_PREFIXES: tuple[str, ...] = ("SCIENCEARENA_",)
STRIPPED_ENV_VARS: frozenset[str] = frozenset({
    "ANTHROPIC_API_KEY",
    "VIBE_ROOT",
    "ARTICLE_REPOSITORY",
    "HABISCI_REF",
})


def hermetic_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """A copy of the environment with our own secrets and roots removed.

    Auth is deliberately left intact — ``CLAUDE_CODE_OAUTH_TOKEN``,
    ``CODEX_HOME``, ``CURSOR_API_KEY`` and the platform keychain are how the
    subprocess reaches the subscription at all. Isolation of *settings* is done
    with the CLI's own flag (``--setting-sources ""``), not by blanking the
    config directory, which was tried first and breaks login.
    """
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key in STRIPPED_ENV_VARS or key.startswith(STRIPPED_ENV_PREFIXES):
            env.pop(key, None)
    return env


def spawn_kwargs(
    argv: Sequence[str],
    *,
    allow_tools: Sequence[str] = (),
    cwd: str | Path | None = None,
) -> tuple[list[str], dict]:
    """One call that returns everything a contained ``subprocess.run`` needs.

    Returns ``(hardened_argv, kwargs)`` where kwargs carries ``cwd`` and ``env``.
    Adapters call this instead of assembling containment themselves — a second
    copy of this logic is a second thing to forget to update.

    ``cwd`` may be overridden by callers that must place a working file next to
    the player (the PDF adapters put the temp PDF in the cwd they pass). It is
    validated: a cwd inside the repo defeats the whole boundary and raises.
    """
    binary = effective_binary(argv) if argv else ""
    hardened = harden_argv(argv, allow_tools=allow_tools)

    if not is_agentic_cli(binary):
        return hardened, {}

    workdir = Path(cwd) if cwd is not None else hermetic_cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    resolved = workdir.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if resolved == repo_root or repo_root in resolved.parents:
        raise UnhardenedCliError(
            f"refusing to spawn {binary!r} with cwd inside the repo ({resolved}). "
            f"The cwd is what put the answer key in scope."
        )

    profile = HERMETIC_PROFILES[binary]
    if profile.cwd_flag and profile.cwd_flag not in hardened:
        # Some CLIs take the workspace as a flag as well as inheriting cwd; set
        # both, so neither alone is load-bearing.
        hardened += [profile.cwd_flag, str(resolved)]

    return hardened, {"cwd": str(resolved), "env": hermetic_env()}


_VERSION_CACHE: dict[str, str] = {}


def cli_version(binary: str) -> str | None:
    """The installed version of an agentic CLI, for the run record.

    A CLI player is ``<model> via <cli>@<version>``, not the model — the harness
    around the weights is part of the instrument, and it auto-updates. Recorded
    per record so a score can be attributed to the thing that produced it.
    Best-effort: never raises, returns ``None`` when the version cannot be read.
    """
    name = normalise_binary(binary)
    if name in _VERSION_CACHE:
        return _VERSION_CACHE[name] or None
    profile = HERMETIC_PROFILES.get(name)
    argv_tail = profile.version_argv if profile else ("--version",)
    try:
        import shutil
        resolved = shutil.which(name)
        if resolved is None:
            return None
        proc = subprocess.run(
            [resolved, *argv_tail], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30, check=False,
        )
        raw = (proc.stdout or proc.stderr or "").strip().splitlines()
        value = raw[0].strip() if raw else ""
    except Exception:
        value = ""
    _VERSION_CACHE[name] = value
    return value or None


def config_fingerprint(binary: str, *, allow_tools: Sequence[str] = ()) -> str:
    """A short, stable hash of everything that makes this containment what it is.

    Folded into the published player version so that changing the containment
    mints a new player instead of appending rows to a leaderboard entry produced
    under different conditions.
    """
    name = normalise_binary(binary)
    profile = HERMETIC_PROFILES.get(name)
    material = "|".join([
        PROFILE_VERSION,
        name,
        ",".join(allow_tools),
        "" if profile is None else repr(profile),
        cli_version(name) or "unknown",
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:8]


def player_instrument_label(model_id: str, binary: str, *,
                            allow_tools: Sequence[str] = ()) -> str:
    """``haiku-4-5 via claude-code@2.1.224 (hermetic)`` — the honest instrument name.

    An unverified profile says so in the label. A CLI player must never share a
    leaderboard row with a bare API call to the same weights un-annotated: it is
    a different instrument, with a different system prompt, different sampling
    defaults and a version that moves on its own.
    """
    name = normalise_binary(binary)
    version = cli_version(name) or "unknown"
    profile = HERMETIC_PROFILES.get(name)
    if profile is None:
        state = "UNCONTAINED"
    elif profile.flags_insufficient:
        state = "UNCONTAINED (flags measured insufficient)"
    elif profile.verified:
        state = "hermetic"
    else:
        state = "hermetic-unverified"
    if allow_tools:
        state = f"{state}; tools={'+'.join(allow_tools)}"
    return f"{model_id} via {name}@{version} ({state})"


# --------------------------------------------------------------------------
# provenance: what a run record says about how the player was contained
# --------------------------------------------------------------------------

#: The three states a scored record can be in, and the ONLY way to tell them
#: apart for records written before 2026-08-19: a CLI record with no
#: ``provenance.containment`` block was produced before containment existed.
#: Absence is the signal, which is why this is derived rather than backfilled —
#: rewriting history to assert a property it never had is the thing this
#: benchmark exists to prevent.
CONTAINMENT_HERMETIC = "hermetic"
CONTAINMENT_UNVERIFIED = "hermetic-unverified"
CONTAINMENT_UNCONTROLLED = "uncontrolled"
CONTAINMENT_NOT_APPLICABLE = "not-applicable"

#: Adapter classes that spawn an agentic CLI. Used as a second, independent
#: signal when `provenance.command` cannot be parsed, so a malformed command
#: string can never downgrade a CLI player to "no agent involved".
CLI_ADAPTER_CLASSES: frozenset[str] = frozenset({
    "SubprocessCliAdapter",
    "LlmCliPdfAdapter", "LlmCliPdfSectionsAdapter", "LlmCliPdfTablesAdapter",
    "LlmPdfReferencesAdapter", "LlmPdfCitationsAdapter",
    "AntigravityCliAdapter",
})


def record_containment(cli_command: Sequence[str] | None,
                       *, allow_tools: Sequence[str] = ()) -> dict | None:
    """The ``provenance.containment`` block for one run record, or ``None``.

    ``None`` for a player that is not an agentic CLI — an HTTP/API player, an R
    tool, a library call. Those have no ambient environment to inherit, so a
    containment claim about them would be noise.
    """
    if not cli_command:
        return None
    binary = effective_binary(cli_command)
    if not is_agentic_cli(binary):
        return None
    profile = HERMETIC_PROFILES.get(binary)
    return {
        "profile_version": PROFILE_VERSION,
        "cli": binary,
        "cli_version": cli_version(binary),
        "allow_tools": list(allow_tools),
        "verified": bool(profile and profile.verified),
        "flags_insufficient": bool(profile and profile.flags_insufficient),
        "verified_note": profile.verified_note if profile else "no profile",
        "fingerprint": config_fingerprint(binary, allow_tools=allow_tools),
    }


def containment_state(record: dict) -> str:
    """Classify one run record: hermetic / hermetic-unverified / uncontrolled / n-a.

    The CC3 decision (2026-08-19) was **keep and label**, not re-run: scores
    produced through the open channel stay on the board and say so. This is the
    function that makes that label real, and every consumer must route through
    it rather than re-deriving the rule.

    A CLI record with no containment block is ``uncontrolled`` — it was written
    before ``framework/hermetic.py`` existed, when the player ran inside the repo
    with its full tool set.
    """
    prov = record.get("provenance") or {}
    block = prov.get("containment")
    if block is None:
        # AUTHORITATIVE SIGNAL: adapter_class. It is present on every record and
        # says exactly which code path spawned the player.
        #
        # `provenance.command` is NOT usable for this and two earlier attempts
        # got it wrong, each visible only by reading the emitted VALUES rather
        # than checking the field exists:
        #   * it is a STRING, not a list, in every record before 2026-08-19
        #     ("claude --print --model X  # player-id");
        #   * for an HTTP player it is literally the adapter class name
        #     ("OpenAIChatCompletionsAdapter"), and for the R tool it is a
        #     quoted absolute path ("'C:/Program Files/R/.../Rscript.exe' ...").
        # Treating an unrecognised first token as "an agent" therefore labelled
        # every API and R player `uncontrolled` — 120 of 127, which is both
        # false and the kind of over-warning that trains a reader to ignore the
        # label entirely.
        adapter = str(prov.get("adapter_class") or "")
        if adapter:
            return (CONTAINMENT_UNCONTROLLED if adapter in CLI_ADAPTER_CLASSES
                    else CONTAINMENT_NOT_APPLICABLE)
        # Only when adapter_class is missing entirely do we fall back to argv.
        command = prov.get("command")
        if isinstance(command, str):
            first = command.strip().split()[0] if command.strip() else ""
        elif isinstance(command, list) and command:
            first = str(command[0])
        else:
            first = ""
        if first and is_agentic_cli(first):
            return CONTAINMENT_UNCONTROLLED
        return CONTAINMENT_NOT_APPLICABLE
    if block.get("flags_insufficient"):
        # A CLI measured reading outside its workspace is uncontrolled however
        # carefully it was invoked. Reporting it as "unverified" would imply
        # "not yet checked"; it WAS checked, and it leaked.
        return CONTAINMENT_UNCONTROLLED
    if block.get("verified"):
        return CONTAINMENT_HERMETIC
    return CONTAINMENT_UNVERIFIED


def summarise_containment(records: Sequence[dict]) -> dict:
    """Per-player containment summary for a report.

    ``worst`` is what a reader needs: a player whose records are MIXED — some
    hermetic, some not — is not a hermetic player, and reporting only the
    majority state would launder the uncontrolled half.
    """
    order = [CONTAINMENT_UNCONTROLLED, CONTAINMENT_UNVERIFIED,
             CONTAINMENT_HERMETIC, CONTAINMENT_NOT_APPLICABLE]
    counts: dict[str, int] = {}
    for r in records:
        state = containment_state(r)
        counts[state] = counts.get(state, 0) + 1
    present = [s for s in order if counts.get(s)]
    return {
        "states": counts,
        "worst": present[0] if present else CONTAINMENT_NOT_APPLICABLE,
        "mixed": len([s for s in present if s != CONTAINMENT_NOT_APPLICABLE]) > 1,
    }
