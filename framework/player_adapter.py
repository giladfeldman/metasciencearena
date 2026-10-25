"""Player adapter ABC + concrete stubs for testing.

Real adapters (RCli, Http, SubprocessCli) live in this module too once the
six v1 players are wired up.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import requests

from framework import hermetic


#: Machine-local values for `${NAME}` placeholders in a player's
#: `start_command`. Deliberately NOT in `framework/publish.py`'s public
#: allowlist and gitignored, because the values are directory layout on the
#: machine that runs the benchmark.
LOCAL_PATHS_FILE = Path(__file__).resolve().parents[1] / "players" / "local_paths.yaml"


def _local_paths() -> dict[str, str]:
    """Read `players/local_paths.yaml`, or `{}` when it is absent.

    Absent is a normal state: a fresh clone, CI, and anyone who does not run the
    locally-installed players all lack it. It must not raise here — the error
    belongs at the point a placeholder actually fails to resolve, where it can
    name the placeholder.
    """
    if not LOCAL_PATHS_FILE.is_file():
        return {}
    import yaml
    loaded = yaml.safe_load(LOCAL_PATHS_FILE.read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in loaded.items()}


def resolve_start_command(command: list[str]) -> list[str]:
    """Expand `${NAME}` placeholders (and `~`) in a player's `start_command`.

    Two rules, and they exist for different reasons.

    `${VIBE_ROOT}` is the portfolio root: it moved out of Dropbox on 2026-08-03
    and may move again, so `players/registry.yaml` must never spell an absolute
    user path — see `Vibe/CLAUDE.md`. Resolution order matches that rule:
    `$VIBE_ROOT`, else `$HOME/Vibe`.

    Every OTHER `${NAME}` resolves from the environment, else from the gitignored
    `players/local_paths.yaml`. That indirection is a DISCLOSURE control, not a
    convenience. `players/registry.yaml` is published to the public mirror
    (`framework/publish.py` lists it), so a launcher path written there tells a
    reader the layout of a private repository. A placeholder tells them only that
    the player is started by a script installed on the machine, which is the true
    and sufficient statement.

    An unresolved placeholder RAISES, naming the placeholder and the file that
    would supply it. It must never silently survive into the command: `cmd /c`
    exits 1 on a nonexistent script without raising, so the only symptom would be
    a 60-second readiness timeout blaming the server for a path that was never
    built.

    Paths come back with forward slashes so a registry entry reads identically on
    every platform and the value a test asserts on is the value Popen receives.
    """
    import re

    vibe_root = os.environ.get("VIBE_ROOT") or str(Path.home() / "Vibe")
    cache: dict[str, str] = {"VIBE_ROOT": vibe_root}
    local: dict[str, str] | None = None

    def value_of(key: str) -> str:
        nonlocal local
        if key in cache:
            return cache[key]
        raw = os.environ.get(key)
        if not raw:
            if local is None:
                local = _local_paths()
            if key not in local:
                raise KeyError(
                    f"start_command references ${{{key}}}, which is set neither in the "
                    f"environment nor in {LOCAL_PATHS_FILE}. That file is gitignored and "
                    f"machine-local by design: add a `{key}: <path>` line to it, or export "
                    f"{key}. It is deliberately not written into players/registry.yaml, "
                    "which is published to the public mirror."
                )
            raw = local[key]
        # A local value may itself be written relative to the portfolio root;
        # ${VIBE_ROOT} is the only nesting allowed, so one pass is enough.
        cache[key] = raw.replace("${VIBE_ROOT}", vibe_root).replace("$VIBE_ROOT", vibe_root)
        return cache[key]

    resolved = []
    for token in command:
        expanded = token.replace("$VIBE_ROOT", vibe_root)
        for key in dict.fromkeys(re.findall(r"\$\{(\w+)\}", expanded)):
            expanded = expanded.replace("${" + key + "}", value_of(key))
        expanded = os.path.expanduser(expanded)
        if expanded != token or "/" in expanded or os.sep in expanded:
            expanded = expanded.replace(os.sep, "/")
        resolved.append(expanded)
    return resolved


def _cli_envelope_error(stdout: str) -> str:
    """The error a CLI's stdout JSON envelope reports, or '' when there is none."""
    try:
        env = json.loads(stdout or "")
    except ValueError:
        return ""
    if not isinstance(env, dict):
        return ""
    parts = [f"{k}={env[k]}" for k in ("subtype", "is_error", "terminal_reason") if k in env]
    if isinstance(env.get("result"), str) and env["result"].strip():
        parts.append(env["result"].strip()[:300])
    return " ".join(parts)


def _extract_json_value(raw: str):
    """Parse the JSON value a CLI emitted, tolerating chatty model output.

    Text-CLI players (notably Opus, which sometimes wraps its answer in prose or
    emits a trailing sentence) do not always return *exactly* one bare JSON value
    on stdout. This permissively recovers the intended object/array:

      1. Strip a leading ```json … ``` fence, then try a strict ``json.loads``.
      2. Else scan for the first balanced ``{...}`` or ``[...]`` block anywhere in
         the text and parse that (handles leading prose, a fence mid-text, and the
         ``Extra data`` case where a sentence trails the JSON).

    Raises ``ValueError`` when there is genuinely no JSON to parse (an empty
    response or a pure-prose refusal) so the runner records it as an honest error
    rather than a fabricated empty object.
    """
    text = raw.strip()
    # 0. Reasoning models wrap a rehearsed draft answer in <think>...</think>; only
    # what follows the LAST closing tag is the answer. Parsing the first balanced
    # block would score the draft (a rehearsed answer once PASSED that way).
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    # 1. Whole-output parse, after stripping a leading fence.
    candidate = text
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        candidate = candidate.rsplit("```", 1)[0]
    candidate = candidate.strip()
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # 2. First *parseable* balanced {...} or [...] block anywhere in the text.
    # Try every candidate opening bracket, not just the first — a non-JSON brace
    # like "{ignored}" earlier in the prose must not abort the search.
    # A balanced block that does not parse (e.g. an outer {"records": [...]} with
    # `//` comments) is malformed AS A WHOLE: its inner records must not be returned
    # as if they were the answer, by either bracket pass.
    dead_spans: list[tuple[int, int]] = []

    def _first_balanced(open_ch: str, close_ch: str):
        search_from = 0
        while True:
            start = text.find(open_ch, search_from)
            if start == -1:
                return None
            inside = next((end for lo, end in dead_spans if lo <= start < end), None)
            if inside is not None:
                search_from = inside
                continue
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            dead_spans.append((start, i + 1))
                            break  # this block is malformed; skip all of it
            search_from = start + 1

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        result = _first_balanced(open_ch, close_ch)
        if result is not None:
            return result
    raise ValueError(
        f"no parseable JSON in CLI output (len={len(raw)}): {raw.strip()[:200]!r}"
    )


#: Why an adapter reports no installed version. FOUR DIFFERENT FACTS that used to
#: arrive as one sentence, "adapter reported no installed version", across all 98
#: UNCHECKED players (measured 2026-08-27).
#:
#: A version this project can compare is the difference between ranking a tool and
#: ranking a label: `player_version` in registry.yaml is an INTENT that drifts,
#: `resolved_tool_version()` is what actually ran. When the second is absent, the
#: audit must be able to say whether that is fine or whether a guard is down.

#: The declared label IS the version -- an LLM whose model id is already in
#: `player_version`, or a script with no package to have a version (pcurve.R).
#: Nothing is wrong and nothing is missing.
VERSION_ABSENT_BY_DESIGN = "by_design"

#: A probe RAN and FAILED. A real version exists somewhere and we could not read
#: it -- a GROBID server that is not up, a binary not on PATH. This is an outage,
#: and it was previously indistinguishable from `by_design`.
VERSION_ABSENT_PROBE_FAILED = "probe_failed"

#: The version is computed by `prepare()`, which the AUDIT deliberately does not
#: call -- preparing a player starts servers and loads multi-gigabyte models.
#: Measured 2026-08-27: all 19 local-GGUF players reported "no installed version"
#: for exactly this reason. `LocalGgufAdapter.__init__` sets `_tool_version = None`
#: (local_gguf.py:280) and only `prepare()` fills it (local_gguf.py:427). Their
#: version is not absent, it is unbuilt -- and it is the richest one we have,
#: `<stem>/<quant>@<engine>@host-<fingerprint>`.
VERSION_ABSENT_NEEDS_PREPARE = "needs_prepare"

#: The adapter has not said. The default, and deliberately not a claim.
VERSION_ABSENT_UNDECLARED = "undeclared"

#: Kinds that mean something is WRONG rather than merely absent. The audit may
#: report these loudly; the others are legitimate states.
VERSION_ABSENT_PROBLEM_KINDS = frozenset({VERSION_ABSENT_PROBE_FAILED})


@dataclass(frozen=True)
class VersionAbsence:
    """(kind, detail) for a player with no resolvable tool version.

    `kind` is one of the VERSION_ABSENT_* constants and is what any gate keys on.
    `detail` is prose for a human reading the audit and must never be parsed.
    """
    kind: str
    detail: str = ""


@dataclass
class PlayerAdapter(ABC):
    player_id: str
    player_version: str
    player_type: str          # "platform" | "tool" | "ai-model" | "human-baseline"
    confidence_strategy: str  # "native" | "implicit-1.0" | "derived"
    deterministic: bool

    #: The argv this adapter actually spawned, for adapters that build their own
    #: command line instead of taking one from the registry's ``cli_command``.
    #: ``framework/runner.py`` falls back to this when writing
    #: ``provenance.containment``; leaving it None on a CLI adapter means every
    #: record it writes is classified ``uncontrolled`` no matter how carefully the
    #: subprocess was contained, because absence of the block IS the signal that a
    #: record predates containment. Only the argv is needed (the binary and its
    #: flags) — never the prompt.
    containment_argv: "Sequence[str] | None" = None

    def prepare(self) -> None:
        """One-time setup (start a server, check binaries, etc.). Default: no-op."""

    @abstractmethod
    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        """Receive a task envelope, return output conforming to arena's output.schema.json."""

    def version_absence_reason(self) -> "VersionAbsence":
        """WHY ``resolved_tool_version()`` returned None. Only the adapter knows.

        `framework audit --versions` reported 225 players as 127 OK / 0 DRIFT /
        98 UNCHECKED, and every one of the 98 read the same sentence: "adapter
        reported no installed version". Accurate, and useless -- it is the single
        string for at least four different facts, one of which is a live outage.

        THE AUDIT CANNOT WORK THIS OUT FROM OUTSIDE, and trying is a known trap:
        a class-override heuristic was attempted on 2026-08-26 and `pcurve`
        disproved it (pcurve.R loads only jsonlite, so it is pure base R BY
        DESIGN, indistinguishable from a broken probe unless the adapter says so).
        Hence a method on the adapter rather than a table in the audit.

        The kind comes from the CODE PATH, never from parsing message text --
        the same discipline `framework/audit.py::SplitNote` uses for split
        failures. `detail` is for a human and must never be matched on.

        Default is ``undeclared``: an adapter that has not thought about this
        says so, instead of inheriting a claim it has not earned. An unlabelled
        assumption is the defect this project has a standing rule against, and
        defaulting to "by design" would silently assert it for every adapter
        anyone adds in future.
        """
        return VersionAbsence(
            VERSION_ABSENT_UNDECLARED,
            f"{type(self).__name__} has not declared why it reports no version",
        )

    def resolved_tool_version(self) -> str | None:
        """Return the ACTUAL installed version of the underlying tool, detected at
        runtime (e.g. ``docpluck.__version__``, ``pdftotext -v``).

        The runner stamps this into every run record so the leaderboard can rank
        on the version a score was actually produced with — not the static label
        declared in registry.yaml (which drifts and silently ranks history). A
        drift check (``framework parity --versions``) compares this against the
        registry's declared ``player_version``.

        Default: ``None`` (no detectable underlying tool — e.g. an LLM whose
        version is the model id already in ``player_version``). Best-effort:
        adapters MUST swallow detection errors and return ``None`` rather than
        raise, so version detection never fails a task.
        """
        return None

    def last_usage(self) -> dict | None:
        """Token usage for the most recent ``play_task`` call, or ``None``.

        Shape: ``{"prompt_tokens": int, "completion_tokens": int,
        "total_tokens": int}`` — whichever the provider actually reported.

        WHY TOKENS AND NOT COST. Every OpenAI-compatible provider returns a
        ``usage`` block on every response and this framework discarded it for
        months, which is why ``cost_usd`` has been declared in the run-record
        schema, aggregated by ``framework/report.py``, and ``null`` in every
        published report since the field was created. Tokens are a MEASURED FACT
        about a run and stay true forever; a dollar figure is a claim about a
        price list that changes without notice, so baking one into a record makes
        the record wrong later. Cost is therefore derived at report time from a
        dated price table, and only the tokens are persisted.

        Default ``None``: local tool players (docpluck, GROBID, statcheck, the R
        tools) consume no tokens, and a CLI-subscription player is not billed per
        token either. ``None`` means "not applicable or not reported" — it must
        never be conflated with zero. Best-effort like ``resolved_tool_version``:
        adapters MUST swallow errors here rather than fail a task over telemetry.
        """
        return None

    def last_request_max_tokens(self) -> int | None:
        """The output ceiling WE put on the wire for the most recent ``play_task``.

        Recorded per record as ``provenance.request_max_tokens``. It is the direct
        companion to ``response_meta.finish_reason``: a completion cut off at the
        token limit fails JSON parsing and scores 0.0, and without the ceiling we
        sent, nobody reading the record can tell OUR configuration from the model's
        own limit. Measured 2026-09-09 on the Cohere Catalyst grant run — answering
        "was that our 4000 or their 4096?" for 7 truncated records required probing
        the live provider API by bisection, because the records could not say.

        Default ``None``, and ``None`` is a FIRST-CLASS ANSWER, never a placeholder:
        a CLI player, an R tool, docpluck and GROBID send no ceiling at all, so
        there is no number to report. A fabricated default here would be worse than
        the missing field it replaced — it would let a reader "confirm" that a
        truncation was our fault when we never set a limit. Best-effort like
        ``last_usage`` and ``resolved_tool_version``: adapters MUST swallow errors
        rather than fail a task over telemetry.
        """
        return None

    def last_request_body(self) -> dict | None:
        """The request body of the most recent model call, minus the prompt text.

        Schema v12 derives the record's ``call_config`` from this: the settings
        that were ON THE WIRE, not the registry's description of them (the two
        diverge exactly when ``openai_extra_body`` overrides a default). The
        messages are replaced by their sha256 so the body carries no task text.
        Default ``None``: an adapter that sends no model request has none.
        """
        return None

    def last_raw_response(self) -> object | None:
        """The provider's raw parsed response for the most recent call, or None.

        Stored gzipped beside the run file (schema v12) so a scorer fix can
        re-score without calling the model again. Default ``None``.
        """
        return None

    def cleanup(self) -> None:
        """One-time teardown (stop server, etc.). Default: no-op."""


class StubPassAdapter(PlayerAdapter):
    """Always returns {'label': 'ok'} — used by framework tests."""

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        return {"label": "ok"}


class StubFailAdapter(PlayerAdapter):
    """Always raises — used by framework tests for error-path coverage."""

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        raise RuntimeError("StubFailAdapter is designed to fail")


class StubSchemaViolationAdapter(PlayerAdapter):
    """Answers, but in a shape the output schema rejects.

    A THIRD failure mode, distinct from both stubs above and from neither of
    which it can be simulated: `StubFailAdapter` raises, so the runner never
    reaches schema validation at all, and `StubPassAdapter` always conforms. So
    until this existed no test could drive `_play_one` down the
    `output_schema_violation` branch, and the structural-adherence stamp on that
    branch could only be checked by reading the source — which is not evidence
    that the real call site executes it.

    Returns a valid `label` plus one unexpected key, so `fake_arena`'s
    `additionalProperties: false` rejects it as `extra_property`. That is the
    single most common class in this repo's real records (12 of 59).
    """

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        return {"label": "ok", "unexpected_key": 1}


_ADAPTER_CLASSES: dict[str, type[PlayerAdapter]] = {
    "StubPassAdapter": StubPassAdapter,
    "StubFailAdapter": StubFailAdapter,
    "StubSchemaViolationAdapter": StubSchemaViolationAdapter,
}


def register_adapter_class(name: str, cls: type[PlayerAdapter]) -> None:
    """Register a new adapter class. Called by tasks that add real adapters."""
    _ADAPTER_CLASSES[name] = cls


def resolve_rscript_binary(explicit: str | None = None) -> str:
    """Resolve the Rscript executable to invoke.

    Precedence: an explicit `rscript_binary` from the registry entry > the
    `RSCRIPT_BINARY` env var > bare `Rscript` from PATH.

    R is routinely installed on Windows WITHOUT being added to PATH (this dev box
    has R 4.4.0 under `C:/Program Files/R/R-4.4.0/bin/`), so a bare `Rscript`
    raises FileNotFoundError and every R reference tool silently records as an
    errored task. That is the worst possible failure for this project: the tool
    that CROSS-VALIDATES an arena's gold reports 0 instead of ~1.00, and a
    `--overwrite` re-run would replace good records with errors. The env var
    makes the R toolchain reproducible without mutating PATH.
    """
    if explicit:
        return explicit
    env = os.environ.get("RSCRIPT_BINARY", "").strip()
    if env:
        return env
    return "Rscript"


class RCliAdapter(PlayerAdapter):
    """Adapter that invokes an R script via Rscript and expects JSON output on stdout."""

    r_script: Path
    rscript_binary: str

    def __init__(self, *args, r_script: str | Path, rscript_binary: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_script = Path(r_script)
        self.rscript_binary = resolve_rscript_binary(rscript_binary)

    def resolved_tool_version(self) -> str | None:
        """Detect the installed version of the R package this adapter drives.

        Closes F7: without this every R reference tool resolved None, so
        ``framework audit --versions`` was blind to drift in exactly the
        deterministic tools that cross-validate arena gold. Best-effort by
        contract — returns None for pure base-R adapters and swallows all errors.
        """
        from players.adapters._tool_version import r_adapter_package, r_package_version
        pkg = r_adapter_package(self.r_script)
        if not pkg:
            return None
        return r_package_version(pkg, self.rscript_binary)

    def version_absence_reason(self) -> VersionAbsence:
        """Ask the SCRIPT, because this is the case that broke the heuristic.

        On 2026-08-26 a class-override table was written to explain the 98, and
        `pcurve` disproved it: `pcurve.R` loads only `jsonlite`, so it is pure
        base R BY DESIGN and has no package version to report. From outside, that
        is indistinguishable from an R package whose version probe failed. From
        inside it is a one-line question -- does this script name a package?
        """
        from players.adapters._tool_version import r_adapter_package
        try:
            pkg = r_adapter_package(self.r_script)
        except Exception as exc:
            return VersionAbsence(
                VERSION_ABSENT_PROBE_FAILED,
                f"could not read {self.r_script}: {type(exc).__name__}: {exc}",
            )
        if not pkg:
            return VersionAbsence(
                VERSION_ABSENT_BY_DESIGN,
                f"{Path(self.r_script).name} loads no R package - it is base R, "
                f"so there is no package version to report",
            )
        return VersionAbsence(
            VERSION_ABSENT_PROBE_FAILED,
            f"{Path(self.r_script).name} uses R package {pkg!r}, but "
            f"packageVersion({pkg!r}) did not return a version - the package is "
            f"probably not installed for this Rscript",
        )

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        proc = subprocess.run(
            [self.rscript_binary, str(self.r_script)],
            input=json.dumps(envelope),
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout_s, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{self.r_script.name} exited {proc.returncode}: {proc.stderr.strip()}")
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"{self.r_script.name} produced invalid JSON: {e}; raw: {proc.stdout[:300]}")


register_adapter_class("RCliAdapter", RCliAdapter)


# Test statistics that are reported WITHOUT degrees of freedom (so the
# df-presence heuristic below cannot catch them): a standard-normal z, the
# Mann-Whitney U, the Wilcoxon W, and the Bayes factor.
_DF_LESS_TESTS = {"z", "U", "W", "BF", "BF10"}


def _escimate_verdict(entry: dict) -> str:
    """Map one ESCImate result row to a typed verification verdict.

    Replaces a five-way OR that collapsed into a single `flagged_suspicious`
    boolean (2026-08-21). The five signals do not mean the same thing, and
    fusing them made an honest "I cannot check this" indistinguishable from
    "this is fabricated". Measured on the stored v2 run: of 53 flagged rows, 10
    were raised purely by `insufficient_data` / `extraction_suspect` /
    `highly_ambiguous` — abstentions scored as accusations.

    Precedence is deliberate. A row carrying BOTH a decision error and an
    ambiguity flag has still made a concrete claim about an inconsistency, so
    `inconsistent` wins over `unverifiable`; ESCImate emits the reason alongside
    (`decision_error_reason`, e.g. `reported_ns_computed_sig`).

    `df_arity_mismatch` is included here for the first time. We requested it in
    `docs/deception-detection-spec-for-escimate.md:78-80` for the
    `swapped_test_label` kind, ESCImate shipped it in 0.3.6, and it has been
    present on 101 of 101 stored extraction rows (true on 2) while no code read
    it — the proposed clause at spec line 143 was never added. It has therefore
    never influenced a published score.
    """
    if entry.get("decision_error") or entry.get("status") == "ERROR":
        return "inconsistent"
    if entry.get("df_arity_mismatch") or entry.get("sign_ci_violation"):
        # A test label whose df arity is impossible, or an estimate outside its
        # own interval. Both are structural contradictions, not uncertainty.
        return "inconsistent"
    if (entry.get("insufficient_data")
            or entry.get("extraction_suspect")
            or entry.get("ambiguity_level") == "highly_ambiguous"):
        return "unverifiable"
    if entry.get("status") == "WARN":
        # A WARN with no decision error and no missing data is a weaker claim of
        # inconsistency (e.g. a reported effect that disagrees with the computed
        # one). It is still a claim, so it is not an abstention.
        return "inconsistent"
    return "ok"


#: How long the one-shot ``/health`` version probe waits. See
#: ``HttpAdapter._probe_health_version`` for why it is not ``_is_up``'s 2s.
HEALTH_VERSION_TIMEOUT_S = 10


class HttpAdapter(PlayerAdapter):
    """POSTs the task input text to an HTTP endpoint, parses JSON response into output schema."""

    endpoint: str
    start_command: list[str] | None

    def __init__(self, *args, endpoint: str, start_command: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoint = endpoint
        self.start_command = start_command
        self._proc = None
        # Provenance read back from the SERVICE, never from registry.yaml. See
        # resolved_tool_version() for why the declared string cannot be trusted.
        self._last_tool_version: str | None = None
        self._last_engine_versions: dict | None = None
        # The /health fallback, probed AT MOST ONCE per adapter instance. See
        # resolved_tool_version(): a played task is stronger evidence, so this
        # only ever fires when no task has answered yet.
        self._health_probed: bool = False
        self._health_tool_version: str | None = None
        self._health_reachable: bool = False

    def prepare(self) -> None:
        # Best-effort: start the local server if it isn't already responding.
        if self._is_up():
            return
        if not self.start_command:
            return
        command = resolve_start_command(self.start_command)
        # Check the launcher exists BEFORE spawning. `cmd /c missing.bat` exits 1
        # without raising, so Popen would succeed and the only symptom would be
        # the 60s "did not become ready" timeout below — a message that blames
        # the server for a path that was never there.
        script = Path(command[-1])
        if script.suffix and not script.exists():
            raise FileNotFoundError(
                f"{self.player_id}: start_command points at {command[-1]}, which does not exist. "
                "The path is assembled from players/registry.yaml plus the ${NAME} values in "
                f"{LOCAL_PATHS_FILE} (gitignored) or the environment -- check whichever supplied "
                "the directory part."
            )
        self._proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            time.sleep(1)
            if self._is_up():
                return
        raise RuntimeError(f"ESCImate did not become ready at {self.endpoint}")

    def cleanup(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None

    def _health_url(self) -> str:
        return self.endpoint.rsplit("/api", 1)[0] + "/health"

    def _is_up(self) -> bool:
        try:
            r = requests.get(self._health_url(), timeout=2)
            return r.status_code < 500
        except Exception:
            return False

    def _probe_health_version(self) -> None:
        """Read the service's version from /health, WITHOUT playing a task.

        WHY THIS EXISTS (measured 2026-09-21). ``resolved_tool_version()`` read
        only the version carried on a ``/process-text`` RESPONSE, so an adapter
        that had not yet played a task returned None -- and
        ``framework audit --versions`` builds adapters and probes them WITHOUT
        running any task. The audit therefore printed::

            [UNCHECKED:undeclared] escimate  declared=0.7.6

        for the one player in this registry whose entire purpose is to
        self-report. It was blind to a live drift the whole time: the service was
        running 0.7.10 while the registry declared 0.7.6, and the audit that is
        documented as the guard against exactly that could not see it. Worse,
        that UNCHECKED row read identically whether the service was DOWN or
        merely idle -- two facts, one sentence, which is the false-green shape
        this module already has a VersionAbsence vocabulary to prevent.

        ``/health`` carries it: measured on the live service, the body is
        ``{"status":["healthy"], "version":["0.7.10"], "engine_versions":{...}}``
        (jsonlite boxes R length-1 vectors, hence the list unwrapping, the same
        shape ``_capture_provenance`` already handles).

        Best-effort and never raises: a provenance probe must not fail a task or
        an audit. It records BOTH outcomes -- whether the service answered at all
        (``_health_reachable``) and what version it named -- because
        ``version_absence_reason`` must distinguish an outage from a service that
        answered and named nothing.

        The timeout is deliberately LONGER than ``_is_up``'s 2s. Observed
        2026-09-21: two probes a second apart against the same running service
        returned 0.7.10 and then nothing, because this is a single-threaded R
        plumber process and a busy one can take longer than 2s to answer
        ``/health``. ``_is_up`` is polled sixty times in a readiness loop and can
        afford to be wrong once; this probe runs ONCE and its answer is reported
        to a human as "the service is not running", so a 2s timeout here would
        publish a transient stall as an outage.
        """
        self._health_probed = True
        try:
            r = requests.get(self._health_url(), timeout=HEALTH_VERSION_TIMEOUT_S)
            if r.status_code >= 500:
                return
            self._health_reachable = True
            body = r.json()
        except Exception:
            return
        if not isinstance(body, dict):
            return
        v = body.get("version")
        if isinstance(v, list):
            v = v[0] if v else None
        if not (isinstance(v, str) and v.strip()):
            engines = body.get("engine_versions")
            if isinstance(engines, dict):
                v = engines.get("effectcheck")
                if isinstance(v, list):
                    v = v[0] if v else None
        if isinstance(v, str) and v.strip():
            self._health_tool_version = v.strip()

    def version_absence_reason(self) -> "VersionAbsence":
        """Why no version: an OUTAGE, or a service that named none. Never both.

        Reached only when :meth:`resolved_tool_version` returned None, which for
        this adapter means the /health probe above also came back empty. Both
        branches are ``probe_failed`` -- a version genuinely exists on the other
        side of this endpoint in either case, and we could not read it -- but the
        detail says which, so a reader is not left guessing at a down server.
        """
        if not self._health_probed:
            self._probe_health_version()
        if not self._health_reachable:
            return VersionAbsence(
                VERSION_ABSENT_PROBE_FAILED,
                f"{self._health_url()} did not answer within "
                f"{HEALTH_VERSION_TIMEOUT_S}s -- the service is not running, or "
                "is not responding. Start it (start_command) and re-run the probe.",
            )
        return VersionAbsence(
            VERSION_ABSENT_PROBE_FAILED,
            f"{self._health_url()} answered but named no version. This service "
            "reports one on `version` and on `engine_versions`; losing it turns "
            "every drift check into a green that tests nothing.",
        )

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        text = envelope["input"]["text"]
        r = requests.post(self.endpoint, json={"text": text}, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        self._capture_provenance(data)
        return self._normalize(data, text)

    def _capture_provenance(self, data: dict) -> None:
        """Read the service's OWN report of what it is running.

        Best-effort and never raises: provenance must not fail a task.
        """
        try:
            v = data.get("effectcheck_version")
            if isinstance(v, list):      # jsonlite may box a length-1 vector
                v = v[0] if v else None
            if isinstance(v, str) and v.strip():
                self._last_tool_version = v.strip()
            ev = data.get("engine_versions")
            if isinstance(ev, dict) and ev:
                self._last_engine_versions = {
                    k: (val[0] if isinstance(val, list) and val else val)
                    for k, val in ev.items()
                }
        except Exception:
            pass

    def resolved_tool_version(self) -> str | None:
        """The version the SERVICE reports, not the one registry.yaml declares.

        ``worker/plumber.R:39`` computes ``EFFECTCHECK_VERSION`` from
        ``utils::packageVersion("effectcheck")``, so it is measured from the
        loaded package and cannot drift from it. It is returned on the body of
        every ``/process`` and ``/process-text`` response (``plumber.R:412,479``).

        This closes F7. It matters because ``start_app.bat:74-81`` REINSTALLS the
        package whenever ``effectcheck/DESCRIPTION`` differs from what is
        installed: on 2026-08-21 the source said 0.7.6 while
        ``players/registry.yaml`` still said 0.6.13, so a run would have executed
        0.7.6 and stamped the record 0.6.13 — a wrong version at write time on a
        player marked ``deterministic: true``. A declared string is an intent; this
        is evidence. ``framework audit`` compares the two and reports the drift.

        TWO SOURCES, IN ORDER OF EVIDENTIAL STRENGTH.

        1. The version carried on the response to a task we actually played.
           That is the version that produced THIS score, and the runner stamps
           it onto the record, so it is preferred whenever it exists.
        2. Failing that, ``/health`` (see :meth:`_probe_health_version`), probed
           at most once per adapter instance. Added 2026-09-21: without it the
           audit -- which never plays a task -- reported UNCHECKED for the only
           self-reporting player in the registry, and was blind to the live
           0.7.6-declared / 0.7.10-installed drift it is documented to catch.

        The order matters for what lands on a RECORD. Source 1 is unchanged, so
        a record of a task that answered still carries the answering service's
        own report, exactly as before. Source 2 can only reach a record of a task
        that produced no answer at all, where the alternative was None.
        """
        if self._last_tool_version:
            return self._last_tool_version
        if not self._health_probed:
            self._probe_health_version()
        return self._health_tool_version

    def _normalize(self, escimate_response: dict, text: str) -> dict:
        """Map ESCImate's response shape into our output schema.

        ESCImate's API surface is documented publicly at
        https://escimate.app/api-docs, and in API.md inside the service's own
        (private) repository. The path to that checkout is deliberately not
        written here: this module is published to the public mirror, and naming
        it would disclose the layout of a private repository. See
        `resolve_start_command` and the gitignored `players/local_paths.yaml`.

        ESCImate result rows use `stat_value` / `effect_reported` for the
        numeric value and `raw_text` for the matched substring; `test_type`
        / `check_type` carry the kind. We map those into our output schema
        and inject a generic `value` field for the scorer.
        """
        extractions = []
        for entry in escimate_response.get("results", []) or []:
            raw = entry.get("raw_text") or ""
            val_num = entry.get("stat_value")
            if val_num is None:
                val_num = entry.get("effect_reported")
            anchor = raw if raw else (str(val_num) if val_num is not None else "")
            cs = text.find(anchor) if anchor else -1
            if cs < 0:
                cs, ce, span_text = 0, 0, ""
            else:
                ce, span_text = cs + len(anchor), anchor
            test_type = entry.get("test_type") or ""
            check_type = entry.get("check_type") or ""
            # Disambiguate nhst_stat vs effect_size by degrees-of-freedom presence.
            # A statistic reported WITH df is a hypothesis-test statistic (t, F,
            # chi2, a correlation r(df), Kruskal-Wallis H, Cochran Q); z / U / W /
            # BF are genuinely df-less tests; everything else df-less is a bare
            # effect size (Cohen's d, eta2, OR/RR, or a correlation reported with
            # only a CI). This is what lets the ambiguous symbols land correctly:
            # "r(28) = .." is an nhst_stat, while a bare "r = -.85" (no df) is an
            # effect_size — previously every "r" was mislabelled nhst_stat,
            # forcing a kind disagreement against the gold.
            has_df = entry.get("df1") is not None
            kind = ("nhst_stat"
                    if check_type == "p_value" or has_df or test_type in _DF_LESS_TESTS
                    else "effect_size")
            fields = dict(entry)
            if "value" not in fields and val_num is not None:
                fields["value"] = val_num
            verdict = _escimate_verdict(entry)
            suspicious = verdict != "ok"
            extractions.append({
                "span": {"text": span_text, "char_start": cs, "char_end": ce},
                "kind": kind,
                "fields": fields,
                # NOT a measurement. ESCImate has no probability to report, so this
                # is a constant this framework invents. It is kept only because the
                # output schema requires the field; the stats-extraction scorer no
                # longer multiplies calibration into the published score, precisely
                # because ECE over `{0.4 x53, 1.0 x48}` measured this line.
                "confidence": 0.4 if suspicious else 1.0,
                "verdict": verdict,
                # Deprecated, kept one version for players/scorers not yet updated.
                "flagged_suspicious": suspicious,
            })
        return {"extractions": extractions, "player_strategy_notes": "escimate (local)"}


register_adapter_class("HttpAdapter", HttpAdapter)


ADAPTER_EXTRA_KWARGS = {
    "r_script", "rscript_binary", "endpoint", "start_command",
    "cli_command",
    "cli_json_envelope", "prompt_template_path",
    # Claude CLI output ceiling, passed as CLAUDE_CODE_MAX_OUTPUT_TOKENS (d-60b28c).
    "cli_max_output_tokens",
    "catalog_path", "api_key_env", "api_url_env",
    # pdf-text-fidelity-v1 adapters:
    "docpluck_level", "pdftotext_binary", "output_message_flag",
    # liteparse OCR/render config. ADDED 2026-08-19: these keys had been declared
    # in registry.yaml since the liteparse players landed but were never on this
    # list, so `build_adapter` dropped them and every liteparse player ran with
    # the adapter defaults. `liteparse-no-ocr` therefore ran WITH OCR and
    # published a score bit-identical to `liteparse-default`
    # (0.7240294726242898, n=70) — the site showed ranks 5 and 6 as an
    # OCR-on/OCR-off comparison that never happened. See
    # framework/tests/test_registry_keys_reach_adapters.py, which is generic over
    # the registry so the next tool cannot repeat this.
    "ocr_enabled", "ocr_language", "dpi",
    # Docling (players/adapters/_docling_common.py). Only these three: the
    # pipeline options that decide EGRESS (enable_remote_services,
    # do_picture_description) are hard-coded in the adapter and deliberately not
    # registry-configurable, so a YAML edit cannot turn a gate-approved local
    # player into a remote one -- the GROBID-endpoint hole, closed by
    # construction rather than by review.
    "docling_vlm", "docling_table_mode", "docling_artifacts_path",
    # Out-of-process extractors in their own pinned venvs (players/tool_envs/<tool>/,
    # added 2026-10-07): Docling 2.135, Marker 2.0, MinerU 4.0. Names WHICH venv,
    # never a path -- the path is resolved in players/adapters/_tool_worker.py.
    "tool_env",
    # GROBID 0.9.1 entries send consolidateFunders explicitly; absent on the 0.8.1
    # entries so their published request stays byte-identical.
    "consolidate_funders",
    # pdf-reference-parsing-v1 CERMINE adapter:
    "cermine_jar", "java_binary",
    # AnyStyle / CERMINE run inside a local container (players/docker/). ADDED
    # 2026-09-28 in the same change that introduced the key -- and only after its
    # first runs went out without it: dropped here, both players fell back to a
    # host `anystyle` / `java` that is not installed.
    "docker_container",
    # stats-from-pdf-v1 pipelines (players/adapters/stats_pipeline.py, 2026-10-17): which
    # extractor feeds which checker, and the LOCAL service URLs (the adapter refuses any
    # non-loopback URL, so a YAML edit cannot send a held-out PDF off the machine).
    "extractor", "checker", "grobid_endpoint", "docpluck_url", "docling_python", "metacheck_lib",
    # OpenAIChatCompletionsAdapter: how a PDF task reaches a text-only chat API
    # ("pdftotext"). Dropped, the adapter JSON-dumped the envelope and the model
    # received a megabyte of base64 instead of the paper (measured 2026-09-28:
    # a Gemini pilot answered with references that are in no task document).
    "document_to_text",
    # regcheck multi-provider adapter (prereg-deviation-v1): pins the LLM provider
    # + model per registry player so one adapter class serves distinct
    # regcheck-groq / regcheck-deepseek / regcheck-openai players.
    "regcheck_client", "regcheck_model",
    # Human evaluators (players/adapters/human_replay.py, T6 2026-09-28): the
    # per-person answers file written by tools/human_score.py.
    "responses_path",
    # Kaggle Benchmarks model-proxy adapter (players/adapters/model_proxy.py):
    # one OpenAI-compatible client serving many models, so the model id and the
    # prompt file are per-player registry data rather than per-adapter code.
    "proxy_model", "proxy_temperature",
    # Generic named-model OpenAI-compatible free-tier adapter
    # (players/adapters/openai_compatible.py). Used for direct provider endpoints
    # or local gateways only when the backend model is stable and explicit.
    "openai_model", "openai_base_url_env", "openai_api_key_env",
    "openai_available_models_env", "openai_chat_path", "openai_temperature",
    # Extra request-body keys, needed to switch OFF chain-of-thought on reasoning
    # models (NVIDIA Nemotron 3.x default it ON, which cost 254s vs 3.7s per task).
    "openai_extra_body",
    "allow_rotating_model",
    # How an arena-owned prompt is filled ("replace" | "str.format"), schema v12.
    "prompt_format",
    # Gemini native generateContent adapter (players/adapters/gemini_native.py).
    "gemini_model", "gemini_api_key_env", "gemini_temperature", "gemini_seed",
    "gemini_thinking_level", "gemini_max_output_tokens", "gemini_response_json",
    "gemini_max_backoff_s",
    # Google Antigravity CLI headless mode (players/adapters/antigravity_cli.py).
    # This is a CLI-agent route, distinct from the legacy Gemini CLI and from
    # direct Gemini API keys.
    # Anthropic Messages API players (players/adapters/anthropic_api.py) -- the
    # one approved API exception (owner, 2026-10-23). Effort is required there.
    "anthropic_model", "anthropic_effort", "anthropic_max_tokens",
    "anthropic_thinking", "anthropic_output_coercer", "anthropic_document_to_text",
    "agy_model", "agy_effort", "agy_agent", "agy_print_timeout",
    "agy_json_schema", "agy_json_schema_path",
    # Local GGUF player served by llama.cpp llama-server on loopback
    # (players/adapters/local_gguf.py). `endpoint` is already listed above and is
    # the ONLY source of the URL for that class, so the held-out egress gate and
    # the request path cannot disagree. `gguf_path` is what the provenance digest
    # is taken from; `gguf_quant` is the escape hatch for a file whose name does
    # not carry a recognisable quant token.
    "gguf_path", "gguf_quant",
    # code-translation-r-v1 vendored-tool adapter: the SPSS/Stata->R converters
    # are R packages, so their outputs are captured once and committed as
    # version-pinned fixtures rather than invoked live (user decision 2026-08-03).
    "fixture_dir", "fixture_tool_version",
}


class SubprocessCliAdapter(PlayerAdapter):
    """Generic CLI adapter: spawn a CLI binary, feed it a templated prompt + envelope on stdin, parse JSON-only stdout."""

    def version_absence_reason(self) -> VersionAbsence:
        """55 of the 98 UNCHECKED players are this class, and all 55 are fine.

        A CLI-driven LLM has no separately-installed tool whose version could
        drift from the label: the model id IS the version, and it is already in
        `player_version`. This is the case that made the old single sentence
        defensible -- and the reason it hid the five GROBID outages sitting
        beside it under identical wording.

        NOTE the CLI BINARY does have a version (`claude --version`), and that is
        recorded separately in the player label as `via claude-code@X.Y` because a
        CLI player is a different instrument from an API call to the same weights.
        That is a player-identity concern, not tool-version drift.
        """
        return VersionAbsence(
            VERSION_ABSENT_BY_DESIGN,
            "a CLI-driven model's version is the model id already in "
            "player_version; there is no separate installed tool to detect",
        )

    cli_command: list[str]
    prompt_template_path: Path

    def __init__(self, *args, cli_command: list[str], prompt_template_path: str | Path,
                 cli_json_envelope: str | None = None,
                 cli_max_output_tokens: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_command = list(cli_command)
        #: The output ceiling for the `claude` CLI, sent as
        #: CLAUDE_CODE_MAX_OUTPUT_TOKENS. Measured 2026-09-30: left unset the CLI
        #: stops at 64000 output tokens ("exceeded the 64000 output token
        #: maximum"), and at least four publication-bias v2 answers stopped at
        #: exactly 64000 -- OUR ceiling, published as the model's failure. Declared
        #: per player so it is stamped into provenance.request_max_tokens like an
        #: API player's max_tokens (d-60b28c). Only honoured for `claude`.
        if cli_max_output_tokens is not None and (
                not isinstance(cli_max_output_tokens, int) or isinstance(cli_max_output_tokens, bool)
                or cli_max_output_tokens <= 0):
            raise ValueError(f"cli_max_output_tokens must be a positive int, got {cli_max_output_tokens!r}")
        if cli_max_output_tokens is not None and hermetic.effective_binary(self.cli_command) != "claude":
            raise ValueError("cli_max_output_tokens is only defined for the claude CLI")
        self.cli_max_output_tokens = cli_max_output_tokens
        self.prompt_template_path = Path(prompt_template_path)
        #: Opt-in wrapper format that carries token usage. "claude" makes this
        #: adapter append `--output-format json`, so the CLI returns an envelope
        #: with usage/stop_reason/cost around the answer instead of bare text.
        #: Left None for every other CLI, whose output shape stays untouched.
        self.cli_json_envelope = cli_json_envelope
        self._template: str | None = None
        self._resolved_command: list[str] | None = None
        self._spawn_kwargs: dict = {}
        self._fresh_session = False
        self._last_usage: dict | None = None
        self._last_meta: dict | None = None

    def prepare(self) -> None:
        self._template = self.prompt_template_path.read_text(encoding="utf-8")
        # Resolve argv[0] via shutil.which so Windows finds .cmd / .bat shims
        # (e.g. npm-installed CLIs like `gemini`, `claude`).
        import shutil
        first = self.cli_command[0]
        resolved = shutil.which(first)
        if resolved is None:
            raise RuntimeError(f"CLI binary not found on PATH: {first}")
        self._resolved_command = [resolved] + self.cli_command[1:]
        if self.cli_json_envelope == "claude" and "--output-format" not in self._resolved_command:
            self._resolved_command += ["--output-format", "json"]
        # CONTAINMENT. An agentic CLI spawned with the runner's cwd runs INSIDE
        # this repo with its full tool set — confirmed 2026-08-16 by planting an
        # unguessable nonce and getting it back verbatim. `framework/hermetic.py`
        # is the single place that decides what a player subprocess inherits;
        # never assemble these flags here. It REFUSES an unprofiled agentic CLI
        # rather than letting it run open.
        # A CLI that carries a conversation between invocations gets its
        # workspace chosen PER TASK instead of once here — see
        # `hermetic.needs_fresh_session`. Baking one workspace in at prepare()
        # is what let eight consecutive tasks share a chat and return the same
        # stale answer.
        self._fresh_session = hermetic.needs_fresh_session(self._resolved_command)
        if not self._fresh_session:
            self._resolved_command, self._spawn_kwargs = hermetic.spawn_kwargs(
                self._resolved_command, allow_tools=(),
            )

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._template is None or self._resolved_command is None:
            self.prepare()
        self._last_usage = None
        self._last_meta = None
        input_payload = envelope["input"]
        if isinstance(input_payload, dict) and "text" in input_payload:
            input_str = input_payload["text"]
        else:
            input_str = json.dumps(input_payload, ensure_ascii=False)
        prompt = self._template.replace("{{INPUT_TEXT}}", input_str)
        # The prompt goes on STDIN wherever the CLI accepts it, never as a
        # positional argument: `--tools` is variadic, so `--tools "" "<prompt>"`
        # swallows the prompt as a second tool name and the CLI dies asking for
        # input.
        #
        # `cursor-agent` is the exception — it has no stdin path at all ("Error:
        # No prompt provided for print mode"), so its prompt must be argv. The
        # hermetic profile records which, because that is a property of the CLI
        # and not of any one player. Hardening deliberately ends on a boolean
        # flag so this trailing positional is safe.
        workspace = None
        if getattr(self, "_fresh_session", False):
            # New workspace per task, so no two tasks can share a conversation.
            import uuid as _uuid
            workspace = hermetic.hermetic_cwd(tag=f"task-{_uuid.uuid4().hex[:12]}")
            argv, self._spawn_kwargs = hermetic.spawn_kwargs(
                self._resolved_command, allow_tools=(), cwd=workspace,
            )
        else:
            argv = list(self._resolved_command)

        stdin_payload: str | None = prompt
        if not hermetic.takes_stdin_prompt(argv):
            if hermetic.prompt_goes_in_a_file(argv):
                # The prompt is written VERBATIM to the workspace so every
                # player is scored on byte-identical instructions — collapsing
                # newlines to fit argv would quietly give this player a
                # different prompt from all the others, which is exactly the
                # comparison the benchmark exists to make.
                workspace = workspace or Path(self._spawn_kwargs.get("cwd", "."))
                prompt_file = Path(workspace) / "task_prompt.txt"
                prompt_file.write_text(prompt, encoding="utf-8")
                argv.append(
                    f"Read the file {prompt_file.name} in your current workspace "
                    f"and follow the instructions in it exactly. Output only what "
                    f"that file asks for."
                )
            else:
                argv.append(prompt)
            stdin_payload = None
        spawn = dict(self._spawn_kwargs)
        if self.cli_max_output_tokens is not None:
            env = dict(spawn.get("env") or os.environ)
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.cli_max_output_tokens)
            spawn["env"] = env
        proc = subprocess.run(
            argv, input=stdin_payload, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout_s, check=False, **spawn,
        )
        if proc.returncode != 0:
            reason = proc.stderr.strip()[:300]
            # The Claude CLI reports its own failures (usage limit, API error) in the
            # stdout JSON envelope and leaves stderr empty (d-daa31d).
            envelope_error = _cli_envelope_error(proc.stdout)
            if envelope_error:
                reason = f"{reason} | {envelope_error}" if reason else envelope_error
            raise RuntimeError(f"{self.cli_command[0]} exited {proc.returncode}: {reason}")
        if self.cli_json_envelope == "claude":
            return _extract_json_value(self._unwrap_claude_envelope(proc.stdout))
        if self.cli_json_envelope == "cursor":
            return _extract_json_value(self._unwrap_cursor_envelope(proc.stdout))
        # Permissively recover the JSON value: CLIs (notably Opus) may wrap it in
        # ```json fences, prepend prose, or append a trailing sentence.
        return _extract_json_value(proc.stdout)

    def _unwrap_claude_envelope(self, stdout: str) -> str:
        """Pull the answer out of `claude --output-format json`, keeping the telemetry.

        The CLI returns `{result, usage, stop_reason, modelUsage, total_cost_usd, ...}`.
        Without this the framework saw only bare text and every Claude player's token
        count was unrecoverable — the same data loss the HTTP adapter had.

        `total_cost_usd` is recorded as PROVIDER-REPORTED and never as money spent:
        these players run on a Claude Max subscription, where the marginal cost of a
        task is not a token price at all. It is an API-equivalent figure, useful for
        comparing against metered players and misleading if read as spend.
        """
        try:
            env = json.loads(stdout)
        except ValueError:
            return stdout  # not an envelope; fall through to permissive extraction
        if not isinstance(env, dict) or "result" not in env:
            return stdout

        u = env.get("usage") or {}
        if isinstance(u, dict):
            inp = u.get("input_tokens")
            out = u.get("output_tokens")
            cache_read = u.get("cache_read_input_tokens")
            cache_new = u.get("cache_creation_input_tokens")
            usage: dict = {}
            # Anthropic splits the input side into fresh / cache-read / cache-write.
            # prompt_tokens is their SUM, because all three are input the model
            # processed; the split is kept separately since it drives cost.
            parts = [v for v in (inp, cache_read, cache_new) if isinstance(v, int)]
            if parts:
                usage["prompt_tokens"] = sum(parts)
            if isinstance(out, int):
                usage["completion_tokens"] = out
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            if isinstance(cache_read, int):
                usage["cache_read_tokens"] = cache_read
            if isinstance(cache_new, int):
                usage["cache_creation_tokens"] = cache_new
            self._last_usage = usage or None

        meta: dict = {"attempts": 1}
        if isinstance(env.get("stop_reason"), str):
            # Anthropic's name for finish_reason. Normalised so one field answers
            # "was this truncated?" regardless of provider.
            meta["finish_reason"] = env["stop_reason"]
        mu = env.get("modelUsage")
        if isinstance(mu, dict) and mu:
            served = self._pick_served_model(mu)
            if isinstance(served, str):
                meta["served_model"] = served
        if isinstance(env.get("session_id"), str):
            meta["provider_request_id"] = env["session_id"]
        if isinstance(env.get("total_cost_usd"), (int, float)):
            meta["provider_reported_cost_usd"] = float(env["total_cost_usd"])
        self._last_meta = meta
        return env["result"] if isinstance(env["result"], str) else json.dumps(env["result"])

    def _unwrap_cursor_envelope(self, stdout: str) -> str:
        """Pull the answer out of `cursor-agent --print --output-format json`.

        Shape (measured 2026-08-19, cursor-agent 2026.08.11-e8db854):
            {"type":"result","subtype":"success","is_error":false,
             "duration_ms":14319,"result":"ok","session_id":"...",
             "request_id":"...",
             "usage":{"inputTokens":17991,"outputTokens":68,
                      "cacheReadTokens":2816,"cacheWriteTokens":0}}

        Different from Anthropic's envelope in every name, so it needs its own
        unwrap rather than a shared one that guesses.

        WORTH KNOWING BEFORE BUDGETING A RUN: `inputTokens` was **17,991 for a
        prompt of five words**. That is the CLI's own system prompt and tool
        definitions, charged on every single task, and it dwarfs any arena
        prompt here (the largest is ~2,700 characters). Cost is therefore
        ~18k x n_tasks almost regardless of which arena you pick.
        """
        # cursor-agent appends a trailing "Shell cwd was reset to ..." line
        # after the JSON, so a strict json.loads FAILS on real output. Falling
        # straight through to the permissive extractor would still recover the
        # answer while silently dropping `usage` — which is exactly how
        # cost_usd came out null in 127 published reports. Recover the ENVELOPE
        # permissively instead, then read usage off it.
        try:
            env = json.loads(stdout)
        except ValueError:
            try:
                env = _extract_json_value(stdout)
            except ValueError:
                return stdout
        if not isinstance(env, dict) or "result" not in env:
            return stdout

        u = env.get("usage")
        if isinstance(u, dict):
            inp = u.get("inputTokens")
            out = u.get("outputTokens")
            cache_read = u.get("cacheReadTokens")
            cache_new = u.get("cacheWriteTokens")
            usage: dict = {}
            # prompt_tokens is the SUM of fresh + cache-read + cache-write: all
            # three are input the model processed. The split is kept separately
            # because it is what drives price.
            parts = [v for v in (inp, cache_read, cache_new) if isinstance(v, int)]
            if parts:
                usage["prompt_tokens"] = sum(parts)
            if isinstance(out, int):
                usage["completion_tokens"] = out
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            if isinstance(cache_read, int):
                usage["cache_read_tokens"] = cache_read
            if isinstance(cache_new, int):
                usage["cache_creation_tokens"] = cache_new
            self._last_usage = usage or None

        meta: dict = {"attempts": 1}
        if isinstance(env.get("request_id"), str):
            meta["provider_request_id"] = env["request_id"]
        if env.get("is_error") is True:
            meta["finish_reason"] = "error"
        elif isinstance(env.get("subtype"), str):
            meta["finish_reason"] = env["subtype"]
        served = self._requested_model()
        if served:
            # cursor-agent does not report which model served the request, so
            # this is the id we ASKED for, not a confirmation. Recorded because
            # the alternative is no provenance at all; never read as measured.
            meta["served_model"] = served
        self._last_meta = meta
        return env["result"] if isinstance(env["result"], str) else json.dumps(env["result"])

    def _requested_model(self) -> str | None:
        """The model id this player asked for, read off its own argv."""
        cmd = self._resolved_command or self.cli_command
        for i, tok in enumerate(cmd):
            if tok == "--model" and i + 1 < len(cmd):
                return cmd[i + 1]
        return None

    def _pick_served_model(self, model_usage: dict) -> str | None:
        """Which entry of `modelUsage` is THIS player's model.

        Not `next(iter(...))`, which is what this did until 2026-08-19.
        Measured that day: `claude --print --model claude-sonnet-5` returns TWO
        modelUsage entries — the requested `claude-sonnet-5` AND an auxiliary
        `claude-haiku-4-5-20251001` call (523 in / 13 out) that the CLI makes on
        every invocation regardless of the model asked for. Taking the first key
        therefore records whichever the CLI happened to serialise first.

        Checked against every stored sonnet/opus record before changing this:
        138 records, all correct, so no published `served_model` was ever wrong.
        The defect is latent and order-dependent, which is exactly the kind that
        surfaces later as an unexplained provenance flip.

        Resolution order: exact match on the requested id, then `canonicalModel`,
        then the entry that consumed the most tokens (the auxiliary call is
        always tiny). Returns None rather than guessing when nothing matches.
        """
        requested = self._requested_model()
        if requested and requested in model_usage:
            return requested
        if requested:
            for name, detail in model_usage.items():
                if isinstance(detail, dict) and detail.get("canonicalModel") == requested:
                    return name
        def _tokens(detail: object) -> int:
            if not isinstance(detail, dict):
                return 0
            return sum(
                v for k, v in detail.items()
                if k.endswith(("Tokens", "InputTokens", "OutputTokens"))
                and isinstance(v, int)
            )
        ranked = sorted(model_usage.items(), key=lambda kv: _tokens(kv[1]), reverse=True)
        return ranked[0][0] if ranked and _tokens(ranked[0][1]) else None

    def last_usage(self) -> dict | None:
        return self._last_usage

    def last_response_meta(self) -> dict | None:
        return self._last_meta

    def last_request_max_tokens(self) -> int | None:
        """The declared CLI ceiling, or None when we set none (the CLI default then
        applies, and a fabricated number would claim a limit we never chose)."""
        return self.cli_max_output_tokens


register_adapter_class("SubprocessCliAdapter", SubprocessCliAdapter)


def build_adapter(registry_entry: dict) -> PlayerAdapter:
    """Construct a PlayerAdapter from a registry entry."""
    cls_name = registry_entry["adapter_class"]
    cls = _ADAPTER_CLASSES.get(cls_name)
    if cls is None:
        raise ValueError(f"Unknown adapter_class: {cls_name}. Known: {sorted(_ADAPTER_CLASSES)}")
    base_kwargs = {k: registry_entry[k] for k in ("player_id", "player_version", "player_type", "confidence_strategy", "deterministic")}
    extra_kwargs = {k: registry_entry[k] for k in ADAPTER_EXTRA_KWARGS if k in registry_entry}
    return cls(**base_kwargs, **extra_kwargs)
