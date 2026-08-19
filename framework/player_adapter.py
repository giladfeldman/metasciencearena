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

import requests

from framework import hermetic


def resolve_start_command(command: list[str]) -> list[str]:
    """Expand `${VIBE_ROOT}` (and `~`) in a player's `start_command`.

    The portfolio root moved out of Dropbox on 2026-08-03 and may move again, so
    `players/registry.yaml` must never spell an absolute user path — see
    `Vibe/CLAUDE.md`. Resolution order matches that rule: `$VIBE_ROOT`, else
    `$HOME/Vibe`.

    Paths come back with forward slashes so a registry entry reads identically on
    every platform and the value a test asserts on is the value Popen receives.
    """
    vibe_root = os.environ.get("VIBE_ROOT") or str(Path.home() / "Vibe")
    resolved = []
    for token in command:
        expanded = token.replace("${VIBE_ROOT}", vibe_root).replace("$VIBE_ROOT", vibe_root)
        expanded = os.path.expanduser(expanded)
        if expanded != token or "/" in expanded or os.sep in expanded:
            expanded = expanded.replace(os.sep, "/")
        resolved.append(expanded)
    return resolved


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
    def _first_balanced(open_ch: str, close_ch: str):
        search_from = 0
        while True:
            start = text.find(open_ch, search_from)
            if start == -1:
                return None
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
                            break  # this block is malformed; advance past it
            search_from = start + 1

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        result = _first_balanced(open_ch, close_ch)
        if result is not None:
            return result
    raise ValueError(
        f"no parseable JSON in CLI output (len={len(raw)}): {raw.strip()[:200]!r}"
    )


@dataclass
class PlayerAdapter(ABC):
    player_id: str
    player_version: str
    player_type: str          # "platform" | "tool" | "ai-model" | "human-baseline"
    confidence_strategy: str  # "native" | "implicit-1.0" | "derived"
    deterministic: bool

    def prepare(self) -> None:
        """One-time setup (start a server, check binaries, etc.). Default: no-op."""

    @abstractmethod
    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        """Receive a task envelope, return output conforming to arena's output.schema.json."""

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


_ADAPTER_CLASSES: dict[str, type[PlayerAdapter]] = {
    "StubPassAdapter": StubPassAdapter,
    "StubFailAdapter": StubFailAdapter,
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


class HttpAdapter(PlayerAdapter):
    """POSTs the task input text to an HTTP endpoint, parses JSON response into output schema."""

    endpoint: str
    start_command: list[str] | None

    def __init__(self, *args, endpoint: str, start_command: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoint = endpoint
        self.start_command = start_command
        self._proc = None

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
                "Check the ${VIBE_ROOT} entry in players/registry.yaml."
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

    def _is_up(self) -> bool:
        health = self.endpoint.rsplit("/api", 1)[0] + "/health"
        try:
            r = requests.get(health, timeout=2)
            return r.status_code < 500
        except Exception:
            return False

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        text = envelope["input"]["text"]
        r = requests.post(self.endpoint, json={"text": text}, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        return self._normalize(data, text)

    def _normalize(self, escimate_response: dict, text: str) -> dict:
        """Map ESCImate's response shape into our output schema.

        ESCImate's API surface is documented at
        https://escimate.app/api-docs and at
        C:/Users/filin/Dropbox/Vibe/MetaScienceTools/ESCIcheckapp/API.md.

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
            suspicious = bool(
                entry.get("decision_error")
                or entry.get("extraction_suspect")
                or entry.get("insufficient_data")
                or entry.get("status") in {"WARN", "ERROR"}
                or entry.get("ambiguity_level") == "highly_ambiguous"
            )
            extractions.append({
                "span": {"text": span_text, "char_start": cs, "char_end": ce},
                "kind": kind,
                "fields": fields,
                "confidence": 0.4 if suspicious else 1.0,
                "flagged_suspicious": suspicious,
            })
        return {"extractions": extractions, "player_strategy_notes": "escimate (local)"}


register_adapter_class("HttpAdapter", HttpAdapter)


ADAPTER_EXTRA_KWARGS = {
    "r_script", "rscript_binary", "endpoint", "start_command",
    "cli_command",
    "cli_json_envelope", "prompt_template_path",
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
    # pdf-reference-parsing-v1 CERMINE adapter:
    "cermine_jar", "java_binary",
    # regcheck multi-provider adapter (prereg-deviation-v1): pins the LLM provider
    # + model per registry player so one adapter class serves distinct
    # regcheck-groq / regcheck-deepseek / regcheck-openai players.
    "regcheck_client", "regcheck_model",
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
    # Google Antigravity CLI headless mode (players/adapters/antigravity_cli.py).
    # This is a CLI-agent route, distinct from the legacy Gemini CLI and from
    # direct Gemini API keys.
    "agy_model", "agy_effort", "agy_agent", "agy_print_timeout",
    "agy_json_schema", "agy_json_schema_path",
    # code-translation-r-v1 vendored-tool adapter: the SPSS/Stata->R converters
    # are R packages, so their outputs are captured once and committed as
    # version-pinned fixtures rather than invoked live (user decision 2026-08-03).
    "fixture_dir", "fixture_tool_version",
}


class SubprocessCliAdapter(PlayerAdapter):
    """Generic CLI adapter: spawn a CLI binary, feed it a templated prompt + envelope on stdin, parse JSON-only stdout."""

    cli_command: list[str]
    prompt_template_path: Path

    def __init__(self, *args, cli_command: list[str], prompt_template_path: str | Path,
                 cli_json_envelope: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_command = list(cli_command)
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
        proc = subprocess.run(
            argv, input=stdin_payload, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout_s, check=False, **self._spawn_kwargs,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{self.cli_command[0]} exited {proc.returncode}: {proc.stderr.strip()[:300]}")
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
