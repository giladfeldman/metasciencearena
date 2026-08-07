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
        self._proc = subprocess.Popen(self.start_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    "cli_command", "prompt_template_path",
    "catalog_path", "api_key_env", "api_url_env",
    # pdf-text-fidelity-v1 adapters:
    "docpluck_level", "pdftotext_binary", "output_message_flag",
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
    # code-translation-r-v1 vendored-tool adapter: the SPSS/Stata->R converters
    # are R packages, so their outputs are captured once and committed as
    # version-pinned fixtures rather than invoked live (user decision 2026-08-03).
    "fixture_dir", "fixture_tool_version",
}


class SubprocessCliAdapter(PlayerAdapter):
    """Generic CLI adapter: spawn a CLI binary, feed it a templated prompt + envelope on stdin, parse JSON-only stdout."""

    cli_command: list[str]
    prompt_template_path: Path

    def __init__(self, *args, cli_command: list[str], prompt_template_path: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_command = list(cli_command)
        self.prompt_template_path = Path(prompt_template_path)
        self._template: str | None = None
        self._resolved_command: list[str] | None = None

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

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._template is None or self._resolved_command is None:
            self.prepare()
        input_payload = envelope["input"]
        if isinstance(input_payload, dict) and "text" in input_payload:
            input_str = input_payload["text"]
        else:
            input_str = json.dumps(input_payload, ensure_ascii=False)
        prompt = self._template.replace("{{INPUT_TEXT}}", input_str)
        proc = subprocess.run(
            self._resolved_command, input=prompt, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout_s, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{self.cli_command[0]} exited {proc.returncode}: {proc.stderr.strip()[:300]}")
        # Permissively recover the JSON value: CLIs (notably Opus) may wrap it in
        # ```json fences, prepend prose, or append a trailing sentence.
        return _extract_json_value(proc.stdout)


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
