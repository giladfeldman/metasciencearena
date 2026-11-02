"""RegcheckShimAdapter — runs JamieCummins/regcheck on prereg-deviation-v1 with
NO OpenAI API key, by routing regcheck's OpenAI SDK calls through a local
codex-backed shim (players/regcheck_shim/openai_shim.py).

Why an env-configured adapter, not a vendored install
-----------------------------------------------------
regcheck is a heavy FastAPI app with its own venv (numpy, PyMuPDF, openai SDK,
fastapi, …). Per the task's hard rules we do NOT commit that clone or venv. The
adapter therefore locates them via environment variables and fails LOUDLY with a
runbook pointer when they are absent — so the committed code is self-describing
but carries no large/binary deps.

Required environment (see players/regcheck_shim/run_regcheck.md):
  REGCHECK_DIR     absolute path to a `git clone` of JamieCummins/regcheck
  REGCHECK_PYTHON  absolute path to a python.exe in a venv with regcheck's
                   (minimal) CLI-path deps installed

Optional:
  REGCHECK_SHIM_CODEX_MODEL   model passed to `codex exec --model` (default: codex default)
  REGCHECK_PARSER_CHOICE      pymupdf (default) | dpt2  — grobid is unavailable (no Java)

On play_task the adapter:
  1. Writes the envelope's preregistration/paper text to .txt files and the
     dimension ids to a dimensions CSV.
  2. Runs `python -m backend.cli general ... --client openai
     --parser-choice <p> --output-format json --output result.json` with
     OPENAI_BASE_URL pointed at the in-process shim and a DUMMY OPENAI_API_KEY.
  3. Converts regcheck's result.json into the arena output schema.

deterministic=False: codex is a non-deterministic LLM.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from framework.player_adapter import PlayerAdapter, register_adapter_class

_SHIM_PKG = Path(__file__).resolve().parents[1] / "regcheck_shim"
if str(_SHIM_PKG.parent.parent) not in sys.path:
    sys.path.insert(0, str(_SHIM_PKG.parent.parent))

from players.regcheck_shim import openai_shim  # noqa: E402
from players.regcheck_shim.regcheck_to_runrecords import (  # noqa: E402
    FULLPAPER_ARENA_ID,
    FULLPAPER_DIMENSIONS_YAML,
    load_dimension_kinds,
    regcheck_items_to_output,
    regcheck_items_to_threeway_output,
)

# Free OpenAI-compatible hosts RegCheck's GPUStack client can be pointed at:
# client name -> (base URL, environment variable holding the key).
GPUSTACK_HOSTS = {
    "groq-gpustack": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "nvidia-gpustack": ("https://integrate.api.nvidia.com/v1", "NVIDIA_NIM_API_KEY"),
}

# Retrieval embeddings a player may pin instead of the shim's hashing stand-in:
# route name -> (base URL, key environment variable, embeddings model).
# nemotron-3-embed-1b is the NIM embeddings model that accepts a plain OpenAI request
# (the asymmetric ones demand an input_type RegCheck does not send; measured 2026-10-30).
EMBEDDING_ROUTES = {
    "nvidia-nim": ("https://integrate.api.nvidia.com/v1", "NVIDIA_NIM_API_KEY",
                   "nvidia/nemotron-3-embed-1b"),
}


class RegcheckShimAdapter(PlayerAdapter):
    """Drives regcheck's `backend.cli general` through the local codex shim."""

    def __init__(self, *args, regcheck_client: str | None = None,
                 regcheck_model: str | None = None, regcheck_commit: str | None = None,
                 regcheck_embeddings: str | None = None,
                 cli_effort: str | None = None, cli_effort_reason: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        #: Effort for the codex calls the shim makes (client "openai"); framework.model_call
        #: refuses those calls without it.
        self.cli_effort = cli_effort
        self.cli_effort_reason = cli_effort_reason
        self._last_model_call: dict | None = None
        self._server = None
        self._thread = None
        self._base_url = None
        self._regcheck_dir: Path | None = None
        self._python: str | None = None
        self._dimension_kinds: dict[str, str] = {}
        self._client: str | None = None
        # Provider + model pinned per registry player (regcheck_client/_model),
        # so regcheck-groq / regcheck-deepseek / regcheck-openai are distinct
        # players from one adapter class. Fall back to the env defaults when the
        # registry entry omits them (back-compat with the single `regcheck` player).
        self._registry_client = (regcheck_client or "").strip().lower() or None
        self._registry_model = (regcheck_model or "").strip() or None
        #: A player may pin the RegCheck commit it measures. Its clone is then
        #: $REGCHECK_HOME/regcheck-<commit> (default ~/regcheck_run), and prepare()
        #: refuses a clone whose HEAD is another commit, so the label names what ran.
        self._registry_commit = (regcheck_commit or "").strip() or None
        #: None = the local shim's embeddings (hashing fallback without sentence-transformers).
        self._embeddings = (regcheck_embeddings or "").strip().lower() or None
        if self._embeddings and self._embeddings not in EMBEDDING_ROUTES:
            raise ValueError(f"unknown regcheck_embeddings {self._embeddings!r}; "
                             f"known: {sorted(EMBEDDING_ROUTES)}")

    # -- lifecycle ----------------------------------------------------------

    def prepare(self) -> None:
        regcheck_dir = os.environ.get("REGCHECK_DIR", "").strip()
        if self._registry_commit:
            home = os.environ.get("REGCHECK_HOME", "").strip() or str(Path.home() / "regcheck_run")
            regcheck_dir = str(Path(home) / f"regcheck-{self._registry_commit}")
        python_exe = os.environ.get("REGCHECK_PYTHON", "").strip()
        if not regcheck_dir or not python_exe:
            raise RuntimeError(
                "regcheck not configured. Set REGCHECK_DIR (a clone of "
                "JamieCummins/regcheck) and REGCHECK_PYTHON (a venv python with its "
                "CLI deps). See players/regcheck_shim/run_regcheck.md."
            )
        self._regcheck_dir = Path(regcheck_dir)
        if not (self._regcheck_dir / "backend" / "cli.py").exists():
            raise RuntimeError(f"REGCHECK_DIR has no backend/cli.py: {self._regcheck_dir}")
        if self._registry_commit:
            head = subprocess.run(["git", "-C", str(self._regcheck_dir), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=False).stdout.strip()
            if not head.startswith(self._registry_commit):
                raise RuntimeError(
                    f"{self._regcheck_dir} is at {head[:7] or 'an unknown commit'}, but this "
                    f"player pins regcheck {self._registry_commit}")
        self._python = python_exe
        self._dimension_kinds = load_dimension_kinds()

        # Client mode:
        #   "openai"   → route the LLM chat through the local codex shim (no key);
        #   "groq"     → regcheck's native Groq path, reading GROQ_API_KEY;
        #   "deepseek" → regcheck's native DeepSeek path, reading DEEPSEEK_API_KEY
        #                (OpenAI-compatible endpoint at https://api.deepseek.com).
        # In every mode the EMBEDDINGS still go to the local shim (neither Groq nor
        # DeepSeek expose an embeddings API), so no real OpenAI key is ever needed.
        # The provider key is read from the live environment only — never placed on
        # argv and never written into the run record's sanitized provenance command.
        self._client = (
            self._registry_client
            or (os.environ.get("REGCHECK_CLIENT", "openai").strip().lower() or "openai")
        )
        #   "groq-gpustack" → regcheck's GPUStack client (its gpt-oss-120b route, built
        #                for the Uni Bern endpoint) pointed at Groq's OpenAI-compatible
        #                API, which serves the same open-weight model on a free tier.
        #                regcheck >= 2026-10 has no native Groq client.
        #   "nvidia-gpustack" → the same GPUStack client pointed at NVIDIA NIM's free
        #                OpenAI-compatible endpoint. Groq's free tier caps every chat
        #                model at <= 8000 tokens/minute, below one full-paper request.
        _key_env = {"groq": "GROQ_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
                    **{c: k for c, (_u, k) in GPUSTACK_HOSTS.items()}}.get(self._client)
        if _key_env and not os.environ.get(_key_env, "").strip():
            raise RuntimeError(
                f"REGCHECK_CLIENT={self._client} but {_key_env} is not set in the "
                "environment. Export it for the run only; never commit it."
            )
        if self._embeddings:
            _emb_key = EMBEDDING_ROUTES[self._embeddings][1]
            if not os.environ.get(_emb_key, "").strip():
                raise RuntimeError(
                    f"regcheck_embeddings={self._embeddings} but {_emb_key} is not set in "
                    "the environment. Export it for the run only; never commit it.")

        # The shim is started in BOTH modes. In "openai" mode it serves chat +
        # embeddings (codex-backed, no key). In "groq" mode the chat goes to Groq
        # natively (fast), but regcheck still builds an OpenAI client for
        # EMBEDDINGS (Groq has no embeddings API) — those calls are routed to the
        # shim's local /v1/embeddings, so no real OpenAI key is ever needed.
        if self._client == "openai" and self._registry_model:
            # The shim answers through `codex exec --model <this>` (ChatGPT subscription).
            openai_shim.CODEX_MODEL = self._registry_model
        if self._client == "openai":
            # Set fresh for every player: never inherit the previous player's level from the module.
            openai_shim.CODEX_EFFORT = (self.cli_effort
                                        or os.environ.get("REGCHECK_SHIM_CODEX_EFFORT", "").strip() or None)
            if not openai_shim.CODEX_EFFORT:
                raise RuntimeError("regcheck via the codex shim needs cli_effort in the registry "
                                   "(or REGCHECK_SHIM_CODEX_EFFORT): an unstated effort is not reproducible")
        self._server = openai_shim.make_server(port=0)
        port = self._server.server_address[1]
        self._base_url = f"http://127.0.0.1:{port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def cleanup(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            finally:
                self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # -- play ---------------------------------------------------------------

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._client is None:
            self.prepare()
        inp = envelope["input"]
        dims = inp["dimensions"]
        with openai_shim._CALL_LOCK:
            openai_shim.CALL_BLOCKS.clear()
        self._last_model_call = None
        try:
            return self._play(envelope, inp, dims, timeout_s)
        finally:
            with openai_shim._CALL_LOCK:
                blocks = list(openai_shim.CALL_BLOCKS)
            if blocks:
                from framework import model_call
                self._last_model_call = model_call.aggregate_blocks(blocks)

    def _play(self, envelope: dict, inp: dict, dims, timeout_s: int) -> dict:
        parser_choice = os.environ.get("REGCHECK_PARSER_CHOICE", "pymupdf").strip() or "pymupdf"

        with tempfile.TemporaryDirectory(prefix="regcheck_task_") as td:
            tdp = Path(td)
            prereg_path = tdp / "prereg.txt"
            paper_path = tdp / "paper.txt"
            dims_csv = tdp / "dimensions.csv"
            result_json = tdp / "result.json"
            prereg_path.write_text(inp["preregistration"], encoding="utf-8")
            paper_path.write_text(inp["paper"], encoding="utf-8")
            dims_csv.write_text(
                "dimension\n" + "\n".join(dims) + "\n", encoding="utf-8"
            )

            env = dict(os.environ)
            # Cap embedding work; our texts are tiny so this is just a guard.
            env.setdefault("MAX_EMBEDDING_SEGMENTS", "200")
            # Embeddings always go to the local shim (no real OpenAI key). In groq
            # mode the LLM chat additionally goes to Groq natively.
            env["OPENAI_BASE_URL"] = self._base_url
            env["OPENAI_API_KEY"] = "dummy-local-shim"  # never a real key
            if self._client == "groq":
                # GROQ_API_KEY is already in os.environ (validated in prepare); it
                # stays in the env only — never on argv, never in the run record's
                # sanitized provenance command.
                client_choice = "groq"
                groq_model = self._registry_model or os.environ.get("REGCHECK_GROQ_MODEL", "").strip()
                if groq_model:
                    env["GROQ_MODEL"] = groq_model
            elif self._client in GPUSTACK_HOSTS:
                client_choice = "gpustack"
                base_url, key_env = GPUSTACK_HOSTS[self._client]
                env["GPUSTACK_BASE_URL"] = base_url
                env["GPUSTACK_API_KEY"] = os.environ[key_env]  # env only, never argv
                if self._registry_model:
                    env["GPUSTACK_MODEL"] = self._registry_model
                # Neither host serves embeddings: retrieval embeddings go to the local shim.
                env["EMBEDDINGS_BASE_URL"] = self._base_url
                env["EMBEDDINGS_API_KEY"] = "dummy-local-shim"
            elif self._client == "deepseek":
                # DEEPSEEK_API_KEY is already in os.environ (validated in prepare).
                # regcheck builds its own OpenAI(base_url="https://api.deepseek.com")
                # client for chat; embeddings still fall back to the local shim.
                client_choice = "deepseek"
                deepseek_model = self._registry_model or os.environ.get("REGCHECK_DEEPSEEK_MODEL", "").strip()
                if deepseek_model:
                    env["DEEPSEEK_MODEL"] = deepseek_model
            else:
                client_choice = "openai"
                env.setdefault("GROQ_API_KEY", "dummy-local-shim")

            cmd = [
                self._python, "-m", "backend.cli", "general",
                "--preregistration", str(prereg_path),
                "--paper", str(paper_path),
                "--dimensions-csv", str(dims_csv),
                "--client", client_choice,
                "--parser-choice", parser_choice,
                "--output-format", "json",
                "--output", str(result_json),
            ]
            if self._embeddings:
                emb_url, emb_key, emb_model = EMBEDDING_ROUTES[self._embeddings]
                env["EMBEDDINGS_BASE_URL"] = emb_url
                env["EMBEDDINGS_API_KEY"] = os.environ[emb_key]  # env only, never argv
                cmd += ["--embedding-model", emb_model]
            proc = subprocess.run(
                cmd,
                cwd=str(self._regcheck_dir),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
            if not result_json.exists():
                raise RuntimeError(
                    f"regcheck produced no result.json (exit {proc.returncode}): "
                    f"{(proc.stderr or '').strip()[-500:]}"
                )
            data = json.loads(result_json.read_text(encoding="utf-8"))
            items = data.get("items", []) if isinstance(data, dict) else []
            if envelope.get("arena_id") == FULLPAPER_ARENA_ID:
                # The full-paper track scores RegCheck's own yes/no/missing three ways.
                return regcheck_items_to_threeway_output(
                    items, load_dimension_kinds(FULLPAPER_DIMENSIONS_YAML))
            return regcheck_items_to_output(items, self._dimension_kinds)


register_adapter_class("RegcheckShimAdapter", RegcheckShimAdapter)
