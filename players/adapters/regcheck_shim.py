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
    load_dimension_kinds,
    regcheck_items_to_output,
)


class RegcheckShimAdapter(PlayerAdapter):
    """Drives regcheck's `backend.cli general` through the local codex shim."""

    def __init__(self, *args, regcheck_client: str | None = None,
                 regcheck_model: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
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

    # -- lifecycle ----------------------------------------------------------

    def prepare(self) -> None:
        regcheck_dir = os.environ.get("REGCHECK_DIR", "").strip()
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
        _key_env = {"groq": "GROQ_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}.get(self._client)
        if _key_env and not os.environ.get(_key_env, "").strip():
            raise RuntimeError(
                f"REGCHECK_CLIENT={self._client} but {_key_env} is not set in the "
                "environment. Export it for the run only; never commit it."
            )

        # The shim is started in BOTH modes. In "openai" mode it serves chat +
        # embeddings (codex-backed, no key). In "groq" mode the chat goes to Groq
        # natively (fast), but regcheck still builds an OpenAI client for
        # EMBEDDINGS (Groq has no embeddings API) — those calls are routed to the
        # shim's local /v1/embeddings, so no real OpenAI key is ever needed.
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
            return regcheck_items_to_output(items, self._dimension_kinds)


register_adapter_class("RegcheckShimAdapter", RegcheckShimAdapter)
