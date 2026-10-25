"""Claude through the Anthropic Messages API: the bare model, no CLI harness.

WHY. Every other Claude player here runs through `claude --print`, which brings
the Claude Code harness with it (system prompt, tool environment, scratchpad --
debts d-1cb7bf / d-f7a10a). A benchmark score should be about the MODEL. This
adapter sends one Messages request with no tools and no system prompt, so the
only thing the model sees is the arena prompt and the task.

POLICY. The project's standing rule is "Claude only via the Max subscription,
never the API". This adapter is the single approved exception (owner,
2026-10-23): API use ONLY for ScienceArena players, funded ONLY by the
Max-included monthly API credit. It uses `requests`, not the `anthropic` SDK, so
the import/dependency commit gate still holds everywhere else. The key is read from SCIENCEARENA_ANTHROPIC_KEY -- never
ANTHROPIC_API_KEY, which the Claude Code CLI would pick up and bill against, and
which hermetic.py strips for that reason. The SCIENCEARENA_ prefix means the
hermetic environment strips this key from every CLI player's subprocess too.

TRANSPARENCY. Effort is REQUIRED on every player (`anthropic_effort`). Models do
not share a default (platform docs 2026-10-23: Opus 5.5 and Haiku 5.5 default to
medium, Sonnet 5.5 to high), so "default settings" is not one condition across
models; an unstated effort is refused rather than inherited. The effort sent, the
thinking setting, the model the API says it served, its request id, stop reason
and token usage all go into the run record.

MODES (SCIENCEARENA_ANTHROPIC_MODE, required -- unset refuses, so no command
spends by accident):
  live    one standard-price request per task. Every request is first counted
          with the free count_tokens endpoint and reserved at its worst case in
          the local spend ledger (_anthropic_budget.py); a request that would
          breach the budget is never sent.
  plan    no paid call: counts tokens, writes the request into
          $SCIENCEARENA_ANTHROPIC_BATCH_DIR/<player_id>/requests.jsonl, then
          raises BatchPlanned. Driven by scripts/anthropic_batch.py, which owns
          the throwaway run output this produces.
  replay  no network at all: answers each task from the downloaded batch results
          ($SCIENCEARENA_ANTHROPIC_BATCH_DIR/<player_id>/results.jsonl), so batch
          answers are scored and recorded by the normal runner. Cost was settled
          at collection time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import requests

from framework.player_adapter import (
    PlayerAdapter,
    _extract_json_value,
    register_adapter_class,
)
from players.adapters import _anthropic_budget as budget

logger = logging.getLogger(__name__)

API_BASE = "https://api.anthropic.com"  # hard-rule-ok: approved ScienceArena API exception, 2026-10-23
API_VERSION = "2023-06-01"
KEY_ENV = "SCIENCEARENA_ANTHROPIC_KEY"
WORKSPACE_ENV = "SCIENCEARENA_ANTHROPIC_WORKSPACE_ID"
MODE_ENV = "SCIENCEARENA_ANTHROPIC_MODE"
BATCH_DIR_ENV = "SCIENCEARENA_ANTHROPIC_BATCH_DIR"
MODES = ("live", "plan", "replay")

EFFORTS = ("low", "medium", "high", "xhigh", "max")
#: Each model's API default effort, from the platform effort docs (2026-10-23).
#: Recorded next to the effort actually sent so a reader can see whether a row is
#: the model's default or a deliberate setting. Never used to fill a missing value.
MODEL_DEFAULT_EFFORT = {
    "claude-opus-5-5": "medium",
    "claude-sonnet-5-5": "high",
    "claude-haiku-5-5": "medium",
}

#: Output ceiling when a player declares none. Includes thinking tokens. Sized
#: from the measured Opus 5 CLI records (2026-10-23 census: max 14,430 output
#: tokens on stats-extraction). A cut-off answer is visible as
#: response_meta.finish_reason == "max_tokens", never silently scored.
DEFAULT_MAX_TOKENS = 16000

_MAX_ATTEMPTS = 3          # 1 + 2 retries, per the plan's retry cap
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
_PDF_PLACEHOLDER = "the PDF document attached to this message"


class BatchPlanned(RuntimeError):
    """Plan mode: the request was queued for a batch; nothing was sent."""


def _user_env(name: str) -> str:
    """A variable from the live environment first, then (Windows) the user's stored
    environment in the registry. A variable saved with scripts/set_anthropic_key.ps1
    only reaches processes started AFTER it was saved; the registry read lets an
    already-running session use it without a restart. Read-only; never logged."""
    value = os.environ.get(name, "").strip()
    if value or os.name != "nt":
        return value
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
            stored, _ = winreg.QueryValueEx(h, name)
        return str(stored).strip()
    except OSError:
        return ""


def read_key() -> str:
    """The API key (see _user_env)."""
    return _user_env(KEY_ENV)


def _headers(key: str) -> dict:
    """Request headers. A key created at organization level (not scoped to a
    workspace) is refused unless the request names a workspace: measured
    2026-10-23, HTTP 400 "This API key is not scoped to a workspace, so this
    request must include the anthropic-workspace-id header". The id is not a
    secret; it comes from SCIENCEARENA_ANTHROPIC_WORKSPACE_ID when set."""
    h = {"x-api-key": key, "anthropic-version": API_VERSION,
         "content-type": "application/json"}
    workspace = _user_env(WORKSPACE_ENV)
    if workspace:
        h["anthropic-workspace-id"] = workspace
    return h


def _input_digest(envelope: dict) -> str:
    raw = json.dumps(envelope.get("input"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def custom_id(player_id: str, envelope: dict, trial: int) -> str:
    """Batch-safe id (^[a-zA-Z0-9_-]{1,64}$) tying one answer to one task + trial."""
    h = hashlib.sha256(f"{player_id}|{envelope['task_id']}|{_input_digest(envelope)}"
                       .encode("utf-8")).hexdigest()[:40]
    return f"{h}-t{trial}"


def build_messages(template: str, envelope: dict, document_to_text: str | None = None,
                   timeout_s: int = 120) -> list[dict]:
    """The user turn: arena prompt plus the task, PDFs as native document blocks.

    With document_to_text="pdftotext" a PDF is instead extracted on this machine by
    the same helper the other players of the arena use (openai_compatible._pdf_to_text)
    and sent as text in {{INPUT_TEXT}} -- for arenas where every AI player is given
    pdftotext input (owner decision 2026-09-29, see llm_pdf.py).

    PDF arenas' prompts were written for CLI players and point at a file path
    ({{PDF_PATH}}); here the model receives the PDF itself, so the placeholder is
    replaced by a phrase naming the attachment. Text arenas use {{INPUT_TEXT}}.
    """
    payload = envelope["input"]
    if document_to_text == "pdftotext" and isinstance(payload, dict) and payload.get("document_bytes_b64"):
        from players.adapters.openai_compatible import _pdf_to_text
        text = _pdf_to_text(payload["document_bytes_b64"], timeout_s)
        return [{"role": "user", "content": template.replace("{{INPUT_TEXT}}", text)}]
    if isinstance(payload, dict) and payload.get("document_bytes_b64"):
        b64 = payload["document_bytes_b64"]
        # Same fallback as the CLI PDF players (llm_pdf.py), so both see one prompt.
        n_pages = payload.get("n_pages", 1)
        text = (template.replace("{{PDF_PATH}}", _PDF_PLACEHOLDER)
                .replace("{{N_PAGES}}", str(n_pages))
                .replace("{{INPUT_TEXT}}", f"[{_PDF_PLACEHOLDER}]"))
        content = [
            {"type": "document",
             "source": {"type": "base64", "media_type": "application/pdf", "data": b64}},
            {"type": "text", "text": text},
        ]
        return [{"role": "user", "content": content}]
    if isinstance(payload, dict) and "text" in payload:
        input_str = payload["text"]
    else:
        input_str = json.dumps(payload, ensure_ascii=False)
    return [{"role": "user", "content": template.replace("{{INPUT_TEXT}}", input_str)}]


def text_of(message: dict) -> str:
    """Concatenate the text blocks of a Messages response (thinking is not the answer)."""
    parts = [b.get("text", "") for b in message.get("content") or []
             if isinstance(b, dict) and b.get("type") == "text"]
    return "".join(parts)


def usage_record(usage: dict | None) -> dict | None:
    """Run-record usage in the framework's canonical shape (same as the CLI envelope
    parser: prompt_tokens = fresh + cache-read + cache-write input)."""
    if not isinstance(usage, dict):
        return None
    def n(k):
        v = usage.get(k)
        return v if isinstance(v, int) and not isinstance(v, bool) else None
    inp, out = n("input_tokens"), n("output_tokens")
    cr, cw = n("cache_read_input_tokens"), n("cache_creation_input_tokens")
    out_rec: dict = {}
    parts = [v for v in (inp, cr, cw) if v is not None]
    if parts:
        out_rec["prompt_tokens"] = sum(parts)
    if out is not None:
        out_rec["completion_tokens"] = out
    if "prompt_tokens" in out_rec and "completion_tokens" in out_rec:
        out_rec["total_tokens"] = out_rec["prompt_tokens"] + out_rec["completion_tokens"]
    if cr is not None:
        out_rec["cache_read_tokens"] = cr
    if cw is not None:
        out_rec["cache_creation_tokens"] = cw
    return out_rec or None


class AnthropicApiAdapter(PlayerAdapter):
    """One task = one Messages request to a pinned model at a stated effort."""

    def __init__(self, *args, anthropic_model: str, prompt_template_path: str | Path,
                 anthropic_effort: str | None = None,
                 anthropic_max_tokens: int | None = None,
                 anthropic_thinking: dict | None = None,
                 anthropic_output_coercer: str | None = None,
                 anthropic_document_to_text: str | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.anthropic_model = (anthropic_model or "").strip()
        self.prompt_template_path = Path(prompt_template_path)
        self.anthropic_effort = anthropic_effort
        self.anthropic_max_tokens = anthropic_max_tokens or DEFAULT_MAX_TOKENS
        #: Optional `thinking` object sent verbatim (e.g. {"type": "disabled"} on
        #: Haiku 5.5 at <= high). None = omit, i.e. the model's adaptive default.
        self.anthropic_thinking = dict(anthropic_thinking) if anthropic_thinking else None
        #: "module:function" that turns the raw answer text into the arena's output
        #: shape. The PDF arenas need the SAME coercer their CLI players use
        #: (e.g. players.adapters.llm_pdf_sections:_coerce_sections), so both
        #: harnesses are scored on identical post-processing. None = generic JSON.
        self.anthropic_output_coercer = anthropic_output_coercer
        if anthropic_document_to_text not in (None, "pdftotext"):
            raise ValueError(f"unknown anthropic_document_to_text {anthropic_document_to_text!r}")
        self.anthropic_document_to_text = anthropic_document_to_text
        self._coerce = None
        self._template: str | None = None
        self._key: str | None = None
        self._mode: str | None = None
        self._ledger: budget.Ledger | None = None
        self._results: dict[str, dict] | None = None
        self._trials: Counter = Counter()
        self._last_usage: dict | None = None
        self._last_meta: dict | None = None
        self._sent: dict[str, dict] = {}
        self._last_request_body: dict | None = None
        self._last_raw_response: dict | None = None
        self._last_sent_max_tokens: int | None = None

    @property
    def latency_measured(self) -> bool:
        """False in replay/plan mode: the runner's clock would time a file lookup."""
        return self._mode == "live"

    # -- setup --------------------------------------------------------------
    def prepare(self) -> None:
        if not self.anthropic_model:
            raise RuntimeError(f"{self.player_id}: anthropic_model must be set")
        if self.anthropic_effort not in EFFORTS:
            raise RuntimeError(
                f"{self.player_id}: anthropic_effort={self.anthropic_effort!r}; it must be "
                f"one of {EFFORTS}. Effort is required on every API player because "
                "models do not share a default.")
        if not isinstance(self.anthropic_max_tokens, int) or isinstance(self.anthropic_max_tokens, bool) \
                or self.anthropic_max_tokens <= 0:
            raise RuntimeError(f"{self.player_id}: anthropic_max_tokens must be a positive int")
        mode = os.environ.get(MODE_ENV, "").strip()
        if mode not in MODES:
            raise RuntimeError(
                f"{self.player_id}: set {MODE_ENV} to one of {MODES}. It has no default so "
                "that no command spends API credit by accident.")
        self._mode = mode
        if self.anthropic_output_coercer:
            import importlib
            mod_name, _, fn = self.anthropic_output_coercer.partition(":")
            self._coerce = getattr(importlib.import_module(mod_name), fn)
        self._template = self.prompt_template_path.read_text(encoding="utf-8")
        if mode in ("live", "plan"):
            key = read_key()
            if not key:
                raise RuntimeError(f"{self.player_id}: {KEY_ENV} is not set. Set it with "
                                   "scripts/set_anthropic_key.ps1; never write the key to a file.")
            self._key = key
            budget.model_prices(self.anthropic_model)  # unpriced model -> refuse now
        if mode == "live":
            self._ledger = budget.Ledger.from_env()
        if mode == "replay":
            self._results = self._load_results()

    def _player_batch_dir(self) -> Path:
        root = os.environ.get(BATCH_DIR_ENV, "").strip()
        if not root:
            raise RuntimeError(f"{self.player_id}: {BATCH_DIR_ENV} must be set in "
                               f"{self._mode} mode")
        return Path(root) / self.player_id

    def _load_results(self) -> dict[str, dict]:
        path = self._player_batch_dir() / "results.jsonl"
        if not path.exists():
            raise RuntimeError(f"{self.player_id}: no batch results at {path}; run "
                               "scripts/anthropic_batch.py collect first")
        out: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["custom_id"]] = row
        # The requests as planned and sent, so each replayed record states the
        # request it answers (see _meta), not the registry as it is today.
        req_path = path.parent / "requests.jsonl"
        if not req_path.exists():
            raise RuntimeError(f"{self.player_id}: no {req_path}; cannot state what was sent")
        self._sent = {}
        for line in req_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                req = json.loads(line)
                self._sent[req["custom_id"]] = req["params"]
        return out

    # -- request ------------------------------------------------------------
    def build_body(self, envelope: dict, timeout_s: int = 120) -> dict:
        assert self._template is not None
        body = {
            "model": self.anthropic_model,
            "max_tokens": self.anthropic_max_tokens,
            "messages": build_messages(self._template, envelope,
                                       self.anthropic_document_to_text, timeout_s),
            "output_config": {"effort": self.anthropic_effort},
        }
        if self.anthropic_thinking:
            body["thinking"] = self.anthropic_thinking
        return body

    def _count_tokens(self, body: dict, timeout_s: int) -> int:
        count_body = {k: v for k, v in body.items() if k in ("model", "messages", "thinking", "system")}
        r = requests.post(f"{API_BASE}/v1/messages/count_tokens", headers=_headers(self._key),
                          json=count_body, timeout=min(timeout_s, 120))
        if r.status_code != 200:
            raise RuntimeError(f"{self.player_id}: count_tokens HTTP {r.status_code}: {r.text[:300]}")
        n = r.json().get("input_tokens")
        if not isinstance(n, int):
            raise RuntimeError(f"{self.player_id}: count_tokens returned no input_tokens")
        return n

    def _meta(self, message: dict, *, attempts: int, tier: str, sent: dict) -> dict:
        # The runner's call_config and raw sidecar (schema v12) read these. The
        # messages are hashed, not kept: the prompt is already pinned by its
        # template hash and the task's input hash.
        self._last_request_body = {k: v for k, v in sent.items() if k != "messages"}
        self._last_request_body["api"] = "anthropic-messages"
        if "messages" in sent:
            self._last_request_body["messages_sha256"] = hashlib.sha256(json.dumps(
                sent["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
        self._last_raw_response = message
        # Effort and thinking are request parameters WE sent, so they go in the
        # run-record schema's existing `sampling` object (response_meta allows no
        # other new keys). The harness needs no field: the adapter class, the
        # `anthropic-api-` player id and the instrument slug all name it.
        # Read from the request body actually sent (`sent`), never from today's
        # registry: a replay of an older batch must state the old request
        # (2026-10-24: a raised max_tokens was recorded on answers sent at 32000).
        self._last_sent_max_tokens = sent.get("max_tokens")
        sampling = {"effort": (sent.get("output_config") or {}).get("effort"),
                    "model_default_effort": MODEL_DEFAULT_EFFORT.get(sent.get("model")),
                    "thinking": sent.get("thinking") or "model default",
                    "max_tokens": sent.get("max_tokens")}
        meta: dict = {"attempts": attempts, "service_tier": tier, "sampling": sampling}
        if isinstance(message.get("stop_reason"), str):
            meta["finish_reason"] = message["stop_reason"]
        if isinstance(message.get("model"), str):
            meta["served_model"] = message["model"]
        if isinstance(message.get("id"), str):
            meta["provider_request_id"] = message["id"]
        return meta

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._mode is None:
            self.prepare()
        # Per-task state cleared FIRST (see openai_compatible.py: a failed task must
        # not inherit the previous task's tokens).
        self._last_usage = None
        self._last_meta = None
        self._last_request_body = None
        self._last_raw_response = None
        self._trials[envelope["task_id"]] += 1
        cid = custom_id(self.player_id, envelope, self._trials[envelope["task_id"]])

        if self._mode == "replay":
            return self._replay(cid)
        body = self.build_body(envelope, timeout_s)
        if self._mode == "plan":
            n_in = self._count_tokens(body, timeout_s)
            d = self._player_batch_dir()
            d.mkdir(parents=True, exist_ok=True)
            with (d / "requests.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"custom_id": cid, "task_id": envelope["task_id"],
                                     "input_tokens": n_in, "params": body}) + "\n")
            raise BatchPlanned(f"{self.player_id}: queued {cid} for a batch (nothing sent)")
        return self._live(cid, body, timeout_s)

    def _live(self, cid: str, body: dict, timeout_s: int) -> dict:
        assert self._ledger is not None
        n_in = self._count_tokens(body, timeout_s)
        worst = budget.worst_case_cost(self.anthropic_model, n_in, self.anthropic_max_tokens,
                                       batch=False)
        rid = f"live-{self.player_id}-{cid}-{int(time.time() * 1000)}"
        self._ledger.reserve(rid, worst, detail=f"{self.anthropic_model} in={n_in} "
                                                f"max_out={self.anthropic_max_tokens}")
        response = None
        attempt = 0
        try:
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                response = requests.post(f"{API_BASE}/v1/messages", headers=_headers(self._key),
                                         json=body, timeout=timeout_s)
                if response.status_code not in _RETRY_STATUSES or attempt == _MAX_ATTEMPTS:
                    break
                logger.warning("%s: HTTP %s (attempt %d/%d), retrying",
                               self.player_id, response.status_code, attempt, _MAX_ATTEMPTS)
                time.sleep(2 ** attempt)
        except requests.RequestException:
            # Transport failure: Anthropic may still have generated (and billed)
            # the answer. The worst-case reservation stays open on purpose.
            raise
        if response.status_code != 200:
            # An HTTP error response means no message was created: not billed.
            self._ledger.release(rid, detail=f"HTTP {response.status_code}")
            raise RuntimeError(f"{self.player_id}: Messages API HTTP {response.status_code}: "
                               f"{response.text[:500]}")
        message = response.json()
        usage = message.get("usage") or {}
        self._ledger.settle(rid, budget.usage_cost(self.anthropic_model, usage, batch=False),
                            detail=json.dumps(usage)[:300])
        self._last_usage = usage_record(usage)
        self._last_meta = self._meta(message, attempts=attempt, tier="standard", sent=body)
        return self._parse(text_of(message))

    def _replay(self, cid: str) -> dict:
        assert self._results is not None
        row = self._results.get(cid)
        if row is None:
            raise RuntimeError(f"{self.player_id}: no batch result for {cid}; the batch did "
                               "not include this task/trial (re-plan with the same flags)")
        result = row.get("result") or {}
        if result.get("type") != "succeeded":
            raise RuntimeError(f"{self.player_id}: batch result for {cid} is "
                               f"{result.get('type')!r}: {json.dumps(result)[:300]}")
        sent = self._sent.get(cid)
        if sent is None:
            raise RuntimeError(f"{self.player_id}: no planned request for {cid} in "
                               "requests.jsonl; cannot state what was sent")
        sent_effort = (sent.get("output_config") or {}).get("effort")
        if sent.get("model") != self.anthropic_model or sent_effort != self.anthropic_effort:
            # The registry entry no longer describes the request this answer came
            # from: replaying would publish it under the wrong instrument.
            raise RuntimeError(f"{self.player_id}: batch request for {cid} was "
                               f"{sent.get('model')} at effort {sent_effort}; the "
                               "registry now says otherwise. Re-run, do not replay.")
        message = result.get("message") or {}
        self._last_usage = usage_record(message.get("usage"))
        self._last_meta = self._meta(message, attempts=1, tier="batch", sent=sent)
        return self._parse(text_of(message))

    def _parse(self, text: str) -> dict:
        return self._coerce(text) if self._coerce else _extract_json_value(text)

    # -- telemetry ------------------------------------------------------------
    def last_usage(self) -> dict | None:
        return self._last_usage

    def last_response_meta(self) -> dict | None:
        return self._last_meta

    def last_request_body(self) -> dict | None:
        return self._last_request_body

    def last_raw_response(self) -> object | None:
        return self._last_raw_response

    def last_request_max_tokens(self) -> int | None:
        # What the last answer was requested with (replay: the saved request).
        return self._last_sent_max_tokens if self._mode == "replay" else self.anthropic_max_tokens

    def resolved_tool_version(self) -> str | None:
        # Vendor-prefixed so the pricing table prices it as metered (a bare
        # `claude-` id is classified as a Max-subscription player). Equal to the
        # registry's player_version, so the version-drift check passes only when
        # the registry names the same model.
        return budget.price_key(self.anthropic_model)


register_adapter_class("AnthropicApiAdapter", AnthropicApiAdapter)
