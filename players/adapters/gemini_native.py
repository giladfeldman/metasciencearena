"""Google Gemini through its NATIVE generateContent API.

Why not the OpenAI-compatible Gemini endpoint the older Gemini players use?
Two measured differences (2026-10-14, free key):

* `seed` is refused there ("Invalid JSON payload received. Unknown name
  \"seed\"") and accepted natively, so a seeded call is only possible here.
* The native response carries `modelVersion`, the model that actually served
  the call. It is not always the one asked for: a request for
  `gemini-3.5-flash` came back `modelVersion: gemini-3.6-flash`, and the
  floating alias `gemini-flash-latest` resolved to `gemini-3.8-flash`. A
  published score has to name the served model, so the alias is recorded as
  requested and `modelVersion` as served.

The free tier is small: `gemini-3.8-flash` allowed 20 requests per day per
project (quotaValue in the HTTP 429 body, 2026-10-14). A 429 whose body names a
per-DAY quota is not retried: waiting seconds cannot help, and retrying only
burns the next day's budget faster.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import requests

from framework.player_adapter import PlayerAdapter, _extract_json_value, register_adapter_class

logger = logging.getLogger(__name__)

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 6
_DEFAULT_MAX_BACKOFF_S = 180.0
_DEFAULT_MAX_TOKENS = 8192
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _is_daily_quota(body_text: str) -> bool:
    return "PerDay" in body_text


class GeminiGenerateContentAdapter(PlayerAdapter):
    """Play a task through Gemini's native API with full call settings recorded."""

    def __init__(
        self,
        *args,
        gemini_model: str,
        prompt_template_path: str | Path,
        gemini_api_key_env: str = "GEMINI_API_KEY",
        gemini_temperature: int | float = 0,
        gemini_seed: int | None = None,
        gemini_thinking_level: str | None = None,
        gemini_max_output_tokens: int = _DEFAULT_MAX_TOKENS,
        gemini_response_json: bool = True,
        gemini_max_backoff_s: float = _DEFAULT_MAX_BACKOFF_S,
        prompt_format: str = "str.format",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if prompt_format not in ("replace", "str.format"):
            raise ValueError(f"unknown prompt_format {prompt_format!r}")
        self.gemini_model = gemini_model.strip()
        self.prompt_template_path = Path(prompt_template_path)
        self.gemini_api_key_env = gemini_api_key_env
        self.gemini_temperature = gemini_temperature
        self.gemini_seed = gemini_seed
        self.gemini_thinking_level = gemini_thinking_level
        self.gemini_max_output_tokens = gemini_max_output_tokens
        self.gemini_response_json = gemini_response_json
        self.gemini_max_backoff_s = float(gemini_max_backoff_s)
        self.prompt_format = prompt_format
        self._template: str | None = None
        self._key: str | None = None
        self._clear()

    def _clear(self) -> None:
        self._last_usage: dict | None = None
        self._last_meta: dict | None = None
        self._last_request_body: dict | None = None
        self._last_raw_response: object | None = None
        self._last_request_max_tokens: int | None = None

    def prepare(self) -> None:
        key = os.environ.get(self.gemini_api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"{self.player_id}: {self.gemini_api_key_env} is not set")
        self._key = key
        self._template = self.prompt_template_path.read_text(encoding="utf-8").replace(chr(13) + chr(10), chr(10))

    def _render(self, envelope: dict) -> str:
        payload = envelope["input"]
        if self.prompt_format == "str.format":
            if not isinstance(payload, dict) or not isinstance(payload.get("prompt_fields"), dict):
                raise RuntimeError(f"{self.player_id}: prompt_format str.format needs input.prompt_fields")
            return self._template.format(**payload["prompt_fields"])
        text = payload["text"] if isinstance(payload, dict) and "text" in payload else json.dumps(payload, ensure_ascii=False)
        return self._template.replace("{{INPUT_TEXT}}", text)

    def _body(self, prompt: str) -> dict:
        gen: dict = {"temperature": self.gemini_temperature, "maxOutputTokens": self.gemini_max_output_tokens}
        if self.gemini_seed is not None:
            gen["seed"] = self.gemini_seed
        if self.gemini_response_json:
            gen["responseMimeType"] = "application/json"
        if self.gemini_thinking_level:
            gen["thinkingConfig"] = {"thinkingLevel": self.gemini_thinking_level}
        return {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen}

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._template is None or self._key is None:
            self.prepare()
        self._clear()
        prompt = self._render(envelope)
        body = self._body(prompt)
        self._last_request_max_tokens = self.gemini_max_output_tokens
        self._last_request_body = {"model": self.gemini_model, "api": "gemini-generate-content",
                                   "generationConfig": body["generationConfig"],
                                   "contents_sha256": hashlib.sha256(json.dumps(
                                       body["contents"], ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()}
        url = f"{_BASE_URL}/models/{self.gemini_model}:generateContent"
        spent, attempt, response = 0.0, 0, None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            response = requests.post(url, json=body, timeout=timeout_s,
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self._key})
            status = response.status_code
            if status not in _RETRY_STATUSES or attempt == _MAX_ATTEMPTS:
                break
            if status == 429 and _is_daily_quota(response.text or ""):
                break
            delay = min(60.0, 5.0 * 2 ** (attempt - 1))
            if spent + delay > self.gemini_max_backoff_s:
                break
            logger.warning("%s: HTTP %s (attempt %d); retrying in %.0fs", self.player_id, status, attempt, delay)
            time.sleep(delay)
            spent += delay
        if response.status_code != 200:
            raise RuntimeError(f"{self.player_id}: Gemini returned HTTP {response.status_code}: "
                               f"{(response.text or '')[:400]}")
        data = response.json()
        self._last_raw_response = data
        cand = (data.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        um = data.get("usageMetadata") or {}
        usage = {k: um[s] for k, s in (("prompt_tokens", "promptTokenCount"),
                                       ("completion_tokens", "candidatesTokenCount"),
                                       ("total_tokens", "totalTokenCount"),
                                       ("reasoning_tokens", "thoughtsTokenCount")) if isinstance(um.get(s), int)}
        self._last_usage = usage or None
        meta = {"attempts": attempt}
        if isinstance(cand.get("finishReason"), str):
            meta["finish_reason"] = cand["finishReason"]
        if isinstance(data.get("modelVersion"), str):
            meta["served_model"] = data["modelVersion"]
        if isinstance(data.get("responseId"), str):
            meta["provider_request_id"] = data["responseId"]
        if isinstance(um.get("serviceTier"), str):
            meta["service_tier"] = um["serviceTier"]
        self._last_meta = meta
        return _extract_json_value(text)

    def last_usage(self) -> dict | None:
        return self._last_usage

    def last_request_max_tokens(self) -> int | None:
        return self._last_request_max_tokens

    def last_response_meta(self) -> dict | None:
        return self._last_meta

    def last_request_body(self) -> dict | None:
        return self._last_request_body

    def last_raw_response(self) -> object | None:
        return self._last_raw_response

    def resolved_tool_version(self) -> str | None:
        """The REQUESTED model id, like OpenAIChatCompletionsAdapter.

        The served model goes to ``response_meta.served_model`` (and the runner
        flags ``model_mismatch``), so intent and evidence stay two fields. A
        floating alias such as ``gemini-flash-latest`` is the declared
        instrument; what it resolved to on each call is the evidence.
        """
        return self.gemini_model


register_adapter_class("GeminiGenerateContentAdapter", GeminiGenerateContentAdapter)
