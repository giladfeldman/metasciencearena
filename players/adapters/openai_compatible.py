"""Generic OpenAI-compatible chat-completions model adapter.

This is the reusable path for named free-tier model endpoints such as NVIDIA
NIM, OpenRouter, Groq, Mistral, Hugging Face provider proxies, and local
gateways when the backend model is explicit. It deliberately refuses "auto" /
router model ids by default: a rotating backend is useful for scouting, but it
must not publish a score under a model name it did not actually run.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

#: Total attempts (1 initial + retries) for a transiently-failing request.
_MAX_ATTEMPTS = 4
#: Statuses worth retrying: a burst limit or a server-side blip. Every other 4xx
#: is the caller's fault and will fail identically forever, so retrying it only
#: burns quota that a later task needs.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
#: Ceiling on cumulative backoff per task, so one throttled provider cannot stall
#: a tournament. Deliberately small: `framework retry-failed` handles the case
#: where the quota is exhausted for minutes rather than milliseconds.
_MAX_TOTAL_BACKOFF_S = 30.0
#: Output ceiling sent on EVERY request. Omitting it does not mean "unlimited" -
#: it means each provider silently applies its own, and they differ (2026-08-13:
#: OpenRouter applied the pinned endpoint's 40960 and reserved credit for all of
#: it, so once the balance fell below the reservation every call died HTTP 402 and
#: 45 of 46 prereg-deviation tasks scored 0.0 for a non-model reason). Two
#: providers scoring one player under different ceilings are not running the same
#: benchmark, so the ceiling belongs to US and is recorded as such.
#: Sized against the MEASURED distribution over all 553 recorded completions, not
#: against a guess: p50=278, p90=5108, p99=14926, max=21531 (nemotron-lightning on
#: code-translation-r-v1, which emits whole R scripts). A first pass picked 12000
#: from "3x the longest I had looked at" and would have truncated 11 real records
#: - the eyeballed maximum was 6x too low. 32000 truncates none of them and still
#: sits under the smallest pinned endpoint ceiling (Cerebras, 40960).
#: Any task that does hit it is visible as `response_meta.finish_reason == "length"`
#: rather than silently scored as incapability. Override per player with
#: `openai_extra_body: {max_tokens: N}`.
_DEFAULT_MAX_TOKENS = 32000
#: Request fields `openai_extra_body` must never set, because they determine what
#: the resulting record is evidence ABOUT rather than how it was obtained. `model`
#: has a dedicated field guarded by `openai_available_models_env`; `messages` is
#: the arena task itself.
_RESERVED_BODY_KEYS = frozenset({"model", "messages"})


def _retry_delay(attempt: int, response: requests.Response | None) -> float:
    """Seconds to wait before `attempt` (1-based). Honours Retry-After when sent."""
    if response is not None:
        raw = response.headers.get("Retry-After") if hasattr(response, "headers") else None
        if raw:
            try:
                return max(0.0, min(float(raw), _MAX_TOTAL_BACKOFF_S))
            except (TypeError, ValueError):
                pass  # Retry-After may be an HTTP-date; fall through to backoff
    return float(2 ** (attempt - 1))  # 1s, 2s, 4s

from framework.player_adapter import (
    PlayerAdapter,
    _extract_json_value,
    register_adapter_class,
)


ROTATING_MODEL_IDS = {
    "auto",
    "autocoding",
    "auto-coding",
    "auto_fast",
    "autofast",
    "openrouter/free",
    "openrouter/auto",
}


def _is_rotating_model_id(model: str) -> bool:
    lowered = model.strip().lower()
    return lowered in ROTATING_MODEL_IDS or lowered.startswith("auto/")


def _parse_model_list(raw: str) -> list[str]:
    """Parse JSON or comma/newline-separated model allowlists."""
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            return [str(m).strip() for m in json.loads(text) if str(m).strip()]
        except (TypeError, ValueError):
            pass
    return [m.strip() for m in text.replace("\n", ",").split(",") if m.strip()]


def env_models(env_name: str | None) -> list[str]:
    """Model ids from an optional environment allowlist."""
    if not env_name:
        return []
    return _parse_model_list(os.environ.get(env_name, ""))


def _parse_sse_chat_content(text: str) -> str:
    """Extract streamed chat content from OpenAI-style SSE lines."""
    chunks: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except ValueError:
            continue
        for choice in event.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                chunks.append(delta["content"])
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                chunks.append(message["content"])
    return "".join(chunks)


def _response_meta(response: requests.Response, attempts: int) -> dict:
    """Provider metadata that says whether a score is TRUSTWORTHY.

    Measured on a real Groq response 2026-08-13, the adapter was discarding all
    of this. The one that matters most is `finish_reason`: a completion cut off
    at the token limit fails JSON parsing, scores 0.0, and is then indistinguishable
    from a model that simply cannot do the task — the benchmark would be
    publishing OUR max_tokens as THEIR incapability. `served_model` matters for the
    same reason `resolved_tool_version` does: this project has published a version
    it was not actually running, and the provider is the only authority on what it
    actually served.

    None of these fields leak document length or content, so unlike token counts
    they survive held-out redaction — which is exactly where they are most needed,
    since a held-out record has no output to inspect.
    """
    meta: dict = {"attempts": attempts}
    try:
        body = response.json()
    except Exception:
        return meta
    if not isinstance(body, dict):
        return meta
    choices = body.get("choices") or []
    if choices and isinstance(choices[0], dict):
        fr = choices[0].get("finish_reason")
        if isinstance(fr, str):
            meta["finish_reason"] = fr
    # `provider` is OpenRouter's: on a ROUTER the model id does NOT identify what
    # ran. The same `openai/gpt-oss-120b` is fanned out over CoreWeave, DeepInfra,
    # Novita, Amazon Bedrock and Google, which differ in quantization and serving
    # stack. Without this the backend behind a published score is unrecoverable.
    for src, dst in (("model", "served_model"),
                     ("provider", "served_by"),
                     ("system_fingerprint", "system_fingerprint"),
                     ("service_tier", "service_tier")):
        v = body.get(src)
        if isinstance(v, str) and v:
            meta[dst] = v
    # Request id: providers put it in different places.
    rid = body.get("id")
    xg = body.get("x_groq")
    if isinstance(xg, dict):
        rid = xg.get("id") or rid
        if isinstance(xg.get("seed"), int):
            meta["provider_seed"] = xg["seed"]
    if isinstance(rid, str) and rid:
        meta["provider_request_id"] = rid
    return meta


def _usage_from_response(response: requests.Response) -> dict | None:
    """Pull the token counts out of an OpenAI-compatible response.

    Every provider tested on 2026-08-13 (Groq, NVIDIA NIM, Mistral) returns this
    block on every call; the framework simply never read it. Returns only the
    three canonical integer fields — providers bolt on extras (queue_time,
    prompt_tokens_details, ...) that are provider-specific and not comparable
    across the leaderboard.

    Best-effort by contract: a missing or malformed usage block must never fail a
    task, because telemetry is not the measurement.
    """
    try:
        usage = response.json().get("usage") or {}
    except Exception:
        return None
    out = {
        k: usage[k]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance(usage.get(k), int)
    }
    return out or None


def _chat_content_from_response(response: requests.Response, player_id: str) -> str:
    try:
        body = response.json()
    except ValueError:
        content = _parse_sse_chat_content(response.text)
        if content:
            return content
        raise RuntimeError(
            f"{player_id}: OpenAI-compatible endpoint returned non-JSON, "
            "non-SSE response"
        )

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        if isinstance(body, dict) and "error" in body:
            raise RuntimeError(
                f"{player_id}: OpenAI-compatible endpoint returned an error: "
                f"{str(body['error'])[:500]}"
            ) from exc
        raise RuntimeError(
            f"{player_id}: OpenAI-compatible endpoint returned an unexpected "
            f"body shape ({exc}); keys={sorted(body) if isinstance(body, dict) else type(body).__name__}"
        ) from exc


class OpenAIChatCompletionsAdapter(PlayerAdapter):
    """Play a task through a named OpenAI-compatible chat-completions model."""

    def __init__(
        self,
        *args,
        openai_model: str,
        prompt_template_path: str | Path,
        openai_base_url_env: str,
        openai_api_key_env: str,
        openai_available_models_env: str | None = None,
        openai_chat_path: str = "/chat/completions",
        openai_temperature: float = 0.0,
        openai_extra_body: dict | None = None,
        allow_rotating_model: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.openai_model = openai_model.strip()
        self.prompt_template_path = Path(prompt_template_path)
        self.openai_base_url_env = openai_base_url_env
        self.openai_api_key_env = openai_api_key_env
        self.openai_available_models_env = openai_available_models_env
        self.openai_chat_path = openai_chat_path
        self.openai_temperature = openai_temperature
        #: Extra top-level keys merged into the request body. Exists for
        #: REASONING MODELS: NVIDIA's Nemotron 3.x ship with chain-of-thought ON by
        #: default, and the thinking tokens come out of the same budget as the
        #: answer. Measured 2026-08-13 on a real p-curve prompt:
        #:
        #:   nemotron-3.5-lightning  thinking on -> HTTP 504 (gateway gave up)
        #:                           thinking off -> 3.7s, 37 completion tokens
        #:   nemotron-3-super-120b   thinking on -> 172.2s, 2510 tokens
        #:                           thinking off ->  32.1s,   39 tokens
        #:
        #: So `{"chat_template_kwargs": {"enable_thinking": false}}` is the
        #: difference between unusable and fast. This was originally misread as
        #: "NVIDIA models are too slow for this benchmark" — the models were fine;
        #: the call was wrong. Older models (nemotron-nano-9b-v2) ignore the flag,
        #: which is harmless.
        self.openai_extra_body = dict(openai_extra_body) if openai_extra_body else None
        self.allow_rotating_model = allow_rotating_model
        self._template: str | None = None
        self._url: str | None = None
        self._key: str | None = None
        self._last_usage: dict | None = None
        self._last_meta: dict | None = None

    def prepare(self) -> None:
        if not self.openai_model:
            raise RuntimeError(f"{self.player_id}: openai_model must be set")
        # `openai_extra_body` is merged with `body.update()`, so without this it can
        # overwrite ANY top-level request field. Two of them decide what the record
        # is evidence ABOUT, and neither is covered by the allowlist below, which
        # only ever inspects `self.openai_model`: a registry entry carrying
        # `{"model": "llama-3.1-8b-instant"}` sent llama on the wire while
        # `resolved_tool_version()` reported gpt-oss-120b, and nothing failed.
        # (Codex review of the max_tokens change, 2026-08-13; reproduced before
        # fixing.) These belong to dedicated, gated fields - extra_body is for
        # provider options like max_tokens, provider pins and reasoning switches.
        reserved = _RESERVED_BODY_KEYS.intersection(self.openai_extra_body or {})
        if reserved:
            raise RuntimeError(
                f"{self.player_id}: openai_extra_body may not set "
                f"{sorted(reserved)} — those decide which artifact the score is "
                f"about and bypass the openai_model allowlist. Use `openai_model` "
                f"and `prompt_template_path` instead."
            )
        if not self.allow_rotating_model and _is_rotating_model_id(self.openai_model):
            raise RuntimeError(
                f"{self.player_id}: openai_model={self.openai_model!r} is a rotating "
                "router id. Use a named stable model for publishable ScienceArena runs."
            )

        base = os.environ.get(self.openai_base_url_env, "").strip().rstrip("/")
        key = os.environ.get(self.openai_api_key_env, "").strip()
        if not base or not key:
            raise RuntimeError(
                f"{self.player_id}: {self.openai_base_url_env} and "
                f"{self.openai_api_key_env} must both be set in the live environment. "
                "Do not commit provider keys."
            )

        models = env_models(self.openai_available_models_env)
        if models and self.openai_model not in models:
            raise RuntimeError(
                f"{self.player_id}: openai_model={self.openai_model!r} is not in "
                f"{self.openai_available_models_env} ({', '.join(sorted(models))}). "
                "Refusing to run rather than publish a score under the wrong model."
            )

        path = self.openai_chat_path if self.openai_chat_path.startswith("/") else f"/{self.openai_chat_path}"
        self._url = f"{base}{path}"
        self._key = key
        self._template = self.prompt_template_path.read_text(encoding="utf-8")

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._template is None or self._url is None or self._key is None:
            self.prepare()

        # Clear FIRST. `_last_usage` is only assigned on a successful response, so
        # without this a task whose call fails (throttle, schema violation, bad
        # body) would inherit the PREVIOUS task's token counts — the runner reads
        # the attribute either way. Wrong tokens attributed to the wrong task is
        # exactly the silently-wrong-value class: nothing throws, and the cost
        # column just quietly lies.
        self._last_usage = None
        self._last_meta = None

        payload = envelope["input"]
        if isinstance(payload, dict) and "text" in payload:
            input_str = payload["text"]
        else:
            input_str = json.dumps(payload, ensure_ascii=False)
        prompt = self._template.replace("{{INPUT_TEXT}}", input_str)

        # Bounded retry on transient failures. Without it a single burst-limit 429
        # became a permanently errored record, which `aggregate()` excludes from
        # the mean — so a throttled provider looked like a provider with no
        # coverage, and three arenas' worth of mistral-large records read as a
        # model that could not do the task (2026-08-12).
        spent = 0.0
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            body = {
                "model": self.openai_model,
                "temperature": self.openai_temperature,
                "stream": False,
                "max_tokens": _DEFAULT_MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.openai_extra_body:
                body.update(self.openai_extra_body)
            response = requests.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=timeout_s,
            )
            # getattr, not attribute access: a response object without a status
            # must fall through to the original non-retrying path rather than
            # raising here. Retry logic must never break a request that used to work.
            status = getattr(response, "status_code", None)
            retryable = status in _RETRY_STATUSES and attempt < _MAX_ATTEMPTS
            if not retryable:
                break

            delay = _retry_delay(attempt, response)
            if spent + delay > _MAX_TOTAL_BACKOFF_S:
                logger.warning(
                    "%s: HTTP %s and the %.0fs backoff budget is spent after %d "
                    "attempt(s); giving up so the tournament can continue. Use "
                    "`framework retry-failed` to fill the gap.",
                    self.player_id, status, _MAX_TOTAL_BACKOFF_S, attempt,
                )
                break
            logger.warning(
                "%s: HTTP %s from %s (attempt %d/%d); retrying in %.1fs",
                self.player_id, status, self.openai_model,
                attempt, _MAX_ATTEMPTS, delay,
            )
            time.sleep(delay)
            spent += delay

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = response.text.strip()
            except Exception:
                detail = ""
            if detail:
                raise RuntimeError(
                    f"{self.player_id}: OpenAI-compatible endpoint returned HTTP "
                    f"{response.status_code}: {detail[:500]}"
                ) from exc
            raise
        # NOTE ON SEMANTICS: this records the usage of the SUCCEEDING attempt only.
        # Tokens burned by earlier attempts that failed or timed out are not
        # captured anywhere, so on a retry-heavy run the published cost UNDERSTATES
        # what the provider actually billed. That is deliberate — the recorded cost
        # is "the cost of the answer we published", which is what makes a
        # cost-per-catch figure comparable across players — but it is a claim worth
        # stating rather than leaving for someone to discover from a bill.
        # (Fable 5 cross-review, 2026-08-15.)
        self._last_usage = _usage_from_response(response)
        self._last_meta = _response_meta(response, attempt)
        content = _chat_content_from_response(response, self.player_id)
        return _extract_json_value(content)

    def last_usage(self) -> dict | None:
        return self._last_usage

    def last_response_meta(self) -> dict | None:
        return self._last_meta

    def resolved_tool_version(self) -> str | None:
        return self.openai_model


register_adapter_class("OpenAIChatCompletionsAdapter", OpenAIChatCompletionsAdapter)
