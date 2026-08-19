"""OpenAI-compatible chat-completions player, for the Kaggle Benchmarks proxy.

WHY A NEW ADAPTER, NOT `HttpAdapter`
------------------------------------
The plan said to reuse `framework.player_adapter.HttpAdapter`. That does not
work: `HttpAdapter` is ESCImate-specific — it POSTs `{"text": ...}` to one
endpoint and its `_normalize` maps ESCImate's `stat_value` / `raw_text` result
rows into the output schema. It is an ESCImate client that happens to speak
HTTP, not a generic one.

So this is a new adapter, and it is the first HTTP-based *model* player: all 69
existing model players shell out through `SubprocessCliAdapter`. It deliberately
mirrors that class's contract exactly —

  * same prompt template files (`players/prompts/*.txt`, `{{INPUT_TEXT}}`),
  * same input flattening (`input.text` if present, else compact JSON),
  * same permissive JSON recovery from a chatty response,

— so a model run through the proxy is scored on the same terms as the same
family of model run through a CLI. If the two disagreed on prompt or parsing,
any comparison between a proxy player and a CLI player would be measuring the
harness rather than the model.

CONFIGURATION
-------------
Kaggle Benchmarks injects two variables into the execution environment:

    MODEL_PROXY_URL      base URL; the OpenAI-shaped API lives at <base>/openapi
    MODEL_PROXY_API_KEY  bearer token for it
    LLMS_AVAILABLE       comma/JSON list of model ids the grant has credit for

The model list is READ from `LLMS_AVAILABLE` rather than hardcoded, so the
registry does not silently claim a model the grant no longer funds. A registry
entry naming a model absent from that list fails loudly at `prepare()` — a run
that quietly fell back to some other model would publish a score under the wrong
model's name, which is a data-integrity fault, not a convenience.

No key is ever read from a committed file, and none is written to a run record.
"""
from __future__ import annotations

import json
import os

import requests

from framework.player_adapter import (
    PlayerAdapter,
    _extract_json_value,
    register_adapter_class,
)

#: Base URL of the Kaggle model proxy.
PROXY_URL_ENV = "MODEL_PROXY_URL"
#: Bearer token for it.
PROXY_KEY_ENV = "MODEL_PROXY_API_KEY"
#: Models the grant currently funds.
MODELS_ENV = "LLMS_AVAILABLE"


def available_models() -> list[str]:
    """Model ids from ``LLMS_AVAILABLE``. Empty list when unset.

    The variable is documented as a list but arrives as either JSON or a
    comma-separated string depending on the runner, so both are accepted. An
    unset variable returns ``[]``, which callers must treat as "unknown", NOT as
    "no models" — see `ModelProxyAdapter.prepare`.
    """
    raw = (os.environ.get(MODELS_ENV) or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            return [str(m).strip() for m in json.loads(raw) if str(m).strip()]
        except (ValueError, TypeError):
            pass
    return [m.strip() for m in raw.replace("\n", ",").split(",") if m.strip()]


class ModelProxyAdapter(PlayerAdapter):
    """Plays a task via an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, *args, proxy_model: str, prompt_template_path: str,
                 proxy_temperature: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.proxy_model = proxy_model
        self.prompt_template_path = prompt_template_path
        self.proxy_temperature = proxy_temperature
        self._template: str | None = None
        self._url: str | None = None
        self._key: str | None = None

    def prepare(self) -> None:
        base = (os.environ.get(PROXY_URL_ENV) or "").strip().rstrip("/")
        key = (os.environ.get(PROXY_KEY_ENV) or "").strip()
        if not base or not key:
            raise RuntimeError(
                f"{PROXY_URL_ENV} and {PROXY_KEY_ENV} must both be set to run "
                f"{self.player_id}. They are provided by the Kaggle Benchmarks "
                f"execution environment; do not commit them."
            )

        models = available_models()
        if models and self.proxy_model not in models:
            # Loud, not a fallback. Scoring a different model under this
            # player's name would publish a wrong attribution that no test
            # could detect after the fact.
            raise RuntimeError(
                f"{self.player_id} declares proxy_model={self.proxy_model!r}, which is "
                f"not in {MODELS_ENV} ({', '.join(sorted(models))}). Refusing to run "
                f"rather than silently scoring a different model under this name."
            )
        # `models == []` means the variable was not set — e.g. a local dry run.
        # That is unknown, not empty, so it is not treated as a rejection.

        with open(self.prompt_template_path, encoding="utf-8") as f:
            self._template = f.read()
        self._url = f"{base}/openapi/chat/completions"
        self._key = key

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._template is None or self._url is None:
            self.prepare()

        # Identical flattening to SubprocessCliAdapter — see the module docstring.
        payload = envelope["input"]
        if isinstance(payload, dict) and "text" in payload:
            input_str = payload["text"]
        else:
            input_str = json.dumps(payload, ensure_ascii=False)
        prompt = self._template.replace("{{INPUT_TEXT}}", input_str)

        response = requests.post(
            self._url,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            json={
                "model": self.proxy_model,
                "temperature": self.proxy_temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"model proxy returned an unexpected body shape ({exc}); "
                f"keys={sorted(body) if isinstance(body, dict) else type(body).__name__}"
            ) from exc
        # Same permissive recovery as the CLI path: models wrap JSON in fences,
        # prepend prose, or append a closing sentence.
        return _extract_json_value(content)


register_adapter_class("ModelProxyAdapter", ModelProxyAdapter)
