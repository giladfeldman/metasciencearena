from __future__ import annotations

import json

import pytest

from players.adapters.openai_compatible import (
    OpenAIChatCompletionsAdapter,
    env_models,
)
from framework.player_adapter import build_adapter


BASE_KWARGS = dict(
    player_id="nvidia-nemotron-free-grim",
    player_version="nvidia/nemotron-nano-9b-v2",
    player_type="ai-model",
    confidence_strategy="native",
    deterministic=False,
)


@pytest.fixture
def prompt_file(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("Return JSON only.\n{{INPUT_TEXT}}", encoding="utf-8")
    return path


def _adapter(prompt_file, model="nvidia/nemotron-nano-9b-v2", **kwargs):
    defaults = dict(
        openai_model=model,
        prompt_template_path=str(prompt_file),
        openai_base_url_env="FREE_BASE_URL",
        openai_api_key_env="FREE_API_KEY",
    )
    defaults.update(kwargs)
    return OpenAIChatCompletionsAdapter(**BASE_KWARGS, **defaults)


def test_env_models_parses_json_and_csv(monkeypatch):
    monkeypatch.setenv("MODELS", '["a", "b"]')
    assert env_models("MODELS") == ["a", "b"]
    monkeypatch.setenv("MODELS", "a,b\nc")
    assert env_models("MODELS") == ["a", "b", "c"]
    assert env_models(None) == []


def test_missing_credentials_raise(prompt_file, monkeypatch):
    monkeypatch.delenv("FREE_BASE_URL", raising=False)
    monkeypatch.delenv("FREE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FREE_BASE_URL"):
        _adapter(prompt_file).prepare()


def test_rotating_model_ids_are_rejected_by_default(prompt_file, monkeypatch):
    monkeypatch.setenv("FREE_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("FREE_API_KEY", "k")
    with pytest.raises(RuntimeError, match="rotating router id"):
        _adapter(prompt_file, model="auto").prepare()
    with pytest.raises(RuntimeError, match="rotating router id"):
        _adapter(prompt_file, model="auto/best-free").prepare()
    with pytest.raises(RuntimeError, match="rotating router id"):
        _adapter(prompt_file, model="openrouter/auto").prepare()


def test_model_allowlist_prevents_misattribution(prompt_file, monkeypatch):
    monkeypatch.setenv("FREE_BASE_URL", "https://api.invalid/v1")
    monkeypatch.setenv("FREE_API_KEY", "k")
    monkeypatch.setenv("FREE_MODELS", "mistral-small-latest,deepseek-chat")
    adapter = _adapter(
        prompt_file,
        model="nvidia/nemotron-nano-9b-v2",
        openai_available_models_env="FREE_MODELS",
    )
    with pytest.raises(RuntimeError, match="not in FREE_MODELS"):
        adapter.prepare()


def test_prompt_request_and_json_recovery(prompt_file, monkeypatch):
    monkeypatch.setenv("FREE_BASE_URL", "https://api.invalid/v1/")
    monkeypatch.setenv("FREE_API_KEY", "secret")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "```json\n{\"ok\": true}\n```"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr("players.adapters.openai_compatible.requests.post", fake_post)
    adapter = _adapter(prompt_file, openai_chat_path="/chat/completions")

    assert adapter.play_task({"input": {"text": "TASK"}}, timeout_s=15) == {"ok": True}
    assert captured["url"] == "https://api.invalid/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "nvidia/nemotron-nano-9b-v2"
    assert captured["body"]["stream"] is False
    assert "TASK" in captured["body"]["messages"][0]["content"]
    assert captured["timeout"] == 15
    assert adapter.resolved_tool_version() == "nvidia/nemotron-nano-9b-v2"


def test_sse_chat_response_is_supported(prompt_file, monkeypatch):
    monkeypatch.setenv("FREE_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("FREE_API_KEY", "local")

    class FakeResponse:
        text = "\n".join(
            [
                'data: {"choices":[{"delta":{"role":"assistant"}}]}',
                'data: {"choices":[{"delta":{"content":"{\\"ok\\":"}}]}',
                'data: {"choices":[{"delta":{"content":" true}"}}]}',
                "data: [DONE]",
            ]
        )

        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(
        "players.adapters.openai_compatible.requests.post",
        lambda url, headers=None, json=None, timeout=None: FakeResponse(),
    )

    assert _adapter(prompt_file).play_task({"input": {"text": "x"}}, timeout_s=5) == {"ok": True}


def test_non_text_input_is_compact_json(prompt_file, monkeypatch):
    monkeypatch.setenv("FREE_BASE_URL", "https://api.invalid/v1")
    monkeypatch.setenv("FREE_API_KEY", "k")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(
        "players.adapters.openai_compatible.requests.post",
        lambda url, headers=None, json=None, timeout=None: (captured.update(body=json), FakeResponse())[1],
    )
    _adapter(prompt_file).play_task({"input": {"doi": "10.1/x", "title": "T"}}, timeout_s=5)
    sent = captured["body"]["messages"][0]["content"]
    assert json.dumps({"doi": "10.1/x", "title": "T"}, ensure_ascii=False) in sent


def test_unexpected_body_shape_raises(prompt_file, monkeypatch):
    monkeypatch.setenv("FREE_BASE_URL", "https://api.invalid/v1")
    monkeypatch.setenv("FREE_API_KEY", "k")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"error": {"message": "quota exceeded"}}

    monkeypatch.setattr(
        "players.adapters.openai_compatible.requests.post",
        lambda url, headers=None, json=None, timeout=None: FakeResponse(),
    )
    with pytest.raises(RuntimeError, match="quota exceeded"):
        _adapter(prompt_file).play_task({"input": {"text": "x"}}, timeout_s=5)


def test_http_error_includes_provider_body(prompt_file, monkeypatch):
    monkeypatch.setenv("FREE_BASE_URL", "https://api.invalid/v1")
    monkeypatch.setenv("FREE_API_KEY", "k")

    class FakeResponse:
        status_code = 429
        text = '{"error":{"message":"rate limit exceeded"}}'

        def raise_for_status(self):
            import requests

            raise requests.HTTPError("429 Client Error")

    monkeypatch.setattr(
        "players.adapters.openai_compatible.requests.post",
        lambda url, headers=None, json=None, timeout=None: FakeResponse(),
    )
    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        _adapter(prompt_file).play_task({"input": {"text": "x"}}, timeout_s=5)


def test_build_adapter_accepts_registry_configuration(prompt_file):
    entry = {
        **BASE_KWARGS,
        "adapter_class": "OpenAIChatCompletionsAdapter",
        "openai_model": "nvidia/nemotron-nano-9b-v2",
        "openai_base_url_env": "FREE_BASE_URL",
        "openai_api_key_env": "FREE_API_KEY",
        "openai_available_models_env": "FREE_MODELS",
        "prompt_template_path": str(prompt_file),
        "openai_temperature": 0.1,
    }

    adapter = build_adapter(entry)

    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.openai_model == "nvidia/nemotron-nano-9b-v2"
    assert adapter.openai_available_models_env == "FREE_MODELS"


# --- 429 backoff (2026-08-12) -------------------------------------------------
#
# Found while promoting mistral-large-latest across 11 arenas: La Plateforme
# returns HTTP 429 on back-to-back requests, and the adapter fired ONE
# requests.post with no retry, so a transient throttle became a permanently
# errored record. 8 of 33 smoke records died that way, and in three arenas EVERY
# task errored — on a leaderboard that is indistinguishable from a model that
# cannot do the task at all.
#
# `framework retry-failed` exists but works at tournament level, a whole extra
# pass per round; it cannot absorb a burst limit that resets within a second.

import players.adapters.openai_compatible as oc


class _FakeResp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


_OK_BODY = {"choices": [{"message": {"content": '{"ok": true}'}}]}


@pytest.fixture
def _prepared(prompt_file, monkeypatch):
    monkeypatch.setenv("FREE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FREE_API_KEY", "k")
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)  # never wait in tests
    a = _adapter(prompt_file)
    a.prepare()
    return a


def test_retries_on_429_then_succeeds(_prepared, monkeypatch):
    """A transient burst limit must not become a permanent errored record."""
    calls = []

    def fake_post(url, **kw):
        calls.append(1)
        return _FakeResp(429, text="rate limited") if len(calls) < 3 else _FakeResp(200, _OK_BODY)

    monkeypatch.setattr(oc.requests, "post", fake_post)
    assert _prepared.play_task({"input": {"text": "hi"}}, timeout_s=30) == {"ok": True}
    assert len(calls) == 3, "should have retried twice before succeeding"


def test_gives_up_after_the_cap_and_still_raises(_prepared, monkeypatch):
    """Bounded, not infinite - and a persistent 429 is still an honest failure."""
    calls = []
    monkeypatch.setattr(
        oc.requests, "post",
        lambda url, **kw: (calls.append(1), _FakeResp(429, text="nope"))[1],
    )
    with pytest.raises(RuntimeError) as exc:
        _prepared.play_task({"input": {"text": "hi"}}, timeout_s=30)
    assert "429" in str(exc.value)
    assert len(calls) == oc._MAX_ATTEMPTS


def test_does_not_retry_a_real_client_error(_prepared, monkeypatch):
    """A 400 is the caller's fault and will never succeed; retrying just burns quota."""
    calls = []
    monkeypatch.setattr(
        oc.requests, "post",
        lambda url, **kw: (calls.append(1), _FakeResp(400, text="bad model"))[1],
    )
    with pytest.raises(RuntimeError):
        _prepared.play_task({"input": {"text": "hi"}}, timeout_s=30)
    assert len(calls) == 1, "400 must not be retried"


def test_usage_is_captured_from_the_response(_prepared, monkeypatch):
    """The `usage` block every provider returns was discarded for months."""
    body = dict(_OK_BODY, usage={"prompt_tokens": 651, "completion_tokens": 170,
                                 "total_tokens": 821})
    monkeypatch.setattr(oc.requests, "post", lambda url, **kw: _FakeResp(200, body))
    _prepared.play_task({"input": {"text": "hi"}}, timeout_s=30)
    assert _prepared.last_usage() == {"prompt_tokens": 651, "completion_tokens": 170,
                                      "total_tokens": 821}


def test_a_failed_call_does_not_inherit_the_previous_task_s_tokens(_prepared, monkeypatch):
    """The stale-attribution bug: `_last_usage` is only set on success, so without
    an explicit reset the runner would stamp task N-1's tokens onto task N."""
    first = dict(_OK_BODY, usage={"prompt_tokens": 651, "completion_tokens": 170,
                                  "total_tokens": 821})
    monkeypatch.setattr(oc.requests, "post", lambda url, **kw: _FakeResp(200, first))
    _prepared.play_task({"input": {"text": "hi"}}, timeout_s=30)
    assert _prepared.last_usage()["prompt_tokens"] == 651

    # Now a call that fails outright.
    monkeypatch.setattr(oc.requests, "post",
                        lambda url, **kw: _FakeResp(400, text="bad request"))
    with pytest.raises(RuntimeError):
        _prepared.play_task({"input": {"text": "different task"}}, timeout_s=30)
    assert _prepared.last_usage() is None, (
        "a failed task must report no usage, not the previous task's tokens"
    )


def test_missing_usage_block_is_none_not_zero(_prepared, monkeypatch):
    monkeypatch.setattr(oc.requests, "post", lambda url, **kw: _FakeResp(200, _OK_BODY))
    _prepared.play_task({"input": {"text": "hi"}}, timeout_s=30)
    assert _prepared.last_usage() is None


def test_an_output_ceiling_is_always_sent(_prepared, monkeypatch):
    """Leaving `max_tokens` unset does not mean "no limit" - it means the PROVIDER
    picks one, and they differ.

    Found 2026-08-13: with nothing sent, OpenRouter applied the pinned endpoint's
    40960-token ceiling and then reserved credit for all of it, so once the key's
    balance fell below that reservation every call died HTTP 402 ("You requested
    up to 40960 tokens, but can only afford 39463") - 45 of 46 tasks on
    prereg-deviation-v1 scored 0.0 for a reason that had nothing to do with the
    model. The same silence is a comparability defect even when it does not fail:
    two providers scoring the same player under different output ceilings are not
    running the same benchmark.
    """
    sent = {}
    monkeypatch.setattr(oc.requests, "post",
                        lambda url, **kw: (sent.update(kw["json"]), _FakeResp(200, _OK_BODY))[1])
    _prepared.play_task({"input": {"text": "hi"}}, timeout_s=30)
    assert sent.get("max_tokens") == oc._DEFAULT_MAX_TOKENS


def test_a_registry_entry_can_raise_the_output_ceiling(_prepared, monkeypatch):
    """The default is a floor against provider drift, not a hard cap: an arena
    whose gold answers are genuinely long must be able to ask for more."""
    sent = {}
    monkeypatch.setattr(oc.requests, "post",
                        lambda url, **kw: (sent.update(kw["json"]), _FakeResp(200, _OK_BODY))[1])
    _prepared.openai_extra_body = {"max_tokens": 32000}
    _prepared.play_task({"input": {"text": "hi"}}, timeout_s=30)
    assert sent["max_tokens"] == 32000


def test_router_backend_is_recorded_not_just_the_model_id(_prepared, monkeypatch):
    """On a ROUTER, `served_model` does not identify what actually ran.

    Found 2026-08-13 while wiring OpenRouter as an independent quota for
    gpt-oss-120b: a probe came back `"model": "openai/gpt-oss-120b",
    "provider": "Amazon Bedrock"`. OpenRouter fans the same model id out over
    CoreWeave / DeepInfra / Novita / Bedrock / Google, which differ in
    quantization and serving stack. Recording only the model id publishes a
    score whose backend is unrecoverable afterwards - the exact failure
    docs/RUN_ARTIFACT_RETENTION.md exists to prevent.
    """
    body = dict(_OK_BODY, model="openai/gpt-oss-120b", provider="Amazon Bedrock")
    monkeypatch.setattr(oc.requests, "post", lambda url, **kw: _FakeResp(200, body))
    _prepared.play_task({"input": {"text": "hi"}}, timeout_s=30)
    meta = _prepared.last_response_meta()
    assert meta["served_model"] == "openai/gpt-oss-120b"
    assert meta["served_by"] == "Amazon Bedrock", (
        "the routed backend must be recorded; it cannot be reconstructed later"
    )


def test_served_by_is_absent_for_a_direct_provider(_prepared, monkeypatch):
    """Direct providers return no `provider` field; absent must stay absent
    rather than becoming a fabricated value."""
    monkeypatch.setattr(oc.requests, "post",
                        lambda url, **kw: _FakeResp(200, dict(_OK_BODY, model="m")))
    _prepared.play_task({"input": {"text": "hi"}}, timeout_s=30)
    assert "served_by" not in _prepared.last_response_meta()


# --- the REAL registry, not a fixture --------------------------------------
#
# Provider tiers are the thing that actually breaks these runs, and a tier lives
# in a constant that outlives the reason it was chosen. These tests pin each
# ceiling to the measurement that justified it.

#: Groq free tier: prompt + max_tokens are counted TOGETHER against one budget,
#: and the check happens before the model runs.
_GROQ_TPM_LIMIT = 8000
#: Largest prompt observed across all 17 text arenas (reference-integrity-v1),
#: measured 2026-08-13 over every recorded `usage.prompt_tokens`.
_WORST_ARENA_PROMPT_TOKENS = 1553


def _registry_players():
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    return yaml.safe_load((root / "players" / "registry.yaml").read_text(encoding="utf-8"))


def test_every_groq_player_fits_the_free_tier_token_budget():
    """Groq answers HTTP 413 BEFORE running the model if prompt + max_tokens
    exceeds its per-minute budget.

    On 2026-08-13 an explicit 32000-token ceiling was added to every request and
    Groq rejected 46 of 46 prereg-deviation tasks with "Requested 33115, Limit
    8000" - an entire arena scored 0.0 without the model being consulted once.
    Any future edit that raises this ceiling, or adds a Groq player without one,
    reproduces that silently until someone reads the error strings.
    """
    groq = [p for p in _registry_players() if p.get("openai_api_key_env") == "GROQ_API_KEY"]
    assert groq, "no Groq players found - has the key env var been renamed?"
    for p in groq:
        cap = (p.get("openai_extra_body") or {}).get("max_tokens")
        assert isinstance(cap, int), (
            f"{p['player_id']}: no max_tokens override, so the adapter default "
            f"({oc._DEFAULT_MAX_TOKENS}) applies and Groq will 413 every request"
        )
        assert cap + _WORST_ARENA_PROMPT_TOKENS <= _GROQ_TPM_LIMIT, (
            f"{p['player_id']}: max_tokens={cap} plus the worst arena prompt "
            f"({_WORST_ARENA_PROMPT_TOKENS}) exceeds Groq's {_GROQ_TPM_LIMIT} TPM"
        )


def test_router_players_pin_a_backend_and_refuse_substitution():
    """A router that silently falls back scores a different artifact under this
    player's name. OpenRouter serves openai/gpt-oss-120b at fp4 on most backends
    and fp16 only on Cerebras, so an unpinned fallback is a quantization change."""
    routed = [p for p in _registry_players()
              if p.get("openai_api_key_env") == "OPENROUTER_API_KEY"]
    assert routed, "no OpenRouter players found"
    for p in routed:
        prov = (p.get("openai_extra_body") or {}).get("provider") or {}
        assert prov.get("order"), f"{p['player_id']}: no provider pin"
        assert prov.get("allow_fallbacks") is False, (
            f"{p['player_id']}: fallbacks enabled - a substitution would be scored "
            f"under this id with no gate to catch it"
        )


def test_extra_body_cannot_swap_the_model(prompt_file, monkeypatch):
    """`openai_extra_body` is merged with `body.update()`, so before this guard a
    registry entry could silently replace `model`.

    Found by a Codex review of the max_tokens change, 2026-08-13, and reproduced:
    with `openai_extra_body: {model: llama-3.1-8b-instant}` the request went out as
    llama while `resolved_tool_version()` still reported gpt-oss-120b - and
    `openai_available_models_env`, the allowlist whose whole job is preventing
    misattribution, never saw it because it only inspects `self.openai_model`.
    Every record from such a player names the wrong artifact, and nothing fails.
    """
    monkeypatch.setenv("FREE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FREE_API_KEY", "k")
    with pytest.raises(RuntimeError, match="openai_extra_body"):
        _adapter(prompt_file, openai_extra_body={"model": "llama-3.1-8b-instant"}).prepare()


def test_extra_body_cannot_replace_the_prompt(prompt_file, monkeypatch):
    """Same hole, other identity-critical field: `messages` IS the task."""
    monkeypatch.setenv("FREE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FREE_API_KEY", "k")
    with pytest.raises(RuntimeError, match="openai_extra_body"):
        _adapter(prompt_file,
                 openai_extra_body={"messages": [{"role": "user", "content": "hi"}]}).prepare()


def test_extra_body_still_allows_genuine_provider_options(prompt_file, monkeypatch):
    """The guard must not block what extra_body is FOR - tier ceilings, provider
    pins, and reasoning switches are all legitimate."""
    monkeypatch.setenv("FREE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FREE_API_KEY", "k")
    a = _adapter(prompt_file, openai_extra_body={
        "max_tokens": 6000,
        "provider": {"order": ["Cerebras"], "allow_fallbacks": False},
        "chat_template_kwargs": {"enable_thinking": False},
    })
    a.prepare()   # must not raise
