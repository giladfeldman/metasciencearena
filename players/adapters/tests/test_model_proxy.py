"""The Kaggle model-proxy adapter must never mis-attribute a score.

The dangerous failure is not a crash. It is running `gpt-4o-mini` because
`gpt-5` was unavailable, and publishing the result under a player named for
gpt-5 — a wrong number about a real model, with nothing failing.
"""
from __future__ import annotations

import json

import pytest

from players.adapters.model_proxy import (
    MODELS_ENV,
    PROXY_KEY_ENV,
    PROXY_URL_ENV,
    ModelProxyAdapter,
    available_models,
)

BASE_KWARGS = dict(
    player_id="kaggle-gpt-5-grim",
    player_version="kaggle-proxy/gpt-5",
    player_type="ai-model",
    confidence_strategy="native",
    deterministic=False,
)


@pytest.fixture
def prompt_file(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("Answer as JSON.\n\n{{INPUT_TEXT}}", encoding="utf-8")
    return p


def _adapter(prompt_file, model="gpt-5"):
    return ModelProxyAdapter(**BASE_KWARGS, proxy_model=model, prompt_template_path=str(prompt_file))


def test_available_models_parses_both_documented_shapes(monkeypatch):
    monkeypatch.setenv(MODELS_ENV, "gpt-5, gemini-pro-3.1 ,deepseek-chat")
    assert available_models() == ["gpt-5", "gemini-pro-3.1", "deepseek-chat"]
    monkeypatch.setenv(MODELS_ENV, '["gpt-5", "o4"]')
    assert available_models() == ["gpt-5", "o4"]


def test_unset_models_env_means_UNKNOWN_not_empty(monkeypatch, prompt_file):
    """An unset variable must not be read as "no models are available".

    Treating unknown as empty would make every player fail on a local dry run;
    treating it as permission is the correct reading, because the check exists
    to catch a model the grant *stopped* funding, not to gate local use.
    """
    monkeypatch.delenv(MODELS_ENV, raising=False)
    monkeypatch.setenv(PROXY_URL_ENV, "https://proxy.invalid")
    monkeypatch.setenv(PROXY_KEY_ENV, "k")
    assert available_models() == []
    _adapter(prompt_file).prepare()  # must not raise


def test_refuses_a_model_the_grant_does_not_fund(monkeypatch, prompt_file):
    """The mis-attribution guard: fail loudly rather than substitute."""
    monkeypatch.setenv(PROXY_URL_ENV, "https://proxy.invalid")
    monkeypatch.setenv(PROXY_KEY_ENV, "k")
    monkeypatch.setenv(MODELS_ENV, "gemini-pro-3.1,deepseek-chat")
    with pytest.raises(RuntimeError, match="not in LLMS_AVAILABLE"):
        _adapter(prompt_file, model="gpt-5").prepare()


def test_missing_credentials_raise_rather_than_run(monkeypatch, prompt_file):
    monkeypatch.delenv(PROXY_URL_ENV, raising=False)
    monkeypatch.delenv(PROXY_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=PROXY_URL_ENV):
        _adapter(prompt_file).prepare()


def test_prompt_and_parsing_match_the_cli_path(monkeypatch, prompt_file):
    """A proxy player and a CLI player must be scored on the same terms.

    Same `{{INPUT_TEXT}}` substitution, same permissive JSON recovery from a
    chatty reply. If these diverged, comparing a proxy player against a CLI
    player would measure the harness rather than the models.
    """
    monkeypatch.setenv(PROXY_URL_ENV, "https://proxy.invalid/")
    monkeypatch.setenv(PROXY_KEY_ENV, "secret-key")
    monkeypatch.delenv(MODELS_ENV, raising=False)

    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            # Deliberately chatty + fenced: the shape CLI players already handle.
            return {"choices": [{"message": {
                "content": "Sure!\n```json\n{\"answer\": 42}\n```\nHope that helps."
            }}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr("players.adapters.model_proxy.requests.post", fake_post)

    adapter = _adapter(prompt_file)
    out = adapter.play_task({"input": {"text": "TASK BODY"}}, timeout_s=30)

    assert out == {"answer": 42}, "permissive JSON recovery diverged from the CLI path"
    assert captured["url"] == "https://proxy.invalid/openapi/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["body"]["model"] == "gpt-5"
    assert "TASK BODY" in captured["body"]["messages"][0]["content"]
    assert "Answer as JSON." in captured["body"]["messages"][0]["content"]


def test_non_text_input_is_flattened_as_compact_json(monkeypatch, prompt_file):
    monkeypatch.setenv(PROXY_URL_ENV, "https://proxy.invalid")
    monkeypatch.setenv(PROXY_KEY_ENV, "k")
    monkeypatch.delenv(MODELS_ENV, raising=False)
    captured = {}

    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(
        "players.adapters.model_proxy.requests.post",
        lambda url, headers=None, json=None, timeout=None: (captured.update(body=json), FakeResponse())[1],
    )
    adapter = _adapter(prompt_file)
    adapter.play_task({"input": {"doi": "10.1/x", "title": "T"}}, timeout_s=5)
    sent = captured["body"]["messages"][0]["content"]
    assert json.dumps({"doi": "10.1/x", "title": "T"}, ensure_ascii=False) in sent


def test_an_unexpected_body_shape_raises_with_a_diagnosis(monkeypatch, prompt_file):
    monkeypatch.setenv(PROXY_URL_ENV, "https://proxy.invalid")
    monkeypatch.setenv(PROXY_KEY_ENV, "k")
    monkeypatch.delenv(MODELS_ENV, raising=False)

    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"error": {"message": "quota exceeded"}}

    monkeypatch.setattr(
        "players.adapters.model_proxy.requests.post",
        lambda url, headers=None, json=None, timeout=None: FakeResponse(),
    )
    with pytest.raises(RuntimeError, match="unexpected body shape"):
        _adapter(prompt_file).play_task({"input": {"text": "x"}}, timeout_s=5)
