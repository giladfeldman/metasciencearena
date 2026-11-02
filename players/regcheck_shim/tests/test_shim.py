"""Tests for the local OpenAI-compatible codex shim.

The shim translates OpenAI chat-completions / embeddings requests into local
calls (codex CLI for chat; sentence-transformers or a hashing fallback for
embeddings) so regcheck can run with NO OpenAI API key.

These tests monkeypatch the network-bound primitives (`codex_complete`) so they
run fast and offline.
"""
from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from players.regcheck_shim import openai_shim


def _post(base_url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        base_url + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


@pytest.fixture()
def running_shim(monkeypatch):
    """Start make_server(0) on an ephemeral port in a thread; yield its base URL.

    Embeddings are forced onto the deterministic hashing fallback via the shim's
    own `REGCHECK_SHIM_NO_ST` switch. Without it these tests were NOT offline as
    the module docstring claims: `/v1/embeddings` lazily constructs
    `SentenceTransformer("all-MiniLM-L6-v2")`, which fetches and loads the model
    on first use and blew past `_post`'s 10s timeout. The symptom was an
    order-dependent flake — whichever embeddings test ran first paid the load and
    failed, the second passed off the in-process cache — so the file reported
    "1 failed, 3 passed" no matter which test was actually at fault.

    `_ST_TRIED` must be reset too: `_get_st_model` short-circuits on it, so the
    env var alone is ignored once any earlier test in the session has already
    loaded the model.

    These tests pin the shim's HTTP contract (shape, determinism, distinctness),
    all of which the hashing fallback satisfies — embedding *quality* is not what
    they measure.
    """
    monkeypatch.setenv("REGCHECK_SHIM_NO_ST", "1")
    monkeypatch.setattr(openai_shim, "_ST_TRIED", False)
    monkeypatch.setattr(openai_shim, "_ST_MODEL", None)
    monkeypatch.setattr(openai_shim, "codex_complete", lambda prompt, **kw: "STUBBED")
    server = openai_shim.make_server(port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_chat_completions_returns_codex_answer(running_shim):
    body = {
        "model": "gpt-5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hi."},
        ],
    }
    data = _post(running_shim, "/v1/chat/completions", body)
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "STUBBED"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in data


def test_chat_completions_concatenates_messages(monkeypatch, running_shim):
    captured = {}

    def fake(prompt, **kw):
        captured["prompt"] = prompt
        return "OK"

    monkeypatch.setattr(openai_shim, "codex_complete", fake)
    body = {
        "model": "gpt-5",
        "messages": [
            {"role": "system", "content": "SYS_MARKER"},
            {"role": "user", "content": "USER_MARKER"},
        ],
    }
    _post(running_shim, "/v1/chat/completions", body)
    assert "SYS_MARKER" in captured["prompt"]
    assert "USER_MARKER" in captured["prompt"]


def test_embeddings_shape_and_determinism(running_shim):
    body = {"model": "text-embedding-3-large", "input": ["alpha", "beta"]}
    data = _post(running_shim, "/v1/embeddings", body)
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert data["data"][0]["object"] == "embedding"
    vec0 = data["data"][0]["embedding"]
    assert isinstance(vec0, list) and len(vec0) > 0
    # Determinism: same input -> same vector.
    data2 = _post(running_shim, "/v1/embeddings", body)
    assert data2["data"][0]["embedding"] == vec0
    # Distinct inputs -> distinct vectors.
    assert data["data"][1]["embedding"] != vec0


def test_embeddings_accepts_single_string(running_shim):
    body = {"model": "text-embedding-3-large", "input": "just one"}
    data = _post(running_shim, "/v1/embeddings", body)
    assert len(data["data"]) == 1


def test_codex_runs_contained_with_the_prompt_on_stdin(monkeypatch):
    # Failing-first 2026-10-30: codex_complete ran `codex exec` with the runner's cwd
    # (the repo, which holds arena answer keys), the user's config, no sandbox, and the
    # whole prompt as argv (full papers exceed Windows' 32,767-char command line).
    import subprocess
    from pathlib import Path
    seen = {}

    def fake_run(argv, input=None, **kw):
        seen.update(argv=list(argv), input=input, cwd=kw.get("cwd"))
        out = argv[argv.index("-o") + 1]
        Path(out).write_text('{"ok": true}', encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(openai_shim.subprocess, "run", fake_run)
    monkeypatch.setattr(openai_shim, "CODEX_EFFORT", "low")
    prompt = "Compare the preregistration and the paper. " * 2000
    assert openai_shim.codex_complete(prompt, want_json=True) == '{"ok": true}'
    argv = seen["argv"]
    assert "--sandbox" in argv and argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in argv
    assert prompt not in argv and seen["input"] == prompt
    repo = Path(__file__).resolve().parents[3]
    cwd = Path(seen["cwd"]).resolve()
    assert cwd != repo and repo not in cwd.parents
    # Found by the 2026-10-30 smoke run: codex exits 2 on a flag given twice
    # ("--skip-git-repo-check cannot be used multiple times"), which the fake hid.
    flags = [a for a in argv if a.startswith("--")]
    assert len(flags) == len(set(flags)), flags
