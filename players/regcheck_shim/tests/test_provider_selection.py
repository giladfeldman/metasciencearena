"""Tests for RegcheckShimAdapter provider selection (groq / deepseek / openai).

These cover the 2026-07-02 multi-provider wiring — one adapter class serving three
distinct players via the registry fields `regcheck_client` / `regcheck_model` —
WITHOUT starting regcheck or hitting any network. We stub out the shim server so
prepare() exercises only the client-resolution + key-validation logic.
"""
from __future__ import annotations

import pytest

from framework.player_adapter import build_adapter
from players.adapters import regcheck_shim
from players.adapters.regcheck_shim import RegcheckShimAdapter


@pytest.fixture(autouse=True)
def _no_server(monkeypatch, tmp_path):
    """Stop prepare() from actually binding a socket / starting a thread."""
    class _FakeServer:
        server_address = ("127.0.0.1", 65535)

        def serve_forever(self):  # never called (thread is stubbed)
            pass

        def shutdown(self):
            pass

        def server_close(self):
            pass

    monkeypatch.setattr(regcheck_shim.openai_shim, "make_server", lambda port=0: _FakeServer())

    class _FakeThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(regcheck_shim.threading, "Thread", _FakeThread)
    # A REGCHECK_DIR with a backend/cli.py so prepare() passes its path check.
    rc = tmp_path / "regcheck"
    (rc / "backend").mkdir(parents=True)
    (rc / "backend" / "cli.py").write_text("# stub\n", encoding="utf-8")
    monkeypatch.setenv("REGCHECK_DIR", str(rc))
    monkeypatch.setenv("REGCHECK_PYTHON", str(tmp_path / "python.exe"))
    # load_dimension_kinds reads the real catalog; keep it working by no-op patch.
    monkeypatch.setattr(regcheck_shim, "load_dimension_kinds", lambda: {})


def _adapter(**registry_extra):
    entry = {
        "player_id": registry_extra.pop("player_id", "regcheck-x"),
        "player_version": "regcheck-test",
        "player_type": "tool",
        "adapter_class": "RegcheckShimAdapter",
        "confidence_strategy": "native",
        "deterministic": False,
        **registry_extra,
    }
    return build_adapter(entry)


def test_registry_client_overrides_env(monkeypatch):
    # env says groq, registry pins deepseek -> deepseek wins.
    monkeypatch.setenv("REGCHECK_CLIENT", "groq")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    a = _adapter(regcheck_client="deepseek", regcheck_model="deepseek-chat")
    assert isinstance(a, RegcheckShimAdapter)
    a.prepare()
    assert a._client == "deepseek"
    assert a._registry_model == "deepseek-chat"


def test_deepseek_requires_its_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    a = _adapter(regcheck_client="deepseek")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        a.prepare()


def test_groq_requires_its_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    a = _adapter(regcheck_client="groq")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        a.prepare()


def test_openai_needs_no_provider_key(monkeypatch):
    # openai routes through the codex shim -> no provider key required to prepare().
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    a = _adapter(regcheck_client="openai")
    a.prepare()  # must not raise
    assert a._client == "openai"


def test_three_players_are_distinct_and_load():
    # Sanity: the three registry ids build distinct adapters of the same class.
    ids = {"regcheck-groq": "groq", "regcheck-deepseek": "deepseek", "regcheck-openai": "openai"}
    built = {pid: _adapter(player_id=pid, regcheck_client=c) for pid, c in ids.items()}
    assert all(isinstance(a, RegcheckShimAdapter) for a in built.values())
    assert len({id(a) for a in built.values()}) == 3
    assert {a.player_id for a in built.values()} == set(ids)
