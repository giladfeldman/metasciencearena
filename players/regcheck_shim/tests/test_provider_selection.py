"""Tests for RegcheckShimAdapter provider selection (groq / deepseek / openai).

These cover the 2026-07-02 multi-provider wiring — one adapter class serving three
distinct players via the registry fields `regcheck_client` / `regcheck_model` —
WITHOUT starting regcheck or hitting any network. We stub out the shim server so
prepare() exercises only the client-resolution + key-validation logic.
"""
from __future__ import annotations

from pathlib import Path

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
        "cli_effort": registry_extra.pop("cli_effort", "low"),
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


def test_a_codex_players_effort_never_carries_over_to_the_next(monkeypatch):
    # 2026-11-01: the shim keeps the effort in a module global; a player without one
    # must be refused, not run at the previous player's level.
    from players.regcheck_shim import openai_shim
    monkeypatch.delenv("REGCHECK_SHIM_CODEX_EFFORT", raising=False)
    monkeypatch.setattr(openai_shim, "CODEX_EFFORT", None)
    _adapter(regcheck_client="openai", cli_effort="high").prepare()
    assert openai_shim.CODEX_EFFORT == "high"
    with pytest.raises(RuntimeError, match="needs cli_effort"):
        _adapter(regcheck_client="openai", cli_effort=None).prepare()


def test_three_players_are_distinct_and_load():
    # Sanity: the three registry ids build distinct adapters of the same class.
    ids = {"regcheck-groq": "groq", "regcheck-deepseek": "deepseek", "regcheck-openai": "openai"}
    built = {pid: _adapter(player_id=pid, regcheck_client=c) for pid, c in ids.items()}
    assert all(isinstance(a, RegcheckShimAdapter) for a in built.values())
    assert len({id(a) for a in built.values()}) == 3
    assert {a.player_id for a in built.values()} == set(ids)


# ---- 2026-10-30: pinned RegCheck versions and the Groq-for-GPUStack route ----------

def _clone(tmp_path, monkeypatch, sha):
    import subprocess
    home = tmp_path / "rc_home"
    d = home / f"regcheck-{sha[:7]}"
    (d / "backend").mkdir(parents=True)
    (d / "backend" / "cli.py").write_text("# stub\n", encoding="utf-8")
    real_run = subprocess.run

    def fake_run(argv, *a, **kw):
        if list(argv[:2]) == ["git", "-C"] and "rev-parse" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=sha + "\n", stderr="")
        return real_run(argv, *a, **kw)

    monkeypatch.setattr(regcheck_shim.subprocess, "run", fake_run)
    monkeypatch.setenv("REGCHECK_HOME", str(home))
    return d


def test_a_pinned_commit_resolves_its_own_clone(monkeypatch, tmp_path):
    d = _clone(tmp_path, monkeypatch, "50c80178e8baa3858760cf0d3d04786e23a8602b")
    a = _adapter(regcheck_client="openai", regcheck_commit="50c8017")
    a.prepare()
    assert a._regcheck_dir == d


def test_a_clone_at_another_commit_is_refused(monkeypatch, tmp_path):
    # The label must name the RegCheck that actually ran (no pretending).
    _clone(tmp_path, monkeypatch, "bdb961d0000000000000000000000000000000000")
    (tmp_path / "rc_home" / "regcheck-bdb961d").rename(tmp_path / "rc_home" / "regcheck-50c8017")
    a = _adapter(regcheck_client="openai", regcheck_commit="50c8017")
    with pytest.raises(RuntimeError, match="50c8017"):
        a.prepare()


def test_groq_gpustack_requires_the_groq_key(monkeypatch, tmp_path):
    _clone(tmp_path, monkeypatch, "50c80178e8baa3858760cf0d3d04786e23a8602b")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    a = _adapter(regcheck_client="groq-gpustack", regcheck_commit="50c8017")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        a.prepare()


def test_groq_gpustack_points_regchecks_gpustack_client_at_groq(monkeypatch, tmp_path):
    import json
    import subprocess
    _clone(tmp_path, monkeypatch, "50c80178e8baa3858760cf0d3d04786e23a8602b")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    a = _adapter(regcheck_client="groq-gpustack", regcheck_model="openai/gpt-oss-120b",
                 regcheck_commit="50c8017")
    a.prepare()
    seen = {}

    def fake_cli(argv, *args, **kw):
        if "backend.cli" in argv:
            seen.update(argv=list(argv), env=kw["env"])
            out = argv[argv.index("--output") + 1]
            Path(out).write_text(json.dumps({"items": []}), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(argv)

    monkeypatch.setattr(regcheck_shim.subprocess, "run", fake_cli)
    a.play_task({"arena_id": "prereg-deviation-v1", "input": {
        "preregistration": "p", "paper": "q", "dimensions": ["hypotheses"]}}, timeout_s=30)
    argv, env = seen["argv"], seen["env"]
    assert argv[argv.index("--client") + 1] == "gpustack"
    assert env["GPUSTACK_BASE_URL"] == "https://api.groq.com/openai/v1"
    assert env["GPUSTACK_API_KEY"] == "gsk-test" and "gsk-test" not in argv
    assert env["GPUSTACK_MODEL"] == "openai/gpt-oss-120b"
    assert env["EMBEDDINGS_BASE_URL"].startswith("http://127.0.0.1:")


def test_openai_registry_model_selects_the_codex_model(monkeypatch, tmp_path):
    _clone(tmp_path, monkeypatch, "50c80178e8baa3858760cf0d3d04786e23a8602b")
    monkeypatch.setattr(regcheck_shim.openai_shim, "CODEX_MODEL", "")
    a = _adapter(regcheck_client="openai", regcheck_model="gpt-6-luna", regcheck_commit="50c8017")
    a.prepare()
    assert regcheck_shim.openai_shim.CODEX_MODEL == "gpt-6-luna"


def test_nvidia_gpustack_points_regchecks_gpustack_client_at_nvidia_nim(monkeypatch, tmp_path):
    # 2026-10-30: Groq's free tier caps every chat model at <= 8000 tokens/minute, below one
    # full-paper RegCheck request (8260 measured), so the free route moves to NVIDIA NIM.
    import json
    import subprocess
    _clone(tmp_path, monkeypatch, "50c80178e8baa3858760cf0d3d04786e23a8602b")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-test")
    a = _adapter(regcheck_client="nvidia-gpustack", regcheck_model="openai/gpt-oss-20b",
                 regcheck_commit="50c8017")
    a.prepare()
    seen = {}

    def fake_cli(argv, *args, **kw):
        if "backend.cli" in argv:
            seen.update(argv=list(argv), env=kw["env"])
            out = argv[argv.index("--output") + 1]
            Path(out).write_text(json.dumps({"items": []}), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(argv)

    monkeypatch.setattr(regcheck_shim.subprocess, "run", fake_cli)
    a.play_task({"arena_id": "prereg-deviation-v1", "input": {
        "preregistration": "p", "paper": "q", "dimensions": ["hypotheses"]}}, timeout_s=30)
    argv, env = seen["argv"], seen["env"]
    assert argv[argv.index("--client") + 1] == "gpustack"
    assert env["GPUSTACK_BASE_URL"] == "https://integrate.api.nvidia.com/v1"
    assert env["GPUSTACK_API_KEY"] == "nvapi-test" and "nvapi-test" not in argv
    assert env["GPUSTACK_MODEL"] == "openai/gpt-oss-20b"


def test_nvidia_gpustack_requires_the_nvidia_key(monkeypatch, tmp_path):
    _clone(tmp_path, monkeypatch, "50c80178e8baa3858760cf0d3d04786e23a8602b")
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    a = _adapter(regcheck_client="nvidia-gpustack", regcheck_commit="50c8017")
    with pytest.raises(RuntimeError, match="NVIDIA_NIM_API_KEY"):
        a.prepare()


def _capture_cli(monkeypatch):
    import json
    import subprocess
    seen = {}

    def fake_cli(argv, *args, **kw):
        if "backend.cli" in argv:
            seen.update(argv=list(argv), env=kw["env"])
            out = argv[argv.index("--output") + 1]
            Path(out).write_text(json.dumps({"items": []}), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(argv)

    monkeypatch.setattr(regcheck_shim.subprocess, "run", fake_cli)
    return seen


@pytest.mark.parametrize("client,model", [("openai", "gpt-6-luna"),
                                          ("nvidia-gpustack", "openai/gpt-oss-20b")])
def test_nim_embeddings_replace_the_hashing_stand_in(monkeypatch, tmp_path, client, model):
    # 2026-10-30 smoke: with the shim's hashing pseudo-embeddings RegCheck's retrieval scored
    # 0.00-0.22 relevance, i.e. near-random passages. A player may pin real embeddings.
    _clone(tmp_path, monkeypatch, "50c80178e8baa3858760cf0d3d04786e23a8602b")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-test")
    a = _adapter(regcheck_client=client, regcheck_model=model, regcheck_commit="50c8017",
                 regcheck_embeddings="nvidia-nim")
    a.prepare()
    seen = _capture_cli(monkeypatch)
    a.play_task({"arena_id": "prereg-deviation-v1", "input": {
        "preregistration": "p", "paper": "q", "dimensions": ["hypotheses"]}}, timeout_s=30)
    argv, env = seen["argv"], seen["env"]
    assert env["EMBEDDINGS_BASE_URL"] == "https://integrate.api.nvidia.com/v1"
    assert env["EMBEDDINGS_API_KEY"] == "nvapi-test" and "nvapi-test" not in argv
    assert argv[argv.index("--embedding-model") + 1] == "nvidia/nemotron-3-embed-1b"


def test_nim_embeddings_require_the_nvidia_key(monkeypatch, tmp_path):
    _clone(tmp_path, monkeypatch, "50c80178e8baa3858760cf0d3d04786e23a8602b")
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    a = _adapter(regcheck_client="openai", regcheck_commit="50c8017", regcheck_embeddings="nvidia-nim")
    with pytest.raises(RuntimeError, match="NVIDIA_NIM_API_KEY"):
        a.prepare()


def test_an_unknown_embeddings_route_is_refused(monkeypatch, tmp_path):
    _clone(tmp_path, monkeypatch, "50c80178e8baa3858760cf0d3d04786e23a8602b")
    with pytest.raises(ValueError, match="regcheck_embeddings"):
        _adapter(regcheck_client="openai", regcheck_commit="50c8017", regcheck_embeddings="magic")
