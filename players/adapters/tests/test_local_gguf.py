"""Local GGUF player: the endpoint must be real, and the quant is part of the identity.

Two project rules meet in this adapter and both have already been violated once
elsewhere, so both are pinned here rather than left to review:

1. THE DECLARED ENDPOINT MUST BE THE URL ACTUALLY DIALLED. ``is_cloud_player``
   (framework/runner.py) decides held-out egress from ``entry["endpoint"]``. A key
   the gate trusts and the code ignores is the GROBID-shaped hole documented at
   framework/runner.py:409 -- the gate would licence a held-out run against a
   player that is quietly talking to someone else's machine.

2. THE QUANTIZATION IS PART OF THE PLAYER IDENTITY. Publishing a 3-bit score
   under the model's plain name is the liteparse-2.0.8 defect (``__version__``
   said 2.0.0 while 2.0.8 ran) with higher stakes, because a quant changes the
   NUMBER rather than just the label.

Written before players/adapters/local_gguf.py existed and watched fail.
"""
from __future__ import annotations

import hashlib

import pytest

from framework.player_adapter import build_adapter

# Imported directly, NOT via pytest.importorskip. A skip is a false green: it
# proves nothing, and a file that skips itself when the module is absent would
# stay silently skipped forever if the adapter were renamed or deleted. This
# import must FAIL loudly instead.
from players.adapters import local_gguf
from players.adapters.local_gguf import LocalGgufAdapter


BASE_KWARGS = dict(
    player_id="local-gpt-oss-20b-grim",
    player_version="local-gguf/gpt-oss-20b",
    player_type="ai-model",
    confidence_strategy="native",
    deterministic=False,
)

LOOPBACK = "http://127.0.0.1:8080/v1"


@pytest.fixture
def prompt_file(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("Return JSON only.\n{{INPUT_TEXT}}", encoding="utf-8")
    return p


@pytest.fixture
def gguf(tmp_path):
    """A stand-in weights file. Content is irrelevant; its DIGEST is the point."""
    p = tmp_path / "gpt-oss-20b-Q8_0.gguf"
    p.write_bytes(b"not really a gguf, but it hashes")
    return p


@pytest.fixture(autouse=True)
def _no_os_probe(monkeypatch):
    """Stub the OS process probe by default.

    `prepare()` shells out to PowerShell/lsof to read the server's argv, which is
    ~2s a call and turned this file from 1.8s into 21s. The tests that are ABOUT
    that probe override this with their own monkeypatch, and
    `test_the_real_probe_returns_none_for_a_dead_port` exercises the unstubbed
    function. The full path was additionally verified end-to-end against a live
    llama-server on 2026-08-21, which is what confirmed it captures
    `--n-gpu-layers 99 --threads 8`.
    """
    monkeypatch.setattr(local_gguf, "_server_command_line", lambda port: None)


def _adapter(prompt_file, gguf, **kw):
    defaults = dict(
        openai_model="gpt-oss-20b",
        prompt_template_path=str(prompt_file),
        endpoint=LOOPBACK,
        gguf_path=str(gguf),
    )
    defaults.update(kw)
    return LocalGgufAdapter(**BASE_KWARGS, **defaults)


# --- 1. the endpoint is real, required, and cannot be remote -----------------

def test_endpoint_is_required(prompt_file, gguf):
    """No endpoint means the gate has no evidence, so it must not be constructible."""
    with pytest.raises(TypeError):
        LocalGgufAdapter(
            **BASE_KWARGS,
            openai_model="gpt-oss-20b",
            prompt_template_path=str(prompt_file),
            gguf_path=str(gguf),
        )


@pytest.mark.parametrize("bad", [
    "https://api.example.com/v1",
    "http://127.0.0.1.attacker.com/v1",   # loopback-looking, remote
    "http://evil.com#127.0.0.1",          # loopback-looking, remote
])
def test_non_loopback_endpoint_is_refused(prompt_file, gguf, bad):
    """A class called Local* must not be pointable off-machine.

    The egress gate would already classify these as cloud. This is the SECOND,
    independent guard: Docling's precedent is that the switch which decides
    egress is hard-coded in the adapter so a registry edit cannot flip it.
    """
    with pytest.raises(Exception) as exc:
        _adapter(prompt_file, gguf, endpoint=bad).prepare()
    assert "loopback" in str(exc.value).lower() or "local" in str(exc.value).lower()


def test_declared_endpoint_is_the_url_actually_dialled(prompt_file, gguf, monkeypatch):
    """The gate reads ``endpoint``; this proves the request goes to that same host.

    Also asserts an env var CANNOT redirect it -- the parent class resolves its URL
    from ``openai_base_url_env``, and inheriting that would mean the registry says
    one destination while the environment picks another.
    """
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LOCAL_LLAMA_BASE_URL", "https://api.example.com/v1")

    seen = {}

    class _Resp:
        status_code = 200
        headers: dict = {}

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": '{"label": "ok"}'},
                             "finish_reason": "stop"}],
                # Must match what _props reports, or the mid-run-swap check
                # fires first and this test stops being about the URL.
                "model": str(gguf),
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }

    def fake_post(url, **kw):
        seen["url"] = url
        seen["body"] = kw.get("json")
        return _Resp()

    monkeypatch.setattr(local_gguf.requests, "post", fake_post)

    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    a.play_task({"input": {"text": "x"}}, timeout_s=30)

    assert seen["url"].startswith("http://127.0.0.1:8080"), seen["url"]
    assert "example.com" not in seen["url"]


# --- 2. provenance names the artifact that produced the number ---------------

def test_resolved_tool_version_carries_quant_and_engine_build(prompt_file, gguf, monkeypatch):
    """Not a bare model name: the quant and the engine build both change the score."""
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    v = a.resolved_tool_version()
    assert "Q8_0" in v, f"quant missing from provenance: {v}"
    assert "b10472" in v, f"engine build missing from provenance: {v}"
    assert v != "gpt-oss-20b"


def test_engine_build_is_labelled_unknown_not_omitted(prompt_file, gguf, monkeypatch):
    """A fallback must be visible. Silently dropping the build reads as 'no engine'."""
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf, build=""))
    a.prepare()
    assert "unknown" in a.resolved_tool_version().lower()


def test_response_meta_carries_the_weights_digest(prompt_file, gguf, monkeypatch):
    """sha256 of the actual weights: the only identifier a rebuilt quant cannot fake."""
    expect = hashlib.sha256(gguf.read_bytes()).hexdigest()
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    meta = a.last_response_meta() or {}
    assert meta.get("gguf_sha256") == expect
    assert meta.get("quant") == "Q8_0"
    assert meta.get("engine_build") == "llama.cpp-b10472-7a556b8f9"


def test_digest_is_cached_in_a_sidecar(prompt_file, gguf, monkeypatch):
    """Hashing 17GB on every run is minutes of wall-clock for a value that cannot change."""
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    sidecar = gguf.with_suffix(gguf.suffix + ".sha256")
    assert sidecar.exists(), "no sidecar written"
    assert sidecar.read_text(encoding="utf-8").strip() == hashlib.sha256(gguf.read_bytes()).hexdigest()


@pytest.mark.parametrize("name,stem,quant", [
    ("gpt-oss-20b-Q8_0.gguf", "gpt-oss-20b", "Q8_0"),
    ("Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf", "Qwen3.6-35B-A3B", "UD-Q3_K_XL"),
    ("Qwen3.8-27B-UD-IQ3_S.gguf", "Qwen3.8-27B", "UD-IQ3_S"),
    ("gpt-oss-20b-F16.gguf", "gpt-oss-20b", "F16"),
    ("Qwen3.6-35B-A3B-MXFP4_MOE.gguf", "Qwen3.6-35B-A3B", "MXFP4_MOE"),
])
def test_quant_parsed_from_filename(name, stem, quant):
    assert local_gguf.parse_gguf_name(name) == (stem, quant)


def test_unparseable_quant_refuses_rather_than_guessing(prompt_file, tmp_path, monkeypatch):
    """An unlabelled quant would publish a number nobody can attribute to an artifact."""
    odd = tmp_path / "mystery-weights.gguf"
    odd.write_bytes(b"x")
    a = _adapter(prompt_file, odd)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(odd))
    with pytest.raises(RuntimeError, match="gguf_quant"):
        a.prepare()


def test_gguf_quant_is_a_fallback_for_an_unparseable_name(prompt_file, tmp_path, monkeypatch):
    """A fallback for a file whose name carries no quant token -- never an
    override of a name that does (pinned in section 5)."""
    odd = tmp_path / "mystery-weights.gguf"
    odd.write_bytes(b"x")
    a = _adapter(prompt_file, odd, gguf_quant="Q4_K_M")
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(odd))
    a.prepare()
    assert "Q4_K_M" in a.resolved_tool_version()


# --- 3. the registry can actually build it ----------------------------------

def test_build_adapter_passes_every_registry_key_through(prompt_file, gguf):
    """ADAPTER_EXTRA_KWARGS gatekeeps registry keys; a dropped key runs the wrong config.

    This is the liteparse-no-ocr defect: ``ocr_enabled`` was declared in registry.yaml,
    silently dropped by build_adapter, and ``liteparse-no-ocr`` ran WITH OCR and
    published a score bit-identical to the OCR-on player.
    """
    entry = dict(
        BASE_KWARGS,
        adapter_class="LocalGgufAdapter",
        openai_model="gpt-oss-20b",
        prompt_template_path=str(prompt_file),
        endpoint=LOOPBACK,
        gguf_path=str(gguf),
        gguf_quant="Q8_0",
        openai_temperature=0.7,
    )
    a = build_adapter(entry)
    assert isinstance(a, LocalGgufAdapter)
    assert a.endpoint == LOOPBACK
    assert str(a.gguf_path) == str(gguf)
    assert a.openai_temperature == 0.7


# --- 4. paths come from the environment, never hardcoded ---------------------

def test_gguf_path_expands_env_vars_and_user(prompt_file, gguf, monkeypatch):
    """`${SCIENCEARENA_GGUF_DIR}/x.gguf` must resolve; the repo forbids absolute
    user paths in config (CLAUDE.md: never hardcode the Vibe root)."""
    monkeypatch.setenv("SCIENCEARENA_GGUF_DIR", str(gguf.parent))
    a = _adapter(prompt_file, "${SCIENCEARENA_GGUF_DIR}/" + gguf.name)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    assert a.gguf_path == gguf
    assert "Q8_0" in a.resolved_tool_version()


def test_unset_path_variable_fails_by_name_rather_than_resolving_wrong(prompt_file, monkeypatch):
    monkeypatch.delenv("SCIENCEARENA_GGUF_DIR", raising=False)
    a = _adapter(prompt_file, "${SCIENCEARENA_GGUF_DIR}/gpt-oss-20b-Q8_0.gguf")
    with pytest.raises(RuntimeError, match="SCIENCEARENA_GGUF_DIR"):
        a.prepare()


# --- 5. the SERVER decides what was run, not the registry --------------------
#
# Codex cross-review, 2026-08-20. The adapter hashed the `gguf_path` the registry
# named and asked the server nothing, so three registry mistakes all published a
# confident, wrong artifact label with every test green:
#
#   * `gguf_quant: Q4_K_M` beside a Q8_0 file -> Q8 weights labelled Q4
#   * `openai_model: gpt-oss-20b` beside some other GGUF -> wrong model name
#   * a server left running on a DIFFERENT GGUF (exactly what happens between
#     benchmark configs) -> the local file's digest published for another file's
#     outputs
#
# llama-server's /props reports `model_path`, `model_ftype` and `build_info`, so
# the served artifact is checkable rather than assumed. Reproduced before fixing.

def _props(model_path, ftype="Q8_0", build="b10472-7a556b8f9"):
    """The shape llama-server's /props really returns, captured from a live server
    on 2026-08-20 rather than invented. `model_alias` defaults to the full path
    when --alias is not passed, and is the string the completion response echoes."""
    return {
        "model_path": str(model_path),
        "model_alias": str(model_path),
        "model_ftype": ftype,
        "build_info": build,
        "default_generation_settings": {"params": {"temperature": 0.8, "seed": 4294967295}},
    }


def test_refuses_when_the_server_serves_a_different_file(prompt_file, gguf, tmp_path, monkeypatch):
    """The defect Codex found: hash one file, score another."""
    other = tmp_path / "gpt-oss-20b-Q4_K_M.gguf"
    other.write_bytes(b"a different quantization entirely")
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(other, ftype="Q4_K_M"))
    with pytest.raises(RuntimeError) as exc:
        a.prepare()
    msg = str(exc.value)
    assert gguf.name in msg and other.name in msg, \
        "the refusal must name BOTH files, or nobody can tell which one to fix"


def test_quant_label_comes_from_the_served_file_not_the_registry(prompt_file, gguf, monkeypatch):
    """`gguf_quant` is a FALLBACK for an unparseable name, never an override.

    An override lets a registry edit relabel a score, which is the whole failure
    this adapter exists to prevent.
    """
    a = _adapter(prompt_file, gguf, gguf_quant="Q4_K_M")
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    v = a.resolved_tool_version()
    assert "Q8_0" in v and "Q4_K_M" not in v, f"registry relabelled the quant: {v}"


def test_refuses_when_props_is_unreachable(prompt_file, gguf, monkeypatch):
    """Unverifiable provenance is refused, not guessed.

    Loud failure beats a confident label nobody can check.
    """
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: None)
    with pytest.raises(RuntimeError, match="/props"):
        a.prepare()


def test_server_reported_ftype_is_recorded_as_independent_evidence(prompt_file, gguf, monkeypatch):
    """model_ftype comes from the GGUF's own metadata, not from its filename."""
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    meta = a.last_response_meta() or {}
    assert meta.get("model_ftype") == "Q8_0"
    # The BASENAME of the served file, never its absolute path: this block is
    # published, and an absolute path here would ship the operator's username.
    assert meta.get("gguf_file") == gguf.name
    assert not any(isinstance(v, str) and (":\\" in v or v.startswith("/"))
                   for k, v in meta.items() if k != "endpoint"), \
        f"an absolute local path reached the published record: {meta}"


def test_a_model_swap_mid_run_is_caught_on_the_very_next_task(prompt_file, gguf, monkeypatch):
    """Sonnet 5 cross-review, 2026-08-20.

    prepare() verifies the served artifact ONCE; a tournament then runs for hours.
    Restart llama-server onto a different GGUF partway through -- a crash-restart,
    or the config-swapping done while benchmarking -- and every later task is
    answered by the new weights but published under the first probe's stem, quant
    and sha256. Nothing throws.

    Measured on a live server 2026-08-20: llama-server IGNORES the requested
    `model` and echoes what it actually loaded, so the check costs no extra call.
    """
    class _Resp:
        status_code = 200
        headers: dict = {}

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"label": "ok"}'},
                                 "finish_reason": "stop"}],
                    "model": "C:/somewhere/else/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1}}

    monkeypatch.setattr(local_gguf.requests, "post", lambda url, **kw: _Resp())
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    with pytest.raises(RuntimeError, match="changed model mid-run"):
        a.play_task({"input": {"text": "x"}}, timeout_s=30)


def test_matching_served_model_does_not_trip_the_swap_check(prompt_file, gguf, monkeypatch):
    """Guard against the opposite failure: a check that fires on every task would
    make the player unusable and would be 'fixed' by deleting it."""
    class _Resp:
        status_code = 200
        headers: dict = {}

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"label": "ok"}'},
                                 "finish_reason": "stop"}],
                    "model": str(gguf),
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1}}

    monkeypatch.setattr(local_gguf.requests, "post", lambda url, **kw: _Resp())
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    assert a.play_task({"input": {"text": "x"}}, timeout_s=30) == {"label": "ok"}


def test_server_sampling_defaults_are_recorded(prompt_file, gguf, monkeypatch):
    """llama-server applies temperature 0.8 and a random seed to anything we do not
    send. A reproducible anchor whose sampling nobody wrote down is not one -- and
    this is what explains two runs of one prompt differing by 359 output tokens."""
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    meta = a.last_response_meta() or {}
    assert (meta.get("server_defaults") or {}).get("temperature") == 0.8


# --- 6. the MACHINE is part of a local player's identity ---------------------
#
# Added 2026-08-21, before a second machine ran a single task.
#
# `players/registry.yaml` records that gpt-oss-20b reproduced 16/16 byte-identical
# completions up to ~2400 tokens, across days, sessions and server processes. That
# bounded determinism is the entire reason this player class exists -- it is the
# project's only drift anchor, and no hosted player has it.
#
# It is a property of ONE CPU and ONE llama.cpp backend. A different SIMD path or
# a different (or absent) iGPU changes the reduction order, one logit flips, and
# the continuation diverges -- same weights, same seed, same temperature.
#
# So the host belongs in `resolved_tool_version` for the same reason the quant
# does: it changes the NUMBER, not just the label.

def test_resolved_tool_version_carries_the_host_fingerprint(prompt_file, gguf, monkeypatch):
    from framework import host

    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    assert f"@host-{host.host_identity()['fingerprint']}" in a.resolved_tool_version()


def test_two_machines_resolve_to_two_different_tool_versions(prompt_file, gguf, monkeypatch):
    """THE regression. Identical weights + identical engine, different machine."""
    from framework import host

    def _version_on(facts):
        monkeypatch.setattr(host, "collect_facts", lambda: facts)
        a = _adapter(prompt_file, gguf)
        monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
        a.prepare()
        return a.resolved_tool_version()

    laptop = _version_on({"cpu": "Intel64 Family 6 Model 170 Stepping 4, GenuineIntel",
                          "arch": "AMD64", "os": "Windows", "os_release": "11",
                          "cpu_logical": 14, "ram_gb": 32})
    desktop = _version_on({"cpu": "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel",
                           "arch": "AMD64", "os": "Windows", "os_release": "11",
                           "cpu_logical": 12, "ram_gb": 32})
    assert laptop != desktop
    # The weights and engine halves must be unchanged -- only the machine moved.
    assert laptop.split("@host-")[0] == desktop.split("@host-")[0]


# --- 7. the server's RUNTIME config, which /props does not report -------------
#
# Measured 2026-08-21 against llama.cpp b10472: GET /props returns build_info,
# model_path, model_ftype, model_alias, total_slots, and a
# default_generation_settings holding only n_ctx and sampling params. It reports
# NOTHING about GPU offload, threads, or the compute backend.
#
# That is the single biggest lever on this hardware. Measured on the laptop the
# same week: -ngl 99 (Vulkan iGPU) gives 8.1-8.5 tok/s decode against 4.7-5.5 for
# -ngl 0, and those two configurations run different kernels -- so they are two
# instruments, and every published record so far was silent about which one ran.
#
# The server's own command line is the only place it is observable, so it is
# captured from the OS. It contains C:/Users/<name>/..., so it is redacted.

def test_server_runtime_records_ctx_and_slots(prompt_file, gguf, monkeypatch):
    a = _adapter(prompt_file, gguf)
    props = _props(gguf)
    props["total_slots"] = 4
    props["default_generation_settings"]["n_ctx"] = 8192
    monkeypatch.setattr(a, "_probe_server_props", lambda: props)
    a.prepare()
    meta = a.last_response_meta() or {}
    # Both feed llama.cpp's slot selection, which registry.yaml names as the
    # mechanism that breaks byte-for-byte reproduction past ~2400 tokens.
    assert meta["server_n_ctx"] == 8192
    assert meta["server_total_slots"] == 4


def test_server_argv_is_captured_and_redacted(prompt_file, gguf, monkeypatch):
    raw = (r"C:\Users\testuser\.unsloth\llama.cpp\build\bin\Release\llama-server.exe "
           r"--model C:/Users/testuser/.sciencearena/models/gpt-oss-20b-Q8_0.gguf "
           r"--n-gpu-layers 99 --threads 8 --ctx-size 8192")
    monkeypatch.setattr(local_gguf, "_server_command_line", lambda port: raw)
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    meta = a.last_response_meta() or {}
    assert "--n-gpu-layers 99" in meta["server_argv"]
    assert "--threads 8" in meta["server_argv"]
    assert "testuser" not in meta["server_argv"]
    assert meta["server_argv_source"] == "os-process"


def test_absent_server_argv_is_labelled_not_omitted(prompt_file, gguf, monkeypatch):
    """A missing field reads as 'not applicable'. This one means 'we could not look'.

    Same rule the engine label already follows: an unknown build is LABELLED
    rather than dropped, because a missing engine reads as "no engine".
    """
    monkeypatch.setattr(local_gguf, "_server_command_line", lambda port: None)
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    meta = a.last_response_meta() or {}
    assert meta["server_argv"] is None
    assert meta["server_argv_source"] == "unavailable"


def test_argv_capture_failure_never_fails_prepare(prompt_file, gguf, monkeypatch):
    """Telemetry must never take down a run -- the rule the version probe follows."""
    def _boom(port):
        raise OSError("no such process")
    monkeypatch.setattr(local_gguf, "_server_command_line", _boom)
    a = _adapter(prompt_file, gguf)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()  # must not raise
    assert (a.last_response_meta() or {})["server_argv_source"] == "unavailable"


def test_the_real_probe_returns_none_for_a_dead_port(monkeypatch):
    """The unstubbed function, against a port nothing is listening on.

    Pins the contract the caller depends on: no exception, no invented value —
    `None`, which `prepare()` turns into an explicit "unavailable" label.
    """
    monkeypatch.undo()
    assert local_gguf._server_command_line(1) is None


def test_argv_from_a_process_that_is_not_llama_server_is_refused(monkeypatch):
    """Codex #6/#7 (Medium), 2026-08-21: the argv could name the wrong process.

    Under SO_REUSEPORT two processes can listen on one port, and a server that
    exits between the /props probe and this one frees the port for anything else.
    Publishing a stranger's command line as this player's offload configuration
    is worse than publishing nothing: a missing field is recoverable, a
    confidently wrong one is the class of defect this change exists to prevent.
    """
    monkeypatch.undo()
    monkeypatch.setattr(local_gguf.subprocess, "run", lambda *a, **k: type(
        "R", (), {"stdout": "C:/other/nginx.exe --port 8080", "returncode": 0})())
    assert local_gguf._server_command_line(8080) is None


# --- the ceiling this adapter installs is the one the record reports ---------
# `provenance.request_max_tokens` (run-record schema v9) is read off the FINISHED
# request body in the parent class, which is what makes this adapter a real test of
# it rather than a second copy of the same logic: LocalGgufAdapter does not send
# the parent's hosted default of 32000. It writes its own, much lower ceiling into
# `openai_extra_body` and then delegates upward — because 32000 tokens on this
# hardware is a multi-hour runaway. A runner that re-read the registry instead
# would report a ceiling this adapter never sends.

@pytest.mark.parametrize("extra_body,expected", [
    (None, local_gguf._LOCAL_MAX_TOKENS),
    ({"max_tokens": 1234}, 1234),
])
def test_the_recorded_ceiling_is_the_one_installed_not_the_parents(
        prompt_file, gguf, monkeypatch, extra_body, expected):
    class _Resp:
        status_code = 200
        headers: dict = {}

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": '{"label": "ok"}'},
                             "finish_reason": "stop"}],
                "model": str(gguf),
            }

    seen = {}

    def fake_post(url, **kw):
        seen["body"] = kw.get("json")
        return _Resp()

    monkeypatch.setattr(local_gguf.requests, "post", fake_post)
    kw = {"openai_extra_body": extra_body} if extra_body is not None else {}
    a = _adapter(prompt_file, gguf, **kw)
    monkeypatch.setattr(a, "_probe_server_props", lambda: _props(gguf))
    a.prepare()
    a.play_task({"input": {"text": "x"}}, timeout_s=30)

    # Two-sided: what went on the wire, and what the adapter reports for the record.
    assert seen["body"]["max_tokens"] == expected, \
        f"the wire carried {seen['body']['max_tokens']}, not {expected}"
    assert a.last_request_max_tokens() == expected
    # The control that makes this test about THIS adapter: the parent's hosted
    # default must not be what we saw.
    from players.adapters.openai_compatible import _DEFAULT_MAX_TOKENS
    assert expected != _DEFAULT_MAX_TOKENS
