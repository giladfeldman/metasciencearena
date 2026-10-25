"""Schema v12: run environment, call configuration, raw responses, the runner gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from framework import container, run_env
from framework.storage import RunRecordValidationError, RunRecordWriter, read_records
from framework.tests._container_fakes import in_process_runner  # noqa: F401  (fixture)


# ------------------------------------------------------------------ hashing
def test_canonical_json_is_sorted_compact_utf8_and_int_floats():
    assert run_env.canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    # 0.0 and 0 must hash alike, or Python (0.0) and TypeScript (0) disagree
    assert run_env.canonical_sha256({"t": 0.0}) == run_env.canonical_sha256({"t": 0})
    assert run_env.canonical_json({"t": 0.7}) == '{"t":0.7}'
    with pytest.raises(ValueError):
        run_env.canonical_json({"t": float("nan")})


def test_canonical_hash_matches_the_citeclaim_library_fixture():
    """Same config, same bytes, same hash as citeclaim's canonicalSha256
    (citeclaim tests/fixtures/python_canonical.json, generated 2026-10-14)."""
    cfg = {"provider": "google", "model_requested": "gemini-flash-latest", "temperature": 0, "seed": 7,
           "top_k": "n/a", "stop": [], "nested": {"b": 1, "a": "é—x"}, "flag": True, "none": None}
    assert run_env.canonical_json(cfg) == (
        '{"flag":true,"model_requested":"gemini-flash-latest","nested":{"a":"é—x","b":1},'
        '"none":null,"provider":"google","seed":7,"stop":[],"temperature":0,"top_k":"n/a"}')


# ------------------------------------------------------------------ call_config
def test_openai_call_config_maps_every_knob_and_keeps_unknown_keys(monkeypatch):
    monkeypatch.setenv("X_BASE", "https://api.example.org/v1/")
    entry = {"adapter_class": "OpenAIChatCompletionsAdapter", "openai_model": "m",
             "openai_base_url_env": "X_BASE", "openai_api_key_env": "GROQ_API_KEY",
             "prompt_format": "str.format"}
    body = {"model": "m", "temperature": 0.0, "max_tokens": 4096, "seed": 7, "reasoning_effort": "low",
            "response_format": {"type": "json_object"}, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}, "mystery_knob": 3, "messages_sha256": "x"}
    cfg = run_env.build_call_config(entry=entry, body=body, prompt_template_sha256="abc",
                                    timeout_s=120, group_size=3)
    assert set(run_env.CALL_CONFIG_FIELDS) <= set(cfg)
    assert cfg["provider"] == "groq" and cfg["endpoint"] == "https://api.example.org/v1"
    assert cfg["api_type"] == "openai-chat-completions" and cfg["model_requested"] == "m"
    assert cfg["temperature"] == 0 and cfg["seed"] == 7 and cfg["max_tokens"] == 4096
    assert cfg["reasoning_effort"] == "low" and cfg["enable_thinking"] is False
    assert cfg["group_size"] == 3 and cfg["prompt_template_sha256"] == "abc"
    assert cfg["top_p"] == "n/a" and cfg["thinking_budget"] == "n/a"   # absent -> n/a, never omitted
    assert cfg["other_body_keys"] == {"mystery_knob": 3}               # unknown -> kept, never dropped


def test_gemini_call_config_reads_generation_config():
    entry = {"adapter_class": "GeminiGenerateContentAdapter", "gemini_model": "gemini-flash-latest"}
    body = {"model": "gemini-flash-latest", "api": "gemini-generate-content", "contents_sha256": "x",
            "generationConfig": {"temperature": 0, "seed": 7, "maxOutputTokens": 8192,
                                 "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": "low"}}}
    cfg = run_env.build_call_config(entry=entry, body=body, prompt_template_sha256=None, timeout_s=60)
    assert cfg["api_type"] == "gemini-generate-content" and cfg["provider"] == "google"
    assert cfg["model_requested"] == "gemini-flash-latest" and cfg["reasoning_effort"] == "low"
    assert cfg["max_tokens"] == 8192 and cfg["response_format"] == "application/json"
    assert cfg["other_body_keys"] == "n/a"


# ------------------------------------------------------------------ stores
def test_stores_are_content_addressed_and_write_once(tmp_path):
    env = run_env.build_env(isolation="host-exempt", runner=None, egress=None,
                            network={"mode": "host"}, resources=None, generator_fingerprint=None,
                            exemption="StubPassAdapter: test")
    sha = run_env.store_env(env, root=tmp_path)
    assert run_env.store_env(env, root=tmp_path) == sha
    assert run_env.load_env(sha, root=tmp_path)["isolation"] == "host-exempt"
    p = tmp_path / "envs" / f"{sha}.json"
    p.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="write-once"):
        run_env.store_env(env, root=tmp_path)


def test_host_exempt_env_must_name_its_exemption():
    with pytest.raises(ValueError):
        run_env.build_env(isolation="host-exempt", runner=None, egress=None, network={},
                          resources=None, generator_fingerprint=None)


# ------------------------------------------------------------------ the writer gate
def _minimal(schema_version: int, env_sha) -> dict:
    rec = {"run_id": "r", "arena_id": "a", "task_set_version": "v1", "task_id": "t", "player_id": "p",
           "player_version": "1", "player_type": "tool", "input_hash": "h", "output": {},
           "score": {"primary": 0.0}, "timestamp_utc": "2026-10-14T00:00:00Z", "schema_version": schema_version,
           "instrument": {"formula_id": "f", "scorer_sha256": None, "gold_sha256": None, "output_sha256": None},
           "player_identity": {"instrument": None, "inference_config": None, "inference_config_sha256": None}}
    if schema_version >= 12:
        rec["env_sha"] = env_sha
        rec["call_config_sha256"] = None
    return rec


def test_writer_rejects_a_v12_record_without_env_sha(tmp_path, monkeypatch):
    env = run_env.build_env(isolation="host-exempt", runner=None, egress=None, network={"mode": "host"},
                            resources=None, generator_fingerprint=None, exemption="StubPassAdapter: test")
    sha = run_env.store_env(env)  # into the per-test store (conftest)
    with RunRecordWriter(tmp_path / "r.jsonl") as w:
        with pytest.raises(RunRecordValidationError):
            w.append(_minimal(12, None))
        w.append(_minimal(12, sha))
        w.append(_minimal(11, None))        # history predates the field
    assert len(read_records(tmp_path / "r.jsonl")) == 2


# ------------------------------------------------------------------ placement
def test_every_registered_adapter_class_is_placed_or_named_exempt():
    """A new adapter class must be either containerised or a NAMED exemption.
    Unclassified, it cannot play (placement raises) - this test says so first."""
    from framework.player_adapter import _ADAPTER_CLASSES
    from framework.runner import _autoload_adapter_modules
    _autoload_adapter_modules()
    unclassified = sorted(c for c in _ADAPTER_CLASSES
                          if c not in container.CONTAINER_ADAPTER_CLASSES
                          and c not in container.EXEMPT_ADAPTER_CLASSES)
    assert unclassified == [], f"classify in framework/container.py: {unclassified}"


def test_placement_rules():
    assert container.placement({"adapter_class": "OpenAIChatCompletionsAdapter"}) == "container"
    assert container.placement({"adapter_class": "OpenAIChatCompletionsAdapter",
                                "document_to_text": "pdftotext"}) == "host-exempt"
    assert container.placement({"adapter_class": "RCliAdapter"}) == "host-exempt"
    with pytest.raises(container.RunnerUnavailable):
        container.placement({"adapter_class": "BrandNewAdapter"})


def test_loopback_endpoint_is_never_sent_to_a_container(monkeypatch):
    monkeypatch.setenv("LOCAL_BASE", "http://127.0.0.1:8080/v1")
    with pytest.raises(container.RunnerUnavailable, match="loopback"):
        container.provider_host({"player_id": "x", "adapter_class": "OpenAIChatCompletionsAdapter",
                                 "openai_base_url_env": "LOCAL_BASE"})


def test_image_holds_code_only_never_task_data():
    staged = [dst for _, dst in container._staged_files()]
    assert any(d == "framework/container_play.py" for d in staged)
    assert any(d.startswith("arenas/citation-claim-fidelity-v1/prompts/") for d in staged)
    for d in staged:
        assert "/task_sets/" not in d and "_held_out" not in d and "arenas_private" not in d
        assert not d.endswith(".jsonl") and ".private_seed" not in d
        assert "/tests/" not in d


# ------------------------------------------------------------------ end to end
GRIM = Path("arenas/grim-consistency-v1")


@pytest.fixture
def fake_http(monkeypatch, in_process_runner):
    monkeypatch.setenv("V12_BASE_URL", "https://api.invalid/v1")
    monkeypatch.setenv("V12_API_KEY", "k")

    class R:
        status_code = 200
        headers: dict = {}
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({"records": []})},
                                 "finish_reason": "stop"}], "model": "fake-v12"}

    monkeypatch.setattr("players.adapters.openai_compatible.requests.post",
                        lambda url, headers=None, json=None, timeout=None: R())
    return in_process_runner


def _entry() -> dict:
    return {"player_id": "v12-probe", "player_type": "ai-model", "player_version": "fake-v12",
            "adapter_class": "OpenAIChatCompletionsAdapter", "openai_model": "fake-v12",
            "openai_base_url_env": "V12_BASE_URL", "openai_api_key_env": "V12_API_KEY",
            "prompt_template_path": "players/prompts/grim_consistency.txt",
            "openai_extra_body": {"seed": 7, "max_tokens": 1000},
            "confidence_strategy": "native", "deterministic": True}


def test_tournament_records_env_call_config_and_raw(tmp_path, fake_http, monkeypatch):
    from framework.runner import run_tournament
    reg = tmp_path / "registry.yaml"
    reg.write_text(yaml.safe_dump([_entry()]), encoding="utf-8")
    out = tmp_path / "run.jsonl"
    run_tournament(arena_dir=GRIM, task_set_version="v1", registry_path=reg, player_ids=["v12-probe"],
                   output_path=out, trials=1, timeout_s=60, seed=0, split="revealed", max_tasks=2,
                   public_only=True)
    recs = read_records(out)
    assert len(recs) == 2 and len(fake_http) == 2        # both tasks went through the container path
    for r in recs:
        assert r["schema_version"] == 12
        env = run_env.load_env(r["env_sha"])
        assert env["isolation"] == "container"
        assert env["network"]["allow_hosts"] == ["api.invalid"]
        cfg = run_env.load_call_config(r["call_config_sha256"])
        assert cfg["model_requested"] == "fake-v12" and cfg["seed"] == 7 and cfg["max_tokens"] == 1000
    raw = run_env.read_raw(tmp_path / "run.raw.jsonl.gz")
    assert [x["run_id"] for x in raw] == [r["run_id"] for r in recs]
    assert raw[0]["raw_response"]["model"] == "fake-v12"


def test_container_required_arena_refuses_a_host_exempt_player(tmp_path, monkeypatch):
    from framework import runner
    real = runner.load_arena

    def with_policy(d):
        a = real(d)
        a["manifest"] = dict(a["manifest"], runner="container-required")
        return a

    monkeypatch.setattr(runner, "load_arena", with_policy)
    reg = tmp_path / "registry.yaml"
    reg.write_text(yaml.safe_dump([{"player_id": "stub", "player_type": "tool", "player_version": "1",
                                    "adapter_class": "StubPassAdapter", "confidence_strategy": "native",
                                    "deterministic": True}]), encoding="utf-8")
    with pytest.raises(container.RunnerUnavailable, match="container-required"):
        runner.run_tournament(arena_dir=GRIM, task_set_version="v1", registry_path=reg, player_ids=["stub"],
                              output_path=tmp_path / "o.jsonl", trials=1, timeout_s=10, seed=0,
                              split="revealed", max_tasks=1, public_only=True)


def test_held_out_raw_goes_to_custody_never_the_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(tmp_path / "repo"))
    run_file = Path("arenas/x/runs/v1/p.jsonl")
    pub = run_env.raw_sidecar_path(run_file, held_out=False, arena_id="x")
    held = run_env.raw_sidecar_path(run_file, held_out=True, arena_id="x")
    assert pub == Path("arenas/x/runs/v1/p.raw.jsonl.gz")
    assert held == tmp_path / "repo" / "ground_truth" / "sciencearena-raw" / "x" / "p.raw.jsonl.gz"
    monkeypatch.delenv("ARTICLE_REPOSITORY")
    with pytest.raises(RuntimeError, match="custody"):
        run_env.raw_sidecar_path(run_file, held_out=True, arena_id="x")


# ------------------------------------------------------------------ consult 2026-10-14 regressions
# Each failed against commit 09f37b2 (the first v12 version) before its fix.
def test_canonical_numbers_are_written_the_javascript_way():
    """Python writes 1e-07 and 1e-05, JavaScript 1e-7 and 0.00001 (Sol, Sonnet)."""
    assert run_env.canonical_json({"t": 1e-7}) == '{"t":1e-7}'
    assert run_env.canonical_json({"t": 1e-5}) == '{"t":0.00001}'
    assert run_env.canonical_json({"t": 0.7}) == '{"t":0.7}'
    assert run_env.canonical_json({"t": 123456.789}) == '{"t":123456.789}'


def test_canonical_refuses_integers_javascript_cannot_hold():
    with pytest.raises(ValueError):
        run_env.canonical_json({"seed": 9007199254740993})
    with pytest.raises(ValueError):
        run_env.canonical_json({"t": 1.5e21})  # a whole float JavaScript also refuses


def test_canonical_keys_sort_by_utf16_code_unit_like_javascript():
    assert run_env.canonical_json({"": 1, "\U00010000": 2}) == '{"\U00010000":2,"":1}'


def test_unmapped_body_text_never_reaches_the_tracked_store():
    """A body key carrying task text is stored as a hash, never verbatim (Sol, Sonnet L2)."""
    entry = {"adapter_class": "OpenAIChatCompletionsAdapter", "openai_model": "m"}
    secret = "An abstract sentence that must not be stored. " * 5
    cfg = run_env.build_call_config(entry=entry, body={"model": "m", "prompt": secret, "flag": True},
                                    prompt_template_sha256=None, timeout_s=10)
    assert secret not in run_env.canonical_json(cfg)
    assert cfg["other_body_keys"]["flag"] is True
    assert set(cfg["other_body_keys"]["prompt"]) == {"sha256", "chars"}


def test_endpoint_keeps_no_credentials_or_query(monkeypatch):
    monkeypatch.setenv("CRED_BASE", "https://user:tok@api.example.org/v1?key=SECRET#x")
    entry = {"adapter_class": "OpenAIChatCompletionsAdapter", "openai_model": "m", "openai_base_url_env": "CRED_BASE"}
    cfg = run_env.build_call_config(entry=entry, body={"model": "m"}, prompt_template_sha256=None, timeout_s=1)
    assert cfg["endpoint"] == "https://api.example.org/v1"


def test_max_completion_tokens_is_not_folded_into_max_tokens():
    entry = {"adapter_class": "OpenAIChatCompletionsAdapter", "openai_model": "m"}
    cfg = run_env.build_call_config(entry=entry, body={"max_tokens": 10, "max_completion_tokens": 20},
                                    prompt_template_sha256=None, timeout_s=1)
    assert cfg["max_tokens"] == 10 and cfg["max_completion_tokens"] == 20


def test_held_out_raw_refuses_a_custody_root_inside_the_repository(monkeypatch):
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(run_env.REPO_ROOT / "tmp" / "fake-custody"))
    with pytest.raises(RuntimeError, match="inside"):
        run_env.raw_sidecar_path(Path("arenas/x/runs/v1/p.jsonl"), held_out=True, arena_id="x")


def test_read_raw_survives_a_torn_final_record(tmp_path):
    p = tmp_path / "r.raw.jsonl.gz"
    run_env.append_raw(p, {"run_id": "a"})
    run_env.append_raw(p, {"run_id": "b"})
    data = p.read_bytes()
    p.write_bytes(data[:-7])  # crash mid-write of the last member
    assert [r["run_id"] for r in run_env.read_raw(p)] == ["a"]


def test_writer_refuses_an_env_sha_with_no_stored_env(tmp_path):
    rec = _minimal(12, "b" * 64)  # well-formed hash, but no run_env/envs/<sha>.json
    with RunRecordWriter(tmp_path / "r.jsonl") as w:
        with pytest.raises(RunRecordValidationError, match="names no stored env"):
            w.append(rec)
