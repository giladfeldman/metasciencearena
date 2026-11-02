"""Offline tests for the Anthropic API player and its spend guard.

No test touches the network: `requests.post` / `requests.get` are replaced, and
every test asserts what WOULD have been sent. The budget tests are the point --
the owner's condition for the API exception (2026-10-23) is that nothing is
spent beyond the credit, so each refusal path is checked to send nothing. The
exception closed on 2026-10-29: `read_key` now refuses unless the reopen flag
is set, and the closure tests check that refusal and that replay still works.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
# The public mirror (contract/public_manifest.txt) leaves this out, so the test runs only here.
_needs = lambda name: pytest.mark.skipif(  # noqa: E731
    not (_SCRIPTS / name).exists(), reason=f"scripts/{name} is private tooling, not in the public mirror")

from players.adapters import _anthropic_budget as budget
from players.adapters import anthropic_api as api

PROMPT = "Task follows.\n{{INPUT_TEXT}}\nAnswer in JSON."
PDF_PROMPT = "Read {{PDF_PATH}} ({{N_PAGES}} pages). Answer in JSON."


class FakeResponse:
    def __init__(self, status: int, body: dict | None = None, text: str | None = None):
        self.status_code = status
        self._body = body or {}
        self.text = text if text is not None else json.dumps(self._body)
        self.content = self.text.encode("utf-8")

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeApi:
    """Records every call; answers count_tokens and messages from fixtures."""

    def __init__(self, *, input_tokens=1000, message=None, status=200):
        self.calls: list[tuple[str, dict]] = []
        self.input_tokens = input_tokens
        self.status = status
        self.message = message or {
            "id": "msg_test", "model": "claude-sonnet-5-5", "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "not the answer"},
                        {"type": "text", "text": '{"verdict": "ok"}'}],
            "usage": {"input_tokens": 1000, "output_tokens": 200},
        }

    def post(self, url, headers=None, json=None, timeout=None):  # noqa: A002
        self.calls.append((url, json))
        assert headers["x-api-key"] == "test-key"
        if url.endswith("/count_tokens"):
            return FakeResponse(200, {"input_tokens": self.input_tokens})
        if url.endswith("/v1/messages"):
            if self.status != 200:
                return FakeResponse(self.status, {"error": {"type": "invalid_request_error"}})
            return FakeResponse(200, self.message)
        raise AssertionError(f"unexpected POST {url}")

    def paid_calls(self) -> int:
        return sum(1 for url, _ in self.calls if url.endswith("/v1/messages"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(PROMPT, encoding="utf-8")
    monkeypatch.setenv(api.KEY_ENV, "test-key")
    # The mechanics below are tested as if the owner had reopened the exception; the closure
    # itself is tested at the end of this file.
    monkeypatch.setenv("SCIENCEARENA_ANTHROPIC_REOPENED", "test")
    monkeypatch.setenv(budget.BUDGET_ENV, "5")
    monkeypatch.setenv(budget.PERIOD_ENV, "test-period")
    monkeypatch.setenv(budget.LEDGER_ENV, str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv(api.BATCH_DIR_ENV, str(tmp_path / "batches"))
    return tmp_path


def make(prompt_path: Path, **kw) -> api.AnthropicApiAdapter:
    args = dict(player_id="anthropic-api-sonnet-5-5-high-test", player_version="anthropic-api/claude-sonnet-5-5",
                player_type="ai-model", confidence_strategy="native", deterministic=False,
                anthropic_model="claude-sonnet-5-5", anthropic_effort="high",
                anthropic_max_tokens=16000, prompt_template_path=prompt_path)
    args.update(kw)
    return api.AnthropicApiAdapter(**args)


ENV = {"task_id": "t-001", "input": {"text": "t(35) = 2.1, p = .04"}}


# -- refusals that must spend nothing -------------------------------------------
def test_mode_unset_refuses(env, monkeypatch):
    monkeypatch.delenv(api.MODE_ENV, raising=False)
    with pytest.raises(RuntimeError, match="SCIENCEARENA_ANTHROPIC_MODE"):
        make(env / "prompt.txt").prepare()


@pytest.mark.parametrize("effort", [None, "", "adaptive", "default"])
def test_effort_is_required(env, monkeypatch, effort):
    monkeypatch.setenv(api.MODE_ENV, "live")
    with pytest.raises(RuntimeError, match="anthropic_effort"):
        make(env / "prompt.txt", anthropic_effort=effort).prepare()


def test_key_required_and_never_anthropic_api_key(env, monkeypatch):
    monkeypatch.setenv(api.MODE_ENV, "live")
    monkeypatch.delenv(api.KEY_ENV)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-be-ignored")
    monkeypatch.setattr(api.os, "name", "posix")  # no registry fallback in this test
    with pytest.raises(RuntimeError, match=api.KEY_ENV):
        make(env / "prompt.txt").prepare()


@pytest.mark.parametrize("missing", [budget.BUDGET_ENV, budget.PERIOD_ENV])
def test_budget_env_required(env, monkeypatch, missing):
    monkeypatch.setenv(api.MODE_ENV, "live")
    monkeypatch.delenv(missing)
    with pytest.raises(budget.BudgetExceeded, match="must both be set"):
        make(env / "prompt.txt").prepare()


def test_unpriced_model_refused(env, monkeypatch):
    monkeypatch.setenv(api.MODE_ENV, "live")
    with pytest.raises(budget.BudgetExceeded, match="no price row"):
        make(env / "prompt.txt", anthropic_model="claude-imaginary-9").prepare()


def test_over_budget_request_is_never_sent(env, monkeypatch):
    # Opus worst case: 1000 in * $4/M + 16000 out * $20/M = $0.324 > a $0.10 budget.
    monkeypatch.setenv(api.MODE_ENV, "live")
    monkeypatch.setenv(budget.BUDGET_ENV, "0.10")
    fake = FakeApi()
    monkeypatch.setattr(api.requests, "post", fake.post)
    a = make(env / "prompt.txt", anthropic_model="claude-opus-5-5", anthropic_effort="medium")
    with pytest.raises(budget.BudgetExceeded, match="refusing to send"):
        a.play_task(ENV, timeout_s=60)
    assert fake.paid_calls() == 0
    assert not (env / "ledger.jsonl").exists()  # nothing was booked either


# -- live path --------------------------------------------------------------------
def test_live_call_records_effort_usage_and_settles(env, monkeypatch):
    monkeypatch.setenv(api.MODE_ENV, "live")
    fake = FakeApi()
    monkeypatch.setattr(api.requests, "post", fake.post)
    a = make(env / "prompt.txt")
    out = a.play_task(ENV, timeout_s=60)
    assert out == {"verdict": "ok"}  # thinking block is not part of the answer
    sent = [b for u, b in fake.calls if u.endswith("/v1/messages")][0]
    assert sent["output_config"] == {"effort": "high"}
    assert sent["max_tokens"] == 16000
    assert "tools" not in sent and "system" not in sent  # bare model
    assert "t(35) = 2.1" in sent["messages"][0]["content"]
    assert a.last_usage() == {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}
    meta = a.last_response_meta()
    assert meta["sampling"]["effort"] == "high" and meta["sampling"]["model_default_effort"] == "high"
    assert meta["service_tier"] == "standard"
    assert meta["served_model"] == "claude-sonnet-5-5" and meta["finish_reason"] == "end_turn"
    led = budget.Ledger.from_env()
    # Sonnet 5.5 standard: 1000 * $2/M + 200 * $10/M = $0.004 -- the reservation is gone.
    assert led.spent_usd() == pytest.approx(0.004)
    assert a.resolved_tool_version() == "anthropic-api/claude-sonnet-5-5"


def test_http_error_releases_reservation(env, monkeypatch):
    monkeypatch.setenv(api.MODE_ENV, "live")
    fake = FakeApi(status=400)
    monkeypatch.setattr(api.requests, "post", fake.post)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        make(env / "prompt.txt").play_task(ENV, timeout_s=60)
    assert budget.Ledger.from_env().spent_usd() == 0.0


def test_transport_failure_keeps_worst_case_reserved(env, monkeypatch):
    monkeypatch.setenv(api.MODE_ENV, "live")
    fake = FakeApi()

    def flaky(url, **kw):
        if url.endswith("/v1/messages"):
            raise api.requests.ConnectionError("dropped")
        return fake.post(url, **kw)
    monkeypatch.setattr(api.requests, "post", flaky)
    with pytest.raises(api.requests.ConnectionError):
        make(env / "prompt.txt").play_task(ENV, timeout_s=60)
    # 1000 * $2/M + 16000 * $10/M = $0.162 stays booked: the answer may have been billed.
    assert budget.Ledger.from_env().spent_usd() == pytest.approx(0.162)


def test_failed_task_does_not_inherit_previous_usage(env, monkeypatch):
    monkeypatch.setenv(api.MODE_ENV, "live")
    fake = FakeApi()
    monkeypatch.setattr(api.requests, "post", fake.post)
    a = make(env / "prompt.txt")
    a.play_task(ENV, timeout_s=60)
    fake.status = 500
    with pytest.raises(RuntimeError):
        a.play_task({"task_id": "t-002", "input": {"text": "x"}}, timeout_s=60)
    assert a.last_usage() is None and a.last_response_meta() is None


# -- batch plan / replay ------------------------------------------------------------
def test_plan_queues_request_and_sends_nothing_paid(env, monkeypatch):
    monkeypatch.setenv(api.MODE_ENV, "plan")
    fake = FakeApi(input_tokens=1234)
    monkeypatch.setattr(api.requests, "post", fake.post)
    a = make(env / "prompt.txt")
    with pytest.raises(api.BatchPlanned):
        a.play_task(ENV, timeout_s=60)
    assert fake.paid_calls() == 0
    rows = [json.loads(l) for l in (env / "batches" / a.player_id / "requests.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["input_tokens"] == 1234
    assert rows[0]["custom_id"] == api.custom_id(a.player_id, ENV, 1)
    assert len(rows[0]["custom_id"]) <= 64


def test_replay_maps_trials_and_sends_nothing(env, monkeypatch):
    monkeypatch.setenv(api.MODE_ENV, "replay")
    monkeypatch.delenv(api.KEY_ENV)  # replay needs no key at all
    monkeypatch.setattr(api.requests, "post", lambda *a, **k: pytest.fail("network in replay"))
    a = make(env / "prompt.txt")
    d = env / "batches" / a.player_id
    d.mkdir(parents=True)

    def row(trial, text, typ="succeeded"):
        msg = {"id": f"m{trial}", "model": "claude-sonnet-5-5", "stop_reason": "end_turn",
               "content": [{"type": "text", "text": text}],
               "usage": {"input_tokens": 10, "output_tokens": 5}}
        return {"custom_id": api.custom_id(a.player_id, ENV, trial),
                "result": {"type": typ, "message": msg} if typ == "succeeded" else {"type": typ}}
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in (
        row(1, '{"n": 1}'), row(2, '{"n": 2}'), row(3, "", "errored"))), encoding="utf-8")
    # The batch was SENT at 32000 although the registry entry now says otherwise.
    sent = {"model": "claude-sonnet-5-5", "max_tokens": 32000, "output_config": {"effort": "high"}}
    (d / "requests.jsonl").write_text("\n".join(json.dumps(
        {"custom_id": api.custom_id(a.player_id, ENV, t), "params": sent}) for t in (1, 2, 3)),
        encoding="utf-8")
    assert a.play_task(ENV, timeout_s=60) == {"n": 1}
    assert a.last_response_meta()["service_tier"] == "batch"
    # The record states the request that was sent, not today's registry.
    assert a.last_response_meta()["sampling"]["max_tokens"] == 32000
    assert a.last_request_max_tokens() == 32000
    # schema v12: the runner's call_config and raw sidecar read these.
    assert a.last_request_body()["api"] == "anthropic-messages"
    assert a.last_request_body()["max_tokens"] == 32000
    assert a.last_raw_response()["id"] == "m1"
    assert a.play_task(ENV, timeout_s=60) == {"n": 2}
    with pytest.raises(RuntimeError, match="errored"):
        a.play_task(ENV, timeout_s=60)


# -- request shape ---------------------------------------------------------------------
def test_pdf_goes_as_native_document_block(env, monkeypatch):
    monkeypatch.setenv(api.MODE_ENV, "plan")
    p = env / "pdf_prompt.txt"
    p.write_text(PDF_PROMPT, encoding="utf-8")
    a = make(p)
    a.prepare()
    body = a.build_body({"task_id": "p1", "input": {"document_bytes_b64": "JVBERi0=", "n_pages": 7}})
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "document", "source": {"type": "base64",
                          "media_type": "application/pdf", "data": "JVBERi0="}}
    assert "{{" not in content[1]["text"] and "7 pages" in content[1]["text"]


def test_output_coercer_is_applied(env, monkeypatch):
    monkeypatch.setenv(api.MODE_ENV, "live")
    fake = FakeApi(message={"id": "m", "model": "claude-sonnet-5-5", "stop_reason": "end_turn",
                            "content": [{"type": "text", "text": '{"sections": [{"label": "methods"}]}'}],
                            "usage": {"input_tokens": 1, "output_tokens": 1}})
    monkeypatch.setattr(api.requests, "post", fake.post)
    a = make(env / "prompt.txt",
             anthropic_output_coercer="players.adapters.llm_pdf_sections:_coerce_sections")
    out = a.play_task(ENV, timeout_s=60)
    assert "sections" in out


# -- spend guard arithmetic ------------------------------------------------------------
def test_haiku_long_context_tier_is_priced():
    short = budget.worst_case_cost("claude-haiku-5-5", 100_000, 0, batch=False)
    long_ = budget.worst_case_cost("claude-haiku-5-5", 100_001, 0, batch=False)
    assert short == pytest.approx(0.01) and long_ == pytest.approx(100_001 * 0.5 / 1e6)


def test_batch_is_half_price():
    std = budget.usage_cost("claude-opus-5-5", {"input_tokens": 1000, "output_tokens": 1000}, batch=False)
    assert std == pytest.approx(0.024)
    assert budget.usage_cost("claude-opus-5-5", {"input_tokens": 1000, "output_tokens": 1000},
                             batch=True) == pytest.approx(0.012)


def test_corrupt_ledger_refuses(env):
    (env / "ledger.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(budget.BudgetExceeded, match="unreadable"):
        budget.Ledger.from_env().spent_usd()


def test_other_periods_do_not_count(env, monkeypatch):
    led = budget.Ledger.from_env()
    led.reserve("a", 1.0)
    led.settle("a", 0.5)
    monkeypatch.setenv(budget.PERIOD_ENV, "next-month")
    assert budget.Ledger.from_env().spent_usd() == 0.0


# -- framework integration -----------------------------------------------------------------
def test_api_records_are_priced_as_metered_not_subscription():
    from framework import pricing
    rec = {"player_id": "anthropic-api-opus-5-5-medium-esconvert",
           "player_version": "anthropic-api/claude-opus-5-5",
           "resolved_tool_version": "anthropic-api/claude-opus-5-5",
           "usage": {"prompt_tokens": 1000, "completion_tokens": 1000}}
    assert not pricing.is_non_token_billed(rec)
    assert pricing.record_cost_usd(rec) == pytest.approx(0.024)


def test_registry_api_players_build_are_cloud_and_sized():
    from framework import timeouts
    from framework.player_adapter import build_adapter
    from framework.registry import load_registry
    from framework.paths import registry_path
    from framework.runner import is_cloud_player
    entries = [e for e in load_registry(registry_path())
               if e.get("adapter_class") == "AnthropicApiAdapter"]
    # 24 lanes x 3 models x 5 effort levels (the effort ladder, plan step 9, 2026-10-27;
    # the stats-from-pdf lane added 2026-10-28); was 69 (default effort only) until then.
    assert len(entries) == 24 * 3 * 5
    assert {e["anthropic_effort"] for e in entries} == {"low", "medium", "high", "xhigh", "max"}
    for e in entries:
        assert not e["player_id"].startswith("claude-")
        assert is_cloud_player(e)
        assert timeouts.declared_max_tokens(e) == e["anthropic_max_tokens"]
        a = build_adapter(e)
        assert a.anthropic_effort in api.EFFORTS
        assert a.resolved_tool_version() == e["player_version"]


@_needs("gen_anthropic_api_players.py")
def test_generated_registry_block_is_current():
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[3]
    r = subprocess.run([sys.executable, str(root / "scripts" / "gen_anthropic_api_players.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_response_meta_fits_the_run_record_schema(env, monkeypatch):
    import jsonschema
    from framework.paths import schema_path
    schema = json.loads(schema_path("run_record.schema.json").read_text(encoding="utf-8"))
    monkeypatch.setenv(api.MODE_ENV, "live")
    fake = FakeApi()
    monkeypatch.setattr(api.requests, "post", fake.post)
    a = make(env / "prompt.txt")
    a.play_task(ENV, timeout_s=60)
    jsonschema.validate(a.last_response_meta(), schema["properties"]["response_meta"])
    jsonschema.validate(a.last_usage(), schema["properties"]["usage"])


def test_key_falls_back_to_the_stored_user_environment(env, monkeypatch):
    # A key saved by set_anthropic_key.ps1 is not in an already-running process's
    # environment; read_key() must find it in the user's stored environment.
    import sys, types
    monkeypatch.delenv(api.KEY_ENV)
    monkeypatch.setattr(api.os, "name", "nt")

    class _H:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    fake = types.SimpleNamespace(HKEY_CURRENT_USER=1, OpenKey=lambda root, sub: _H(),
                                 QueryValueEx=lambda h, name: ("  stored-key  ", 1))
    monkeypatch.setitem(sys.modules, "winreg", fake)
    assert api.read_key() == "stored-key"
    fake.QueryValueEx = lambda h, name: (_ for _ in ()).throw(OSError("absent"))
    assert api.read_key() == ""


def test_workspace_header_only_when_configured(monkeypatch):
    monkeypatch.setattr(api.os, "name", "posix")
    monkeypatch.delenv(api.WORKSPACE_ENV, raising=False)
    assert "anthropic-workspace-id" not in api._headers("k")
    monkeypatch.setenv(api.WORKSPACE_ENV, "wrkspc_test")
    assert api._headers("k")["anthropic-workspace-id"] == "wrkspc_test"


# -- batch submit that dies on a network error (2026-10-23 sweep: 3 handshake resets) --

def _batch_script():
    import importlib.util
    path = Path(__file__).resolve().parents[3] / "scripts" / "anthropic_batch.py"
    spec = importlib.util.spec_from_file_location("anthropic_batch_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _submit_with_network_error(env, monkeypatch, server_batches):
    mod = _batch_script()
    monkeypatch.setattr(mod.Path, "home", classmethod(lambda cls: env))  # isolate known ids
    d = env / "batches" / "p1"
    d.mkdir(parents=True)
    (d / "requests.jsonl").write_text(json.dumps(
        {"custom_id": "abc-t1", "input_tokens": 100, "params": {"model": "claude-haiku-5-5",
         "max_tokens": 1000, "messages": []}}) + "\n", encoding="utf-8")
    (d / "plan.json").write_text(json.dumps({"arena": "a", "task_set": "v1", "split": "revealed",
                                             "n_requests": 1}), encoding="utf-8")
    monkeypatch.setattr(mod, "_player_entry", lambda pid: {"anthropic_model": "claude-haiku-5-5"})
    monkeypatch.setattr(mod, "_batch_dir", lambda args: d)

    def boom(*a, **k):
        raise mod.requests.ConnectionError("reset during handshake")
    monkeypatch.setattr(mod.requests, "post", boom)
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: FakeResponse(
        200, {"data": server_batches, "has_more": False}))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    rc = mod.cmd_submit(type("A", (), {"player": "p1", "yes": True})())
    return rc, d, budget.Ledger.from_env()


@_needs("anthropic_batch.py")
def test_submit_network_error_with_no_server_batch_releases(env, monkeypatch):
    rc, d, led = _submit_with_network_error(env, monkeypatch, [])
    assert rc == 1 and not (d / "batch.json").exists()
    assert led.spent_usd() == 0.0  # reservation released, not left counting as spent


@_needs("anthropic_batch.py")
def test_submit_network_error_adopts_a_batch_the_server_did_create(env, monkeypatch):
    orphan = {"id": "msgbatch_orphan", "processing_status": "in_progress",
              "request_counts": {"processing": 1, "succeeded": 0, "errored": 0,
                                 "canceled": 0, "expired": 0}}
    rc, d, led = _submit_with_network_error(env, monkeypatch, [orphan])
    assert rc == 0
    assert json.loads((d / "batch.json").read_text(encoding="utf-8"))["id"] == "msgbatch_orphan"
    assert led.spent_usd() > 0  # worst case stays reserved until collect settles it


def test_served_model_is_compared_to_the_bare_api_model():
    # player_version is "anthropic-api/<model>"; comparing the served id to it flagged
    # every API record as a model mismatch (caught in the 2026-10-23 pass-1 replay).
    from framework.runner import _build_response_meta

    class A:
        def last_response_meta(self):
            return {"served_model": "claude-haiku-5-5"}
    entry = {"anthropic_model": "claude-haiku-5-5", "player_version": "anthropic-api/claude-haiku-5-5"}
    assert _build_response_meta(A(), entry, "public")["model_mismatch"] is False
    entry["anthropic_model"] = "claude-sonnet-5-5"
    assert _build_response_meta(A(), entry, "public")["model_mismatch"] is True


def test_pdftotext_mode_sends_the_extracted_text_not_the_pdf(env, monkeypatch):
    # Reference parsing / citation matching v2 give every AI player pdftotext input;
    # the API players must get the same text through the same helper, never the PDF.
    import players.adapters.openai_compatible as oc
    monkeypatch.setattr(oc, "_pdf_to_text", lambda b64, t: "EXTRACTED REFERENCES TEXT")
    a = make(env / "prompt.txt", anthropic_document_to_text="pdftotext")
    a._template = PROMPT
    body = a.build_body({"task_id": "p1", "input": {"document_bytes_b64": "JVBERi0="}})
    content = body["messages"][0]["content"]
    assert isinstance(content, str) and "EXTRACTED REFERENCES TEXT" in content
    assert "document" not in json.dumps(body["messages"])
    with pytest.raises(ValueError):
        make(env / "prompt.txt", anthropic_document_to_text="ocr")


def test_only_live_mode_claims_a_measured_latency(env):
    a = make(env / "prompt.txt")
    for mode, measured in (("live", True), ("replay", False), ("plan", False)):
        a._mode = mode
        assert a.latency_measured is measured

def test_replay_refuses_when_the_registry_no_longer_matches_the_request(env, monkeypatch):
    monkeypatch.setenv(api.MODE_ENV, "replay")
    a = make(env / "prompt.txt")  # registry: sonnet at effort high
    d = env / "batches" / a.player_id
    d.mkdir(parents=True)
    cid = api.custom_id(a.player_id, ENV, 1)
    msg = {"id": "m", "model": "claude-sonnet-5-5", "stop_reason": "end_turn",
           "content": [{"type": "text", "text": "{}"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
    (d / "results.jsonl").write_text(json.dumps({"custom_id": cid, "result": {"type": "succeeded", "message": msg}}),
                                     encoding="utf-8")
    (d / "requests.jsonl").write_text(json.dumps({"custom_id": cid, "params": {
        "model": "claude-sonnet-5-5", "max_tokens": 16000, "output_config": {"effort": "low"}}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Re-run, do not replay"):
        a.play_task(ENV, timeout_s=60)


def test_call_config_reads_an_anthropic_request_as_anthropic():
    from framework import run_env
    body = {"api": "anthropic-messages", "model": "claude-opus-5-5", "max_tokens": 16000,
            "output_config": {"effort": "medium"}, "messages_sha256": "x"}
    cfg = run_env.build_call_config(entry={"adapter_class": "players.adapters.anthropic_api.AnthropicApiAdapter"},
                                    body=body, prompt_template_sha256=None, timeout_s=60)
    assert cfg["api_type"] == "anthropic-messages"
    assert cfg["provider"] == "anthropic"
    assert cfg["model_requested"] == "claude-opus-5-5"
    assert cfg["reasoning_effort"] == "medium"
    assert cfg["max_tokens"] == 16000
    assert cfg["other_body_keys"] == run_env.NA



# -- the exception is CLOSED (owner, 2026-10-29) ---------------------------------------
# sciencearena-review 2026-11-01, F1: "No more API or cloud usage" was enforced nowhere in
# code. With the key still in the user environment, MODE=live (or a batch submit) would have
# billed a closed account. Every paid path reads the key through read_key(); replay does not.
def test_closed_exception_refuses_every_key_read(env, monkeypatch):
    monkeypatch.delenv(api.REOPEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="CLOSED"):
        api.read_key()
    for mode in ("live", "plan"):
        monkeypatch.setenv(api.MODE_ENV, mode)
        with pytest.raises(RuntimeError, match="CLOSED"):
            make(env / "prompt.txt").prepare()


#: scripts/ is not part of the public mirror (mirrored-suite run, 2026-11-02): the tests that import
#: its batch and sweep drivers skip there by name instead of failing.
NEEDS_SCRIPTS = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[3] / "scripts" / "anthropic_batch.py").is_file(),
    reason="scripts/ is not in the public mirror -- NOT verified here")


@NEEDS_SCRIPTS
def test_closed_exception_refuses_the_batch_scripts_key(env, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import anthropic_batch
    monkeypatch.delenv(api.REOPEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="CLOSED"):
        anthropic_batch._key()


def test_closed_exception_still_replays(env, monkeypatch):
    """Control: replay needs no key, so the closure leaves scoring of collected batches alone."""
    monkeypatch.delenv(api.REOPEN_ENV, raising=False)
    monkeypatch.setenv(api.MODE_ENV, "replay")
    a = make(env / "prompt.txt")
    (env / "batches" / a.player_id).mkdir(parents=True)
    for name in ("results.jsonl", "requests.jsonl"):
        (env / "batches" / a.player_id / name).write_text("", encoding="utf-8")
    a.prepare()
    assert a._key is None


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 1000}, {"input_tokens": "1000", "output_tokens": 200}])
def test_a_reply_without_usable_usage_settles_at_the_worst_case(env, monkeypatch, usage):
    """sciencearena-review 2026-11-01, F4: a 200 reply with a missing or partial `usage`
    block settled at $0.00, replacing the worst-case reservation, so the ledger under-counted
    exactly what it exists to bound. It must keep the worst case instead."""
    monkeypatch.setenv(api.MODE_ENV, "live")
    fake = FakeApi()
    if usage is None:
        fake.message.pop("usage")
    else:
        fake.message["usage"] = usage
    monkeypatch.setattr(api.requests, "post", fake.post)
    make(env / "prompt.txt").play_task(ENV, timeout_s=60)
    worst = budget.worst_case_cost("claude-sonnet-5-5", 1000, 16000, batch=False)
    assert budget.Ledger.from_env().spent_usd() == pytest.approx(worst)


@NEEDS_SCRIPTS
def test_sweep_levels_and_players_match_the_generated_registry():
    """sciencearena-review 2026-11-01, F6: the sweep's `ladder` expansion had no test, so a
    LEVELS or MODELS edit could plan players the registry does not hold and fail only late."""
    import sys
    import yaml
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "scripts"))
    import anthropic_sweep as sweep
    import gen_anthropic_api_players as gen
    assert set(sweep.LEVELS) == set(gen.REGISTRY_EFFORTS)
    assert sweep.MODELS == {m: d for m, (_, d) in gen.MODELS.items()}
    assert set(sweep.LANES) <= set(gen.LANES)  # the sweep may run fewer lanes, never others
    reg = yaml.safe_load((root / "players" / "registry.yaml").read_text(encoding="utf-8"))
    ids = {e["player_id"] for e in (reg["players"] if isinstance(reg, dict) else reg)
           if isinstance(e, dict) and "player_id" in e}
    planned = {pid for _, _, _, pid in sweep._players(["default", "ladder"])}
    assert planned and planned <= ids, sorted(planned - ids)[:5]
    # `ladder` is every level except the model's default, `default` is that one level.
    assert len(planned) == len(sweep.LANES) * len(sweep.MODELS) * len(sweep.LEVELS)


@pytest.mark.parametrize("value", ["0", "false", "No", "off", " "])
def test_a_reopen_flag_that_reads_as_off_keeps_the_exception_closed(env, monkeypatch, value):
    """sciencearena-review 2026-11-01 (attempt 2, nit): any non-blank value reopened the API,
    so SCIENCEARENA_ANTHROPIC_REOPENED=0, meant as 'off', would have read the key."""
    monkeypatch.setenv(api.REOPEN_ENV, value)
    with pytest.raises(RuntimeError, match="CLOSED"):
        api.read_key()
