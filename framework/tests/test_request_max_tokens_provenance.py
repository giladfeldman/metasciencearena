"""`provenance.request_max_tokens` — the output ceiling WE sent, recorded per record.

WHY THIS EXISTS. `players/adapters/openai_compatible.py::_response_meta` states the
reason `finish_reason` is captured at all: "a completion cut off at the token limit
fails JSON parsing, scores 0.0, and is then indistinguishable from a model that
simply cannot do the task — the benchmark would be publishing OUR max_tokens as
THEIR incapability". But until 2026-09-11 the ceiling itself was recorded nowhere.
Measured that day over every run file in the repo: `provenance.request_max_tokens`
was absent from all 11,709 records, and `response_meta.max_tokens` was present in
only the 542 written by the local llama.cpp adapter — so for every hosted-provider
record, `finish_reason == "length"` could not be attributed to our configuration or
to the model's own limit.

It was not hypothetical. The Cohere Catalyst grant run produced 7 records truncated
at exactly 4000 completion tokens, and answering "ours or theirs?" required probing
the live Cohere API by bisection because the records could not answer it. The true
ceilings are 8192/4096/4096; the registry had been set to 8000/4000/4000, i.e.
192/96/96 BELOW the models' own limits, which published at least one true 1.000 as
a 0.0.

TWO-SIDED ON PURPOSE. A test that only checks the adapter default passes even when
a player's `openai_extra_body.max_tokens` override is silently dropped — and the
override case is the entire point, because that is where the registry value and the
value on the wire diverge. So both directions are asserted, against the SAME code
path, plus a third player that sends no ceiling at all and must record `null` rather
than a fabricated default.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from framework.runner import run_tournament
from framework.storage import read_records
from players.adapters.openai_compatible import _DEFAULT_MAX_TOKENS
from framework.tests._container_fakes import in_process_runner  # noqa: F401  (fixture)

ARENA = Path("arenas/grim-consistency-v1")

#: The override under test. Deliberately NOT equal to `_DEFAULT_MAX_TOKENS`, and
#: deliberately not a round number that could coincide with some other constant.
_OVERRIDE = 4096

_VALID_OUTPUT = {"records": []}


def _openai_entry(player_id: str, extra_body: dict | None) -> dict:
    entry = {
        "player_id": player_id,
        "player_type": "ai-model",
        "player_version": "fake/ceiling-probe",
        "adapter_class": "OpenAIChatCompletionsAdapter",
        "openai_model": "fake-ceiling-probe",
        "openai_base_url_env": "CEILING_PROBE_BASE_URL",
        "openai_api_key_env": "CEILING_PROBE_API_KEY",
        "prompt_template_path": "players/prompts/grim_consistency.txt",
        "confidence_strategy": "native",
        "deterministic": True,
    }
    if extra_body is not None:
        entry["openai_extra_body"] = extra_body
    return entry


#: A player that sends no ceiling because it speaks no HTTP at all. `scrutiny-grim`
#: is a real registry entry (RCliAdapter) already used by test_runner_provenance.
_NO_CEILING_PLAYER = "scrutiny-grim"


@pytest.fixture
def fake_provider(monkeypatch, in_process_runner):
    """Capture every request body the adapter actually puts on the wire."""
    monkeypatch.setenv("CEILING_PROBE_BASE_URL", "https://api.invalid/v1")
    monkeypatch.setenv("CEILING_PROBE_API_KEY", "k")
    sent: list[dict] = []

    class FakeResponse:
        status_code = 200
        headers: dict = {}

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": json.dumps(_VALID_OUTPUT)},
                             "finish_reason": "stop"}],
                "model": "fake-ceiling-probe",
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        return FakeResponse()

    monkeypatch.setattr("players.adapters.openai_compatible.requests.post", fake_post)
    return sent


def _run(tmp_path, entries, player_ids, out_name="probe.jsonl"):
    reg = tmp_path / "registry.yaml"
    reg.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")
    out = tmp_path / out_name
    # public_only: grim's tasks are all `public`, and it keeps the held-out egress
    # gate out of this test — the gate is about disclosure, not about ceilings.
    run_tournament(arena_dir=ARENA, task_set_version="v1", registry_path=reg,
                   player_ids=player_ids, output_path=out, trials=1,
                   timeout_s=120, seed=0, split="revealed", max_tasks=1,
                   public_only=True)
    return read_records(out)


def test_override_and_default_are_both_recorded(tmp_path, fake_provider):
    """The value ACTUALLY SENT, for a player with an override and one without.

    Both players run in the SAME tournament so neither can pass on a stale
    attribute left by the other — the failure mode a per-player run would hide.
    """
    entries = [
        _openai_entry("probe-override-grim", {"max_tokens": _OVERRIDE}),
        _openai_entry("probe-default-grim", None),
    ]
    recs = _run(tmp_path, entries, [e["player_id"] for e in entries])
    by_player = {r["player_id"]: r for r in recs}
    assert set(by_player) == {"probe-override-grim", "probe-default-grim"}, \
        f"expected one record per player, got {sorted(by_player)}"

    # Control: the request bodies really did carry two different ceilings, so a
    # match below is evidence about the record and not about a constant.
    on_the_wire = sorted(b["max_tokens"] for b in fake_provider)
    assert on_the_wire == sorted([_OVERRIDE, _DEFAULT_MAX_TOKENS]), \
        f"the adapter did not send two distinct ceilings: {on_the_wire}"

    assert by_player["probe-override-grim"]["provenance"]["request_max_tokens"] == _OVERRIDE, \
        ("a player's openai_extra_body.max_tokens override was NOT recorded — this is "
         "the exact bug a default-only test passes through")
    assert by_player["probe-default-grim"]["provenance"]["request_max_tokens"] == _DEFAULT_MAX_TOKENS, \
        "a player with no override must record the adapter's own default"


def test_a_player_that_sends_no_ceiling_records_null(tmp_path):
    """No fabricated default for adapters that put no ceiling on any wire.

    A wrong number here is worse than a missing one: it would let a reader
    "confirm" that a CLI player's truncation was our configuration's fault.
    """
    recs = _run(tmp_path, yaml.safe_load(Path("players/registry.yaml").read_text(encoding="utf-8")),
                [_NO_CEILING_PLAYER], out_name="cli.jsonl")
    assert recs, "no records written"
    prov = recs[0]["provenance"]
    assert prov["adapter_class"] == "RCliAdapter", \
        f"control failed: expected a non-HTTP adapter, got {prov['adapter_class']!r}"
    assert "request_max_tokens" in prov, \
        "the key must be PRESENT and null, so absence keeps meaning 'predates the field'"
    assert prov["request_max_tokens"] is None, \
        f"a player that sends no ceiling recorded {prov['request_max_tokens']!r}"


# --------------------------------------------------------------- held-out boundary
# The brief that asked for this field said to CONFIRM the redaction verdict against
# `framework/holdout.py` rather than assume it. The reasoning is that the ceiling is
# a number WE chose, so unlike `usage` token counts (a near-linear proxy for the
# length of a document we deliberately do not distribute) it discloses nothing about
# the task — and a held-out record is exactly where it is most needed, because such
# a record has no `output` left to inspect. `_response_meta` makes the same argument
# for `finish_reason`, its companion. This pins that verdict so a future change to
# the single redaction boundary cannot silently take the field with it.

from framework.holdout import held_out_leak_reasons, redact_held_out_record


def _held_out_record(cap):
    return {
        "task_id": "t1",
        "task_visibility": "held_out",
        "input_hash": "a" * 64,
        "output": {"records": []},
        "usage": {"prompt_tokens": 12345, "completion_tokens": 7},
        "score": {"primary": 1.0, "breakdown": {"n_rows": 9}},
        "response_meta": {"finish_reason": "length"},
        "provenance": {"seed": 424242, "request_max_tokens": cap,
                       "request_timeout_s": 300},
    }


def test_the_ceiling_survives_held_out_redaction():
    red = redact_held_out_record(_held_out_record(_OVERRIDE))

    # CONTROLS FIRST. If redaction did not actually run, "the field survived" is a
    # statement about an unredacted record and proves nothing.
    assert red["output"] == {}, "control: redaction did not run at all"
    assert "usage" not in red, "control: the length oracle was not dropped"
    assert red["provenance"]["seed"] == "<redacted>", \
        "control: the private seed was not redacted"

    assert red["provenance"]["request_max_tokens"] == _OVERRIDE, \
        ("the ceiling is a number WE chose and says nothing about the document — it "
         "must survive, or a held-out truncation becomes unattributable forever")
    assert held_out_leak_reasons(red, kind="record") == [], \
        "the redacted record must be clean by the boundary's own leak check"


# ------------------------------------------------------- backward compatibility
# `provenance` is `additionalProperties: false`, which is the one place where
# "add a field" and "require a field" are a single edit apart — and the difference
# decides whether the 11,709 records already on disk stay valid. Records are never
# backfilled here, so every one of them carries the ABSENCE, which as of v9 means
# "this run predates the field" and must never mean "unknown ceiling, invalid
# record". Raised by a peer session on 2026-09-11 before the commit landed.

def test_a_record_written_before_v9_still_validates():
    """The key is OPTIONAL. A v8-shaped provenance block must not become invalid."""
    from jsonschema import Draft202012Validator

    schema = json.loads(
        Path("framework/contract/schemas/run_record.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    legacy = {
        "run_id": "r1", "arena_id": "grim-consistency-v1", "task_set_version": "v1",
        "task_id": "t1", "player_id": "p", "player_version": "v", "player_type": "ai-model",
        "input_hash": "a" * 64, "output": {}, "score": {"primary": 1.0, "breakdown": {}},
        "timestamp_utc": "2026-08-13T00:00:00Z", "schema_version": 8,
        # schema_version >= 7 conditionally requires `instrument`; supplied so this
        # test fails for the ONE reason it is about and not for an unrelated rule.
        "instrument": {"formula_id": "f", "scorer_sha256": "b" * 64},
        "provenance": {"request_temperature": 0.0, "request_timeout_s": 300},
    }
    assert list(validator.iter_errors(legacy)) == [], \
        "a pre-v9 record was invalidated by making the new ceiling field required"

    # Two-sided: the SAME record with the field present is equally valid, and an
    # unknown sibling is still rejected — so the check above is about optionality
    # and not about a schema that stopped enforcing anything.
    with_field = {**legacy, "schema_version": 9,
                  "provenance": {**legacy["provenance"], "request_max_tokens": 4096}}
    assert list(validator.iter_errors(with_field)) == []

    bogus = {**legacy, "provenance": {**legacy["provenance"], "request_max_tokns": 4096}}
    assert list(validator.iter_errors(bogus)), \
        "control: provenance is no longer a CLOSED object, so this test proves nothing"
