"""Token capture and cost derivation.

Context: `cost_usd` has been declared in run_record.schema.json and aggregated by
framework/report.py since those files were written, and is `null` in every report
ever published — no adapter populated it, and `_chat_content_from_response`
discarded the `usage` block every provider returns on every call.

The fix records TOKENS (a measured fact) and derives cost from a dated price
table (a claim that drifts). These tests pin the distinctions that make that
honest, especially: absent must never be read as zero.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from framework import pricing
from framework.player_adapter import PlayerAdapter, StubPassAdapter


def test_base_adapter_reports_no_usage_by_default():
    """A local tool consumes no tokens; it must say None, not 0."""
    a = StubPassAdapter(player_id="p", player_version="v", player_type="tool",
                        confidence_strategy="implicit-1.0", deterministic=True)
    assert a.last_usage() is None
    assert PlayerAdapter.last_usage(a) is None


def test_cost_is_none_not_zero_when_unknown():
    """Every 'we cannot say' path must return None.

    0.0 would render as a free run on a cost-per-catch table, which is a
    different and usually false claim.
    """
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    assert pricing.cost_usd(None, usage) is None                    # no model
    assert pricing.cost_usd("openai/gpt-oss-120b", None) is None    # no usage
    assert pricing.cost_usd("some-unpriced-model", usage) is None   # not in table
    assert pricing.cost_usd("openai/gpt-oss-120b", {}) is None      # empty usage


def test_subscription_players_are_not_token_billed():
    """Claude via Claude Max has no per-token marginal cost. Saying $0 would be
    as wrong as saying $5 — the question does not apply."""
    usage = {"prompt_tokens": 10_000, "completion_tokens": 10_000}
    for model in ("claude-opus-4-8", "claude-sonnet-5", "antigravity-gemini-3.6"):
        assert pricing.cost_usd(model, usage) is None, model


def test_known_model_computes_the_arithmetic():
    # gpt-oss-120b: $0.15/1M in, $0.75/1M out
    c = pricing.cost_usd("openai/gpt-oss-120b",
                         {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
    assert c == pytest.approx(0.15 + 0.75)


def test_free_routes_carry_their_LIST_price_not_zero():
    """'Free for us' is a property of our quota, not of the model.

    A cost table printing 0.00 for every free route would claim the benchmark is
    free to run, which is not what the number is for.
    """
    for model in ("openai/gpt-oss-120b", "nvidia/nemotron-3.5-lightning-30b-a3b"):
        inp, out = pricing.PRICES[model]
        assert inp > 0 and out > 0, f"{model} recorded as free"


def test_staleness_is_detectable():
    """A price table with no expiry silently publishes last year's prices."""
    assert pricing.stale_days(pricing.CHECKED_ON) == 0
    assert not pricing.is_stale(pricing.CHECKED_ON)
    old = date(pricing.CHECKED_ON.year + 1, pricing.CHECKED_ON.month, pricing.CHECKED_ON.day)
    assert pricing.is_stale(old), "a year-old table must report stale"


def test_summarise_separates_priced_from_merely_measured():
    """n_priced vs n_with_usage is the honesty column: how much of the cost figure
    is actually backed by a known price rather than quietly dropped."""
    records = [
        {"resolved_tool_version": "openai/gpt-oss-120b",
         "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}},
        {"resolved_tool_version": "claude-opus-4-8",          # not token-billed
         "usage": {"prompt_tokens": 500, "completion_tokens": 500}},
        {"resolved_tool_version": "statcheck"},                # local tool, no usage
    ]
    s = pricing.summarise(records)
    assert s["n_with_usage"] == 2, "the claude record still reports tokens"
    assert s["n_priced"] == 1, "only one record had a usable price"
    assert s["prompt_tokens"] == 1_000_500
    assert s["cost_usd"] == pytest.approx(0.15)
    assert s["price_table_checked_on"] == pricing.CHECKED_ON.isoformat()


def test_summarise_reports_none_cost_when_nothing_is_priced():
    s = pricing.summarise([{"resolved_tool_version": "claude-opus-4-8",
                            "usage": {"prompt_tokens": 10, "completion_tokens": 10}}])
    assert s["cost_usd"] is None, "no priced record must yield None, not 0.0"
    assert s["n_with_usage"] == 1


# --- Claude CLI envelope (2026-08-13) -----------------------------------------
#
# `claude --print --output-format json` returns {result, usage, stop_reason,
# modelUsage, total_cost_usd}. Without opting into it the framework saw bare text
# and every Claude player's tokens were unrecoverable — the same data loss the
# HTTP adapter had. Opt-in per player so no other CLI's output shape changes.

from framework.player_adapter import SubprocessCliAdapter


def _cli_adapter(tmp_path, envelope_mode="claude"):
    tmpl = tmp_path / "p.txt"
    tmpl.write_text("{{INPUT_TEXT}}", encoding="utf-8")
    return SubprocessCliAdapter(
        player_id="p", player_version="claude-haiku-4-5", player_type="ai-model",
        confidence_strategy="native", deterministic=False,
        cli_command=["claude", "--print"], prompt_template_path=str(tmpl),
        cli_json_envelope=envelope_mode,
    )


CLAUDE_ENVELOPE = json.dumps({
    "result": '{"flags": []}',
    "stop_reason": "end_turn",
    "session_id": "sess-123",
    "total_cost_usd": 0.0497234,
    "modelUsage": {"claude-haiku-4-5": {"inputTokens": 9}},
    "usage": {"input_tokens": 9, "output_tokens": 583,
              "cache_read_input_tokens": 24694, "cache_creation_input_tokens": 22165},
})


def test_claude_envelope_yields_the_answer_and_the_telemetry(tmp_path):
    a = _cli_adapter(tmp_path)
    assert json.loads(a._unwrap_claude_envelope(CLAUDE_ENVELOPE)) == {"flags": []}
    u = a.last_usage()
    # prompt_tokens is the SUM of the three input-side buckets Anthropic reports.
    assert u["prompt_tokens"] == 9 + 24694 + 22165
    assert u["completion_tokens"] == 583
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert u["cache_read_tokens"] == 24694
    assert u["cache_creation_tokens"] == 22165
    m = a.last_response_meta()
    assert m["finish_reason"] == "end_turn"       # Anthropic's name for it
    assert m["served_model"] == "claude-haiku-4-5"
    assert m["provider_reported_cost_usd"] == pytest.approx(0.0497234)


def test_non_envelope_stdout_falls_through_untouched(tmp_path):
    """A CLI that ignores --output-format must still work.

    Falling through to the permissive extractor means enabling the envelope can
    never turn a working player into a failing one.
    """
    a = _cli_adapter(tmp_path)
    assert a._unwrap_claude_envelope('```json\n{"flags": []}\n```').strip().startswith("```")
    assert a.last_usage() is None


def test_envelope_is_opt_in(tmp_path):
    """Every other CLI player's output shape must be untouched."""
    a = _cli_adapter(tmp_path, envelope_mode=None)
    assert a.cli_json_envelope is None
    assert a.last_usage() is None


def test_provider_reported_cost_is_not_treated_as_spend(tmp_path):
    """Claude players run on a subscription: pricing must still refuse to price them.

    The CLI's own total_cost_usd is an API-EQUIVALENT figure. If `cost_usd` started
    honouring it, the benchmark would report subscription runs as money spent.
    """
    assert pricing.cost_usd("claude-haiku-4-5",
                            {"prompt_tokens": 46868, "completion_tokens": 583}) is None


def test_a_claude_cli_record_is_classified_as_subscription_not_unmetered():
    """CLI adapters leave resolved_tool_version None — their version IS the model id
    in player_version. Keying only on resolved_tool_version mislabelled every Claude
    record "not_metered", i.e. "consumes no tokens", which is the opposite of true:
    a claude --print call consumes ~47k tokens, they are just not billed per token.
    """
    rec = {"player_id": "claude-haiku-4-5-siglang", "player_version": "claude-haiku-4-5",
           "player_type": "ai-model", "resolved_tool_version": None,
           "timestamp_utc": "2026-08-13T10:00:00Z"}
    assert pricing.classify_missing_usage(rec) == "subscription"


def test_a_genuinely_unmetered_tool_still_says_not_metered():
    rec = {"player_id": "statcheck", "player_type": "tool",
           "resolved_tool_version": "statcheck-1.5.0",
           "timestamp_utc": "2026-08-13T10:00:00Z"}
    assert pricing.classify_missing_usage(rec) == "not_metered"


# ---------------------------------------------------------------------------
# The Python and Node cost engines must agree.
#
# Both publish numbers under the same arena name: framework/report.py produces
# the tool-feedback bundles, leaderboard-app/scripts/lib/report.mjs produces the
# reports the SITE serves. If they disagreed about whether a run was priced, the
# same player would carry two different costs depending on which artifact you
# read. Added 2026-08-14 with the shared contract/pricing.json table.
# ---------------------------------------------------------------------------


def test_subscription_player_is_unpriced_even_with_a_priced_model_string():
    """The three-field check, not just the model id.

    A CLI adapter leaves resolved_tool_version unset and puts the model in
    player_version/player_id. Keying on resolved_tool_version ALONE would price a
    Claude Max run that has no per-token marginal cost at all.
    """
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    rec = {"player_id": "claude-opus-4-8", "player_version": "claude-opus-4-8",
           "resolved_tool_version": None, "usage": usage}
    assert pricing.record_cost_usd(rec) is None
    assert pricing.is_non_token_billed(rec) is True


def test_record_cost_matches_the_node_engine_rules():
    """Same inputs, same rules as isNonTokenBilled/costUsd in pricing.mjs."""
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    priced = {"player_id": "groq-gpt-oss-120b", "player_version": "1.0",
              "resolved_tool_version": "openai/gpt-oss-120b", "usage": usage}
    assert pricing.record_cost_usd(priced) == pytest.approx(0.90)

    unpriced_model = dict(priced, resolved_tool_version="some-model-not-in-table")
    assert pricing.record_cost_usd(unpriced_model) is None

    no_usage = dict(priced, usage=None)
    assert pricing.record_cost_usd(no_usage) is None


def test_price_table_is_the_shared_json_contract():
    """Both languages read contract/pricing.json. If the Python module ever grows
    its own inline table again, they will drift and the drift will surface as a
    wrong published dollar figure rather than an error."""
    import json
    table = json.loads(pricing._TABLE_PATH.read_text(encoding="utf-8"))
    assert pricing._TABLE_PATH.name == "pricing.json"
    assert pricing._TABLE_PATH.parent.name == "contract"
    assert set(table["usd_per_1m"]) == set(pricing.PRICES), (
        "PRICES no longer matches the shared table"
    )
    assert tuple(table["non_token_billed_prefixes"]) == pricing.NON_TOKEN_BILLED


# ---------------------------------------------------------------------------
# Cross-engine parity with leaderboard-app/scripts/lib/pricing.mjs.
#
# Found by the Fable 5 cross-model review, 2026-08-15. The Node engine received
# a strict-integer fix (Codex review, 2026-08-14) that was never mirrored back
# into Python, and Python's `summarise` returned raw sums — so the two engines
# published DIFFERENT values under the same field names. Reproduced on both
# sides before these tests were written.
# ---------------------------------------------------------------------------

_SUBSCRIPTION = {"player_id": "claude-opus-5-x", "player_version": "claude-opus-5",
                 "player_type": "model", "timestamp_utc": "2026-08-14T10:00:00Z"}
_LOCAL_TOOL = {"player_id": "statcheck", "player_type": "tool",
               "timestamp_utc": "2026-08-14T10:00:00Z"}


def test_absent_usage_reports_none_not_zero():
    """A subscription or local player consumed tokens we did not measure (or none
    at all). Publishing 0 asserts "this player consumed nothing" — the opposite
    of the truth for a Claude-via-CLI run, and the exact claim pricing.py's own
    module docstring forbids ("Absent is not zero").

    The Node engine already returns null here (pricing.mjs `tokens_prompt:
    nWithUsage ? ... : null`); Python returned 0, so `framework report` and the
    published site disagreed on a field with the same name.
    """
    for label, rec in (("subscription", _SUBSCRIPTION), ("local tool", _LOCAL_TOOL)):
        got = pricing.summarise([rec])
        assert got["prompt_tokens"] is None, f"{label}: 0 tokens read as measured"
        assert got["completion_tokens"] is None, label
        assert got["total_tokens"] is None, label
        assert got["n_with_usage"] == 0, label


def test_absent_usage_reports_why():
    """`usage_absent_reason` is the honesty field: "not recorded" and "consumed
    nothing" look identical in the data and mean different things. Node published
    it; Python omitted it entirely, so the two report shapes differed."""
    assert pricing.summarise([_SUBSCRIPTION])["usage_absent_reason"] == "subscription"
    assert pricing.summarise([_LOCAL_TOOL])["usage_absent_reason"] == "not_metered"
    priced = {"player_id": "g", "resolved_tool_version": "openai/gpt-oss-120b",
              "player_type": "model", "timestamp_utc": "2026-08-14T10:00:00Z",
              "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    assert pricing.summarise([priced])["usage_absent_reason"] is None


def test_non_integer_token_counts_are_not_summed():
    """A float/string token count is a BROKEN measurement, not a fractional one.

    Python did `usage.get("prompt_tokens", 0) or 0`, which published 780.5 as a
    token count (and would raise TypeError on a string); Node's Number.isInteger
    guard skipped it. Neither may silently invent a number.
    """
    bad = {"player_id": "g", "resolved_tool_version": "openai/gpt-oss-120b",
           "player_type": "model", "timestamp_utc": "2026-08-14T10:00:00Z",
           "usage": {"prompt_tokens": 780.5, "completion_tokens": 212}}
    got = pricing.summarise([bad])
    assert got["prompt_tokens"] == 0, "a non-integer token count must not be summed"
    assert got["completion_tokens"] == 212
    assert got["total_tokens"] == 212


def test_booleans_are_not_token_counts():
    """isinstance(True, int) is True in Python and Number.isInteger(true) is false
    in JS — the same bool-coercion class as the `_close(2, True)` defect the
    2026-08-04 cross-review caught."""
    rec = {"player_id": "g", "resolved_tool_version": "openai/gpt-oss-120b",
           "player_type": "model", "timestamp_utc": "2026-08-14T10:00:00Z",
           "usage": {"prompt_tokens": True, "completion_tokens": True}}
    assert pricing.cost_usd("openai/gpt-oss-120b", rec["usage"]) is None
    assert pricing.summarise([rec])["prompt_tokens"] == 0


def test_empty_usage_block_is_no_measurement():
    """`usage: {}` is an empty measurement, not a zero one. Python skipped it
    ({} is falsy) while Node counted it (truthy) and then published 0 tokens."""
    rec = {"player_id": "g", "resolved_tool_version": "openai/gpt-oss-120b",
           "player_type": "model", "timestamp_utc": "2026-08-14T10:00:00Z",
           "usage": {}}
    got = pricing.summarise([rec])
    assert got["n_with_usage"] == 0
    assert got["total_tokens"] is None
