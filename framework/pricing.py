"""Dated price table, kept OUT of the run records on purpose.

A run record stores token counts, which are a measured fact about a run and stay
true forever. A dollar figure is a claim about a published price list that
changes without notice — bake one into a record and the record silently becomes
wrong, with nothing to signal it. So cost is derived here, at report time, from a
table that carries the date each price was checked.

This also makes the honest answer to "what did this cost?" possible: for a
CLI-subscription player (Claude via Claude Max) or a local tool (docpluck,
GROBID, statcheck) the per-task marginal cost is not a token price at all, and
this module returns None rather than inventing one. Absent is not zero.

USD per 1,000,000 tokens. Update `CHECKED_ON` whenever you touch a number, and
re-check before publishing any cost figure — see `stale_days()`.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

#: The table lives in `framework/contract/pricing.json` — INSIDE the installed
#: package, beside the schemas, not at the repo root. A repo-root `contract/`
#: is not a package and is not installed, so a wheel would import fine here and
#: raise FileNotFoundError on a clean `pip install` (caught by the clean-install
#: gate in CI, 2026-08-14). The Node build reads the same file, so that
#: produces the PUBLISHED reports reads the SAME numbers. Two hand-maintained
#: tables in two languages would drift silently, and the drift would show up as
#: a wrong published dollar figure rather than as an error.
_TABLE_PATH = Path(__file__).resolve().parent / "contract" / "pricing.json"
_TABLE = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))

#: The day every price below was last verified against the provider's page.
CHECKED_ON = date.fromisoformat(_TABLE["checked_on"])

#: How old the table may get before a cost figure should not be published.
#: Deliberately short: provider pricing moved several times during 2026.
MAX_AGE_DAYS = int(_TABLE["max_age_days"])

#: model id (as recorded in `resolved_tool_version`) -> (input, output) USD/1M.
#: Free-tier routes are recorded at their PUBLISHED list price, not 0.00, because
#: "free for us right now" is an account property, not a property of the model —
#: a cost-per-catch table that printed 0.00 for every free route would say the
#: benchmark is free rather than that our quota is.
PRICES: dict[str, tuple[float, float]] = {
    model: (p["input"], p["output"]) for model, p in _TABLE["usd_per_1m"].items()
}

#: Players whose marginal per-task cost is NOT a token price. Listed explicitly so
#: a missing price is never confused with a free run. (Claude Max subscription via
#: the CLI; Google Antigravity CLI on OAuth; `gemini-` is retired but kept so the
#: classification stays honest for historical records.)
NON_TOKEN_BILLED = tuple(_TABLE["non_token_billed_prefixes"])


def is_non_token_billed(record: dict) -> bool:
    """True when this player has no per-token marginal cost at all.

    Checks the served model AND the player id, because CLI adapters leave
    ``resolved_tool_version`` unset — keying on it alone mislabels every Claude
    record as un-metered, the opposite of the truth for a subscription player
    that consumes plenty. Mirrored by `isNonTokenBilled` in
    leaderboard-app/scripts/lib/pricing.mjs; the two engines must not disagree,
    since both publish numbers under the same arena name.
    """
    for field in ("resolved_tool_version", "player_version", "player_id"):
        value = record.get(field) or ""
        if any(value.startswith(p) for p in NON_TOKEN_BILLED):
            return True
    return False


def record_cost_usd(record: dict) -> float | None:
    """Cost of one RECORD, applying the same three-field subscription check the
    Node build applies. Prefer this over :func:`cost_usd` when you have a whole
    record rather than a bare model id."""
    if is_non_token_billed(record):
        return None
    return cost_usd(record.get("resolved_tool_version"), record.get("usage"))


def stale_days(today: date | None = None) -> int:
    """Days since the table was verified. Compare against MAX_AGE_DAYS."""
    return ((today or date.today()) - CHECKED_ON).days


def is_stale(today: date | None = None) -> bool:
    return stale_days(today) > MAX_AGE_DAYS


def _token_count(value) -> int | None:
    """Strict token reader: an int that is not a bool, else None.

    A float, string, or None is a BROKEN measurement, not a fractional one —
    summing it publishes an invented number (780.5 tokens) or raises. Booleans
    are rejected explicitly because ``isinstance(True, int)`` is True in Python
    while ``Number.isInteger(true)`` is false in JS: the same bool-coercion class
    as the ``_close(2, True)`` defect the 2026-08-04 cross-review caught.

    Mirrors ``Number.isInteger`` in leaderboard-app/scripts/lib/pricing.mjs. The
    two engines publish the same field names; they must not disagree on what
    counts as a measurement. (Fable 5 cross-review, 2026-08-15.)
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def cost_usd(model: str | None, usage: dict | None) -> float | None:
    """Cost of one task, or None when the question does not apply.

    Returns None — never 0.0 — when the model is unpriced, the player is not
    token-billed, or no usage was recorded. A zero would read as "this was free",
    which is a different and usually false claim.
    """
    if not model or not usage:
        return None
    if any(model.startswith(p) for p in NON_TOKEN_BILLED):
        return None
    price = PRICES.get(model)
    if price is None:
        return None
    inp, out = price
    prompt = _token_count(usage.get("prompt_tokens"))
    completion = _token_count(usage.get("completion_tokens"))
    if prompt is None or completion is None:
        return None
    return (prompt * inp + completion * out) / 1_000_000.0


def summarise(records) -> dict:
    """Aggregate tokens and cost over run records.

    `n_priced` vs `n_with_usage` is the honesty column: it says how much of the
    cost figure is actually backed by a known price rather than quietly dropped.
    """
    tokens_in = tokens_out = 0
    n_with_usage = n_priced = 0
    total = 0.0
    for r in records:
        usage = r.get("usage")
        # An EMPTY usage block is an absent measurement, not a zero one. `{}` is
        # falsy here and truthy in JS, so this line is where the two engines used
        # to part company; pricing.mjs now applies the same emptiness test.
        if not usage:
            continue
        n_with_usage += 1
        # Strict, not `or 0`: a malformed count must not add as zero (which reads
        # as a cheap run) nor as a float (which publishes 992.5 tokens).
        prompt = _token_count(usage.get("prompt_tokens"))
        completion = _token_count(usage.get("completion_tokens"))
        tokens_in += prompt or 0
        tokens_out += completion or 0
        c = record_cost_usd(r)
        if c is not None:
            n_priced += 1
            total += c

    # Why usage is absent, when it is entirely absent. Reported as the single
    # dominant reason, so a consumer can say "not priced — subscription" rather
    # than the false "consumed 0 tokens".
    reason = None
    if n_with_usage == 0:
        counts: Counter = Counter()
        for r in records:
            why = classify_missing_usage(r)
            if why:
                counts[why] += 1
        reason = counts.most_common(1)[0][0] if counts else None

    return {
        "n_with_usage": n_with_usage,
        "n_priced": n_priced,
        "usage_absent_reason": reason,
        # None, never 0, when nothing was measured. A zero asserts "this player
        # consumed no tokens", which is the opposite of the truth for a
        # subscription player that consumed plenty. Mirrors pricing.mjs.
        "prompt_tokens": tokens_in if n_with_usage else None,
        "completion_tokens": tokens_out if n_with_usage else None,
        "total_tokens": (tokens_in + tokens_out) if n_with_usage else None,
        "cost_usd": round(total, 6) if n_priced else None,
        "price_table_checked_on": CHECKED_ON.isoformat(),
        "price_table_stale": is_stale(),
    }


#: The day `OpenAIChatCompletionsAdapter` began recording the `usage` block.
#: Records older than this have no token data and none can be recovered: it was
#: never stored, no local logs captured it, and Groq / Mistral / NVIDIA all return
#: 404 for usage-history endpoints (probed 2026-08-13).
INSTRUMENTED_ON = date.fromisoformat(_TABLE["instrumented_on"])


def classify_missing_usage(record: dict) -> str | None:
    """Why does this record have no `usage`? Returns None if it has some.

    Four causes that look identical in the data and mean different things:

      "not_metered"    local tool player — consumed no tokens at all
      "subscription"   CLI player on a flat subscription — no per-token price
      "held_out"       stripped, because prompt_tokens leaks document length
      "pre_instrument" ran before 2026-08-13 — the data was simply never captured

    Only the last is a gap in our records rather than a property of the run, and
    conflating it with the others would make the benchmark look like it measured
    something it did not.
    """
    if record.get("usage"):
        return None
    if record.get("task_visibility") == "held_out":
        return "held_out"
    # Check BOTH the served model and the player id. CLI adapters leave
    # `resolved_tool_version` None (their version is already the model id in
    # player_version), so keying on it alone mislabelled every Claude record as
    # "not_metered" — which reads as "this player consumes no tokens", the exact
    # opposite of the truth for a subscription player that consumes plenty.
    for field in ("resolved_tool_version", "player_version", "player_id"):
        value = record.get(field) or ""
        if any(value.startswith(p) for p in NON_TOKEN_BILLED):
            return "subscription"
    if record.get("player_type") in ("tool", "platform"):
        return "not_metered"
    ts = (record.get("timestamp_utc") or "")[:10]
    if ts and ts < INSTRUMENTED_ON.isoformat():
        return "pre_instrument"
    return "not_metered"
