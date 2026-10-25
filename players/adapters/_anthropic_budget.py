"""Local spend guard for the Anthropic API players.

WHY THIS EXISTS. The owner approved API use for ScienceArena on one condition: it
is funded ONLY by the Max-included monthly API credit and must never cost money
beyond it (standing "Claude Max only" rule, exception of 2026-10-23). The
Console already stops requests at a zero balance (auto-reload off) and caps spend
at the monthly credit. This module is the third, local guard: it refuses to SEND
a request whose worst-case cost would take the period's spend past the budget the
operator declared, so a script loop or a mis-sized batch is stopped on this
machine before Anthropic ever sees it.

HOW. Every request is RESERVED at its worst case (counted input tokens + the full
`max_tokens` ceiling as output) before it goes on the wire, then SETTLED at the
cost its own `usage` block reports, or RELEASED when the API says it billed
nothing. Spent = settled charges + reservations still open. An open reservation
is deliberately counted at its worst case: a request whose fate is unknown
(client timeout, crash mid-batch) may still have been billed.

Configuration, all from the live environment, none with a default that spends:
  SCIENCEARENA_ANTHROPIC_BUDGET_USD  required; the most this period may spend
  SCIENCEARENA_ANTHROPIC_PERIOD      required; a label for the credit window,
                                     e.g. "2026-10-credit" (one per monthly grant)
  SCIENCEARENA_ANTHROPIC_LEDGER      optional; default ~/.sciencearena/anthropic_spend.jsonl
                                     (outside every repo and worktree, so all of
                                     them share one ledger)

Prices come from framework/contract/pricing.json (`anthropic-api/<model>` rows),
the same table the published reports read. A model with no row is refused: an
unpriced request has an unknowable worst case.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

BUDGET_ENV = "SCIENCEARENA_ANTHROPIC_BUDGET_USD"
PERIOD_ENV = "SCIENCEARENA_ANTHROPIC_PERIOD"
LEDGER_ENV = "SCIENCEARENA_ANTHROPIC_LEDGER"

#: Message Batches API price relative to standard (platform pricing page,
#: checked 2026-10-23: "a 50 percent discount on both input and output tokens").
BATCH_FACTOR = 0.5

_PRICING_PATH = Path(__file__).resolve().parents[2] / "framework" / "contract" / "pricing.json"


class BudgetExceeded(RuntimeError):
    """Raised BEFORE a request is sent when its worst case would breach the budget."""


def price_key(model: str) -> str:
    """Pricing-table key for an API model. Vendor-prefixed on purpose: the table
    classifies anything starting `claude-` as a Max-subscription (un-metered)
    player, and an API call is metered."""
    return f"anthropic-api/{model}"


def model_prices(model: str, pricing_path: Path = _PRICING_PATH,
                 input_tokens: int = 0) -> tuple[float, float]:
    """(input, output) USD per 1M tokens at the STANDARD price. Raises if unpriced.

    A row may carry a long-context tier (Haiku 5.5 bills prompts over 100,000
    tokens at a higher rate); `input_tokens` selects it."""
    table = json.loads(pricing_path.read_text(encoding="utf-8"))["usd_per_1m"]
    row = table.get(price_key(model))
    if not row:
        raise BudgetExceeded(
            f"no price row {price_key(model)!r} in {pricing_path.name}; refusing to send "
            "a request whose worst-case cost cannot be computed")
    threshold = row.get("long_context_threshold")
    if isinstance(threshold, int) and input_tokens > threshold:
        return float(row["long_context_input"]), float(row["long_context_output"])
    return float(row["input"]), float(row["output"])


def usage_cost(model: str, usage: dict, *, batch: bool,
               pricing_path: Path = _PRICING_PATH) -> float:
    """What a response's own `usage` block says it cost.

    Cache writes bill at 1.25x input and cache reads at 0.1x (5-minute cache;
    pricing page 2026-10-23). These players set no cache_control, so both should
    be zero; they are priced anyway so a future change cannot under-count.
    Output tokens include thinking tokens (they are billed as output).
    """
    def n(key: str) -> int:
        v = usage.get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) else 0
    prompt = n("input_tokens") + n("cache_creation_input_tokens") + n("cache_read_input_tokens")
    inp, out = model_prices(model, pricing_path, input_tokens=prompt)
    usd = (n("input_tokens") * inp
           + n("cache_creation_input_tokens") * inp * 1.25
           + n("cache_read_input_tokens") * inp * 0.1
           + n("output_tokens") * out) / 1_000_000.0
    return usd * (BATCH_FACTOR if batch else 1.0)


def worst_case_cost(model: str, input_tokens: int, max_tokens: int, *, batch: bool,
                    pricing_path: Path = _PRICING_PATH) -> float:
    """Upper bound for one request: every allowed output token is spent."""
    inp, out = model_prices(model, pricing_path, input_tokens=input_tokens)
    usd = (input_tokens * inp + max_tokens * out) / 1_000_000.0
    return usd * (BATCH_FACTOR if batch else 1.0)


@dataclass
class Ledger:
    path: Path
    period: str
    budget_usd: float

    @classmethod
    def from_env(cls) -> "Ledger":
        raw_budget = os.environ.get(BUDGET_ENV, "").strip()
        period = os.environ.get(PERIOD_ENV, "").strip()
        if not raw_budget or not period:
            raise BudgetExceeded(
                f"{BUDGET_ENV} and {PERIOD_ENV} must both be set before any Anthropic "
                f"API call (e.g. {BUDGET_ENV}=150, {PERIOD_ENV}=2026-10-credit). There "
                "is deliberately no default: an unset budget must stop the run, not "
                "allow unlimited spend.")
        try:
            budget = float(raw_budget)
        except ValueError as exc:
            raise BudgetExceeded(f"{BUDGET_ENV}={raw_budget!r} is not a number") from exc
        if budget <= 0:
            raise BudgetExceeded(f"{BUDGET_ENV} must be positive, got {budget}")
        default = Path.home() / ".sciencearena" / "anthropic_spend.jsonl"
        path = Path(os.environ.get(LEDGER_ENV, "").strip() or default)
        return cls(path=path, period=period, budget_usd=budget)

    # -- reading -----------------------------------------------------------
    def _entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                # A corrupt line could be hiding a charge. Refuse rather than
                # under-count spend.
                raise BudgetExceeded(f"unreadable line in spend ledger {self.path}; "
                                     "fix it by hand before spending more")
            if e.get("period") == self.period:
                out.append(e)
        return out

    def spent_usd(self) -> float:
        """Settled charges plus every reservation still open, for this period."""
        state: dict[str, dict] = {}
        for e in self._entries():
            state[e["id"]] = e  # last word on each id wins
        total = 0.0
        for e in state.values():
            if e["kind"] in ("reserve", "settle"):
                total += float(e["usd"])
        return total

    def remaining_usd(self) -> float:
        return self.budget_usd - self.spent_usd()

    # -- writing -----------------------------------------------------------
    def _append(self, kind: str, rid: str, usd: float, detail: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "period": self.period, "kind": kind, "id": rid,
                           "usd": round(float(usd), 6), "detail": detail})
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def reserve(self, rid: str, usd: float, detail: str = "") -> None:
        """Book the worst case, or raise BudgetExceeded without booking anything."""
        remaining = self.remaining_usd()
        if usd > remaining:
            raise BudgetExceeded(
                f"refusing to send {rid}: worst case ${usd:.4f} exceeds the "
                f"${remaining:.4f} left of the ${self.budget_usd:.2f} budget for period "
                f"{self.period!r} (ledger {self.path}). Lower max_tokens, split the "
                "batch, or raise the budget deliberately.")
        self._append("reserve", rid, usd, detail)

    def settle(self, rid: str, usd: float, detail: str = "") -> None:
        """Replace a reservation with the cost the API reported."""
        self._append("settle", rid, usd, detail)

    def release(self, rid: str, detail: str = "") -> None:
        """The API billed nothing for this id (error before generation)."""
        self._append("release", rid, 0.0, detail)
