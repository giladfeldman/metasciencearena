"""Scorer for citation-claim-fidelity-v1: guarded, evidence-scoped item accuracy.

Per sentence, the player's verdict is GUARDED exactly as the citeclaim library
guards it before a user sees it: a "consistent" or "contradicts" verdict whose
evidence quote is not found in the abstract counts as NO verdict. "silent" needs
no quote. Quote matching follows one spec shared with citeclaim's locateQuote and
pinned by tests/quote_parity.json (the same file, same sha256, in both repos).

Item ids must be exactly the sentence numbers (a JSON integer, or a digit string
without a leading zero). Any other id, or an id outside 1..n, makes the WHOLE
answer unattributable, so every item is no verdict: a shifted attribution would
score verdicts against the wrong sentences (consult 2026-10-14).

Correct when the guarded verdict equals the gold verdict; for gold
"not_consistent" (a misrepresented finding, a wrong paper) either "contradicts"
or "silent" is correct. For gold "not_contradicts" (task set v3, a sentence human coders
found substantiated by the source's FULL text, so its abstract may support it or be silent)
either "consistent" or "silent" is correct. No verdict is never correct.

score.primary = share of the task's sentences judged correctly.
"""
from __future__ import annotations

import re
import unicodedata

FORMULA_ID = "ccf-guarded-item-accuracy-v1"
VERDICTS = ("consistent", "contradicts", "silent")

#: Explicit whitespace list, identical to citeclaim's (Python's and JavaScript's
#: \s disagree on U+FEFF and U+001C..U+001F, so neither regex class is used).
WHITESPACE = frozenset([0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680,
                        *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF])
SINGLE_QUOTES = frozenset([0x2018, 0x2019, 0x201A, 0x201B, 0x2032])
DOUBLE_QUOTES = frozenset([0x201C, 0x201D, 0x201E, 0x201F, 0x2033])
TRIM = " \"'.,;:"


def _map_code_point(ch: str) -> str:
    cp = ord(ch)
    if cp in WHITESPACE:
        return " "
    if cp in SINGLE_QUOTES:
        return "'"
    if cp in DOUBLE_QUOTES:
        return '"'
    if 0x2010 <= cp <= 0x2015 or cp == 0x2212:
        return "-"
    return ch.lower()  # per code point: no context rules (e.g. final sigma), as in citeclaim


def normalize_for_match(s: str) -> str:
    """citeclaim normalizeForMatch: NFKC, per-code-point mapping, spaces collapsed, trimmed."""
    out: list[str] = []
    prev_space = True
    for ch in unicodedata.normalize("NFKC", s):
        m = _map_code_point(ch)
        if m == " ":
            if not prev_space:
                out.append(" ")
                prev_space = True
        else:
            prev_space = False
            out.append(m)
    if out and out[-1] == " ":
        out.pop()
    return "".join(out)


def quote_pieces(quote: str) -> list[str]:
    pieces = [normalize_for_match(p).strip(TRIM) for p in re.split(r"\.\.\.|…", quote)]
    return [p for p in pieces if p]


def quote_in_abstract(abstract: str, quote) -> bool:
    """citeclaim locateQuote as a yes/no. A non-string quote never matches."""
    if not isinstance(quote, str):
        return False
    pieces = quote_pieces(quote)
    if not pieces or any(len(p) < 3 for p in pieces):  # len() counts code points, as citeclaim does
        return False
    norm = normalize_for_match(abstract)
    pos = 0
    for p in pieces:
        at = norm.find(p, pos)
        if at < 0:
            return False
        pos = at + len(p)
    return True


_DIGITS = re.compile(r"[1-9][0-9]{0,5}")


def parse_item_id(v) -> int | None:
    """citeclaim parseItemId: a JSON integer (never a bool), or a digit string without a leading zero."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():  # JSON 1.0 is the number 1 in JavaScript too
        return int(v)
    if isinstance(v, str) and _DIGITS.fullmatch(v):
        return int(v)
    return None


def guarded_verdicts(output: dict, n_items: int, abstract: str) -> list[str | None]:
    items = output.get("items") if isinstance(output, dict) else None
    if not isinstance(items, list):
        return [None] * n_items
    by_id: dict[int, dict] = {}
    dup: set[int] = set()
    for it in items:
        k = parse_item_id(it.get("id")) if isinstance(it, dict) else None
        if k is None or not 1 <= k <= n_items:
            return [None] * n_items
        if k in by_id:
            dup.add(k)
        by_id[k] = it
    out: list[str | None] = []
    for k in range(1, n_items + 1):
        it = by_id.get(k)
        v = it.get("verdict") if it is not None and k not in dup else None
        v = v.strip().lower() if isinstance(v, str) else None
        if v not in VERDICTS:
            out.append(None)
        elif v == "silent":
            out.append("silent")
        else:
            out.append(v if quote_in_abstract(abstract, it.get("evidence_quote")) else None)
    return out


def is_correct(gold: str, verdict: str | None) -> bool:
    if verdict is None:
        return False
    if gold == "not_consistent":
        return verdict in ("contradicts", "silent")
    if gold == "not_contradicts":
        return verdict in ("consistent", "silent")
    return verdict == gold


def _error_category(gold: str, verdict: str | None) -> str | None:
    if is_correct(gold, verdict):
        return None
    if verdict is None:
        return "no_verdict"
    if gold == "consistent":
        return "false_alarm_on_clean" if verdict == "contradicts" else "wrong_silent"
    if gold == "contradicts":
        return "missed_contradiction"
    if gold == "not_consistent":
        return "missed_not_consistent"
    if gold == "silent":
        return "overclaimed_support" if verdict == "consistent" else "false_alarm_on_clean"
    if gold == "not_contradicts":
        return "false_alarm_on_clean"
    return "no_verdict"


def score(output: dict, ground_truth: dict) -> dict:
    gold_items = ground_truth["items"]
    verdicts = guarded_verdicts(output, len(gold_items), ground_truth["abstract"])
    correct = [is_correct(g["gold"], v) for g, v in zip(gold_items, verdicts)]
    findings: dict[str, int] = {}
    for g, v in zip(gold_items, verdicts):
        cat = _error_category(g["gold"], v)
        if cat:
            findings[cat] = findings.get(cat, 0) + 1
    n = len(gold_items)
    return {
        "primary": sum(correct) / n if n else 0.0,
        "breakdown": {"n_items": n, "n_correct": sum(correct),
                      "n_no_verdict": sum(1 for v in verdicts if v is None)},
        "findings": [{"category": c, "count": k} for c, k in sorted(findings.items())],
    }
