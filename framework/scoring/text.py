"""Tokenization + text-similarity helpers shared by multiple arenas.

`pdf-text-fidelity-v1` carries its own private `_normalize.py` (kept for
back-compat). New arenas (`pdf-section-structure-v1`,
`pdf-table-extraction-v1`) consume these helpers directly so the same
token-F1 / Levenshtein math is applied across the PDF-extraction family.

The helpers are intentionally tiny and dependency-light:
- rapidfuzz is used for Levenshtein when available, with a pure-Python
  fallback so tests still pass on machines that haven't installed it.
"""
from __future__ import annotations

import string
import unicodedata
from collections import Counter

_PUNCT = str.maketrans("", "", string.punctuation)


def tokenize(text: str) -> list[str]:
    """Lowercase, NFC, strip ASCII punctuation, split on whitespace."""
    text = unicodedata.normalize("NFC", text).lower()
    text = text.translate(_PUNCT)
    return [t for t in text.split() if t]


def token_f1(a: str, b: str) -> float:
    """Multiset token-F1 in [0, 1]. Both empty -> 1.0; one empty -> 0.0."""
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(tb)
    recall = overlap / len(ta)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# Above this DP-matrix size (len(a) * len(b)) the pure-Python fallback is
# intractable: a real paper's full body (~35k chars) vs an extractor's output
# (~40k chars) is ~1.4 billion cells and effectively hangs forever. rapidfuzz's
# C implementation does the same in ~50ms, so it is a hard project dependency
# (pyproject.toml). If it is somehow absent we FAIL FAST with a clear message
# (the runner records it as a soft adapter error) rather than hang the whole
# tournament — the failure mode that masked the full-body gold rebuild on
# 2026-06-12 when the .venv lacked rapidfuzz. This guard MUST stay in sync with
# arenas/pdf-text-fidelity-v1/_normalize.py (which now delegates here). DR-0001.
_PUREPY_LEVENSHTEIN_MAX_CELLS = 4_000_000


def levenshtein_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]; identical strings -> 1.0.

    Uses rapidfuzz's C implementation when available. The pure-Python fallback
    is correct but O(len(a)*len(b)); above ``_PUREPY_LEVENSHTEIN_MAX_CELLS`` it
    raises ``RuntimeError`` instead of hanging (rapidfuzz is the declared, fast
    path — install it). Note we catch only ``ImportError`` around the rapidfuzz
    import, so a genuine ``RuntimeError`` from the guard is NOT swallowed.
    """
    if a == b:
        return 1.0
    if not a and not b:
        return 1.0
    try:
        from rapidfuzz.distance import Levenshtein  # type: ignore
    except ImportError:
        Levenshtein = None
    if Levenshtein is not None:
        return Levenshtein.normalized_similarity(a, b)

    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    if m * n > _PUREPY_LEVENSHTEIN_MAX_CELLS:
        raise RuntimeError(
            f"levenshtein_similarity: inputs too large for the pure-Python fallback "
            f"({m}x{n} = {m * n:,} cells > {_PUREPY_LEVENSHTEIN_MAX_CELLS:,}). "
            f"Install rapidfuzz (a declared dependency) — `pip install rapidfuzz`."
        )
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    edit = prev[n]
    return 1.0 - (edit / max(m, n))


def normalize_heading(text: str) -> str:
    """Normalize a heading string for set-based comparison.

    Strips leading numbering ("1.", "1.1", "I.", "A."), lowercases,
    collapses whitespace, drops trailing colons/punctuation. Used by
    section-structure scorers to compare heading_text across player and
    gold without penalizing format differences.
    """
    text = unicodedata.normalize("NFC", text).strip()
    # Strip common heading numbering prefixes
    import re
    text = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[IVXLCDM]+\.|[A-Z]\.)\s+", "", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(":. ")
    return text
