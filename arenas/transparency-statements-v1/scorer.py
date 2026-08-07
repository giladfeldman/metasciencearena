"""Scorer for transparency-statements-v1.

Compares the player's extracted transparency field map to gold and computes:
  - field_accuracy = macro accuracy over the present/absent (and on_request) flags
                     across the six transparency fields.
  - field_f1       = F1 of statement-present vs absent across fields. The T2
                     paraphrase tier is the trap here: a reworded-but-real
                     statement is still PRESENT — flagging extra (or dropping real)
                     statements tanks precision / recall.
  - url_judgement  = fraction of URL availability claims judged correctly: a real
                     repository link must be marked available; an "available on
                     request" hedge or a placeholder/broken URL must NOT be.
  - calibration    = 1 - ECE over the player's overall confidence.
  - composite      = field_accuracy * url_judgement * calibration.

Findings (categories MUST match arena.yaml#error_categories):
  - statement_missed         (major) — a present statement was reported absent.
  - statement_false_positive (major) — an absent statement was reported present.
  - url_misjudged            (minor) — a real link treated as unavailable, or an
                             on-request/placeholder claim treated as a real repo.
"""
from __future__ import annotations

CLAIM_FIELDS = ["coi", "funding", "data", "code", "materials", "prereg"]
OPEN_FIELDS = ["data", "code", "materials", "prereg"]


def _ece(predictions, n_bins: int = 5) -> float:
    """Expected Calibration Error. predictions = [(confidence, is_correct), ...]."""
    if not predictions:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for conf, correct in predictions:
        conf = min(max(conf, 0.0), 1.0)
        idx = min(n_bins - 1, int(conf * n_bins))
        bins[idx].append((conf, correct))
    total = len(predictions)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(1 for _, ok in b if ok) / len(b)
        ece += (len(b) / total) * abs(avg_conf - avg_acc)
    return ece


def _gold_has_claim(field: str, g: dict) -> bool:
    """Did the manuscript make ANY transparency claim for this field (present)?"""
    if field in ("coi", "funding"):
        return bool(g.get("present"))
    if field == "prereg":
        return bool(g.get("available")) or g.get("url") is not None
    return bool(g.get("available")) or bool(g.get("on_request")) or g.get("url") is not None


def _pred_has_claim(field: str, p: dict) -> bool:
    if field in ("coi", "funding"):
        return bool(p.get("present"))
    if field == "prereg":
        return bool(p.get("available")) or p.get("url") is not None
    return bool(p.get("available")) or bool(p.get("on_request")) or p.get("url") is not None


def _field_flags_match(field: str, g: dict, p: dict) -> bool:
    """Exact match of the truth-bearing flags for a field (drives field_accuracy)."""
    if field in ("coi", "funding"):
        return bool(g.get("present")) == bool(p.get("present"))
    if field == "prereg":
        return bool(g.get("available")) == bool(p.get("available"))
    return (bool(g.get("available")) == bool(p.get("available"))
            and bool(g.get("on_request")) == bool(p.get("on_request")))


def score(player_output: dict, ground_truth: dict) -> dict:
    findings: list[dict] = []

    # --- field accuracy (macro over flag-correctness) ---
    correct = 0
    for field in CLAIM_FIELDS:
        g = ground_truth.get(field, {})
        p = player_output.get(field, {}) or {}
        if _field_flags_match(field, g, p):
            correct += 1
    field_accuracy = correct / len(CLAIM_FIELDS)

    # --- statement present/absent F1 (T2 false-alarm trap lives here) ---
    tp = fp = fn = 0
    for field in CLAIM_FIELDS:
        g = ground_truth.get(field, {})
        p = player_output.get(field, {}) or {}
        gold_claim = _gold_has_claim(field, g)
        pred_claim = _pred_has_claim(field, p)
        if gold_claim and pred_claim:
            tp += 1
        elif gold_claim and not pred_claim:
            fn += 1
            findings.append({"category": "statement_missed",
                             "anchor": {"field": field},
                             "correct_value": "present"})
        elif (not gold_claim) and pred_claim:
            fp += 1
            findings.append({"category": "statement_false_positive",
                             "anchor": {"field": field},
                             "evidence": str(p.get("statement") or p.get("url") or "claimed present")})
        # else: true negative.

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    field_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # --- URL judgement on the open-practices fields ---
    url_total = 0
    url_correct = 0
    for field in OPEN_FIELDS:
        g = ground_truth.get(field, {})
        p = player_output.get(field, {}) or {}
        # Only fields where the manuscript made an availability-style claim are
        # scored for url judgement (real link, on-request hedge, or placeholder).
        gold_made_url_claim = _gold_has_claim(field, g)
        if not gold_made_url_claim:
            continue
        url_total += 1
        gold_available = bool(g.get("available"))
        pred_available = bool(p.get("available"))
        if gold_available == pred_available:
            url_correct += 1
        else:
            findings.append({"category": "url_misjudged",
                             "anchor": {"field": field},
                             "evidence": str(p.get("url")),
                             "correct_value": "available" if gold_available else "not_a_real_repo"})
    url_judgement = url_correct / url_total if url_total > 0 else 1.0

    # --- calibration over the player's overall confidence ---
    conf = player_output.get("confidence", None)
    calib: list[tuple[float, bool]] = []
    if conf is not None:
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        # One calibration point per field: was that field's flag set correct?
        for field in CLAIM_FIELDS:
            g = ground_truth.get(field, {})
            p = player_output.get(field, {}) or {}
            calib.append((conf, _field_flags_match(field, g, p)))
    calibration = 1.0 - _ece(calib) if calib else 1.0

    composite = field_accuracy * url_judgement * calibration

    return {
        "primary": composite,
        "breakdown": {
            "field_accuracy": field_accuracy,
            "field_f1": field_f1,
            "precision": precision,
            "recall": recall,
            "url_judgement": url_judgement,
            "calibration": calibration,
            "composite": composite,
            "n_fields": len(CLAIM_FIELDS),
            "n_fields_correct": correct,
            "tp": tp, "fp": fp, "fn": fn,
            "url_total": url_total, "url_correct": url_correct,
        },
        "findings": findings,
    }
