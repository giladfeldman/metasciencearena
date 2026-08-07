"""Scorer for significance-language-v1.

Matches player-flagged spans to gold spans and computes:
  - detection precision/recall/F1 on span localisation (a predicted flag matches a
    gold flag when their character ranges overlap).
  - category_accuracy = of correctly-localised flags, fraction with the right
    category.
  - calibration       = 1 - ECE over the player's per-flag confidence.
  - composite (primary) = f1 * calibration.

The T2 false-alarm trap is the key discriminator: the excerpt is built entirely
from clean controls (legitimate hedging, exact-significant claims, randomised
causal claims), so any flag is a false alarm that tanks precision -> F1 -> composite.

error_categories (arena.yaml): flag_missed (major), flag_false_alarm (major),
category_mislabel (minor).
"""
from __future__ import annotations


def _ece(predictions: list[tuple[float, bool]], n_bins: int = 5) -> float:
    """Expected Calibration Error. predictions = [(confidence, is_correct), ...]."""
    if not predictions:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
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


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """True if [a_start,a_end) and [b_start,b_end) share at least one character."""
    return a_start < b_end and b_start < a_end


def _as_span(obj: dict) -> tuple[int | None, int | None]:
    span = obj.get("span") or {}
    if not isinstance(span, dict):
        return None, None
    try:
        return int(span.get("char_start")), int(span.get("char_end"))
    except (TypeError, ValueError):
        return None, None


def score(player_output: dict, ground_truth: dict) -> dict:
    gold = ground_truth.get("flags", [])
    preds = player_output.get("flags", [])

    findings: list[dict] = []

    # Greedy one-to-one matching of predicted flags to gold flags by span overlap.
    gold_matched = [False] * len(gold)
    pred_matched = [False] * len(preds)
    calib: list[tuple[float, bool]] = []

    tp = 0
    category_correct = 0
    category_total = 0

    for pi, p in enumerate(preds):
        p_start, p_end = _as_span(p)
        conf = p.get("confidence", 0.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        if p_start is None or p_end is None:
            # Malformed span: counts as a false alarm (can't localise it).
            continue
        best = None
        for gi, g in enumerate(gold):
            if gold_matched[gi]:
                continue
            g_start, g_end = _as_span(g)
            if g_start is None or g_end is None:
                continue
            if _overlaps(p_start, p_end, g_start, g_end):
                best = gi
                break
        if best is not None:
            gold_matched[best] = True
            pred_matched[pi] = True
            tp += 1
            category_total += 1
            g = gold[best]
            if p.get("category") == g.get("category"):
                category_correct += 1
                calib.append((conf, True))
            else:
                # Localised correctly but mislabelled the category.
                calib.append((conf, False))
                findings.append({
                    "category": "category_mislabel",
                    "anchor": {"char_start": p_start, "char_end": p_end},
                    "evidence": str(p.get("category")),
                    "correct_value": g.get("category"),
                })

    # Unmatched predictions are false alarms.
    for pi, p in enumerate(preds):
        if pred_matched[pi]:
            continue
        p_start, p_end = _as_span(p)
        conf = p.get("confidence", 0.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        calib.append((conf, False))
        anchor = None
        if p_start is not None and p_end is not None:
            anchor = {"char_start": p_start, "char_end": p_end}
        findings.append({
            "category": "flag_false_alarm",
            "anchor": anchor,
            "evidence": str(p.get("category")),
        })

    # Unmatched gold flags are misses.
    for gi, g in enumerate(gold):
        if gold_matched[gi]:
            continue
        g_start, g_end = _as_span(g)
        anchor = None
        if g_start is not None and g_end is not None:
            anchor = {"char_start": g_start, "char_end": g_end}
        findings.append({
            "category": "flag_missed",
            "anchor": anchor,
            "correct_value": g.get("category"),
        })

    fp = sum(1 for m in pred_matched if not m)
    fn = sum(1 for m in gold_matched if not m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    category_accuracy = category_correct / category_total if category_total > 0 else 1.0
    calibration = 1.0 - _ece(calib) if calib else 1.0
    composite = f1 * calibration

    return {
        "primary": composite,
        "breakdown": {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "category_accuracy": category_accuracy,
            "calibration": calibration,
            "composite": composite,
            "n_gold_flags": len(gold),
            "n_pred_flags": len(preds),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "findings": findings,
    }
