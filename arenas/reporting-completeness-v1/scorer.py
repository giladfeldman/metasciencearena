"""Scorer for reporting-completeness-v1.

Matches player flags to gold flags by CATEGORY + SPAN OVERLAP (containment OR
IoU >= 0.5) and computes:
  - detection precision / recall / F1 over flagged defects
  - calibration = 1 - ECE over the player's per-flag confidence
  - composite   = detection_f1 * calibration

The T2 false-alarm trap is the key discrimination: a paragraph where every test
is reported completely and precisely must yield ZERO flags. Flagging a clean
report shows up as flag_false_alarm and tanks precision.

Findings (must match arena.yaml#error_categories):
  flag_missed       (major) - a gold defect the player did not flag (false neg)
  flag_false_alarm  (major) - a player flag with no matching gold defect (false pos)
  category_mislabel (minor) - player flag overlaps a gold defect but wrong category
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


def _bounds(span: dict) -> tuple[int, int]:
    try:
        a = int(span.get("char_start"))
        b = int(span.get("char_end"))
    except (TypeError, ValueError):
        return (0, -1)  # empty / invalid
    return (a, b) if b >= a else (b, a)


def _overlaps(a: dict, b: dict) -> bool:
    """True if spans overlap by containment OR IoU >= 0.5."""
    a0, a1 = _bounds(a)
    b0, b1 = _bounds(b)
    if a1 <= a0 or b1 <= b0:
        return False
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return False
    len_a = a1 - a0
    len_b = b1 - b0
    # Containment: one span fully inside the other.
    if inter == len_a or inter == len_b:
        return True
    union = len_a + len_b - inter
    iou = inter / union if union > 0 else 0.0
    return iou >= 0.5


def score(player_output: dict, ground_truth: dict) -> dict:
    gold_flags = list(ground_truth.get("flags", []))
    preds = list(player_output.get("flags", []))

    findings: list[dict] = []
    gold_matched = [False] * len(gold_flags)
    calib: list[tuple[float, bool]] = []

    tp = 0          # flag overlaps a gold defect AND category matches
    fp = 0          # flag with no overlapping gold defect, or wrong category
    category_mislabels = 0

    for p in preds:
        p_span = p.get("span", {}) or {}
        p_cat = p.get("category")
        conf = p.get("confidence", 0.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0

        # Find the best unmatched gold flag this prediction overlaps.
        match_idx = -1
        for gi, g in enumerate(gold_flags):
            if gold_matched[gi]:
                continue
            if _overlaps(p_span, g.get("span", {}) or {}):
                match_idx = gi
                break

        if match_idx < 0:
            # No overlapping gold defect => false alarm.
            fp += 1
            calib.append((conf, False))
            findings.append({
                "category": "flag_false_alarm",
                "anchor": {"char_start": _bounds(p_span)[0], "char_end": _bounds(p_span)[1]},
                "evidence": str(p_span.get("text", "")),
            })
            continue

        g = gold_flags[match_idx]
        gold_matched[match_idx] = True
        if p_cat == g.get("category"):
            tp += 1
            calib.append((conf, True))
        else:
            # Overlap but wrong category: counts against precision AND recall is
            # not credited (gold defect remains effectively unmatched in label).
            fp += 1
            category_mislabels += 1
            calib.append((conf, False))
            findings.append({
                "category": "category_mislabel",
                "anchor": {"char_start": _bounds(g.get("span", {}))[0],
                           "char_end": _bounds(g.get("span", {}))[1]},
                "evidence": str(p_cat),
                "correct_value": str(g.get("category")),
            })

    # Any gold defect never correctly matched (by category) is a miss.
    fn = 0
    for gi, g in enumerate(gold_flags):
        if not gold_matched[gi]:
            fn += 1
            findings.append({
                "category": "flag_missed",
                "anchor": {"char_start": _bounds(g.get("span", {}))[0],
                           "char_end": _bounds(g.get("span", {}))[1]},
                "correct_value": str(g.get("category")),
            })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    detection_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    calibration = 1.0 - _ece(calib) if calib else 1.0
    composite = detection_f1 * calibration

    return {
        "primary": composite,
        "breakdown": {
            "detection_f1": detection_f1,
            "precision": precision,
            "recall": recall,
            "calibration": calibration,
            "composite": composite,
            "n_gold_flags": len(gold_flags),
            "n_pred_flags": len(preds),
            "tp": tp, "fp": fp, "fn": fn,
            "category_mislabels": category_mislabels,
        },
        "findings": findings,
    }
