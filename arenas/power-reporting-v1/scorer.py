"""Scorer for power-reporting-v1 (FIELD-MAP arena).

Compares a player's power-analysis field-map to gold:
  - detection   = 1.0 if has_power_analysis matches gold, else 0.0
  - kind        = kind-classification accuracy WHEN a power analysis is present
                  (1.0 when gold has no power analysis — kind is not applicable)
  - field_f1    = F1 of the extracted {test, sample, alpha, power, effect_size,
                  software} field-map against gold (precision over reported
                  fields, recall over gold fields)
  - calibration = 1 - ECE over the player's confidence on the detection call
  - composite   = mean(detection, kind, field_f1) * calibration

Multiplicative calibration term: a confidently-wrong player is punished. The T2
false-alarm trap is the key discrimination — a methods excerpt that mentions
sample size / alpha but reports NO power analysis must not be flagged as having
one (power_false_alarm), and a correctly-labelled post-hoc analysis must not be
relabelled a-priori (kind_mislabel, the inverse of the headline trap).

error_categories: power_missed (major), power_false_alarm (major),
kind_mislabel (major), field_wrong (minor).
"""
from __future__ import annotations

FIELDS = ["test", "sample", "alpha", "power", "effect_size", "software"]


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


def _norm(v) -> str:
    """Normalise a field value for comparison (case/whitespace-insensitive)."""
    return " ".join(str(v).strip().lower().split())


def score(player_output: dict, ground_truth: dict) -> dict:
    gold_has = bool(ground_truth.get("has_power_analysis", False))
    gold_kind = ground_truth.get("kind")
    gold_fields = ground_truth.get("fields", {}) or {}

    pred_has = bool(player_output.get("has_power_analysis", False))
    pred_kind = player_output.get("kind")
    pred_fields = player_output.get("fields", {}) or {}
    if not isinstance(pred_fields, dict):
        pred_fields = {}

    conf = player_output.get("confidence", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0

    findings: list[dict] = []

    # --- Detection ---------------------------------------------------------
    detection = 1.0 if pred_has == gold_has else 0.0
    if gold_has and not pred_has:
        findings.append({
            "category": "power_missed",
            "anchor": {"field": "has_power_analysis"},
            "evidence": "false",
            "correct_value": True,
        })
    elif (not gold_has) and pred_has:
        findings.append({
            "category": "power_false_alarm",
            "anchor": {"field": "has_power_analysis"},
            "evidence": str(pred_kind),
            "correct_value": False,
        })

    # --- Kind classification ----------------------------------------------
    # Only applicable when a power analysis truly exists. When gold has none,
    # kind is not scored (1.0) so a correct "no PA" call is not double-penalised.
    if gold_has:
        if pred_has:
            kind_score = 1.0 if pred_kind == gold_kind else 0.0
            if pred_kind != gold_kind:
                findings.append({
                    "category": "kind_mislabel",
                    "anchor": {"field": "kind"},
                    "evidence": str(pred_kind),
                    "correct_value": gold_kind,
                })
        else:
            # Missed the analysis entirely — already a power_missed; kind is 0.
            kind_score = 0.0
    else:
        kind_score = 1.0

    # --- Field-map F1 ------------------------------------------------------
    # Precision over reported fields, recall over gold fields. A field counts as
    # correct only if it is present in gold AND the (normalised) value matches.
    correct = 0
    for f in FIELDS:
        in_gold = f in gold_fields
        in_pred = f in pred_fields
        if in_gold and in_pred:
            if _norm(pred_fields[f]) == _norm(gold_fields[f]):
                correct += 1
            else:
                findings.append({
                    "category": "field_wrong",
                    "anchor": {"field": f},
                    "evidence": str(pred_fields[f]),
                    "correct_value": gold_fields[f],
                })
        elif in_gold and not in_pred:
            findings.append({
                "category": "field_wrong",
                "anchor": {"field": f},
                "evidence": "[missing]",
                "correct_value": gold_fields[f],
            })
        elif in_pred and not in_gold:
            # Hallucinated a field the excerpt does not report.
            findings.append({
                "category": "field_wrong",
                "anchor": {"field": f},
                "evidence": str(pred_fields[f]),
                "correct_value": None,
            })

    n_pred = sum(1 for f in FIELDS if f in pred_fields)
    n_gold = len(gold_fields)
    if n_gold == 0 and n_pred == 0:
        field_f1 = 1.0  # no fields to extract, none invented — perfect.
    else:
        precision = correct / n_pred if n_pred > 0 else (1.0 if n_gold == 0 else 0.0)
        recall = correct / n_gold if n_gold > 0 else (1.0 if n_pred == 0 else 0.0)
        field_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # --- Calibration -------------------------------------------------------
    calibration = 1.0 - _ece([(conf, pred_has == gold_has)])

    accuracy = (detection + kind_score + field_f1) / 3.0
    composite = accuracy * calibration

    return {
        "primary": composite,
        "breakdown": {
            "detection": detection,
            "kind": kind_score,
            "field_f1": field_f1,
            "calibration": calibration,
            "accuracy": accuracy,
            "composite": composite,
            "n_gold_fields": n_gold,
            "n_pred_fields": n_pred,
            "n_fields_correct": correct,
        },
        "findings": findings,
    }
