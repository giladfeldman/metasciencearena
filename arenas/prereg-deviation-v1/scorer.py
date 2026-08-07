"""Scorer for prereg-deviation-v1.

Matches player per-dimension deviation judgements to gold by dimension id and
computes:
  - detection precision/recall/F1 on deviation vs no-deviation
  - kind_accuracy  = of correctly-flagged deviations, fraction with the right kind
  - calibration    = 1 - ECE over the player's confidence
  - composite      = detection_f1 * kind_accuracy * calibration

The T2 paraphrase tier is the key trap: rewording is NOT a deviation, so flagging
it shows up as deviation_false_alarm and tanks precision.
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


def score(player_output: dict, ground_truth: dict) -> dict:
    gold = ground_truth.get("dimensions", [])
    gold_by_dim = {g["dimension"]: g for g in gold}
    preds = player_output.get("deviations", [])

    findings: list[dict] = []
    pred_by_dim: dict[str, dict] = {}
    for p in preds:
        dim = p.get("dimension")
        if dim in gold_by_dim:
            pred_by_dim[dim] = p  # last write wins on duplicates
        else:
            findings.append({"category": "unknown_dimension",
                             "anchor": {"dimension": dim}, "evidence": str(dim)})

    tp = fp = fn = 0
    kind_correct = 0
    kind_total = 0
    calib: list[tuple[float, bool]] = []

    for dim, g in gold_by_dim.items():
        truth = bool(g["deviation"])
        p = pred_by_dim.get(dim)
        if p is None:
            findings.append({"category": "dimension_missed", "anchor": {"dimension": dim}})
            if truth:
                fn += 1
                findings.append({"category": "deviation_missed",
                                 "anchor": {"dimension": dim},
                                 "correct_value": g["deviation_kind"]})
            continue

        pred_dev = bool(p.get("deviation"))
        conf = p.get("confidence", 0.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        calib.append((conf, pred_dev == truth))

        if truth and pred_dev:
            tp += 1
            kind_total += 1
            if p.get("deviation_kind") == g["deviation_kind"]:
                kind_correct += 1
            else:
                findings.append({"category": "kind_mislabel",
                                 "anchor": {"dimension": dim},
                                 "evidence": str(p.get("deviation_kind")),
                                 "correct_value": g["deviation_kind"]})
        elif truth and not pred_dev:
            fn += 1
            findings.append({"category": "deviation_missed",
                             "anchor": {"dimension": dim},
                             "correct_value": g["deviation_kind"]})
        elif (not truth) and pred_dev:
            fp += 1
            findings.append({"category": "deviation_false_alarm",
                             "anchor": {"dimension": dim},
                             "evidence": str(p.get("deviation_kind"))})
        # else: true negative — correctly left unflagged.

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    detection_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    kind_accuracy = kind_correct / kind_total if kind_total > 0 else 1.0
    calibration = 1.0 - _ece(calib) if calib else 1.0
    composite = detection_f1 * kind_accuracy * calibration

    return {
        "primary": composite,
        "breakdown": {
            "detection_f1": detection_f1,
            "precision": precision,
            "recall": recall,
            "kind_accuracy": kind_accuracy,
            "calibration": calibration,
            "composite": composite,
            "n_gold_dimensions": len(gold_by_dim),
            "n_deviating": sum(1 for g in gold_by_dim.values() if g["deviation"]),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "findings": findings,
    }
