"""Scorer for grim-consistency-v1.

Matches the player's per-statistic granularity-consistency judgements to gold by
stat_id and computes:
  - detection precision/recall/F1 on flagged vs not-flagged
  - kind_accuracy = of correctly-flagged statistics, fraction with the right issue_kind
  - calibration   = 1 - ECE over the player's confidence
  - composite     = detection_f1 * kind_accuracy * calibration

The T2 controls tier is the key trap: values that look odd (small N, many decimals,
arbitrary-looking percentages) are nonetheless GRIM-consistent. Flagging them shows
up as grim_false_alarm and tanks precision.

Findings categories (must match arena.yaml#error_categories):
  - grim_missed       (major)  a GRIM-inconsistent mean or percentage was not flagged
  - grim_false_alarm  (major)  a consistent value was flagged
  - kind_mislabel     (minor)  a correctly-flagged statistic got the wrong issue_kind
                               — e.g. an impossible percentage labelled
                               grim_inconsistent instead of grim_percent_inconsistent
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
    gold = ground_truth.get("records", [])
    gold_by_id = {g["stat_id"]: g for g in gold}
    preds = player_output.get("records", [])

    findings: list[dict] = []
    pred_by_id: dict[str, dict] = {}
    for p in preds:
        sid = p.get("stat_id")
        if sid in gold_by_id:
            pred_by_id[sid] = p  # last write wins on duplicates
        else:
            # A record for a stat id the gold doesn't know. If the player flagged
            # it, that's a grim_false_alarm on a phantom statistic.
            if bool(p.get("flagged")):
                findings.append({
                    "category": "grim_false_alarm",
                    "anchor": {"stat_id": sid},
                    "evidence": str(p.get("issue_kind")),
                })

    tp = fp = fn = 0
    kind_correct = 0
    kind_total = 0
    calib: list[tuple[float, bool]] = []

    for sid, g in gold_by_id.items():
        truth = bool(g["flagged"])
        p = pred_by_id.get(sid)
        if p is None:
            # No record at all for this statistic. If it was truly inconsistent that
            # is a miss; if consistent, a missing record on a clean stat is harmless
            # (true negative).
            if truth:
                fn += 1
                findings.append({
                    "category": "grim_missed",
                    "anchor": {"stat_id": sid},
                    "correct_value": g.get("issue_kind"),
                })
            continue

        pred_flag = bool(p.get("flagged"))
        conf = p.get("confidence", 0.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        calib.append((conf, pred_flag == truth))

        if truth and pred_flag:
            tp += 1
            kind_total += 1
            if p.get("issue_kind") == g.get("issue_kind"):
                kind_correct += 1
            else:
                findings.append({
                    "category": "kind_mislabel",
                    "anchor": {"stat_id": sid},
                    "evidence": str(p.get("issue_kind")),
                    "correct_value": g.get("issue_kind"),
                })
        elif truth and not pred_flag:
            fn += 1
            findings.append({
                "category": "grim_missed",
                "anchor": {"stat_id": sid},
                "correct_value": g.get("issue_kind"),
            })
        elif (not truth) and pred_flag:
            fp += 1
            findings.append({
                "category": "grim_false_alarm",
                "anchor": {"stat_id": sid},
                "evidence": str(p.get("issue_kind")),
            })
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
            "n_gold_statistics": len(gold_by_id),
            "n_inconsistent": sum(1 for g in gold_by_id.values() if g["flagged"]),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "findings": findings,
    }
