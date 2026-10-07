"""Scorer for stats-extraction-v1.

Player extractions are matched to ground-truth items by span overlap
(containment, else IoU >= 0.5) plus `kind` agreement, via greedy
maximum-overlap assignment. Scoring then has TWO HEADS, because the arena asks
two different questions and used to fuse them into one boolean:

  EXTRACTION   — what does the page say?
    precision = TP / (TP + FP), recall = TP / (TP + FN), where a match is a TP
    iff the player's number is within 0.01 of the printed one. Blind to verdicts.

  VERIFICATION — is what it says coherent?
    accuracy over ANSWERABLE checkable matched items, against the
    `expected_verdict` gold derives by recomputation. An "unverifiable" answer is
    never counted as an accusation — but since 2026-08-27 it is not free either:
    declining an item the gold can answer stays in the denominator as a non-hit.
    Only a genuinely AMBIGUOUS item (`coherent=False`) leaves it. See
    `_verification_cell` for the three-way split and the measurement that forced
    it: while all three shared one cell, answering "unverifiable" to everything
    scored a perfect 1.0000 and was the single best strategy in the arena.

  composite = precision * recall * verification_score

`calibration` (1 - ECE) is still computed and reported, but is NOT in the
composite. For every deterministic tool player it measured this framework rather
than the tool: `framework/player_adapter.py` derived `confidence` mechanically
from the suspicion flag (`0.4 if suspicious else 1.0`), so ESCImate's confidence
across 101 extractions was exactly {0.4 x53, 1.0 x48}. See the block above
`_player_verdict` for the full history.
"""
from __future__ import annotations

#: Names the scoring FORMULA this module implements, stamped into every record's
#: `instrument` block. Two records with different formula_id are not comparable and
#: must never be averaged into one leaderboard row. On 2026-08-22 this arena moved
#: from `precision * recall * calibration` to `precision * recall * verification`
#: and the records gave a reader no way to tell which one produced a number.
#: Bump the suffix when the FORMULA changes; never when the code merely moves.
#:
#: -v2 (2026-08-27). The product is unchanged (P x R x V) but V is not, so the
#: same output against the same gold can now score differently and the two are
#: NOT comparable. Two changes, both in the verification head: a player that says
#: "unverifiable" about an item the gold CAN answer now stays in the denominator
#: as a non-hit instead of leaving it (declining everything scored a perfect
#: 1.0), and `expected_verdict == "unverifiable"` gained its own branch instead
#: of falling through to `inconsistent` (a correct "unverifiable" earned nothing
#: while a false accusation was credited as a hit).
#:
#: -v3 (2026-08-29). The -v2 repair covered only COHERENT items: a refusal aimed
#: at an incoherent one still reached the `abstain` cell, still left the
#: denominator, and on a task where EVERY checkable item is incoherent still
#: emptied `v_total` into the neutral 1.0. That is the -v2 defect one branch over.
#: `_verification_cell` no longer consults coherence for a refusal.
#:
#: MEASURED before the bump, so this is a fact rather than a precaution: every
#: revealed record in this arena was rescored under both scorers, and 4 of 612
#: move -- 2 on v2, 2 on v3, and ZERO on v4 or v5, whose task sets contain no
#: incoherent items at all. A bump is still correct: the same stored formula_id
#: would otherwise cover two functions that disagree on real records, which is
#: exactly the ambiguity this field exists to remove.
FORMULA_ID = "stats-extraction-PxRxV-v3"


def _safe_conf(extraction: dict) -> float:
    """Read a player extraction's confidence, defaulting to 0.0.

    Players (especially LLM ones) frequently omit or malform `confidence`; a bare
    `float(extraction["confidence"])` would raise and the runner would record the
    whole task as an adapter error instead of degrading gracefully. Mirrors the
    grim scorer's defensive read (DR-0004).
    """
    try:
        return float(extraction["confidence"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _match_score(gt_span: dict, ext_span: dict) -> float:
    """How well does an extracted span localize a GT anchor?

    GT spans in this arena are short value anchors (e.g. "5.34"); player
    extractions often quote a longer phrase. So pure IoU at threshold 0.5 is
    too strict. Score 1.0 if either span fully contains the other, otherwise
    fall back to IoU. Returns 0 if either span is degenerate (zero length).
    """
    a_lo, a_hi = gt_span["char_start"], gt_span["char_end"]
    b_lo, b_hi = ext_span["char_start"], ext_span["char_end"]
    if a_hi <= a_lo or b_hi <= b_lo:
        return 0.0
    if b_lo <= a_lo and b_hi >= a_hi:
        return 1.0
    if a_lo <= b_lo and a_hi >= b_hi:
        return 1.0
    inter = max(0, min(a_hi, b_hi) - max(a_lo, b_lo))
    union = max(a_hi, b_hi) - min(a_lo, b_lo)
    return inter / union if union > 0 else 0.0


def _greedy_match(gt_items: list[dict], extractions: list[dict], match_thresh: float = 0.5) -> list[tuple[int, int]]:
    """Return list of (gt_idx, ext_idx) pairs, greedy by descending match score."""
    pairs: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_items):
        for ei, ext in enumerate(extractions):
            if gt["kind"] != ext["kind"]:
                continue
            score = _match_score(gt["span"], ext["span"])
            if score >= match_thresh:
                pairs.append((score, gi, ei))
    pairs.sort(reverse=True)
    used_gt: set[int] = set()
    used_ext: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gi, ei in pairs:
        if gi in used_gt or ei in used_ext:
            continue
        used_gt.add(gi); used_ext.add(ei)
        matches.append((gi, ei))
    return matches


_VALUE_SYNONYMS = ("value", "stat_value", "statistic", "test_value", "effect_reported", "effect")


def _ext_value(ext: dict):
    f = ext.get("fields", {})
    for k in _VALUE_SYNONYMS:
        v = f.get(k)
        if v is not None:
            return v
    return None


def _is_correct(gt_item: dict, ext: dict) -> bool:
    """EXTRACTION correctness only: did the player read the printed number?

    Deliberately blind to the verdict. Before 2026-08-21 a deception item was
    "correct" iff the player set one boolean, which fused two different questions
    — *what does the page say* and *is what it says coherent* — into a single
    score. They are now two heads: this one, and `_verification_score` below.

    A deception item is still an extraction target: the fabricated statistic is
    printed on the page, and a reader who transcribes it accurately has done the
    extraction half correctly whatever they conclude about it.
    """
    gt_v = gt_item["fields"].get("value")
    ext_v = _ext_value(ext)
    if gt_v is None or ext_v is None:
        return False
    try:
        return abs(float(gt_v) - float(ext_v)) < 0.01
    except (ValueError, TypeError):
        return False


# --- the verification head (2026-08-21) --------------------------------------
#
# WHY. `players/prompts/stats_extraction.txt:3,23-24` ORDERS every player to
# "Recompute the two-sided p implied by the statistic and its df ... Flag it",
# and promised "A well-written result must NOT be flagged". Measured 2026-08-21,
# both halves were false against this file: `generator.py` drew p independently
# of the statistic, so 37 of 84 truthful `nhst_stat` gold items were real
# statcheck decision errors, and flagging one emitted `deception_false_alarm`
# which cost neither precision nor recall but dropped the task from primary
# 1.000 to 0.400 through the calibration term. The better a player followed our
# own instructions, the worse it scored: `deception_false_alarm` was the top
# finding for every LLM player (sonnet 65, opus 57, haiku 54, gpt-oss 39).
#
# Verified against live effectcheck 0.7.6 (docs/reports/evidence/): on
# `r(129) = -0.82, p = 0.393` it returns decision_error=True,
# reason='reported_ns_computed_sig', p_computed=4.7e-33. The tool is right.
#
# Gold now carries `expected_verdict` per item, derived by recomputation rather
# than asserted (generator._label_verification), and a player answers with a
# typed `verdict` instead of one boolean.

_VERDICTS = ("ok", "inconsistent", "unverifiable", "not_a_stat")


def _player_verdict(ext: dict) -> str:
    """The player's verdict, falling back to the legacy boolean.

    `flagged_suspicious` remains in the output schema for one version so a player
    that has not been updated still scores. The fallback is deliberately coarse —
    a bare boolean genuinely cannot distinguish "this is inconsistent" from "I
    cannot check this", which is the whole reason the field is being replaced.
    """
    v = ext.get("verdict")
    if isinstance(v, str) and v in _VERDICTS:
        return v
    return "inconsistent" if ext.get("flagged_suspicious") else "ok"


def _gold_verdict(gt_item: dict) -> tuple[bool, str, bool]:
    """(checkable, expected_verdict, coherent) for a gold item, with a legacy fallback.

    Gold produced by `generator._label_verification` carries both explicitly,
    derived by recomputation. Gold that does not — hand-written test fixtures,
    an older stored answer key, a third-party arena — falls back to the pre-2026-08-21
    semantics: a planted deception is `inconsistent`, anything else is `ok`.

    The fallback is safe because it only reproduces the OLD behaviour on gold that
    carries no new information; it cannot reintroduce the defect, which lived in
    the *generator* labelling 37 genuinely inconsistent items `truthful: true`.
    Silently scoring unlabelled gold as unverifiable would be worse: it would make
    a whole arena's verification head vanish with no error, and this project has
    twice published a check that could not fire.
    """
    if "expected_verdict" in gt_item or "checkable" in gt_item:
        return (bool(gt_item.get("checkable", False)),
                gt_item.get("expected_verdict", "ok"),
                bool(gt_item.get("coherent", True)))
    return True, ("ok" if gt_item.get("truthful", True) else "inconsistent"), True


def _verification_cell(expected: str, said: str, coherent: bool = True) -> str:
    """One of: hit | miss | false_alarm | declined | abstain.

    THREE things can happen that are not a right or a wrong answer, and until
    2026-08-27 all three were the single cell `abstain`, which left the
    denominator. Collapsing them is what made refusing to work the best strategy
    in the arena, so they are now named separately and priced separately:

    `abstain` — the item is genuinely AMBIGUOUS, so neither crediting nor
    charging the player is defensible. It leaves the denominator. Exactly one
    thing reaches it: an ACCUSATION ("inconsistent" / "not_a_stat") against an
    item with `coherent=False`, whose reported p does not match the recomputed p
    at the reported precision but which does not cross alpha. Flagging one is
    CORRECT at a finer granularity than `expected_verdict` asks about. Without
    this, statcheck (which reports `error` as well as `decision_error`) took 31
    false alarms for being right. Measured: 74 of 84 truthful NHST items differ
    at the reported precision while only 37 cross alpha.

    A REFUSAL against the same item does NOT reach `abstain` (corrected
    2026-08-28; see the `said == "unverifiable"` branch). The forgiveness is for
    reading an item more finely than the gold asks, not for declining to read it.
    Measured over both task sets: `coherent=False` occurs only ever alongside
    `expected_verdict == "ok"`, so this branch has always been doing that one job.

    `declined` — the player said "unverifiable" about an item the gold CAN
    answer. This is still not an accusation, so it is not a false alarm and emits
    no `deception_false_alarm` finding; the 2026-08-21 rule that an honest "I
    cannot check this" must never be scored as a fabrication claim is intact. But
    it is not free either: it stays in the denominator as a non-hit. Before this,
    a player that declined EVERY item emptied the denominator and collected
    `verification_score = 1.0`, beating every real verifier. Measured over the 36
    revealed v3 tasks with a perfect extractor: always-"unverifiable" 1.0000,
    always-"ok" 0.8078, always-"inconsistent" 0.2931.

    `expected == "unverifiable"` — the gold itself says this item cannot be
    checked (7 of 149 checkable revealed items). Saying so is then the RIGHT
    answer, and asserting an inconsistency in it is a false alarm. This branch did
    not exist: the function handled "ok", then "not_a_stat", then assumed the
    remainder was "inconsistent", so a correct "unverifiable" earned nothing while
    a false accusation was credited as a hit.
    """
    if expected == "unverifiable":
        # Gold cannot check this item, and says so. The player that agrees is
        # right; the player that claims to have found a problem in it is not.
        if said == "unverifiable":
            return "hit"
        if said in ("inconsistent", "not_a_stat"):
            return "false_alarm"
        return "miss"
    if said == "unverifiable":
        # Unconditionally `declined` — coherence does NOT forgive a refusal.
        # Incoherence forgives an ACCUSATION, because an accusation can be right
        # at a finer granularity than `expected_verdict` asks about. A refusal
        # cannot be: "I cannot check this" is not a sharper reading of an
        # answerable item, it is the absence of a reading. Until 2026-08-28 this
        # read `"declined" if coherent else "abstain"`, which left the F1 defect
        # alive one branch over — on a task where every checkable item was
        # incoherent, an all-"unverifiable" player emptied `v_total` and took the
        # neutral 1.0. Measured: 3 such tasks in v3, 0 in v4.
        return "declined"
    if expected == "ok":
        if said == "ok":
            return "hit"
        return "false_alarm" if coherent else "abstain"
    if expected == "not_a_stat":
        # Rejecting an impostor as "not a statistic" is the precise answer;
        # calling it inconsistent is the right instinct with the wrong label, and
        # is credited rather than split hairs over at this sample size (n = 2).
        return "hit" if said in ("not_a_stat", "inconsistent") else "miss"
    # expected == "inconsistent"
    return "hit" if said in ("inconsistent", "not_a_stat") else "miss"


def _ece(predictions: list[tuple[float, bool]], n_bins: int = 5) -> float:
    """Expected Calibration Error. predictions = [(confidence, is_correct), ...]."""
    if not predictions:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in predictions:
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


def _gt_anchor(gt_item: dict) -> dict:
    span = gt_item.get("span", {}) or {}
    return {
        "char_start": span.get("char_start", 0),
        "char_end": span.get("char_end", 0),
        "kind": gt_item.get("kind", ""),
    }


def _ext_anchor(ext: dict) -> dict:
    span = ext.get("span", {}) or {}
    return {
        "char_start": span.get("char_start", 0),
        "char_end": span.get("char_end", 0),
        "kind": ext.get("kind", ""),
    }


def _try_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_OVERCONFIDENT_THRESHOLD = 0.9


def _emit_findings(gt_items: list[dict], extractions: list[dict],
                   matches: list[tuple[int, int]]) -> list[dict]:
    findings: list[dict] = []
    matched_gt = {gi for gi, _ in matches}
    matched_ext = {ei for _, ei in matches}

    # value_off on every matched pair (extraction head), then the verification cell
    for gi, ei in matches:
        gt = gt_items[gi]
        ext = extractions[ei]
        gv = _try_float(gt.get("fields", {}).get("value"))
        ev = _try_float(_ext_value(ext))
        if gv is not None and ev is not None and abs(gv - ev) >= 0.01:
            findings.append({
                "category": "value_off",
                "anchor": _gt_anchor(gt),
                "evidence": str(_ext_value(ext)),
                "correct_value": gv,
            })

        checkable, expected, coherent = _gold_verdict(gt)
        if not checkable:
            # Nothing to verify — a bare effect size with no interval and no test
            # statistic. Probed against live effectcheck 0.7.6 on 2026-08-21: all
            # six bare forms this arena emits return ZERO rows, because there is
            # nothing to check them against. Charging a verification failure here
            # measures a tool outside its contract.
            continue
        cell = _verification_cell(expected, _player_verdict(ext), coherent)
        if cell == "false_alarm":
            findings.append({
                "category": "deception_false_alarm",
                "anchor": _gt_anchor(gt),
                "evidence": f"said inconsistent; gold verdict is {expected}",
            })
        elif cell == "miss":
            findings.append({
                "category": "deception_missed",
                "anchor": _gt_anchor(gt),
                "correct_value": gt.get("deception_kind") or expected,
            })
        elif cell == "declined":
            findings.append({
                "category": "abstain_on_checkable",
                "anchor": _gt_anchor(gt),
                "evidence": f"declined to verify an answerable item (gold: {expected})",
            })
        elif cell == "abstain":
            # Genuinely ambiguous — right at a finer granularity than the gold
            # asks about. Not charged, and not reported as a player error either.
            pass

    # extraction_missed for unmatched gold
    for gi, gt in enumerate(gt_items):
        if gi in matched_gt:
            continue
        finding = {
            "category": "extraction_missed",
            "anchor": _gt_anchor(gt),
        }
        gv = gt.get("fields", {}).get("value")
        if gv is not None:
            finding["correct_value"] = gv
        findings.append(finding)

    # extraction_extra (or kind_mismatch when an unmatched extraction overlaps a gold span with wrong kind)
    for ei, ext in enumerate(extractions):
        if ei in matched_ext:
            continue
        overlap_with_wrong_kind = None
        for gi, gt in enumerate(gt_items):
            if gt["kind"] == ext.get("kind"):
                continue
            if _match_score(gt.get("span", {}), ext.get("span", {})) >= 0.5:
                overlap_with_wrong_kind = gt
                break
        if overlap_with_wrong_kind is not None:
            findings.append({
                "category": "kind_mismatch",
                "anchor": _gt_anchor(overlap_with_wrong_kind),
                "evidence": ext.get("kind", ""),
                "correct_value": overlap_with_wrong_kind["kind"],
            })
        else:
            ev = _ext_value(ext)
            findings.append({
                "category": "extraction_extra",
                "anchor": _ext_anchor(ext),
                "evidence": str(ev) if ev is not None else "(no value)",
            })

    # calibration_overconfident
    for gi, ei in matches:
        gt = gt_items[gi]
        ext = extractions[ei]
        if _is_correct(gt, ext):
            continue
        conf = _try_float(ext.get("confidence"))
        if conf is not None and conf >= _OVERCONFIDENT_THRESHOLD:
            findings.append({
                "category": "calibration_overconfident",
                "anchor": _gt_anchor(gt),
                "evidence": f"confidence={conf:.2f} on incorrect match",
            })
    for ei, ext in enumerate(extractions):
        if ei in matched_ext:
            continue
        conf = _try_float(ext.get("confidence"))
        if conf is not None and conf >= _OVERCONFIDENT_THRESHOLD:
            findings.append({
                "category": "calibration_overconfident",
                "anchor": _ext_anchor(ext),
                "evidence": f"confidence={conf:.2f} on FP extraction",
            })

    return findings


def score(player_output: dict, ground_truth: dict) -> dict:
    gt_items = ground_truth.get("items", [])
    extractions = player_output.get("extractions", [])

    matches = _greedy_match(gt_items, extractions)
    matched_gt = {gi for gi, _ in matches}
    matched_ext = {ei for _, ei in matches}

    tp = sum(1 for gi, ei in matches if _is_correct(gt_items[gi], extractions[ei]))
    fp = len(extractions) - len(matched_ext) + (len(matches) - tp)
    fn = len(gt_items) - len(matched_gt) + (len(matches) - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # --- verification head ---------------------------------------------------
    # Accuracy over CHECKABLE matched items only. Abstentions leave the
    # denominator; an unmatched gold item is an extraction failure and is already
    # charged there, so charging it twice would double-count one mistake.
    v_hits = v_total = 0
    v_cells = {"hit": 0, "miss": 0, "false_alarm": 0, "declined": 0, "abstain": 0}
    for gi, ei in matches:
        gt = gt_items[gi]
        checkable, expected, coherent = _gold_verdict(gt)
        if not checkable:
            continue
        cell = _verification_cell(expected, _player_verdict(extractions[ei]), coherent)
        v_cells[cell] += 1
        if cell == "abstain":
            continue
        v_total += 1
        if cell == "hit":
            v_hits += 1
    # Neutral (1.0) only when there was NOTHING ANSWERABLE — gold with no
    # checkable items, or only ambiguous ones. That case has no verification
    # evidence in either direction, so the head drops out of the product rather
    # than scoring a zero it did not measure.
    #
    # It used to also fire when the player DECLINED every answerable item, and
    # those two are opposite situations: one is the arena having nothing to ask,
    # the other is the player refusing to answer. Conflating them made universal
    # abstention strictly dominant — 1.0000 against 0.8078 for a trivial
    # always-"ok" guesser, measured on all 36 revealed v3 tasks. A declined item
    # now stays in the denominator (see `_verification_cell`), so an all-declining
    # player scores 0.0 here, not 1.0.
    #
    # 2026-08-28: that repair covered only COHERENT items. A decline aimed at an
    # incoherent one was still forgiven, so on the 3 v3 tasks whose every
    # checkable item is incoherent the all-declining player still emptied
    # `v_total` and still took the 1.0. `_verification_cell` no longer consults
    # coherence for a refusal, so the only remaining route to this line is the
    # intended one: gold with nothing checkable, or only accusations against
    # ambiguous items.
    verification_score = (v_hits / v_total) if v_total else 1.0

    predictions: list[tuple[float, bool]] = []
    for gi, ei in matches:
        predictions.append((_safe_conf(extractions[ei]),
                            _is_correct(gt_items[gi], extractions[ei])))
    for ei in range(len(extractions)):
        if ei not in matched_ext:
            predictions.append((_safe_conf(extractions[ei]), False))
    calibration = 1.0 - _ece(predictions) if predictions else 1.0

    # CALIBRATION IS NO LONGER IN THE HEADLINE. It was one of three multiplied
    # terms, and for every deterministic tool player it measured OUR adapter:
    # `framework/player_adapter.py` set `confidence = 0.4 if suspicious else 1.0`,
    # so ESCImate's confidence across 101 extractions was exactly {0.4 x53,
    # 1.0 x48} — a two-point mass this framework invented, whose ECE says nothing
    # about the tool. It is still computed and reported, as a non-ranking column.
    composite = precision * recall * verification_score

    return {
        "primary": composite,
        "breakdown": {
            "precision": precision,
            "recall": recall,
            "verification_score": verification_score,
            "calibration": calibration,
            "composite": composite,
            "n_gt": len(gt_items),
            "n_extracted": len(extractions),
            "n_matched": len(matches),
            "n_verifiable": v_total,
            "verification_hits": v_hits,
            "verification_misses": v_cells["miss"],
            "verification_false_alarms": v_cells["false_alarm"],
            # Two distinct non-answers, reported separately because they are
            # priced differently: `declined` counts in `n_verifiable` as a
            # non-hit, `abstentions` does not count at all.
            "verification_declined": v_cells["declined"],
            "verification_abstentions": v_cells["abstain"],
            "tp": tp, "fp": fp, "fn": fn,
        },
        "findings": _emit_findings(gt_items, extractions, matches),
    }
