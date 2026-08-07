"""Generator for stats-extraction-v1.

All-procedural. Deterministic from (task_set_version, seed).
Tier 6 composition + tiers 4-5 deception kinds are added in Task 12.

Deception taxonomy (tier 5/6, cycled by index for parity):
  internal_inconsistency, swapped_test_label, statistic_impostor,
  missing_info_impostor, fabricated_value (original), plus
  nhst_inconsistent and wrong_df (2026-07-01, statcheck-style failure modes).
Truthful NHST items now report a p that is CONSISTENT with their statistic+df
(recomputed two-sided), so a correctly-reported result is the matched clean
control for nhst_inconsistent (closes the staged "TODO item E").
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import yaml

try:  # scipy is a declared project dependency (pyproject: scipy>=1.11); guard
    from scipy import stats as _scipy_stats  # so a broken env degrades, not crashes.
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only on a scipy-less box
    _scipy_stats = None
    _HAVE_SCIPY = False

ARENA_DIR = Path(__file__).resolve().parent
CATALOGS_DIR = ARENA_DIR / "catalogs"
TEMPLATES_DIR = ARENA_DIR / "templates"

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

# Significance threshold the consistency checks (and a reader) decide against.
_ALPHA = 0.05


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ocr_corrupt(s: str) -> str:
    return s.replace("0", "O").replace("1", "l")


def _format_value(val: float, *, decimals: int = 2, use_comma_decimal: bool = False, drop_leading_zero: bool = False) -> str:
    s = f"{val:.{decimals}f}"
    if drop_leading_zero and s.startswith("0."):
        s = s[1:]
    if drop_leading_zero and s.startswith("-0."):
        s = "-" + s[2:]
    if use_comma_decimal:
        s = s.replace(".", ",")
    return s


def _two_sided_p(stat_id: str, value, df1: int, df2: int) -> float | None:
    """The true two-sided p-value implied by a reported test statistic + df.

    Pure math, no rng — so calling it never perturbs the seed-driven content and
    parity is unaffected. Mirrors statcheck's recomputation: t/r/z are two-sided,
    F/chi2 (and the chi2-family H/Q) are one-sided by construction. Returns
    ``None`` when scipy is unavailable or the statistic id has no closed form, so
    callers fall back to a deterministic stand-in rather than crashing.
    """
    if not _HAVE_SCIPY:
        return None
    v = abs(float(value))
    d1 = max(int(df1), 1)
    d2 = max(int(df2), 1)
    try:
        if stat_id == "t":
            return float(2.0 * _scipy_stats.t.sf(v, d1))
        if stat_id == "F":
            return float(_scipy_stats.f.sf(v, d1, d2))
        if stat_id == "r":
            if v >= 1.0:
                return 0.0
            t = v * (d1 ** 0.5) / ((1.0 - v * v) ** 0.5)
            return float(2.0 * _scipy_stats.t.sf(abs(t), d1))
        if stat_id in ("chi2", "H", "Q"):
            return float(_scipy_stats.chi2.sf(v, d1))
        if stat_id == "z":
            return float(2.0 * _scipy_stats.norm.sf(v))
    except Exception:  # pragma: no cover - scipy edge values
        return None
    return None


def _round_p(p: float) -> float:
    """Round a p to the 3-dp APA grid, never reporting an exact 0.000."""
    r = round(min(max(p, 0.0), 0.999), 3)
    return r if r > 0.0 else 0.001


def _consistent_p(rng: random.Random, stat_id: str, value, df1: int, df2: int) -> float:
    """A reported p that AGREES with the statistic (same significance decision).

    This is what makes a truthful NHST item a genuine clean control for
    ``nhst_inconsistent``: a player that recomputes p from the stat finds no
    discrepancy. Falls back to a plainly-significant small p when the true p is
    unavailable (no scipy) so the control still reads as a coherent result.
    """
    p = _two_sided_p(stat_id, value, df1, df2)
    if p is None:
        return round(rng.uniform(0.001, 0.06), 3)
    return _round_p(p)


def _decisive_sig_stat(rng: random.Random, spec: dict, df1: int, df2: int) -> float:
    """A statistic value whose true two-sided p is decisively < .01 for this df.

    Used by ``nhst_inconsistent`` so the injected (non-significant) p is an
    unambiguous reporting error, not a borderline call. Built from the critical
    value at a strict alpha plus a positive nudge; signed randomly for symmetric
    statistics. Falls back to the catalog range when scipy is unavailable.
    """
    sid = spec["id"]
    if not _HAVE_SCIPY:
        return round(rng.uniform(*spec["range"]), 2)
    a = 0.002
    d1 = max(int(df1), 1)
    d2 = max(int(df2), 1)
    try:
        if sid == "t":
            crit = float(_scipy_stats.t.isf(a / 2, d1)); val = crit + rng.uniform(0.3, 2.0)
            return round(val if rng.random() < 0.5 else -val, 2)
        if sid == "F":
            crit = float(_scipy_stats.f.isf(a, d1, d2)); return round(crit + rng.uniform(0.3, 3.0), 2)
        if sid == "r":
            val = min(0.94, 0.6 + rng.uniform(0.0, 0.34)); return round(val if rng.random() < 0.5 else -val, 2)
        if sid in ("chi2", "H", "Q"):
            crit = float(_scipy_stats.chi2.isf(a, d1)); return round(crit + rng.uniform(0.3, 4.0), 2)
        if sid == "z":
            val = 3.1 + rng.uniform(0.0, 1.5); return round(val if rng.random() < 0.5 else -val, 2)
    except Exception:  # pragma: no cover
        pass
    return round(rng.uniform(*spec["range"]), 2)


def _nonsig_p(rng: random.Random) -> float:
    """A clearly non-significant p (the wrong call for a decisive statistic)."""
    return round(rng.uniform(0.20, 0.45), 3)


# Rendered df-arity per statistic label: F reports two df, t/r/chi2/H/Q one, and
# z/U/W/BF none. Used to render a coherent "{label}(df) = value" / "label = value".
_DF_ARITY = {"t": 1, "F": 2, "r": 1, "chi2": 1, "H": 1, "Q": 1, "z": 0, "U": 0, "W": 0, "BF10": 0}


def _render_stat_str(fields: dict) -> str:
    """Render a coherent "label(df) = value" string with the right df count.

    The value is rendered with the plain 2-dp form (``_format_value``), which is
    one of the renderings ``_anchor_span`` searches, so the gold span anchors on
    it. F gets two df, t/r/chi2 one, z/U/W/BF none.
    """
    label = fields["label"]
    arity = _DF_ARITY.get(label, 1)
    val = _format_value(fields["value"])
    if arity == 2:
        return f"{label}({fields['df1']}, {fields['df2']}) = {val}"
    if arity == 1:
        return f"{label}({fields['df1']}) = {val}"
    return f"{label} = {val}"


def _build_nhst_fields(rng: random.Random, nhst_catalog: list[dict]) -> dict:
    spec = rng.choice(nhst_catalog)
    df1 = rng.randint(8, 200)
    df2 = rng.randint(2, 12)
    val = round(rng.uniform(*spec["range"]), 2)
    # NOTE: p here is independent of the statistic. The tier 1-4 templates render a
    # FIXED label (e.g. "t(...)") that does not always match `spec`, so a p
    # recomputed from (spec, val) would not be consistent with the *rendered*
    # statistic anyway (the label<->stat mismatch is the staged "TODO item D").
    # The recompute-consistent clean control for nhst_inconsistent is instead the
    # dedicated, fully-coherent `nhst_consistent` truthful item (see
    # _build_nhst_consistent_fields), where label, value and p are built together.
    p = round(rng.uniform(0.0001, 0.5), 3)
    return {
        "stat_id": spec["id"], "label": spec["label_pattern"],
        "df1": df1, "df2": df2, "value": val, "p": p,
    }


def _build_nhst_consistent_fields(rng: random.Random, nhst_catalog: list[dict]) -> dict:
    """A fully-coherent truthful, SIGNIFICANT NHST result (the clean control).

    This is the matched clean control for ``nhst_inconsistent``. The statistic is
    drawn DECISIVELY significant (true two-sided p < .01) for its df, the df-arity
    matches the test (F gets two df, t/r/chi2 one), and the reported p is the
    recomputed two-sided p — so prose ("the test was significant"), statistic, and
    p all agree, and a player that recomputes p finds NO discrepancy. It differs
    from the nhst_inconsistent deception in exactly one place: the reported p is
    correct here, wrong there. Only closed-form stats are used so it is verifiable.
    """
    verifiable = [s for s in nhst_catalog if s["id"] in ("t", "F", "r", "chi2", "z")]
    spec = rng.choice(verifiable)
    sid = spec["id"]
    df1 = rng.randint(8, 200)
    df2 = rng.randint(2, 12)
    if sid == "r":
        df1 = rng.randint(15, 200)  # keep |t(r)| finite and p well-defined
    val = _decisive_sig_stat(rng, spec, df1, df2)
    p = _consistent_p(rng, sid, val, df1, df2)
    return {
        "stat_id": sid, "label": spec["label_pattern"],
        "df1": df1, "df2": df2, "value": val, "p": p,
    }


def _build_anova_design_fields(rng: random.Random) -> dict:
    """Fields for a one-way ANOVA stated WITH its sample size and group count.

    Returns a realistic design: ``k`` groups, ``n_total`` participants, with the
    *correct* numerator/denominator df (df1 = k-1, df2 = n_total-k) plus a
    ``df2_wrong`` that is inconsistent with N. The matched pair powers ``wrong_df``
    (mistake = df2_wrong) and its clean control (truthful = df2). The reported F
    is decisively significant so the df is the only thing in question.
    """
    k = rng.randint(3, 5)
    n_per = rng.randint(12, 40)
    n_total = k * n_per
    df1 = k - 1
    df2 = n_total - k
    val = _decisive_sig_stat(rng, {"id": "F", "range": [0.1, 30]}, df1, df2)
    p = _consistent_p(rng, "F", val, df1, df2)
    # A denominator df that contradicts the stated N (off by a chunk, never equal).
    df2_wrong = df2 + rng.randint(12, 40)
    return {
        "stat_id": "F", "label": "F", "k_groups": k, "n_total": n_total,
        "df1": df1, "df2": df2, "df2_wrong": df2_wrong, "value": val, "p": p,
    }


def _build_es_fields(rng: random.Random, es_catalog: list[dict]) -> dict:
    spec = rng.choice(es_catalog)
    es = round(rng.uniform(*spec["range"]), 2)
    ci_lo = round(es - rng.uniform(0.05, 0.3), 2)
    ci_hi = round(es + rng.uniform(0.05, 0.3), 2)
    return {
        "es_id": spec["id"], "es_label": spec["label"],
        "es": es, "ci_lo": ci_lo, "ci_hi": ci_hi,
    }


def _build_es_rounding_fields(rng: random.Random, es_catalog: list[dict]) -> dict:
    """An effect size whose point estimate was re-rounded so it no longer matches its CI.

    A real ESCIcheck/statcheck-family value-integrity failure: the CI is the TRUE
    symmetric interval about a 2-dp estimate ``es_true`` (so its midpoint recovers
    ``es_true``), but the estimate that got written into the text (``es_reported``)
    was rounded to ONE decimal and drifted away from ``es_true`` by a visible margin.
    A player that recomputes the CI midpoint ((ci_lo+ci_hi)/2) and compares it with
    the stated estimate sees the disagreement. We redraw until 1-dp rounding actually
    moves the number by >= 0.05, so the inconsistency is real (not a rounding tie).

    The GOLD anchor value is ``es_reported`` — that is the number rendered into the
    text (the CI itself is the true one), so ``_anchor_span`` finds it. Only the
    estimate is corrupted; matched clean control is ``clean_es`` (tier 1), where
    the estimate equals the CI midpoint. Uses ``_format_value`` for render parity.
    """
    spec = rng.choice(es_catalog)
    # Draw a true 2-dp estimate and a symmetric CI about it, redrawing until the
    # 1-dp re-rounding lands a VISIBLE (>= 0.05) distance from the true estimate.
    for _ in range(64):  # a handful of draws suffices; bound the loop defensively
        es_true = round(rng.uniform(*spec["range"]), 2)
        es_reported = round(es_true, 1)
        if abs(es_reported - es_true) >= 0.05:
            break
    else:  # pragma: no cover - only if a pathological range never yields a gap
        es_reported = round(es_true + 0.05, 1)
    h = round(rng.uniform(0.05, 0.3), 2)
    ci_lo = round(es_true - h, 2)
    ci_hi = round(es_true + h, 2)
    ci_mid = round((ci_lo + ci_hi) / 2, 2)  # recovers es_true up to 2-dp rounding
    return {
        "es_id": spec["id"], "es_label": spec["label"],
        "es_reported": es_reported, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "es_true": es_true, "ci_mid": ci_mid,
    }


def _build_clean_es_fields(rng: random.Random, es_catalog: list[dict]) -> dict:
    """A correctly-rounded effect size whose estimate EQUALS its CI midpoint.

    The matched clean control (look-alike) for ``effect_size_rounding``: identical
    shape — an estimate plus a symmetric 95% CI — but here the reported estimate is
    exactly the CI midpoint (``es`` = (ci_lo+ci_hi)/2 up to 2-dp rounding), so a
    player recomputing the midpoint finds NO discrepancy and must NOT flag it. Lives
    in tier 1 (clean APA baseline); like ``nhst_consistent`` it is emitted once per
    tier-1 task so both splits carry it equally and parity is unaffected.
    """
    spec = rng.choice(es_catalog)
    es = round(rng.uniform(*spec["range"]), 2)
    h = round(rng.uniform(0.05, 0.3), 2)
    ci_lo = round(es - h, 2)
    ci_hi = round(es + h, 2)
    return {
        "es_id": spec["id"], "es_label": spec["label"],
        "es": es, "ci_lo": ci_lo, "ci_hi": ci_hi,
    }


def _instantiate_template(template: str, fields: dict) -> str:
    """Fill a template with a permissive .format-style substitution.

    Tier-specific variants (e.g. {value_comma}, {p_no_lead}, {value_ocr}) are
    derived from base fields here so templates can request them without the
    field-builder knowing about tiers.
    """
    derived = {**fields}
    if "value" in fields:
        derived["value_comma"] = _format_value(fields["value"], use_comma_decimal=True)
        derived["value_ocr"] = _ocr_corrupt(_format_value(fields["value"]))
    if "p" in fields:
        derived["p_no_lead"] = _format_value(fields["p"], decimals=3, drop_leading_zero=True)
        derived["p_comma"] = _format_value(fields["p"], decimals=3, use_comma_decimal=True)
        derived["p_ocr"] = _ocr_corrupt(_format_value(fields["p"], decimals=3))
        derived["p_threshold"] = ".05"
    if "df1" in fields:
        derived["df1_ocr"] = _ocr_corrupt(str(fields["df1"]))
    if "es" in fields:
        derived["es_no_lead"] = _format_value(fields["es"], decimals=2, drop_leading_zero=True)
    if "ci_lo" in fields:
        derived["ci_lo_no_lead"] = _format_value(fields["ci_lo"], drop_leading_zero=True)
        derived["ci_lo_comma"] = _format_value(fields["ci_lo"], use_comma_decimal=True)
    if "ci_hi" in fields:
        derived["ci_hi_no_lead"] = _format_value(fields["ci_hi"], drop_leading_zero=True)
        derived["ci_hi_comma"] = _format_value(fields["ci_hi"], use_comma_decimal=True)
    return template.format(**derived)


def _value_renderings(value) -> list[str]:
    """Every way a numeric value may appear in task text given the templates.

    Tiers render the SAME value in different forms — raw float ``48.7``, fixed
    2-decimal ``48.70``, comma-decimal ``48,70`` (tier 2), leading-zero-dropped
    ``.72`` (effect sizes), OCR-swapped ``2l.85`` (tier 3). The gold span anchor
    must match whichever form was actually emitted, so we enumerate them all,
    most-specific (longest) first so a short form like ``0.7`` is tried after
    longer ones and never anchors inside a longer number such as ``0.75``.
    """
    fv = _format_value(value)
    cands = [
        _ocr_corrupt(fv),                                # tier-3 {value_ocr}
        _format_value(value, use_comma_decimal=True),    # tier-2 {value_comma}
        fv,                                              # plain 2-decimal
        _format_value(value, drop_leading_zero=True),    # effect-size {es_no_lead}
        f"{value}",                                      # raw {value}/{es} (Python str)
    ]
    seen: set[str] = set()
    uniq: list[str] = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    uniq.sort(key=len, reverse=True)
    return uniq


def _anchor_span(text: str, value, start: int = 0) -> dict:
    """Locate the gold span for ``value`` as actually rendered into ``text``.

    Tries every plausible rendering (see :func:`_value_renderings`) and requires
    digit boundaries so a short rendering does not match inside a longer adjacent
    number. Returns a degenerate ``[0,0]`` span only when no rendering of the
    value is present at all (e.g. a deception value deliberately not written).
    """
    for cand in _value_renderings(value):
        idx = text.find(cand, start)
        while idx != -1:
            before = text[idx - 1] if idx > 0 else ""
            after_i = idx + len(cand)
            after = text[after_i] if after_i < len(text) else ""
            if not before.isdigit() and not after.isdigit():
                return {"text": text[idx:after_i], "char_start": idx, "char_end": after_i}
            idx = text.find(cand, idx + 1)
    return {"text": "", "char_start": 0, "char_end": 0}


def _assign_spans(text: str, items: list[dict]) -> None:
    """Attach a gold span to each item, anchored on its rendered value.

    Items are emitted in text order, so a forward cursor disambiguates repeated
    values (two items sharing a value anchor to successive occurrences). Falls
    back to a from-start search when the cursor overshoots.
    """
    cursor = 0
    for item in items:
        span = _anchor_span(text, item["fields"]["value"], cursor)
        if span["char_end"] <= span["char_start"]:
            span = _anchor_span(text, item["fields"]["value"], 0)
        item["span"] = span
        if span["char_end"] > span["char_start"]:
            cursor = span["char_end"]


def _build_task(task_id: str, rng: random.Random, tier: int, density: int,
                nhst_catalog, es_catalog) -> tuple[dict, dict]:
    """Generate one task envelope + its ground truth."""
    templates_path = TEMPLATES_DIR / f"tier{tier}_{ {1:'baseline',2:'notation',3:'formatting'}[tier] }.yaml"
    templates = _load_yaml(templates_path)

    parts: list[str] = []
    items: list[dict] = []  # ground-truth items
    n_stats = density
    for _ in range(n_stats):
        kind = rng.choice(["nhst", "effect_size"]) if templates.get("effect_size") else "nhst"
        if kind == "nhst":
            fields = _build_nhst_fields(rng, nhst_catalog)
            tmpl = rng.choice(templates["nhst"])
            sentence = _instantiate_template(tmpl, fields)
            items.append({
                "kind": "nhst_stat",
                "fields": {"test_type": fields["stat_id"], "df1": fields["df1"],
                           "df2": fields["df2"], "value": fields["value"], "p": fields["p"]},
                "truthful": True, "deception_kind": None,
            })
        else:
            fields = _build_es_fields(rng, es_catalog)
            tmpl = rng.choice(templates["effect_size"])
            sentence = _instantiate_template(tmpl, fields)
            items.append({
                "kind": "effect_size",
                "fields": {"effect_size_type": fields["es_id"], "value": fields["es"],
                           "ci_low": fields["ci_lo"], "ci_high": fields["ci_hi"]},
                "truthful": True, "deception_kind": None,
            })
        parts.append(sentence)

    # Tier 1 carries one fully-coherent "nhst_consistent" control per task — the
    # matched clean look-alike for the tier-5 nhst_inconsistent deception (correct
    # label/df, p = recomputed two-sided p). Emitted deterministically (always one,
    # after the density items) so both splits carry it equally and parity is
    # unaffected; only the wording/value is seed-driven. Tier 1 is the natural home
    # (clean APA baseline); the OCR/comma noise of tiers 2-3 would obscure the point.
    if tier == 1 and templates.get("nhst_consistent"):
        cf = _build_nhst_consistent_fields(rng, nhst_catalog)
        cf["stat_str"] = _render_stat_str(cf)
        parts.append(_instantiate_template(rng.choice(templates["nhst_consistent"]), cf))
        items.append({
            "kind": "nhst_stat",
            "fields": {"test_type": cf["stat_id"], "df1": cf["df1"], "df2": cf["df2"],
                       "value": cf["value"], "p": cf["p"]},
            "truthful": True, "deception_kind": None,
        })

    # Tier 1 also carries one "clean_es" control per task — the matched clean
    # look-alike for the tier-5 effect_size_rounding deception (same shape: an
    # estimate + a symmetric 95% CI), but here the estimate EQUALS the CI midpoint,
    # so a player recomputing the midpoint finds no discrepancy and must NOT flag
    # it. Emitted deterministically after the nhst_consistent control (in text
    # order, so _assign_spans's forward cursor anchors it correctly); both splits
    # carry it equally, so parity is unaffected. Tier 1 is the clean home (the
    # OCR/comma noise of tiers 2-3 would obscure the estimate<->CI comparison).
    if tier == 1 and templates.get("clean_es"):
        ce = _build_clean_es_fields(rng, es_catalog)
        parts.append(_instantiate_template(rng.choice(templates["clean_es"]), ce))
        items.append({
            "kind": "effect_size",
            "fields": {"effect_size_type": ce["es_label"], "value": ce["es"],
                       "ci_low": ce["ci_lo"], "ci_high": ce["ci_hi"]},
            "truthful": True, "deception_kind": None,
        })

    text = " ".join(parts)
    _assign_spans(text, items)

    envelope = {
        "task_id": task_id,
        "arena_id": "stats-extraction-v1",
        "task_set_version": "v1",
        "visibility": "public",
        "difficulty": {"tier": tier, "density": density, "deception_rate": 0},
        "input": {"text": text, "tier": tier, "format_hint": "plain_text"},
    }
    return envelope, {"items": items}


def _build_tier4_task(task_id, rng, density, nhst, es):
    templates = _load_yaml(TEMPLATES_DIR / "tier4_density.yaml")
    parts, items = [], []
    for _ in range(density):
        f1 = _build_nhst_fields(rng, nhst)
        f2 = _build_nhst_fields(rng, nhst)
        f1["value2"] = f2["value"]; f1["p2"] = f2["p"]
        tmpl = rng.choice(templates["nhst"])
        parts.append(_instantiate_template(tmpl, f1))
        items.append({"kind": "nhst_stat",
                      "fields": {"test_type": f1["stat_id"], "df1": f1["df1"], "df2": f1["df2"],
                                 "value": f1["value"], "p": f1["p"]},
                      "truthful": True, "deception_kind": None})
    # One clean ANOVA control per task: the matched truthful look-alike for the
    # tier-5 wrong_df deception (df2 = N - k is consistent here). Emitted
    # deterministically (always, after the density items) so both splits carry it
    # equally and parity is unaffected; only the wording is seed-driven.
    af = _build_anova_design_fields(rng)
    parts.append(_instantiate_template(rng.choice(templates["clean_anova"]), af))
    items.append({"kind": "nhst_stat",
                  "fields": {"test_type": af["stat_id"], "df1": af["df1"], "df2": af["df2"],
                             "value": af["value"], "p": af["p"]},
                  "truthful": True, "deception_kind": None})
    text = " ".join(parts)
    _assign_spans(text, items)
    return {"task_id": task_id, "arena_id": "stats-extraction-v1", "task_set_version": "v1",
            "visibility": "public",
            "difficulty": {"tier": 4, "density": density, "deception_rate": 0},
            "input": {"text": text, "tier": 4, "format_hint": "plain_text"}}, {"items": items}


def _build_tier5_task(task_id, rng, density, nhst, es, deception_kinds_yaml, kind_offset=0):
    """Build one deception task.

    Deception KINDS are assigned deterministically by cycling through every
    declared kind (driven by `kind_offset`, a seed-independent running counter),
    so every benchmark split covers the full array of injected mistakes equally.
    Only the concrete VALUES/templates stay rng-driven, so distinct seeds still
    produce distinct content across the revealed and private splits. This is what
    lets framework/parity.py pass at count_tolerance 0. See contract/README.md.
    """
    templates = _load_yaml(TEMPLATES_DIR / "tier5_adversarial.yaml")
    deception_ids = [d["id"] for d in deception_kinds_yaml]
    parts, items = [], []
    for i in range(density):
        kind = deception_ids[(kind_offset + i) % len(deception_ids)]
        tmpl = rng.choice(templates[kind])
        if kind == "nhst_inconsistent":
            # A decisively-significant statistic (true two-sided p < .01) reported
            # with a non-significant p — a real statcheck DECISION error. Built
            # coherently (correct label + df-arity via {stat_str}, decisive value)
            # so the ONLY defect is the p; the label/df are right, which is what
            # separates this kind from swapped_test_label. The gold value is the
            # statistic as rendered, so its span anchors normally.
            f = _build_nhst_consistent_fields(rng, nhst)
            f["stat_str"] = _render_stat_str(f)
            f["wrong_p"] = _nonsig_p(rng)
            sentence = _instantiate_template(tmpl, f)
            anchor_value = f["value"]
        elif kind == "wrong_df":
            # A correctly-labelled F whose denominator df is inconsistent with the
            # stated N and group count (df2 should be N - k). The clean control is
            # the truthful ANOVA template that reports the consistent df2.
            f = _build_anova_design_fields(rng)
            sentence = _instantiate_template(tmpl, f)
            anchor_value = f["value"]
        elif kind == "effect_size_rounding":
            # An effect-size point estimate re-rounded to fewer decimals so it no
            # longer matches its own CI (the CI midpoint disagrees with the stated
            # estimate). Only the estimate is corrupted; the CI is the true one. The
            # value actually written (and thus anchorable) is es_reported. This kind
            # has no test statistic / df / p, so it is emitted as an effect_size
            # item below (like statistic_impostor), never touching f["p"].
            f = _build_es_rounding_fields(rng, es)
            sentence = _instantiate_template(tmpl, f)
            anchor_value = f["es_reported"]
        else:
            f = _build_nhst_fields(rng, nhst)
            f["wrong_p"] = round(min(0.99, f["p"] + 0.4), 3)
            f["wrong_value"] = round(f["value"] + 1.0, 2)
            ef = _build_es_fields(rng, es)
            f["es"] = ef["es"]; f["es_label"] = ef["es_label"]
            sentence = _instantiate_template(tmpl, f)
            # statistic_impostor renders "Cohen's d = {es}", so the value actually
            # written into the text (and thus anchorable) is the es, not the
            # unrelated nhst value that the other kinds report.
            anchor_value = f["es"] if kind == "statistic_impostor" else f["value"]
        parts.append(sentence)
        # effect_size_rounding is an effect-size record (type/value/CI), like
        # statistic_impostor; the nhst-shaped kinds carry test_type/df/p. Building
        # item_kind/item_fields per shape keeps f["p"] off the effect-size path
        # (its fields dict has no p) and yields a well-formed record for each.
        if kind == "effect_size_rounding":
            item_kind = "effect_size"
            item_fields = {"effect_size_type": f["es_label"], "value": anchor_value,
                           "ci_low": f["ci_lo"], "ci_high": f["ci_hi"]}
        else:
            item_kind = "effect_size" if kind == "statistic_impostor" else "nhst_stat"
            item_fields = {"test_type": f["stat_id"], "df1": f.get("df1"), "df2": f.get("df2"),
                           "value": anchor_value, "p": f["p"]}
        items.append({
            "kind": item_kind,
            "fields": item_fields,
            "truthful": False,
            "deception_kind": kind,
        })
    text = " ".join(parts)
    _assign_spans(text, items)
    return {"task_id": task_id, "arena_id": "stats-extraction-v1", "task_set_version": "v1",
            "visibility": "public",
            "difficulty": {"tier": 5, "density": density, "deception_rate": 5},
            "input": {"text": text, "tier": 5, "format_hint": "plain_text"}}, {"items": items}


def _build_tier6_task(task_id, rng, density, nhst, es, deception_kinds_yaml):
    """Compose a multi-section results section out of tier 1-5 paragraphs."""
    composition = _load_yaml(TEMPLATES_DIR / "tier6_results_section.yaml")
    titles = composition["section_titles"]
    section_tmpl = composition["section_template"]
    crossrefs = composition["cross_reference_phrases"]

    sections = []
    items_all: list[dict] = []
    text_pieces: list[str] = []
    for i, title in enumerate(titles):
        # alternate among lower tiers
        sub_tier = (i % 5) + 1
        if sub_tier == 5:
            # Deterministic kind offset (seed-independent) so tier-6 deception
            # coverage is identical across splits too.
            env, gt = _build_tier5_task(f"{task_id}-sub{i}", rng, density, nhst, es,
                                        deception_kinds_yaml, kind_offset=i)
        elif sub_tier == 4:
            env, gt = _build_tier4_task(f"{task_id}-sub{i}", rng, density, nhst, es)
        else:
            env, gt = _build_task(f"{task_id}-sub{i}", rng, sub_tier, density, nhst, es)
        body = env["input"]["text"]
        if i > 0:
            body = body + " " + rng.choice(crossrefs) + "."
        sections.append(section_tmpl.format(title=title, body=body))
        items_all.extend(gt["items"])
    full_text = "\n\n".join(sections)

    # Recompute spans against the full composed text.
    _assign_spans(full_text, items_all)

    return {"task_id": task_id, "arena_id": "stats-extraction-v1", "task_set_version": "v1",
            "visibility": "public",
            "difficulty": {"tier": 6, "density": density, "deception_rate": 3},
            "input": {"text": full_text, "tier": 6, "format_hint": "plain_text"}}, {"items": items_all}


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    """Yield the task envelopes for one benchmark `split`.

    Both splits run the IDENTICAL tier × density × deception matrix below; only
    the seed (chosen by the caller per arena.yaml#benchmark_splits) and the
    visibility/split tags differ. This is what makes the revealed and private
    suites parity-matched by construction (see framework/parity.py). This arena
    is synthetic-only, so the private split has no real-world holdout to append.
    """
    visibility = "public" if split == "revealed" else "held_out"
    nhst = _load_yaml(CATALOGS_DIR / "nhst_stats.yaml")
    es = _load_yaml(CATALOGS_DIR / "effect_sizes.yaml")
    deception_kinds = _load_yaml(CATALOGS_DIR / "deception_kinds.yaml")
    n_per_cell = 2
    densities = [1, 2, 3]
    t5_offset = 0  # seed-independent running counter -> deterministic deception-kind coverage
    for tier in (1, 2, 3, 4, 5, 6):
        for density in densities:
            for k in range(n_per_cell):
                tid = f"t-tier{tier}-d{density}-{k}-s{seed}"
                rng = random.Random(_seed_int(task_set_version, seed, tier, density, k))
                if tier in (1, 2, 3):
                    env, gt = _build_task(tid, rng, tier, density, nhst, es)
                elif tier == 4:
                    env, gt = _build_tier4_task(tid, rng, density, nhst, es)
                elif tier == 5:
                    env, gt = _build_tier5_task(tid, rng, density, nhst, es, deception_kinds,
                                                kind_offset=t5_offset)
                    t5_offset += density
                else:
                    env, gt = _build_tier6_task(tid, rng, density, nhst, es, deception_kinds)
                env["split"] = split
                env["visibility"] = visibility
                _GROUND_TRUTH_CACHE[tid] = gt
                yield env


def ground_truth(task_id: str) -> dict:
    """Return gold for a task, regenerated from seed via the in-process cache.

    The runner (and dump_revealed_gold / check_parity) always call generate()
    before ground_truth(), which fills _GROUND_TRUTH_CACHE. Serving from that
    cache keeps the served gold byte-identical to what generate() produces — so
    a fix to the span/value logic takes effect with no external answer-key
    rebuild — and makes the private split's secret the seed, not a stored key
    (see LESSONS.md 2026-06-06 "New arenas should be registry-free (gold
    regenerated from seed)"). Previously this read the article-finder eval-only
    registry, which could drift out of sync with the generator.
    """
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(
            f"No cached gold for {task_id!r}; call generate() for the matching "
            "split/seed before ground_truth()."
        )
    return _GROUND_TRUTH_CACHE[task_id]


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)
