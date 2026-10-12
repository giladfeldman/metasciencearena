"""Generator for stats-extraction-v1.

All-procedural. Deterministic from (task_set_version, seed).
Tier 6 composition + tiers 4-5 deception kinds are added in Task 12.

Deception taxonomy (tier 5/6, cycled by index for parity):
  internal_inconsistency, swapped_test_label, statistic_impostor,
  missing_info_impostor, fabricated_value (original), plus
  nhst_inconsistent and wrong_df (2026-07-01, statcheck-style failure modes).
Truthful NHST items do NOT all report a p consistent with their statistic+df:
`_build_nhst_fields` draws the two independently, deliberately, for the generic
tier 1-4 items (see the NOTE there). Only `nhst_consistent` is built coherent.
An earlier version of this docstring claimed otherwise and was false for as long
as it stood -- measured 2026-08-21, 37 of 84 truthful `nhst_stat` items in task
set v2 are real statcheck decision errors. Each item therefore carries an
explicit `expected_verdict` stamped by `_label_verification`, so the property is
DECLARED per item rather than asserted in prose here.
"""
from __future__ import annotations

import hashlib
import random
import re
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
        # No closed form (U, W, BF10). Draw clear of the boundary in BOTH
        # directions rather than straddling it, so the item is unambiguous.
        return round(rng.uniform(0.001, 0.04), 3) if rng.random() < 0.5             else round(rng.uniform(0.06, 0.45), 3)
    r = _round_p(p)
    # Rounding to the 3-dp APA grid can move a p ACROSS alpha: a true .04995
    # prints as .050, and `.050 < .05` is False, so the "consistent" control
    # reads as a decision error. Measured on the first v3 draft, this alone
    # produced one crossing item. Nudge to the same side as the true value.
    if (r < _ALPHA) != (p < _ALPHA):
        r = round(_ALPHA - 0.001, 3) if p < _ALPHA else round(_ALPHA + 0.001, 3)
    return r


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


def _pins_labels(task_set_version: str) -> bool:
    """Whether item D (statistic drawn to match the template's label) applies.

    v1 is FROZEN. The generator is shared across task sets, so pinning the
    statistic for every version silently rewrote history: skipping the old
    `rng.choice(nhst_catalog)` draw shifts the entire downstream RNG stream, and
    only 7 of 36 v1 tasks still matched `task_sets/v1/_ground_truth.json` —
    `t-tier1-d1-0-s0` went from `r(28) = 48.7` (gold `test_type: Q`) to
    `t(13) = -0.09`. v1's gold is a committed artifact and its records are kept as
    labelled history; anyone reproducing a published v1 score must get the text
    that score was computed against, impossible-looking correlations included.

    Same gate, same reason, as `_seed_tag`. Found by a Codex review pass on
    2026-08-09 and reproduced before acting.
    """
    return task_set_version != "v1"


def _is_v3_plus(task_set_version: str) -> bool:
    """Whether the task set 3 stimulus changes apply.

    Same gate, same reason, as `_pins_labels`: the generator is shared across
    task sets and is APPEND-ONLY. v1 and v2 gold are committed artifacts whose
    published scores were computed against exactly that text, so a change that
    shifts the RNG stream must be version-gated or it silently rewrites frozen
    history. (Pinning labels for every version once left only 7 of 36 v1 tasks
    matching their committed gold.)

    v3 changes three things v1/v2 must not see:
      * truthful NHST items get a p CONSISTENT with their statistic (see
        `_build_nhst_fields`) -- v2's were drawn independently, making 44% of
        them real statcheck decision errors labelled `truthful: true`;
      * some tier-1 items render an effect size BESIDE its test statistic, the
        canonical APA form v1/v2 never emitted once;
      * tier 6 gets a fifth section title, which makes its deception branch
        reachable instead of dead.
    """
    return task_set_version not in ("v1", "v2")


def _is_v4_plus(task_set_version: str) -> bool:
    """Whether the v4 p-REPORTING change applies. Same append-only reason as above.

    v4 changes ONE thing v1/v2/v3 must not see: a p whose true value falls beyond
    the 3-dp APA grid is printed as an INEQUALITY (`p < .001`, `p > .999`) instead
    of being clamped to a point value.

    Why this is a correctness fix and not a cosmetic one. `_round_p` clamps: any
    true p below .0005 prints `p = .001`, any above .999 prints `p = .999`. Those
    are FALSE STATEMENTS about the statistic, and a tool that recomputes p is
    right to flag them. Measured on v3 revealed, 44 of the 45 truthful NHST items
    labelled incoherent are at a clamp and exactly ONE is a real precision
    mismatch. So the arena was manufacturing 44 reporting errors and then
    forgiving them, and statcheck's apparent weakness was largely it detecting a
    defect we had planted by accident.

    Printing the inequality is also simply what APA requires and what a real paper
    prints, so v4 stimulus text is MORE realistic than v3's, not less.

    THE RULE IS NOT "REPLACE EVERY CLAMP WITH AN INEQUALITY". A p that genuinely
    rounds to .001 (true p in [.0005, .0015)) must keep printing `p = .001` --
    `p < .001` would be false of it. Measured: converting every clamp blindly
    turns 11 correct items incoherent. `_apa_p_rel` therefore keys on the TRUE p,
    never on the printed value.
    """
    return task_set_version not in ("v1", "v2", "v3")


def _is_v5_plus(task_set_version: str) -> bool:
    """Whether the v5 PRINTED-CLAIM change applies. Same append-only reason as above.

    v4 stamped `p_rel` onto every truthful NHST gold item, but three templates
    never interpolated it, so the ANSWER KEY recorded a relation the PASSAGE
    never printed:

        tier2_notation.yaml  `*t*({df1}) = {value}, *p* = {p_no_lead}.`
        tier2_notation.yaml  `t({df1}) = {value}, p < {p_threshold}.`
        tier3_formatting.yaml `t({df1})\n= {value}, p &lt; {p_threshold}`

    The second and third are worse than a wrong relation: `p_threshold` is the
    CONSTANT ".05" that `_instantiate_template` supplies, so the drawn p is never
    rendered and the gold graded a number the reader cannot see.

    MEASURED on v4 revealed, with every one of the 112 nhst gold items located
    (the first survey located 55 and found 6; this is the full count):
      * 7 items print a relation the gold does not record;
      * 11 print the `.05` threshold rather than the gold p;
      * 5 print a claim that is FALSE of their own recomputed p while the answer
        key says `expected_verdict: ok`.
    On the last 5, a tool that recomputes p and correctly flags the misreport is
    charged a `false_alarm` and one that says "ok" is credited a hit -- the
    scoring is inverted against exactly the competence this arena measures.

    THE FIX IS NOT A RELABEL. Re-labelling those items `inconsistent` would leave
    the passage printing `p = .001` for a true p of 7.3e-06, which is precisely
    the manufactured error v4 exists to remove. v5 changes the TEXT so the
    printed claim is true and the key grades that claim -- which is why it is a
    new task set and a full re-run rather than a rescore.

    Templates carry v5 wording under a separate `nhst_v5` key rather than being
    edited in place, the same shape as tier 6's `section_titles_v3`: v1-v4 must
    keep rendering byte-identical text or their committed gold and published
    scores stop referring to tasks that exist.
    """
    return task_set_version not in ("v1", "v2", "v3", "v4")


def _is_v6_plus(task_set_version: str) -> bool:
    """Whether the v6 SEED-INDEPENDENT COMPOSITION change applies. Append-only.

    Through v5, `_build_task` picked each density slot's clause shape by drawing
    from `rng`: the kind (`rng.choice(["nhst", "effect_size"])`) and, from v3,
    whether an nhst slot also printed its effect size (`rng.random() < 0.5`). The
    revealed and private splits are generated from DIFFERENT seeds, so both draws
    land differently on each split and the two halves of the benchmark stop being
    matched on what they are made of.

    That is not cosmetic, because `scorer.py`'s `fn` term charges EVERY unmatched
    gold item as a false negative. A tool that only reads significance tests --
    statcheck, the real instrument this arena is built around -- therefore has its
    per-task recall capped at nhst_items / all_items, and that cap is a property
    of the seed rather than of the tool. MEASURED on v5 with the real split seeds,
    mean over all 36 tasks:

        NHST-only recall ceiling    revealed 0.6516    private 0.6901   (-0.0385)

        per-tier nhst/all:  t1 11/28 vs 11/26   t2 8/12 vs 7/12   t3 5/12 vs 7/12
                            t4 18/18 both       t5 9/12 both      t6 53/81 vs 51/81

    Tiers 2 and 3 hold IDENTICAL item counts on both splits and still differ in
    composition, which is why making cardinality alone deterministic would leave
    the larger half of the defect standing. Tiers 4 and 5 already match exactly:
    neither draws a kind, and tier 5's deception kinds are index-cycled.

    v6 replaces both draws with `_slot_plan_v6`, an index cycle in the same shape
    as the tier-5 deception-kind assignment -- the technique is already proven
    here, since generating v5 over 40 seeds yields exactly 24 injected items on
    every one.

    The cycles are sized to land on v5's MEAN composition rather than on some new
    one, so v6 removes the variance without moving the arena's difficulty. What
    that is worth is a measurement, not an argument. Over the same 40 seeds:

                          v5 (per seed)            v6 (every seed)
        distinct profiles 40                       1
        nhst items        95..114   mean 104.95    105
        all gold items    157..168  mean 162.05    162
        NHST ceiling      0.5641..0.7581  m 0.6545 0.6508

    The direct cross-split test, eight revealed/private seed pairs: v5 matches on
    0 of 8, v6 on 8 of 8.

    v6 fixes COMPOSITION only, never content: over five seeds all 36 task cells
    still render five distinct texts, so the private split keeps its secret and
    the two halves remain different tasks that are merely made of the same parts.

    v1-v5 keep the `rng` draws, and must -- their committed gold and published run
    records were computed from this generator, and `_seed_int` includes the task
    set version, so any change to what they render orphans those scores.
    """
    return task_set_version not in ("v1", "v2", "v3", "v4", "v5")


#: v6 density-slot cycles, keyed by which template pools a tier actually offers.
#: Each slot emits a clause shape by INDEX, never by seed, so both benchmark
#: splits carry the same gold-item cardinality AND the same kind composition.
#:
#: The four-slot cycle reproduces v5's expected mix EXACTLY rather than inventing
#: a new one. Per slot, v5 drew kind 50/50 and then fired `nhst_with_es` on half
#: of the nhst slots, giving E[nhst items] = 0.50, E[effect_size items] = 0.75,
#: E[items] = 1.25. Over the four slots below: `nhst_with_es` emits one item of
#: each kind, `nhst` emits one nhst item, and the two `effect_size` slots emit one
#: each -- 2 nhst and 3 effect-size items per 4 slots, the same three numbers.
#:
#: That equality only holds over a WHOLE number of cycles, and a task holds just
#: 1-3 slots. Restarting the cycle at every task therefore over-weights its first
#: entry: measured, that variant put 117 nhst items in 168 across the split where
#: v5 averages 104.95 in 162.05 over 40 seeds -- seed-independent, but a different
#: arena. `generate` instead carries a running per-tier slot counter across tasks
#: (`v6_slot_offsets`), the same seed-independent running-counter shape as tier
#: 5's `t5_offset`, so each tier consumes complete cycles and the mix lands on
#: v5's expectation instead of on its first cycle entry.
_V6_SLOT_CYCLES = {
    # tier 1: has both an `effect_size` and an `nhst_with_es` pool.
    (True, True): ("nhst_with_es", "effect_size", "nhst", "effect_size"),
    # tiers 2 and 3: an `effect_size` pool but no `nhst_with_es`, so v5's second
    # draw never fired there and only the kind alternation needs pinning.
    (True, False): ("nhst", "effect_size"),
    # Fallbacks for template pools no tier currently ships. They exist so a future
    # tier cannot reintroduce a seed-dependent composition by omitting a pool, and
    # they are exercised by framework/tests, not by any generated task set today.
    (False, True): ("nhst_with_es", "nhst"),
    (False, False): ("nhst",),
}


def _slot_plan_v6(i: int, templates: dict) -> str:
    """Clause shape for density slot `i`, chosen by index and never by seed.

    `i` is the RUNNING slot number for this tier across the whole split, not the
    within-task index -- see `_V6_SLOT_CYCLES` for why that distinction changes
    the arena's composition rather than merely its bookkeeping.

    Returns one of "nhst_with_es" (one nhst item AND its effect size, from one
    clause), "nhst" (one nhst item) or "effect_size" (one effect-size item).
    """
    cycle = _V6_SLOT_CYCLES[(bool(templates.get("effect_size")),
                             bool(templates.get("nhst_with_es")))]
    return cycle[i % len(cycle)]


#: The alpha the `p < {p_threshold}` templates print, as a number. `_ALPHA` is
#: the decision threshold the labeller compares against; this is the value that
#: appears IN THE TEXT and, from v5, in the gold that grades it. Same number,
#: two different jobs, so a future change to one does not silently move the other.
_ALPHA_PRINTED = 0.05

#: `<` / `>` / `=` in the entity-escaped form tier 3 emits.
_HTML_REL = {"<": "&lt;", ">": "&gt;", "=": "="}


def _apa_p_rel(stat_id: str, value, df1, df2, p) -> str:
    """The relation a real paper would print for this p: "<", ">" or "=".

    Pure and RNG-free by construction: it is called AFTER the fields are drawn and
    consumes no randomness, so adding it cannot shift the stream that frozen task
    sets depend on. Returns "=" whenever the true p is unavailable (U, W, BF10 have
    no closed form), which keeps those items exactly as they are today.
    """
    if p is None:
        return "="
    tp = _two_sided_p(stat_id, value, df1, df2)
    if tp is None:
        return "="
    if tp < 0.0005:
        return "<"
    if tp > 0.999:
        return ">"
    return "="


def _apa_alpha_rel(stat_id: str, value, df1, df2, p) -> str:
    """The relation a paper would print against the .05 THRESHOLD: "<", ">" or "=".

    A different question from `_apa_p_rel`, which asks where the true p sits
    relative to the 3-dp APA grid. This one asks where it sits relative to alpha,
    because that is the only claim `p {rel} .05` makes. Pure and RNG-free for the
    same reason: it is called after the fields are drawn and consumes nothing.

    Falls back to the REPORTED p when the statistic has no closed form, so the
    printed claim is still the one the gold grades. Every template that uses this
    hardcodes `t(`, so the fallback is unreachable today and exists so a future
    template cannot silently print an unfounded inequality.

    Returns "=" only when the true p is EXACTLY .05, where `p = .05` is the one
    true statement of the three. Measure-zero, and handled rather than rounded
    past: `<` and `>` are both false there, so a missing branch would manufacture
    a reporting error out of an item that is correct.
    """
    tp = _two_sided_p(stat_id, value, df1, df2)
    if tp is None:
        tp = p
    if tp is None:
        return "="
    if tp < _ALPHA:
        return "<"
    if tp > _ALPHA:
        return ">"
    return "="


def _apply_apa_p(render_fields: dict, gold_fields: dict, v4: bool,
                 tmpl: str = "", v5: bool = False) -> None:
    """Stamp the APA relation onto BOTH the render fields and the gold item.

    Two dicts on purpose. The render dict feeds `{p_rel}` in the template, so the
    task TEXT reads `p < .001`. The gold dict feeds `_label_verification`, which
    cannot judge whether a printed claim is true without knowing which claim was
    printed. Writing one and not the other is the failure mode this signature is
    shaped to prevent: the text would say `p < .001` while the answer key still
    graded it as `p = .001`.

    `tmpl` is the template about to render these fields, and v5 needs it because
    ONE template family makes a different claim from the rest. `p {rel}
    {p_threshold}` prints the constant `.05`, never the drawn p, so for those
    items the gold's `p` must BECOME the .05 the reader sees and the relation
    must be measured against alpha. Writing the stamp without consulting the
    template is exactly how v4 came to grade a claim its own text never printed
    (see `_is_v5_plus`), so the template is threaded here rather than guessed at.
    """
    if not v4:
        return
    rel = _apa_p_rel(render_fields.get("stat_id"), render_fields.get("value"),
                     render_fields.get("df1"), render_fields.get("df2"),
                     render_fields.get("p"))
    render_fields["p_rel"] = rel
    gold_fields["p_rel"] = rel
    if not (v5 and "{p_threshold}" in tmpl):
        return
    arel = _apa_alpha_rel(render_fields.get("stat_id"), render_fields.get("value"),
                          render_fields.get("df1"), render_fields.get("df2"),
                          render_fields.get("p"))
    render_fields["p_alpha_rel"] = arel
    render_fields["p_alpha_rel_html"] = _HTML_REL[arel]
    # The drawn p is DISCARDED for these items, deliberately. It was never
    # printed, so an answer key that keeps it asserts a value absent from the
    # input -- the same shape as the ghost CIs `_ci_is_printed` removed.
    gold_fields["p"] = _ALPHA_PRINTED
    gold_fields["p_rel"] = arel


def _build_nhst_coherent_fields(rng: random.Random, nhst_catalog: list[dict],
                                stat_id: str | None = None) -> dict:
    """A truthful NHST result whose reported p IS the p its statistic implies.

    v3+. `_build_nhst_fields` draws the statistic and the p independently, which
    is right for v1/v2 (frozen) and wrong as a description of a "truthful" item:
    measured on v2, 37 of 84 straddle alpha and 74 of 84 differ at the reported
    precision, so the arena's clean class was 10 items by luck.

    Only closed-form statistics are used, so the p is verifiable rather than
    stipulated; anything else falls back to the independent draw and is labelled
    honestly by `_label_verification`.
    """
    spec = None
    if stat_id is not None:
        spec = next((s for s in nhst_catalog if s["id"] == stat_id), None)
    if spec is None:
        spec = rng.choice(nhst_catalog)
    df1 = rng.randint(8, 200)
    df2 = rng.randint(2, 12)
    if spec["id"] == "r":
        df1 = rng.randint(15, 200)
    val = round(rng.uniform(*spec["range"]), 2)
    p = _consistent_p(rng, spec["id"], val, df1, df2)
    return {"stat_id": spec["id"], "label": spec["label_pattern"],
            "df1": df1, "df2": df2, "value": val, "p": p}


def _template_stat_id(tmpl: str, nhst_catalog: list[dict] | None = None) -> str | None:
    """The statistic a template HARDCODES, or None when it renders `{stat_str}`.

    Tier 1-4 templates spell the notation out — ``t({df1})``, ``F({df1}, {df2})``,
    ``r({df1})``, ``chi2({df1})`` — while the statistic was drawn independently
    from the catalog. That is TODO item D: a Cochran Q (range 0.1-50) rendered as
    ``r(28) = 48.7``, a correlation outside [-1, 1], with gold recording ``Q`` for
    text that says ``r``.

    Matched on the LABEL AS A TOKEN followed by ``(`` (or ``=`` for df-less
    labels), tolerating the italic markers tier 2 adds (``*t*(``) — never a bare
    substring search. Longest label first, because ``BF10`` contains ``F`` and a
    shortest-first scan would render every Bayes factor as an F-test.
    """
    catalog = nhst_catalog if nhst_catalog is not None else _load_yaml(CATALOGS_DIR / "nhst_stats.yaml")
    for spec in sorted(catalog, key=lambda s: len(s["label_pattern"]), reverse=True):
        label = re.escape(spec["label_pattern"])
        # `\*?` for tier 2's *t*(...); the lookbehind stops `t` matching inside a word.
        if re.search(rf"(?<![A-Za-z0-9_])\*?{label}\*?\s*[(=]", tmpl):
            return spec["id"]
    return None


def _build_nhst_fields(rng: random.Random, nhst_catalog: list[dict],
                       stat_id: str | None = None) -> dict:
    """Draw a truthful NHST result.

    ``stat_id`` pins which statistic to draw so it matches the template that will
    render it (see :func:`_template_stat_id`). Callers pass it for every template
    that hardcodes its notation; without it the draw is free, which is correct only
    for templates rendering a pre-built ``{stat_str}``.

    An unknown ``stat_id`` falls back to a free draw rather than raising: a new
    template with unrecognised notation should degrade to the old behaviour, not
    break generation. The label-coherence tests would catch it.
    """
    spec = None
    if stat_id is not None:
        spec = next((s for s in nhst_catalog if s["id"] == stat_id), None)
    if spec is None:
        spec = rng.choice(nhst_catalog)
    df1 = rng.randint(8, 200)
    df2 = rng.randint(2, 12)
    val = round(rng.uniform(*spec["range"]), 2)
    # NOTE: p remains independent of the statistic here, deliberately. These are the
    # generic tier 1-4 items; the recompute-consistent clean control is the dedicated
    # `nhst_consistent` item (see _build_nhst_consistent_fields), where label, value
    # and p are built together. Item D fixed the label<->statistic mismatch; it did
    # not make every truthful item p-consistent, which is a separate property.
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
        # v4 stamps "<" / ">" here. Everything older has no p_rel, and "=" makes
        # `p {p_rel} {p}` render EXACTLY the `p = {p}` those task sets emitted, so
        # the template edit is byte-safe for frozen history.
        derived.setdefault("p_rel", "=")
    if "p2" in fields:
        derived.setdefault("p2_rel", "=")
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
                nhst_catalog, es_catalog, *, pin_labels: bool = True, v3: bool = False,
                v4: bool = False, v5: bool = False, v6: bool = False,
                slot_offsets: dict | None = None) -> tuple[dict, dict]:
    """Generate one task envelope + its ground truth."""
    templates_path = TEMPLATES_DIR / f"tier{tier}_{ {1:'baseline',2:'notation',3:'formatting'}[tier] }.yaml"
    templates = _load_yaml(templates_path)

    parts: list[str] = []
    items: list[dict] = []  # ground-truth items
    n_stats = density
    # v6 only: where this task's slots sit in its tier's running cycle. Advanced
    # below so the NEXT task continues the cycle instead of restarting it; tier 6
    # shares the same counters, so its sub-sections continue them too.
    _slot_base = 0 if slot_offsets is None else slot_offsets.get(tier, 0)
    for i in range(n_stats):
        if v6:
            # v6: composition by INDEX, so both splits are made of the same thing.
            # `rng` is deliberately NOT consulted here -- consuming it is what made
            # per-task kind mix and cardinality a property of the seed. See
            # _is_v6_plus for the measured cross-split gap this closes.
            _slot = _slot_plan_v6(_slot_base + i, templates)
            kind = "effect_size" if _slot == "effect_size" else "nhst"
            with_es = _slot == "nhst_with_es"
        else:
            # v1-v5, frozen: the draws stay, and stay in this order, because the
            # committed gold and published scores were computed from this stream.
            kind = rng.choice(["nhst", "effect_size"]) if templates.get("effect_size") else "nhst"
            with_es = bool(v3 and kind == "nhst" and templates.get("nhst_with_es")
                           and rng.random() < 0.5)
        if with_es:
            # v3+: the canonical APA form, an effect size reported BESIDE the
            # statistic it came from. Yields TWO gold items from one clause.
            # `{stat_str}` is pre-rendered so label and df-arity are coherent.
            tmpl = rng.choice(templates["nhst_with_es"])
            nf = _build_nhst_coherent_fields(rng, nhst_catalog)
            nf["stat_str"] = _render_stat_str(nf)
            ef = _build_clean_es_fields(rng, es_catalog)
            nf.update({k: ef[k] for k in ("es_label", "es", "ci_lo", "ci_hi")})
            _nhst_gold = {"test_type": nf["stat_id"], "df1": nf["df1"],
                          "df2": nf["df2"], "value": nf["value"], "p": nf["p"]}
            _apply_apa_p(nf, _nhst_gold, v4, tmpl, v5)
            sentence = _instantiate_template(tmpl, nf)
            items.append({
                "kind": "nhst_stat", "fields": _nhst_gold,
                "truthful": True, "deception_kind": None,
            })
            items.append({
                "kind": "effect_size",
                "fields": {"effect_size_type": ef["es_label"], "value": ef["es"],
                           "ci_low": ef["ci_lo"], "ci_high": ef["ci_hi"]},
                "truthful": True, "deception_kind": None,
            })
            parts.append(sentence)
            continue
        if kind == "nhst":
            # Template FIRST, then draw the statistic it hardcodes (item D). Drawing
            # them independently is what rendered a Cochran Q as `r(28) = 48.7`.
            #
            # v1 keeps the ORIGINAL order (fields, then template). The reorder alone
            # shifts the RNG stream — 23/36 v1 tasks drifted from their committed gold
            # even with the statistic left unpinned — and v1 is frozen history.
            if pin_labels:
                # v5 wording lives under its own key so v1-v4 keep rendering the
                # exact bytes their committed gold was computed on. Falls back to
                # `nhst` when a tier has no v5 variant (tiers 1 and 4 already
                # render `{p_rel}` everywhere and need none).
                pool = templates.get("nhst_v5") if v5 else None
                tmpl = rng.choice(pool or templates["nhst"])
                build = _build_nhst_coherent_fields if v3 else _build_nhst_fields
                fields = build(rng, nhst_catalog, _template_stat_id(tmpl, nhst_catalog))
            else:
                fields = _build_nhst_fields(rng, nhst_catalog)
                tmpl = rng.choice(templates["nhst"])
            _nhst_gold = {"test_type": fields["stat_id"], "df1": fields["df1"],
                          "df2": fields["df2"], "value": fields["value"], "p": fields["p"]}
            _apply_apa_p(fields, _nhst_gold, v4, tmpl, v5)
            sentence = _instantiate_template(tmpl, fields)
            items.append({
                "kind": "nhst_stat", "fields": _nhst_gold,
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

    if slot_offsets is not None:
        slot_offsets[tier] = _slot_base + n_stats

    # Tier 1 carries one fully-coherent "nhst_consistent" control per task — the
    # matched clean look-alike for the tier-5 nhst_inconsistent deception (correct
    # label/df, p = recomputed two-sided p). Emitted deterministically (always one,
    # after the density items) so both splits carry it equally and parity is
    # unaffected; only the wording/value is seed-driven. Tier 1 is the natural home
    # (clean APA baseline); the OCR/comma noise of tiers 2-3 would obscure the point.
    if tier == 1 and templates.get("nhst_consistent"):
        cf = _build_nhst_consistent_fields(rng, nhst_catalog)
        cf["stat_str"] = _render_stat_str(cf)
        _cf_gold = {"test_type": cf["stat_id"], "df1": cf["df1"], "df2": cf["df2"],
                    "value": cf["value"], "p": cf["p"]}
        _cf_tmpl = rng.choice(templates["nhst_consistent"])
        _apply_apa_p(cf, _cf_gold, v4, _cf_tmpl, v5)
        parts.append(_instantiate_template(_cf_tmpl, cf))
        items.append({
            "kind": "nhst_stat", "fields": _cf_gold,
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


def _build_tier4_task(task_id, rng, density, nhst, es, *, pin_labels: bool = True, v3: bool = False,
                      v4: bool = False, v5: bool = False):
    templates = _load_yaml(TEMPLATES_DIR / "tier4_density.yaml")
    parts, items = [], []
    for _ in range(density):
        # Template first (item D). `{value2}` is rendered inside the SAME hardcoded
        # notation as `{value}` in the two-stat tier-4 templates, so f2 is drawn for
        # that statistic too — otherwise the second number in the sentence is out of
        # range for the label printed beside it.
        if pin_labels:
            tmpl = rng.choice(templates["nhst"])
            stat_id = _template_stat_id(tmpl, nhst)
            build = _build_nhst_coherent_fields if v3 else _build_nhst_fields
            f1 = build(rng, nhst, stat_id)
            f2 = build(rng, nhst, stat_id)
        else:  # v1 order, frozen
            f1 = _build_nhst_fields(rng, nhst)
            f2 = _build_nhst_fields(rng, nhst)
            tmpl = rng.choice(templates["nhst"])
        f1["value2"] = f2["value"]; f1["p2"] = f2["p"]
        _f1_gold = {"test_type": f1["stat_id"], "df1": f1["df1"], "df2": f1["df2"],
                    "value": f1["value"], "p": f1["p"]}
        _apply_apa_p(f1, _f1_gold, v4, tmpl, v5)
        # The SECOND statistic is printed but carries no gold item, so nothing
        # grades it -- which is exactly why it must not be left printing a false
        # `p = .001`. A tool that recomputes p would flag it correctly and collect
        # a false alarm against an item the answer key does not contain.
        if v4:
            f1["p2_rel"] = _apa_p_rel(f2.get("stat_id"), f2.get("value"),
                                      f2.get("df1"), f2.get("df2"), f2.get("p"))
        parts.append(_instantiate_template(tmpl, f1))
        items.append({"kind": "nhst_stat", "fields": _f1_gold,
                      "truthful": True, "deception_kind": None})
    # One clean ANOVA control per task: the matched truthful look-alike for the
    # tier-5 wrong_df deception (df2 = N - k is consistent here). Emitted
    # deterministically (always, after the density items) so both splits carry it
    # equally and parity is unaffected; only the wording is seed-driven.
    af = _build_anova_design_fields(rng)
    _af_gold = {"test_type": af["stat_id"], "df1": af["df1"], "df2": af["df2"],
                "value": af["value"], "p": af["p"]}
    _af_tmpl = rng.choice(templates["clean_anova"])
    _apply_apa_p(af, _af_gold, v4, _af_tmpl, v5)
    parts.append(_instantiate_template(_af_tmpl, af))
    items.append({"kind": "nhst_stat", "fields": _af_gold,
                  "truthful": True, "deception_kind": None})
    text = " ".join(parts)
    _assign_spans(text, items)
    return {"task_id": task_id, "arena_id": "stats-extraction-v1", "task_set_version": "v1",
            "visibility": "public",
            "difficulty": {"tier": 4, "density": density, "deception_rate": 0},
            "input": {"text": text, "tier": 4, "format_hint": "plain_text"}}, {"items": items}


def _build_tier5_task(task_id, rng, density, nhst, es, deception_kinds_yaml, kind_offset=0, *, pin_labels: bool = True):
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
            # Item D applies to the kinds whose LABEL is meant to be right — the
            # planted error is a wrong p / a wrong value / a missing df, not a wrong
            # test name. `swapped_test_label` is the deliberate exception: its whole
            # deception is rendering the statistic under the wrong notation, so it
            # keeps drawing freely and must never be coerced to match its template.
            stat_id = (None if (kind == "swapped_test_label" or not pin_labels)
                       else _template_stat_id(tmpl, nhst))
            f = _build_nhst_fields(rng, nhst, stat_id)
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


def _deception_rate_bucket(n_deception: int, n_items: int) -> int:
    """Map a realised deception count onto arena.yaml's declared 0-5 axis.

    `difficulty_axes` documents `deception_rate` as "0=none, 5=>=50%". This is
    the only place that mapping exists, so the stamp is derived from the items
    rather than written by hand beside them.
    """
    if n_items <= 0 or n_deception <= 0:
        return 0
    frac = n_deception / n_items
    for bucket, ceiling in enumerate((0.10, 0.20, 0.30, 0.40, 0.50), start=1):
        if frac < ceiling:
            return bucket
    return 5


def _build_tier6_task(task_id, rng, density, nhst, es, deception_kinds_yaml, *, pin_labels: bool = True, v3: bool = False,
                      v4: bool = False, v5: bool = False, v6: bool = False,
                      slot_offsets: dict | None = None):
    """Compose a multi-section results section out of tier 1-5 paragraphs."""
    composition = _load_yaml(TEMPLATES_DIR / "tier6_results_section.yaml")
    # FOUR titles make `sub_tier = (i % 5) + 1` top out at 4, so the deception
    # branch below was unreachable dead code while the envelope stamped
    # `deception_rate: 3`. Measured on task set v2: tier 6 held 66 of 144 gold
    # items (46%) and contained ZERO deception. v3 adds a fifth title; v1/v2 keep
    # four, because their gold is frozen.
    titles = composition["section_titles_v3"] if v3 else composition["section_titles"]
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
                                        deception_kinds_yaml, kind_offset=i,
                                        pin_labels=pin_labels)
        elif sub_tier == 4:
            env, gt = _build_tier4_task(f"{task_id}-sub{i}", rng, density, nhst, es,
                                        pin_labels=pin_labels, v3=v3, v4=v4, v5=v5)
        else:
            env, gt = _build_task(f"{task_id}-sub{i}", rng, sub_tier, density, nhst, es,
                                  pin_labels=pin_labels, v3=v3, v4=v4, v5=v5, v6=v6,
                                  slot_offsets=slot_offsets)
        body = env["input"]["text"]
        if i > 0:
            body = body + " " + rng.choice(crossrefs) + "."
        sections.append(section_tmpl.format(title=title, body=body))
        items_all.extend(gt["items"])
    full_text = "\n\n".join(sections)

    # Recompute spans against the full composed text.
    _assign_spans(full_text, items_all)

    # COMPUTE the stamp, never assert it. The literal `3` here was false for every
    # v1/v2 task: the branch that would have produced deception was unreachable.
    # A difficulty axis is read by every published per-player report, so a wrong
    # stamp mislabels a breakdown no reader can check.
    n_deception = sum(1 for it in items_all if not it.get("truthful", True))
    deception_rate = _deception_rate_bucket(n_deception, len(items_all)) if v3 else 3
    return {"task_id": task_id, "arena_id": "stats-extraction-v1", "task_set_version": "v1",
            "visibility": "public",
            "difficulty": {"tier": 6, "density": density, "deception_rate": deception_rate},
            "input": {"text": full_text, "tier": 6, "format_hint": "plain_text"}}, {"items": items_all}


def _require_scipy() -> None:
    """Refuse to generate anything when scipy is unavailable.

    The module-level guard above exists so that IMPORTING this file on a
    scipy-less box degrades instead of crashing — reading `ground_truth()` or a
    pure helper is harmless. GENERATING is not: without scipy,
    `_decisive_sig_stat` falls back to `rng.uniform(*spec["range"])`, a draw with
    no significance property, and the nhst_consistent control — the one item in
    the arena whose entire purpose is to be internally coherent — starts emitting
    text like "the test was significant, chi2(158) = 3.09, p = 0.057".

    That is not degraded output, it is WRONG output, and it is silent. On
    2026-08-10 a corrupted numpy in the local venv (`numpy/_core/_dtype.py`
    missing) tripped exactly this: two tests went red in a way that read as a
    regression in frozen v1 history, while `pip check` reported "No broken
    requirements found" because it never imports what it checks. Had gold been
    regenerated on that box instead, text and gold would have drifted TOGETHER
    and every test would have passed — publishing incoherent statistics as
    ground truth with nothing to catch it.

    scipy is a declared, locked dependency (pyproject `scipy>=1.11`,
    requirements.lock `scipy==1.17.1`). Its absence is a broken environment, and
    the correct response to a broken environment is to stop.
    """
    if _HAVE_SCIPY:
        return
    raise RuntimeError(
        "stats-extraction-v1 cannot generate tasks: scipy failed to import, so "
        "significance values would be drawn at random and the coherent controls "
        "would contradict their own prose. Repair the environment "
        "(`pip install --force-reinstall numpy scipy`) and re-run — do not "
        "regenerate gold until `from scipy import stats` succeeds."
    )


# --- verification labelling (added 2026-08-21) -------------------------------
#
# WHY THIS EXISTS. This module's docstring claimed, from the day "TODO item E"
# was staged, that "Truthful NHST items now report a p that is CONSISTENT with
# their statistic+df". The code at _build_nhst_fields says the opposite in an
# explicit NOTE, and only `nhst_consistent` was ever built that way. Measured
# 2026-08-21 over task set v2 revealed: 37 of 84 truthful `nhst_stat` gold items
# have a reported p and a recomputed p on OPPOSITE SIDES of alpha = .05 — real
# statcheck decision errors, labelled `truthful: true`.
#
# That combination was incoherent with the rest of the arena:
# `players/prompts/stats_extraction.txt:3,23-24` ORDERS every player to
# "Recompute the two-sided p implied by the statistic and its df ... Flag it",
# so the better a player followed our own instructions the worse it scored.
# Verified against live effectcheck 0.7.6 (docs/reports/evidence/): on
# `r(129) = -0.82, p = 0.393` it returns decision_error=True,
# reason='reported_ns_computed_sig', p_computed=4.7e-33. It is right.
#
# The fix separates two axes that were conflated in one boolean:
#   `truthful`          — did we PLANT a lie? (intent; only a generator knows)
#   `expected_verdict`  — does the reported set COHERE? (recomputable)
# plus `checkable`, which says whether the question is even askable of this item.
#
# This runs as a pure post-pass over already-built items: no rng call, so the
# task TEXT is byte-identical and `input_hash` is stable. That is what makes this
# a scoring-version change rather than a task-set change — every stored player
# output can be replayed. A shared generator is append-only; reordering an rng
# call here would silently rewrite frozen history.

_UNCHECKABLE_DECEPTIONS = {"statistic_impostor", "missing_info_impostor"}

# A bare "." is NOT a sentence boundary in this arena: every value is a decimal,
# so splitting on it truncates `95% CI [2.56, 2.72]` at its first bound and makes
# every printed interval look absent. Require a terminator followed by whitespace
# or end-of-text, or a newline.
_SENTENCE_BREAK = re.compile(r"[.!?](?:\s|$)|\n")


def _ci_is_printed(text: str, item: dict) -> bool:
    """Is this effect size's CI actually RENDERED in the task text?

    Gold recorded `ci_low`/`ci_high` from the drawn fields unconditionally, but
    two templates -- `"We found a sizeable effect ({es_label} = {es})."` and the
    tier-6 sentences built from it -- print no interval at all. Measured
    2026-08-21: 4 of 49 effect-size gold items in task set v2 carry a GHOST CI,
    present in the answer key and absent from the input.

    Nothing scored CI bounds, so this had never surfaced a wrong number -- it is
    exactly the latent shape where gold asserts something the input does not
    contain, and any future CI check would have charged every player for missing
    an interval that was never printed.

    Checked by looking for BOTH bounds, in any APA rendering, inside the
    sentence carrying the item -- exact, not a bracket-shaped heuristic.
    """
    f = item.get("fields", {}) or {}
    lo, hi = f.get("ci_low"), f.get("ci_high")
    if lo is None or hi is None:
        return False
    cs = item.get("span", {}).get("char_start", 0)
    ce = item.get("span", {}).get("char_end", cs)
    # A bare "." is NOT a sentence boundary here: every value in this arena is a
    # decimal, so splitting on it truncates `[2.56, 2.72]` at its first bound.
    # Require a terminator followed by whitespace or end-of-text.
    starts = [m.end() for m in _SENTENCE_BREAK.finditer(text[:cs])]
    start = starts[-1] if starts else 0
    m = _SENTENCE_BREAK.search(text[ce:])
    end = ce + m.end() if m else len(text)
    sentence = text[start:end]
    return (any(r in sentence for r in _value_renderings(lo))
            and any(r in sentence for r in _value_renderings(hi)))


def _label_verification(gt: dict, text: str = "", *, v4: bool = False) -> dict:
    """Stamp `checkable` + `expected_verdict` on every gold item. Pure math.

    ``v4`` is threaded explicitly rather than inferred from the presence of
    ``p_rel``, because a v4 DECEPTION item carries no ``p_rel`` and would
    otherwise be silently graded under the v3 rule.
    """
    for item in gt.get("items", []):
        kind = item.get("kind")
        f = item.get("fields", {}) or {}
        dk = item.get("deception_kind")
        truthful = item.get("truthful", True)

        if dk == "statistic_impostor":
            # Not a statistic at all. A correct reader rejects it; there is
            # nothing to recompute, so it gets its own verdict rather than
            # being folded into "inconsistent".
            item["checkable"] = True
            item["expected_verdict"] = "not_a_stat"
            continue
        if dk == "missing_info_impostor":
            # Presented as complete with a critical field missing (no df). The
            # honest answer is "I cannot check this" — which is exactly what
            # effectcheck's `insufficient_data` says.
            item["checkable"] = True
            item["expected_verdict"] = "unverifiable"
            continue
        if not truthful:
            # Every other planted kind is a genuine internal contradiction.
            item["checkable"] = True
            item["expected_verdict"] = "inconsistent"
            continue

        if kind == "nhst_stat":
            tp = _two_sided_p(f.get("test_type"), f.get("value"),
                              f.get("df1"), f.get("df2"))
            p = f.get("p")
            if tp is None or p is None:
                item["checkable"] = False
                item["expected_verdict"] = "unverifiable"
            else:
                p, tp = float(p), float(tp)
                rel = f.get("p_rel", "=")
                # v4 prints an INEQUALITY when the true p falls beyond the 3-dp
                # grid, so the claim under test is "tp < .001" / "tp > .999", not
                # "tp rounds to .001". Grading an inequality as a point value is
                # what manufactured 84 of the arena's 85 reporting errors.
                if rel == "<":
                    # `p < .001` asserts significance, and so does `p < .05` --
                    # but `.05 < .05` is False, so a strict comparison reads the
                    # commonest significance claim in psychology as a null one.
                    # The bound is INCLUSIVE for a `<` claim.
                    #
                    # Provably inert for v1-v4, which is why it is not version
                    # gated: `rel` is non-"=" only from v4, where "<" requires the
                    # true p to be below .0005, and `_round_p`'s floor then prints
                    # .001. No v1-v4 item has rel "<" with p == .05. The
                    # byte-for-byte regeneration of all four is the check.
                    claim_sig = p <= _ALPHA
                    coherent_claim = tp < p
                elif rel == ">":
                    claim_sig = False             # `p > .999` asserts null
                    coherent_claim = tp > p
                else:
                    claim_sig = p < 0.05
                    coherent_claim = None         # decided by the rounding test below
                crosses = claim_sig != (tp < 0.05)
                item["checkable"] = True
                item["expected_verdict"] = "inconsistent" if crosses else "ok"
                item["p_recomputed"] = round(tp, 8)
                # TWO GRANULARITIES, and they are not the same question. statcheck
                # calls the first `decision_error` and the second `error`:
                #   crosses alpha        -> the paper's stated CONCLUSION does not
                #                           follow. This is what `expected_verdict`
                #                           requires a player to catch.
                #   differs at the       -> the reported p is simply not the p the
                #   REPORTED precision      statistic implies. Also a real defect.
                #
                # Measured 2026-08-21 over these 84 items: 37 cross alpha, but 74
                # differ at the reported precision. Only 10 are genuinely coherent.
                # So a player flagging a non-alpha-crossing mismatch is RIGHT, and
                # the first version of this labelling charged it as a false alarm —
                # reproducing, one granularity finer, the exact defect this whole
                # change exists to remove. statcheck took 31 such false alarms.
                #
                # `coherent` is therefore recorded separately, and the scorer never
                # charges a false alarm against an item that is not coherent.
                if coherent_claim is None:
                    dec = len(str(f.get("p")).split(".")[1]) if "." in str(f.get("p")) else 0
                    coherent_claim = round(tp, dec) == round(p, dec)
                item["coherent"] = coherent_claim
                # THE FOLD, v4 ONLY. Once the printing is fixed, an incoherent
                # truthful item is a GENUINE misreport (measured: exactly one on
                # the revealed split), so the answer key can simply say so and the
                # scorer's forgiveness cell is no longer needed. Folding this on v3
                # instead would have published 84 of our own rounding clamps as
                # reporting errors -- which is why that option was measured and
                # rejected rather than shipped.
                if v4 and not coherent_claim:
                    item["expected_verdict"] = "inconsistent"
            continue

        if kind == "effect_size":
            # An effect size is verifiable only against something. With a CI we
            # can check the midpoint; bare, there is nothing to check.
            #
            # This is not a convenience: probed against live effectcheck 0.7.6
            # on 2026-08-21, ALL SIX bare-effect-size forms this arena emits
            # return ZERO rows, while all three paired APA forms
            # (`t(48) = 2.31, p = .025, d = 0.65`) parse and PASS. A bare effect
            # size has nothing to verify, so scoring one as a verification
            # failure measures a tool outside its contract.
            has_ci = _ci_is_printed(text, item)
            if not has_ci:
                # Drop the ghost: an answer key must not claim a value the input
                # never showed. See _ci_is_printed.
                f.pop("ci_low", None)
                f.pop("ci_high", None)
            item["checkable"] = bool(has_ci)
            item["expected_verdict"] = "ok" if has_ci else "unverifiable"
            continue

        item["checkable"] = False
        item["expected_verdict"] = "unverifiable"
    return gt


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    """Yield the task envelopes for one benchmark `split`.

    Both splits run the IDENTICAL tier × density × deception matrix below; only
    the seed (chosen by the caller per arena.yaml#benchmark_splits) and the
    visibility/split tags differ. This is what makes the revealed and private
    suites parity-matched by construction (see framework/parity.py). This arena
    is synthetic-only, so the private split has no real-world holdout to append.
    """
    _require_scipy()
    visibility = "public" if split == "revealed" else "held_out"
    # Item D applies to v2+ only; v1 is frozen history. See _pins_labels.
    pin_labels = _pins_labels(task_set_version)
    v3 = _is_v3_plus(task_set_version)
    v4 = _is_v4_plus(task_set_version)
    v5 = _is_v5_plus(task_set_version)
    v6 = _is_v6_plus(task_set_version)
    nhst = _load_yaml(CATALOGS_DIR / "nhst_stats.yaml")
    es = _load_yaml(CATALOGS_DIR / "effect_sizes.yaml")
    deception_kinds = _load_yaml(CATALOGS_DIR / "deception_kinds.yaml")
    n_per_cell = 2
    densities = [1, 2, 3]
    t5_offset = 0  # seed-independent running counter -> deterministic deception-kind coverage
    # v6 only: one seed-independent running slot counter per sub-tier, so each
    # tier consumes whole `_V6_SLOT_CYCLES` cycles across the split rather than
    # restarting mid-cycle at every task. Same shape and purpose as t5_offset.
    v6_slot_offsets: dict[int, int] = {}
    for tier in (1, 2, 3, 4, 5, 6):
        for density in densities:
            for k in range(n_per_cell):
                tid = f"t-tier{tier}-d{density}-{k}-s{_seed_tag(task_set_version, seed, split)}"
                rng = random.Random(_seed_int(task_set_version, seed, tier, density, k))
                if tier in (1, 2, 3):
                    env, gt = _build_task(tid, rng, tier, density, nhst, es,
                                          pin_labels=pin_labels, v3=v3, v4=v4, v5=v5, v6=v6,
                                          slot_offsets=v6_slot_offsets)
                elif tier == 4:
                    env, gt = _build_tier4_task(tid, rng, density, nhst, es,
                                                pin_labels=pin_labels, v3=v3, v4=v4, v5=v5)
                elif tier == 5:
                    env, gt = _build_tier5_task(tid, rng, density, nhst, es, deception_kinds, pin_labels=pin_labels,
                                                kind_offset=t5_offset)
                    t5_offset += density
                else:
                    env, gt = _build_tier6_task(tid, rng, density, nhst, es, deception_kinds,
                                                pin_labels=pin_labels, v3=v3, v4=v4, v5=v5, v6=v6,
                                                slot_offsets=v6_slot_offsets)
                env["split"] = split
                env["visibility"] = visibility
                # The four _build_* helpers hardcode "v1" in the envelope literal.
                # Stamp the version actually being generated, the same way split and
                # visibility are stamped above — otherwise every v2 envelope, and
                # every run record written from it, announces itself as v1.
                env["task_set_version"] = task_set_version
                _GROUND_TRUTH_CACHE[tid] = _label_verification(
                    gt, env["input"]["text"],
                    v4=_is_v4_plus(env.get("task_set_version", "")))
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


def _seed_tag(task_set_version: str, seed: int, split: str = "revealed") -> str:
    """The seed discriminator that goes INTO a `task_id`.

    v1 emitted the raw seed (`…-s<the 9-digit secret>`; the value is deliberately
    not quoted here). Private run records are committed, so
    `runs/v1/escimate__private__v0_6_13.jsonl` publishes the "secret" private seed
    in 36 task_ids — the exposure task-set v2 exists to rotate. Rotating the value
    alone would have re-leaked it after one tournament, so v2+ publishes a
    truncated SHA-256 instead: deterministic and stable, but not invertible.

    v1 keeps the legacy format deliberately. Its 36 stored records key off those
    ids, as do build-data.mjs, the task-detail route, compare-task-diff and
    player-report; changing them would orphan every stored record and link. v1's
    seed stays in `KNOWN_EXPOSED` — the debt is closed by v2 superseding it, not
    by rewriting history.

    v2..v5 hash the seed ALONE, so all four share the same 36 ids with different
    text (a frozen board makes its task links inert for exactly this reason, and
    `_GROUND_TRUTH_CACHE` is keyed on task_id). Frozen, so left as is. From v6 the
    version goes into the hash, so a new task set can never reuse an old id.

    THE v2..v5 HASH IS INVERTIBLE (debt d-c7642f, measured 2026-09-30): 20 of 23
    private seeds are 9 digits and plain-Python sha256 runs ~441k/s on one core,
    so every candidate is hashed in ~38 minutes (seconds on a GPU). Any
    deterministic function of a small secret is. So from v6 the PRIVATE split's
    tag carries nothing derived from the seed -- a constant plus the version, as
    publication-bias v2's `_v2_split_tag` does. The revealed seed is public, so
    its tag may stay a hash. v2..v5 private ids stay as committed; minting v6
    with a fresh private seed is what retires the exposed one.
    """
    if task_set_version == "v1":
        return str(seed)
    if _is_v6_plus(task_set_version):
        if split != "revealed":
            return f"private{task_set_version}"
        return hashlib.sha256(f"{task_set_version}:{seed}".encode("utf-8")).hexdigest()[:8]
    return hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:8]
