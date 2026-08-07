"""Generator for reporting-completeness-v1.

Builds results-section paragraphs out of several reported statistical tests and
injects KNOWN statistical-reporting COMPLETENESS / precision defects. We do NOT
recompute test statistics (that is statcheck / stats-extraction-v1) — we grade
whether each test is reported completely and precisely.

Injected defect categories (these ARE the mistake_kinds, cycled deterministically
so the revealed set covers all nine):
    missing_effect_size     - a significant test reported with no effect size
    imprecise_p             - "p < .05" where an exact p is knowable
    impossible_p_zero       - "p = .000" (a p-value can never be exactly zero)
    missing_ci              - an estimate reported with no confidence interval
    missing_df              - a test statistic reported without its degrees of freedom
    nonsig_as_support       - a p > .05 result framed as supporting the hypothesis
    unspecified_test        - a significance claim with a p-value but NO test statistic
                              named (just "the groups differed significantly, p = .03")
    no_correction           - many comparisons interpreted at p < .05 with NO
                              multiplicity correction (Bonferroni/Holm/FDR) mentioned
    percent_count_mismatch  - a reported "<count> of <N> (<pct>%)" where the % is
                              arithmetically inconsistent with count/N

Each new kind has a MATCHED CLEAN CONTROL look-alike (the honest version that must
NOT be flagged), rendered via CLEAN_CONTROL_MODES and surfaced in the T2 false-alarm
trap and the T4 subtle tier:
    clean_count_match - a count/percent clause whose % IS consistent (vs
                        percent_count_mismatch)
    clean_corrected   - a many-comparisons report that DOES state a correction (vs
                        no_correction)
    (unspecified_test's clean look-alike is the ordinary complete sentence, which
     names the test statistic.)

Dual-benchmark (revealed/private): both splits run the IDENTICAL tier matrix and
assign the injected-defect KIND deterministically (index-driven, seed-independent),
so every split covers all nine categories. Only the concrete VALUES / wording are
rng-driven, so revealed and private content differ. This is what makes
framework/parity.py pass. The revealed seed (0) is committed in arena.yaml; the
private seed is the gitignored secret. Gold is regenerated from the seed and served
from the in-process cache (registry-free) — see ground_truth().
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import yaml

ARENA_DIR = Path(__file__).resolve().parent
CATALOGS_DIR = ARENA_DIR / "catalogs"

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

# The nine injected defects. Order is load-bearing: T3/T4 cycle through this list
# by index so every category is covered deterministically in every split.
DEFECT_KINDS = [
    "missing_effect_size",
    "imprecise_p",
    "impossible_p_zero",
    "missing_ci",
    "missing_df",
    "nonsig_as_support",
    "unspecified_test",
    "no_correction",
    "percent_count_mismatch",
]

# Clean-control modes: a modes entry of None means "render a clean, complete
# sentence" (no defect). These named clean modes render the *confusable look-alike*
# of a specific defect — a complete, honest sentence that carries the same
# structural element the defect corrupts (a consistent percent/count clause; a
# many-comparisons report that DOES state a correction). They produce NO gold flag
# and are the T2-trap / T4-subtle controls that must NOT be flagged.
CLEAN_CONTROL_MODES = [
    "clean_count_match",   # look-alike for percent_count_mismatch
    "clean_corrected",     # look-alike for no_correction
]

# Maps each new defect kind to the clean look-alike used beside it in the T4 subtle
# tier. unspecified_test's look-alike is the ordinary complete sentence (None),
# which names the test statistic the defect omits.
_CONFUSABLE_CLEAN = {
    "percent_count_mismatch": "clean_count_match",
    "no_correction": "clean_corrected",
    "unspecified_test": None,
}


def _load_tests() -> list[dict]:
    with (CATALOGS_DIR / "tests.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _render_sentence(opt: dict, defect: str | None) -> tuple[str, dict | None]:
    """Render one reported-test sentence.

    `defect` is one of: None (clean complete sentence), a member of DEFECT_KINDS
    (inject that defect), or a member of CLEAN_CONTROL_MODES (a clean look-alike
    that carries a defect's structural element but is honest -> no flag).

    Returns (sentence_text, span_meta) where span_meta, if not None, locates the
    defect WITHIN the sentence as {sub: substring, category}. The caller resolves
    sub -> absolute char offsets in the assembled paragraph.

    A clean sentence reports: statistic (incl. df) + exact p + effect size + CI,
    plus an honest interpretation. Each defect knocks out / corrupts exactly one
    of those while leaving the rest complete.
    """
    stat = opt["stat"]
    stat_nodf = opt["stat_nodf"]
    p_exact = opt["p_exact"]
    es = opt["es"]
    ci = opt["ci"]
    supported = opt["claim_supported"]

    if defect is None:
        # Complete, precise, honestly framed.
        s = f"The effect was significant, {stat}, {p_exact}, {es}, {ci}, {supported}."
        return s, None

    # ----- clean-control look-alikes (carry a defect's structure, but honest) ----
    if defect == "clean_count_match":
        # A consistent percent/count clause in front of a complete report.
        good = opt["count_clause_good"]
        s = f"{good}. The effect was significant, {stat}, {p_exact}, {es}, {ci}, {supported}."
        return s, None

    if defect == "clean_corrected":
        # A many-comparisons report that DOES state a multiplicity correction.
        n_tests = opt["n_tests_clause"]
        corrected = opt["claim_corrected"]
        s = f"{n_tests}, the effect was significant, {stat}, {p_exact}, {es}, {ci}, {corrected}."
        return s, None

    if defect == "missing_effect_size":
        # Significant test, df + exact p + CI present, but NO effect size.
        s = f"The effect was significant, {stat}, {p_exact}, {ci}, {supported}."
        # Anchor the flag on the statistic+p clause that lacks an ES.
        return s, {"sub": f"{stat}, {p_exact}", "category": defect}

    if defect == "imprecise_p":
        p_vague = opt["p_vague"]
        s = f"The effect was significant, {stat}, {p_vague}, {es}, {ci}, {supported}."
        return s, {"sub": p_vague, "category": defect}

    if defect == "impossible_p_zero":
        p_zero = opt["p_zero"]
        s = f"The effect was significant, {stat}, {p_zero}, {es}, {ci}, {supported}."
        return s, {"sub": p_zero, "category": defect}

    if defect == "missing_ci":
        # Complete except the CI is absent.
        s = f"The effect was significant, {stat}, {p_exact}, {es}, {supported}."
        return s, {"sub": f"{es}", "category": defect}

    if defect == "missing_df":
        # Statistic reported WITHOUT degrees of freedom.
        s = f"The effect was significant, {stat_nodf}, {p_exact}, {es}, {ci}, {supported}."
        return s, {"sub": stat_nodf, "category": defect}

    if defect == "nonsig_as_support":
        # A genuinely non-significant p (> .05) framed as supporting H.
        p_nonsig = opt["p_nonsig_exact"]
        claim = opt["claim_nonsig_as_support"]
        s = f"The effect was not significant, {stat}, {p_nonsig}, {es}, {ci}, {claim}."
        return s, {"sub": claim, "category": defect}

    if defect == "unspecified_test":
        # A significance claim with a p-value + effect size + CI but NO test
        # statistic named at all (no t/F/chi^2/r and thus no df either). Distinct
        # from missing_df, which keeps the statistic but drops only the df.
        s = f"The groups differed significantly, {p_exact}, {es}, {ci}, {supported}."
        # Anchor on the bare significance claim that lacks any test statistic.
        return s, {"sub": "The groups differed significantly", "category": defect}

    if defect == "no_correction":
        # Many comparisons interpreted at p < .05 with NO multiplicity correction.
        n_tests = opt["n_tests_clause"]
        uncorrected = opt["claim_uncorrected"]
        s = f"{n_tests}, the effect was significant, {stat}, {p_exact}, {es}, {ci}, {uncorrected}."
        return s, {"sub": uncorrected, "category": defect}

    if defect == "percent_count_mismatch":
        # A "<count> of <N> (<pct>%)" clause whose percentage does NOT match
        # count/N; the surrounding test report stays complete.
        bad = opt["count_clause_bad"]
        s = f"{bad}. The effect was significant, {stat}, {p_exact}, {es}, {ci}, {supported}."
        return s, {"sub": bad, "category": defect}

    raise ValueError(f"unknown defect {defect!r}")


def _resolve_span(paragraph: str, sentence_start: int, sub: str) -> dict:
    """Locate `sub` within the sentence beginning at sentence_start, return span."""
    local = paragraph.find(sub, sentence_start)
    if local < 0:  # pragma: no cover - defensive; sub always present by construction
        raise ValueError(f"span substring not found: {sub!r}")
    return {"text": sub, "char_start": local, "char_end": local + len(sub)}


def _assemble(task_id, tier, sentences_modes, rng, split, visibility) -> tuple[dict, dict]:
    """sentences_modes: list of (test_template, defect_or_None)."""
    parts: list[str] = []
    pending_spans: list[tuple[int, str, str]] = []  # (sentence_index, sub, category)
    for test in sentences_modes:
        template, defect = test
        opt = rng.choice(template["value_options"])
        sentence, span_meta = _render_sentence(opt, defect)
        sent_idx = len(parts)
        parts.append(sentence)
        if span_meta is not None:
            pending_spans.append((sent_idx, span_meta["sub"], span_meta["category"]))

    # Join into a paragraph and resolve every pending span to absolute offsets.
    paragraph = " ".join(parts)
    # Precompute each sentence's absolute start offset in the joined paragraph.
    starts: list[int] = []
    cursor = 0
    for i, sent in enumerate(parts):
        starts.append(cursor)
        cursor += len(sent) + 1  # +1 for the joining space

    flags: list[dict] = []
    kinds: list[str] = []
    for sent_idx, sub, category in pending_spans:
        span = _resolve_span(paragraph, starts[sent_idx], sub)
        flags.append({"span": span, "category": category})
        kinds.append(category)

    n_defects = len(flags)
    envelope = {
        "task_id": task_id,
        "arena_id": "reporting-completeness-v1",
        "task_set_version": "v1",
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "n_defects": n_defects},
        "input": {
            "text": paragraph,
        },
    }
    gold = {
        "flags": flags,
        # mistake_kinds: the injected-defect labels in this task. [] / ["clean"]
        # for a clean control (parity checker buckets by this).
        "mistake_kinds": kinds if kinds else [],
    }
    return envelope, gold


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    visibility = "public" if split == "revealed" else "held_out"
    tests = _load_tests()
    n = len(tests)
    n_kinds = len(DEFECT_KINDS)

    def emit(tier, idx, sentences_modes):
        tid = f"rc-t{tier}-{idx}-s{seed}"
        rng = random.Random(_seed_int(task_set_version, seed, tier, idx))
        env, gt = _assemble(tid, tier, sentences_modes, rng, split, visibility)
        _GROUND_TRUTH_CACHE[tid] = gt
        return env

    # T1: clean/simple — every reported test is complete and precise. No defects.
    for k in range(3):
        # A small number of clean sentences (rotated so wording varies).
        chosen = [tests[(k + j) % n] for j in range(3)]
        yield emit(1, k, [(t, None) for t in chosen])

    # T2: FALSE-ALARM TRAP — paragraphs that look defect-prone (every test
    # reported) but are ALL complete & precise. A good player must flag NOTHING.
    # We rotate through different templates so the trap covers varied prose, AND
    # we interleave the clean-control look-alikes for the structural defects (a
    # CONSISTENT percent/count clause; a many-comparisons report that DOES state a
    # correction) — these look the most "flaggable" yet are honest, so they are the
    # hardest part of the trap. Assignment is index-driven (no rng) -> parity holds.
    n_clean = len(CLEAN_CONTROL_MODES)
    for k in range(3):
        chosen = [tests[(2 * k + j) % n] for j in range(4)]
        modes = []
        for j, t in enumerate(chosen):
            # Slot one clean-control look-alike per task at a rotating position;
            # the rest are ordinary complete sentences.
            if j == k % len(chosen):
                modes.append((t, CLEAN_CONTROL_MODES[k % n_clean]))
            else:
                modes.append((t, None))
        yield emit(2, k, modes)

    # T3: exactly ONE injected defect, cycling through ALL six categories so every
    # kind is covered. The defect rides on one test; the rest are clean.
    for i in range(n_kinds):
        defect = DEFECT_KINDS[i]
        # Put the defect on test i (mod n); surround with clean reports.
        modes = []
        defect_pos = i % n
        for j in range(n):
            modes.append((tests[j], defect if j == defect_pos else None))
        # Keep paragraphs compact: defect sentence + two clean neighbours.
        compact = [modes[defect_pos]]
        compact += [modes[(defect_pos + 1) % n], modes[(defect_pos + 2) % n]]
        yield emit(3, i, compact)

    # T4: SUBTLE — one defect hidden in a longer paragraph of otherwise complete
    # reports. Same cycling so every category appears subtly too. For the kinds
    # with a CONFUSABLE clean look-alike (percent_count_mismatch, no_correction),
    # we place that honest look-alike in the very next slot, so the player must
    # distinguish a mismatched % from a consistent one (and an uncorrected
    # many-comparisons report from a corrected one) in the same paragraph. This is
    # where the realism lives. Assignment stays index-driven (no rng) -> parity.
    for i in range(n_kinds):
        defect = DEFECT_KINDS[i]
        defect_pos = (i + 3) % n
        lookalike = _CONFUSABLE_CLEAN.get(defect)
        lookalike_pos = (defect_pos + 1) % n if lookalike is not None else None
        modes = []
        for j in range(n):
            if j == defect_pos:
                modes.append((tests[j], defect))
            elif lookalike_pos is not None and j == lookalike_pos:
                modes.append((tests[j], lookalike))
            else:
                modes.append((tests[j], None))
        yield emit(4, i, modes)  # all n tests, exactly one defective

    # T5: MULTIPLE co-occurring defects (deterministic distinct categories), rest
    # complete. Three defects per task, cycling the starting category.
    for k in range(3):
        defects_here = [DEFECT_KINDS[(k * 2 + o) % n_kinds] for o in range(3)]
        modes = []
        di = 0
        for j in range(n):
            if di < len(defects_here) and j % 2 == 0:
                modes.append((tests[j], defects_here[di]))
                di += 1
            else:
                modes.append((tests[j], None))
        yield emit(5, k, modes)

    # T6: COMPOSITION — every test present, a deterministic mix of clean and all
    # nine defect categories across the paragraph. The per-task offset strides by
    # n tests so successive tasks rotate the whole kind set into view (with 7 tests
    # and 9 kinds, one task can't show them all; three tasks do). Index-driven.
    for k in range(3):
        modes = []
        for j, t in enumerate(tests):
            slot = (j + k * n) % (n_kinds + 1)  # 0..n_kinds; index n_kinds => clean
            defect = DEFECT_KINDS[slot] if slot < n_kinds else None
            modes.append((t, defect))
        yield emit(6, k, modes)


def ground_truth(task_id: str) -> dict:
    """Return gold for a task, regenerated from the seed via the in-process cache.

    Registry-free: the runner always calls generate() (for the right split/seed)
    before ground_truth(); the secret is the private seed, not a stored answer key.
    """
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(
            f"No cached gold for {task_id!r}; call generate() for the matching "
            "split/seed before ground_truth()."
        )
    return _GROUND_TRUTH_CACHE[task_id]
