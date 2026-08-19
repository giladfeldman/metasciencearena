"""Tests for the stats-extraction generator."""
from pathlib import Path
import importlib.util
import sys

import pytest

ARENA_DIR = Path(__file__).resolve().parents[1]

def _load(mod_name, filename):
    """Load an arena module under a UNIQUE name.

    A bare `import scorer` after `sys.path.insert(ARENA_DIR)` registers
    `sys.modules["scorer"]`, and all 19 arenas ship a `scorer.py`. The first
    arena imported wins, so every later arena's tests silently exercised the
    WRONG arena's code. 119 tests were also being dropped as duplicate modules
    before `consider_namespace_packages` was enabled (2026-08-07).
    """
    spec = importlib.util.spec_from_file_location(mod_name, ARENA_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod



generator = _load("_stats_extraction_v1_generator", "generator.py")


def test_generate_is_deterministic():
    a = list(generator.generate("v1", seed=42))
    b = list(generator.generate("v1", seed=42))
    assert len(a) == len(b)
    assert all(x["task_id"] == y["task_id"] for x, y in zip(a, b))
    assert all(x["input"]["text"] == y["input"]["text"] for x, y in zip(a, b))


def test_generate_emits_envelopes_with_required_fields():
    tasks = list(generator.generate("v1", seed=0))
    assert len(tasks) > 0
    for t in tasks:
        assert {"task_id", "arena_id", "task_set_version", "difficulty", "input"} <= t.keys()
        assert "text" in t["input"]
        assert "tier" in t["input"]


def test_ground_truth_returns_extractions_for_known_task():
    tasks = list(generator.generate("v1", seed=0))
    # ground_truth() serves from the in-process cache that generate() just filled.
    gt = generator.ground_truth(tasks[0]["task_id"])
    assert "items" in gt
    assert isinstance(gt["items"], list)


def test_ground_truth_is_served_from_seed_cache():
    # Gold is regenerated from seed (registry-free): generate() fills the cache,
    # ground_truth() reads it, and an uncached id raises rather than silently
    # serving stale/empty gold (LESSONS.md 2026-06-06).
    tasks = list(generator.generate("v1", seed=0))
    tid = tasks[0]["task_id"]
    assert generator.ground_truth(tid) is generator._GROUND_TRUTH_CACHE[tid]
    with pytest.raises(KeyError):
        generator.ground_truth("t-never-generated")


def test_gold_spans_are_non_degenerate_and_offset_accurate():
    # Regression for the gold-span bug: the anchor was recomputed with a fixed
    # 2-decimal format that did not match how templates actually rendered the
    # value (raw 48.7, comma 48,70, leading-zero-dropped .72, OCR 2l.85), so 26%
    # of gold spans collapsed to a degenerate [0,0] that no extraction can match.
    for split, seed in (("revealed", 0), ("private", 999)):
        for env in generator.generate("v1", seed, split=split):
            text = env["input"]["text"]
            for item in generator._GROUND_TRUTH_CACHE[env["task_id"]]["items"]:
                sp = item["span"]
                assert sp["char_end"] > sp["char_start"], (
                    f"degenerate span in {env['task_id']} for value "
                    f"{item['fields']['value']!r}")
                assert text[sp["char_start"]:sp["char_end"]] == sp["text"]


def test_statistic_impostor_value_matches_rendered_effect_size():
    # statistic_impostor renders "Cohen's d = {es}"; the gold value must be that
    # es (so its span anchors), not the unrelated nhst value it used to store.
    found = 0
    for env in generator.generate("v1", seed=0):
        for item in generator._GROUND_TRUTH_CACHE[env["task_id"]]["items"]:
            if item.get("deception_kind") == "statistic_impostor":
                found += 1
                sp = item["span"]
                rendered = env["input"]["text"][sp["char_start"]:sp["char_end"]]
                assert rendered in generator._value_renderings(item["fields"]["value"])
    assert found > 0, "expected at least one statistic_impostor item in revealed split"


def test_tier_distribution_covers_all_tiers_when_seeded_widely():
    # With enough seeds the generator should emit tasks from at least tiers 1..3 (others come in Task 12)
    tiers_seen = set()
    for s in range(10):
        for t in generator.generate("v1", seed=s):
            tiers_seen.add(t["input"]["tier"])
    assert {1, 2, 3}.issubset(tiers_seen)


def test_split_defaults_to_revealed_and_tags_visibility():
    tasks = list(generator.generate("v1", seed=0))
    assert tasks, "generator emitted nothing"
    for t in tasks:
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=7, split="private"))
    assert tasks
    for t in tasks:
        assert t["split"] == "private"
        assert t["visibility"] == "held_out"


def test_revealed_and_private_share_the_same_difficulty_matrix():
    """Parity by construction: both splits hit the identical tier x density cells."""
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=999, split="private"))

    def cells(tasks):
        from collections import Counter
        return Counter((t["difficulty"]["tier"], t["difficulty"]["density"]) for t in tasks)

    assert cells(rev) == cells(priv)


def test_different_seeds_produce_different_content_across_splits():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=999, split="private"))
    rev_text = [t["input"]["text"] for t in rev]
    priv_text = [t["input"]["text"] for t in priv]
    # Same number of tasks, but the actual generated content differs (distinct seeds).
    assert len(rev_text) == len(priv_text)
    assert rev_text != priv_text


# --- 2026-07-01: statcheck-style deception kinds (nhst_inconsistent, wrong_df) ---

import re  # noqa: E402

_FN_RE = re.compile(r"F\((\d+),\s*(\d+)\)")
_NPART_RE = re.compile(r"(\d+)\s+participants")
_KGROUP_RE = re.compile(r"(\d+)\s+(?:groups|conditions)")


def _items_of(env):
    return generator._GROUND_TRUTH_CACHE[env["task_id"]]["items"]


def _iter_items(seed=0, split="revealed", kind=None):
    for env in generator.generate("v1", seed, split=split):
        for it in _items_of(env):
            if kind is None or it.get("deception_kind") == kind:
                yield env, it


def test_new_deception_kinds_are_emitted_in_both_splits():
    for split, seed in (("revealed", 0), ("private", 999)):
        kinds = {it.get("deception_kind") for _, it in _iter_items(seed, split)}
        assert "nhst_inconsistent" in kinds, f"nhst_inconsistent missing from {split}"
        assert "wrong_df" in kinds, f"wrong_df missing from {split}"


def test_nhst_inconsistent_is_a_real_decision_error_with_coherent_label():
    # The statistic must be DECISIVELY significant (true two-sided p < .01) while
    # the reported p is non-significant — a statcheck decision error. The label/df
    # must be coherent (only the p is wrong), so the rendered text shows the
    # statistic with a p clearly >= .05 and the true p is small.
    seen = 0
    for env, it in _iter_items(kind="nhst_inconsistent"):
        f = it["fields"]
        true_p = generator._two_sided_p(f["test_type"], f["value"], f["df1"], f["df2"])
        if true_p is not None:  # scipy present (declared dep) -> assert the flip
            assert true_p < 0.01, f"nhst_inconsistent stat not decisive: true_p={true_p}"
        # the reported (non-significant) p is rendered into the text
        assert ", p = 0." in env["input"]["text"]
        assert it["truthful"] is False
        seen += 1
    assert seen > 0, "no nhst_inconsistent items generated"


def test_wrong_df_denominator_is_inconsistent_with_n_and_groups():
    # F is correctly labelled (df1 = k - 1) but the RENDERED denominator df is
    # inconsistent with the stated N and group count (should be N - k); the GOLD
    # stores the CORRECT df2 (= N - k).
    seen = 0
    for env, it in _iter_items(kind="wrong_df"):
        text = env["input"]["text"]
        n_part = _NPART_RE.search(text)
        k_grp = _KGROUP_RE.search(text)
        f_clause = _FN_RE.search(text)
        assert n_part and k_grp and f_clause, f"wrong_df text malformed: {text!r}"
        n_total, k = int(n_part.group(1)), int(k_grp.group(1))
        rendered_df1, rendered_df2 = int(f_clause.group(1)), int(f_clause.group(2))
        assert rendered_df1 == k - 1, "wrong_df numerator df should stay correct (k-1)"
        assert rendered_df2 != n_total - k, "wrong_df denominator df should be inconsistent"
        # gold records the CORRECT denominator df for downstream consistency checks
        assert it["fields"]["df2"] == n_total - k
        assert it["truthful"] is False
        seen += 1
    assert seen > 0, "no wrong_df items generated"


def test_clean_anova_control_is_consistent_and_truthful():
    # The matched clean control for wrong_df: a truthful one-way ANOVA whose df IS
    # consistent with N and groups (df1 = k-1, df2 = N-k). A good player must not
    # flag it. Lives in tier 4 (one per task).
    seen = 0
    for env in generator.generate("v1", 0, split="revealed"):
        if env["input"]["tier"] != 4:
            continue
        text = env["input"]["text"]
        for m in re.finditer(
            r"(\d+)\s+participants across\s+(\d+)\s+groups, F\((\d+),\s*(\d+)\)", text
        ):
            n_total, k, df1, df2 = (int(m.group(i)) for i in range(1, 5))
            assert df1 == k - 1 and df2 == n_total - k, (
                f"clean_anova control inconsistent in {env['task_id']}: {m.group(0)!r}"
            )
            seen += 1
    assert seen > 0, "no clean_anova control rendered in tier 4"


# CLOSED 2026-08-08. This carried `@pytest.mark.xfail(strict=True)` for the
# defect where `_build_nhst_fields` drew p independently of the statistic, so a
# "clean control" could render "the test was significant" beside p = 0.057 — which
# statcheck/escimate correctly flag, scoring a spurious `deception_false_alarm`
# against every honest player.
#
# The marker did exactly what it was put there to do: the generator was fixed, the
# test started passing, and `strict=True` turned that into a suite failure instead
# of letting a stale marker sit forever. The fix is `_build_nhst_consistent_fields`
# (generator.py) — it draws a decisively significant statistic via
# `_decisive_sig_stat` and reports the RECOMPUTED p via `_consistent_p`, so prose,
# statistic and p are built together. `_build_nhst_fields` still draws p
# independently by design; it renders the non-significance-claiming templates this
# test's regex deliberately does not match.
#
# Verified by measurement before removing the marker, not by reading the diff:
# swept seeds 0-59 plus 999 across both splits — 732 significant-prose controls
# rendered, 0 paired with p >= .05.
#
# STILL OPEN, and deliberately not closed here: fixing the generator CHANGED TASK
# TEXT, so the stored records were invalidated and never re-run. `framework audit
# --fresh` reports 100 stale records in this arena (claude-haiku-4-5-stats,
# claude-opus-4-8-stats, claude-sonnet-5-stats, escimate, statcheck — 20 each),
# which means the published leaderboard for stats-extraction-v1 is currently
# ranking on scores computed against text that no longer exists. That re-run is
# the other half of TODO item E and needs a deliberate tournament — see TODO.md.
#
# It was invisible until 2026-08-07: this file was shadowed in the full suite by
# another arena's identically-named test module, so 22 of its 24 tests silently
# did not run.
def test_nhst_consistent_control_significant_prose_matches_small_p():
    # The matched clean control for nhst_inconsistent: a coherent, SIGNIFICANT
    # result reported with its recomputed (small) p. The "significant"/"reliable"
    # prose must never be paired with a non-significant p — otherwise the control
    # itself would look inconsistent and unfairly invite a flag.
    pat = re.compile(
        r"(?:statistically reliable|As predicted, the test was significant), "
        r"[A-Za-z0-9]+(?:\([0-9, ]+\))? = -?[0-9]+\.[0-9]+, p = ([01]\.[0-9]+)"
    )
    seen = 0
    for split, seed in (("revealed", 0), ("private", 999)):
        for env in generator.generate("v1", seed, split=split):
            if env["input"]["tier"] != 1:
                continue
            for m in pat.finditer(env["input"]["text"]):
                assert float(m.group(1)) < 0.05, (
                    f"nhst_consistent control claims significance but p>=.05: {m.group(0)!r}"
                )
                seen += 1
    assert seen > 0, "no nhst_consistent control rendered in tier 1"


def test_every_gold_span_slices_its_exact_text_across_all_kinds():
    # Span-integrity invariant (the arena's known gold trap): for EVERY gold item
    # in BOTH splits — including the new nhst_inconsistent / wrong_df deceptions
    # and the nhst_consistent / clean_anova controls — the [char_start:char_end]
    # slice of the rendered text must equal the recorded span text, and be
    # non-degenerate. A break here is the ~26%-FP/FN bug the 2026-06-08 fix closed.
    checked = 0
    for split, seed in (("revealed", 0), ("private", 999)):
        for env in generator.generate("v1", seed, split=split):
            text = env["input"]["text"]
            for it in _items_of(env):
                sp = it["span"]
                assert sp["char_end"] > sp["char_start"], (
                    f"degenerate span in {env['task_id']} ({it.get('deception_kind')}) "
                    f"for value {it['fields']['value']!r}"
                )
                assert text[sp["char_start"]:sp["char_end"]] == sp["text"], (
                    f"span slice mismatch in {env['task_id']} ({it.get('deception_kind')})"
                )
                checked += 1
    assert checked > 200, f"span check covered too few items ({checked})"


def test_new_kinds_have_identical_per_kind_counts_across_splits():
    # Parity at count_tolerance 0: the deterministic index-cycled kind assignment
    # must give the new kinds the SAME count in revealed and private (this is what
    # tools/check_parity.py enforces; assert it here too for a fast local signal).
    from collections import Counter

    def counts(seed, split):
        c = Counter()
        for _, it in _iter_items(seed, split):
            if it.get("deception_kind"):
                c[it["deception_kind"]] += 1
        return c

    rev, priv = counts(0, "revealed"), counts(999, "private")
    for kind in ("nhst_inconsistent", "wrong_df"):
        assert rev[kind] == priv[kind] > 0, (
            f"{kind} parity broken: revealed={rev[kind]} private={priv[kind]}"
        )


# --- 2026-07-01 (cycle 5): effect_size_rounding + its clean_es control ---

# "Cohen's d = 0.5, 95% CI [0.62, 0.94]" — capture the stated estimate and CI so a
# test can recompute the midpoint and compare it with the estimate (the real check).
_ES_CI_RE = re.compile(
    r"= (-?[0-9]+(?:\.[0-9]+)?)(?:\s*\(|,)\s*95% CI \[(-?[0-9.]+),\s*(-?[0-9.]+)\]"
)


def test_effect_size_rounding_is_emitted_in_both_splits():
    # The new kind must appear in the deterministic index-cycle for BOTH splits, so
    # the public and private suites probe this ESCIcheck-style failure equally.
    for split, seed in (("revealed", 0), ("private", 999)):
        kinds = {it.get("deception_kind") for _, it in _iter_items(seed, split)}
        assert "effect_size_rounding" in kinds, f"effect_size_rounding missing from {split}"


def test_effect_size_rounding_estimate_disagrees_with_ci_midpoint():
    # The whole point of the kind: the reported estimate must be VISIBLY inconsistent
    # with its own CI (the CI midpoint disagrees by >= 0.05). Parse the rendered text
    # to confirm the number a reader sees is the inconsistent one, and the gold marks
    # it non-truthful. Only the estimate is corrupted; the CI is the true interval.
    seen = 0
    for env, it in _iter_items(kind="effect_size_rounding"):
        text = env["input"]["text"]
        sp = it["span"]
        # the anchored value is the (re-rounded) estimate actually written into text
        rendered = text[sp["char_start"]:sp["char_end"]]
        assert rendered in generator._value_renderings(it["fields"]["value"])
        m = _ES_CI_RE.search(text)
        assert m, f"effect_size_rounding text malformed: {text!r}"
        stated, ci_lo, ci_hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
        midpoint = (ci_lo + ci_hi) / 2
        assert abs(stated - midpoint) >= 0.05, (
            f"effect_size_rounding not visibly inconsistent: stated={stated} "
            f"midpoint={midpoint} in {env['task_id']}"
        )
        # gold's value is the rendered (inconsistent) estimate, and its CI is the true one
        assert it["fields"]["value"] == stated
        assert it["fields"]["ci_low"] == ci_lo and it["fields"]["ci_high"] == ci_hi
        assert it["truthful"] is False
        seen += 1
    assert seen > 0, "no effect_size_rounding items generated"


def test_clean_es_control_estimate_matches_ci_midpoint():
    # The matched clean control for effect_size_rounding: a correctly-rounded effect
    # size whose estimate EQUALS its CI midpoint (up to 2-dp rounding). A good player
    # recomputing the midpoint finds no discrepancy, so it must NOT be flagged. Lives
    # in tier 1 (one per task), the same clean home as nhst_consistent. The control is
    # identified by its distinctive template wording (same approach as
    # test_clean_anova_control_is_consistent_and_truthful) — plain tier-1 effect_size
    # density items also carry deception_kind=None but are NOT midpoint-centred.
    clean_es_pat = re.compile(
        r"(?:The estimate was well determined|Consistent with expectations), "
        r".+? = (-?[0-9]+(?:\.[0-9]+)?)(?:\s*\(|,)\s*95% CI "
        r"\[(-?[0-9.]+),\s*(-?[0-9.]+)\]"
    )
    seen = 0
    for split, seed in (("revealed", 0), ("private", 999)):
        for env in generator.generate("v1", seed, split=split):
            if env["input"]["tier"] != 1:
                continue
            for m in clean_es_pat.finditer(env["input"]["text"]):
                stated, ci_lo, ci_hi = (float(m.group(i)) for i in range(1, 4))
                midpoint = (ci_lo + ci_hi) / 2
                assert abs(stated - midpoint) <= 0.01, (
                    f"clean_es control estimate != CI midpoint in {env['task_id']}: "
                    f"stated={stated} midpoint={midpoint} ({m.group(0)!r})"
                )
                seen += 1
    assert seen > 0, "no clean_es control rendered in tier 1"

    # And the gold item for the control is truthful (unflaggable). The control is the
    # single tier-1 effect_size item whose stored estimate equals its stored midpoint.
    truthful_centred = 0
    for env in generator.generate("v1", 0, split="revealed"):
        if env["input"]["tier"] != 1:
            continue
        for it in _items_of(env):
            if it.get("deception_kind") is not None or it["kind"] != "effect_size":
                continue
            f = it["fields"]
            if abs(f["value"] - round((f["ci_low"] + f["ci_high"]) / 2, 2)) <= 0.01:
                assert it["truthful"] is True
                truthful_centred += 1
    assert truthful_centred > 0, "no midpoint-centred clean_es gold item in tier 1"


def test_effect_size_rounding_parity_counts_match():
    # Parity at count_tolerance 0 for the new kind specifically (check_parity.py
    # enforces this globally; assert here for a fast local signal).
    from collections import Counter

    def counts(seed, split):
        c = Counter()
        for _, it in _iter_items(seed, split):
            if it.get("deception_kind"):
                c[it["deception_kind"]] += 1
        return c

    rev, priv = counts(0, "revealed"), counts(999, "private")
    assert rev["effect_size_rounding"] == priv["effect_size_rounding"] > 0, (
        f"effect_size_rounding parity broken: "
        f"revealed={rev['effect_size_rounding']} private={priv['effect_size_rounding']}"
    )


# --- taxonomy drift-guard (cycle-4 lesson: the 4-place sync) --------------------
#
# This arena's mistake taxonomy lives in three places that must stay in sync: the
# generator (which kinds are actually injected, via deception_kinds.yaml + the
# tier-5 dispatcher), the tier-5 template catalog (a template block per kind), and
# the PLAYER PROMPT (which must teach every mode so a player knows to flag it). The
# output schema has NO deception enum here — deception is signalled by the
# `flagged_suspicious` boolean — so the durable drift-guard is prompt coverage, not
# an enum check (per project lessons.md 2026-07-01: "if the output schema has no
# enum, assert the prompt documents each mode instead"). Broadening a kind without
# updating the prompt is the exact silent half-fix that deflated cycle-1 scores.

_DECEPTION_YAML = ARENA_DIR / "catalogs" / "deception_kinds.yaml"
_TIER5_TEMPLATES = ARENA_DIR / "templates" / "tier5_adversarial.yaml"
_STATS_PROMPT = ARENA_DIR.parents[1] / "players" / "prompts" / "stats_extraction.txt"


def _declared_deception_ids():
    import yaml
    return [d["id"] for d in yaml.safe_load(_DECEPTION_YAML.read_text(encoding="utf-8"))]


def test_every_declared_deception_kind_has_a_tier5_template():
    # Each injected kind needs a template block, or the tier-5 dispatcher KeyErrors
    # when the index-cycle reaches it.
    import yaml
    templates = yaml.safe_load(_TIER5_TEMPLATES.read_text(encoding="utf-8"))
    for kind in _declared_deception_ids():
        assert kind in templates and templates[kind], (
            f"deception kind {kind!r} has no tier5_adversarial.yaml template block"
        )


def test_stats_prompt_documents_every_deception_kind():
    # The player prompt (stats_extraction.txt) must NAME every deception mode the
    # generator injects, so a player is told what to flag. If a future cycle adds a
    # kind to deception_kinds.yaml + the generator but forgets the prompt, this fails
    # — closing the 4-place-sync gap that silently deflated cycle-1 scores.
    prompt = _STATS_PROMPT.read_text(encoding="utf-8").lower()
    missing = [k for k in _declared_deception_ids() if k.replace("_", " ") not in prompt and k not in prompt]
    assert not missing, (
        f"stats_extraction.txt does not document these injected deception kinds: {missing}. "
        f"Add each to the prompt's deception-modes list (name it so the player flags it)."
    )


def test_generate_refuses_to_run_without_scipy(monkeypatch):
    """A scipy-less box must FAIL, not quietly emit incoherent statistics.

    Found 2026-08-10 by /ship's QA gate. numpy/scipy were corrupted in the local
    venv (`numpy/_core/_dtype.py` missing), so `from scipy import stats` raised
    and the module-level guard set `_HAVE_SCIPY = False`. `_decisive_sig_stat`
    then silently fell back to `rng.uniform(*spec["range"])` — a draw with no
    significance property at all — and the nhst_consistent CONTROL, whose entire
    job is to be coherent, rendered:

        "As predicted, the test was significant, chi2(158) = 3.09, p = 0.057"

    against a gold that says p = 0.002. Two tests went red and looked exactly
    like a code regression in frozen v1 history; the code was fine.

    That is the cheap half of the cost. The expensive half: `pip check` reported
    "No broken requirements found" (it never imports anything), so nothing in the
    pipeline could tell a working scipy from a broken one — and regenerating gold
    on that box would have published incoherent task text as GROUND TRUTH, with
    every test green, because gold and text would have drifted together.

    Fallbacks are fine; unlabelled ones are not. This one produces scientifically
    wrong output, so it is refused rather than labelled.
    """
    monkeypatch.setattr(generator, "_HAVE_SCIPY", False)
    with pytest.raises(RuntimeError, match="scipy"):
        list(generator.generate("v2", 0, "revealed"))
