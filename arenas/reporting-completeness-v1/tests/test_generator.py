"""Tests for the reporting-completeness generator."""
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_reporting_completeness_generator", ARENA_DIR / "generator.py"
)
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_reporting_completeness_generator"] = generator
_SPEC.loader.exec_module(generator)

# The three defect kinds added in the 2026-07 broadening, each with a matched
# clean-control look-alike (must NOT produce a gold flag).
_NEW_KINDS = {"unspecified_test", "no_correction", "percent_count_mismatch"}


def test_generate_is_deterministic():
    a = list(generator.generate("v1", seed=3))
    b = list(generator.generate("v1", seed=3))
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert [t["input"]["text"] for t in a] == [t["input"]["text"] for t in b]


def test_envelopes_have_required_fields_and_tags():
    tasks = list(generator.generate("v1", seed=0, split="revealed"))
    assert tasks
    for t in tasks:
        assert {"task_id", "arena_id", "task_set_version", "difficulty", "input"} <= t.keys()
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"
        assert "text" in t["input"]
        assert isinstance(t["input"]["text"], str) and t["input"]["text"]


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_t1_and_t2_have_no_defects():
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] in (1, 2):
            assert t["difficulty"]["n_defects"] == 0
            gt = generator.ground_truth(t["task_id"])
            assert gt["flags"] == []
            assert gt["mistake_kinds"] == []


def test_ground_truth_returns_flag_records():
    tasks = list(generator.generate("v1", seed=0))
    # Pick a task with at least one defect (a T3 task).
    t3 = next(t for t in tasks if t["difficulty"]["tier"] == 3)
    gt = generator.ground_truth(t3["task_id"])
    assert "flags" in gt and "mistake_kinds" in gt
    assert gt["flags"]
    for f in gt["flags"]:
        assert {"span", "category"} <= f.keys()
        assert {"text", "char_start", "char_end"} <= f["span"].keys()


def test_gold_spans_match_input_text():
    """Every gold span's char offsets must extract exactly its recorded text."""
    for t in generator.generate("v1", seed=0):
        text = t["input"]["text"]
        gt = generator.ground_truth(t["task_id"])
        for f in gt["flags"]:
            span = f["span"]
            assert text[span["char_start"]:span["char_end"]] == span["text"]


def test_ground_truth_missing_raises_keyerror():
    try:
        generator.ground_truth("rc-does-not-exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError for an unknown task_id")


def test_revealed_set_covers_every_defect_kind():
    """The public benchmark must exercise the full array of injected defects."""
    all_kinds = set(generator.DEFECT_KINDS)
    seen = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        for k in gt["mistake_kinds"]:
            seen.add(k)
    assert seen == all_kinds


def test_splits_share_difficulty_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert [t["input"]["text"] for t in rev] != [t["input"]["text"] for t in priv]


# --------------------------------------------------------------------------- #
# Broadened taxonomy (2026-07): new kinds + matched clean controls + parity.   #
# --------------------------------------------------------------------------- #

def _kind_counts(tasks):
    c = Counter()
    for t in tasks:
        gt = generator.ground_truth(t["task_id"])
        for k in gt["mistake_kinds"]:
            c[k] += 1
    return c


def test_new_defect_kinds_are_registered():
    """All three broadened kinds are in DEFECT_KINDS (so T3/T4 cycle them)."""
    assert _NEW_KINDS <= set(generator.DEFECT_KINDS)
    # 9 kinds total after the broadening.
    assert len(generator.DEFECT_KINDS) == 9
    assert len(generator.DEFECT_KINDS) == len(set(generator.DEFECT_KINDS))


def test_each_new_kind_is_actually_emitted_in_gold():
    """Each new kind must appear as a real gold flag (with a non-empty span)."""
    counts = _kind_counts(generator.generate("v1", seed=0, split="revealed"))
    for kind in _NEW_KINDS:
        assert counts.get(kind, 0) >= 1, f"{kind} never emitted"
    # And the emitted flags carry a span that extracts exactly its text.
    for t in generator.generate("v1", seed=0, split="revealed"):
        text = t["input"]["text"]
        gt = generator.ground_truth(t["task_id"])
        for f in gt["flags"]:
            if f["category"] in _NEW_KINDS:
                span = f["span"]
                assert span["text"].strip()
                assert text[span["char_start"]:span["char_end"]] == span["text"]


def test_clean_control_lookalikes_produce_no_flag():
    """Tasks that embed a clean-control look-alike must not flag it.

    The T2 trap and T4 subtle tiers render the consistent-% clause and the
    corrected many-comparisons report. No gold flag may carry a clean-control mode
    label, and the T2 trap must stay flag-free even though it now contains those
    confusable-but-honest sentences.
    """
    clean_modes = set(generator.CLEAN_CONTROL_MODES)
    assert clean_modes  # the controls exist
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        # A clean-control mode is never a flagged category.
        assert not (clean_modes & set(gt["mistake_kinds"]))
        if t["difficulty"]["tier"] == 2:
            assert gt["flags"] == [], "T2 false-alarm trap must carry zero flags"
            assert gt["mistake_kinds"] == []


def test_t2_trap_actually_contains_the_clean_lookalikes():
    """The hardened T2 trap must contain at least one consistent-% clause AND at
    least one corrected many-comparisons report across its tasks (else the
    false-alarm discrimination for the new kinds is untested)."""
    t2_text = " ".join(
        t["input"]["text"]
        for t in generator.generate("v1", seed=0, split="revealed")
        if t["difficulty"]["tier"] == 2
    )
    # A consistent percent/count clause renders "(NN.N%)" with a matching count.
    assert re.search(r"\(\d{1,3}\.\d%\)", t2_text), "no percent/count clause in T2 trap"
    # A corrected report names a multiplicity correction.
    assert re.search(r"Bonferroni|Holm|Tukey|FDR|correction", t2_text), (
        "no corrected many-comparisons report in T2 trap"
    )


def test_unspecified_test_has_no_test_statistic_but_keeps_a_p_value():
    """The unspecified_test defect omits the statistic yet still reports p/ES/CI."""
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        text = t["input"]["text"]
        for f in gt["flags"]:
            if f["category"] != "unspecified_test":
                continue
            span = f["span"]
            # The flagged span is the bare significance claim with no statistic.
            assert "= " not in span["text"]  # no "t = ", "F = ", etc. in the claim
            # The same report still carries a p-value (precise reporting elsewhere):
            # look in the window immediately following the claim.
            window = text[span["char_end"]: span["char_end"] + 40]
            assert re.search(r"p = \.\d", window), window


def test_percent_count_mismatch_is_genuinely_inconsistent():
    """Cross-reference every "<count> of <N> (<pct>%)" clause against gold.

    A clause covered by a percent_count_mismatch gold flag MUST be arithmetically
    inconsistent; any other count/percent clause (a clean_count_match look-alike)
    MUST be internally consistent. Both must occur somewhere in the suite. This
    handles tasks that carry BOTH the defect and its clean look-alike (the T4
    subtle tier), regardless of which appears first.
    """
    # Two renderings: "Of the <N> ..., <count> (<pct>%)" and "<count> of <N> (<pct>%)".
    clause_re = re.compile(
        r"(?:of the (?P<n1>\d+)[^,()]*,\s*(?P<c1>\d+)|(?P<c2>\d+)\s+of\s+(?P<n2>\d+))"
        r"\s*\((?P<pct>\d{1,3}\.\d)%\)",
        re.I,
    )

    def parse(m):
        if m.group("n1") is not None:
            n, count = int(m.group("n1")), int(m.group("c1"))
        else:
            count, n = int(m.group("c2")), int(m.group("n2"))
        return count, n, float(m.group("pct"))

    saw_bad = saw_good = 0
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        text = t["input"]["text"]
        bad_spans = [
            (f["span"]["char_start"], f["span"]["char_end"])
            for f in gt["flags"]
            if f["category"] == "percent_count_mismatch"
        ]
        for m in clause_re.finditer(text):
            count, n, pct = parse(m)
            true_pct = round(100.0 * count / n, 1)
            # Is this clause inside a percent_count_mismatch gold span?
            flagged = any(s <= m.start() and m.end() <= e for s, e in bad_spans)
            if flagged:
                assert abs(true_pct - pct) > 0.05, (
                    f"flagged mismatch is actually consistent: {count}/{n}={true_pct} vs {pct}"
                )
                saw_bad += 1
            else:
                assert abs(true_pct - pct) <= 0.05, (
                    f"clean count clause is inconsistent: {count}/{n}={true_pct} vs {pct} "
                    f"in {t['task_id']!r}"
                )
                saw_good += 1
    assert saw_bad >= 1, "no flagged percent_count_mismatch clause found"
    assert saw_good >= 1, "no clean count-clause control found"


def test_per_kind_counts_match_across_splits_at_zero_tolerance():
    """Parity is by construction: index-driven assignment => IDENTICAL per-kind and
    per-tier counts in revealed and private (tolerance 0)."""
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=98765, split="private"))
    assert _kind_counts(rev) == _kind_counts(priv)
    assert (Counter(t["difficulty"]["tier"] for t in rev)
            == Counter(t["difficulty"]["tier"] for t in priv))


def test_every_gold_category_is_in_schema_enum_and_named_in_prompt():
    """Drift guard: a kind the output schema or the player prompt doesn't know is
    untestable. Every gold category must be in the schema enum AND in the prompt."""
    schema = json.loads((ARENA_DIR / "schemas" / "output.schema.json").read_text(encoding="utf-8"))
    enum = set(schema["properties"]["flags"]["items"]["properties"]["category"]["enum"])
    prompt = (ARENA_DIR.parents[1] / "players" / "prompts" / "reporting_completeness.txt").read_text(
        encoding="utf-8"
    )
    gold_cats = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gold_cats.update(generator.ground_truth(t["task_id"])["mistake_kinds"])
    assert gold_cats, "no gold categories produced"
    missing_from_enum = gold_cats - enum
    assert not missing_from_enum, f"gold categories missing from schema enum: {missing_from_enum}"
    missing_from_prompt = {c for c in gold_cats if c not in prompt}
    assert not missing_from_prompt, f"gold categories not named in player prompt: {missing_from_prompt}"
    # DEFECT_KINDS itself must be fully covered by both, too.
    assert set(generator.DEFECT_KINDS) <= enum
    assert all(k in prompt for k in generator.DEFECT_KINDS)
