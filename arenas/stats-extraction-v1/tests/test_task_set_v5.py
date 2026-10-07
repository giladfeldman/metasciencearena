"""Task set v5 pins the defect it was minted to fix (2026-08-29).

v4 stamped `p_rel` onto every truthful NHST gold item, but three templates never
interpolated it, so THE ANSWER KEY RECORDED A RELATION THE PASSAGE NEVER PRINTED.
Two of them are worse than a wrong relation: `p {rel} {p_threshold}` renders the
CONSTANT ".05" and never the drawn p at all, so the gold graded a number the
reader cannot see.

Found by a three-provider cross-model consult round on the v4 scoring diff
(seat: Grok 4.6 / xAI via Cursor CLI) on 2026-08-28. Neither other seat found it.
Reproduced and re-measured locally 2026-08-29 with a locator that finds ALL 112
nhst gold items -- the first survey located 55 of them and therefore reported a
LOWER BOUND of 6.

MEASURED, revealed split, before any line was changed:

                                          v3     v4     v5
    nhst gold items located            108/108 112/112 104/104
    printed relation != gold relation        0      7      0
    printed number   != gold number          8     11      0
    printed claim FALSE while key says ok   44      5      0

Every test here states the measured v4 value it is moving away from, and every
one was watched fail against v4 before v5 existed. `test_..._is_a_v5_property`
is the two-sided control that keeps that honest: it asserts v3 and v4 STILL
measure their old values, so making v5 green by rewriting frozen history turns
that control red instead.

THE FIX IS NOT A RELABEL, and the rejected option is the tempting one.
Re-labelling those 5 items `inconsistent` in the gold would need no re-run, and
would leave the passage printing `p = .001` for a true p of 7.3e-06 -- which is
exactly the manufactured error v4 exists to remove. v5 changes the TEXT so the
printed claim is true and the key grades that claim.

Full write-up: docs/reports/2026-08-28-v4-answer-key-p-relation-defect.md
"""
from pathlib import Path
import importlib.util
import json
import re
import sys

import pytest

ARENA_DIR = Path(__file__).resolve().parents[1]

#: The task set under test, and the two it must not have disturbed.
CURRENT = "v5"
FROZEN = ("v1", "v2", "v3", "v4")

#: The committed answer keys are in `framework/publish.py`'s NEVER_PUBLISH set, so
#: the PUBLIC MIRROR ships this file without them. A byte-for-byte comparison
#: against a file that is absent by design is a SKIP, not a failure: the public
#: suite must be able to go green, and a red badge on the public repo is its own
#: false claim. Locally the keys are present and every one of these runs.
#: Found by `scripts/sync_public_mirror.sh --verify-tests`, which runs the
#: mirrored suite -- the local suite cannot see this and never will.
_GOLD_PRESENT = all((ARENA_DIR / "task_sets" / v / "_ground_truth.json").is_file()
                    for v in FROZEN)
needs_committed_gold = pytest.mark.skipif(
    not _GOLD_PRESENT,
    reason="committed _ground_truth.json is NEVER_PUBLISH, so a public-mirror "
           "checkout cannot run the byte-for-byte frozen comparison")


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, ARENA_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


generator = _load("_stats_extraction_v1_generator_v5", "generator.py")
scorer = _load("_stats_extraction_v1_scorer_v5", "scorer.py")


# --- locating the claim the reader actually sees -----------------------------
#
# Anchored on the gold item's own SPAN, never on text order. Text order is wrong
# here: the tier-4 dual template prints a second statistic that carries no gold
# item, so a positional zip drifts by one and manufactures defects that are not
# there. Tolerates italics (`*p*`), the HTML entities tier 3 emits, comma
# decimals, and tier 3's OCR corruption (`_ocr_corrupt` maps 0->O and 1->l).

_P_CLAIM = re.compile(r"[,;:\s]\s*\*?p\*?\s*(&lt;|&gt;|<|>|=)\s*")
_NUMBER = re.compile(r"[-+]?[0-9OlI]*[.,][0-9OlI]+|[-+]?[0-9OlI]+")
_HTML = {"&lt;": "<", "&gt;": ">"}


def _decode(raw):
    """(value, decimals printed) for a number in any rendering the templates emit."""
    s = raw.replace("O", "0").replace("l", "1").replace("I", "1").replace(",", ".")
    try:
        return float(s), (len(s.split(".")[1]) if "." in s else 0)
    except ValueError:
        return None, 0


def _printed_claim(text, item):
    """(relation, value, decimals, raw) the passage states about this item's p."""
    span = item.get("span") or {}
    if span.get("char_end", 0) <= span.get("char_start", 0):
        return None
    m = _P_CLAIM.search(text, span["char_end"])
    if m is None:
        return None
    nm = _NUMBER.search(text[m.end(): m.end() + 24])
    if nm is None:
        return None
    val, dec = _decode(nm.group(0))
    return _HTML.get(m.group(1), m.group(1)), val, dec, nm.group(0)


def _claim_is_true(rel, val, dec, true_p):
    if val is None or true_p is None:
        return None
    if rel == "<":
        return true_p < val
    if rel == ">":
        return true_p > val
    return round(true_p, dec) == round(val, dec)


def _nhst(version, split="revealed", truthful_only=True):
    """(task_id, text, item) for every nhst gold item of one version.

    Clears the gold cache first. `task_id` is NOT unique across task-set versions
    -- `_seed_tag` hashes the seed only, so v2..v5 share all 36 ids -- and
    `_GROUND_TRUTH_CACHE` is keyed on task_id alone. A cache left over from
    another version serves that version's gold and every count here becomes a
    measurement of the wrong task set.
    """
    generator._GROUND_TRUTH_CACHE.clear()
    envelopes = list(generator.generate(version, 0, split=split))
    assert envelopes, f"generator produced no {version}/{split} tasks -- the probe is broken"
    out = []
    for env in envelopes:
        text = env["input"]["text"]
        for item in generator.ground_truth(env["task_id"])["items"]:
            if item.get("kind") != "nhst_stat":
                continue
            if truthful_only and not item.get("truthful", True):
                continue  # a planted deception prints a wrong p ON PURPOSE
            out.append((env["task_id"], text, item))
    return out


def _relation_mismatches(version):
    """Items whose gold `p_rel` is not the relation the passage prints."""
    bad = []
    for tid, text, item in _nhst(version):
        claim = _printed_claim(text, item)
        if claim is None:
            continue
        rel, _val, _dec, raw = claim
        gold_rel = (item.get("fields") or {}).get("p_rel")
        if gold_rel is not None and gold_rel != rel:
            bad.append(f"{tid}: text prints 'p {raw}' with relation {rel!r}, "
                       f"gold records {gold_rel!r}")
    return bad


def _number_mismatches(version):
    """Items whose gold `p` is not the number the passage prints."""
    bad = []
    for tid, text, item in _nhst(version):
        claim = _printed_claim(text, item)
        if claim is None:
            continue
        _rel, val, _dec, raw = claim
        gold_p = (item.get("fields") or {}).get("p")
        if val is not None and gold_p is not None and abs(val - float(gold_p)) > 1e-9:
            bad.append(f"{tid}: text prints {raw!r}, gold grades p = {gold_p}")
    return bad


def _false_claims_graded_ok(version):
    """The defect that actually corrupts scores: a FALSE printed claim graded `ok`."""
    bad = []
    for tid, text, item in _nhst(version):
        claim = _printed_claim(text, item)
        if claim is None:
            continue
        rel, val, dec, raw = claim
        f = item.get("fields") or {}
        true_p = item.get("p_recomputed")
        if true_p is None:
            true_p = generator._two_sided_p(f.get("test_type"), f.get("value"),
                                            f.get("df1"), f.get("df2"))
        if _claim_is_true(rel, val, dec, true_p) is False and item.get("expected_verdict") == "ok":
            bad.append(f"{tid}: text asserts 'p {rel} {raw}' but the true p is {true_p:.6g}")
    return bad


# --- the two-sided control on the locator ------------------------------------

def test_every_nhst_item_has_a_findable_printed_claim():
    """Without this, a zero from every test below is meaningless -- a locator that
    matches nothing looks exactly like an arena with no defects. The first survey
    of this defect located 55 of 112 items and reported its 6 as a LOWER BOUND."""
    for version in (CURRENT, "v4", "v3"):
        items = _nhst(version, truthful_only=False)
        missing = [tid for tid, text, it in items if _printed_claim(text, it) is None]
        assert not missing, (
            f"{version}: no printed p-claim found for {len(missing)} of {len(items)} "
            f"nhst gold items: {missing[:5]}")


# --- the defect: the key graded a claim the passage did not print ------------

def test_no_truthful_item_prints_a_relation_the_gold_does_not_record():
    """v4 measured 7. `*p* = {p_no_lead}` printed `=` while gold stamped `<`, and
    `p < {p_threshold}` printed `<` while gold stamped `=`."""
    bad = _relation_mismatches(CURRENT)
    assert not bad, (
        f"{len(bad)} item(s) print a relation the answer key does not record:\n  "
        + "\n  ".join(bad))


def test_no_truthful_item_prints_a_number_the_gold_does_not_record():
    """v4 measured 11, all from `p {rel} {p_threshold}`: `.05` is a CONSTANT that
    `_instantiate_template` supplies, so the drawn p was never rendered and the
    gold asserted a value absent from the input. Same shape as the ghost CIs
    `_ci_is_printed` removed in v3."""
    bad = _number_mismatches(CURRENT)
    assert not bad, (
        f"{len(bad)} item(s) are graded against a p the passage never printed:\n  "
        + "\n  ".join(bad))


def test_no_item_with_a_false_printed_claim_is_graded_ok():
    """THE SCIENTIFIC INVARIANT, and the one that inverts scoring.

    An item whose printed claim is FALSE of its own recomputed p is a genuine
    reporting error. Grading it `ok` charges a `false_alarm` to the tool that
    catches it and credits a `hit` to the tool that misses it -- against exactly
    the competence this arena exists to measure. v4 measured 5, v3 measured 44.
    """
    bad = _false_claims_graded_ok(CURRENT)
    assert not bad, (
        f"{len(bad)} item(s) print a FALSE claim about p yet are graded 'ok':\n  "
        + "\n  ".join(bad))


def test_the_repair_is_a_v5_property_not_a_relabelled_v3_or_v4():
    """TWO-SIDED CONTROL, and the reason it is here.

    v3 and v4 are frozen: v3 has published scores and v4 has 180 committed run
    records. If either number moves, someone made v5 green by editing history
    rather than by changing v5's text, and every score computed on that history
    now refers to tasks that no longer exist.
    """
    assert len(_relation_mismatches("v4")) == 7, (
        f"v4 now measures {len(_relation_mismatches('v4'))} relation mismatches, not 7")
    assert len(_number_mismatches("v4")) == 11, (
        f"v4 now measures {len(_number_mismatches('v4'))} number mismatches, not 11")
    assert len(_false_claims_graded_ok("v4")) == 5, (
        f"v4 now measures {len(_false_claims_graded_ok('v4'))} false claims graded ok, not 5")
    assert len(_false_claims_graded_ok("v3")) == 44, (
        f"v3 now measures {len(_false_claims_graded_ok('v3'))} false claims graded ok, not 44")


# --- what the repair must NOT have done --------------------------------------

def test_the_threshold_notation_survives():
    """CONTROL against the cheapest wrong fix: deleting the template.

    `p < .05` is ordinary APA prose and tier 2 exists to test notation variation.
    Removing the form would make every test above pass while narrowing the arena,
    so v5 must still print a threshold claim -- in BOTH directions, since the
    relation is now measured against alpha rather than hardcoded.
    """
    rels = {"<": 0, ">": 0}
    for _tid, text, item in _nhst(CURRENT):
        claim = _printed_claim(text, item)
        if claim is None:
            continue
        rel, val, _dec, _raw = claim
        if val is not None and abs(val - generator._ALPHA_PRINTED) < 1e-9 and rel in rels:
            rels[rel] += 1
    assert rels["<"] > 0 and rels[">"] > 0, (
        f"v5 no longer prints a threshold claim in both directions: {rels}. "
        "The notation was deleted rather than repaired.")


def test_the_alpha_relation_is_true_of_the_statistic():
    """Every threshold claim v5 prints must be TRUE of the recomputed p."""
    for tid, _text, item in _nhst(CURRENT):
        f = item.get("fields") or {}
        if f.get("p") != generator._ALPHA_PRINTED:
            continue
        tp = generator._two_sided_p(f.get("test_type"), f.get("value"),
                                    f.get("df1"), f.get("df2"))
        assert tp is not None, f"{tid}: threshold claim on a statistic with no closed-form p"
        rel = f.get("p_rel")
        if rel == "<":
            assert tp < generator._ALPHA_PRINTED, f"{tid}: printed p < .05 but true p is {tp}"
        elif rel == ">":
            assert tp > generator._ALPHA_PRINTED, f"{tid}: printed p > .05 but true p is {tp}"
        else:
            assert abs(tp - generator._ALPHA_PRINTED) < 5e-3, (
                f"{tid}: printed p = .05 but true p is {tp}")


def test_a_significance_claim_at_the_boundary_reads_as_significant():
    """`p < .05` ASSERTS significance, and `.05 < .05` is False.

    A strict comparison in `_label_verification` would read the commonest
    significance claim in psychology as a null one, and every v5 threshold item
    below alpha would be stamped `inconsistent`. Unit-level so the rule is pinned
    where it lives, not only through the 36-task aggregate.
    """
    gt = {"items": [{
        "kind": "nhst_stat", "truthful": True, "deception_kind": None,
        # t(100) = 3.0 -> true p ~= .0034, comfortably significant.
        "fields": {"test_type": "t", "df1": 100, "df2": 0, "value": 3.0,
                   "p": 0.05, "p_rel": "<"},
    }]}
    out = generator._label_verification(gt, "t(100) = 3.00, p < .05.", v4=True)
    item = out["items"][0]
    assert item["expected_verdict"] == "ok", (
        "a true `p < .05` claim was graded a decision error: "
        f"{item['expected_verdict']}")
    assert item["coherent"] is True


def test_no_truthful_item_is_incoherent():
    """Inherited from v4 and must not regress: a truthful item whose reported p is
    not true of its statistic is a defect WE manufactured. v3 measured 45."""
    bad = [(it["fields"].get("p"), it["fields"].get("p_rel"), it.get("p_recomputed"))
           for _tid, _text, it in _nhst(CURRENT) if it.get("coherent") is False]
    assert not bad, f"{len(bad)} truthful NHST items are incoherent on {CURRENT}: {bad[:5]}"


def test_the_deliberate_deception_class_is_intact():
    """v5 removes an ACCIDENTAL error class. Removing the deliberate one too would
    leave an arena that cannot distinguish any player from any other. Measured
    identical on v3, v4 and v5: 24 deceptions, 8 kinds, 15 `inconsistent` items."""
    generator._GROUND_TRUTH_CACHE.clear()
    kinds, n_inconsistent = {}, 0
    for t in generator.generate(CURRENT, 0, "revealed"):
        for it in generator.ground_truth(t["task_id"])["items"]:
            if not it.get("truthful", True):
                kinds[it.get("deception_kind")] = kinds.get(it.get("deception_kind"), 0) + 1
            if it.get("expected_verdict") == "inconsistent":
                n_inconsistent += 1
    assert sum(kinds.values()) == 24, f"deception count moved: {kinds}"
    assert len(kinds) == 8, f"a deception KIND disappeared: {sorted(kinds)}"
    assert n_inconsistent == 15, f"the graded error class moved: {n_inconsistent}"


def test_the_clean_class_is_not_empty():
    """Same floor as the v3 and v4 guards. Measured on v5: 80."""
    n_ok = sum(1 for _tid, _text, it in _nhst(CURRENT)
               if it.get("expected_verdict") == "ok")
    assert n_ok >= 40, f"only {n_ok} genuinely coherent truthful NHST items"


def test_the_forgiveness_cell_has_no_cases():
    """v3 measured 45 forgiven items, which is why statcheck was graded on 39 of
    its 73 checkable items while escimate was graded on 70 of 72. The scorer is
    unchanged; the rule is inert because the arena no longer manufactures
    incoherent items."""
    generator._GROUND_TRUTH_CACHE.clear()
    forgiven = 0
    for t in generator.generate(CURRENT, 0, "revealed"):
        for it in generator.ground_truth(t["task_id"])["items"]:
            checkable, _expected, coherent = scorer._gold_verdict(it)
            if checkable and not coherent:
                forgiven += 1
    assert forgiven == 0, f"{forgiven} {CURRENT} items still reach the forgiveness cell"


# --- the append-only invariant -----------------------------------------------

@needs_committed_gold
@pytest.mark.parametrize("version", FROZEN)
def test_frozen_task_sets_reproduce_byte_for_byte(version):
    """A shared generator is APPEND-ONLY, and this is the only check that proves it.

    Merely REORDERING two `rng` calls rewrote a frozen task set in this repo
    before -- 7 of 36 v1 tasks stopped matching their committed gold, and every
    test still passed. v5 adds a template pool, three derived render fields and a
    branch in `_apply_apa_p`; none of them may consume or displace a draw.

    Compares the ENVELOPE TEXT and every GOLD FIELD, not just the text: v5 also
    rewrites `fields['p']` for the threshold items, which is a gold-only change
    that a text-only comparison would not see.

    REVEALED ONLY, and it says so rather than implying otherwise. Only the
    revealed split has a committed answer key to compare against; the private
    one is covered by `test_frozen_private_split_reproduces_its_fingerprint`
    below. Splitting the two is deliberate -- a guard that checks one split while
    asserting both is the PARTIAL-GUARD shape `framework/audit.py` documents,
    where a fingerprint baselined over half a task set reports OK forever.
    """
    committed = json.loads(
        (ARENA_DIR / "task_sets" / version / "_ground_truth.json").read_text(encoding="utf-8"))
    assert committed, f"{version}: committed gold is empty -- the probe is broken"
    generator._GROUND_TRUTH_CACHE.clear()
    regen = {t["task_id"]: (t, generator.ground_truth(t["task_id"]))
             for t in generator.generate(version, 0, "revealed")}
    assert len(regen) == len(committed), (
        f"{version}: regenerated {len(regen)} tasks, committed gold has {len(committed)}")

    def _canon(o):
        return json.dumps(o, sort_keys=True, ensure_ascii=False)

    text_drift, gold_drift = [], []
    for tid, rec in committed.items():
        env, gt = regen[tid]
        if rec["envelope"]["input"]["text"] != env["input"]["text"]:
            text_drift.append(tid)
        if _canon(rec["ground_truth"]) != _canon(gt):
            gold_drift.append(tid)
    assert not text_drift, f"{version}: {len(text_drift)} task(s) drifted in TEXT: {text_drift[:3]}"
    assert not gold_drift, f"{version}: {len(gold_drift)} task(s) drifted in GOLD: {gold_drift[:3]}"


@pytest.mark.parametrize("version", FROZEN)
def test_frozen_private_split_reproduces_its_fingerprint(version):
    """The half the committed answer key cannot cover.

    Raised by the cross-model consult on 2026-08-29 (seat: Grok 4.6 / xAI): the
    test above asserts an append-only invariant about the GENERATOR while
    exercising one of the two splits. The private split runs the identical gated
    code path on a different seed, so a v5 branch that shifted the stream would
    show there too -- but nothing in this file would have noticed.

    The private answer key is deliberately never committed, so this compares the
    generator against `arena.yaml`'s recorded `generator_fingerprint`, which is
    computed over BOTH splits (`framework.audit.generator_fingerprint`). A
    mismatch means one split moved; which one is then a debugging question.

    SKIPS LOUDLY without the gitignored `.private_seed`. Every fresh clone is in
    that state, and a machine without the secret computes a different,
    revealed-only hash -- reporting DRIFT THAT IS NOT DRIFT. Skipping is honest;
    passing would not be.
    """
    import yaml as _yaml

    if not (ARENA_DIR / "task_sets" / version / ".private_seed").exists():
        pytest.skip(f"{version}: no .private_seed on this machine — the private "
                    "split is UNCHECKED here, not verified")
    from framework.audit import generator_fingerprint

    manifest = _yaml.safe_load((ARENA_DIR / "arena.yaml").read_text(encoding="utf-8"))
    entry = next(v for v in manifest["task_set_versions"] if v["version"] == version)
    expected = entry.get("generator_fingerprint")
    assert expected, f"{version} has no recorded generator_fingerprint — it is UNGUARDED"

    notes = []
    actual, n = generator_fingerprint(ARENA_DIR, version, errors=notes)
    assert actual is not None, f"{version}: fingerprint could not be computed: {notes}"
    assert n == 72, (
        f"{version}: fingerprint covers {n} tasks, not the 72 both splits produce. "
        "A fingerprint that matched over half a task set is a rubber stamp — see "
        "framework/audit.py's PARTIAL-GUARD verdict.")
    assert actual == expected, (
        f"{version}: generator no longer reproduces its recorded fingerprint over "
        f"both splits.\n  recorded: {expected}\n  actual:   {actual}")


@needs_committed_gold
def test_the_frozen_check_can_actually_detect_drift():
    """CONTROL for the test above. Comparing a version against ANOTHER version's
    committed gold must fail loudly -- otherwise a byte comparison that silently
    matches nothing would report every frozen set clean."""
    committed = json.loads(
        (ARENA_DIR / "task_sets" / "v3" / "_ground_truth.json").read_text(encoding="utf-8"))
    generator._GROUND_TRUTH_CACHE.clear()
    drift = [t["task_id"] for t in generator.generate("v4", 0, "revealed")
             if committed[t["task_id"]]["envelope"]["input"]["text"] != t["input"]["text"]]
    assert len(drift) == 36, (
        "v3 and v4 task text should differ on all 36 tasks; the comparison found "
        f"{len(drift)}, so it is not comparing what it claims to")
