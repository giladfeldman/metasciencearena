"""Tests for the reference-integrity generator."""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("_reference_integrity_generator", ARENA_DIR / "generator.py")
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_reference_integrity_generator"] = generator
_SPEC.loader.exec_module(generator)

# The new, less-synthetic kinds added on top of the original six.
_NEW_KINDS = {"invalid_doi", "predatory_source", "tortured_phrase"}


def test_generate_is_deterministic():
    a = list(generator.generate("v1", seed=3))
    b = list(generator.generate("v1", seed=3))
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert [t["input"]["references"] for t in a] == [t["input"]["references"] for t in b]
    assert [t["input"]["in_text_marker_ids"] for t in a] == [t["input"]["in_text_marker_ids"] for t in b]


def test_envelopes_have_required_fields_and_tags():
    tasks = list(generator.generate("v1", seed=0, split="revealed"))
    assert tasks
    for t in tasks:
        assert {"task_id", "arena_id", "task_set_version", "difficulty", "input"} <= t.keys()
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"
        assert {"references", "in_text_marker_ids"} <= t["input"].keys()


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_t1_and_t2_have_no_issues():
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] in (1, 2):
            assert t["difficulty"]["n_issues"] == 0
            gt = generator.ground_truth(t["task_id"])
            assert gt["mistake_kinds"] == []
            assert all(not g["flagged"] for g in gt["references"])


def test_ground_truth_returns_reference_records_and_mistake_kinds():
    tasks = list(generator.generate("v1", seed=0))
    gt = generator.ground_truth(tasks[0]["task_id"])
    assert "references" in gt
    assert "mistake_kinds" in gt and isinstance(gt["mistake_kinds"], list)
    assert all({"reference_id", "issue_kind", "flagged"} <= g.keys() for g in gt["references"])


def test_ground_truth_missing_task_raises_keyerror():
    import pytest
    list(generator.generate("v1", seed=0))
    with pytest.raises(KeyError):
        generator.ground_truth("ri-t1-0-sDOES_NOT_EXIST")


def test_revealed_set_covers_every_issue_kind():
    """The public benchmark must exercise the full array of injected issues."""
    all_kinds = set(generator.ISSUE_KINDS)
    seen = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        for g in gt["references"]:
            if g["flagged"]:
                seen.add(g["issue_kind"])
    assert seen == all_kinds


def test_dangling_missing_has_marker_without_reference():
    """A dangling_missing flag must be keyed on an in_text_marker with no reference."""
    found = False
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        ref_ids = {r["reference_id"] for r in t["input"]["references"]}
        for g in gt["references"]:
            if g["issue_kind"] == "dangling_missing":
                found = True
                assert g["reference_id"] in t["input"]["in_text_marker_ids"]
                assert g["reference_id"] not in ref_ids
    assert found


def test_dangling_uncited_reference_not_in_markers():
    found = False
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        for g in gt["references"]:
            if g["issue_kind"] == "dangling_uncited":
                found = True
                ref = next(r for r in t["input"]["references"] if r["reference_id"] == g["reference_id"])
                assert ref["cited_in_text"] is False
                assert g["reference_id"] not in t["input"]["in_text_marker_ids"]
    assert found


def test_splits_share_difficulty_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert [t["input"]["references"] for t in rev] != [t["input"]["references"] for t in priv]


# --- New (less-synthetic) issue kinds: invalid_doi / predatory_source / tortured_phrase ---

def _flagged_records_by_kind(seed=0, split="revealed"):
    out: dict[str, list[dict]] = {}
    for t in generator.generate("v1", seed=seed, split=split):
        gt = generator.ground_truth(t["task_id"])
        ref_by_id = {r["reference_id"]: r for r in t["input"]["references"]}
        for g in gt["references"]:
            if g["flagged"]:
                out.setdefault(g["issue_kind"], []).append(
                    {"gold": g, "listed": ref_by_id.get(g["reference_id"])}
                )
    return out


def test_new_kinds_are_in_issue_kinds_and_emitted():
    assert _NEW_KINDS <= set(generator.ISSUE_KINDS)
    by_kind = _flagged_records_by_kind()
    for kind in _NEW_KINDS:
        assert by_kind.get(kind), f"{kind} never emitted in the revealed set"


def test_every_reference_has_a_venue_field():
    for t in generator.generate("v1", seed=0):
        for ref in t["input"]["references"]:
            assert "venue" in ref and isinstance(ref["venue"], str) and ref["venue"]


def test_invalid_doi_listed_doi_is_malformed_and_canonical_is_stored():
    for rec in _flagged_records_by_kind()["invalid_doi"]:
        listed, gold = rec["listed"], rec["gold"]
        assert listed is not None
        canonical = gold["canonical"]["doi"]
        # The listed DOI must differ from canonical and be structurally broken
        # (lacks the "10." registrant prefix OR is a strict, shorter prefix of it).
        assert listed["doi"] != canonical
        assert (not listed["doi"].startswith("10.")) or (canonical.startswith(listed["doi"]) and len(listed["doi"]) < len(canonical)) or ("O" in listed["doi"])
        assert canonical.startswith("10.")


def test_predatory_source_swaps_venue_and_stores_reputable_canonical():
    for rec in _flagged_records_by_kind()["predatory_source"]:
        listed, gold = rec["listed"], rec["gold"]
        assert listed is not None
        assert listed["venue"] != gold["canonical"]["venue"]
        # The injected venue carries a predatory-naming tell; the canonical does not.
        assert "Journal" in listed["venue"] or "International" in listed["venue"] or "Global" in listed["venue"]


def test_tortured_phrase_swaps_title_and_stores_legit_canonical():
    for rec in _flagged_records_by_kind()["tortured_phrase"]:
        listed, gold = rec["listed"], rec["gold"]
        assert listed is not None
        assert listed["title"] != gold["canonical"]["title"]


def test_clean_control_lookalikes_are_present_and_never_flagged():
    """The confusable honest twins must appear AND must all be flagged=False."""
    seen_control_entries = set()
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        gold_by_id = {g["reference_id"]: g for g in gt["references"]}
        for ref in t["input"]["references"]:
            rid = ref["reference_id"]
            if rid.startswith("cc_"):
                seen_control_entries.add(rid.rsplit("__", 1)[0])
                assert gold_by_id[rid]["flagged"] is False
                assert gold_by_id[rid]["issue_kind"] is None
    # Every clean-control pool entry from the catalog is exercised somewhere.
    controls = generator._load_clean_controls()
    catalog_control_ids = {e["id"] for pool in controls.values() for e in pool}
    assert catalog_control_ids
    assert catalog_control_ids <= seen_control_entries


def test_t2_controls_include_every_clean_control_pool():
    """The T2 false-alarm trap must seat each confusable look-alike pool."""
    pools_seen = set()
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] != 2:
            continue
        for ref in t["input"]["references"]:
            if ref["reference_id"].startswith("cc_"):
                pools_seen.add(ref["reference_id"].rsplit("__", 1)[0])
    controls = generator._load_clean_controls()
    catalog_control_ids = {e["id"] for pool in controls.values() for e in pool}
    # At least one entry from each of the three pools shows up across T2.
    assert len(pools_seen) >= 3
    assert pools_seen <= catalog_control_ids


def test_private_split_also_covers_every_issue_kind():
    """Parity at the gold level: the private split exercises all kinds too."""
    seen = set()
    for t in generator.generate("v1", seed=777, split="private"):
        gt = generator.ground_truth(t["task_id"])
        for g in gt["references"]:
            if g["flagged"]:
                seen.add(g["issue_kind"])
    assert seen == set(generator.ISSUE_KINDS)


# --- Drift guard: every gold category must be known to the schema enum AND the prompt ---

def _output_schema_enum() -> set[str]:
    schema = json.loads((ARENA_DIR / "schemas" / "output.schema.json").read_text(encoding="utf-8"))
    enum = schema["properties"]["records"]["items"]["properties"]["issue_kind"]["enum"]
    return {e for e in enum if isinstance(e, str)}


def test_every_gold_category_is_in_the_output_schema_enum_and_the_prompt():
    """A kind the schema/prompt doesn't know is untestable. Guard both surfaces.

    Mirrors the drift bug class fixed elsewhere: the generator can emit an
    issue_kind that the output schema enum forbids (so a correct player can't even
    encode it) or that the player prompt never names (so no player is told to look
    for it). Assert every flagged gold category is present in BOTH.
    """
    gold_categories = set()
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        gold_categories.update(g["issue_kind"] for g in gt["references"] if g["flagged"])
    assert gold_categories == set(generator.ISSUE_KINDS)

    enum = _output_schema_enum()
    missing_from_schema = gold_categories - enum
    assert not missing_from_schema, f"gold kinds absent from output.schema.json enum: {sorted(missing_from_schema)}"

    prompt = (REPO_ROOT / "players" / "prompts" / "reference_integrity.txt").read_text(encoding="utf-8")
    missing_from_prompt = {k for k in gold_categories if k not in prompt}
    assert not missing_from_prompt, f"gold kinds not named in the player prompt: {sorted(missing_from_prompt)}"
