"""Tests for the open-practices-repro generator."""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("_open_practices_repro_generator", ARENA_DIR / "generator.py")
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_open_practices_repro_generator"] = generator
_SPEC.loader.exec_module(generator)

STATEMENT_KINDS = ["dead_data_link", "available_upon_request", "materials_claim_no_link"]


def _gold_record_for(gt, target):
    return next((r for r in gt["records"] if r["target"] == target), None)


def _files_by_name(task):
    return {f["name"]: f for f in task["input"]["files"]}


def _papers(tasks):
    """A content fingerprint per task (repo_url + concatenated file contents)."""
    return [t["input"]["repo_url"] + "".join(f["content"] for f in t["input"]["files"]) for t in tasks]


def test_generate_is_deterministic():
    a = list(generator.generate("v1", seed=3))
    b = list(generator.generate("v1", seed=3))
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert _papers(a) == _papers(b)


def test_envelopes_have_required_fields_and_tags():
    tasks = list(generator.generate("v1", seed=0, split="revealed"))
    assert tasks
    for t in tasks:
        assert {"task_id", "arena_id", "task_set_version", "difficulty", "input"} <= t.keys()
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"
        assert {"repo_url", "files", "targets"} <= t["input"].keys()


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_t1_and_t2_have_no_issues():
    """T2 is the false-alarm trap: suspicious-LOOKING files that are CLEAN."""
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] in (1, 2):
            assert t["difficulty"]["n_issues"] == 0
            gt = generator.ground_truth(t["task_id"])
            assert gt["mistake_kinds"] == []
            assert all(r["issue_kind"] is None for r in gt["records"])


def test_ground_truth_returns_records_and_mistake_kinds():
    tasks = list(generator.generate("v1", seed=0))
    gt = generator.ground_truth(tasks[0]["task_id"])
    assert "records" in gt and "mistake_kinds" in gt
    assert all({"target", "issue_kind"} <= r.keys() for r in gt["records"])


def test_ground_truth_unknown_task_raises():
    import pytest
    with pytest.raises(KeyError):
        generator.ground_truth("or-t9-99-s999")


def test_revealed_set_covers_every_issue_kind():
    """The public benchmark must exercise the full array of injected defects."""
    all_kinds = set(generator.ISSUE_KINDS)
    seen = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        seen.update(gt["mistake_kinds"])
    assert seen == all_kinds


def test_broken_link_target_is_the_repo_url():
    """The broken_link defect's target must be the repo_url string, not a file."""
    found = False
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        for r in gt["records"]:
            if r["issue_kind"] == "broken_link":
                assert r["target"] == t["input"]["repo_url"]
                found = True
    assert found


def test_missing_file_load_target_is_absent_from_files():
    """A missing_file_load points at a file the script load()s but isn't present."""
    checked = False
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        present = {f["name"] for f in t["input"]["files"]}
        for r in gt["records"]:
            if r["issue_kind"] == "missing_file_load":
                # the flagged script is present; the file it loads is not.
                assert r["target"] in present
                content = next(f["content"] for f in t["input"]["files"] if f["name"] == r["target"])
                assert "_raw" in content or "raw_" in content  # loads the absent *_raw file
                checked = True
    assert checked


def test_splits_share_difficulty_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert _papers(rev) != _papers(priv)


# ---- Open-practices-reporting kinds (dead_data_link / available_upon_request /
# ---- materials_claim_no_link) and their matched clean look-alikes. -----------

def test_n_tasks_matches_manifest():
    """20 = T1:2 + T2:2 + T3:7 + T4:5 + T5:2 + T6:2 (7 issue kinds)."""
    assert len(list(generator.generate("v1", seed=0))) == 20


def test_issue_kinds_include_open_practices_kinds():
    for k in STATEMENT_KINDS:
        assert k in generator.ISSUE_KINDS


def test_each_statement_kind_is_emitted_on_a_statement_target():
    """Each new kind appears in the revealed set, flagged on its statement doc."""
    emitted: dict[str, str] = {}
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        for r in gt["records"]:
            if r["issue_kind"] in STATEMENT_KINDS:
                emitted[r["issue_kind"]] = r["target"]
                # the flagged target is an availability-statement doc, present in files[]
                assert r["target"] in _files_by_name(t)
                assert r["target"].endswith(".md")
    assert set(emitted) == set(STATEMENT_KINDS), emitted


def test_dead_data_link_statement_cites_a_placeholder_url():
    """A dead_data_link statement claims availability AT a placeholder/404-style URL."""
    checked = False
    placeholders = ("XXXXX", "USERNAME/REPO", "coming-soon", "localhost")
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        for r in gt["records"]:
            if r["issue_kind"] == "dead_data_link":
                content = _files_by_name(t)[r["target"]]["content"]
                assert "available" in content.lower()
                assert any(p in content for p in placeholders), content
                # and it must NOT cite a genuine resolvable DOI
                assert "doi.org" not in content
                checked = True
    assert checked


def test_available_upon_request_statement_has_request_language_and_no_open_link():
    checked = False
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        for r in gt["records"]:
            if r["issue_kind"] == "available_upon_request":
                content = _files_by_name(t)[r["target"]]["content"].lower()
                assert "upon" in content and "request" in content
                assert "doi.org" not in content and "http" not in content
                checked = True
    assert checked


def test_materials_claim_no_link_has_no_link_and_no_materials_file():
    """The bad materials claim: no URL in the statement AND no materials/ file present."""
    checked = False
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        for r in gt["records"]:
            if r["issue_kind"] == "materials_claim_no_link":
                content = _files_by_name(t)[r["target"]]["content"]
                assert "http" not in content and "doi.org" not in content
                names = set(_files_by_name(t))
                assert not any(n.startswith("materials/") for n in names), names
                checked = True
    assert checked


def test_clean_statement_controls_are_not_flagged():
    """Genuine DOI data statements and real-materials-repo statements stay clean.

    Across the revealed set, any DATA_AVAILABILITY.md citing a resolvable DOI and
    any MATERIALS.md citing a real repo (with a materials/ file present) must have
    gold issue_kind == None (the confusable clean look-alike the player must not flag).
    """
    saw_clean_data = saw_clean_materials = False
    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        files = _files_by_name(t)
        for name, f in files.items():
            if name == "DATA_AVAILABILITY.md" and "doi.org" in f["content"]:
                assert _gold_record_for(gt, name)["issue_kind"] is None
                saw_clean_data = True
            if name == "MATERIALS.md" and "http" in f["content"]:
                assert _gold_record_for(gt, name)["issue_kind"] is None
                # the materials file the clean statement points to is present
                assert any(n.startswith("materials/") for n in files)
                saw_clean_materials = True
    assert saw_clean_data and saw_clean_materials


def test_revealed_and_private_category_counts_match_exactly():
    """count_tolerance is 0: index-cycled kinds => identical category counts."""
    def cats(seed, split):
        c: Counter = Counter()
        for t in generator.generate("v1", seed=seed, split=split):
            gt = generator.ground_truth(t["task_id"])
            for k in (gt["mistake_kinds"] or ["clean"]):
                c[k] += 1
        return c
    assert cats(0, "revealed") == cats(987654, "private")


def test_output_schema_enum_matches_issue_kinds():
    """Drift-guard: the output schema's issue_kind enum == ISSUE_KINDS (+ null).

    Prevents the 'untestable kind' bug class — a generator kind with no schema slot.
    """
    schema = json.loads((ARENA_DIR / "schemas" / "output.schema.json").read_text(encoding="utf-8"))
    enum = schema["properties"]["records"]["items"]["properties"]["issue_kind"]["enum"]
    non_null = [k for k in enum if k is not None]
    assert None in enum  # clean targets use issue_kind=null
    assert set(non_null) == set(generator.ISSUE_KINDS)
    assert len(non_null) == len(generator.ISSUE_KINDS)  # no dupes


def test_player_prompt_documents_every_issue_kind():
    """Drift-guard: every ISSUE_KIND must be taught in the player prompt."""
    prompt = (ARENA_DIR.parents[1] / "players" / "prompts" / "open_practices_repro.txt").read_text(encoding="utf-8")
    for k in generator.ISSUE_KINDS:
        assert k in prompt, f"issue kind {k!r} is not documented in the player prompt"
