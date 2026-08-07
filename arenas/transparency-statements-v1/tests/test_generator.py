"""Tests for the transparency-statements generator."""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("_transparency_statements_generator", ARENA_DIR / "generator.py")
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_transparency_statements_generator"] = generator
_SPEC.loader.exec_module(generator)


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
        assert isinstance(t["input"]["text"], str)


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_t1_and_t2_have_no_mistakes():
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] in (1, 2):
            assert t["difficulty"]["n_mistakes"] == 0
            gt = generator.ground_truth(t["task_id"])
            assert gt["mistake_kinds"] == []


def test_ground_truth_returns_field_map_with_mistake_kinds():
    tasks = list(generator.generate("v1", seed=0))
    gt = generator.ground_truth(tasks[0]["task_id"])
    assert "mistake_kinds" in gt and isinstance(gt["mistake_kinds"], list)
    for field in ("coi", "funding", "data", "code", "materials", "prereg"):
        assert field in gt
    assert {"present", "statement"} <= gt["coi"].keys()
    assert {"available", "on_request", "url"} <= gt["data"].keys()
    assert {"available", "url"} <= gt["prereg"].keys()


def test_ground_truth_unknown_task_raises():
    try:
        generator.ground_truth("ts-does-not-exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown task_id")


def test_revealed_set_covers_every_mistake_kind():
    """The public benchmark must exercise the full array of injected mistakes."""
    all_kinds = set(generator.MISTAKE_KINDS)
    seen = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        seen.update(gt["mistake_kinds"])
    assert seen == all_kinds


def test_t3_single_mistake_each_kind():
    """T3 has exactly one mistake per task and cycles through all kinds."""
    t3 = [t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 3]
    kinds = []
    for t in t3:
        gt = generator.ground_truth(t["task_id"])
        assert len(gt["mistake_kinds"]) == 1
        kinds.append(gt["mistake_kinds"][0])
    assert set(kinds) == set(generator.MISTAKE_KINDS)


def test_missing_coi_omits_coi_line_and_marks_absent():
    t3 = [t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 3]
    target = next(t for t in t3 if generator.ground_truth(t["task_id"])["mistake_kinds"] == ["missing_coi"])
    gt = generator.ground_truth(target["task_id"])
    assert gt["coi"]["present"] is False
    assert "Competing interests" not in target["input"]["text"]


def test_on_request_field_marks_available_false_on_request_true():
    tasks = list(generator.generate("v1", seed=0))
    found = False
    for t in tasks:
        gt = generator.ground_truth(t["task_id"])
        if "data_on_request_not_real" in gt["mistake_kinds"]:
            # at least one open field must be on_request with no real url
            on_req = [f for f in ("data", "code", "materials")
                      if gt[f].get("on_request") and gt[f].get("available") is False
                      and gt[f].get("url") is None]
            # prereg has no on_request flag; handle separately
            prereg_on_req = (gt["prereg"]["available"] is False and gt["prereg"]["url"] is None)
            assert on_req or prereg_on_req
            found = True
    assert found


def test_placeholder_url_marks_available_false_with_nonnull_url():
    tasks = list(generator.generate("v1", seed=0))
    found = False
    for t in tasks:
        gt = generator.ground_truth(t["task_id"])
        if "placeholder_url" in gt["mistake_kinds"]:
            offenders = [f for f in ("data", "code", "materials", "prereg")
                         if gt[f].get("available") is False and gt[f].get("url") is not None]
            assert offenders
            found = True
    assert found


def test_splits_share_difficulty_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert [t["input"]["text"] for t in rev] != [t["input"]["text"] for t in priv]


# --- broadened taxonomy: false_open_claim / false_prereg_claim / funding_on_request ---

NEW_KINDS = {"false_open_claim", "false_prereg_claim", "funding_on_request"}


def test_mistake_kinds_match_documented_taxonomy():
    """Drift guard: the closed kind set is exactly the seven documented kinds.

    There is no player-facing kind ENUM (output is a fixed field map), so this
    pins the generator's taxonomy instead — if a kind is added/removed without
    updating this list (and the prompt + catalog), the test fails.
    """
    assert set(generator.MISTAKE_KINDS) == {
        "missing_coi",
        "missing_funding",
        "data_on_request_not_real",
        "placeholder_url",
        "false_open_claim",
        "false_prereg_claim",
        "funding_on_request",
    }
    # No duplicates, deterministic order preserved.
    assert len(generator.MISTAKE_KINDS) == len(set(generator.MISTAKE_KINDS))


def test_new_kinds_are_emitted_in_revealed_and_each_only_once_per_task_in_t3():
    """Every new kind must appear, and stand alone exactly once in a T3 task."""
    tasks = list(generator.generate("v1", seed=0, split="revealed"))
    seen = set()
    for t in tasks:
        seen.update(generator.ground_truth(t["task_id"])["mistake_kinds"])
    assert NEW_KINDS <= seen

    t3 = [t for t in tasks if t["difficulty"]["tier"] == 3]
    t3_singletons = {
        generator.ground_truth(t["task_id"])["mistake_kinds"][0]
        for t in t3
        if len(generator.ground_truth(t["task_id"])["mistake_kinds"]) == 1
    }
    assert NEW_KINDS <= t3_singletons


def test_false_open_claim_asserts_openness_but_no_link():
    """false_open_claim: a bare 'openly available' assertion with NO url.

    Gold marks the field available=False, on_request=False, url=None, yet the
    rendered text DOES make an openness claim (so the field is not omitted).
    """
    tasks = list(generator.generate("v1", seed=0))
    target = next(
        t for t in tasks
        if generator.ground_truth(t["task_id"])["mistake_kinds"] == ["false_open_claim"]
    )
    gt = generator.ground_truth(target["task_id"])
    offenders = [
        f for f in ("data", "code", "materials")
        if gt[f]["available"] is False and gt[f]["on_request"] is False and gt[f]["url"] is None
    ]
    assert offenders, gt
    # The text asserts openness for the offending field (line is present, not omitted).
    headers = {"data": "Data availability.", "code": "Code availability.",
               "materials": "Materials availability."}
    for f in offenders:
        assert headers[f] in target["input"]["text"]
    assert "openly available in a public repository" in target["input"]["text"] \
        or "publicly available in an online repository" in target["input"]["text"]


def test_false_prereg_claim_bare_claim_no_link():
    """false_prereg_claim: 'was preregistered' with NO registry link/ID."""
    tasks = list(generator.generate("v1", seed=0))
    target = next(
        t for t in tasks
        if generator.ground_truth(t["task_id"])["mistake_kinds"] == ["false_prereg_claim"]
    )
    gt = generator.ground_truth(target["task_id"])
    assert gt["prereg"]["available"] is False
    assert gt["prereg"]["url"] is None
    assert "Preregistration." in target["input"]["text"]
    assert "preregistered prior to data collection" in target["input"]["text"]


def test_funding_on_request_is_not_a_real_funding_statement():
    """funding_on_request: a deferred funding line is NOT a funding disclosure."""
    tasks = list(generator.generate("v1", seed=0))
    target = next(
        t for t in tasks
        if generator.ground_truth(t["task_id"])["mistake_kinds"] == ["funding_on_request"]
    )
    gt = generator.ground_truth(target["task_id"])
    assert gt["funding"]["present"] is False
    assert gt["funding"]["statement"] == ""
    # A funding line IS rendered (the deferral), so the section keeps the header.
    assert "Funding." in target["input"]["text"]
    assert "request" in target["input"]["text"].lower()


def test_no_funding_clean_control_is_present_and_unflagged():
    """The genuine 'no external funding' declaration is a CLEAN control (present)."""
    tasks = list(generator.generate("v1", seed=0))
    nf_phrases = ("no external funding", "no specific grant", "no funding was received")
    controls = [
        t for t in tasks
        if any(p in t["input"]["text"].lower() for p in nf_phrases)
    ]
    assert controls, "expected at least one no-funding clean control task"
    for t in controls:
        gt = generator.ground_truth(t["task_id"])
        # It is a real disclosure: present=True, and it is NOT an injected mistake.
        assert gt["funding"]["present"] is True
        assert "funding_on_request" not in gt["mistake_kinds"]


def test_t6_union_covers_every_kind_without_field_collision():
    """T6 spans every kind across its tasks; no kind is silently overwritten."""
    t6 = [t for t in generator.generate("v1", seed=0) if t["difficulty"]["tier"] == 6]
    union = set()
    for t in t6:
        gt = generator.ground_truth(t["task_id"])
        mk = gt["mistake_kinds"]
        # n_mistakes is the count of actually-injected kinds (no dedup/overwrite).
        assert t["difficulty"]["n_mistakes"] == len(mk)
        # funding can't be both missing AND deferred in one task.
        assert not ({"missing_funding", "funding_on_request"} <= set(mk))
        union.update(mk)
    assert union == set(generator.MISTAKE_KINDS)


def test_every_injected_kind_is_reflected_in_gold_field_state():
    """No collision anywhere: each injected kind shows up in the field map."""
    def reflected(kind, gt):
        if kind == "missing_coi":
            return gt["coi"]["present"] is False
        if kind in ("missing_funding", "funding_on_request"):
            return gt["funding"]["present"] is False
        if kind == "false_prereg_claim":
            return gt["prereg"]["available"] is False and gt["prereg"]["url"] is None
        if kind == "data_on_request_not_real":
            open_hit = any(gt[f].get("on_request") and gt[f]["available"] is False
                           and gt[f].get("url") is None for f in ("data", "code", "materials"))
            return open_hit or (gt["prereg"]["available"] is False and gt["prereg"]["url"] is None)
        if kind == "placeholder_url":
            return any(gt[f]["available"] is False and gt[f].get("url") is not None
                       for f in ("data", "code", "materials", "prereg"))
        if kind == "false_open_claim":
            return any(gt[f]["available"] is False and gt[f].get("on_request") is False
                       and gt[f].get("url") is None for f in ("data", "code", "materials"))
        raise AssertionError(f"unhandled kind {kind}")

    for t in generator.generate("v1", seed=0):
        gt = generator.ground_truth(t["task_id"])
        for kind in gt["mistake_kinds"]:
            assert reflected(kind, gt), (t["task_id"], kind, gt)


def test_prompt_documents_new_failure_modes():
    """The untestable-kind guard: the player prompt must teach the new modes."""
    prompt = (ARENA_DIR.parents[1] / "players" / "prompts" / "transparency_statements.txt").read_text(encoding="utf-8")
    low = prompt.lower()
    # bare openness claim, bare prereg claim, deferred funding all called out.
    assert "openly available in a public repository" in low or "bare openness" in low
    assert "this study was preregistered" in low
    assert "available on request" in low and "details" in low
    assert "no external funding" in low or "no specific grant" in low
