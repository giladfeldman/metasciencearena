"""Tests for the prereg-extraction generator."""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ARENA_DIR.parents[1]
PROMPT_PATH = REPO_ROOT / "players" / "prompts" / "prereg_extraction.txt"
_SPEC = importlib.util.spec_from_file_location("_prereg_extraction_generator", ARENA_DIR / "generator.py")
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_prereg_extraction_generator"] = generator
_SPEC.loader.exec_module(generator)

# The integrity abuses mined from AbusingPreReg (Tier D modules 12/13/17) added
# on top of the original three field-map mistakes.
_NEW_KINDS = {
    "viewonly_instead_of_doi",
    "embargoed_at_publication",
    "withdrawn_still_cited",
}


def _gt_by_kind(kind, seed=0):
    for t in generator.generate("v1", seed=seed, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        if kind in gt["mistake_kinds"]:
            return t, gt
    raise AssertionError(f"no task with injected kind {kind!r}")


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


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_t2_is_a_clean_false_alarm_trap():
    """T2 decoy tasks must be clean controls (no injected mistake) whose gold is
    prereg_found=False — the false-alarm trap."""
    for t in generator.generate("v1", seed=0):
        if t["difficulty"]["tier"] == 2:
            gt = generator.ground_truth(t["task_id"])
            assert gt["prereg_found"] is False
            assert gt["mistake_kinds"] == []


def test_ground_truth_shape():
    tasks = list(generator.generate("v1", seed=0))
    gt = generator.ground_truth(tasks[0]["task_id"])
    assert {"prereg_found", "platform", "link", "fields", "mistake_kinds"} <= gt.keys()
    assert set(gt["fields"].keys()) == {"hypotheses", "design", "sample_size", "analysis_plan"}


def test_ground_truth_raises_for_unknown_task():
    try:
        generator.ground_truth("does-not-exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown task_id")


def test_revealed_set_covers_every_injected_kind():
    """The public benchmark must exercise the full array of injected mistakes."""
    all_kinds = set(generator.INJECTED_KINDS)
    seen = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        seen.update(gt["mistake_kinds"])
    assert all_kinds <= seen


def test_both_platforms_appear_in_revealed():
    platforms = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        if gt["platform"]:
            platforms.add(gt["platform"])
    assert platforms == {"osf", "aspredicted"}


def test_splits_share_tier_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter(t["difficulty"]["tier"] for t in tasks)

    assert cells(rev) == cells(priv)
    assert [t["input"]["text"] for t in rev] != [t["input"]["text"] for t in priv]


# --- New integrity-abuse modes (AbusingPreReg modules 12/13/17) ---

def test_new_integrity_kinds_are_emitted():
    """Each new injected abuse must actually appear in the revealed suite."""
    seen = set()
    for t in generator.generate("v1", seed=0, split="revealed"):
        seen.update(generator.ground_truth(t["task_id"])["mistake_kinds"])
    assert _NEW_KINDS <= seen


def test_viewonly_gold_is_osf_with_canonical_link():
    """viewonly_instead_of_doi: real reg, platform osf, canonical (non-view_only)
    link, fields present. The view-only token must NOT be in the gold link."""
    _, gt = _gt_by_kind("viewonly_instead_of_doi")
    assert gt["prereg_found"] is True
    assert gt["platform"] == "osf"
    assert gt["link"] and "view_only" not in gt["link"]
    assert gt["link"].startswith("https://osf.io/")
    assert all(gt["fields"][f] for f in ("hypotheses", "design", "sample_size", "analysis_plan"))


def test_viewonly_text_actually_shows_a_view_only_link():
    """The injected task text must contain the anonymized view-only URL (so the
    canonical-link recovery is a real challenge), even though gold drops it."""
    t, _ = _gt_by_kind("viewonly_instead_of_doi")
    assert "view_only=" in t["input"]["text"]


def test_embargoed_gold_found_but_fields_null():
    """embargoed_at_publication: registration referenced (found=True, platform
    osf) but contents not public -> every field null."""
    _, gt = _gt_by_kind("embargoed_at_publication")
    assert gt["prereg_found"] is True
    assert gt["platform"] == "osf"
    assert gt["link"]
    assert all(gt["fields"][f] is None for f in ("hypotheses", "design", "sample_size", "analysis_plan"))


def test_withdrawn_gold_is_not_found():
    """withdrawn_still_cited: a tombstone is not a usable prereg -> found=False,
    no platform/link/fields, even though the text claims pre-registration."""
    t, gt = _gt_by_kind("withdrawn_still_cited")
    assert gt["prereg_found"] is False
    assert gt["platform"] is None
    assert gt["link"] is None
    assert all(gt["fields"][f] is None for f in ("hypotheses", "design", "sample_size", "analysis_plan"))
    # The trap: the text asserts pre-registration and shows a (dead) link.
    assert "withdrawn" in t["input"]["text"].lower()


def test_clean_controls_carry_no_mistake_and_are_found():
    """The three confusable clean look-alikes (DOI link, embargo-lifted,
    live-cited) must be unflagged real regs with extractable fields."""
    # Identify them by their surface markers in the rendered text.
    markers = {
        "doi.org/10.17605/osf.io/": False,        # doi_clean
        "embargo lifted prior to publication": False,  # embargo_lifted
        "registration remains publicly available": False,  # live_cited
    }
    for t in generator.generate("v1", seed=0, split="revealed"):
        gt = generator.ground_truth(t["task_id"])
        text = t["input"]["text"].lower()
        for marker in markers:
            if marker in text:
                markers[marker] = True
                # A clean control: no injected mistake, found, all fields present.
                assert gt["mistake_kinds"] == [], (marker, gt["mistake_kinds"])
                assert gt["prereg_found"] is True
                assert gt["platform"] == "osf"
                assert all(gt["fields"][f] for f in ("hypotheses", "design", "sample_size", "analysis_plan"))
    assert all(markers.values()), f"missing clean-control surface forms: {markers}"


def test_every_injected_kind_appears_in_t3_single():
    """T3 (single-mistake) must cover all six injected kinds exactly once."""
    t3_kinds = Counter()
    for t in generator.generate("v1", seed=0, split="revealed"):
        if t["difficulty"]["tier"] == 3:
            for k in generator.ground_truth(t["task_id"])["mistake_kinds"]:
                t3_kinds[k] += 1
    assert set(t3_kinds) == set(generator.INJECTED_KINDS)
    assert all(v == 1 for v in t3_kinds.values())


def test_t4_pairs_each_abuse_with_its_clean_lookalike():
    """T4 (subtle) must contain BOTH the abuse and its confusable clean twin for
    each new integrity mode — that adjacency is the point of the tier."""
    t4_kinds = set()
    t4_clean_markers = {"doi.org/10.17605/osf.io/": False,
                        "embargo lifted prior to publication": False,
                        "registration remains publicly available": False}
    for t in generator.generate("v1", seed=0, split="revealed"):
        if t["difficulty"]["tier"] != 4:
            continue
        t4_kinds.update(generator.ground_truth(t["task_id"])["mistake_kinds"])
        text = t["input"]["text"].lower()
        for m in t4_clean_markers:
            if m in text:
                t4_clean_markers[m] = True
    assert _NEW_KINDS <= t4_kinds                 # the abuses
    assert all(t4_clean_markers.values()), t4_clean_markers  # their clean twins


def test_splits_have_identical_category_counts_exact_parity():
    """count_tolerance is 0: every injected category count must match EXACTLY
    between revealed and private (form placement is index-cycled, seed-free)."""
    def cats(seed, split):
        c = Counter()
        for t in generator.generate("v1", seed=seed, split=split):
            ks = generator.ground_truth(t["task_id"])["mistake_kinds"] or ["clean"]
            for k in ks:
                c[k] += 1
        return c

    assert cats(0, "revealed") == cats(1_000_003, "private")


def test_prompt_documents_each_new_mode():
    """Drift guard (the untestable-kind bug class): the player prompt MUST explain
    every new judgment call, or the mode is untestable. Mirrors the enum/prompt
    sync we enforce elsewhere — here the contract lives in the prompt, not an enum."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8").lower()
    # withdrawn -> prereg_found=false
    assert "withdrawn" in prompt and "tombstone" in prompt
    # embargoed -> found but fields null
    assert "embargo" in prompt
    # view-only OSF link is still platform osf; recover the canonical link
    assert "view_only" in prompt or "view-only" in prompt
    # OSF DOI form is still platform "osf"
    assert "10.17605/osf.io" in prompt
