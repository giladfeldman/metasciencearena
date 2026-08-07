"""The public/private boundary must be computed, agreed, and non-vacuous.

Every assertion here guards a mistake that cannot be undone once the mirror is
pushed: a held-out corpus, a private seed, or a real-paper gold path reaching a
public repository. See framework/publish.py for the rule itself.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from framework import publish

REPO = Path(__file__).resolve().parents[2]
ARENAS = REPO / "arenas"

pytestmark = pytest.mark.skipif(
    not ARENAS.is_dir(), reason="arenas/ not present (running from an installed package)"
)


@pytest.fixture(scope="module")
def classified():
    rows = publish.classify_arenas(ARENAS)
    assert rows, "classify_arenas found no arenas — a vacuous pass"
    return rows


def test_the_gold_veto_never_fires_on_a_seeded_arena(classified):
    """The two conditions are NOT symmetric, and the plan was wrong to say so.

    Measured 2026-08-07, after fixing a stripper bug that hid two real imports:

        seeded      16     <- this is the partition: 16 public / 6 private
        gold-free   20     <- 2 arenas import framework.jats
        publishable 16

    So the seed condition alone produces the split, and the gold condition is a
    one-way VETO: it can only make an arena more private, never less. Four
    arenas "disagree" (unseeded but gold-free) and that is expected — they are
    private because their tasks are not seed-reproducible, which is reason
    enough on its own.

    What IS worth pinning is that the veto is currently inert: no arena is
    seed-reproducible AND dependent on real-paper gold. An arena that becomes
    both is a disclosure judgement a human must make, so it fails here.
    """
    both = [
        f"{a.arena_id}: seeded, but depends on {sorted(set(a.gold_markers))}"
        for a in classified
        if a.seed_says_public and a.gold_markers
    ]
    assert both == [], (
        "a seed-reproducible arena now depends on the real-paper gold path:\n  "
        + "\n  ".join(both)
        + "\nThe seed rule would publish it; the gold rule would not. Decide "
          "deliberately and record the decision — do not let the veto resolve it silently."
    )


def test_the_gold_veto_still_recognises_a_real_dependency(classified):
    """Non-vacuity: the veto must actually see the imports that exist.

    It reported ZERO gold markers repo-wide until `code_only` was fixed — an
    inert check that looked like a clean bill of health. At least one arena
    genuinely imports `framework.jats`; if that count drops to zero, either the
    arenas changed or the detector broke, and both need a human.
    """
    with_markers = [a.arena_id for a in classified if a.gold_markers]
    assert with_markers, (
        "no arena anywhere shows a real-paper gold dependency. Some do import "
        "framework.jats, so the detector has stopped detecting — the same failure "
        "as the token-join bug that produced a silently inert veto."
    )


def test_code_only_keeps_a_real_marker_but_drops_prose():
    """The stripper must not hide a dependency, and must not invent one.

    Both directions are load-bearing. Under-stripping misclassified two arenas
    as private because a DOCSTRING said "no article-finder registry needed".
    Over-stripping would hide `AF_DIR = Path.home() / ".claude" / "skills" /
    "article-finder"`, which is a real dependency written as a string literal.
    """
    src = "\n".join([
        "# comment mentioning article-finder",
        '"""Docstring saying no article-finder registry is needed."""',
        'AF_DIR = Path.home() / ".claude" / "skills" / "article-finder"',
        "def f():",
        '    """Inner docstring mentioning framework.gold."""',
        "    from framework.jats import parse_tables",
        "    return parse_tables",
        "",
    ])
    code = publish.code_only(src)

    # A DOTTED import is the shape that actually broke: an earlier version
    # rebuilt the source by joining tokens with spaces, turning this into
    # "from framework . jats import", which no marker matches. Three real-paper
    # arenas were reported gold-free because of it.
    assert "framework.jats" in code, (
        f"over-stripped: a real dotted import was reshaped or removed: {code!r}"
    )
    # A string-literal dependency must survive too.
    assert "article-finder" in code, "over-stripped: a real string-literal dependency vanished"
    assert code.count("article-finder") == 1, (
        f"prose survived stripping, so a comment can still condemn an arena: {code!r}"
    )
    assert "framework.gold" not in code, "an inner docstring survived stripping"


def test_the_split_is_complementary_and_covers_everything(classified):
    """No arena is both, none is neither — the partition has no gap."""
    public = {a.arena_id for a in classified if a.publishable}
    private = {a.arena_id for a in classified if not a.publishable}
    assert public & private == set()
    assert public | private == {a.arena_id for a in classified}
    assert public, "no arena is publishable — the mirror would ship no arenas at all"
    assert private, (
        "EVERY arena is publishable. That is either a real change or the predicate "
        "stopped discriminating; the real-paper arenas must stay private."
    )


def test_real_paper_arenas_are_never_publishable(classified):
    """The named real-paper arenas stay private, whatever the predicate computes.

    A belt-and-braces pin: these six are private for reasons (copyrighted APA
    full text, a shared 30-paper PMC holdout) that outlive any refactor of the
    rule above.
    """
    must_stay_private = {
        "pdf-citation-matching-v1",
        "pdf-reference-parsing-v1",
        "pdf-section-structure-v1",
        "pdf-table-extraction-v1",
        "pdf-text-fidelity-v1",
        "replication-target-lookup-v1",
    }
    by_id = {a.arena_id: a for a in classified}
    for arena_id in sorted(must_stay_private):
        if arena_id not in by_id:
            continue  # arena retired; nothing to protect
        assert not by_id[arena_id].publishable, (
            f"{arena_id} became publishable. It resolves gold from real papers "
            f"(copyrighted or shared-holdout), so this is a disclosure decision, "
            f"not a refactor."
        )


def test_manifest_never_includes_a_private_path():
    files = publish.public_manifest(REPO)
    assert files, "the public manifest is empty — nothing would be mirrored"
    posix = [p.as_posix() for p in files]

    banned = [p for p in posix if publish.NEVER_PUBLISH.search(p)]
    assert banned == [], f"denylisted paths reached the manifest: {banned}"

    for p in posix:
        assert "_held_out" not in p, p
        assert not p.endswith(".private_seed"), p
        assert not p.endswith("_ground_truth.json"), p
        assert "/runs/" not in p, p
        # The Next.js app is private in full: it holds auth, the admin surface
        # and the built data bundle.
        assert not p.startswith("leaderboard-app/"), p
        assert not p.startswith("docs/outreach/"), p


def test_manifest_excludes_every_private_arena():
    private_ids = {a.arena_id for a in publish.classify_arenas(ARENAS) if not a.publishable}
    assert private_ids, "no private arenas — this test would prove nothing"
    posix = [p.as_posix() for p in publish.public_manifest(REPO)]
    for arena_id in private_ids:
        hits = [p for p in posix if p.startswith(f"arenas/{arena_id}/")]
        assert hits == [], f"{arena_id} is private but {len(hits)} of its files are in the manifest: {hits[:5]}"


def test_manifest_includes_what_reproduction_actually_needs():
    """A mirror that omits the scorer cannot reproduce a score."""
    posix = {p.as_posix() for p in publish.public_manifest(REPO)}
    assert "framework/scoring/__init__.py" in posix or any(
        p.startswith("framework/scoring/") for p in posix
    ), "the scoring package is missing — the whole point of the mirror"
    assert "framework/holdout.py" in posix, "the contamination boundary must be auditable"
    assert "framework/publish.py" in posix, (
        "the publish rule itself must be public: a benchmark claiming contamination "
        "resistance should let readers check the boundary, not take it on trust"
    )
    assert any(p.startswith("framework/contract/schemas/") for p in posix)
    public_ids = {a.arena_id for a in publish.classify_arenas(ARENAS) if a.publishable}
    for arena_id in sorted(public_ids):
        assert f"arenas/{arena_id}/scorer.py" in posix or f"arenas/{arena_id}/generator.py" in posix, (
            f"{arena_id} is publishable but neither its generator nor its scorer is mirrored"
        )


def test_leak_scan_is_not_vacuous():
    """The scanner must refuse to certify when it has nothing to compare against."""
    empty = Path(__file__).parent / "__nonexistent_repo__"
    reasons = publish.scan_for_leaks(empty, [])
    assert reasons, "a scan with no seeds and no held-out material reported CLEAN"
    assert "REFUSING TO CERTIFY" in reasons[0]


def test_leak_scan_detects_a_pasted_held_out_passage(tmp_path):
    """Construct the exact accident a path rule cannot see, and catch it."""
    secret = (
        "Participants completed a battery of twelve counterbalanced vignettes "
        "describing morally ambiguous workplace scenarios, rating each on a "
        "seven-point scale anchored at 1 (completely unacceptable) and 7 "
        "(completely acceptable), with attention checks interleaved."
    )
    held = tmp_path / "arenas" / "a1" / "task_sets" / "v1" / "_held_out"
    held.mkdir(parents=True)
    (held / "case.txt").write_text(secret, encoding="utf-8")
    seedfile = tmp_path / "arenas" / "a1" / "task_sets" / "v1" / ".private_seed"
    seedfile.write_text("s3cr3t-seed-value-0001", encoding="utf-8")

    # A file that a path allowlist would happily mirror: it is a scorer fixture.
    innocent_looking = tmp_path / "arenas" / "a1" / "tests" / "fixtures.py"
    innocent_looking.parent.mkdir(parents=True)
    innocent_looking.write_text(f'SAMPLE = """{secret}"""\n', encoding="utf-8")

    rel = innocent_looking.relative_to(tmp_path)
    reasons = publish.scan_for_leaks(tmp_path, [rel])
    assert any("verbatim" in r for r in reasons), (
        f"a held-out passage pasted into a mirrored fixture was not detected: {reasons}"
    )


def test_leak_scan_detects_a_private_seed_value(tmp_path):
    held = tmp_path / "arenas" / "a1" / "task_sets" / "v1" / "_held_out"
    held.mkdir(parents=True)
    (held / "case.txt").write_text("x" * 400, encoding="utf-8")
    (tmp_path / "arenas" / "a1" / "task_sets" / "v1" / ".private_seed").write_text(
        "9f2c1ab77e5d4410", encoding="utf-8"
    )
    leaky = tmp_path / "arenas" / "a1" / "arena.yaml"
    leaky.write_text("benchmark_splits:\n  private:\n    seed: 9f2c1ab77e5d4410\n", encoding="utf-8")

    reasons = publish.scan_for_leaks(tmp_path, [leaky.relative_to(tmp_path)])
    assert any(".private_seed value" in r for r in reasons), reasons


def test_leak_scan_passes_a_genuinely_clean_file(tmp_path):
    """Non-vacuity in the other direction: it must not flag everything."""
    held = tmp_path / "arenas" / "a1" / "task_sets" / "v1" / "_held_out"
    held.mkdir(parents=True)
    (held / "case.txt").write_text("A held-out passage. " * 40, encoding="utf-8")
    (tmp_path / "arenas" / "a1" / "task_sets" / "v1" / ".private_seed").write_text(
        "abc123", encoding="utf-8"
    )
    clean = tmp_path / "arenas" / "a1" / "scorer.py"
    clean.write_text("def score(output, gold):\n    return {'primary': 1.0}\n", encoding="utf-8")

    assert publish.scan_for_leaks(tmp_path, [clean.relative_to(tmp_path)]) == []


def test_the_real_manifest_is_clean():
    """The end-to-end assertion: today's mirror leaks nothing."""
    files = publish.public_manifest(REPO)
    reasons = publish.scan_for_leaks(REPO, files)
    assert reasons == [], "the CURRENT public manifest would leak:\n  " + "\n  ".join(reasons)


def test_the_gold_authoring_path_is_never_mirrored():
    """`framework/gold/` and `framework/jats.py` must stay private.

    They reached the FIRST real mirror build through a blanket
    `framework/**/*.py` allowlist, and nothing automated caught it — the
    verbatim scanner cannot, because these files contain code that *reads* the
    answer key rather than the answer key itself. A human reading the output
    file list caught it (2026-08-07).

    `framework/gold/__init__.py` documents itself as "the ONLY way an arena
    obtains ground truth" and warns that "a player that can read this module can
    read the answer key". Publishing it publishes the route to every answer.
    """
    posix = {p.as_posix() for p in publish.public_manifest(REPO)}
    leaked = sorted(p for p in posix if p.startswith("framework/gold/") or p == "framework/jats.py")
    assert leaked == [], (
        "the gold-AUTHORING path is in the public manifest: " + ", ".join(leaked)
    )


def test_any_module_reaching_the_gold_path_is_refused():
    """The class-level rule, not just the two known files.

    A future `framework/whatever.py` that imports the gold client must be
    refused by the scan, not by someone remembering to add it to a denylist.
    """
    fake = REPO / "framework" / "__does_not_exist__.py"
    # Build the check against a synthetic tree so it does not depend on a real
    # offending file existing (there is none, by design).
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "arenas" / "a1" / "task_sets" / "v1" / "_held_out").mkdir(parents=True)
        (root / "arenas" / "a1" / "task_sets" / "v1" / "_held_out" / "c.txt").write_text(
            "held out material " * 40, encoding="utf-8"
        )
        (root / "arenas" / "a1" / "task_sets" / "v1" / ".private_seed").write_text("s", encoding="utf-8")
        mod = root / "framework" / "resolver.py"
        mod.parent.mkdir(parents=True)
        mod.write_text(
            "from framework.gold import fetch\n\n\ndef go():\n    return fetch()\n", encoding="utf-8"
        )
        reasons = publish.scan_for_leaks(root, [mod.relative_to(root)])
    assert any("gold path" in r for r in reasons), (
        f"a module importing framework.gold was not refused: {reasons}"
    )
    assert not fake.exists()  # sanity: the test invented nothing in the real tree
