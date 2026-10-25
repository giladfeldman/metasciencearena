"""No file the public mirror ships may identify a HELD-OUT paper.

WHY THIS FILE EXISTS. The held-out split's whole value is that nobody can tune
against it. `framework/holdout.py` strips held-out gold, output, `input_hash`
and breakdown from anything published, and `framework/publish.py` treats every
path outside `PUBLIC_TOP_LEVEL` as private by default. Nothing checked the one
remaining channel: a mirrored file that simply NAMES a held-out paper in prose.

Membership is not gold. It is still a step toward the contamination the split
exists to prevent — knowing a paper is in the held-out set tells you what to
tune on, and a per-paper score printed beside it tells you how well you are
doing. This is the edge the portfolio CLAUDE.md means by "adding a consumer-side
contract test at an unchecked edge is worth more than almost any feature work".

WHAT IT CAUGHT, 2026-09-02, on the tree that created it:

  * `framework/tests/test_arenas_private_is_gitignored.py` hardcoded one real
    held-out PMC id as example data. Fixed in `e30a901` before this gate existed
    — the article-custody scan flagged the file for prose density, and reading
    the file rather than trusting the flag's stated reason is what found it.
  * `players/registry.yaml` — 3 real held-out accessions in a prose comment,
    each printed beside that paper's per-player score delta. Redacted the same
    day; the comment's point survives without them. The block ALSO identified
    one paper indirectly ("the paper docpluck's release note names"), which no
    regex here would ever catch — an indirect pointer leaks membership exactly
    as well as an accession, and it went too.

THE MEASUREMENT MISTAKE THAT MADE THIS GATE NECESSARY, and the reason
`_mirrored_files()` looks the way it does. The first probe of this question
reported "0 PMC identifiers across every file in `contract/public_manifest.txt`"
and that number reached a commit message. It was wrong. The manifest is CRLF,
the probe did `[ -f "$f" ]` without stripping the trailing `\\r`, and so it opened
**zero** files — a zero manufactured by the filter, not observed in the data. The
true count was five. That is why `test_the_mirrored_file_list_is_not_vacuous` below is not
decoration: a scan that reads nothing reports the same clean green as a scan that
reads everything and finds nothing.

WHAT IT MISSED FOR THREE DAYS, and why there are now TWO identifier families.
`_PMC` was the only pattern in this file, and there was no DOI pattern anywhere.
The `pdf-text-fidelity-v1` held-out corpus is keyed by PMC accession, so it was
covered. The `pdf-section-structure-v1` held-out corpus is keyed by DOI ONLY, so
those papers were not covered at all: a mirrored file could name any of them and
this gate stayed green. That is not hypothetical either — `framework/article_custody.py`,
written while moving the held-out PDFs into article-finder's custody, carried one
real held-out DOI in a docstring example, and this file said the mirror was clean.
Found 2026-09-05 by scanning the mirror against the corpus rather than trusting
the green; scrubbed to the synthetic prefix in the same change that added the
DOI half below.

WHY THE DOI HALF IS SHAPED DIFFERENTLY FROM THE PMC HALF. A PMC accession is
essentially never legitimate in this repo, so a bare pattern is a usable gate.
DOIs are the opposite: measured 2026-09-05, 14 mirrored files legitimately name
48 distinct DOIs (arena catalogs, prompts, READMEs), none of them held out. So a
bare "any DOI is a leak" rule would flag 14 files on a clean tree and be turned
off within a week. The DOI half is therefore two complementary checks:

  * `DOI_BASELINE` — a self-contained INVENTORY of legitimate DOI usage, FILE and
    exact COUNT only, enforced by equality. It works in the public checkout,
    where the corpus does not exist, and it fires on the case that matters: a
    NEW mirrored file (or a new DOI in an existing one) naming a paper. A count
    that grew is a review trigger, not an accusation — bump it once you have
    read what changed.
  * the corpus-aware check — in a private checkout the held-out DOIs can be read
    from each paper's `source.json`, so the exact set is known and a SUBSTITUTED
    DOI (count unchanged) is caught too. Skipped where the corpus is absent.

Neither half writes a real identifier into this file, and neither failure message
prints one: both report FILE and COUNT, as the note on KNOWN_EXPOSED requires.

WHY A PATTERN AND NOT THE REAL CORPUS. This module is itself mirrored
(`framework/**/*.py`), so it runs in the public checkout, where the held-out
corpus does not exist and cannot be enumerated. A self-contained regex is the
only form of this check that works in the place the disclosure would actually
land. It also means this file must never name a real id itself — including in
KNOWN_EXPOSED, which therefore records the FILE and the COUNT and never the
identifiers. Writing them here would be the very leak being policed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "contract" / "public_manifest.txt"

#: PMC accession numbers. The held-out corpora for `pdf-text-fidelity-v1` and
#: `pdf-section-structure-v1` are keyed by these.
_PMC = re.compile(r"PMC\d{6,}")

#: A LEADING ZERO marks a synthetic accession — the convention already in use for
#: fixtures (`PMC0000001` in `test_holdout_breakdown_allowlist.py`) and for the
#: placeholder this repo adopted in `e30a901` (`PMC00000000`).
#:
#: Measured 2026-09-02 before adopting it: 0 of the 30 real held-out accessions
#: begin with a zero — they are all in the PMC12/13-million range. So this cannot
#: mask a genuine one, which is the only way a placeholder rule could turn this
#: gate into a rubber stamp. Checked rather than assumed, because a too-generous
#: placeholder pattern is indistinguishable from a working gate.
_PLACEHOLDER = re.compile(r"^PMC0\d*$")

#: DOI strings. The `pdf-section-structure-v1` held-out corpus is keyed by these
#: and by nothing else, which is why `_PMC` alone left it unprotected.
#:
#: The trailing class stops at whitespace and at the punctuation that ends a
#: sentence or closes a bracket; a trailing period is stripped separately by
#: `_dois`, because "…see 10.0000/x.1." is one DOI and a full stop, not a DOI
#: ending in a dot.
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>,;)\]}]+")

#: `10.0000` is not an assigned DOI registrant prefix, so it is the DOI analogue
#: of the leading-zero `PMC0…` placeholder above, and fixtures use it.
#:
#: Measured 2026-09-05 against every held-out paper's `source.json` before
#: adopting it: 0 of the 36 real held-out DOIs begin with `10.0000/`. So this
#: cannot mask a genuine one — the only way a placeholder rule could quietly
#: turn a gate into a rubber stamp. Checked rather than assumed, for the same
#: reason the `PMC0…` claim above was checked.
_DOI_PLACEHOLDER = re.compile(r"^10\.0000/")

#: Legitimate, NON-held-out DOI usage in the mirror: FILE -> exact count of
#: distinct DOIs, measured 2026-09-05 (14 files, 48 DOIs).
#:
#: This is an INVENTORY, not a dispensation. It is enforced by EQUALITY, so a
#: file that gains a DOI goes red even though the file is already listed. That
#: is the whole point: on a clean tree every one of these is an arena catalog,
#: a player prompt or a README citing published work, and the disclosure risk is
#: a paper appearing where one did not appear before.
#:
#: The identifiers themselves are deliberately NOT written here — see the note
#: on KNOWN_EXPOSED. A count is not an identifier.
DOI_BASELINE: dict[str, int] = {
    "arenas/open-practices-repro-v1/catalogs/repo_templates.yaml": 4,
    "arenas/p-curve-v1/README.md": 1,
    "arenas/prereg-extraction-v1/CHANGELOG.md": 1,
    "arenas/prereg-extraction-v1/README.md": 1,
    "arenas/prereg-extraction-v1/generator.py": 2,
    "arenas/prereg-extraction-v1/tests/test_generator.py": 2,
    "arenas/reference-integrity-v1/catalogs/clean_controls.yaml": 6,
    "arenas/reference-integrity-v1/catalogs/references.yaml": 17,
    "arenas/transparency-statements-v1/catalogs/statements.yaml": 4,
    "framework/tests/test_scimeto_replication.py": 2,
    # INSPECT-SR (BMJ 2026), the published paper the T2 arena family is modelled
    # on; cited as a provenance basis ref. Published work, not a held-out paper.
    "framework/tests/test_task_provenance.py": 1,
    "players/adapters/metacheck.R": 3,
    "players/prompts/open_practices_repro.txt": 3,
    "players/prompts/prereg_extraction.txt": 1,
    "players/prompts/reference_integrity.txt": 1,
    # 2026-09-28: synthetic reference DOI on the Crossref example prefix (cannot resolve)
    "framework/paperforge/docbuild.py": 1,
    # 2026-10-20: synthetic reference DOI on the Crossref example prefix 10.5555 (cannot resolve),
    # built from a seeded random number, in the PDF-universe generator
    "framework/pdfforge/content.py": 1,
    "framework/pdfforge/refstyle.py": 1,
    # 2026-09-28: synthetic reference DOI on the Crossref example prefix (cannot resolve)
    "framework/tests/test_paperforge.py": 1,
    "taxonomy/leaves/02-preregistration/trial-registration-consistency.yaml": 1,
    "taxonomy/leaves/06-results-reporting/baseline-allocation-plausibility.yaml": 1,
    "taxonomy/leaves/06-results-reporting/correlation-matrix-consistency.yaml": 1,
    "taxonomy/leaves/06-results-reporting/participant-flow-consistency.yaml": 1,
    "taxonomy/leaves/07-writing-claims/text-integrity-duplication.yaml": 1,
    "taxonomy/leaves/12-corrections/post-publication-notice-detection.yaml": 1,
    "taxonomy/leaves/15-synthesis/rct-trustworthiness-composite.yaml": 1,
}

#: Where the held-out corpora live, for the corpus-aware half. Absent in the
#: public checkout, which is why that half skips rather than fails there.
_HELD_OUT_CORPORA = (
    "arenas/pdf-text-fidelity-v1/task_sets/v1/_held_out/pmc",
    "arenas/pdf-section-structure-v1/task_sets/v1/_held_out/apa",
)

#: Files whose exposure is KNOWN, with what has to happen about it.
#:
#: An entry is a DEBT, not a dispensation — the idiom is taken from
#: `test_private_seed_not_in_tracked_files.py`. A blanket skip would hide the
#: next leak exactly as well as having no test; naming the one offender keeps the
#: gate live for the other 525 mirrored files.
#:
#: WHEN YOU ADD AN ENTRY, PAIR IT WITH A STRICT-XFAIL TEST asserting the debt is
#: still outstanding. That makes the entry self-deleting: clean the file and
#: forget to prune this, and the XPASS turns the suite red. Delete BOTH together
#: — an emptied dict leaves such a test vacuously true, so `strict=True` would
#: then fail on a tree that is actually clean. That is why no xfail remains here.
#:
#: NOTE the identifiers are deliberately NOT written here. See the module
#: docstring — this file is mirrored too.
KNOWN_EXPOSED: dict[str, str] = {
    # EMPTY, and that is the point: the one debt was paid the day it was
    # recorded. Anything added here must record the FILE and the COUNT and
    # never the identifiers, because this module is mirrored too.
}


def _mirrored_files() -> list[Path]:
    """Every path in the public manifest that exists on disk.

    `.strip()` is load-bearing: the manifest is CRLF, and a trailing `\\r` makes
    every existence check fail, which silently turns this whole module into a
    scan of nothing. See the module docstring.
    """
    out: list[Path] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        p = REPO / rel
        if p.is_file():
            out.append(p)
    return out


def _real_ids(text: str) -> set[str]:
    """Held-out-shaped identifiers, placeholders excluded."""
    return {m for m in _PMC.findall(text) if not _PLACEHOLDER.match(m)}


def _dois(text: str) -> set[str]:
    """DOI-shaped strings, normalised and placeholders excluded.

    Lower-cased because DOIs are case-insensitive, and a trailing period is
    stripped because sentence punctuation is not part of the identifier. Both
    normalisations must match the one used to read `source.json`, or a real
    held-out DOI would fail to compare equal and the corpus-aware check would
    report a clean mirror it never actually tested.
    """
    return {d for d in (m.lower().rstrip(".") for m in _DOI.findall(text))
            if not _DOI_PLACEHOLDER.match(d)}


def _names_verbatim(text: str, ids: set[str]) -> bool:
    """True if `text` contains any of `ids` as a bounded substring.

    WHY THIS EXISTS ALONGSIDE `_dois`. Extraction-then-intersection — run the
    DOI regex, compare what it returns — is the right tool for the BASELINE
    inventory below, which needs to COUNT distinct DOIs. It is the wrong tool
    for asking "does this file name a held-out paper", and the difference was
    measured on 2026-09-08: a tracked file carried a real DOI concatenated with
    the following word by a PDF text-layer defect (`<doi>Personality`). `_DOI`
    greedily consumes that suffix, so the extracted string matched nothing in
    the held-out set and the line read CLEAN — while any reader, or scraper,
    sees the paper plainly.

    Boundary rules, and note the first one is deliberately narrow: the preceding
    character must not be ALPHANUMERIC (and nothing more — the commonest way to
    write a DOI is `https://doi.org/<doi>`, where a stricter rule rejecting `/`
    would ignore the likeliest form of the leak), and the following character
    must not be a DIGIT, because one more digit makes a different identifier.

    DELIBERATELY DUPLICATED from `test_tracked_files_name_no_heldout_paper.py`
    rather than imported. Both modules are mirrored, and a cross-module import
    between two test files is exactly the kind of thing that resolves on a
    workstation and fails collection in the public checkout — this repo lost 16
    days to one such difference already. Ten self-contained lines cost less than
    a mirrored import that only the public CI can falsify. If you change the
    rules here, change them there, and both files' controls will tell you.
    """
    low = text.lower()
    for ident in ids:
        start = 0
        while True:
            i = low.find(ident, start)
            if i < 0:
                break
            before = low[i - 1] if i > 0 else ""
            after = low[i + len(ident)] if i + len(ident) < len(low) else ""
            if not before.isalnum() and not after.isdigit():
                return True
            start = i + 1
    return False


def _doi_counts() -> dict[str, int]:
    """mirrored path -> number of distinct real-shaped DOIs it names."""
    counts: dict[str, int] = {}
    for p in _mirrored_files():
        try:
            found = _dois(p.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if found:
            counts[p.relative_to(REPO).as_posix()] = len(found)
    return counts


def _held_out_dois() -> set[str]:
    """Every held-out paper's DOI, read from its `source.json`.

    Returns an empty set where the corpora are absent — the public checkout —
    which is what makes the corpus-aware test skip rather than certify.
    """
    out: set[str] = set()
    for rel in _HELD_OUT_CORPORA:
        corpus = REPO / rel
        if not corpus.is_dir():
            continue
        for record in corpus.glob("*/source.json"):
            try:
                import json
                doi = json.loads(record.read_text(encoding="utf-8")).get("doi")
            except (OSError, ValueError):
                continue
            if doi:
                out.add(doi.lower().rstrip("."))
    return out


def _held_out_keys() -> set[str]:
    """Every held-out paper's DIRECTORY NAME, lower-cased.

    THE THIRD IDENTIFIER FAMILY, added 2026-09-08 after cross-model review
    (Grok 4.6) pointed out that this module had exactly two: `_PMC` for
    accessions and `_DOI` for DOIs. The APA-keyed corpus identifies its papers
    by an `<author>_<year>_<journal>` slug — no accession, and the slug is not
    the DOI. A mirrored file naming one matched NEITHER family and this gate
    stayed green. That is the same shape of miss that left the DOI half absent
    until 2026-09-05: a family nobody had enumerated.

    The shape guard (length plus a digit) keeps a generic corpus directory name
    out of the set, for the same reason it does in
    `test_tracked_files_name_no_heldout_paper.py`.
    """
    keys: set[str] = set()
    for rel in _HELD_OUT_CORPORA:
        corpus = REPO / rel
        if not corpus.is_dir():
            continue
        for record in corpus.glob("*/source.json"):
            name = record.parent.name
            if len(name) >= 8 and any(c.isdigit() for c in name):
                keys.add(name.lower())
    return keys


def _corpus_contributions() -> dict[str, int]:
    """corpus path -> how many identifiers it contributed. For the floor below."""
    out: dict[str, int] = {}
    for rel in _HELD_OUT_CORPORA:
        corpus = REPO / rel
        if not corpus.is_dir():
            continue
        out[rel] = len(list(corpus.glob("*/source.json")))
    return out


def _offenders() -> dict[str, set[str]]:
    """mirrored path (repo-relative, forward slashes) -> real ids it names."""
    found: dict[str, set[str]] = {}
    for p in _mirrored_files():
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        ids = _real_ids(text)
        if ids:
            found[p.relative_to(REPO).as_posix()] = ids
    return found


@pytest.mark.skipif(not MANIFEST.is_file(),
                    reason="contract/public_manifest.txt absent — not a repo checkout")
def test_the_mirrored_file_list_is_not_vacuous():
    """Refuse to certify a run that read nothing — and say which half it certifies.

    RENAMED AND WIDENED 2026-09-11, after the same defect was found and fixed in
    the sibling module `test_tracked_files_name_no_heldout_paper.py` (`bfd02c2`).
    The old name was `test_the_scan_is_not_vacuous`, and "the scan" here is the
    product of TWO inputs:

      * the mirrored FILE LIST, from `contract/public_manifest.txt`;
      * the held-out IDENTIFIER SET, from each paper's `source.json`.

    It asserted the first and said nothing about the second, under a name that
    promised both. An empty identifier set does not merely weaken the corpus-aware
    gates, it removes them: all four of them skip on it.

    MEASURED on a corpus-less checkout of `db5f376` — a disposable worktree with
    both paper corpora moved out, which is what every CI runner, every fresh clone
    and the public mirror look like: `5 passed, 4 skipped`. The
    skips were `test_no_mirrored_file_names_a_HELD_OUT_paper_by_DOI`,
    `test_the_corpus_KEY_family_is_covered`,
    `test_the_corpus_aware_DOI_gate_is_not_vacuous` — a test whose own name claims
    non-vacuity and which then skips — and
    `test_the_verbatim_matcher_catches_what_extraction_drops`. This test was among
    the PASSES. With the corpus present, the same worktree gives `9 passed, 0
    skipped` off exactly 72 identifiers (36 DOIs + 36 corpus keys) from 36 papers.

    HOW THIS MODULE DIFFERS FROM ITS SIBLING, and it matters: a corpus-less run
    here is NOT wholly vacuous. Two live gates genuinely ran in that measurement —
    `test_no_mirrored_file_names_a_HELD_OUT_paper`, which needs only the `_PMC`
    regex, and `test_no_mirrored_file_names_a_DOI_outside_the_baseline`, which
    compares 540 mirrored files against the 14-file / 48-DOI `DOI_BASELINE`
    inventory. So the defect fixed here is the misleading NAME plus an unasserted
    second input, not the total false green the sibling had. The name now states
    what a runner can actually certify.

    THE SECOND HALF CANNOT BE MADE TO GO RED ON A RUNNER, and that limit is
    deliberate rather than unfinished work. The corpora are gitignored and must
    NEVER exist on CI; closing the skip would take a committed digest of ~30 PMC
    accessions, which is brute-forceable in seconds. So it reports its own absence
    as a SKIP. That is the whole point of the change — a pass and a skip are
    different verdicts, and only one of them is a claim.

    The file-list half is asserted BEFORE the skip deliberately, so a runner keeps
    that half enforced rather than losing both. That half is the guard against the
    exact mistake that produced the wrong "0" (see the module docstring): a green
    from an empty file list is indistinguishable from a green from a clean mirror.

    BOTH identifier families are checked, not their union. `_held_out_dois()` and
    `_held_out_keys()` are consumed separately downstream — the KEY gate reads
    only the second, the two DOI controls read only the first — so a union that
    was non-empty because one family survived would certify a state in which the
    other family's gate had silently skipped.
    """
    files = _mirrored_files()
    assert len(files) > 400, (
        f"the manifest resolved to only {len(files)} readable files, so this "
        "scan checked almost nothing and its verdict is void. The known cause is "
        "failing to strip the CRLF line terminator before the existence check."
    )

    dois, keys = _held_out_dois(), _held_out_keys()
    if not (dois and keys):
        pytest.skip(
            f"HALF-CERTIFIED ONLY. The mirrored file list is sound ({len(files)} "
            "readable paths), and the two corpus-free gates below did run against "
            f"it. But the held-out identifier set is incomplete — {len(dois)} "
            f"DOI(s) and {len(keys)} corpus key(s) were read — so the four "
            "corpus-aware tests in this module matched nothing and skipped, and "
            "NOTHING here certifies that a mirrored file naming a held-out paper "
            "by DOI or by corpus slug would be seen. This is the expected state "
            "on a CI runner, a fresh clone and the public mirror: the corpora are "
            "gitignored and must never exist there, and closing this would need a "
            "committed digest of the accessions, which is brute-forceable. It is "
            "reported as a SKIP rather than a PASS so no summary line can be read "
            "as 'the scan was not vacuous'. Only a run on a machine that holds "
            "both corpora certifies both halves. COUNTS ONLY above, never the "
            "identifiers — this module is itself mirrored."
        )


def test_the_matcher_separates_real_ids_from_placeholders():
    """Two-sided control on the matcher itself, with no file I/O.

    Without this, a regex that matched nothing would make every verdict below a
    silent pass.
    """
    # COMPOSED AT RUNTIME so the literal never appears in this file's source.
    # This module is itself in the manifest, so the live gate below scans it —
    # and a real-shaped accession written out here would be flagged, by this
    # module, in this module. That is not a flaw: it is the gate proving it
    # covers everything the mirror ships, including its own text. Caught exactly
    # this way on 2026-09-02, one run after the manifest was regenerated.
    shaped = "PMC" + "1234567"
    assert _real_ids(f"see {shaped} for details") == {shaped}, (
        "the matcher failed to find a real-shaped accession — every 'clean' "
        "result from it is worthless"
    )
    assert _real_ids("placeholder PMC00000000 only") == set(), (
        "the all-zero placeholder was treated as a real accession; fixtures "
        "would be unable to express an example path"
    )
    assert _real_ids("fixture PMC0000001 in a redaction test") == set(), (
        "a leading-zero synthetic accession was treated as real — this is the "
        "convention test_holdout_breakdown_allowlist.py already uses"
    )
    assert _real_ids("no identifiers here at all") == set()


@pytest.mark.skipif(not MANIFEST.is_file(),
                    reason="contract/public_manifest.txt absent — not a repo checkout")
def test_no_mirrored_file_names_a_HELD_OUT_paper():
    """THE LIVE GATE. Any mirrored file not on the debt list must be clean."""
    new = {path: ids for path, ids in _offenders().items()
           if path not in KNOWN_EXPOSED}
    assert not new, (
        "a file the PUBLIC mirror ships names a held-out paper by accession:\n  "
        + "\n  ".join(f"{p} ({len(ids)} id(s))" for p, ids in sorted(new.items()))
        + "\n\nThat publishes which papers are in the held-out split. Membership "
        "is not gold, but it tells a reader what to tune on, and any score "
        "printed beside it tells them how they are doing.\n\n"
        "FIX: use the all-zero placeholder PMC00000000 for examples and fixtures "
        "— nothing here resolves an accession, so a fake one tests identically. "
        "If the identifier is genuinely load-bearing, add a KNOWN_EXPOSED entry "
        "recording the FILE and the COUNT, never the identifiers themselves."
    )


def test_the_DOI_matcher_separates_real_ids_from_placeholders():
    """Two-sided control on the DOI matcher, with no file I/O.

    Same reasoning as the PMC control above: a regex that matched nothing would
    make every DOI verdict below a silent pass. The real-shaped literal is
    COMPOSED AT RUNTIME so it never appears in this file's source — this module
    is in the manifest, so the live gates below scan its own text.
    """
    shaped = "10." + "1234/example.2026.99"
    assert _dois(f"see {shaped} for details") == {shaped}, (
        "the DOI matcher failed to find a real-shaped DOI — every 'clean' "
        "result from it is worthless"
    )
    assert _dois("trailing punctuation " + shaped + ".") == {shaped}, (
        "a sentence-final period was treated as part of the identifier, so a "
        "real DOI at the end of a sentence would never compare equal"
    )
    assert _dois("placeholder 10.0000/example.2025.1 only") == set(), (
        "the unassigned 10.0000 registrant was treated as a real DOI; fixtures "
        "and docstrings would be unable to show an example record"
    )
    assert _dois("no identifiers here at all") == set()


@pytest.mark.skipif(not MANIFEST.is_file(),
                    reason="contract/public_manifest.txt absent — not a repo checkout")
def test_no_mirrored_file_names_a_DOI_outside_the_baseline():
    """THE SELF-CONTAINED DOI GATE. Works in the public checkout.

    Enforced by EQUALITY in both directions: a file that gains a DOI is red, and
    a stale baseline entry for a file that lost one is red too, so the inventory
    cannot quietly drift into a blanket allowance.
    """
    actual = _doi_counts()
    added = {p: n for p, n in actual.items() if p not in DOI_BASELINE}
    grew = {p: (DOI_BASELINE[p], n) for p, n in actual.items()
            if p in DOI_BASELINE and n > DOI_BASELINE[p]}
    shrank = {p: (DOI_BASELINE[p], actual.get(p, 0)) for p in DOI_BASELINE
              if actual.get(p, 0) < DOI_BASELINE[p]}
    assert not (added or grew or shrank), (
        "the mirror's DOI inventory moved:\n"
        + "".join(f"  NEW FILE   {p} names {n} DOI(s)\n"
                 for p, n in sorted(added.items()))
        + "".join(f"  GREW       {p}: {was} -> {now}\n"
                 for p, (was, now) in sorted(grew.items()))
        + "".join(f"  SHRANK     {p}: {was} -> {now}\n"
                 for p, (was, now) in sorted(shrank.items()))
        + "\nA DOI naming a HELD-OUT paper publishes its membership in the "
        "split, which tells a reader what to tune on.\n\n"
        "FIX: read what changed. If the DOI is a held-out paper, remove it and "
        "use the unassigned placeholder prefix 10.0000/ instead — nothing here "
        "resolves a DOI, so a fake one tests identically. If it is ordinary "
        "published work being cited, update the count in DOI_BASELINE in the "
        "same commit. Never paste the identifier into this file."
    )


@pytest.mark.skipif(not MANIFEST.is_file(),
                    reason="contract/public_manifest.txt absent — not a repo checkout")
def test_no_mirrored_file_names_a_HELD_OUT_paper_by_DOI():
    """THE CORPUS-AWARE DOI GATE. Private checkout only, and exact.

    The baseline above cannot see a SUBSTITUTION — swap a cited DOI for a
    held-out one and the count is unchanged. Here the real set is known, so that
    case is caught. This is the half that runs pre-push, which is where a leak is
    actually prevented: the mirror is built FROM this tree.

    Failure reports FILE and COUNT only. Printing the identifier would write the
    leak into the test output, and from there into a log.
    """
    held = _held_out_dois() | _held_out_keys()
    if not held:
        pytest.skip("held-out corpora absent — this is a public checkout")

    # TWO FLOORS, both from cross-model review (Grok 4.6, 2026-09-08). Without
    # them, deleting one whole corpus leaves this test RUNNING against the other
    # corpus's identifiers and reporting a clean mirror for papers it no longer
    # knows about — a partial identifier set is a silent partial gate.
    contributions = _corpus_contributions()
    assert len(held) >= 30, (
        f"only {len(held)} held-out identifiers were read. A short set means a "
        "corpus is partly missing, and this verdict covers only what remains."
    )
    empty = [rel for rel, n in contributions.items() if n == 0]
    assert not empty and len(contributions) >= 2, (
        f"held-out corpora present: {len(contributions)}, of which "
        f"{len(empty)} contributed nothing. An entire corpus has gone missing "
        "or been emptied; its papers are no longer covered by this gate."
    )

    offenders = {}
    for p in _mirrored_files():
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # BOTH matchers, and the union. `_dois` gives an exact count for the
        # message; `_names_verbatim` catches the concatenated form that `_dois`
        # provably drops. Either one firing is a leak.
        hits = _dois(text) & held
        if hits or _names_verbatim(text, held):
            offenders[p.relative_to(REPO).as_posix()] = max(len(hits), 1)
    assert not offenders, (
        "a file the PUBLIC mirror ships names a HELD-OUT paper by DOI:\n  "
        + "\n  ".join(f"{p} ({n} held-out DOI(s))"
                      for p, n in sorted(offenders.items()))
        + "\n\nThe identifiers are deliberately not printed. Open the file and "
        "replace each one with the unassigned placeholder prefix 10.0000/."
    )


def test_the_corpus_KEY_family_is_covered():
    """A held-out paper named by its corpus SLUG must be seen.

    The regression control for the third identifier family (Grok 4.6,
    2026-09-08). Before it, `_PMC` covered accessions and `_DOI` covered DOIs,
    and an `<author>_<year>_<journal>` slug matched neither — a mirrored file
    could name an APA-corpus paper and this module stayed green.

    Both literals are read off disk and never printed.
    """
    keys = _held_out_keys()
    if not keys:
        pytest.skip("held-out corpora absent — this is a public checkout")
    slugs = sorted(k for k in keys if not k.lower().startswith("pmc"))
    assert slugs, (
        "no non-accession corpus key was found, so this control is asserting "
        "nothing. The APA corpus keys its papers by slug; if that changed, this "
        "test must change with it rather than silently pass."
    )
    held = _held_out_dois() | keys
    planted = slugs[0]
    assert _names_verbatim(f"as described in {planted} we find", held), (
        "a real held-out corpus slug planted in prose was NOT recognised"
    )
    assert not _names_verbatim("see author_1999_none here", held), (
        "an unrelated slug matched — the matcher is a yes-machine"
    )


def test_the_corpus_aware_DOI_gate_is_not_vacuous():
    """A held-out DOI planted in a scanned file must be SEEN.

    Without this, an empty `held` set, a normalisation mismatch between
    `source.json` and `_dois`, or a regex that stopped matching would all
    present as the same clean green. Skipped, not passed, where the corpus is
    absent — a check that cannot run must never report success.
    """
    held = _held_out_dois()
    if not held:
        pytest.skip("held-out corpora absent — this is a public checkout")
    planted = sorted(held)[0]
    assert _dois(f"a mirrored file mentioning {planted} in prose") & held, (
        "a real held-out DOI planted in text was NOT recognised, so the "
        "corpus-aware gate above is scanning for something it can never find"
    )
    assert not (_dois("unrelated 10.0000/example.2025.1 placeholder") & held), (
        "a placeholder matched the held-out set — the gate is a yes-machine"
    )


def test_the_verbatim_matcher_catches_what_extraction_drops():
    """Two-sided control on `_names_verbatim`, against the real held-out set.

    This is the control that would have caught the gap it exists to close: the
    CONCATENATED form, measured in a tracked file on 2026-09-08, which `_dois`
    provably fails to match. The planted string is built from a real held-out
    DOI read off disk and is never printed.
    """
    held = _held_out_dois()
    if not held:
        pytest.skip("held-out corpora absent — this is a public checkout")
    planted = sorted(held)[0]

    assert not (_dois(planted + "Personality") & held), (
        "extraction unexpectedly matched the concatenated form. If this ever "
        "passes, `_dois` changed and the duplication below may no longer be "
        "needed — but verify before removing it."
    )
    assert _names_verbatim(planted + "Personality", held), (
        "THE CONCATENATION CASE. The verbatim matcher failed to see a held-out "
        "DOI glued to a following word, which is the one thing it was added for."
    )
    assert _names_verbatim("https://doi.org/" + planted, held), (
        "a resolver URL suppressed the match — the before-rule is too strict"
    )
    assert not _names_verbatim("placeholder 10.0000/example.2025.1 only", held), (
        "a placeholder matched the held-out set — the matcher is a yes-machine"
    )
    assert not _names_verbatim(planted + "7", held), (
        "one more DIGIT makes a different identifier and must not match"
    )
