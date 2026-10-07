"""No TRACKED file may NAME a held-out paper, outside the answer key itself.

WHY THIS FILE EXISTS. Naming which papers are held out IS the answer key. A
player who knows the held-out DOIs can fetch and pre-process exactly those
papers before the run — the split's entire value is that nobody can do that.
Gold, output, `input_hash` and breakdown are already stripped by
`framework/holdout.py`; `framework/publish.py` keeps everything outside
`PUBLIC_TOP_LEVEL` private by default; and
`test_public_mirror_names_no_heldout_paper.py` guards the files the mirror
actually ships. None of them watch the channel this module watches: an
identifier written into a tracked file that is private TODAY.

THE INCIDENT, 2026-09-07. `framework/article_custody.py` — which IS mirrored —
carried a docstring saying "this module is mirrored publicly, so it names no
real held-out paper" and then used, as its worked example, a DOI that is one of
`pdf-text-fidelity-v1`'s held-out papers. Writing that finding up, the same
session quoted the DOI into `TODO.md`. It caught itself. Nothing in the repo
would have. Full write-up: LESSONS.md, 2026-09-07.

WHY "PRIVATE TODAY" IS NOT SAFE. `framework/publish.py`'s `public_manifest()`
enumerates the FILESYSTEM against `PUBLIC_TOP_LEVEL` globs. A file that is
private today becomes mirrored the moment it matches a glob — nobody edits it,
nobody reviews it, and the first anyone hears of it is a publish.

WHAT IT FOUND ON THE TREE THAT CREATED IT, 2026-09-08: 61 lines across 15
tracked files, plus 3 more `README.md` files found only after the cross-model
review below. Fifteen were remediated in the same change with document-local
pseudonyms. The other three could NOT be, and are recorded in `_ARCHIVED_DEBT`
below — they live under `runs/**/_archive/`, where `scripts/verify_records.py`
blocks any edit at pre-commit because an archive that moves is not an archive.
Two legitimate rules point opposite ways there; the debt records the conflict
instead of resolving it by quietly weakening one of them. None of the eighteen
had ever been in the mirror manifest, which was checked rather than assumed.

THE SCOPING TRAP, STATED SO THE NEXT READER DOES NOT "FIX" IT. This scan is
deliberately scoped to `git ls-files`, so it reports on GIT, not on the working
tree. Elsewhere in this portfolio that scoping is a MEASURED DEFECT — the
article-custody scan's `--tracked-only` and `--history` forms both reported
0 leaks against a planted article render in a gitignored `tmp/`, and only the
flagless full-tree form saw it. Here the narrow scope is CORRECT and
load-bearing, because the question this module asks is precisely "is this
identifier committed?" A gitignored scratch file naming a held-out paper cannot
be pushed, cannot be mirrored, and cannot be read by anyone else. Widening this
to the whole working tree would flag every scratch file and get the gate
switched off within a week. The held-out corpora themselves are gitignored,
which is exactly why the identifier SET is built from the working tree while the
SCAN is built from the index.

WHAT A CROSS-MODEL REVIEW FOUND IN THE FIRST DRAFT (Sonnet 5 + Grok 4.6,
2026-09-08; Sol was out of quota and is recorded as a refusal). Fourteen
findings, every one of them a FALSE GREEN — the gate reporting clean while a
tracked file really did name a paper. The ones that were reproduced and fixed,
each now carrying a control:

  * the identifier set was keyed off `source.json`, so a held-out paper
    directory that had not been given one yet contributed NOTHING (Grok);
  * the floor counted TOKENS, not papers — deleting the entire 6-paper APA
    corpus still left ~60 PMC tokens and the floor passed (Grok). This is the
    single most dangerous one, because deleting files a gate reads is a silent
    change to the gate;
  * `git ls-files` failing returned `[]`, so the live gate asserted `not {}`
    and PASSED while the vacuity check merely SKIPPED (Grok);
  * the answer-key exclusion was a PATH prefix, so `README.md` files under
    `arenas/*/runs/` were invisible — and three of them really did name papers
    (Sonnet, reproduced immediately);
  * identifiers only matched as one contiguous run, so a line-wrapped DOI, a
    `%2F`-encoded one, a soft hyphen or zero-width space pasted in from a PDF,
    or `PMC-12345678` all read clean (both);
  * an unreadable file was skipped silently, and a staged blob was never read
    at all — a leak could be staged while the working tree looked clean (Grok).

The one finding NOT fixed, because it cannot be, is recorded in
`test_the_identifier_set_is_present_or_this_is_a_corpusless_checkout`.

WHY THIS MODULE NAMES NO IDENTIFIER, ANYWHERE. `framework/**/*.py` is in
`PUBLIC_TOP_LEVEL`, so this file is mirrored. Every literal here is synthetic,
every control composes its example at runtime, and no failure message prints a
matched identifier — it reports FILE, LINE and COUNT. Printing the identifier
would write the leak into the test output and from there into a CI log.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Held-out corpora live under each task set. They are GITIGNORED — the papers
#: are in article-finder's custody — so the identifier set is read from the
#: WORKING TREE even though the scan reads what git tracks.
_HELD_OUT_ROOTS = "arenas/*/task_sets/*/_held_out"

#: A held-out paper's DIRECTORY NAME is an identifier whatever its shape. The
#: PMC-keyed corpus names each directory by accession; the APA-keyed corpus uses
#: an author-year-journal slug, which resolves to the paper in one search just as
#: readably as a DOI. Covering only accessions would have left the most
#: human-readable form of the disclosure unguarded.
#:
#: The shape guard stops that generality becoming a noise generator: a corpus
#: directory called `paper1`, or the `code-translation-r-v1` corpus's TASK names
#: (`paired-ttest-repeated` — a task, not a paper), must not enter the set.
#: Requiring length AND a digit admits every real paper key and refuses those.
_MIN_NAME_LEN = 8


def _is_identifying(name: str) -> bool:
    """Is this corpus directory name specific enough to be an identifier?"""
    return len(name) >= _MIN_NAME_LEN and any(c.isdigit() for c in name)


#: Paths where a held-out identifier is the record's PRIMARY KEY rather than
#: prose: ground truth, seed lists, and run records keyed by a task id built
#: from the accession. Scrubbing these would destroy the benchmark — this repo
#: is private precisely so it can hold its own answer key.
#:
#: THE EXTENSION HALF IS LOAD-BEARING AND WAS ADDED AFTER REVIEW. The first
#: draft excluded the whole path PREFIX, which silently exempted every
#: `README.md` written beside a run — and three of them named papers, in prose,
#: which is exactly the case this gate exists for. Only machine-readable record
#: formats are exempt; anything a human writes is scanned wherever it lives.
_ANSWER_KEY_PREFIX = re.compile(r"^arenas/[^/]+/(runs|task_sets)/")
_ANSWER_KEY_SUFFIX = {".json", ".jsonl", ".txt"}


def _is_answer_key(rel: str) -> bool:
    return bool(_ANSWER_KEY_PREFIX.match(rel)) and Path(rel).suffix.lower() in _ANSWER_KEY_SUFFIX


#: Files that DO name a held-out paper and CANNOT be scrubbed in place, with the
#: count and the reason. This is a DEBT, not a dispensation — the idiom is taken
#: from `KNOWN_EXPOSED` in `test_public_mirror_names_no_heldout_paper.py`.
#:
#: WHY THESE THREE CANNOT BE FIXED THE OBVIOUS WAY. They are `README.md` files
#: sitting under `runs/**/_archive/`, and `scripts/verify_records.py` blocks, at
#: pre-commit, ANY modify/delete/rename under `_archive/`. That guard is a flat
#: rule on purpose — "an archive that moves when the scorer moves is not an
#: archive" — and it is protecting published evidence about a superseded run.
#: Scrubbing the prose would rewrite what a reviewer was told the run showed.
#:
#: So two legitimate rules point opposite ways here, and this records the
#: conflict rather than quietly resolving it by weakening one of them. Neither
#: `--no-verify` nor a looser archive guard is an acceptable way out.
#:
#: WHAT BOUNDS THE RISK, measured 2026-09-08 rather than assumed: none of the
#: three is in `contract/public_manifest.txt`, so none is published, and
#: `test_no_debt_file_is_mirrored` below asserts that stays true. If one ever
#: becomes mirrored, the debt is no longer tolerable and that test goes red.
#:
#: The identifiers are deliberately NOT written here — this module is mirrored.
#: FILE and COUNT only, as the mirror gate's note requires.
_ARCHIVED_DEBT: dict[str, int] = {
    "arenas/pdf-section-structure-v1/runs/v1/_archive/"
    "2026-08-28-docpluck-2.4.136-superseded-by-2.4.137/README.md": 3,
    "arenas/pdf-table-extraction-v1/runs/v1/_archive/"
    "2026-08-28-docpluck-2.4.136-superseded-by-2.4.137/README.md": 3,
    "arenas/pdf-text-fidelity-v1/runs/v2/_archive/"
    "2026-08-28-docpluck-2.4.136-superseded-by-2.4.137/README.md": 3,
}

#: Genuinely OPAQUE formats: images, fonts, archives, bytecode. Skipping these
#: is safe — they carry no prose a reader could lift an identifier from.
_OPAQUE = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".svgz",
           ".woff", ".woff2", ".ttf", ".otf", ".zip", ".gz", ".pyc", ".mp4"}

#: Formats that DO carry readable text but that this scanner cannot decode.
#: Reviewed defect: a tracked spreadsheet with an accession column would never
#: be scanned. Measured 2026-09-08 the repo tracks NONE of these, so the honest
#: fix is to assert that stays true rather than to ship a document parser —
#: see `test_no_tracked_file_is_skipped_as_binary`. If it goes red, read the
#: file; do not add its extension to `_OPAQUE`.
_TEXT_BEARING_BINARY = {".pdf", ".xlsx", ".xls", ".docx", ".doc", ".pptx", ".odt"}

_BINARY = _OPAQUE | _TEXT_BEARING_BINARY

#: Characters that carry no meaning but break a contiguous match: zero-width
#: space/non-joiner/joiner, BOM, and the soft hyphen a PDF paste leaves behind.
_INVISIBLE = "​‌‍﻿­"


def _normalise(text: str) -> str:
    """Undo the encodings and line-breaks that hide an identifier in plain text.

    Each substitution here answers a CONCRETE way a real identifier survives in
    a document while defeating a naive substring search: percent- and
    entity-encoding from a copied URL, an invisible character from a PDF paste,
    and — the commonest — a word wrap that puts a line break inside the token.

    Whitespace is removed ENTIRELY rather than collapsed. Identifiers never
    contain whitespace, so joining can only ever REVEAL a split one; the risk of
    two unrelated adjacent tokens concatenating into a 20-character DOI is not a
    real risk. This is the transform that catches a DOI wrapped across lines.
    """
    low = text.lower()
    for ch in _INVISIBLE:
        low = low.replace(ch, "")
    low = low.replace("%2f", "/").replace("&#47;", "/").replace("&sol;", "/")
    low = low.replace("%2e", ".")
    return re.sub(r"\s+", "", low)


# Every character a real document uses where a hyphen was typed. Matching only
# the ASCII "-" makes this gate report clean on exactly the files it is about:
# word processors, LaTeX and PDF text layers substitute these silently, and the
# subject matter here IS documents that have been through those pipelines.
# Removed from BOTH text and identifier, so folding can only ever REVEAL a
# disguised identifier -- it cannot make two different ones collide, which the
# negative control in the dash test pins.
_DASHES = "-‐‑‒–—―−"


def _dehyphen(text: str) -> str:
    """Normalised, with hyphens and underscores removed as well.

    Applied to the TEXT and to the IDENTIFIER together, so the comparison stays
    honest: it catches `PMC-12345678` and an APA slug written with dashes
    instead of underscores, without letting a DOI that legitimately contains a
    hyphen match something it should not.
    """
    out = _normalise(text)
    for ch in _DASHES:
        out = out.replace(ch, "")
    return out.replace("_", "")


def _held_out_papers() -> list[Path]:
    """Every held-out PAPER directory on disk.

    A directory counts as a paper if it carries a `source.json` OR its name
    passes the shape guard. The `OR` is the fix for a reviewed defect: keying
    this off `source.json` alone meant a paper directory that had not been given
    one yet — the corpus growth path — contributed no identifier at all, and a
    tracked file naming it read CLEAN.
    """
    papers: list[Path] = []
    for root in REPO.glob(_HELD_OUT_ROOTS):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            if (path / "source.json").is_file() or _is_identifying(path.name):
                papers.append(path)
    return papers


def _held_out_identifiers() -> set[str]:
    """Every held-out paper's DOI and directory name, lower-cased.

    A trailing period is stripped because sentence punctuation is not part of an
    identifier, and a normalisation mismatch here would silently make every
    comparison fail to match.
    """
    ids: set[str] = set()
    for paper in _held_out_papers():
        record = paper / "source.json"
        if record.is_file():
            try:
                doi = json.loads(record.read_text(encoding="utf-8")).get("doi")
            except (OSError, ValueError):
                doi = None
            if doi:
                ids.add(doi.lower().rstrip("."))
        if _is_identifying(paper.name):
            ids.add(paper.name.lower())
    return ids


def _find(text: str, ids: set[str]) -> bool:
    """True if `text` names any identifier in `ids`, in any of its real forms.

    SUBSTRING, not extract-then-intersect, and the difference is not academic.
    The tree that created this module had a live case where a real DOI was
    concatenated with the following word by a PDF text-layer defect
    (`<doi>Personality`). A DOI regex greedily consumes that suffix, so
    extracting and intersecting produced NO match on a line that plainly names
    the paper. A reader — or a scraper — has no such trouble.

    THREE PASSES, because one contiguous-substring pass was shown by review to
    miss every encoded and line-wrapped form: the raw text, the normalised text
    (invisibles and whitespace removed, percent/entity decoding applied), and
    the de-hyphenated text with the identifiers de-hyphenated to match.

    Two boundary rules keep it precise:

    * the character BEFORE must not be ALPHANUMERIC, so an identifier is not
      "found" glued inside some longer unrelated token;
    * the character AFTER must not be a DIGIT, because one more digit makes a
      DIFFERENT accession or DOI. A following LETTER is deliberately still a
      hit — that is the concatenation case above.

    THE BEFORE-RULE TESTS `isalnum()` AND NOTHING ELSE. The first draft also
    rejected a preceding `.` or `/`, reasoning that those "continue an
    identifier". They do — but the commonest way anyone writes a DOI is
    `https://doi.org/<doi>`, where the preceding character is exactly `/`. That
    draft passed every control written for it and would have ignored the single
    most likely form of the leak.

    THE BEFORE-RULE IS RELAXED ON THE NORMALISED PASSES, and this was found by
    the control below failing rather than by reasoning. Removing whitespace to
    reunite a line-wrapped DOI also glues the PRECEDING word onto it, so the
    character before the match becomes a letter and the strict rule rejects the
    very match the pass exists to make. On those passes the rule is therefore
    "not a DIGIT" — still enough to refuse a match inside a longer number, which
    is the case the rule was for, without refusing every wrapped identifier.
    """
    for haystack, needles, strict in (
        (text.lower(), ids, True),
        (_normalise(text), {_normalise(i) for i in ids}, False),
        (_dehyphen(text), {_dehyphen(i) for i in ids}, False),
    ):
        for ident in needles:
            if not ident:
                continue
            start = 0
            while True:
                i = haystack.find(ident, start)
                if i < 0:
                    break
                before = haystack[i - 1] if i > 0 else ""
                after = haystack[i + len(ident)] if i + len(ident) < len(haystack) else ""
                blocked = before.isalnum() if strict else before.isdigit()
                if not blocked and not after.isdigit():
                    return True
                start = i + 1
    return False


def _git_lines(*args: str) -> list[str]:
    """NUL-separated `git` output as a list. Empty list on any failure.

    An empty list from a git failure is indistinguishable from a genuinely
    empty result, which is why every caller that could be made VACUOUS by it
    asserts its own input size rather than trusting this.
    """
    proc = subprocess.run(
        ["git", "-C", str(REPO), *args, "-z"], capture_output=True
    )
    if proc.returncode != 0:
        return []
    return [p for p in proc.stdout.decode("utf-8", "ignore").split("\0") if p]


def _tracked() -> list[str]:
    """Every tracked path, POSIX-separated. Empty if this is not a checkout."""
    return _git_lines("ls-files")


def _staged_blobs() -> dict[str, str]:
    """path -> STAGED content, for paths whose index differs from HEAD.

    Closes a reviewed gap: the scan reads working-tree bytes, so a leak that is
    STAGED while the working tree has been cleaned would commit unseen. Only
    changed paths are read, so this stays cheap.
    """
    proc = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--cached", "--name-only", "-z"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return {}
    out: dict[str, str] = {}
    for rel in (p for p in proc.stdout.decode("utf-8", "ignore").split("\0") if p):
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", f":{rel}"], capture_output=True
        )
        if blob.returncode == 0:
            out[rel] = blob.stdout.decode("utf-8", "ignore")
    return out


def _scan() -> tuple[dict[str, list[int]], int]:
    """(path -> offending 1-indexed lines, count of files that could not be read).

    A line list may legitimately be EMPTY when the identifier is split across
    lines — the whole-text pass sees it and no single line does. That is
    reported as a hit with no line number rather than dropped.

    NOTHING TRACKED IS EXEMPT, and one exemption used to be. Until 2026-09-11
    every path in `git ls-files --deleted` was skipped here, commented as "a
    deletion in flight ... on their way OUT of the tracked set". MEASURED that
    day, that premise is false in both directions:

      * a deletion that HAS been staged never reaches this loop at all.
        `git ls-files` stops listing a path the moment its deletion enters the
        index — measured 1 before staging, 0 after — so a removal genuinely in
        flight is already out of scope and never needed an exemption;
      * what the exemption therefore covered was only the UNSTAGED case: a
        deletion nobody recorded. That is not a removal in flight, it is an
        ABANDONED one. The path is still tracked, still in HEAD, and still
        names whatever it named.

    Four files sat in exactly that state in September 2026, owned by nobody.
    Two of them named held-out papers on 8 lines, and this gate reported GREEN
    for as long as they sat there; the same tree in a clean worktree, with the
    corpus replicated, goes RED on those lines. The exemption was the whole
    difference between the two verdicts.

    So a tracked path whose working-tree file is gone now falls through to the
    HEAD-blob read below. The committed content is what is still tracked, and
    it is what gets checked. Somebody mid-removal is not blocked — they stage
    the deletion, which is what recording it means.
    """
    ids = _held_out_identifiers()
    if not ids:
        return {}, 0
    found: dict[str, list[int]] = {}
    unreadable = 0
    staged = _staged_blobs()
    for rel in _tracked():
        if _is_answer_key(rel) or Path(rel).suffix.lower() in _BINARY:
            continue
        try:
            text = (REPO / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            # Tracked but the working tree cannot produce the bytes — a DELETED
            # file (see the docstring), a sparse checkout, a permission problem,
            # or a path the filesystem cannot open. The COMMITTED content is
            # still in scope, so fall back to the HEAD blob rather than counting
            # it. That keeps `unreadable` a real error instead of a routine
            # number everyone learns to ignore, which is the only way the
            # assertion on it stays worth making.
            blob = subprocess.run(
                ["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True
            )
            if blob.returncode == 0:
                text = blob.stdout.decode("utf-8", "ignore")
            elif rel in staged:
                # Never committed, and no working-tree file: a path ADDED to the
                # index and then removed from disk without staging the removal.
                # There is no HEAD blob to read and nothing was missed — the
                # staged blob below is the whole of this path's content, so it
                # IS fully checked. Counting it unreadable would redden the gate
                # for a file it actually read.
                text = ""
            else:
                unreadable += 1
                text = ""
        for source in (text, staged.get(rel, "")):
            if source and _find(source, ids):
                found.setdefault(rel, [])
                found[rel] = sorted(set(found[rel]) | {
                    n for n, line in enumerate(source.splitlines(), 1)
                    if _find(line, ids)
                })
    return found, unreadable


# ---------------------------------------------------------------------------
# Guards on the guard. Every one of these exists because the live gate below
# reports the same clean green whether it examined everything or nothing.
# ---------------------------------------------------------------------------

def test_the_identifier_set_is_present_or_this_is_a_corpusless_checkout():
    """THE MOST IMPORTANT ASSERTION IN THIS FILE.

    An empty identifier set makes `_find` return False for every line of every
    file, and the live gate then reports a clean tree it never examined.

    THE LIMIT THIS CANNOT FIX, stated plainly because both reviewers raised it
    and it is real. The corpora are gitignored and must never exist on a runner,
    so on CI this test SKIPS and the live gate skips with it: a pull request
    that writes a held-out DOI into `TODO.md` goes green on CI. This gate is
    therefore an OPERATOR-MACHINE gate, and it is effective there because
    `/ship` is the portfolio's only sanctioned route to a push and runs the full
    suite on the machine that does hold the corpus. Do not read a green CI as
    this gate having run. Closing the CI half would need a committed,
    non-reversible digest of the identifiers, and a digest of 30 PMC accessions
    is brute-forceable in seconds — which is why it has not been done.
    """
    ids = _held_out_identifiers()
    if not ids:
        pytest.skip(
            "no held-out corpus on disk — the papers are gitignored and live in "
            "article-finder's custody, so this is a fresh clone, a CI runner, or "
            "the public mirror. The live gate skips too. A green run here is NOT "
            "evidence that any tracked file was checked."
        )
    assert len(ids) >= 30, (
        f"only {len(ids)} held-out identifiers were read. A short set means the "
        "corpus is partly missing or the record shape changed, and every "
        "'clean' verdict below is void."
    )


def test_no_held_out_corpus_has_been_partly_emptied():
    """Deleting files a gate reads is a silent change to the gate.

    Reviewed defect, and the most dangerous one found: the floor above counts
    TOKENS. Delete the entire 6-paper APA corpus and ~60 PMC tokens remain, so
    the floor still passes while six papers silently stop being protected.

    Two independent floors close it. PAPERS, not tokens — and at least two
    distinct corpus ROOTS must still yield papers, which is what a whole-corpus
    deletion breaks. Plus: any root holding a `source.json` at all must yield at
    least one paper, which catches a partial emptying.
    """
    papers = _held_out_papers()
    if not papers:
        pytest.skip("no held-out corpus on disk — see the test above")

    assert len(papers) >= 30, (
        f"only {len(papers)} held-out PAPERS are on disk. A count that dropped "
        "means a corpus was partly deleted, and the papers that went are no "
        "longer protected by anything."
    )

    roots_with_papers = {
        r for r in REPO.glob(_HELD_OUT_ROOTS)
        if any(str(p).startswith(str(r)) for p in papers)
    }
    assert len(roots_with_papers) >= 2, (
        f"only {len(roots_with_papers)} held-out corpus root(s) yielded any "
        "paper. An entire corpus has gone missing; the token floor above cannot "
        "see that, which is exactly why this test exists."
    )

    for root in REPO.glob(_HELD_OUT_ROOTS):
        if not any(root.rglob("source.json")):
            continue
        assert any(str(p).startswith(str(root)) for p in papers), (
            f"{root.relative_to(REPO).as_posix()} holds source.json records but "
            "yielded no identifying paper — the corpus shape changed and its "
            "papers are unprotected."
        )


def test_every_real_corpus_key_passes_the_shape_guard():
    """No real held-out paper may be silently dropped by the shape guard.

    A key that fails `_is_identifying` and has no `source.json` contributes
    nothing. That loss is invisible — the suite goes green either way.
    """
    papers = _held_out_papers()
    if not papers:
        pytest.skip("no held-out corpus on disk")
    dropped = [p for p in papers
               if not _is_identifying(p.name) and not (p / "source.json").is_file()]
    assert not dropped, (
        f"{len(dropped)} held-out paper director(ies) contribute NO identifier: "
        "their names fail the shape guard and they carry no source.json, so no "
        "tracked file naming them would ever be flagged. The names are not "
        "printed — this module is mirrored."
    )


def test_the_matcher_finds_and_bounds_identifiers():
    """Two-sided control on `_find`. No file I/O; every literal is synthetic.

    Without this, a matcher that matched NOTHING would make the live gate a
    permanent silent pass. Literals are composed at runtime so no
    identifier-shaped string appears in this file's source — this module is
    mirrored, and the mirror's own name gate scans it.
    """
    doi = "10." + "1234/example.2026.99"
    pmc = "PMC" + "1234567"
    slug = "author" + "_2099_jrnl"
    ids = {doi, pmc.lower(), slug}

    # --- must FIND: every form these identifiers really take in a document ---
    assert _find("as reported in " + doi + " we see", ids), "plain DOI missed"
    assert _find("the " + pmc + " record", ids), "plain accession missed"
    assert _find("the " + slug + " paper", ids), (
        "an author-year-journal slug was missed — the APA corpus keys its "
        "papers by exactly this shape"
    )
    assert _find("see " + doi.upper() + " here", ids), "matching must be case-insensitive"
    assert _find("trailing punctuation " + doi + ".", ids), "sentence period suppressed it"
    assert _find("https://doi.org/" + doi, ids), (
        "THE URL CASE, and why the before-rule tests isalnum() and nothing "
        "else. A resolver URL puts `/` immediately before the DOI."
    )
    assert _find("concatenated " + doi + "Personality", ids), (
        "THE CONCATENATION CASE, live in a real tracked file: a text-layer "
        "defect glues the next word on. Extract-then-intersect drops it."
    )
    # Reviewed defects: forms that defeated the contiguous-substring draft.
    assert _find("wrapped " + doi[:12] + "\n" + doi[12:] + " end", ids), (
        "A LINE-WRAPPED identifier. Word wrap inside a token is the commonest "
        "way a DOI survives in prose while defeating a naive search."
    )
    assert _find("encoded " + doi.replace("/", "%2F"), ids), "percent-encoded `/` missed"
    assert _find("entity " + doi.replace("/", "&#47;"), ids), "HTML-entity `/` missed"
    assert _find("pasted " + doi[:8] + "​" + doi[8:], ids), (
        "a zero-width space from a PDF paste hid the identifier"
    )
    assert _find("hyphenated PMC-" + pmc[3:], ids), "PMC-<digits> form missed"
    assert _find("dashed " + slug.replace("_", "-"), ids), (
        "an APA slug written with dashes instead of underscores"
    )

    # --- must NOT find: the ways a looser matcher becomes a noise generator ---
    assert not _find("unrelated longer " + doi + "7 identifier", ids), (
        "one more DIGIT makes a different DOI; flagging it would get this gate "
        "turned off"
    )
    assert not _find(pmc + "8 is another paper", ids), "one more digit, different accession"
    assert not _find("no identifiers here at all", ids)
    assert not _find("a plainly-named " + doi + " paper", set()), (
        "an EMPTY identifier set must match nothing — this is the shape of the "
        "false green the first test exists to prevent"
    )


def test_the_matcher_survives_typographic_dash_substitution():
    """A dash class that matches only ASCII `-` reports clean on real documents.

    Found 2026-09-09 by verifying this gate from the other end, on a standing
    caution to check it both ways. Real prose carries U+2212 MINUS
    SIGN, U+2013 EN DASH and U+2014 EM DASH wherever a hyphen was typed: word
    processors, LaTeX and PDF text layers substitute them silently, and this
    project's whole subject matter is documents that have been through exactly
    those pipelines.

    `_dehyphen` stripped the ASCII hyphen and underscore from both sides and left
    every Unicode form in place, so an identifier containing a hyphen evaded the
    gate the moment its document had been typeset. Watched RED before the fix:
    U+2212, U+2013 and U+2014 all returned False against a real held-out key.

    Literals are composed at runtime for the reason given in the matcher test
    above -- this module is mirrored, and the mirror's own name gate scans it.
    """
    doi = "10." + "1234/example-2026-99"
    slug = "author" + "_2099_jrnl"
    ids = {doi, slug}

    assert _find("plain " + doi, ids), "positive control: the ASCII form must match"

    for name, ch in (
        ("U+2212 MINUS SIGN", "−"),
        ("U+2013 EN DASH", "–"),
        ("U+2014 EM DASH", "—"),
        ("U+2010 HYPHEN", "‐"),
        ("U+2011 NON-BREAKING HYPHEN", "‑"),
    ):
        assert _find("typeset " + doi.replace("-", ch), ids), (
            name + " substituted for the hyphen defeated the matcher"
        )
        assert _find("slug " + slug.replace("_", ch), ids), (
            name + " substituted for the underscore defeated the matcher"
        )

    assert not _find("10." + "1234/example-2026-98", ids), (
        "negative control: one changed digit is a DIFFERENT identifier and must "
        "still not match, or folding the dash class has made a noise generator"
    )


def test_the_shape_guard_admits_real_keys_and_refuses_generic_ones():
    """`_is_identifying` must admit every real corpus key and refuse noise."""
    assert _is_identifying("PMC" + "1234567"), "an accession must be admitted"
    assert _is_identifying("author" + "_2099_jrnl"), "an APA slug must be admitted"
    assert not _is_identifying("paper1"), "too short to identify anything"
    assert not _is_identifying("supplementary"), "no digit — not a paper key"
    assert not _is_identifying("paired-ttest-repeated"), (
        "a TASK name from the code-translation corpus is not a paper, and "
        "admitting it would flag half the repo"
    )


def test_the_tracked_file_list_is_not_vacuous():
    """Refuse to certify a run that read almost no files — BOTH halves.

    RENAMED AND WIDENED 2026-09-11. The old name was
    `test_the_scan_is_not_vacuous`, and it promised the whole check while
    asserting half of it. A scan here is the product of TWO inputs: the tracked
    FILE LIST and the held-out IDENTIFIER SET. `_find` returns False for every
    line of every file when the second one is empty, so an empty identifier set
    makes the scan just as vacuous as an empty file list — and this test said
    nothing about it.

    MEASURED on a corpus-less checkout, which is what every CI runner is:
    `6 passed, 6 skipped`, with `test_no_tracked_file_names_a_HELD_OUT_paper`
    and `test_no_answer_key_file_is_mirrored` among the SKIPS and this test
    among the PASSES. A reader scanning the summary for the word "vacuous" saw
    a pass and concluded the scan was not vacuous. It was entirely vacuous.

    Both halves are asserted now, and the second one cannot be made to go RED
    on a runner: the corpora are gitignored and must NEVER exist there, and the
    only way to close that would be a committed digest of ~30 PMC accessions,
    which is brute-forceable in seconds. So it reports its own absence as a
    SKIP. That is the whole point of the change — a pass and a skip are
    different verdicts, and only one of them is a claim.

    The file-list half is asserted BEFORE the skip deliberately, so a runner
    still gets that half enforced rather than losing both.
    """
    tracked = _tracked()
    if not tracked:
        pytest.skip("`git ls-files` returned nothing — not a git checkout")
    assert len(tracked) > 500, (
        f"only {len(tracked)} tracked paths were listed, so this scan examined "
        "almost nothing and its verdict is void."
    )

    if not _held_out_identifiers():
        pytest.skip(
            f"HALF-CERTIFIED ONLY. The file list is sound ({len(tracked)} "
            "tracked paths) but the identifier set is EMPTY, so `_find` matched "
            "nothing anywhere and the live gate scanned no identifiers at all. "
            "This is the expected state on a CI runner, a fresh clone and the "
            "public mirror — the corpora are gitignored and must never exist "
            "there. It is reported as a SKIP rather than a PASS so that no "
            "summary line can be read as 'the scan was not vacuous'. Only a run "
            "on the machine that holds the corpus certifies both halves."
        )


def test_no_tracked_file_is_skipped_as_binary():
    """The `_BINARY` skip list must stay empty of real tracked files.

    Reviewed defect: a tracked `.xlsx` with an accession column would never be
    scanned. Measured 2026-09-08 the repo tracks none, so the cheap fix is to
    assert that rather than to write a spreadsheet parser. If this goes red,
    read the file — do not extend the skip list.
    """
    tracked = _tracked()
    if not tracked:
        pytest.skip("not a git checkout")
    skipped = [r for r in tracked if Path(r).suffix.lower() in _TEXT_BEARING_BINARY]
    assert not skipped, (
        f"{len(skipped)} tracked file(s) carry readable text but are skipped as "
        "binary, so they are never checked. Read each one; do NOT move its "
        "extension into _OPAQUE — that is the hole this test keeps shut. "
        "never checked for held-out identifiers:\n  " + "\n  ".join(sorted(skipped))
    )


def test_an_abandoned_deletion_is_still_scanned_and_a_staged_one_is_not(tmp_path,
                                                                        monkeypatch):
    """TWO-SIDED CONTROL on the exemption `_scan` carried until 2026-09-11.

    Everything here is SYNTHETIC — a throwaway `git init` under `tmp_path`, a
    fake corpus, the unassigned `10.0000/` registrant prefix and a leading-zero
    `PMC0...` accession. No real identifier is read, written or matched, which
    is why this control can live in a mirrored module.

    THREE STATES, because the middle one alone would be half a control and the
    last one alone could pass vacuously:

      * file PRESENT and tracked -> FOUND. Proves the fixture is live; without
        it an empty identifier set would make the third state pass having
        matched nothing;
      * file DELETED from the working tree, deletion NOT staged -> FOUND. This
        is the ABANDONED deletion. The path is still tracked and still in HEAD,
        so the committed content is read and reported. The old exemption made
        this state report clean, which is how four unowned files — two naming
        held-out papers on 8 lines — sat GREEN in September 2026;
      * deletion STAGED -> NOT FOUND, and the path is no longer TRACKED. This
        is the genuinely in-flight removal, and it passes without any exemption
        because `git ls-files` stops listing a path the moment its deletion
        enters the index. Asserting the tracked-list fact alongside the verdict
        is what pins WHY it passes: if a future git kept listing it, this goes
        red here rather than silently blocking somebody's removal.

    `unreadable` is asserted at 0 in the middle state on purpose. A deleted file
    must be READ from its HEAD blob, not counted as unread — counting it would
    redden the live gate with the wrong diagnosis.
    """
    repo = tmp_path / "fixture-repo"
    paper = repo / "arenas" / "a-v1" / "task_sets" / "v1" / "_held_out" / "PMC00000001"
    paper.mkdir(parents=True)
    fake_doi = "10.0000/synthetic-control-" + "abandoned-deletion"
    (paper / "source.json").write_text(
        json.dumps({"doi": fake_doi}), encoding="utf-8"
    )
    note = repo / "notes.md"
    note.write_text("see " + fake_doi + " for the method", encoding="utf-8")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo),
             # A throwaway fixture must not run this repo's own hooks.
             "-c", "core.hooksPath=" + str(tmp_path / "no-hooks"),
             "-c", "user.name=fixture", "-c", "user.email=fixture@invalid",
             *args],
            capture_output=True,
        )

    if subprocess.run(["git", "init", "-q", str(repo)],
                      capture_output=True).returncode != 0:
        pytest.skip(
            "`git init` failed, so THIS CONTROL DID NOT RUN. The exemption it "
            "guards against could be reintroduced without anything going red."
        )
    # `notes.md` only — the corpus is the identifier SOURCE and must stay
    # untracked, exactly as the real one is gitignored.
    assert git("add", "--", "notes.md").returncode == 0
    assert git("commit", "-q", "-m", "fixture").returncode == 0

    monkeypatch.setitem(globals(), "REPO", repo)

    ids = _held_out_identifiers()
    assert ids, (
        "the fixture corpus produced no identifiers, so every verdict below "
        "would be vacuous. Check the shape guard and the source.json shape."
    )
    assert "notes.md" in _tracked(), "the fixture never committed the file"

    found, unreadable = _scan()
    assert "notes.md" in found, (
        "the fixture file names the fixture identifier and is tracked, yet the "
        "scan reported it clean. The matcher or the fixture is broken, and the "
        "two states below would both pass for the wrong reason."
    )

    note.unlink()
    assert "notes.md" in _git_lines("ls-files", "--deleted"), (
        "the fixture is not in the abandoned-deletion state this control tests"
    )
    found, unreadable = _scan()
    assert "notes.md" in found, (
        "ABANDONED DELETION READ AS CLEAN. A tracked file was removed from the "
        "working tree without the removal being staged, and the scan skipped "
        "it. That is the exact false green this control exists to prevent: the "
        "path is still tracked and its committed content still names the "
        "paper. Do not reintroduce a `git ls-files --deleted` exemption — a "
        "removal that is genuinely in flight is already out of scope because "
        "staging it drops the path from `git ls-files` entirely."
    )
    assert unreadable == 0, (
        f"{unreadable} unreadable file(s): a deleted-but-tracked path must be "
        "read from its HEAD blob, not counted as unread."
    )

    assert git("add", "-A", "--", "notes.md").returncode == 0
    assert "notes.md" not in _tracked(), (
        "git still lists a path whose deletion is STAGED. The removal of the "
        "deleted-file exemption relies on git dropping it here; if that has "
        "changed, `_scan` now blocks genuine in-progress removals and needs an "
        "exemption scoped to STAGED deletions only."
    )
    found, _ = _scan()
    assert "notes.md" not in found, (
        "a deletion that HAS been staged was flagged. An in-flight removal must "
        "still pass — otherwise this gate blocks the very fix it asks for."
    )

    # FOURTH STATE, covering the `rel in staged` branch the fix added. A path
    # ADDED to the index and then removed from disk without staging the removal
    # is tracked, has no working-tree bytes AND no HEAD blob — the only source
    # of its content is the staged blob. It must be FOUND (the index really does
    # carry the identifier) and it must NOT be counted unreadable, because the
    # scan read every byte this path has.
    fresh = repo / "staged-only.md"
    fresh.write_text("see " + fake_doi + " for the method", encoding="utf-8")
    assert git("add", "--", "staged-only.md").returncode == 0
    fresh.unlink()
    assert "staged-only.md" in _tracked(), "the fixture never staged the file"
    found, unreadable = _scan()
    assert "staged-only.md" in found, (
        "a path whose ONLY content is its staged blob was read as clean. The "
        "identifier is in the index and would be committed by the next commit."
    )
    assert unreadable == 0, (
        f"{unreadable} unreadable file(s): a staged-only path has no HEAD blob "
        "by definition, and counting it unread would redden the live gate for a "
        "file whose every byte was in fact scanned."
    )


def test_no_tracked_file_names_a_HELD_OUT_paper():
    """THE LIVE GATE.

    Failure reports FILE and LINE and never the identifier, because this module
    is mirrored and a test log is a publication channel too.
    """
    ids = _held_out_identifiers()
    if not ids:
        pytest.skip(
            "no held-out corpus on disk — fresh clone, CI runner, or public "
            "mirror. See test_the_identifier_set_is_present_or_this_is_a_"
            "corpusless_checkout: a green here is not evidence."
        )
    tracked = _tracked()
    assert len(tracked) > 500, (
        f"`git ls-files` returned {len(tracked)} paths. THIS GATE IS ASSERTING "
        "ITS OWN INPUT because a git failure returns an empty list, and an "
        "empty list makes the assertion below pass having checked nothing — a "
        "reviewed false green, not a hypothetical."
    )

    offenders, unreadable = _scan()
    # The recorded debt is subtracted here rather than skipped in `_scan`, so
    # `test_the_recorded_debt_is_still_outstanding` can still see those files
    # and go red the moment one of them is cleaned or removed.
    offenders = {p: n for p, n in offenders.items() if p not in _ARCHIVED_DEBT}
    assert unreadable == 0, (
        f"{unreadable} tracked file(s) could not be read, so they were not "
        "checked. A skipped file is not a clean file."
    )
    assert not offenders, (
        "a TRACKED file names a HELD-OUT paper — by DOI, by PMC accession, or "
        "by the author-year-journal slug its corpus keys it under:\n"
        + "".join(
            "  " + path + (":" + ",".join(str(n) for n in lines) if lines
                           else " (identifier is split across lines)") + "\n"
            for path, lines in sorted(offenders.items())
        )
        + "\nNaming which papers are held out IS the answer key: a player who "
        "knows them can fetch and pre-process exactly those papers before the "
        "run, which is the contamination the split exists to prevent.\n\n"
        "FIX: open each line and replace the identifier with a placeholder — "
        "the unassigned registrant prefix 10.0000/ for a DOI, a leading-zero "
        "PMC0... for an accession. Nothing in this repo resolves either, so a "
        "synthetic one documents identically. Where a document must distinguish "
        "several held-out papers, use document-local labels. The identifiers "
        "are deliberately not printed above; do not paste one into a commit "
        "message, an issue, or this file while fixing it."
    )


def test_no_answer_key_file_is_mirrored():
    """What keeps the answer-key exclusion from being a hole.

    Ground truth, seed lists and run records legitimately name held-out papers
    and are excluded. That is only safe while none of them is published.
    `framework/publish.py` makes everything outside `PUBLIC_TOP_LEVEL` private
    by DEFAULT, but `public_manifest()` enumerates the filesystem, so this
    asserts the OUTCOME rather than trusting the rule.
    """
    manifest = REPO / "contract" / "public_manifest.txt"
    if not manifest.is_file():
        pytest.skip("contract/public_manifest.txt absent — not a repo checkout")
    # `.strip()` is load-bearing: the manifest is CRLF, and a stray `\r` makes
    # every comparison fail, silently emptying this check.
    mirrored = {line.strip() for line in
                manifest.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert mirrored, "the manifest read as empty, so this check is vacuous"
    ids = _held_out_identifiers()
    if not ids:
        pytest.skip("no held-out corpus on disk")
    naming = []
    for rel in sorted(p for p in mirrored if _is_answer_key(p)):
        try:
            if _find((REPO / rel).read_text(encoding="utf-8", errors="ignore"), ids):
                naming.append(rel)
        except OSError:
            continue
    assert not naming, (
        "a file excluded from the gate above as 'answer key' is in the PUBLIC "
        "manifest AND names a held-out paper:\n  " + "\n  ".join(naming)
        + "\n\nThe exclusion assumed these are never published. Either remove "
        "the path from PUBLIC_TOP_LEVEL or stop excluding it."
    )


def test_the_recorded_debt_is_still_outstanding():
    """Every `_ARCHIVED_DEBT` entry must still be a real, unfixable offender.

    THIS IS WHAT MAKES THE DEBT SELF-DELETING, and it is why the debt is
    subtracted in the live gate rather than skipped during the scan. Three ways
    an entry can go stale, all of them red here:

      * the file was cleaned anyway (the archive rule changed, or the run was
        re-published) — then the entry is a silent exemption for nothing;
      * the file was deleted — same;
      * it never named a paper and the entry was wrong.

    Delete the entry and this test's expectation TOGETHER. An emptied dict makes
    this test vacuously true, which is why it asserts the dict is non-empty
    first: a debt list that quietly became a no-op reads exactly like a repo
    with no debt.
    """
    ids = _held_out_identifiers()
    if not ids:
        pytest.skip("no held-out corpus on disk — see the first test")
    assert _ARCHIVED_DEBT, (
        "the debt list is empty, so this test asserts nothing. If the debt is "
        "genuinely paid, DELETE THIS TEST in the same change — do not leave a "
        "vacuous guard standing where a real one used to be."
    )
    stale = []
    for rel in _ARCHIVED_DEBT:
        path = REPO / rel
        if not path.is_file():
            stale.append(f"{rel} (gone)")
            continue
        if not _find(path.read_text(encoding="utf-8", errors="ignore"), ids):
            stale.append(f"{rel} (clean now)")
    assert not stale, (
        "a recorded debt is no longer outstanding:\n  " + "\n  ".join(stale)
        + "\n\nRemove the entry from _ARCHIVED_DEBT in the same commit. An "
        "exemption that outlives the thing it exempts is how a gate quietly "
        "stops covering a file nobody remembers excluding."
    )


def test_no_debt_file_is_mirrored():
    """The debt is only tolerable while none of it is PUBLISHED.

    These files name held-out papers and cannot be scrubbed, so the single fact
    that bounds the risk is that the mirror does not ship them. That is asserted
    here rather than assumed, because `public_manifest()` enumerates the
    filesystem and a path can enter it without anybody editing the file.
    """
    manifest = REPO / "contract" / "public_manifest.txt"
    if not manifest.is_file():
        pytest.skip("contract/public_manifest.txt absent — not a repo checkout")
    mirrored = {line.strip() for line in
                manifest.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert mirrored, "the manifest read as empty, so this check is vacuous"
    published = sorted(set(_ARCHIVED_DEBT) & mirrored)
    assert not published, (
        "a file on the held-out DEBT list is now in the PUBLIC manifest:\n  "
        + "\n  ".join(published)
        + "\n\nThe debt was accepted ONLY because these are private. It is no "
        "longer acceptable: either keep the path out of PUBLIC_TOP_LEVEL, or "
        "resolve the archive-vs-disclosure conflict properly."
    )
