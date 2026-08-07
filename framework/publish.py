"""What may be published, decided by a rule rather than a list.

THE PROBLEM THIS SOLVES
-----------------------
The repo is private and stays that way: the app, the held-out corpora and the
private seeds live here. But the funder-facing claim is that a published score
is *reproducible*, and that requires the scoring logic, the task generators and
the contract to be public. So one tree has to be split in two, repeatedly, as
arenas are added.

A hand-maintained "these files are public" list drifts the first time someone
adds an arena and forgets to update it — and it drifts in the dangerous
direction, because forgetting means a NEW arena's held-out data quietly inherits
whatever the wildcard above it allows.

So publishability is *computed*:

    An arena is publishable iff
      (a) its task set has a `.private_seed` — meaning the secret is the seed
          VALUE (gitignored, never in git), not the code; AND
      (b) nothing under it depends on the real-paper gold path
          (`framework.gold`, `framework.jats`, or the local `article-finder`
          skill), which is how gold for copyrighted / real papers is authored.

THE TWO CONDITIONS ARE NOT SYMMETRIC
------------------------------------
An earlier plan recorded that the two halves "agree exactly, 16 + 6 = 22, zero
overlap, zero gap". Measured, that is not what they do:

    seeded      16   <- THIS is the partition: 16 public, 6 private
    gold-free   20   <- 2 arenas import framework.jats
    publishable 16

Condition (a) produces the split on its own. Condition (b) is a one-way VETO: it
can only move an arena from public to private, never the reverse. Four arenas
are unseeded yet gold-free, and that is fine — not being seed-reproducible is
sufficient reason to stay private.

Keeping the veto is still the conservative direction: an arena that acquires a
seed while still authoring gold from real papers would be published by the seed
rule alone, and that mistake cannot be undone once it is on PyPI. Today no arena
is in that state, and `test_publish.py` fails the build if one ever is.

WHAT A PATH RULE CANNOT DO
--------------------------
A path allowlist blocks `_held_out/` but it cannot block an answer that was
copied into `arena.yaml`, into a scorer fixture, or into a doc comment. That is
why `scan_for_leaks` exists: it reads the actual held-out material and refuses
to mirror any file that reproduces a long verbatim run of it. See that
function's docstring for exactly what it does and does not catch.
"""
from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ArenaPublishability",
    "PRIVATE_GOLD_MARKERS",
    "PRIVATE_PACKAGES",
    "PRIVATE_TEST_FILES",
    "imports_a_private_package",
    "PUBLIC_ARENA_GLOBS",
    "PUBLIC_TOP_LEVEL",
    "classify_arenas",
    "code_only",
    "MIRROR_RENAMES",
    "held_out_shingles",
    "PUBLISHED_ARENAS_FILE",
    "public_manifest",
    "published_arena_ids",
    "scan_for_leaks",
]

#: Importing or path-inserting any of these means an arena's gold comes from
#: real papers, which are not ours to republish.
PRIVATE_GOLD_MARKERS = (
    "framework.gold",
    "framework.jats",
    "article-finder",
    "article_finder",
)

#: Repo-root paths mirrored verbatim. Anything not matched here is private by
#: DEFAULT — the list says what goes out, never what stays.
PUBLIC_TOP_LEVEL = (
    "framework/**/*.py",
    "framework/contract/schemas/*.json",
    # Test FIXTURES.  never matched these, so the mirrored
    # tests referenced fake arenas and registries that were not shipped.
    "framework/tests/fixtures/**/*.yaml",
    "framework/tests/fixtures/**/*.yml",
    "framework/tests/fixtures/**/*.json",
    "framework/tests/fixtures/**/*.jsonl",
    "contract/README.md",
    "contract/arena.example.yaml",
    "taxonomy/**/*.yaml",
    "taxonomy/**/*.md",
    "pyproject.toml",
    "requirements.lock",
    "LICENSE",
    "PACKAGE_README.md",
    "CITATION.cff",
    "DATA_HANDLING.md",
    "publish_templates/*.yml",
    "contract/published_arenas.json",
)

#: Paths that land under a DIFFERENT name in the mirror.
#:
#: Two cases, both deliberate:
#:  * the repo README is repo-facing (private layout, app, outreach); the
#:    registry-facing one is PACKAGE_README.md and becomes the mirror's README.
#:  * workflow files live under `publish_templates/` here rather than
#:    `.github/workflows/`, so the PRIVATE repo never tries to run a PyPI
#:    publish. Trusted publishing is configured for the public repo only, so a
#:    stray tag here would fail — noisily, but for the wrong reason.
MIRROR_RENAMES = {
    "PACKAGE_README.md": "README.md",
    "publish_templates/publish-pypi.yml": ".github/workflows/publish-pypi.yml",
    "publish_templates/public-ci.yml": ".github/workflows/ci.yml",
}

#: Per-arena paths mirrored for a PUBLISHABLE arena. Generators and scorers,
#: not generated data: a public task set is reproducible from the arena's
#: revealed seed, so shipping the code ships the tasks without shipping a second
#: copy that can drift from — or leak beyond — what the generator emits.
PUBLIC_ARENA_GLOBS = (
    "arena.yaml",
    "README.md",
    "CHANGELOG.md",
    "difficulty.yaml",
    # Every module at the arena root, not just generator/scorer: arenas have
    # sibling helpers (r_runner.py, _normalize.py, make_dataset.py) that the
    # scorer imports. Naming only the two entry points shipped a scorer whose
    # import failed. Per-file rules still apply — a helper that reaches the gold
    # path or a private package is dropped by the checks below.
    "*.py",
    "catalogs/**/*.yaml",
    "schemas/*.json",
    "tests/**/*.py",
    "tools/**/*.py",
)

#: Per-arena paths that are private no matter which arena they belong to.
#: `tools/build_gold.py` is the gold-AUTHORING path: it reaches into the local
#: `article-finder` skill to resolve real papers. Publishing it would publish
#: the route to the answer key even for an arena whose tasks are seed-generated.
PRIVATE_ARENA_PATHS = ("tools/build_gold.py",)

#: Packages that exist only in the PRIVATE repo. A mirrored file with an
#: unguarded module-level import of one of these cannot run for anybody who
#: installs the package — not a leak, a breakage, but a public repo with a red X
#: is its own kind of unbacked claim.
#:
#: Three `framework/tests/*` files imported `players.adapters.*` and broke CI on
#: the public repo within minutes of the first release (2026-08-07). They are
#: integration tests for adapters the package does not ship, so they belong with
#: the private repo.
PRIVATE_PACKAGES = frozenset({"players"})

#: Test files that are bound to PRIVATE data and cannot run in the mirror.
#:
#: Determined empirically by running the mirrored suite (not guessed), then
#: encoded here with the reason. Every one reads something the package does not
#: ship — `players/prompts/*.txt`, `players/registry.yaml`, a private arena's
#: modules, or the private side of the publish boundary itself.
#:
#: This exists because the first release turned the public repo's CI red: the
#: mirror was shipping the private repo's INTEGRATION tests, which reference
#: assets only a checkout has. A public repo with a failing badge is its own kind
#: of claim the project cannot back — it was the first thing anyone following the
#: funder link would have seen.
#:
#: `scripts/build_public_mirror.py --verify-tests` RUNS the mirrored suite, so
#: this list cannot quietly go stale.
PRIVATE_TEST_FILES = frozenset({
    # Assert a prompt in players/prompts/ documents every deception kind.
    "arenas/grim-consistency-v1/tests/test_generator.py",
    "arenas/open-practices-repro-v1/tests/test_generator.py",
    "arenas/prereg-extraction-v1/tests/test_generator.py",
    "arenas/reference-integrity-v1/tests/test_generator.py",
    "arenas/reporting-completeness-v1/tests/test_generator.py",
    "arenas/significance-language-v1/tests/test_generator.py",
    "arenas/stats-extraction-v1/tests/test_generator.py",
    "arenas/transparency-statements-v1/tests/test_generator.py",
    "arenas/stats-extraction-v1/tests/test_coverage.py",
    # Read players/registry.yaml, private arenas, or run the private side of the
    # publish boundary (whose answer differs by construction inside the mirror).
    "framework/tests/test_publish.py",
    "framework/tests/test_parity.py",
    "framework/tests/test_registry_attribution.py",
    "framework/tests/test_retry_failed.py",
    "framework/tests/test_runner_provenance.py",
    "framework/tests/test_scoring_text.py",
    "framework/tests/test_stale_input_audit.py",
    # Asserts PRIVATE-repo layout invariants: the 19-arena module-name collision,
    # the pyproject collection config, and that every arena with a scorer has
    # tests. The mirror has 16 arenas and deliberately omits some tests, so the
    # invariant is legitimately false there.
    "framework/tests/test_test_collection_integrity.py",
    # code-translation-r-v1 builds gold by EXECUTING R over
    # `source_scripts/` — which contains `source_scripts/gold/*.json`. Whether
    # that (revealed) gold should be published is a deliberate disclosure
    # decision for the project owner, not something to infer: under-publishing
    # is reversible, over-publishing is not. Until it is decided, the arena's
    # generator and scorer ship (so the LOGIC is auditable) but these two tests,
    # which need the source scripts to run, do not. See TODO.md.
    "arenas/code-translation-r-v1/tests/test_generator.py",
    "arenas/code-translation-r-v1/tests/test_scorer.py",
})

#: Files that contain the marker strings as VOCABULARY rather than as a
#: dependency: this module declares them, and its test exercises them. Without
#: this the rule condemns the rule — the same self-reference that made the theme
#: guard fire on its own documentation.
MARKER_VOCABULARY_FILES = frozenset({
    "framework/publish.py",
    "framework/tests/test_publish.py",
})

#: Belt-and-braces denylist applied AFTER the allowlist.
#:
#: `framework/gold/` and `framework/jats.py` are here because a blanket
#: `framework/**/*.py` allowlist swept them into the first real mirror build
#: (caught by reading the file list, 2026-08-07). `framework/gold/__init__.py`
#: describes itself as "the ONLY way an arena obtains ground truth" and warns
#: that "a player that can read this module can read the answer key" — it is the
#: gold-AUTHORING path, and publishing it publishes the route to every answer.
#: Both reach into the local `article-finder` skill.
#:
#: Note what did NOT catch this: the verbatim scanner. These files contain code
#: that *reads* gold, not gold text, so there was nothing to match. An allowlist
#: that is too broad and a content scan that only sees copied text leave exactly
#: this gap between them.
NEVER_PUBLISH = re.compile(
    r"(^|/)(_held_out|runs|node_modules|__pycache__|\.venv|\.git)(/|$)"
    r"|(^|/)\.private_seed$"
    r"|(^|/)_ground_truth\.json$"
    r"|(^|/)\.env"
    r"|^framework/gold(/|$)"
    r"|^framework/jats\.py$"
)

#: Windows that DO overlap held-out material but are not answers.
#:
#: Each entry is a blake2b fingerprint, never the text: this module is itself
#: mirrored, so pasting the overlapping string here would put a fragment of
#: held-out material into the public repo — the very thing the scanner exists to
#: prevent. The reason is what a reviewer audits; the fingerprint is what the
#: code matches.
#:
#: Adding an entry is a deliberate act. It suppresses ONE exact window, so any
#: other overlap in the same file still fails.
ACKNOWLEDGED_OVERLAPS: dict[str, str] = {
    # arenas/code-translation-r-v1/tests/test_scorer.py — the arena's own
    # harvest CONTRACT (the R named-list shape a translated script must print).
    # It appears in the held-out R sources because every case must satisfy it.
    # Publishing the contract is the point of the arena; it reveals no answer.
    "d0cb6a0da66a5d6a4cba7e51": "code-translation-r-v1 harvest contract (public API, not gold)",
    # arenas/open-practices-repro-v1/catalogs/repo_templates.yaml — the stock
    # "data available from the corresponding author upon reasonable request"
    # sentence. It is a published journal cliché the arena GENERATES from, and
    # it is in the catalog precisely so tasks can be synthesised; a held-out
    # case containing it is a coincidence of the phrase being ubiquitous.
    "118dba40a0705026638354af": "open-practices-repro-v1 data-availability boilerplate (1/3)",
    "535aee089e12455d4f39fdbe": "open-practices-repro-v1 data-availability boilerplate (2/3)",
    "ac2083f169dd8ac49d82913b": "open-practices-repro-v1 data-availability boilerplate (3/3)",
}

#: Length of the verbatim run `scan_for_leaks` detects. Long enough that normal
#: shared boilerplate (licence headers, import blocks) does not collide, short
#: enough to catch a pasted paragraph of a held-out paper.
SHINGLE_CHARS = 96
SHINGLE_STRIDE = 24

_TEXT_SUFFIXES = {
    ".py", ".md", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".toml",
    ".cff", ".csv", ".tsv", ".r", ".R", ".sps", ".do", ".cfg", ".lock",
}


@dataclass
class ArenaPublishability:
    arena_id: str
    root: Path
    has_private_seed: bool
    gold_markers: list[str] = field(default_factory=list)

    @property
    def seed_says_public(self) -> bool:
        return self.has_private_seed

    @property
    def gold_says_public(self) -> bool:
        return not self.gold_markers

    @property
    def publishable(self) -> bool:
        return self.seed_says_public and self.gold_says_public

    @property
    def conditions_agree(self) -> bool:
        return self.seed_says_public == self.gold_says_public

    def why(self) -> str:
        if self.publishable:
            return "seeded, no real-paper gold dependency"
        bits = []
        if not self.has_private_seed:
            bits.append("no .private_seed (tasks are not seed-reproducible)")
        if self.gold_markers:
            bits.append(f"depends on real-paper gold via {sorted(set(self.gold_markers))}")
        return "; ".join(bits)


def code_only(src: str) -> str:
    """Python source with comments and docstrings removed.

    WHY THIS IS NEEDED
    ------------------
    Marker matching over raw text reads PROSE as if it were a dependency. Two
    arenas were misclassified as unpublishable by exactly that: the matched text
    in `prereg-deviation-v1/generator.py` is a docstring saying *"no
    article-finder registry needed"* — the guard fired on documentation that
    asserted the opposite of what it was accused of.

    String LITERALS are deliberately kept. `AF_DIR = Path.home() / ".claude" /
    "skills" / "article-finder"` is a real dependency expressed as a string, and
    stripping all strings would hide it. Only comments and standalone string
    expressions (docstrings) go.

    Over-stripping is the dangerous direction, which is what
    `test_code_only_keeps_a_real_marker_but_drops_prose` pins.

    IMPLEMENTATION NOTE — WHY SPANS, NOT A TOKEN JOIN
    -------------------------------------------------
    The first version rebuilt the source by joining surviving tokens with a
    space. That silently broke every DOTTED marker: `from framework.jats import
    parse_tables` came back as `from framework . jats import parse_tables`, so
    `"framework.jats"` no longer matched and three real-paper arenas were
    reported as having no gold dependency at all. The self-test missed it
    because its marker was a single string token, which a join preserves.

    So: erase the spans of removed tokens and leave every other byte exactly
    where it was. Nothing that survives can be reshaped by the stripper.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Unparseable source: fall back to the raw text. Failing OPEN here would
        # mean a syntactically broken file could smuggle a marker past the check.
        return src

    lines = src.splitlines(keepends=True)
    kill: list[tuple[int, int, int, int]] = []
    at_statement_start = True
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            kill.append((*tok.start, *tok.end))
            continue
        if tok.type == tokenize.STRING and at_statement_start:
            kill.append((*tok.start, *tok.end))  # docstring / bare string expression
            continue
        if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                        tokenize.DEDENT, tokenize.ENCODING):
            at_statement_start = True
            continue
        at_statement_start = False

    # Blank back-to-front so earlier spans keep their coordinates.
    for srow, scol, erow, ecol in reversed(kill):
        si, ei = srow - 1, erow - 1
        if si < 0 or ei >= len(lines):
            continue
        if si == ei:
            lines[si] = lines[si][:scol] + " " * (ecol - scol) + lines[si][ecol:]
        else:
            lines[si] = lines[si][:scol] + "\n"
            for i in range(si + 1, ei):
                lines[i] = "\n"
            lines[ei] = " " * ecol + lines[ei][ecol:]
    return "".join(lines)


def classify_arenas(arenas_root: Path) -> list[ArenaPublishability]:
    """Apply the predicate to every arena under ``arenas_root``."""
    out: list[ArenaPublishability] = []
    for arena in sorted(p for p in arenas_root.iterdir() if (p / "arena.yaml").is_file()):
        # The seed lives at task_sets/<version>/.private_seed, not the arena root.
        has_seed = any(arena.glob("task_sets/*/.private_seed"))
        markers: list[str] = []
        for py in arena.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            rel = py.relative_to(arena).as_posix()
            # A marker inside a file that is never mirrored cannot make the
            # ARENA unpublishable — the gold-authoring tool is private on its
            # own terms, and letting it condemn its whole arena would withhold
            # 21 seed-generated task families for no gain.
            if rel in PRIVATE_ARENA_PATHS:
                continue
            try:
                text = py.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            code = code_only(text)
            markers.extend(m for m in PRIVATE_GOLD_MARKERS if m in code)
        out.append(ArenaPublishability(arena.name, arena, has_seed, markers))
    return out


#: Committed list of arenas that exist in the public mirror.
#:
#: The Next.js app needs this and CANNOT compute it: publishability turns on
#: `.private_seed`, which is gitignored and therefore absent from the Vercel
#: build. Without it the site deep-linked every arena into the public repo, and
#: the six private ones 404'd the moment REPO_URL was switched on — a link the
#: site could not back, which is the exact defect this whole stream exists to
#: remove. `test_publish.py` fails if this file drifts from the computed set.
PUBLISHED_ARENAS_FILE = "contract/published_arenas.json"


def published_arena_ids(arenas_root: Path) -> list[str]:
    """Sorted ids of arenas that appear in the public mirror."""
    return sorted(a.arena_id for a in classify_arenas(arenas_root) if a.publishable)


def imports_a_private_package(path: Path) -> str | None:
    """Name of the private package a file imports at module scope, else None.

    MODULE-LEVEL AND UNGUARDED ONLY. An import inside a function, or inside a
    `try: ... except ImportError:`, is a deliberate optional dependency —
    `framework/runner.py` imports `players.adapters` exactly that way so the
    in-repo adapters load in a checkout and are simply absent from an install.
    Flagging those would exclude the runner itself.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return None
    for node in tree.body:  # tree.BODY: top level only, so no try/except or def
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        for name in names:
            if name in PRIVATE_PACKAGES:
                return name
    return None


def public_manifest(repo_root: Path) -> list[Path]:
    """Repo-relative paths that make up the public mirror, sorted."""
    picked: set[Path] = set()

    for pattern in PUBLIC_TOP_LEVEL:
        for p in repo_root.glob(pattern):
            if p.is_file():
                picked.add(p.relative_to(repo_root))

    arenas_root = repo_root / "arenas"
    if arenas_root.is_dir():
        for arena in classify_arenas(arenas_root):
            if not arena.publishable:
                continue
            for pattern in PUBLIC_ARENA_GLOBS:
                for p in arena.root.glob(pattern):
                    if not p.is_file():
                        continue
                    # PRIVATE_ARENA_PATHS was previously consulted only when
                    # CLASSIFYING an arena, never here — so `tools/build_gold.py`
                    # was excluded from condemning its arena and then cheerfully
                    # mirrored. Both call sites must honour it.
                    if p.relative_to(arena.root).as_posix() in PRIVATE_ARENA_PATHS:
                        continue
                    picked.add(p.relative_to(repo_root))

    kept = [
        p for p in picked
        if not NEVER_PUBLISH.search(p.as_posix())
        # A file that cannot import in the published package must not ship.
        and not (p.suffix == ".py" and imports_a_private_package(repo_root / p))
        and p.as_posix() not in PRIVATE_TEST_FILES
    ]
    return sorted(kept)


def _fingerprint(window: str) -> str:
    """Stable digest of one window. Stable ACROSS PROCESSES — see _shingles."""
    return hashlib.blake2b(window.encode("utf-8"), digest_size=12).hexdigest()


def _shingles(text: str, stride: int) -> set[str]:
    """Hashed fixed-length character windows of whitespace-normalised text.

    STRIDE IS NOT SYMMETRIC, AND THAT MATTERS
    -----------------------------------------
    Both sides were originally indexed at the same coarse stride, which meant a
    copied passage was only detected when it happened to land at a multiple of
    that stride. A paste one character off — e.g. wrapped in `SAMPLE = \"\"\"…` —
    produced ZERO overlap. The scanner would have reported clean on the exact
    accident it exists to catch.

    So the candidate side is indexed at stride 1 (every window) and the held-out
    side at the coarse stride. Then any verbatim run of at least
    ``SHINGLE_CHARS + SHINGLE_STRIDE - 1`` characters must contain at least one
    whole coarse window, and that window is guaranteed to be in the candidate's
    stride-1 set regardless of alignment.

    The digest is blake2b rather than the builtin `hash()`: `hash()` is salted
    per process, so an acknowledged-overlap allowlist keyed on it would silently
    stop matching on the next run — and an allowlist that stops matching turns
    into a build failure nobody can reproduce.
    """
    flat = " ".join(text.split())
    if len(flat) < SHINGLE_CHARS:
        return set()
    return {
        _fingerprint(flat[i:i + SHINGLE_CHARS])
        for i in range(0, len(flat) - SHINGLE_CHARS + 1, stride)
    }


#: Shortest verbatim run the scan is guaranteed to catch, at any alignment.
MIN_DETECTED_RUN = SHINGLE_CHARS + SHINGLE_STRIDE - 1


def held_out_shingles(repo_root: Path) -> tuple[set[str], int]:
    """Fingerprints of every held-out artifact, plus how many files fed them.

    The count is returned so callers can refuse a VACUOUS scan: matching zero
    files against an empty fingerprint set proves nothing, and would report a
    clean bill of health on a repo whose held-out corpora had simply been moved.
    """
    shingles: set[str] = set()
    n_files = 0
    for held_out_dir in repo_root.glob("arenas/*/task_sets/*/_held_out"):
        for p in held_out_dir.rglob("*"):
            if not p.is_file():
                continue
            n_files += 1
            try:
                shingles |= _shingles(p.read_text(encoding="utf-8", errors="replace"), SHINGLE_STRIDE)
            except OSError:
                continue
    # Seeds are short, so they are matched verbatim rather than by shingle.
    return shingles, n_files


def private_seed_values(repo_root: Path) -> set[str]:
    """Every private seed's literal content. Any occurrence anywhere is a leak."""
    values = set()
    for p in repo_root.glob("arenas/*/task_sets/*/.private_seed"):
        try:
            v = p.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if v:
            values.add(v)
    return values


def scan_for_leaks(repo_root: Path, files: list[Path]) -> list[str]:
    """Reasons the given files must NOT be mirrored. Empty list == clean.

    WHAT THIS CATCHES
      * any file containing a private seed value verbatim;
      * any file sharing a {MINRUN}-character verbatim run (at ANY alignment)
        with a held-out
        artifact — i.e. a paragraph of a held-out paper pasted into a scorer
        fixture, an arena.yaml, or a comment, none of which a path rule sees.

    WHAT IT DOES NOT CATCH
      * paraphrase, translation, or summarisation of held-out content;
      * a gold ANSWER that never appears in a `_held_out/` file (e.g. one only
        ever computed at run time);
      * verbatim runs shorter than {MINRUN} characters.

    It is a backstop for copy-paste, which is the realistic accident. It is not
    a proof of non-disclosure, and the docs must not claim it is.
    """
    reasons: list[str] = []
    seeds = private_seed_values(repo_root)
    ho_shingles, ho_files = held_out_shingles(repo_root)

    # A module that REACHES the gold path is as dangerous as gold itself, and
    # the verbatim scan below cannot see it: such a file contains code that
    # reads the answer key, not the answer key. `framework/gold/` reached the
    # first mirror build through a blanket `framework/**/*.py` allowlist, and
    # only a human reading the file list caught it. So the marker check that
    # already governs ARENAS now runs over every mirrored Python file too.
    for rel in files:
        if rel.suffix != ".py" or rel.as_posix() in MARKER_VOCABULARY_FILES:
            continue
        try:
            code = code_only((repo_root / rel).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        hits = sorted({m for m in PRIVATE_GOLD_MARKERS if m in code})
        if hits:
            reasons.append(
                f"{rel}: reaches the real-paper gold path ({', '.join(hits)}). Publishing a "
                f"module that resolves gold publishes the route to the answer key, even "
                f"though the file contains no gold text for the verbatim scan to find."
            )

    # Non-vacuity. A scan that examined nothing must not report "clean" —
    # that is the exact shape of false confidence this project keeps re-learning.
    if not seeds and not ho_shingles:
        reasons.append(
            "REFUSING TO CERTIFY: found no private seeds and no held-out material to "
            "scan against. Either the arenas root is wrong or the corpora moved — "
            "a scan with nothing to compare cannot clear anything."
        )
        return reasons

    for rel in files:
        p = repo_root / rel
        if p.suffix not in _TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            reasons.append(f"{rel}: unreadable ({exc}) — cannot certify, so not mirrored")
            continue
        for seed in seeds:
            if seed in text:
                reasons.append(f"{rel}: contains a .private_seed value verbatim")
        if ho_shingles:
            # Stride 1 on the candidate: see _shingles for why symmetry here
            # silently defeats the whole check.
            overlap = (_shingles(text, 1) & ho_shingles) - set(ACKNOWLEDGED_OVERLAPS)
            if overlap:
                reasons.append(
                    f"{rel}: shares {len(overlap)} verbatim {SHINGLE_CHARS}-char window(s) "
                    f"with held-out material. If a window is shared BOILERPLATE rather "
                    f"than an answer, add its fingerprint to ACKNOWLEDGED_OVERLAPS with "
                    f"a reason: {sorted(overlap)[:3]}"
                )
    return reasons


scan_for_leaks.__doc__ = (scan_for_leaks.__doc__ or "").replace("{MINRUN}", str(MIN_DETECTED_RUN))
