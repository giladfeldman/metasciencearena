"""Every real-paper load site must build the SAME task from custody.

Why this is a correctness test and not a plumbing test: a held-out paper's PDF
bytes are base64'd INTO the task envelope (`input.document_bytes_b64`) and the
envelope is NOT persisted -- every `document_bytes_b64` in a published
`_ground_truth.json` is the literal string `<pdf_bytes_omitted>`. So the bytes
the generator reads ARE the task. Resolving a different manifestation of the
same paper does not merely change a score, it changes the question, under an
unchanged `task_id`, and the published task set silently stops reproducing its
fingerprint with nothing going red.

One of the held-out papers is exactly that trap: the article repository's copy
and the arena's copy have the same DOI, the same byte count and byte-identical
extracted text, and different sha256. A size, filename or DOI check passes that
pair. Only the content hash separates them.

SEVEN LOAD SITES, NOT FOUR. The custody hand-off named four. Grepping the tracked
tree for `paper.pdf` found three more: `pdf-citation-matching-v1`,
`pdf-reference-parsing-v1` and `pdf-table-extraction-v1` all point their own
`HELD_OUT_PMC_DIR` at `pdf-text-fidelity-v1`'s shared corpus and read the file
themselves. Wiring only the named four and deleting the local copies would have
left those three arenas emitting ZERO real-paper tasks -- a benchmark that
quietly shrank by three arenas' worth of held-out tasks, with every suite green.
That is the failure `framework.holdout.require_corpus` exists to prevent, and it
would have walked straight past it, because the directory still exists.

HOW IT IS CHECKED, and why it does not go stale. Each site is run twice over the
same corpus: once against a mirror holding NO local PDF (custody resolves them)
and once against a mirror where the PDFs have been materialised as local files.
The two must emit byte-identical envelopes. Written RED first: before the wiring,
the custody-only half emitted nothing at every site. It keeps its teeth after the
local copies are deleted, because the local-file half is reconstructed from
custody rather than read from the tree.

This file is published to the public mirror, so it names no held-out paper.
Directories are enumerated, identifiers are hashed, and no count is pinned --
see `test_public_mirror_names_no_heldout_paper.py`.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.article_custody import (  # noqa: E402
    SOURCE_RECORD,
    has_paper,
    held_out_pdf_bytes,
)

PMC_DIR = REPO_ROOT / "arenas" / "pdf-text-fidelity-v1" / "task_sets" / "v1" / "_held_out" / "pmc"
APA_DIR = REPO_ROOT / "arenas" / "pdf-section-structure-v1" / "task_sets" / "v1" / "_held_out" / "apa"

#: (arena_id, generator function, corpus-dir attribute, corpus, task set version)
#:
#: All seven sites that turn a held-out paper into a task envelope. Keep this in
#: step with `git grep -n 'HELD_OUT_.*_DIR' arenas/*/generator.py` -- a site
#: missing from here is a site whose bytes nothing checks.
SITES = [
    ("pdf-section-structure-v1", "_real_pmc_tasks", "HELD_OUT_PMC_DIR", PMC_DIR, "v1"),
    ("pdf-section-structure-v1", "_real_apa_tasks", "HELD_OUT_APA_DIR", APA_DIR, "v1"),
    ("pdf-text-fidelity-v1", "_real_pmc_tasks", "HELD_OUT_PMC_DIR", PMC_DIR, "v1"),
    ("pdf-text-fidelity-v1", "_real_pmc_tasks_v2", "HELD_OUT_PMC_DIR", PMC_DIR, "v2"),
    ("pdf-citation-matching-v1", "_real_pmc_tasks", "HELD_OUT_PMC_DIR", PMC_DIR, "v1"),
    ("pdf-reference-parsing-v1", "_real_pmc_tasks", "HELD_OUT_PMC_DIR", PMC_DIR, "v1"),
    ("pdf-table-extraction-v1", "_real_pmc_tasks", "HELD_OUT_PMC_DIR", PMC_DIR, "v1"),
]
SITE_IDS = [f"{a}:{f}" for a, f, _attr, _d, _v in SITES]


def _paper_dirs(corpus: Path) -> list[Path]:
    if not corpus.is_dir():
        return []
    return sorted(d for d in corpus.iterdir() if d.is_dir() and has_paper(d))


def _recorded_sha(paper_dir: Path) -> str | None:
    record = paper_dir / SOURCE_RECORD
    if not record.is_file():
        return None
    return json.loads(record.read_text(encoding="utf-8"))["sha256"]


def _load_generator(arena_id: str):
    path = REPO_ROOT / "arenas" / arena_id / "generator.py"
    spec = importlib.util.spec_from_file_location(f"_custody_parity_{arena_id}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def mirrors(tmp_path_factory):
    """Two copies of each corpus: custody-only, and with local PDFs restored.

    Session-scoped because materialising the PDFs is the expensive part and
    every site reuses the same two corpora. The local half is written FROM
    custody, so this fixture keeps working once the tree holds no PDF at all --
    which is the whole point of the move it guards.
    """
    root = tmp_path_factory.mktemp("custody_parity")
    built: dict[Path, tuple[Path, Path]] = {}
    for corpus in (PMC_DIR, APA_DIR):
        papers = _paper_dirs(corpus)
        if not papers or any(_recorded_sha(d) is None for d in papers):
            continue
        custody_only = root / f"{corpus.name}_custody"
        with_local = root / f"{corpus.name}_local"
        shutil.copytree(corpus, custody_only,
                        ignore=shutil.ignore_patterns("paper.pdf"))
        shutil.copytree(custody_only, with_local)
        for paper in papers:
            (with_local / paper.name / "paper.pdf").write_bytes(held_out_pdf_bytes(paper))
        built[corpus] = (custody_only, with_local)
    return built


def _run(arena_id: str, func_name: str, attr: str, corpus_dir: Path, version: str) -> dict[str, dict]:
    mod = _load_generator(arena_id)
    original = getattr(mod, attr)
    try:
        setattr(mod, attr, corpus_dir)
        return {env["task_id"]: env for env in getattr(mod, func_name)(version, 12345)}
    finally:
        setattr(mod, attr, original)


@pytest.mark.parametrize("arena_id,func_name,attr,corpus,version", SITES, ids=SITE_IDS)
def test_load_site_builds_identical_tasks_from_custody(
    arena_id, func_name, attr, corpus, version, mirrors
):
    """Custody and a local file must produce byte-identical envelopes.

    Before the wiring the custody-only half emitted NOTHING at every site, which
    is what made this red: a load site that reads a local `paper.pdf` silently
    shrinks the benchmark rather than failing when the file is gone.
    """
    pair = mirrors.get(corpus)
    if pair is None:
        pytest.skip(f"held-out corpus absent or unrecorded at {corpus}")
    custody_only, with_local = pair

    from_custody = _run(arena_id, func_name, attr, custody_only, version)
    from_local = _run(arena_id, func_name, attr, with_local, version)

    assert from_local, (
        f"{arena_id}:{func_name} emitted no envelopes even WITH local PDFs — "
        "this site's other preconditions are unmet, so the comparison below "
        "would be vacuous"
    )
    assert set(from_custody) == set(from_local), (
        f"{arena_id}:{func_name} emitted {len(from_custody)} task(s) from custody "
        f"but {len(from_local)} from local files. A site that reads a local "
        f"paper.pdf emits nothing once the file is gone, which shrinks the "
        f"benchmark instead of failing."
    )
    for task_id, env in from_custody.items():
        assert env == from_local[task_id], (
            "an envelope built from custody differs from the one built from the "
            "local file — a different manifestation is a DIFFERENT TASK under "
            "the same task_id"
        )


@pytest.mark.parametrize("arena_id,func_name,attr,corpus,version", SITES, ids=SITE_IDS)
def test_every_envelope_carries_the_bytes_source_json_names(
    arena_id, func_name, attr, corpus, version, mirrors
):
    """The bytes in the envelope must hash to what the paper's record declares.

    The parity test above would still pass if BOTH halves resolved the same
    wrong manifestation. This pins them to the recorded content hash.
    """
    pair = mirrors.get(corpus)
    if pair is None:
        pytest.skip(f"held-out corpus absent or unrecorded at {corpus}")
    custody_only, _with_local = pair
    expected = {d.name: _recorded_sha(d) for d in _paper_dirs(corpus)}

    envelopes = _run(arena_id, func_name, attr, custody_only, version)
    if not envelopes:
        pytest.skip(f"{arena_id}:{func_name} emits no real-paper tasks here")
    for task_id, env in envelopes.items():
        doc_id = task_id.rsplit("-", 1)[-1]
        assert doc_id in expected, f"envelope {task_id} maps to no paper directory"
        raw = base64.b64decode(env["input"]["document_bytes_b64"])
        assert hashlib.sha256(raw).hexdigest() == expected[doc_id], (
            f"{arena_id}:{func_name} put bytes in an envelope that are not the "
            f"ones {SOURCE_RECORD} names for that paper"
        )


@pytest.mark.parametrize("corpus", [PMC_DIR, APA_DIR], ids=["pmc", "apa"])
def test_custody_returns_the_same_bytes_as_a_local_copy(corpus):
    """While local copies still exist, custody must return them byte-for-byte.

    This is the check that licenses the irreversible delete. It skips once the
    local copies are gone — by then the parity test above is the live guarantee,
    and it no longer depends on the tree holding a PDF.
    """
    papers = [d for d in _paper_dirs(corpus) if (d / "paper.pdf").is_file()]
    if not papers:
        pytest.skip(f"no local paper.pdf remains under {corpus}")
    checked = 0
    for paper in papers:
        if _recorded_sha(paper) is None:
            continue
        assert held_out_pdf_bytes(paper) == (paper / "paper.pdf").read_bytes(), (
            "custody resolved bytes that differ from the local copy"
        )
        checked += 1
    assert checked, f"{corpus}: nothing was actually compared"


def test_no_arena_generator_still_reads_a_local_paper_pdf():
    """No load site may reach past custody for a file on disk.

    A grep, deliberately, and over every arena generator rather than the seven
    sites listed above: the list is maintained by hand, and this is what catches
    an EIGHTH site added later. The hand-off that started this work named four
    sites when there were seven, so the list being wrong is the expected failure,
    not a hypothetical one.
    """
    offenders = []
    for gen in sorted((REPO_ROOT / "arenas").glob("*/generator.py")):
        text = gen.read_text(encoding="utf-8")
        if 'sub / "paper.pdf"' in text or "sub / 'paper.pdf'" in text:
            offenders.append(gen.relative_to(REPO_ROOT).as_posix())
    assert not offenders, (
        "these arena generators still read a held-out paper from a local file, "
        "which is empty now that article-finder holds the papers:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse framework.article_custody.held_out_pdf_bytes(sub) and "
        "has_paper(sub) instead, and add the site to SITES above."
    )


def test_the_leak_scan_still_fingerprints_the_papers_themselves(tmp_path):
    """Moving the PDFs into custody must not shrink the mirror's leak scan.

    `publish.held_out_shingles` fingerprints every file under `_held_out/` and
    `publish.scan_for_leaks` refuses to mirror any file sharing a long verbatim
    run with one. Until 2026-09-05 the papers were among those files. Deleting
    them dropped the fingerprint set from ~3.42M shingles over 143 files to
    437k over 107 — 12.8% retained — with the gate still reporting a clean
    build. Losing coverage silently is exactly the failure the vacuity guard in
    `held_out_shingles` exists to prevent, and a non-zero file count walks past
    it.

    Article-PROSE coverage was never at risk: `jats.xml` and `gold.json` stayed,
    and a pasted paragraph is the realistic accident the scan is a backstop for.
    What was lost is the narrow case of a held-out paper's raw bytes reaching a
    mirrored file. This test pins it, because a gate that quietly covers less
    than it did is worse than one that never covered it: nothing tells you.

    Written RED first — before `held_out_shingles` resolved the papers from
    custody, the planted run below was not flagged.
    """
    from framework import publish

    papers = [d for d in _paper_dirs(PMC_DIR) + _paper_dirs(APA_DIR)
              if (d / SOURCE_RECORD).is_file()]
    if not papers:
        pytest.skip("no held-out paper records on disk")

    shingles, n_files = publish.held_out_shingles(REPO_ROOT)
    assert n_files >= len(papers), (
        f"the leak scan fingerprinted {n_files} held-out files but there are "
        f"{len(papers)} papers alone — the papers themselves are not being "
        f"resolved from custody, so a mirrored file could carry one verbatim"
    )

    # Two-sided, at the real call site: a run taken from a paper's own bytes
    # must be refused, and a file that carries no such run must not be.
    raw = held_out_pdf_bytes(papers[0]).decode("utf-8", errors="replace")
    window = raw[2000:2000 + 4 * publish.MIN_DETECTED_RUN]

    # THE PROBE IS WRITTEN OUTSIDE THE REPOSITORY, DELIBERATELY.
    #
    # It used to be written to `framework/_leak_scan_probe_tmp.py` and removed
    # in a `finally`. A `finally` does not survive the process being killed: on
    # 2026-09-11 an interrupted suite left 896 bytes of verbatim held-out paper
    # text in the working tree for 18 minutes. `public_manifest` enumerates the
    # FILESYSTEM, not git, so while that file existed it was IN the computed
    # public mirror — a publish or a mirror sync in that window would have
    # carried it, and `test_publish.py::test_the_real_manifest_is_clean`
    # reported it as a held-out leak in the public manifest. Several sessions
    # share this one working tree, so a concurrent process globbing
    # `framework/` saw it too; that produced two full-suite failures which did
    # not reproduce serially.
    #
    # `scan_for_leaks` joins `repo_root / rel`, and joining an ABSOLUTE path
    # yields that path unchanged, so a probe anywhere on disk is scanned
    # exactly as an in-tree file would be. `pytest`'s `tmp_path` owns the
    # cleanup, and nothing in the repository is touched even if this process is
    # killed mid-assert.
    probe = (tmp_path / "_leak_scan_probe.py").resolve()
    probe.write_text(window, encoding="utf-8")

    # Pin the orphan case shut rather than assuming any directory is safe:
    # whatever path the probe is given, the manifest must not reach it.
    reachable = {(REPO_ROOT / rel).resolve() for rel in publish.public_manifest(REPO_ROOT)}
    assert probe not in reachable, (
        f"the leak-scan probe at {probe} is inside the computed public "
        f"manifest — while this test runs, a mirror sync would publish "
        f"verbatim held-out paper text"
    )

    flagged = publish.scan_for_leaks(REPO_ROOT, [probe], gates="all")
    assert flagged, (
        "a verbatim run from a held-out PAPER was not refused by the leak "
        "scan — the scan no longer fingerprints the papers themselves"
    )
    clean = publish.scan_for_leaks(
        REPO_ROOT, [Path("framework/article_custody.py")], gates="all")
    assert not clean, (
        "a known-clean mirrored file was refused — the scan is a "
        "yes-machine and its refusals mean nothing"
    )
