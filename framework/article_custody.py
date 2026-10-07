"""Resolve a held-out paper's PDF bytes from the article repository.

article-finder is the portfolio's sole custodian of publications: no project
stores article PDFs in its own directory or git history. ScienceArena's two PDF
arenas were the last holdout -- 36 papers living as `paper.pdf` inside
gitignored `_held_out/` directories -- because those bytes are not a corpus the
arena merely reads, they ARE the task inputs.

`_real_pmc_tasks` and `_real_apa_tasks` base64 the PDF straight into the task
envelope, and the envelope is not persisted: every `document_bytes_b64` in
`task_sets/v1/_ground_truth.json` is the literal string `<pdf_bytes_omitted>`.
So the loose files were the only copy of what the published task sets were
generated from, and any substitution changes the TASK, not merely a score.

That is why resolution here pins the CONTENT HASH and never merely the DOI. For
one of the held-out papers the repository's copy and the arena's copy were the
SAME SIZE, same title, same DOI, with byte-identical extracted text -- and a
different sha256. Size, filename and DOI checks all pass such a pair; only the
content hash separates them. article-finder now holds both, as two
manifestations of one DOI, and this module picks the exact one a published task
set was built from.

(This module is mirrored publicly, so it names no real held-out paper. The
identifiers and byte counts live with the measurement, in the private
article-finder repo.)

Each held-out paper directory carries a `source.json`:

    {"doi": "10.0000/example.2025.1", "sha256": "<64 hex>", "size_bytes": 1234567}

A directory with no `source.json` but a local `paper.pdf` still works: that is
what `tools/fetch_pmc.py` produces for a newly fetched paper, and breaking it
would break the way the corpus grows.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class HeldOutPaperUnresolvable(RuntimeError):
    """A declared held-out paper could not be resolved from custody.

    Raised rather than skipped, and never resolved to a near-miss. Emitting the
    wrong manifestation would produce a DIFFERENT task under the SAME task_id
    and report the regeneration as a success -- the published task set would
    stop reproducing its fingerprint with nothing going red. Skipping instead
    would silently shrink the benchmark, which is the failure
    `framework.holdout.require_corpus` already exists to prevent.
    """


SOURCE_RECORD = "source.json"
_CHUNK = 1 << 20


def repository_root() -> Path:
    """The article repository, from ARTICLE_REPOSITORY.

    Never a hardcoded or guessed path: the repository has moved before, and a
    guessed default named a private directory in the public mirror (2026-10-07).
    Unset means no repository on this machine, which is the same condition as a
    missing index, so it raises the same error.
    """
    explicit = os.environ.get("ARTICLE_REPOSITORY")
    if not explicit:
        raise HeldOutPaperUnresolvable(
            "ARTICLE_REPOSITORY is not set. Set it to the article repository root "
            "to resolve held-out papers."
        )
    return Path(explicit)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def read_source_record(paper_dir: Path) -> dict | None:
    """The paper's custody pointer, or None when the directory has no record."""
    record = Path(paper_dir) / SOURCE_RECORD
    if not record.is_file():
        return None
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HeldOutPaperUnresolvable(
            f"{record} is present but unreadable: {exc}"
        ) from exc
    for field in ("doi", "sha256"):
        if not data.get(field):
            raise HeldOutPaperUnresolvable(f"{record} has no {field!r}")
    return data


def _candidate_paths(entry: dict, repo: Path) -> list[tuple[str | None, Path]]:
    """(recorded sha256 or None, path) for the primary and every manifestation.

    Mirrors `article-finder`'s `_lib.manifestations`, deliberately by reading
    `index.json` rather than importing the skill: the arena must resolve papers
    with nothing but the standard library and a repository path.
    """
    rows: list[tuple[str | None, Path]] = []
    filename = entry.get("filename")
    if filename:
        rows.append((entry.get("sha256"),
                     repo / entry.get("subdir", "fulltext") / filename))
    for man in entry.get("manifestations") or []:
        if man.get("filename"):
            rows.append((man.get("sha256"),
                         repo / man.get("subdir", "fulltext") / man["filename"]))
    return rows


def resolve_pdf(doi: str, sha256: str, repo: Path | None = None) -> Path:
    """The repository file for `doi` whose CONTENT hashes to `sha256`.

    The recorded hash is only ever a prefilter -- the returned file is hashed
    before it is returned. The index is a CLAIM about the store and only the
    store settles it: measured 2026-09-03, grepping `index.json` reported 31 of
    36 papers in custody while hashing the files reported 34.
    """
    repo = repo or repository_root()
    index_path = repo / "index.json"
    if not index_path.is_file():
        raise HeldOutPaperUnresolvable(
            f"no article repository index at {index_path}. Set ARTICLE_REPOSITORY "
            f"to the repository root."
        )
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HeldOutPaperUnresolvable(f"{index_path} is unreadable: {exc}") from exc

    entry = index.get(doi) or index.get(doi.lower())
    if entry is None:
        raise HeldOutPaperUnresolvable(
            f"{doi} is not in the article repository index ({index_path}). "
            f"Take custody first: ingest-local-pdf.py <pdf> --doi {doi}"
        )

    candidates = _candidate_paths(entry, repo)
    # A recorded hash that matches is checked first because it costs nothing;
    # 93.9% of index entries (9,540 of 10,164, measured 2026-09-04) predate the
    # `sha256` field, so an unrecorded hash must be settled against the file
    # rather than read as "not held".
    ordered = ([p for rec, p in candidates if rec == sha256]
               + [p for rec, p in candidates if not rec])
    for path in ordered:
        if path.is_file() and _sha256_file(path) == sha256:
            return path

    held = ", ".join(sorted(p.name for _rec, p in candidates)) or "nothing"
    raise HeldOutPaperUnresolvable(
        f"{doi} is in custody but no manifestation has sha256 {sha256}. "
        f"The repository holds: {held}. Register the exact bytes with "
        f"ingest-local-pdf.py --as-manifestation before regenerating."
    )


def held_out_pdf_bytes(paper_dir: Path, repo: Path | None = None) -> bytes | None:
    """The PDF bytes for one held-out paper directory, or None if there is none.

    `source.json` wins over a local `paper.pdf` whenever it is present. A stale
    local file that drifted from the recorded hash would otherwise silently
    change the task, which is precisely the local-copy ambiguity that moving
    these papers into custody removes.
    """
    paper_dir = Path(paper_dir)
    record = read_source_record(paper_dir)
    if record is None:
        local = paper_dir / "paper.pdf"
        return local.read_bytes() if local.is_file() else None

    expected_size = record.get("size_bytes")
    path = resolve_pdf(record["doi"], record["sha256"], repo=repo)
    data = path.read_bytes()
    # Re-check after reading: `resolve_pdf` hashed the file, this pins the exact
    # bytes handed to the generator, so no read-time race can slip past.
    if _sha256_bytes(data) != record["sha256"]:
        raise HeldOutPaperUnresolvable(
            f"{path} changed between resolution and read for {record['doi']}"
        )
    if expected_size is not None and len(data) != expected_size:
        raise HeldOutPaperUnresolvable(
            f"{paper_dir / SOURCE_RECORD} records size_bytes={expected_size} but "
            f"the content it names is {len(data)} bytes. A record whose two "
            f"facts disagree is not trustworthy for either."
        )
    return data


def has_paper(paper_dir: Path) -> bool:
    """True when this directory declares a paper at all (custody or local).

    Cheap: does not resolve or hash. The loaders use it exactly where they used
    `pdf_path.exists()`, so a directory holding only gold still skips.
    """
    paper_dir = Path(paper_dir)
    return (paper_dir / SOURCE_RECORD).is_file() or (paper_dir / "paper.pdf").is_file()
