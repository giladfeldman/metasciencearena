"""Held-out PDFs resolve from the article repository, by DOI *and* content hash.

Written 2026-09-04, RED first, to close the last step of the custody move:
article-finder is the portfolio's sole custodian of publications, and these PDFs
were the last publication bytes ScienceArena held in its own tree.

The reason resolution must pin the CONTENT HASH and not merely the DOI:
`_real_pmc_tasks` / `_real_apa_tasks` base64 `paper.pdf`'s bytes straight INTO
the task envelope, and those bytes are not persisted (every
`document_bytes_b64` in `task_sets/v1/_ground_truth.json` is the literal string
`<pdf_bytes_omitted>`). The PDF bytes ARE the task input. Two files can be the
same paper under the same DOI and still be different tasks: measured on one of
the APA held-out papers, the repository's copy and the arena's copy were both
231,529 bytes with byte-identical extracted text (92,169 chars) and different
sha256, so size, filename and DOI checks all passed a pair that differed.
Resolving that DOI to "a PDF of that paper" would silently change a published
task set and break its fingerprint.

So `source.json` records the hash, and a mismatch is an error rather than a
substitution.

THIS FILE IS MIRRORED PUBLICLY (`framework/**/*.py`), so the DOIs below are
SYNTHETIC. Naming a real held-out paper here would leak membership, which is a
step toward the contamination the split exists to prevent — the same rule
`test_public_mirror_names_no_heldout_paper` enforces for PMC accessions. The
real identifiers and byte counts live with the measurement, in the private
article-finder repo (`tests/test_ingest_manifestations.py`).
"""

import hashlib
import json

import pytest

from framework import article_custody

#: SYNTHETIC. `10.0000/...` is not an assigned registrant prefix, so this cannot
#: collide with a real paper — the DOI analogue of the leading-zero `PMC0…`
#: placeholder convention already used across this suite.
DOI = "10.0000/synthetic.manifestation.demo"
OTHER_DOI = "10.0000/synthetic.not.held"

PRIMARY = b"%PDF-1.7\nthe repository's own copy\n"
ARENA = b"%PDF-1.7\nthe arena's copy, same paper, different bytes\n"


def _write_repo(tmp_path, entries):
    """Build a minimal ArticleRepository: fulltext/ + index.json."""
    repo = tmp_path / "ArticleRepository"
    (repo / "fulltext").mkdir(parents=True)
    index = {}
    for doi, files in entries.items():
        primary, *rest = files
        stem = doi.replace("/", "__")
        (repo / "fulltext" / f"{stem}.pdf").write_bytes(primary)
        index[doi] = {
            "tier": 1,
            "format": "pdf",
            "filename": f"{stem}.pdf",
            "subdir": "fulltext",
            "sha256": hashlib.sha256(primary).hexdigest(),
            "size_bytes": len(primary),
        }
        mans = []
        for body in rest:
            sha = hashlib.sha256(body).hexdigest()
            name = f"{stem}~{sha[:16]}.pdf"
            (repo / "fulltext" / name).write_bytes(body)
            mans.append({"sha256": sha, "filename": name, "subdir": "fulltext",
                         "size_bytes": len(body)})
        if mans:
            index[doi]["manifestations"] = mans
    (repo / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return repo


def _paper_dir(tmp_path, name, doi, body):
    d = tmp_path / "held_out" / name
    d.mkdir(parents=True)
    (d / "source.json").write_text(json.dumps({
        "doi": doi,
        "sha256": hashlib.sha256(body).hexdigest(),
        "size_bytes": len(body),
    }), encoding="utf-8")
    return d


def test_resolves_the_primary_manifestation(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path, {DOI: [PRIMARY]})
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(repo))
    d = _paper_dir(tmp_path, "paper_a", DOI, PRIMARY)
    assert article_custody.held_out_pdf_bytes(d) == PRIMARY


def test_resolves_the_second_manifestation_not_the_primary(tmp_path, monkeypatch):
    """The whole point: same DOI, and the arena's exact bytes come back."""
    repo = _write_repo(tmp_path, {DOI: [PRIMARY, ARENA]})
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(repo))
    d = _paper_dir(tmp_path, "paper_a", DOI, ARENA)
    got = article_custody.held_out_pdf_bytes(d)
    assert got == ARENA
    assert got != PRIMARY


def test_a_hash_the_repository_does_not_hold_raises(tmp_path, monkeypatch):
    """Never substitute. A missing manifestation must stop the generator.

    Emitting the primary instead would produce a DIFFERENT task under the SAME
    task_id and report the regeneration as a success -- the published task set
    would stop reproducing its fingerprint with nothing going red.
    """
    repo = _write_repo(tmp_path, {DOI: [PRIMARY]})
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(repo))
    d = _paper_dir(tmp_path, "paper_a", DOI, ARENA)
    with pytest.raises(article_custody.HeldOutPaperUnresolvable) as e:
        article_custody.held_out_pdf_bytes(d)
    assert DOI in str(e.value)


def test_a_doi_absent_from_the_index_raises(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path, {DOI: [PRIMARY]})
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(repo))
    d = _paper_dir(tmp_path, "paper_b", OTHER_DOI, PRIMARY)
    with pytest.raises(article_custody.HeldOutPaperUnresolvable):
        article_custody.held_out_pdf_bytes(d)


def test_corrupted_repository_bytes_are_caught(tmp_path, monkeypatch):
    """The index is a CLAIM about the store; only the store settles it.

    Measured 2026-09-03: grepping index.json said 31 of 36 papers were in
    custody; hashing the files said 34. A resolver that trusts the record alone
    would hand the generator whatever is on disk under that name.
    """
    repo = _write_repo(tmp_path, {DOI: [PRIMARY]})
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(repo))
    stem = DOI.replace("/", "__")
    (repo / "fulltext" / f"{stem}.pdf").write_bytes(b"%PDF-1.7\nsomething else\n")
    d = _paper_dir(tmp_path, "paper_a", DOI, PRIMARY)
    with pytest.raises(article_custody.HeldOutPaperUnresolvable):
        article_custody.held_out_pdf_bytes(d)


def test_a_local_paper_pdf_is_used_when_there_is_no_source_json(tmp_path, monkeypatch):
    """The fetcher still writes a bare paper.pdf; that path must keep working.

    `tools/fetch_pmc.py` drops a new paper into the held-out tree with no
    `source.json`. Requiring one unconditionally would break fetching new
    papers, which is how the corpus grows.
    """
    repo = _write_repo(tmp_path, {DOI: [PRIMARY]})
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(repo))
    d = tmp_path / "held_out" / "paper_fresh"
    d.mkdir(parents=True)
    (d / "paper.pdf").write_bytes(b"%PDF-1.7\nfreshly fetched\n")
    assert article_custody.held_out_pdf_bytes(d) == b"%PDF-1.7\nfreshly fetched\n"


def test_source_json_wins_over_a_stale_local_pdf(tmp_path, monkeypatch):
    """Custody is the source of truth once it has been declared.

    A leftover `paper.pdf` that drifted from the recorded hash would otherwise
    silently win and change the task -- reintroducing exactly the local-copy
    ambiguity this move exists to remove.
    """
    repo = _write_repo(tmp_path, {DOI: [PRIMARY, ARENA]})
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(repo))
    d = _paper_dir(tmp_path, "paper_a", DOI, ARENA)
    (d / "paper.pdf").write_bytes(b"%PDF-1.7\na stale local copy\n")
    assert article_custody.held_out_pdf_bytes(d) == ARENA


def test_no_pdf_and_no_source_json_returns_none(tmp_path, monkeypatch):
    """An empty directory is 'no task here', not an error -- the loaders skip."""
    repo = _write_repo(tmp_path, {DOI: [PRIMARY]})
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(repo))
    d = tmp_path / "held_out" / "empty"
    d.mkdir(parents=True)
    assert article_custody.held_out_pdf_bytes(d) is None


def test_the_recorded_size_must_agree_with_the_recorded_hash(tmp_path, monkeypatch):
    """A source.json whose two facts disagree is not trustworthy for either."""
    repo = _write_repo(tmp_path, {DOI: [PRIMARY]})
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(repo))
    d = _paper_dir(tmp_path, "paper_a", DOI, PRIMARY)
    rec = json.loads((d / "source.json").read_text(encoding="utf-8"))
    rec["size_bytes"] = rec["size_bytes"] + 1
    (d / "source.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(article_custody.HeldOutPaperUnresolvable):
        article_custody.held_out_pdf_bytes(d)
