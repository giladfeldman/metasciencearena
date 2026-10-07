"""Task provenance sidecar: the schema is loaded and enforced, and the public projection leaks nothing.

Field names are fixed by docs/design/2026-09-arena-expansion/task-provenance.md. The
leak test matters most: `injected_issues` is the answer key for a private-split task.
"""
from __future__ import annotations

import copy
import json

import pytest

from framework import provenance


def _generated() -> dict:
    return {
        "task_id": "t-0001",
        "arena_id": "participant-flow-consistency-v1",
        "introduced_in": "v1",
        "origin": "generated",
        "generator": {
            "module": "arenas/participant-flow-consistency-v1/generator.py",
            "generator_fingerprint": "sha256:" + "a" * 64,
            "split": "private",
            "realism_profile": None,
        },
        "basis": {
            "summary": "CONSORT flow where randomised minus lost does not equal analysed.",
            "refs": [{"kind": "doi", "value": "10.1136/bmj-2026-100611", "note": None}],
        },
        "injected_issues": [{"catalog": "flow", "id": "arithmetic-gap", "location": "fig1"}],
        "license": "CC-BY-4.0",
    }


def _contributed() -> dict:
    r = _generated()
    del r["generator"]
    r["origin"] = "contributed"
    r["contributor"] = {
        "orcid": "0000-0002-1825-0097",
        "display": "pseudonymous",
        "pseudonym": "Contributor 7F3A",
        "declared_relationships": [{"player_family": "docpluck", "relation": "competitor"}],
    }
    r["review"] = {
        "submission_id": "sub-1",
        "triage_report_sha256": "b" * 64,
        "moderator": "gilad",
        "decision": "accepted",
        "decided_at": "2026-09-28T10:00:00Z",
    }
    return r


def test_valid_generated_and_contributed_records_pass():
    provenance.validate(_generated())
    provenance.validate(_contributed())


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda r: r.pop("generator"), "generator"),
        (lambda r: r.update(origin="made-up"), "origin"),
        (lambda r: r.update(introduced_in="1"), "introduced_in"),
        (lambda r: r.update(license="MIT"), "license"),
        (lambda r: r.update(answer="leak"), "answer"),
        (lambda r: r["basis"]["refs"].append({"kind": "fulltext", "value": "x"}), "kind"),
        (lambda r: r["generator"].update(generator_fingerprint="abc"), "generator_fingerprint"),
    ],
)
def test_invalid_records_are_rejected(mutate, needle):
    r = _generated()
    mutate(r)
    with pytest.raises(ValueError, match="invalid provenance"):
        provenance.validate(r)


def test_contributed_requires_contributor_and_review():
    r = _contributed()
    del r["review"]
    with pytest.raises(ValueError):
        provenance.validate(r)


def test_generated_record_cannot_carry_a_contributor():
    r = _generated()
    r["contributor"] = _contributed()["contributor"]
    with pytest.raises(ValueError):
        provenance.validate(r)


def test_public_projection_never_carries_the_answer_key():
    for rec in (_generated(), _contributed()):
        text = json.dumps(provenance.public_projection(rec))
        for secret in ("arithmetic-gap", "injected_issues", "fig1", "generator_fingerprint",
                       "private", "10.1136/bmj-2026-100611", "triage_report_sha256"):
            assert secret not in text, secret


def test_public_projection_hides_orcid_unless_identified():
    pseud = _contributed()
    assert provenance.public_projection(pseud)["contributor"]["orcid"] is None
    ident = copy.deepcopy(pseud)
    ident["contributor"]["display"] = "identified"
    assert provenance.public_projection(ident)["contributor"]["orcid"] == "0000-0002-1825-0097"


def test_write_jsonl_round_trips_and_refuses_bad_input(tmp_path):
    a, b = _generated(), _generated()
    b["task_id"] = "t-0002"
    out = tmp_path / "prov.jsonl"
    assert provenance.write_jsonl([a, b], out) == 2
    assert [json.loads(l)["task_id"] for l in out.read_text(encoding="utf-8").splitlines()] == ["t-0001", "t-0002"]

    with pytest.raises(ValueError, match="empty"):
        provenance.write_jsonl([], tmp_path / "empty.jsonl")
    assert not (tmp_path / "empty.jsonl").exists()

    with pytest.raises(ValueError, match="duplicate"):
        provenance.write_jsonl([a, copy.deepcopy(a)], tmp_path / "dup.jsonl")

    bad = copy.deepcopy(b)
    bad["origin"] = "nope"
    with pytest.raises(ValueError):
        provenance.write_jsonl([a, bad], tmp_path / "bad.jsonl")
    assert not (tmp_path / "bad.jsonl").exists()
