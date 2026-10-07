"""Community contributions — the local half (framework/contributions.py).

Spec: docs/design/2026-09-arena-expansion/t5-submissions-spec.md. Every check is
tested from both sides: a case it must pass AND a case it must fail, so a check
that always passes cannot survive here. The AI review is replaced by a stub; the
real headless-Claude run is exercised by `scripts/contributions.py check`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from framework import contributions as C
from framework.publish import NEVER_PUBLISH, _shingles, SHINGLE_STRIDE

REPO = Path(__file__).resolve().parents[2]
FIX = Path(__file__).parent / "fixtures" / "contributions"
GRIM = "grim-consistency-v1"

#: The duplicate and dispute checks compare against every committed task input,
#: which lives in `task_sets/*/_ground_truth.json`. The public mirror never ships
#: those files, so there the checks have nothing to compare against -- an honest
#: skip, not a vacuous pass.
_needs_committed_inputs = pytest.mark.skipif(
    not any((REPO / "arenas" / GRIM / "task_sets").glob("*/_ground_truth.json")),
    reason="no committed _ground_truth.json (public mirror): nothing to check duplicates against",
)


@pytest.fixture()
def cases() -> list[dict]:
    return json.loads((FIX / "grim_cases.json").read_text(encoding="utf-8"))["cases"]


@pytest.fixture()
def golds() -> list[dict]:
    return json.loads((FIX / "grim_golds.json").read_text(encoding="utf-8"))


@pytest.fixture()
def arenas(tmp_path: Path) -> Path:
    """A private copy of the GRIM arena (no run files) to write bundles into."""
    src = REPO / "arenas" / GRIM
    dst = tmp_path / "arenas" / GRIM
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("runs", "__pycache__", "_held_out", ".private_seed"))
    return tmp_path / "arenas"


# --- canonical JSON ---------------------------------------------------------

def test_canonical_json_matches_the_typescript_implementation():
    data = json.loads((REPO / "framework" / "contract" / "fixtures" / "canonical-vectors.json").read_text(encoding="utf-8"))
    assert len(data["vectors"]) >= 5
    for v in data["vectors"]:
        obj = json.loads(v["input"])
        assert C.canonical_json(obj) == v["canonical"]
        assert C.canonical_sha256(obj) == v["sha256"]


@pytest.mark.parametrize("x,js", [
    (1.0, "1"), (0.1, "0.1"), (2.5e-5, "0.000025"), (1e-7, "1e-7"), (1e21, "1e+21"),
    (1e20, "100000000000000000000"), (-0.5, "-0.5"), (-0.0, "0"), (123456789012345678901, "123456789012345680000"),
])
def test_number_layout_is_javascripts(x, js):
    assert C.canonical_json(x) == js


# --- the GRIM fixture really is what it claims -------------------------------

def _grim_ok(value: float, n: int, dec: int, pct: bool) -> bool:
    top = n if pct else n * 10
    return any(round((100 * k if pct else k) / n, dec) == value for k in range(top + 1))


def test_fixture_arithmetic(cases):
    """The fixture's labels must be TRUE, or every test built on it is wrong."""
    for c in cases:
        flags = {r["stat_id"]: r["flagged"] for r in c["expected_output"]["records"]}
        for s in c["input"]["statistics"]:
            pct = s["stat_type"] == "percent"
            ok = _grim_ok(s["percent"] if pct else s["mean"], s["n"], s["decimals"], pct)
            assert flags[s["stat_id"]] is (not ok), s["stat_id"]


# --- checks, two-sided ------------------------------------------------------

def test_schema_check(cases):
    adir = C.arena_dir(GRIM)
    assert C.check_schema(adir, cases)["status"] == "pass"
    broken = json.loads(json.dumps(cases))
    del broken[0]["input"]["statistics"][0]["n"]
    broken[1]["expected_output"]["records"][0]["confidence"] = "high"
    r = C.check_schema(adir, broken)
    assert r["status"] == "fail" and len(r["detail"]) == 2


def test_custody_check_fails_on_heldout_text_and_passes_synthetic(cases):
    fake_article = ("Participants were randomly assigned to one of two conditions and completed the "
                    "manipulation check before the dependent measure was administered by the experimenter. ") * 3
    corpus = (_shingles(fake_article, SHINGLE_STRIDE), 1)
    pasted = [{"input": {"statistics": [{"label": fake_article[20:260]}]}, "expected_output": {}}]
    assert C.check_custody(pasted, corpus)["status"] == "fail"
    assert C.check_custody(cases, corpus)["status"] == "pass"
    # An unrun comparison is never a pass.
    assert C.check_custody(cases, None)["status"] == "warn"


def test_custody_lists_article_identifiers(cases):
    c = json.loads(json.dumps(cases))
    c[0]["input"]["statistics"][0]["label"] = "Liking (see 10.0000/xge0000001)"
    r = C.check_custody(c, (set(), 1))
    assert r["status"] == "warn" and r["detail"]["article_identifiers"]


@_needs_committed_inputs
def test_duplicate_check(cases):
    adir = C.arena_dir(GRIM)
    assert C.check_duplicates(adir, cases)["status"] == "pass"
    existing = next(iter(C.existing_inputs(adir).values()))
    dup = [{"input": existing, "expected_output": {}}]
    assert C.check_duplicates(adir, dup)["status"] == "fail"
    assert C.check_duplicates(adir, [cases[0], cases[0]])["status"] == "fail"


def test_duplicate_check_with_nothing_committed_is_skipped_not_passed(cases, tmp_path):
    """No `_ground_truth.json` (public mirror, fresh clone): the comparison against
    committed tasks cannot run, so it must not read as a pass. A duplicate WITHIN
    the submission is still caught."""
    empty = tmp_path / "arenas" / GRIM
    (empty / "task_sets" / "v1").mkdir(parents=True)
    assert C.check_duplicates(empty, cases)["status"] == "skipped"
    assert C.check_duplicates(empty, [cases[0], cases[0]])["status"] == "fail"


def test_self_consistency(cases, golds):
    adir = C.arena_dir(GRIM)
    chk, scores = C.check_self_consistency(adir, cases, golds)
    assert chk["status"] == "pass" and all(s >= C.SELF_CONSISTENCY_MIN for s in scores)
    wrong = json.loads(json.dumps(golds))
    wrong[0]["records"][1]["flagged"] = True
    wrong[0]["records"][1]["issue_kind"] = "grim_inconsistent"
    assert C.check_self_consistency(adir, cases, wrong)[0]["status"] == "fail"
    assert C.check_self_consistency(adir, cases, [None, None])[0]["status"] == "skipped"


@_needs_committed_inputs
def test_dispute_target():
    adir = C.arena_dir(GRIM)
    tid = next(iter(C.existing_inputs(adir))).split("/", 1)[1]
    assert C.check_dispute_target(adir, tid)["status"] == "pass"
    assert C.check_dispute_target(adir, "no-such-task")["status"] == "fail"


# --- whole-submission triage -------------------------------------------------

def _sub(cases, **over):
    s = {"id": "s1", "kind": "test_cases", "arenaId": GRIM, "title": "t", "payloadSha256": "p",
         "payload": {"cases": cases}, "declaredRelationships": [{"player_family": "scrutiny", "relation": "developer"}]}
    s.update(over)
    return s


def test_triage_uses_ai_drafts_and_never_decides(cases, golds):
    ai = lambda _adir, _sub: {"cases": [{"index": i, "gold_draft": g} for i, g in enumerate(golds)], "rationale": "ok"}
    r = C.triage_submission(_sub(cases), heldout=(set(), 1), ai=ai)
    assert r["summary"]["failed"] == []
    assert [c["gold_draft"] for c in r["cases"]] == golds
    flat = json.dumps(r)
    for word in ('"status": "accepted"', '"decision"', '"bundle"'):
        assert word not in flat


def test_triage_reports_an_ai_failure_instead_of_passing(cases):
    def boom(_a, _s):
        raise RuntimeError("claude returned no output")
    r = C.triage_submission(_sub(cases), heldout=(set(), 1), ai=boom)
    assert "ai_review" in r["summary"]["failed"]
    # Without drafts there is no gold, so self-consistency cannot pass.
    assert r["checks"]["self_consistency"]["status"] == "skipped"


def test_favouring_marks_declared_families_and_skips_on_bad_gold(cases, golds):
    registry = [{"player_id": "p-good", "instrument": "scrutiny-1-0-0", "adapter_class": "StubPassAdapter"},
                {"player_id": "p-bad", "instrument": "othertool-2-0", "adapter_class": "StubPassAdapter"}]

    def fake_favour(adir, cs, gs, declared, *, registry):
        def play(entry, envelope, t):
            idx = int(envelope["task_id"][-2:]) - 1
            return cs[idx]["expected_output"] if entry["player_id"] == "p-good" else {"records": []}
        C_arena = adir
        orig = C.arena_player_ids
        C.arena_player_ids = lambda _a: ["p-good", "p-bad"]
        try:
            return C.favouring(C_arena, cs, gs, declared, registry=registry, play=play,
                               include=["p-good", "p-bad"])
        finally:
            C.arena_player_ids = orig

    ai = lambda _a, _s: {"cases": [{"index": i, "gold_draft": g} for i, g in enumerate(golds)]}
    r = C.triage_submission(_sub(cases), heldout=(set(), 1), ai=ai, favour=fake_favour, registry=registry)
    rows = {p["player_id"]: p for p in r["favouring"]["players"]}
    assert rows["p-good"]["declared_relation"] == "developer" and rows["p-good"]["mean_score"] == 1.0
    assert rows["p-bad"]["declared_relation"] is None and rows["p-bad"]["mean_score"] < 1.0

    bad_ai = lambda _a, _s: {"cases": [{"index": 0, "gold_draft": {"records": []}}, {"index": 1, "gold_draft": {"records": []}}]}
    r2 = C.triage_submission(_sub(cases), heldout=(set(), 1), ai=bad_ai, favour=fake_favour, registry=registry)
    assert r2["checks"]["favouring"]["status"] == "skipped"


# --- export -------------------------------------------------------------------

def _bundle(cases, golds, *, split="revealed", version="v2", orcid=None, display="pseudonymous"):
    sid = "3f2a9c1e-0000-4000-8000-000000000001"
    tasks = [{"task_id": f"contrib-3f2a9c1e-{i + 1:02d}", "input": c["input"], "expected_output": c["expected_output"],
              "gold": g, "note": c.get("note")} for i, (c, g) in enumerate(zip(cases, golds))]
    prov = [{"task_id": t["task_id"], "arena_id": GRIM, "introduced_in": version, "origin": "contributed",
             "basis": {"summary": "s", "refs": []},
             "contributor": {"orcid": orcid, "display": display, "pseudonym": "Contributor 7F3A2C",
                             "declared_relationships": [{"player_family": "*", "relation": "none"}]},
             "review": {"submission_id": sid, "triage_report_sha256": "a" * 64, "moderator": "m",
                        "decision": "accepted", "decided_at": "2026-09-28T12:00:00.000Z"},
             "license": "CC-BY-4.0"} for t in tasks]
    return {"bundle_version": 1, "submission_id": sid, "arena_id": GRIM, "split": split, "introduced_in": version,
            "tasks": tasks, "provenance": prov}


def test_write_revealed_bundle_and_load_for_version(arenas, cases, golds):
    b = _bundle(cases, golds)
    paths = C.write_bundle(b, C.canonical_sha256(b), root=arenas)
    names = sorted(p.name for p in paths)
    assert names == ["bundle.json", "commitment.json", "provenance.jsonl"]
    loaded = C.load_for_version(arenas / GRIM, "v2")
    assert [x["task"]["task_id"] for x in loaded] == [t["task_id"] for t in b["tasks"]]
    assert C.load_for_version(arenas / GRIM, "v3") == []


def test_private_bundle_writes_only_the_commitment(arenas, cases, golds):
    b = _bundle(cases, golds, split="private")
    sha = C.canonical_sha256(b)
    paths = C.write_bundle(b, sha, root=arenas)
    assert [p.name for p in paths] == ["commitment.json"]
    out = arenas / GRIM / "contributed" / b["submission_id"]
    assert not (out / "bundle.json").exists()
    with pytest.raises(FileNotFoundError, match="materialise"):
        C.load_for_version(arenas / GRIM, "v2")
    C.write_private(b, sha, root=arenas)
    assert len(C.load_for_version(arenas / GRIM, "v2")) == 2
    # The private payload path is never mirrored.
    rel = f"arenas/{GRIM}/contributed/{b['submission_id']}/private/bundle.json"
    assert NEVER_PUBLISH.search(rel)
    assert not NEVER_PUBLISH.search(f"arenas/{GRIM}/contributed/{b['submission_id']}/commitment.json")


@pytest.mark.parametrize("mutate,match", [
    (lambda b: b.update(introduced_in="v1") or [p.update(introduced_in="v1") for p in b["provenance"]], "not newer"),
    (lambda b: b["provenance"][0].update(origin="generated"), "provenance|invalid"),
    (lambda b: b["provenance"][0]["review"].update(submission_id="other"), "different submission"),
    (lambda b: b["tasks"][0].update(gold={"records": []}), "self-consistency"),
    (lambda b: b["tasks"][0]["input"]["statistics"][0].pop("n"), "schema"),
    (lambda b: b["provenance"].pop(), "one provenance record per task"),
])
def test_exporter_refusals(arenas, cases, golds, mutate, match):
    b = _bundle(cases, golds)
    mutate(b)
    with pytest.raises((C.BundleRefused, ValueError), match=match):
        C.write_bundle(b, C.canonical_sha256(b), root=arenas)
    assert not (arenas / GRIM / "contributed").exists()


def test_exporter_refuses_a_hash_that_is_not_the_commitment(arenas, cases, golds):
    b = _bundle(cases, golds)
    with pytest.raises(C.BundleRefused, match="does not match"):
        C.write_bundle(b, "0" * 64, root=arenas)


def test_load_for_version_refuses_a_tampered_bundle(arenas, cases, golds):
    b = _bundle(cases, golds)
    C.write_bundle(b, C.canonical_sha256(b), root=arenas)
    f = arenas / GRIM / "contributed" / b["submission_id"] / "bundle.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    data["tasks"][0]["gold"]["records"][0]["flagged"] = False
    f.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="commitment"):
        C.load_for_version(arenas / GRIM, "v2")


# --- review findings 2026-09-30: identity in bundles, private-split egress -----

ORCID = "0000-0002-1825-0097"


def test_exporter_refuses_an_orcid_in_a_pseudonymous_bundle(arenas, cases, golds):
    """Defence in depth: an ORCID the contributor did not agree to publish never reaches a tracked file."""
    b = _bundle(cases, golds, orcid=ORCID, display="pseudonymous")
    with pytest.raises(C.BundleRefused, match="ORCID"):
        C.write_bundle(b, C.canonical_sha256(b), root=arenas)
    assert not (arenas / GRIM / "contributed").exists()
    # Control: an identified contributor's ORCID is exported as before.
    ok = _bundle(cases, golds, orcid=ORCID, display="identified")
    assert [p.name for p in C.write_bundle(ok, C.canonical_sha256(ok), root=arenas)][0] == "commitment.json"


def _favour_capturing(seen: list, include: list[str]):
    def favour(adir, cs, gs, declared, *, registry, **kw):
        def play(entry, envelope, t):
            seen.append((entry["player_id"], envelope))
            return cs[int(envelope["task_id"][-2:]) - 1]["expected_output"]
        orig = C.arena_player_ids
        C.arena_player_ids = lambda _a: []
        try:
            return C.favouring(adir, cs, gs, declared, registry=registry, play=play, include=include, **kw)
        finally:
            C.arena_player_ids = orig
    return favour


_REG = [{"player_id": "p-local", "instrument": "docpluck-2-4", "adapter_class": "DocpluckLibraryAdapter"},
        {"player_id": "p-cloud", "instrument": "claude-opus", "adapter_class": "LlmCliAdapter"}]


def _drafts(golds):
    return lambda _a, _s: {"cases": [{"index": i, "gold_draft": g} for i, g in enumerate(golds)]}


@pytest.mark.parametrize("field", ["requestedSplit", "split"])
def test_private_split_favouring_envelope_is_held_out(cases, golds, field):
    from framework.holdout import HELD_OUT
    seen: list = []
    C.triage_submission(_sub(cases, **{field: "private"}), heldout=(set(), 1), ai=None, golds=golds,
                        favour=_favour_capturing(seen, ["p-local"]), registry=_REG)
    assert seen, "the local player should still run on a private-split case"
    assert all(env["visibility"] == HELD_OUT and env["split"] == "private" for _, env in seen), seen[0][1]


def test_public_split_favouring_is_unchanged(cases, golds):
    seen: list = []
    r = C.triage_submission(_sub(cases, requestedSplit="revealed"), heldout=(set(), 1), ai=None, golds=golds,
                            favour=_favour_capturing(seen, ["p-local", "p-cloud"]), registry=_REG)
    assert {pid for pid, _ in seen} == {"p-local", "p-cloud"}
    assert all(env["visibility"] == "public" and env["split"] == "revealed" for _, env in seen)
    assert r["checks"]["favouring"]["status"] == "pass"


def test_private_split_refuses_cloud_players(cases, golds):
    seen: list = []
    with pytest.raises(C.CloudEgressRefused, match="p-cloud"):
        C.triage_submission(_sub(cases, requestedSplit="private"), heldout=(set(), 1), ai=None, golds=golds,
                            favour=_favour_capturing(seen, ["p-local", "p-cloud"]), registry=_REG)
    assert not [pid for pid, _ in seen if pid == "p-cloud"], "a cloud player was sent a private-split case"


def test_private_split_skips_ai_review_and_says_why(cases, golds):
    calls: list = []

    def ai(adir, sub):
        calls.append(sub["id"])
        return _drafts(golds)(adir, sub)

    r = C.triage_submission(_sub(cases, requestedSplit="private"), heldout=(set(), 1), ai=ai)
    assert calls == [], "ai_review sent a private-split case to a cloud model"
    assert r["checks"]["ai_review"] == {"status": "skipped", "detail": "skipped: private split — no cloud egress"}
    assert "ai_review" in r["summary"]["warned"] and "ai_review" not in r["summary"]["failed"]
    # Control: a revealed-split submission is still reviewed.
    C.triage_submission(_sub(cases, requestedSplit="revealed"), heldout=(set(), 1), ai=ai)
    assert calls == ["s1"]


@pytest.mark.skipif(not (REPO / "scripts" / "contributions.py").is_file(),
                    reason="scripts/ is private maintainer tooling, not in the public mirror")
def test_cli_triage_exits_nonzero_naming_cloud_players_for_a_private_split(cases, golds, monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("contrib_cli", REPO / "scripts" / "contributions.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    posted: list = []

    def api(method, path, body=None):
        if method == "GET":
            return {"submissions": [_sub(cases, requestedSplit="private", id="priv-1")]}
        posted.append(path)
        return {}

    monkeypatch.setattr(cli, "_api", api)
    monkeypatch.setattr(cli, "_heldout", lambda: (set(), 1))
    monkeypatch.setattr(cli, "_registry", lambda: _REG)
    monkeypatch.setattr(cli.C, "ai_review", _drafts(golds))
    monkeypatch.setattr(cli.C, "arena_player_ids", lambda _a: [])
    rc = cli.main(["triage", "--players", "p-local", "p-cloud"])
    out = capsys.readouterr().out
    assert rc != 0
    assert "p-cloud" in out and "priv-1" in out
    assert posted == [], "a refused triage must not be posted"
