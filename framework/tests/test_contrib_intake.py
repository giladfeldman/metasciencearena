"""Intake-v2 triage (framework/contrib_intake.py, framework/contributions.py::_triage_intake_v2).

Fixtures are SYNTHETIC rows under the real header names of the RegCheck validation
study (github.com/beth099/RegCheck-Validation-Study, headers read 2026-10-16) — no
article text, no real coder judgements. Each rule is tested from both sides.
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from framework import contrib_intake as CI
from framework import contributions as C

DISAGREEMENTS_HEADER = ("coder_pair,article_num,doi_link,registration_url,study_num,dimension,prereg_c1,paper_c1,"
                        "deviation_c1,concerns_c1,conf_c1,comments_c1,dev_text_c1,prereg_c2,paper_c2,deviation_c2,"
                        "concerns_c2,conf_c2,comments_c2,dev_text_c2,final_deviation_decision")

QUOTE = "We will recruit participants until we reach the planned sample, and the analysis will " * 5  # > LONG_CELL


def _disagreements_csv(n: int = 6) -> bytes:
    labels = ["Yes", "No", "Mixed", "Too vague or not reported"]
    rows = [DISAGREEMENTS_HEADER]
    for i in range(n):
        lab = labels[i % 4]
        rows.append(",".join([
            "A-B", str(i), f"https://doi.org/10.0000/synthetic{i}", f"https://osf.io/abc{i}", "1", "hypothesis",
            f'"{QUOTE}"', f'"{QUOTE}"', lab, "no significant", "very confident", "", f'"{QUOTE}"',
            f'"{QUOTE}"', f'"{QUOTE}"', lab, "no significant", "confident", "", f'"{QUOTE}"', lab,
        ]))
    return ("\n".join(rows) + "\n").encode("utf-8")


class TestFileInventory:
    def test_quote_columns_are_dropped_and_never_echoed(self):
        inv = CI.inventory_file("disagreements_to_resolve.csv", _disagreements_csv())
        assert inv["status"] == "read" and inv["rows"] == 6
        for col in ("prereg_c1", "paper_c1", "dev_text_c1", "prereg_c2", "paper_c2", "dev_text_c2"):
            assert col in inv["dropped_quote_columns"]
            assert col not in inv["columns"]
        assert "We will recruit" not in json.dumps(inv)

    def test_label_columns_are_kept_with_their_vocabulary(self):
        inv = CI.inventory_file("d.csv", _disagreements_csv())
        assert set(inv["small_vocabularies"]["final_deviation_decision"]) == {"Yes", "No", "Mixed", "Too vague or not reported"}

    def test_a_long_cell_under_an_innocent_header_is_flagged_not_echoed(self):
        data = ("id,notes\n1," + '"' + QUOTE + '"' + "\n").encode()
        inv = CI.inventory_file("x.csv", data)
        assert inv["long_text_columns"] == {"notes": 1}
        assert "notes" not in inv["small_vocabularies"]
        assert "We will recruit" not in json.dumps(inv)

    def test_zip_bomb_guard_refuses_before_parsing(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("xl/huge.bin", b"\0" * (CI.MAX_ZIP_UNCOMPRESSED + 1))
        assert CI.inventory_file("bomb.xlsx", buf.getvalue())["status"] == "refused"

    def test_a_broken_file_is_a_finding_not_a_crash(self):
        assert CI.inventory_file("bad.json", b"{not json")["status"] == "unreadable"


class TestLinks:
    def _fetch(self, url: str) -> bytes:
        if url == "https://api.github.com/repos/beth099/RegCheck-Validation-Study":
            return json.dumps({"default_branch": "main", "license": {"spdx_id": "MIT"}}).encode()
        if url.startswith("https://api.github.com/repos/beth099/RegCheck-Validation-Study/git/trees/main"):
            return json.dumps({"tree": [
                {"path": "Sample_selection/feed_to_regcheck/MASTER_list.csv", "type": "blob"},
                {"path": "Analyses/disagreement_spreadsheet/disagreements_to_resolve.csv", "type": "blob"},
                {"path": "Analyses/OSF_preregs/x.pdf", "type": "blob"},
                {"path": "README.md", "type": "blob"},
            ]}).encode()
        raise AssertionError(f"unexpected fetch {url}")

    def test_github_repo_inventory_lists_data_files_and_licence(self):
        inv = CI.inventory_link("https://github.com/beth099/RegCheck-Validation-Study", self._fetch)
        assert inv["status"] == "read" and inv["licence"] == "MIT"
        assert "Sample_selection/feed_to_regcheck/MASTER_list.csv" in inv["data_files"]
        assert inv["pdfs_in_repo"] == 1

    def test_linked_spreadsheets_are_read_in_memory_answer_files_first(self):
        inv = CI.inventory_link("https://github.com/beth099/RegCheck-Validation-Study", self._fetch)
        fetched: list[str] = []

        def fetch(url: str) -> bytes:
            fetched.append(url)
            return _disagreements_csv()

        files = CI.inventory_linked_files(inv, fetch)
        assert [f["name"] for f in files][0].endswith("disagreements_to_resolve.csv")  # 'resolve' hint ranks first
        assert all(u.startswith("https://raw.githubusercontent.com/beth099/RegCheck-Validation-Study/main/") for u in fetched)
        assert all(not u.endswith(".pdf") for u in fetched)
        assert "We will recruit" not in json.dumps(files)

    def test_hosts_off_the_allow_list_are_never_fetched(self):
        with pytest.raises(PermissionError):
            CI.http_get("https://example.org/data.csv")
        out = CI.inventory_link("https://drive.example.org/folder", lambda u: (_ for _ in ()).throw(AssertionError("fetched")))
        assert out["status"] == "skipped"

    def test_a_doi_is_noted_for_local_custody_not_fetched(self):
        out = CI.inventory_link("https://doi.org/10.0000/synthetic-doi", lambda u: (_ for _ in ()).throw(AssertionError("fetched")))
        assert out["kind"] == "doi" and "local article custody" in out["note"]


class TestMergeContributed:
    """`python -m framework contributed add <arena> <version>`: explicit, new versions only."""

    def _arena(self, tmp_path, name="demo-v1", status="current", with_runs=False):
        import yaml

        adir = tmp_path / name
        (adir / "task_sets" / "v2").mkdir(parents=True)
        (adir / "arena.yaml").write_text(yaml.safe_dump({"arena_id": name, "task_set_versions": [
            {"version": "v1", "status": "frozen"}, {"version": "v2", "status": status}]}), encoding="utf-8")
        (adir / "task_sets" / "v2" / "_ground_truth.json").write_text(json.dumps({"gen-1": {"envelope": {}, "ground_truth": {}}}), encoding="utf-8")
        bundle = {"bundle_version": 1, "submission_id": "s1", "arena_id": name, "split": "revealed", "introduced_in": "v2",
                  "tasks": [{"task_id": "contrib-aaaaaaaa-01", "input": {"x": 1}, "expected_output": {"y": 1}, "gold": {"y": 1}, "note": None}],
                  "provenance": [{"task_id": "contrib-aaaaaaaa-01", "origin": "contributed"}]}
        cdir = adir / "contributed" / "s1"
        cdir.mkdir(parents=True)
        (cdir / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
        (cdir / "commitment.json").write_text(json.dumps({"submission_id": "s1", "arena_id": name, "introduced_in": "v2",
                                                          "split": "revealed", "bundle_sha256": C.canonical_sha256(bundle)}), encoding="utf-8")
        if with_runs:
            (adir / "runs" / "v2").mkdir(parents=True)
            (adir / "runs" / "v2" / "p.jsonl").write_text("{}\n", encoding="utf-8")
        return adir

    def test_adds_to_a_new_version_with_provenance_and_a_fingerprint(self, tmp_path):
        import yaml

        adir = self._arena(tmp_path)
        out = C.merge_contributed(adir, "v2")
        assert out["added"] == 1 and out["written"]
        gt = json.loads((adir / "task_sets" / "v2" / "_ground_truth.json").read_text(encoding="utf-8"))
        assert gt["contrib-aaaaaaaa-01"]["envelope"]["difficulty"] == {"origin": "contributed"}
        assert "gen-1" in gt  # the generator's tasks are untouched
        assert (adir / "task_sets" / "v2" / "provenance.jsonl").read_text(encoding="utf-8").count("contributed") == 1
        man = yaml.safe_load((adir / "task_sets" / "v2" / "manifest.yaml").read_text(encoding="utf-8"))
        assert man["contributed"]["n"] == 1 and len(man["contributed"]["sha256"]) == 64

    def test_dry_run_writes_nothing(self, tmp_path):
        adir = self._arena(tmp_path)
        before = (adir / "task_sets" / "v2" / "_ground_truth.json").read_text(encoding="utf-8")
        assert C.merge_contributed(adir, "v2", dry_run=True)["written"] is False
        assert (adir / "task_sets" / "v2" / "_ground_truth.json").read_text(encoding="utf-8") == before

    @pytest.mark.parametrize("kw,version,needle", [
        ({"status": "frozen"}, "v2", "frozen"),
        ({"with_runs": True}, "v2", "run records"),
        ({}, "v9", "not in"),
        ({"name": "prereg-deviation-v1"}, "v2", "study"),
    ])
    def test_refusals(self, tmp_path, kw, version, needle):
        adir = self._arena(tmp_path, **kw)
        with pytest.raises(C.MergeRefused, match=needle):
            C.merge_contributed(adir, version)

    def test_a_second_run_refuses_rather_than_duplicating(self, tmp_path):
        adir = self._arena(tmp_path)
        C.merge_contributed(adir, "v2")
        with pytest.raises(C.MergeRefused, match="already in"):
            C.merge_contributed(adir, "v2")


class TestDatasetMapping:
    def test_maps_yes_no_too_vague_and_leaves_mixed_unmapped(self):
        inv = CI.inventory_file("disagreements_to_resolve.csv", _disagreements_csv())
        m = CI.dataset_mapping([inv])
        assert m["column"] == "final_deviation_decision"  # preferred over deviation_c1/_c2 on a tie
        assert m["mapped"] == {"Yes": "deviation", "No": "no_deviation", "Too vague or not reported": "insufficient_evidence"}
        assert m["unmapped"] == ["Mixed"]

    def test_maps_their_dimension_codes_and_leaves_transformations_for_a_person(self):
        rows = [DISAGREEMENTS_HEADER]
        for i, (dim, lab) in enumerate([("hypothesis", "Yes"), ("samplesize", "No"), ("transf", "Yes"), ("inex", "No")]):
            rows.append(",".join(["A-B", str(i), "", "", "1", dim, "", "", lab, "", "", "", "", "", "", lab, "", "", "", "", ""]))
        inv = CI.inventory_file("d.csv", ("\n".join(rows) + "\n").encode())
        m = CI.dataset_mapping([inv])
        assert m["dimension_map"]["mapped"] == {"hypothesis": "hypotheses", "samplesize": "sample_size", "inex": "data_exclusion"}
        assert m["dimension_map"]["unmapped"] == ["transf"]
        assert "inex" in m["dimension_map"]["approximate"]
        # final_deviation_decision is empty in every row: say so, and say which column was used instead.
        assert "final_deviation_decision is empty" in m["note"]

    def test_no_mapping_when_no_column_speaks_the_vocabulary(self):
        inv = CI.inventory_file("x.csv", b"a,b\n1,2\n3,4\n")
        assert CI.dataset_mapping([inv]) is None


def _runner(reply: dict):
    seen: list[str] = []

    def run(prompt: str) -> str:
        seen.append(prompt)
        return "Here you go:\n" + json.dumps(reply)
    run.seen = seen  # type: ignore[attr-defined]
    return run


# prereg-deviation-v1 is a private arena: the public mirror ships neither its tasks nor its key.
_needs_prereg_arena = pytest.mark.skipif(
    not (C.arena_dir("prereg-deviation-v1") / "task_sets" / "v1" / "_ground_truth.json").exists(),
    reason="prereg-deviation-v1 is private, not in the public mirror")


class TestTriageIntakeV2:
    @_needs_prereg_arena
    def test_feedback_triage_targets_the_test_and_returns_suggested_reasons(self, tmp_path):
        sub = {"id": "s1", "kind": "stimulus_feedback", "arenaId": "prereg-deviation-v1", "taskId": "pd-t1-0-s0",
               "payloadSha256": "p", "payload": {"category": "unrealistic", "comment": "Too short to be a real preregistration.",
                                                "anchor": {"block": "input", "quote": "We recruited 150 participants."}}}
        run = _runner({"summary": "s", "classification": "unrealistic", "is_it_right": "yes", "proposed_action": "longer preregs",
                       "suggested_reason": {"accept": "Thank you — you are right.", "reject": "Thank you, but…", "bogus": "x"}})
        report = C.triage_submission(sub, heldout=None, runner=run)
        assert report["checks"]["flag_target"]["status"] == "pass"
        assert report["checks"]["ai_review"]["status"] == "pass"
        assert set(report["suggested_reason"]) == {"accept", "reject"}
        assert "<feedback>" in run.seen[0] and "Too short" in run.seen[0]
        assert "never" not in report["summary"]["verdict"]

    def test_feedback_on_a_missing_test_fails_the_target_check(self):
        sub = {"id": "s1", "kind": "stimulus_feedback", "arenaId": "prereg-deviation-v1", "taskId": "no-such-task",
               "payload": {"category": "other", "comment": "something long enough"}}
        report = C.triage_submission(sub, heldout=None, ai=None)
        assert report["checks"]["flag_target"]["status"] == "fail"
        assert report["checks"]["ai_review"]["status"] == "skipped"

    def test_material_triage_counts_files_and_maps_the_dataset(self):
        sub = {"id": "s2", "kind": "test_material", "arenaId": "prereg-deviation-v1",
               "payload": {"description": "coded deviations", "links": [], "files": [{"filename": "d.csv"}]}}
        report = C.triage_submission(sub, heldout=None, ai=None, files=[("d.csv", _disagreements_csv())])
        assert report["checks"]["files"]["status"] == "pass"
        assert report["dataset_mapping"]["unmapped"] == ["Mixed"]
        assert "We will recruit" not in json.dumps(report)

    @_needs_prereg_arena
    def test_material_is_translated_and_the_cases_get_the_test_case_checks(self):
        # Built from a real public prereg-deviation-v1 test so the schema check can pass.
        gt = json.loads((C.arena_dir("prereg-deviation-v1") / "task_sets" / "v1" / "_ground_truth.json").read_text(encoding="utf-8"))
        env = next(v for v in gt.values() if v["envelope"].get("visibility") == "public")
        case_input = env["envelope"]["input"]
        dims = case_input["dimensions"]
        expected = {"deviations": [{"dimension": d, "deviation": False, "deviation_kind": None, "confidence": 0.9} for d in dims]}
        prompts: list[str] = []

        def run(prompt: str) -> str:
            prompts.append(prompt)
            if "<input_schema>" in prompt:  # the translation call
                return json.dumps({"cases": [{"input": case_input, "expected_output": expected, "note": "no deviations", "source": "row 1"}],
                                   "restatement": "We read your material as one case with no deviations.", "not_translated": ""})
            if "<scorer>" in prompt:  # ai_review: gold drafts for the translated case
                return json.dumps({"cases": [{"index": 0, "gold_draft": env["ground_truth"], "agrees_with_expected_output": True,
                                              "unambiguous": True, "ambiguity": ""}], "looks_synthetic": True, "rationale": "fine"})
            return json.dumps({"summary": "s", "usable_as": "synthetic test cases", "what_we_still_need": [], "suggested_reason": {"accept": "Thanks!"}})

        sub = {"id": "s3", "kind": "test_material", "arenaId": "prereg-deviation-v1", "requestedSplit": "revealed",
               "payload": {"description": "one made-up case", "links": [], "files": [{"filename": "c.csv"}]}}
        report = C.triage_submission(sub, heldout=None, files=[("c.csv", b"case,notes\n1,made up\n")], runner=run)
        assert report["checks"]["translation"]["status"] == "pass"
        assert report["translation"]["cases"][0]["source"] == "row 1"
        assert report["checks"]["schema"]["status"] == "pass"
        assert report["checks"]["self_consistency"]["status"] in ("pass", "fail")  # it RAN on the translated case
        assert report["cases"][0]["gold_draft"] == env["ground_truth"]
        assert sub["payload"] == {"description": "one made-up case", "links": [], "files": [{"filename": "c.csv"}]}  # payload untouched
        assert any("<material>" in p and "made up" in p for p in prompts)

    def test_a_translated_perfect_answer_is_certain(self):
        """Measured 2026-10-17: confidence 0.95 in an expected output capped the perfect
        answer at 0.95 and failed self-consistency on a correct case."""
        a = C.arena_dir("prereg-deviation-v1")
        case = {"input": {"paper": "We recruited 154.", "dimensions": ["sample_size"], "preregistration": "We will recruit 200."},
                "expected_output": CI._certain({"deviations": [{"deviation": True, "dimension": "sample_size", "confidence": 0.95,
                                                                "deviation_kind": "sample_size_change"}]})}
        gold = {"dimensions": [{"deviation": True, "dimension": "sample_size", "deviation_kind": "sample_size_change"}]}
        check, scores = C.check_self_consistency(a, [case], [gold])
        assert check["status"] == "pass" and scores == [1.0]
        assert CI._certain({"a": [{"confidence": 0.2, "x": 0.2}]}) == {"a": [{"confidence": 1.0, "x": 0.2}]}

    def test_private_split_material_is_never_sent_for_translation(self):
        def run(prompt: str) -> str:
            raise AssertionError("a private-split submission reached the cloud model")

        sub = {"id": "s4", "kind": "test_material", "arenaId": "prereg-deviation-v1", "requestedSplit": "private",
               "payload": {"description": "secret cases", "links": [], "files": []}}
        report = C.triage_submission(sub, heldout=None, runner=run)
        assert report["checks"]["translation"]["status"] == "skipped"
        assert report["checks"]["ai_review"]["status"] == "skipped"

    def test_an_idea_gets_a_brief_and_unclear_ground_truth_is_a_warning(self):
        seen: list[str] = []

        def run(prompt: str) -> str:
            seen.append(prompt)
            return json.dumps({"one_line": "Checks mediation reporting.", "input_artifact": "a results section",
                               "output_to_judge": "missing CI flags", "failure_modes": ["selective reporting"],
                               "ground_truth_strategy": "unclear", "feasibility": "low", "feasibility_why": "no coded set",
                               "questions_for_proposer": ["How would we know?"],
                               "suggested_reason": {"accept": "Thanks — worth designing.", "reject": "Not now, because…"}})

        sub = {"id": "i1", "kind": "arena_proposal", "payload": {"goes_wrong": "Mediation effects reported without CIs.", "how_known": "not sure"}}
        report = C.triage_submission(sub, heldout=None, runner=run)
        assert report["brief"]["one_line"] == "Checks mediation reporting."
        assert "suggested_reason" not in report["brief"]
        assert set(report["suggested_reason"]) == {"accept", "reject"}
        assert report["checks"]["ground_truth"]["status"] == "warn"
        assert "<idea>" in seen[0]
        # The arena list comes from leaderboard-app/lib/arena-plain.ts, which the public mirror
        # does not ship; there the prompt must say so rather than list nothing silently.
        if (C.arenas_root().parent / "leaderboard-app" / "lib" / "arena-plain.ts").exists():
            assert "grim-consistency-v1" in seen[0]  # the arena list reached the prompt
        else:
            assert "(arena list unavailable)" in seen[0]

    def test_material_with_a_missing_file_fails(self):
        sub = {"id": "s2", "kind": "test_material", "arenaId": "prereg-deviation-v1",
               "payload": {"links": [], "files": [{"filename": "d.csv"}, {"filename": "e.csv"}]}}
        report = C.triage_submission(sub, heldout=None, ai=None, files=[("d.csv", b"a,b\n1,2\n")])
        assert report["checks"]["files"]["status"] == "fail"


class TestArenaBriefCatalogue:
    """The brief prompt carries the iterate skill's failure-mode catalogue, found by
    shape (`*-iterate/...`) so the published module names no private skill path."""

    def _prompt(self, root):
        seen = {}

        def run(prompt):
            seen["prompt"] = prompt
            return json.dumps({"title": "t"})

        CI.draft_arena_brief({"payload": {"goes_wrong": "x"}}, root, run)
        return seen["prompt"]

    def test_the_catalogue_is_found_under_any_iterate_skill(self, tmp_path):
        ref = tmp_path / ".claude" / "skills" / "demo-iterate" / "references"
        ref.mkdir(parents=True)
        (ref / "failure-mode-catalog.md").write_text("CATALOGUE-MARKER-7", encoding="utf-8")
        assert "CATALOGUE-MARKER-7" in self._prompt(tmp_path)

    def test_no_catalogue_says_so(self, tmp_path):
        assert "(catalogue unavailable)" in self._prompt(tmp_path)

    def test_the_published_module_names_no_private_skill(self):
        from pathlib import Path
        src = Path(CI.__file__).read_text(encoding="utf-8")
        assert "sciencearena-iterate" not in src
