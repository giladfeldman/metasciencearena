"""citation-claim-fidelity-v1: scorer, generator, prompt identity, arena self-tests."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ARENA = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"ccf_{name}", ARENA / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scorer = _load("scorer")
generator = _load("generator")

ABSTRACT = ("In a randomized laboratory experiment, 84 undergraduate students were assigned to a "
            "restricted-sleep or a normal-sleep condition. Students in the restricted-sleep condition "
            "recalled fewer words (d = 0.45). Self-reported mood did not differ.")


def _gt(*golds):
    return {"abstract": ABSTRACT, "items": [{"id": i + 1, "gold": g, "type": "x"} for i, g in enumerate(golds)]}


def _out(*items):
    return {"items": [{"id": i + 1, "verdict": v, "evidence_quote": q, "reason": ""} for i, (v, q) in enumerate(items)]}


# ------------------------------------------------------------------ scorer
def test_correct_verdicts_with_verbatim_quotes_score_one():
    s = scorer.score(_out(("consistent", "recalled fewer words"), ("contradicts", "Self-reported mood did not differ"),
                          ("silent", "")), _gt("consistent", "contradicts", "silent"))
    assert s["primary"] == 1.0 and s["findings"] == []


def test_quote_guard_turns_an_unquoted_verdict_into_no_verdict():
    s = scorer.score(_out(("contradicts", "students recalled more words")), _gt("contradicts"))
    assert s["primary"] == 0.0
    assert s["findings"] == [{"category": "no_verdict", "count": 1}]


def test_quote_guard_forgives_typography_not_words():
    assert scorer.quote_in_abstract(ABSTRACT, "“RECALLED  fewer words.”")  # case, spaces, end quotes/period
    assert scorer.quote_in_abstract(ABSTRACT, "restricted–sleep condition")    # en dash -> hyphen
    assert scorer.quote_in_abstract(ABSTRACT, "84 undergraduate ... recalled fewer words")
    assert not scorer.quote_in_abstract(ABSTRACT, "recalled fewer ... 84 undergraduate")  # order matters
    assert not scorer.quote_in_abstract(ABSTRACT, "recalled more words")
    assert not scorer.quote_in_abstract(ABSTRACT, "")


def test_not_consistent_accepts_either_non_consistent_verdict():
    assert scorer.score(_out(("silent", "")), _gt("not_consistent"))["primary"] == 1.0
    assert scorer.score(_out(("contradicts", "recalled fewer words")), _gt("not_consistent"))["primary"] == 1.0
    assert scorer.score(_out(("consistent", "recalled fewer words")), _gt("not_consistent"))["primary"] == 0.0


def test_missing_duplicate_and_unknown_items_are_no_verdict():
    out = {"items": [{"id": 1, "verdict": "silent"}, {"id": 1, "verdict": "silent"}, {"id": 3, "verdict": "maybe"}]}
    s = scorer.score(out, _gt("silent", "silent", "silent"))
    assert s["primary"] == 0.0 and s["breakdown"]["n_no_verdict"] == 3
    assert scorer.score({"nope": 1}, _gt("silent"))["primary"] == 0.0


def test_error_categories_are_declared_in_arena_yaml():
    import yaml
    declared = {c["id"] for c in yaml.safe_load((ARENA / "arena.yaml").read_text(encoding="utf-8"))["error_categories"]}
    cases = [("consistent", "contradicts"), ("consistent", "silent"), ("contradicts", "consistent"),
             ("not_consistent", "consistent"), ("silent", "consistent"), ("silent", "contradicts"), ("silent", None)]
    for gold, v in cases:
        assert scorer._error_category(gold, v) in declared


# ------------------------------------------------------------------ self-tests
def _constant_player(verdict: str):
    def play(env, gt):
        n = env["input"]["group_size"]
        quote = gt["abstract"][:40] if verdict != "silent" else ""
        return {"items": [{"id": i + 1, "verdict": verdict, "evidence_quote": quote} for i in range(n)]}
    return play


def _item_results(play) -> tuple[list[dict], float]:
    items, scores = [], []
    for env in generator.generate("v1", 0, split="revealed"):
        gt = generator.ground_truth(env["task_id"])
        out = play(env, gt)
        scores.append(scorer.score(out, gt)["primary"])
        verdicts = scorer.guarded_verdicts(out, len(gt["items"]), gt["abstract"])
        items += [{"type": g["type"], "gold": g["gold"], "verdict": v} for g, v in zip(gt["items"], verdicts)]
    return items, sum(scores) / len(scores)


NEUTRAL_GROUPING = {"mean_difference": 0.0, "ci95_lower": 0.0, "n": 1}


def _small_sample_bars(bars_mod):
    """The real bars with sample-size preconditions lowered for the 13-item public
    split, and the upper bounds opened (a CI on 5 items is wide by construction).
    Rates and recall minima are unchanged, so a pass or fail is about substance."""
    import copy
    bars = copy.deepcopy(bars_mod.load_bars())
    bars["recall"]["min_n"] = 1
    for k in ("clean_false_alarm", "silent_false_alarm"):
        bars[k]["min_n"] = 1
        bars[k]["max_upper95"] = 1.0
    return bars


@pytest.mark.parametrize("verdict", ["contradicts", "consistent", "silent"])
def test_constant_players_fail_the_pass_bars(verdict):
    """Arena self-test (arena.yaml pass_bars.self_test): a player that always
    says the same thing, even with a valid quote, must FAIL the bars. Per-type
    minimum n is lowered to 1 for the small public split so recall is evaluated."""
    import copy
    bars_mod = _load("bars")
    bars = _small_sample_bars(bars_mod)
    items, acc = _item_results(_constant_player(verdict))
    report = bars_mod.evaluate(items, bars, grouping=NEUTRAL_GROUPING)
    assert report["passed"] is False, (verdict, report)
    # it fails on SUBSTANCE (a false-alarm or recall bar), not on a precondition
    assert any(c["pass"] is False and "precondition" not in c["detail"] for c in report["checks"])
    assert acc < 0.75, (verdict, acc)


def test_the_oracle_passes_the_bars_it_must_pass():
    """Control: the gold answers pass every bar, so a constant player's failure
    is the bars working, not bars nobody can meet."""
    import copy
    bars_mod = _load("bars")
    bars = _small_sample_bars(bars_mod)

    def oracle(env, gt):
        return {"items": [{"id": it["id"], "verdict": "silent" if it["gold"] == "not_consistent" else it["gold"],
                           "evidence_quote": gt["abstract"][:40]} for it in gt["items"]]}

    items, acc = _item_results(oracle)
    report = bars_mod.evaluate(items, bars, grouping=NEUTRAL_GROUPING)
    assert report["passed"] is True, report
    assert acc == 1.0


def test_an_oracle_player_scores_one_on_the_revealed_split():
    """Control for the self-test: the gold itself, with verbatim quotes, scores 1.0."""
    for env in generator.generate("v1", 0, split="revealed"):
        gt = generator.ground_truth(env["task_id"])
        out = {"items": [{"id": it["id"],
                          "verdict": "silent" if it["gold"] == "not_consistent" else it["gold"],
                          "evidence_quote": gt["abstract"][:40]} for it in gt["items"]]}
        assert scorer.score(out, gt)["primary"] == 1.0, env["task_id"]


# ------------------------------------------------------------------ generator
def test_revealed_envelopes_validate_and_cover_every_type():
    schema = json.loads((ARENA / "schemas" / "input.schema.json").read_text(encoding="utf-8"))
    v = Draft202012Validator(schema)
    types = set()
    for env in generator.generate("v1", 0, split="revealed"):
        assert env["visibility"] == "public"
        assert not list(v.iter_errors(env["input"]))
        types |= {it["type"] for it in generator.ground_truth(env["task_id"])["items"]}
    assert {"A4", "A2", "C1", "C3", "A1", "B3", "PARA", "ORIG-consistent", "ORIG-silent"} <= types


def test_private_split_refuses_without_custody(monkeypatch):
    monkeypatch.delenv("ARTICLE_REPOSITORY", raising=False)
    with pytest.raises(generator.CustodyTasksUnavailable):
        list(generator.generate("v1", 123, split="private"))


def test_private_split_refuses_a_file_that_fails_its_hash(tmp_path, monkeypatch):
    base = tmp_path / "ground_truth" / "citeclaim-2026-10"
    (base / "build_v1").mkdir(parents=True)
    (base / "build_v1" / "tasks_v1.jsonl").write_text('{"task_key": "k"}\n', encoding="utf-8")
    (base / "MANIFEST.json").write_text(json.dumps({"build_v1/tasks_v1.jsonl": {"sha256": "0" * 64}}), encoding="utf-8")
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(tmp_path))
    with pytest.raises(generator.CustodyTasksUnavailable, match="does not match"):
        list(generator.generate("v1", 123, split="private"))


def test_private_task_ids_are_salted_and_name_no_doi(tmp_path, monkeypatch):
    base = tmp_path / "ground_truth" / "citeclaim-2026-10"
    (base / "build_v1").mkdir(parents=True)
    task = {"task_key": "abc", "abstract_doi": "10.0000/x", "title": "T", "year": 2020, "abstract": "A " * 120,
            "items": [{"item_id": "i", "sentence": "S.", "type": "ORIG-consistent", "gold": "consistent"}]}
    p = base / "build_v1" / "tasks_v1.jsonl"
    p.write_text(json.dumps(task) + "\n", encoding="utf-8")
    (base / "MANIFEST.json").write_text(json.dumps({"build_v1/tasks_v1.jsonl": {
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}}), encoding="utf-8")
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(tmp_path))
    a = [e["task_id"] for e in generator.generate("v1", 111, split="private")]
    b = [e["task_id"] for e in generator.generate("v1", 222, split="private")]
    assert a != b and all("10.0000" not in t for t in a + b)
    env = next(generator.generate("v1", 111, split="private"))
    assert env["visibility"] == "held_out"


def _custody_with(tmp_path, files: dict[str, list[dict]]):
    base = tmp_path / "ground_truth" / "citeclaim-2026-10"
    manifest = {}
    for rel, tasks in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
        manifest[rel] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (base / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


def _task(key: str) -> dict:
    return {"task_key": key, "abstract_doi": "10.1/x", "title": "T", "year": 2020, "abstract": "A " * 120,
            "items": [{"item_id": key, "sentence": "S.", "type": "ORIG-consistent", "gold": "consistent"}]}


def test_v2_reads_its_own_file_and_stamps_its_version(tmp_path, monkeypatch):
    """v2 (C1 top-up, 2026-10-15) is a separate custody file; v1 output is unchanged by it."""
    _custody_with(tmp_path, {"build_v1/tasks_v1.jsonl": [_task("a")],
                             "build_v2/tasks_v2.jsonl": [_task("a"), _task("b")]})
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(tmp_path))
    v1 = list(generator.generate("v1", 7, split="private"))
    v2 = list(generator.generate("v2", 7, split="private"))
    assert len(v1) == 1 and len(v2) == 2
    assert {e["task_set_version"] for e in v1} == {"v1"} and {e["task_set_version"] for e in v2} == {"v2"}
    # The shared task keeps its id across versions (same seed, same task_key).
    assert v1[0]["task_id"] in {e["task_id"] for e in v2}
    assert {e["task_set_version"] for e in generator.generate("v2", 0, split="revealed")} == {"v2"}


def test_v2_refuses_when_only_v1_is_in_custody(tmp_path, monkeypatch):
    _custody_with(tmp_path, {"build_v1/tasks_v1.jsonl": [_task("a")]})
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(tmp_path))
    with pytest.raises(generator.CustodyTasksUnavailable):
        list(generator.generate("v2", 7, split="private"))


def test_build_c1_number_check(monkeypatch, tmp_path):
    """The v2 build keeps a C1 edit only when a number the sentence shares with the
    abstract is gone from the edit; markers, years and 0-2 are not numbers to edit."""
    monkeypatch.setenv("ARTICLE_REPOSITORY", str(tmp_path))
    spec = importlib.util.spec_from_file_location("ccf_build_t", ARENA / "tools" / "build_tasks.py")
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    assert bt._numbers("In 2019, Smith [12] found 45% of 312 adults (d = 0.41) in 2 studies", "12") == {"45", "312", "0.41"}
    bt.BUILD = tmp_path / "b2"
    bt.BUILD.mkdir()
    seed = {"seed_id": "s1", "cited_doi": "10.1/x", "abstract": "We surveyed 312 adults; 45% agreed.",
            "sentence": "Smith [3] found that 45% of 312 adults agreed.", "marker": "3"}
    (bt.BUILD / "seeds.jsonl").write_text(json.dumps(seed) + "\n", encoding="utf-8")
    variants = [("Smith [3] found that 65% of 312 adults agreed.", True),      # 45 -> 65: kept
                ("Smith [3] found that 45% of 312 adults disagreed.", False),  # no number changed
                ("Smith [4] found that 65% of 312 adults agreed.", False)]     # marker lost
    for sentence, keep in variants:
        row = {"seed_id": "s1", "variants": [{"type": "C1", "sentence": sentence}]}
        (bt.BUILD / "perturbations_raw.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        assert bool(bt.candidates_c1()) is keep


# ------------------------------------------------------------------ prompt
def test_prompt_is_byte_identical_to_citeclaim_claim_pair_v1():
    """The arena prompt and the citeclaim library's claim-pair-v1 prompt are one
    text; the library pins the same hash (its tests/fixtures/python_canonical.json)."""
    raw = (ARENA / "prompts" / "v1.md").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == "dbca576a5da1a2bc5f1f3632c0ad44843f39501fa079a64b63eb80158d98a592"


def test_render_sentences_matches_the_library():
    assert generator.render_sentences(["a", 'b "c"']) == '1. "a"\n2. "b "c""'
