"""Task set v2 of publication-bias-v1: score the STATISTICS, not a verdict.

Owner decision 2026-09-17. These tests pin the four seams where v2 can silently
publish a wrong number without anything crashing:

  * the gold file drifting from the datasets it was computed on (input_sha256);
  * v1 generation changing under a v2 edit (a shared generator is append-only);
  * the scorer, the output schema, the prompt and the gold builder disagreeing on
    WHICH statistics are scored;
  * a RoBMA tolerance band tighter than RoBMA's own MCMC noise.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ARENA_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ARENA_DIR.parents[1]


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


generator = _load("_pubbias_v2_generator", ARENA_DIR / "generator.py")
scorer = _load("_pubbias_v2_scorer", ARENA_DIR / "scorer.py")
build_gold = _load("_pubbias_v2_build_gold", ARENA_DIR / "tools" / "build_gold_v2.py")

V2_SCHEMA = json.loads((ARENA_DIR / "schemas" / "v2" / "output.schema.json").read_text(encoding="utf-8"))
V1_SCHEMA = json.loads((ARENA_DIR / "schemas" / "output.schema.json").read_text(encoding="utf-8"))
PROMPT_V2 = (REPO_ROOT / "players" / "prompts" / "publication_bias_v2.txt").read_text(encoding="utf-8")
REVEALED_GOLD = ARENA_DIR / "task_sets" / "v2" / "_r_gold.json"
STABILITY = ARENA_DIR / "task_sets" / "v2" / "robma_stability.json"

# The public mirror ships none of these (gold and run records are NEVER_PUBLISH),
# so tests that read them skip there instead of failing; locally every one runs.
# Found by scripts/sync_public_mirror.sh --verify-tests on 2026-10-07.
_V1_GOLD = ARENA_DIR / "task_sets" / "v1" / "_ground_truth.json"
needs_v1_gold = pytest.mark.skipif(not _V1_GOLD.is_file(),
                                   reason="v1 _ground_truth.json is not published")
needs_v2_gold = pytest.mark.skipif(not REVEALED_GOLD.is_file(),
                                   reason="v2 _r_gold.json is not published")
needs_stability = pytest.mark.skipif(not STABILITY.is_file(),
                                     reason="robma_stability.json is not published")
needs_v1_runs = pytest.mark.skipif(not any((ARENA_DIR / "runs" / "v1").glob("*.jsonl")),
                                   reason="run records are not published")


def _revealed_v2():
    return list(generator.generate("v2", 0, split="revealed"))


def _gold_as_output(gt):
    """A player that reports exactly the gold statistics."""
    return json.loads(json.dumps(gt["gold_stats"]))


# --------------------------------------------------------------------------- #
# v1 is untouched.
# --------------------------------------------------------------------------- #
@needs_v1_gold
def test_v1_generation_is_byte_identical_to_its_committed_gold():
    committed = json.loads((ARENA_DIR / "task_sets" / "v1" / "_ground_truth.json").read_text(encoding="utf-8"))
    generated = {e["task_id"]: e for e in generator.generate("v1", 0, split="revealed")}
    assert set(generated) == set(committed) and len(generated) == 12
    for tid, entry in committed.items():
        assert generated[tid]["input"] == entry["envelope"]["input"], tid


def test_v1_scoring_is_unchanged():
    gt = {"bias_detected": True, "egger_p": 0.001}
    out = scorer.score({"bias_detected": True, "confidence": 1.0}, gt)
    assert out["primary"] == 1.0
    assert scorer.formula_id_for(gt) is None  # falls through to the v1 constant
    assert scorer.FORMULA_ID == "publication-bias-ACC85-CAL15-v1"


def test_gold_of_neither_shape_is_refused_not_scored():
    out = scorer.score({"bias_detected": True, "confidence": 1.0}, {"label": "x"})
    assert out["primary"] == 0.0 and "error" in out["breakdown"]


# --------------------------------------------------------------------------- #
# v2 datasets and gold binding.
# --------------------------------------------------------------------------- #
def test_v2_task_ids_never_collide_with_v1():
    v1 = {e["task_id"] for e in generator.generate("v1", 0)}
    v2 = {e["task_id"] for e in _revealed_v2()}
    assert len(v2) == 12 and not (v1 & v2)
    assert all(t.startswith("pubbias2-") for t in v2)


@needs_v2_gold
def test_v2_revealed_gold_covers_every_task_and_binds_to_its_dataset():
    assert REVEALED_GOLD.is_file(), "build it with tools/build_gold_v2.py --split revealed"
    for env in _revealed_v2():
        gt = generator.ground_truth(env["task_id"])
        assert gt["input_sha256"] == generator.v2_input_sha256(env["input"]["studies"])
        build_gold._assert_complete(env["task_id"], gt["gold_stats"])


@needs_v2_gold
def test_v2_stale_gold_fails_loudly(tmp_path, monkeypatch):
    gold = json.loads(REVEALED_GOLD.read_text(encoding="utf-8"))
    victim = sorted(k for k in gold if k != "_meta")[0]
    gold[victim]["input_sha256"] = "0" * 64
    fake = tmp_path / "_r_gold.json"
    fake.write_text(json.dumps(gold), encoding="utf-8")
    monkeypatch.setattr(generator, "_v2_gold_path", lambda v, s: fake)
    with pytest.raises(ValueError, match="different dataset"):
        list(generator.generate("v2", 0, split="revealed"))


def test_v2_missing_gold_yields_datasets_but_refuses_to_serve_gold(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "_v2_gold_path", lambda v, s: tmp_path / "absent.json")
    envs = list(generator.generate("v2", 0, split="revealed"))
    assert len(envs) == 12
    with pytest.raises(FileNotFoundError, match="build_gold_v2.py"):
        generator.ground_truth(envs[0]["task_id"])


@needs_v2_gold
def test_v2_datasets_spread_the_statistics():
    """v1's biased sets pinned Egger p at ~0 (down to 1e-13), which makes a p-value
    a constant. v2 must put Egger p on both sides of .05 and away from 0."""
    ps = [generator.ground_truth(e["task_id"])["gold_stats"]["egger"]["p"] for e in _revealed_v2()]
    assert min(ps) > 1e-6
    assert sum(p < 0.10 for p in ps) >= 1 and sum(p > 0.30 for p in ps) >= 4


def test_v2_every_dataset_populates_both_selection_intervals():
    for env in _revealed_v2():
        n_sig = sum(generator._one_sided_p(s["yi"], s["sei"]) < 0.025 for s in env["input"]["studies"])
        assert n_sig >= 3 and env["input"]["k"] - n_sig >= 3, env["task_id"]


# --------------------------------------------------------------------------- #
# One list of statistics, four places.
# --------------------------------------------------------------------------- #
def _scored_keys():
    return {f"{m}.{s}" for m, s, *_ in scorer.V2_STATISTICS}


def test_scorer_builder_schema_and_prompt_agree_on_the_statistics():
    builder = {f"{m}.{s}" for m, fields in build_gold.EXPECTED.items() for s in fields}
    schema = {f"{m}.{s}" for m, blk in V2_SCHEMA["properties"].items() for s in blk["properties"]}
    assert _scored_keys() == builder == schema
    assert len(_scored_keys()) == 11
    for key in _scored_keys():
        assert key in PROMPT_V2, f"prompt never asks for {key}"


def test_every_method_block_is_weighted():
    blocks = {b for bs in scorer.V2_METHODS.values() for b in bs}
    assert blocks == {m for m, *_ in scorer.V2_STATISTICS}
    assert len(scorer.V2_METHODS) == 6
    w = scorer.V2_METHOD_WEIGHTS
    assert sum(w.values()) == pytest.approx(1.0)
    # Owner decision 2026-10-01: the modern methods outweigh the classic ones.
    assert w["robma"] + w["pet_peese"] + w["three_psm"] == pytest.approx(0.75)
    assert min(w["robma"], w["pet_peese"], w["three_psm"]) > max(w["egger"], w["begg"], w["trimfill"])


@needs_v2_gold
def test_v2_schema_accepts_gold_and_rejects_the_v1_answer():
    from jsonschema import Draft202012Validator
    v = Draft202012Validator(V2_SCHEMA)
    for env in _revealed_v2():
        assert not list(v.iter_errors(_gold_as_output(generator.ground_truth(env["task_id"]))))
    assert list(v.iter_errors({"bias_detected": True, "confidence": 0.9}))
    assert list(Draft202012Validator(V1_SCHEMA).iter_errors({"egger": {"p": 0.2}}))


# --------------------------------------------------------------------------- #
# The scorer.
# --------------------------------------------------------------------------- #
@needs_v2_gold
def test_reporting_gold_scores_one_and_names_the_v2_formula():
    for env in _revealed_v2():
        gt = generator.ground_truth(env["task_id"])
        out = scorer.score(_gold_as_output(gt), gt)
        assert out["primary"] == pytest.approx(1.0), env["task_id"]
        assert out["findings"] == []
        assert out["breakdown"]["n_reported"] == 11
        assert scorer.formula_id_for(gt) == scorer.FORMULA_ID_V2


@needs_v2_gold
def test_omitting_everything_scores_zero_with_eleven_named_misses():
    gt = generator.ground_truth(_revealed_v2()[0]["task_id"])
    out = scorer.score({}, gt)
    assert out["primary"] == 0.0 and out["breakdown"]["n_reported"] == 0
    assert [f["category"] for f in out["findings"]] == ["statistic_missing"] * 11


@needs_v2_gold
def test_omission_is_never_cheaper_than_a_wrong_value():
    gt = generator.ground_truth(_revealed_v2()[0]["task_id"])
    full = _gold_as_output(gt)
    omitted = {k: v for k, v in full.items() if k != "egger"}
    g = gt["gold_stats"]["egger"]["p"]
    wrong = dict(full, egger={"p": g - 0.5 if g > 0.5 else g + 0.5})
    assert scorer.score(omitted, gt)["primary"] == scorer.score(wrong, gt)["primary"]
    assert scorer.score(omitted, gt)["primary"] == pytest.approx(1 - scorer.V2_METHOD_WEIGHTS["egger"])


@needs_v2_gold
def test_band_is_linear_between_full_and_zero():
    gt = generator.ground_truth(_revealed_v2()[0]["task_id"])
    g = gt["gold_stats"]["pet"]["estimate"]
    for delta, credit in [(0.001, 1.0), (0.002, 1.0), (0.011, 0.5), (0.02, 0.0), (0.5, 0.0)]:
        out = _gold_as_output(gt)
        out["pet"]["estimate"] = g + delta
        b = scorer.score(out, gt)["breakdown"]
        assert b["stat_pet_estimate"] == pytest.approx(credit, abs=1e-9), delta
        # PET-PEESE is one method of two statistics.
        assert b["method_pet_peese"] == pytest.approx((credit + 1.0) / 2)


@needs_v2_gold
def test_trimfill_count_must_be_exact():
    gt = generator.ground_truth(_revealed_v2()[0]["task_id"])
    k0 = gt["gold_stats"]["trimfill"]["k0"]
    for reported, credit in [(k0, 1.0), (float(k0), 1.0), (k0 + 1, 0.0), (k0 + 0.5, 0.0)]:
        out = _gold_as_output(gt)
        out["trimfill"]["k0"] = reported
        assert scorer.score(out, gt)["breakdown"]["stat_trimfill_k0"] == credit, reported


@needs_v2_gold
def test_bayes_factor_is_scored_on_the_log_scale():
    gt = generator.ground_truth(_revealed_v2()[0]["task_id"])
    bf = gt["gold_stats"]["robma"]["bf_bias"]
    for factor, credit in [(math.exp(0.04), 1.0), (1 / math.exp(0.04), 1.0), (math.exp(0.275), 0.5), (math.exp(0.5), 0.0)]:
        out = _gold_as_output(gt)
        out["robma"]["bf_bias"] = bf * factor
        assert scorer.score(out, gt)["breakdown"]["stat_robma_bf_bias"] == pytest.approx(credit, abs=1e-9)


@needs_v2_gold
def test_a_boolean_or_string_is_not_a_statistic():
    gt = generator.ground_truth(_revealed_v2()[0]["task_id"])
    out = _gold_as_output(gt)
    out["egger"]["p"] = True
    out["begg"]["tau"] = str(out["begg"]["tau"])
    b = scorer.score(out, gt)["breakdown"]
    assert b["stat_egger_p"] == 0.0 and b["stat_begg_tau"] == 0.0 and b["n_reported"] == 9


def test_v2_findings_categories_are_declared_in_the_manifest():
    import yaml
    declared = {c["id"] for c in yaml.safe_load((ARENA_DIR / "arena.yaml").read_text(encoding="utf-8"))["error_categories"]}
    assert {"statistic_off", "statistic_missing"} <= declared


# --------------------------------------------------------------------------- #
# RoBMA is MCMC: its band must cover its own seed-to-seed noise.
# --------------------------------------------------------------------------- #
def _robma_refits():
    """Every RoBMA refit of a revealed dataset under a seed other than the gold's:
    the stability file (tools/build_gold_v2.py --stability-seeds) plus the reference
    player's published run (RoBMA seed 1), which is the same measurement taken
    through the runner."""
    fits = []
    stab = json.loads(STABILITY.read_text(encoding="utf-8"))
    for key, st in stab["fits"].items():
        fits.append((key.split("|")[0], st["robma"]))
    for f in (ARENA_DIR / "runs" / "v2").glob("metafor-robma-pubbias__revealed__*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out = rec.get("output") or {}
                if isinstance(out.get("robma"), dict):
                    fits.append((rec["task_id"], out["robma"]))
    return fits


@needs_stability
def test_robma_full_credit_band_covers_measured_mcmc_noise():
    assert STABILITY.is_file(), "build it with tools/build_gold_v2.py --stability-seeds"
    bands = {s: full for m, s, _k, full, _z in scorer.V2_STATISTICS if m == "robma"}
    gold = {e["task_id"]: generator.ground_truth(e["task_id"])["gold_stats"]["robma"] for e in _revealed_v2()}
    fits = _robma_refits()
    assert {t for t, _ in fits} == set(gold), "every revealed dataset needs at least one refit"
    worst = {"bf_bias": 0.0, "mu": 0.0}
    for tid, other in fits:
        worst["bf_bias"] = max(worst["bf_bias"], abs(math.log(other["bf_bias"]) - math.log(gold[tid]["bf_bias"])))
        worst["mu"] = max(worst["mu"], abs(other["mu"] - gold[tid]["mu"]))
    assert len(fits) >= 12
    assert worst["bf_bias"] <= bands["bf_bias"], worst
    assert worst["mu"] <= bands["mu"], worst


# --------------------------------------------------------------------------- #
# Players are shown the v2 prompt; the v1 prompt stays readable.
# --------------------------------------------------------------------------- #
@needs_v1_runs
def test_llm_players_read_the_v2_prompt_and_v1_records_name_the_archived_one():
    sys.path.insert(0, str(REPO_ROOT))
    from framework.registry import load_registry
    from framework.runner import _prompt_template_sha
    entries = [e for e in load_registry(REPO_ROOT / "players" / "registry.yaml")
               if e["player_id"].endswith("-pubbias") and e.get("prompt_template_path")]
    assert len(entries) >= 10
    assert {e["prompt_template_path"] for e in entries} == {"players/prompts/publication_bias_v2.txt"}
    archived = REPO_ROOT / "players" / "prompts" / "_archive" / "publication_bias.c2ded7861e854256.txt"
    assert _prompt_template_sha(str(archived)) == "c2ded7861e854256"
    hashes = set()
    for f in (ARENA_DIR / "runs" / "v1").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                h = (json.loads(line).get("provenance") or {}).get("prompt_template_sha256")
                if h:
                    hashes.add(h)
    assert hashes == {"c2ded7861e854256"}


@needs_v2_gold
def test_an_impossible_value_costs_one_statistic_not_the_whole_task():
    """2026-09-30: cohere-command-r-plus reported robma.bf_bias = 0.0. A schema with
    exclusiveMinimum rejected the WHOLE answer, voiding ten other statistics, while the
    design (and Sonnet's consult) says a wrong statistic scores 0 for itself only."""
    from jsonschema import Draft202012Validator
    gt = generator.ground_truth(_revealed_v2()[0]["task_id"])
    out = _gold_as_output(gt)
    out["robma"]["bf_bias"] = 0.0
    out["egger"]["p"] = 1.7
    assert not list(Draft202012Validator(V2_SCHEMA).iter_errors(out))
    b = scorer.score(out, gt)["breakdown"]
    assert b["stat_robma_bf_bias"] == 0.0 and b["stat_egger_p"] == 0.0
    assert b["method_pet_peese"] == 1.0 and b["method_three_psm"] == 1.0
