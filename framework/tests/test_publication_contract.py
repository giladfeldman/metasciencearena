"""The publication contract (plan D, 2026-11-01): a record publishes only with its evidence.

Both copies are pinned here: framework/publication_contract.py (QA gate C12) and the site build's
leaderboard-app/scripts/lib/publication-contract.mjs must give the same verdict on the same record.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from framework import publication_contract as pc

REPO = Path(__file__).resolve().parents[2]
SITE_LIB = REPO / "leaderboard-app" / "scripts" / "lib" / "publication-contract.mjs"
FROZEN = "2026-11-01T10:00:00Z"
CONFIGS = {"api": {"api_type": "anthropic-messages"}, "cli": {"api_type": "cli:claude"}}

CASES = {
    "v13 tool, env only": ({"schema_version": 13, "player_type": "tool", "env_sha": "e"}, None),
    "v13 without env": ({"schema_version": 13, "player_type": "tool"}, "v13 record without env_sha"),
    "v13 model without config": ({"schema_version": 13, "player_type": "ai-model", "env_sha": "e"},
                                 "v13 language-model record without call_config_sha256"),
    "v13 model, config missing": ({"schema_version": 13, "player_type": "ai-model", "env_sha": "e",
                                   "call_config_sha256": "f" * 64},
                                  f"call settings {'f' * 12} cited but not in run_env/call_configs/"),
    "v13 API model": ({"schema_version": 13, "player_type": "ai-model", "env_sha": "e", "call_config_sha256": "api"},
                      None),
    "v13 CLI model without block": ({"schema_version": 13, "player_type": "ai-model", "env_sha": "e",
                                     "call_config_sha256": "cli", "model_call": None},
                                    "v13 command-line-tool record without its model_call block"),
    "v13 CLI model with block": ({"schema_version": 13, "player_type": "ai-model", "env_sha": "e",
                                  "call_config_sha256": "cli",
                                  "model_call": {"recorder_version": 1, "launch": {"binary": "claude"}}}, None),
    "v13 CLI model, empty aggregate": ({"schema_version": 13, "player_type": "ai-model", "env_sha": "e",
                                        "call_config_sha256": "cli", "model_call": {"blocks": []}},
                                       "v13 command-line-tool record without its model_call block"),
    "v13 CLI model, empty object": ({"schema_version": 13, "player_type": "ai-model", "env_sha": "e",
                                     "call_config_sha256": "cli", "model_call": {}},
                                    "v13 command-line-tool record without its model_call block"),
    "v13 as a float": ({"schema_version": 13.0, "player_type": "tool", "env_sha": "e"}, None),
    "legacy before freeze": ({"schema_version": 12, "timestamp_utc": "2026-10-31T02:35:10Z"}, None),
    "legacy after freeze": ({"schema_version": 12, "timestamp_utc": "2026-11-02T00:00:00Z"},
                            f"legacy (pre-v13) record written after the inventory was frozen ({FROZEN})"),
    "legacy, no version, no time": ({}, f"legacy (pre-v13) record written after the inventory was frozen ({FROZEN})"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_record_verdicts(name):
    rec, want = CASES[name]
    assert pc.record_problem(rec, frozen_utc=FROZEN, call_config_of=CONFIGS.get) == want


def test_no_inventory_refuses_every_legacy_record():
    assert pc.record_problem({"schema_version": 5, "timestamp_utc": "2020-01-01T00:00:00Z"},
                             frozen_utc=None) == "legacy (pre-v13) record and no frozen legacy inventory"


def _corpus(tmp_path: Path, recs: list[dict]) -> Path:
    runs = tmp_path / "arenas" / "a1" / "runs" / "v1"
    runs.mkdir(parents=True)
    (runs / "run.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    (runs / "run.retry-r1.jsonl").write_text(json.dumps({"schema_version": 1}) + "\n", encoding="utf-8")
    return tmp_path / "arenas"


def test_check_counts_legacy_against_the_frozen_cap(tmp_path):
    old = {"schema_version": 12, "timestamp_utc": "2026-10-01T00:00:00Z", "task_id": "t"}
    arenas = _corpus(tmp_path, [old, old])
    assert pc.check(arenas, {"frozen_utc": FROZEN, "arenas": {"a1": 2}}) == []
    fails = pc.check(arenas, {"frozen_utc": FROZEN, "arenas": {"a1": 1}})
    assert fails == ["a1: 2 legacy (pre-v13) records, the frozen inventory allows 1"]


def test_an_empty_corpus_is_not_a_pass(tmp_path):
    (tmp_path / "arenas" / "a1").mkdir(parents=True)
    assert pc.check(tmp_path / "arenas", {"frozen_utc": FROZEN, "arenas": {}})[0].startswith("no run records found")


def test_freeze_then_check_passes_and_a_new_legacy_record_fails(tmp_path):
    old = {"schema_version": 12, "timestamp_utc": "2026-10-01T00:00:00Z", "task_id": "t"}
    arenas = _corpus(tmp_path, [old])
    inv_path = tmp_path / "inv.json"
    inv = pc.freeze("test", arenas=arenas, path=inv_path)
    assert inv["arenas"] == {"a1": 1} and inv["total"] == 1  # the retry temp is not counted
    assert pc.check(arenas, pc.load_inventory(inv_path)) == []
    late = {"schema_version": 12, "timestamp_utc": "2099-01-01T00:00:00Z", "task_id": "late"}
    with (arenas / "a1" / "runs" / "v1" / "run.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(late) + "\n")
    fails = pc.check(arenas, pc.load_inventory(inv_path))
    assert any("written after the inventory was frozen" in f for f in fails)
    assert any("2 legacy (pre-v13) records, the frozen inventory allows 1" in f for f in fails)


@pytest.mark.skipif(not pc.INVENTORY.is_file(),
                    reason="run_env/ and the run records are not in the public mirror -- NOT verified here")
def test_the_committed_inventory_admits_todays_corpus():
    """Two-sided: the live corpus passes under the committed inventory, and fails without one."""
    inv = pc.load_inventory()
    assert inv.get("frozen_utc"), "run_env/legacy_inventory.json is missing or unfrozen"
    assert pc.check(inventory=inv) == []
    assert pc.check(inventory={"frozen_utc": None, "arenas": {}}) != []


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed -- NOT verified")
@pytest.mark.skipif(not SITE_LIB.is_file(), reason="leaderboard-app/ is not in the public mirror -- NOT verified here")
def test_the_site_build_gives_the_same_verdicts():
    script = ("import { recordProblem } from " + json.dumps(SITE_LIB.as_uri()) + ";"
              "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>{const {cases,frozen,configs}=JSON.parse(s);"
              "const out={};for(const [k,[rec]] of Object.entries(cases))"
              "out[k]=recordProblem(rec,{frozenUtc:frozen,callConfigOf:(sha)=>configs[sha]??null});"
              "process.stdout.write(JSON.stringify(out));});")
    payload = json.dumps({"cases": CASES, "frozen": FROZEN, "configs": CONFIGS})
    out = subprocess.run(["node", "--input-type=module", "-e", script], input=payload, capture_output=True,
                         text=True, encoding="utf-8", check=True, timeout=60).stdout
    js = json.loads(out)
    for name, (rec, _want) in CASES.items():
        assert js[name] == pc.record_problem(rec, frozen_utc=FROZEN, call_config_of=CONFIGS.get), name
