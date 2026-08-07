import json
from pathlib import Path
from jsonschema import Draft202012Validator
from framework.paths import schema_path

SCHEMA = json.loads(schema_path("run_record.schema.json").read_text(encoding="utf-8"))
V = Draft202012Validator(SCHEMA)

BASE = {
    "run_id": "r1", "arena_id": "a", "task_set_version": "v1", "task_id": "t1",
    "player_id": "p", "player_version": "1.0", "player_type": "tool",
    "input_hash": "h", "output": {}, "score": {"primary": 1.0},
    "timestamp_utc": "2026-06-07T00:00:00Z",
}

def test_legacy_record_without_provenance_validates():
    assert list(V.iter_errors(BASE)) == []

def test_record_with_provenance_validates():
    rec = dict(BASE, provenance={
        "tested_at_utc": "2026-06-07T00:00:00Z", "host": "win11-local",
        "adapter_class": "RCliAdapter", "command": "Rscript players/adapters/metacheck.R",
        "tool_version_detail": "metacheck 0.1.0 @abc1234", "trials": 1, "seed": 0, "split": "revealed",
    })
    assert list(V.iter_errors(rec)) == []

def test_provenance_rejects_unknown_field():
    rec = dict(BASE, provenance={"api_key": "leak"})
    assert list(V.iter_errors(rec)) != []
