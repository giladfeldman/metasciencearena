from pathlib import Path
from framework.runner import run_tournament
from framework.storage import read_records

ARENA = Path("arenas/grim-consistency-v1")
REG = Path("players/registry.yaml")

def test_runner_writes_provenance(tmp_path):
    out = tmp_path / "scrutiny-grim.jsonl"
    run_tournament(arena_dir=ARENA, task_set_version="v1", registry_path=REG,
                   player_ids=["scrutiny-grim"], output_path=out, trials=1,
                   timeout_s=120, seed=0, split="revealed", max_tasks=1)
    recs = read_records(out)
    assert recs, "no records written"
    p = recs[0]["provenance"]
    assert p["adapter_class"] == "RCliAdapter"
    assert p["split"] == "revealed"
    assert p["trials"] == 1
    assert p["seed"] == 0
    assert p["tested_at_utc"].endswith("Z")
    assert "scrutiny-grim" in p["command"]
