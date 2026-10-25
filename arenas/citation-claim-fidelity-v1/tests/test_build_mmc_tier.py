"""build_mmc_tier: frontier labels -> tasks in the library's grouping (fictional data)."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ccf_build_mmc_tier", ARENA / "tools" / "build_mmc_tier.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def test_undecidable_dropped_and_labels_carried(tmp_path, monkeypatch):
    stage = tmp_path / "stage_e"
    (stage / "gold").mkdir(parents=True)
    (stage / "jobs").mkdir()
    job = {"citations": [{"id": i, "sentence": f"S{i} (Okafor, 2031)."} for i in ("c1", "c2", "c3")],
           "abstracts": [{"doi": "10.9/x", "title": "T", "year": 2031, "abstract": "A" * 250}]}
    (stage / "jobs" / "p.json").write_text(json.dumps(job), encoding="utf-8")
    gold = [{"citation_id": "c1", "label": "consistent", "label_source": "frontier_consensus",
             "discipline": "psychology", "sentence_sources": 1},
            {"citation_id": "c2", "label": "undecidable", "label_source": "adjudicated",
             "discipline": "psychology", "sentence_sources": 1},
            {"citation_id": "c3", "label": "contradicts", "label_source": "adjudicated",
             "discipline": "economics", "sentence_sources": 3}]
    (stage / "gold" / "gold_mmc_v1.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gold), encoding="utf-8")
    monkeypatch.setattr(B, "STAGE_E", stage)
    monkeypatch.setattr(B.panel, "groups", lambda: [
        {"job": "p", "doi": "10.9/x", "citation_ids": ["c1", "c2"], "rendered_sha256": "a"},
        {"job": "p", "doi": "10.9/x", "citation_ids": ["c2"], "rendered_sha256": "b"},
        {"job": "p", "doi": "10.9/x", "citation_ids": ["c3"], "rendered_sha256": "c"}])
    rep = Counter()
    tasks = B.mmc_tasks(rep)
    assert len(tasks) == 2 and rep["tasks_dropped_empty"] == 1 and rep["excluded_undecidable"] == 2
    assert [it["gold"] for it in tasks[0]["items"]] == ["consistent"]
    it = tasks[1]["items"][0]
    assert it["type"] == "MMC-contradicts" and it["meta"] == {
        "source": "adjudicated", "discipline": "economics", "multi_marker_sentence": True}
    assert all(t["tier"] == "rd-mmc-pilot" for t in tasks)
