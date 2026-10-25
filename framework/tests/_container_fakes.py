"""Run the containerised-adapter path without Docker, for tests.

Only the three Docker touch points are replaced: building/inspecting images,
starting the egress proxy, and `docker run`. The `docker run` replacement calls
the SAME entrypoint the image runs (framework.container_play.main) on the SAME
job file, so serialisation of the entry and envelope, the result contract and
the host-side ContainerizedAdapter are all exercised. What it cannot show is
isolation; that is proven against real Docker in test_container_live.py.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

FAKE_RUNNER = {"image": "sciencearena/runner:env-test", "image_id": "sha256:" + "0" * 64,
               "code_hash": "0" * 64, "python": "3.14-test", "pip_freeze": ["requests==test"],
               "pip_freeze_sha256": "0" * 64, "dockerfile_sha256": "0" * 64, "requirements_sha256": "0" * 64}
FAKE_EGRESS = {"image": "sciencearena/egress:1", "image_id": "sha256:" + "1" * 64, "dockerfile_sha256": "1" * 64}


@pytest.fixture
def in_process_runner(monkeypatch):
    from framework import container, container_play

    calls: list[dict] = []

    def fake_run(entry, envelope, timeout_s, runner, host):
        work = Path(tempfile.mkdtemp(prefix="sa-job-test-"))
        try:
            job = work / "job.json"
            out = work / "result.json"
            job.write_text(json.dumps({"entry": entry, "envelope": envelope, "timeout_s": timeout_s},
                                      ensure_ascii=False), encoding="utf-8")
            container_play.main([str(job), str(out)])
            res = json.loads(out.read_text(encoding="utf-8"))
            calls.append({"entry": entry["player_id"], "host": host, "task_id": envelope.get("task_id")})
            return res
        finally:
            shutil.rmtree(work, ignore_errors=True)

    monkeypatch.setattr(container, "ensure_image", lambda: dict(FAKE_RUNNER))
    monkeypatch.setattr(container, "ensure_egress_image", lambda: dict(FAKE_EGRESS))
    monkeypatch.setattr(container, "ensure_proxy", lambda host: "sa-egress-test")
    monkeypatch.setattr(container, "run_in_container", fake_run)
    return calls
