"""The out-of-process extractors stay on this machine, and a crash is a failure.

framework/runner.py lists DoclingSectionsAdapter and the Marker / MinerU adapter
classes as LOCAL, which is what lets them read held-out (licensed) papers. That
classification is only honest while the things below hold, so they are asserted
here rather than trusted from a comment. These tests read the worker sources and
exercise the client with a fake worker; they need none of the tool venvs.
"""
from __future__ import annotations

import ast
import json
import sys
import textwrap
from pathlib import Path

import pytest

from framework.runner import is_cloud_player
from players.adapters import _tool_worker as tw

TOOL_ENVS = Path(tw.TOOL_ENVS)


@pytest.mark.parametrize("cls", [
    "DoclingSectionsAdapter", "MarkerTextAdapter", "MarkerSectionsAdapter", "MarkerTablesAdapter",
    "MineruTextAdapter", "MineruSectionsAdapter", "MineruTablesAdapter",
])
def test_new_adapter_classes_are_local_only_without_an_endpoint(cls):
    assert is_cloud_player({"player_id": "x", "adapter_class": cls}) is False
    assert is_cloud_player({"player_id": "x", "adapter_class": cls, "endpoint": "https://example.org"}) is True


@pytest.mark.skipif(not (TOOL_ENVS / "marker" / "worker.py").is_file(),
                    reason="players/tool_envs/ (the extractor workers) is not in the public mirror")
def test_marker_worker_never_enables_an_llm():
    src = (TOOL_ENVS / "marker" / "worker.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    config = next(n.value for n in ast.walk(tree)
                  if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "CONFIG" for t in n.targets))
    keys = {k.value for k in config.keys}
    assert keys == {"disable_ocr", "mode"}
    assert "use_llm" not in src.replace("`use_llm`", "")


@pytest.mark.skipif(not (TOOL_ENVS / "mineru" / "worker.py").is_file(),
                    reason="players/tool_envs/ (the extractor workers) is not in the public mirror")
def test_mineru_worker_never_goes_remote_and_pins_llm_features_off():
    src = (TOOL_ENVS / "mineru" / "worker.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert '"--remote"' not in code and "'--remote'" not in code
    assert "title_leveling: false" in src and "cross_page_table_cell_merge: false" in src
    assert '"telemetry", "disable"' in src or 'telemetry_action("disable")' in src
    assert "refusing to parse" in src


def test_mineru_home_is_outside_the_repository(monkeypatch):
    monkeypatch.delenv("SCIENCEARENA_MINERU_HOME", raising=False)
    home = Path(tw.mineru_env()["MINERU_HOME"]).resolve()
    assert tw.REPO_ROOT.resolve() not in home.parents


def test_task_pdf_is_written_outside_the_repository_and_removed():
    with tw.pdf_on_disk(b"%PDF-1.4 x") as path:
        p = Path(path)
        assert p.read_bytes() == b"%PDF-1.4 x"
        assert tw.REPO_ROOT.resolve() not in p.resolve().parents
    assert not p.exists()


# ------------------------------------------------- crash / timeout with a fake

def _fake_worker(tmp_path: Path, monkeypatch, body: str) -> tw.ToolWorker:
    tool_dir = tmp_path / "fake"
    tool_dir.mkdir()
    (tool_dir / "worker.py").write_text(textwrap.dedent(body), encoding="utf-8")
    monkeypatch.setattr(tw, "TOOL_ENVS", tmp_path)
    monkeypatch.setattr(tw, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("SCIENCEARENA_TOOL_ENV_FAKE", sys.executable)
    return tw.ToolWorker("fake", start_timeout_s=60)


_READY = 'import json, os, sys\nprint(json.dumps({"ready": True, "versions": {"v": "1"}}), flush=True)\n'


def test_a_worker_that_dies_mid_task_raises_with_its_exit_code(tmp_path, monkeypatch):
    w = _fake_worker(tmp_path, monkeypatch, _READY + "sys.stdin.readline()\nos._exit(3)\n")
    with pytest.raises(tw.ToolWorkerCrashed, match=r"exit(ed)? with code 3"):
        w.call({"pdf_path": "x"}, timeout_s=30)
    w.stop()


def test_a_hung_worker_becomes_a_timeout_and_is_killed(tmp_path, monkeypatch):
    w = _fake_worker(tmp_path, monkeypatch, _READY + "import time\nsys.stdin.readline()\ntime.sleep(600)\n")
    with pytest.raises(TimeoutError):
        w.call({"pdf_path": "x"}, timeout_s=3)
    assert w._proc is None


def test_a_reported_tool_error_is_raised_not_returned(tmp_path, monkeypatch):
    w = _fake_worker(tmp_path, monkeypatch, _READY +
                     'sys.stdin.readline()\nprint(json.dumps({"ok": False, "error": "ValueError: bad pdf"}), flush=True)\n'
                     "sys.stdin.readline()\n")
    with pytest.raises(RuntimeError, match="bad pdf"):
        w.call({"pdf_path": "x"}, timeout_s=30)
    w.stop()


def test_a_crashed_worker_is_restarted_for_the_next_task(tmp_path, monkeypatch):
    marker = tmp_path / "count"
    body = _READY + textwrap.dedent(f"""
        p = {str(marker)!r}
        n = int(open(p).read()) if os.path.exists(p) else 0
        open(p, "w").write(str(n + 1))
        line = sys.stdin.readline()
        if n == 0:
            os._exit(5)
        print(json.dumps({{"ok": True, "result": {{"n": n}}}}), flush=True)
        sys.stdin.readline()
    """)
    w = _fake_worker(tmp_path, monkeypatch, body)
    with pytest.raises(tw.ToolWorkerCrashed):
        w.call({"pdf_path": "x"}, timeout_s=30)
    assert w.call({"pdf_path": "x"}, timeout_s=30) == {"n": 1}
    assert w.restarts == 1
    assert w.versions() == {"v": "1"}
    w.stop()
