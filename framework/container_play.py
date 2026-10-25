"""Entrypoint INSIDE the arena runner image: play one task with one adapter.

    python -I -m framework.container_play /job/job.json /out/result.json

The job holds the registry entry, the task envelope (never the gold) and the
clock. The result holds everything the host runner reads off an adapter after
``play_task``: the output or the error, usage, response metadata, the request
ceiling, the request body (prompt replaced by its hash) and the raw provider
response. The host scores; nothing here sees an answer key.
"""
from __future__ import annotations

import json
import sys
import traceback


def _safe(fn):
    try:
        return fn()
    except Exception:  # telemetry never fails a task, same contract as the host
        return None


def main(argv: list[str]) -> int:
    from framework.player_adapter import build_adapter
    # Only the containerised adapter modules: importing framework.runner would
    # pull the whole orchestrator and its dependencies into the image.
    import players.adapters.gemini_native  # noqa: F401  (registers its class)
    import players.adapters.openai_compatible  # noqa: F401
    job = json.loads(open(argv[0], encoding="utf-8").read())
    result: dict = {"output": None, "error": None}
    adapter = None
    try:
        adapter = build_adapter(job["entry"])
        adapter.prepare()
        result["output"] = adapter.play_task(job["envelope"], timeout_s=int(job["timeout_s"]))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
    if adapter is not None:
        result["usage"] = _safe(adapter.last_usage)
        result["response_meta"] = _safe(getattr(adapter, "last_response_meta", lambda: None))
        result["request_max_tokens"] = _safe(adapter.last_request_max_tokens)
        result["request_body"] = _safe(adapter.last_request_body)
        result["raw_response"] = _safe(adapter.last_raw_response)
        result["resolved_tool_version"] = _safe(adapter.resolved_tool_version)
        _safe(adapter.cleanup)
    with open(argv[1], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
