"""Local OpenAI-compatible HTTP shim backed by the codex CLI (no API key).

regcheck talks to OpenAI via the `openai` Python SDK, which honours the
`OPENAI_BASE_URL` environment variable. This module stands up a tiny
`http.server` that implements the two endpoints regcheck's `backend.cli general`
path actually calls:

  * POST /v1/chat/completions  -> flattens the OpenAI `messages` into one prompt,
                                  shells out to `codex exec` (the owner's ChatGPT
                                  subscription — no API key, no API cost), and
                                  returns an OpenAI chat-completion envelope.
  * POST /v1/embeddings        -> embeds inputs with sentence-transformers
                                  (all-MiniLM-L6-v2) when importable, else a
                                  deterministic hashing-based pseudo-embedding
                                  fallback (degraded retrieval, but offline).

Point regcheck at it with:
    OPENAI_BASE_URL=http://127.0.0.1:<port>/v1
    OPENAI_API_KEY=dummy-local-shim          (never a real key)

No secrets live here. `codex_complete` is a module-level function so tests can
monkeypatch it without spawning the real CLI.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

logger = logging.getLogger("regcheck_shim")

# ---------------------------------------------------------------------------
# Chat: codex CLI round-trip
# ---------------------------------------------------------------------------

CODEX_BINARY = os.environ.get("REGCHECK_SHIM_CODEX", "codex")
# Default model left to codex's own default; override via env if desired.
CODEX_MODEL = os.environ.get("REGCHECK_SHIM_CODEX_MODEL", "").strip()
CODEX_TIMEOUT_S = int(os.environ.get("REGCHECK_SHIM_CODEX_TIMEOUT_S", "600"))


def _messages_to_prompt(messages: list[dict], *, want_json: bool) -> str:
    """Flatten OpenAI chat `messages` into a single codex prompt."""
    parts: list[str] = []
    for m in messages or []:
        role = (m.get("role") or "user").upper()
        content = m.get("content")
        if isinstance(content, list):
            # OpenAI content-parts form: keep only text parts.
            content = "".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        parts.append(f"[{role}]\n{content or ''}")
    prompt = "\n\n".join(parts)
    if want_json:
        prompt += (
            "\n\n[OUTPUT CONTRACT]\n"
            "Respond with a SINGLE valid JSON object and nothing else. "
            "No markdown, no code fences, no commentary before or after the JSON."
        )
    return prompt


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # drop the opening fence line and the trailing fence
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def codex_complete(prompt: str, *, want_json: bool = False, timeout_s: int | None = None, **_) -> str:
    """Run `codex exec` on `prompt` and return ONLY the final assistant answer.

    Uses `-o/--output-last-message <file>` so the final message is recovered
    cleanly regardless of the surrounding event log on stdout.
    """
    timeout_s = timeout_s or CODEX_TIMEOUT_S
    with tempfile.TemporaryDirectory(prefix="regcheck_shim_") as td:
        last_path = os.path.join(td, "last.txt")
        cmd = [CODEX_BINARY, "exec", "--skip-git-repo-check", "-o", last_path]
        if CODEX_MODEL:
            cmd += ["--model", CODEX_MODEL]
        cmd += [prompt]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        answer = ""
        if os.path.exists(last_path):
            with open(last_path, "r", encoding="utf-8", errors="replace") as fh:
                answer = fh.read().strip()
        if not answer:
            # Fallback: codex prints the final answer last on stdout.
            answer = (proc.stdout or "").strip()
        if proc.returncode != 0 and not answer:
            raise RuntimeError(
                f"codex exec exited {proc.returncode}: {(proc.stderr or '').strip()[:400]}"
            )
        answer = _strip_code_fences(answer)
        if want_json:
            answer = _coerce_json_object(answer)
        return answer


def _coerce_json_object(text: str) -> str:
    """Best-effort: return a JSON-object string from possibly-noisy model text."""
    t = text.strip()
    try:
        json.loads(t)
        return t
    except Exception:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            return candidate
    return t


# ---------------------------------------------------------------------------
# Embeddings: sentence-transformers if available, else deterministic hashing
# ---------------------------------------------------------------------------

_ST_MODEL = None
_ST_TRIED = False
_HASH_DIM = 384  # matches all-MiniLM-L6-v2 dim; arbitrary for the fallback.


def _get_st_model():
    global _ST_MODEL, _ST_TRIED
    if _ST_TRIED:
        return _ST_MODEL
    _ST_TRIED = True
    if os.environ.get("REGCHECK_SHIM_NO_ST"):
        _ST_MODEL = None
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("regcheck_shim: using sentence-transformers all-MiniLM-L6-v2 for embeddings")
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning(
            "regcheck_shim: sentence-transformers unavailable (%s); using DEGRADED "
            "hashing-based pseudo-embeddings. Retrieval quality is reduced.",
            exc,
        )
        _ST_MODEL = None
    return _ST_MODEL


def _hash_embed(text: str, dim: int = _HASH_DIM) -> list[float]:
    """Deterministic bag-of-token hashing embedding, L2-normalised.

    DEGRADED fallback. Tokens are hashed into `dim` buckets; the vector is
    normalised so cosine similarity is well-defined. Good enough to let
    regcheck's retrieval run offline, but far weaker than a real model.
    """
    vec = [0.0] * dim
    tokens = [tok for tok in _simple_tokens(text)]
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _simple_tokens(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def embed_texts(inputs: list[str]) -> list[list[float]]:
    model = _get_st_model()
    if model is not None:
        arr = model.encode(list(inputs), normalize_embeddings=True)
        return [list(map(float, row)) for row in arr]
    return [_hash_embed(t) for t in inputs]


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D401 - silence default stderr logging
        return

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        # Minimal health/models endpoints so SDK probes don't 404 fatally.
        if self.path.rstrip("/").endswith("/models"):
            self._send_json({"object": "list", "data": [{"id": "gpt-5", "object": "model"}]})
            return
        self._send_json({"status": "ok"})

    def do_POST(self):  # noqa: N802 - http.server API
        try:
            body = self._read_body()
        except Exception as exc:
            self._send_json({"error": {"message": f"bad request body: {exc}"}}, status=400)
            return

        path = self.path.split("?", 1)[0].rstrip("/")
        try:
            if path.endswith("/chat/completions"):
                self._send_json(self._handle_chat(body))
            elif path.endswith("/embeddings"):
                self._send_json(self._handle_embeddings(body))
            else:
                self._send_json({"error": {"message": f"unsupported path: {path}"}}, status=404)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("regcheck_shim handler error")
            self._send_json({"error": {"message": str(exc), "type": "shim_error"}}, status=500)

    # -- handlers -----------------------------------------------------------

    def _handle_chat(self, body: dict) -> dict:
        messages = body.get("messages") or []
        model = body.get("model") or "gpt-5"
        want_json = (body.get("response_format") or {}).get("type") == "json_object"
        prompt = _messages_to_prompt(messages, want_json=want_json)
        answer = codex_complete(prompt, want_json=want_json)
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(answer) // 4)
        return {
            "id": f"chatcmpl-shim-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _handle_embeddings(self, body: dict) -> dict:
        raw_input = body.get("input")
        model = body.get("model") or "text-embedding-3-large"
        if isinstance(raw_input, str):
            inputs = [raw_input]
        elif isinstance(raw_input, list):
            # OpenAI also allows token-id arrays; coerce everything to str.
            inputs = ["".join(map(str, x)) if isinstance(x, list) else str(x) for x in raw_input]
        else:
            inputs = [str(raw_input)]
        vectors = embed_texts(inputs)
        data = [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ]
        total = sum(max(1, len(t) // 4) for t in inputs)
        return {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {"prompt_tokens": total, "total_tokens": total},
        }


class _ThreadingShimServer(ThreadingHTTPServer):
    # Concurrent connections: the openai SDK (httpx) keeps a pooled HTTP/1.1
    # keep-alive connection per endpoint, so embeddings + chat requests can be
    # in flight at once. A single-threaded server serializes them and the idle
    # connection's socket buffer fills → WinError 10053/10054 resets. Threading
    # one request per connection fixes it. daemon threads so shutdown is clean.
    daemon_threads = True
    # Don't crash the server thread on a client that drops mid-response.
    def handle_error(self, request, client_address):  # noqa: D401
        return


def make_server(port: int = 0, host: str = "127.0.0.1") -> HTTPServer:
    """Create (but do not start) a threaded HTTPServer bound to `host:port`.

    Pass port=0 for an ephemeral port (read it back via `server.server_address`).
    """
    return _ThreadingShimServer((host, port), _Handler)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Local OpenAI-compatible codex shim for regcheck.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("REGCHECK_SHIM_PORT", "8765")))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    server = make_server(port=args.port, host=args.host)
    actual_port = server.server_address[1]
    print(f"regcheck_shim listening on http://{args.host}:{actual_port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
