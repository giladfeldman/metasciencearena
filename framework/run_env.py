"""The run environment and the call configuration, as versioned records (schema v12).

Two content-addressed objects, each stored ONCE and referenced by hash from
every run record that used it:

* ``env``  -> ``run_env/envs/<env_sha>.json``: WHERE the answer was produced.
  The runner image (id, Dockerfile hash, lock hash, Python), the egress proxy
  image, the network policy (allowed hosts), resource limits, the
  orchestrator's Python/lock, and the arena's generator fingerprint. Any change
  to any field is a new env version with a new hash. Records never embed it.

* ``call_config`` -> ``run_env/call_configs/<sha>.json``: HOW the model was
  asked. Everything that can change an answer (owner directive: effort level
  alone moves accuracy a lot), derived from the request body that was ON THE
  WIRE, not from the registry's description of it. A field is the string "n/a"
  when the request did not set that knob (the provider's own default applied,
  or the provider has no such knob); it is never omitted, so "not set" and "not
  recorded" stay distinguishable. Request keys this module does not map are
  kept under ``other_body_keys``: numbers, booleans and short strings verbatim,
  any longer text only as {sha256, chars}, so task text can never reach this
  tracked store.

Hashing is canonical JSON: sorted keys, compact separators, UTF-8 kept
(``ensure_ascii=False``) and whole-number floats written as integers, so the
same configuration hashes identically here and in the TypeScript citeclaim
library (`canonicalSha256`), whose verdicts carry the same hash.

Both stores are write-once: an existing file with different content for the
same hash is a hash collision or a tampered file, and raises.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE_ROOT = REPO_ROOT / "run_env"
#: Tests point the stores at a temp dir with this variable (conftest.py), so a
#: test run never writes into the tracked run_env/ tree.
STORE_ENV_VAR = "SCIENCEARENA_RUN_ENV_DIR"


def store_root() -> Path:
    override = os.environ.get(STORE_ENV_VAR)
    return Path(override) if override else STORE_ROOT
ENV_SCHEMA_VERSION = 1
CALL_CONFIG_SCHEMA_VERSION = 1
NA = "n/a"


def _normalise(value):
    """Whole-number floats -> int, recursively (0.0 -> 0), for cross-language hashes."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"non-finite number in a hashed record: {value}")
        if value.is_integer():
            # JavaScript cannot tell 1.5e21 from an integer and refuses it beyond
            # 2^53 (citeclaim canonicalJson); the same rule here keeps the bytes equal.
            return int(value)
        return value
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            # str(True) is "True" in Python and "true" in JavaScript: refuse rather than diverge.
            raise TypeError("hashed records take string keys only")
        return {k: _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return value


MAX_SAFE_INTEGER = 2 ** 53 - 1


def _js_number(x) -> str:
    """A number written exactly as JavaScript's JSON.stringify writes it.

    The citeclaim library hashes the same configurations in TypeScript, so the
    bytes must agree: Python writes 1e-07 where JavaScript writes 1e-7, and
    0.00001 where Python writes 1e-05 (tier-3 consult, 2026-10-14). Integers
    beyond 2^53 are refused: JavaScript cannot hold them exactly.
    """
    from decimal import Decimal

    if isinstance(x, int):
        if abs(x) > MAX_SAFE_INTEGER:
            raise ValueError(f"integer {x} is outside the range JavaScript holds exactly")
        return str(x)
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    t = Decimal(repr(abs(x))).normalize().as_tuple()   # shortest round-trip digits
    digits = "".join(map(str, t.digits))
    k = len(digits)
    n = k + t.exponent                                  # value = 0.d1..dk x 10^n
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        body = digits[0] + ("." + digits[1:] if k > 1 else "") + "e" + ("+" if e >= 0 else "-") + str(abs(e))
    return sign + body


def _canon(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _js_number(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ",".join(_canon(x) for x in v) + "]"
    if isinstance(v, dict):
        # JavaScript sorts keys by UTF-16 code unit, Python by code point; they
        # differ for keys mixing astral characters with U+E000..U+FFFF.
        keys = sorted(v, key=lambda k: k.encode("utf-16-be"))
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _canon(v[k]) for k in keys) + "}"
    raise TypeError(f"canonical_json: unsupported type {type(v).__name__}")


def canonical_json(obj) -> str:
    """Sorted keys (UTF-16 order), compact, UTF-8 kept, numbers as JavaScript writes
    them; byte-identical to citeclaim's canonicalJson for the same value."""
    return _canon(_normalise(obj))


def canonical_sha256(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _store(kind: str, obj: dict, root: Path | None = None) -> str:
    sha = canonical_sha256(obj)
    d = (root or store_root()) / kind
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{sha}.json"
    text = json.dumps(_normalise(obj), sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != sha:
            raise RuntimeError(f"{path} exists with different content; the {kind} store is write-once")
        return sha
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return sha


def store_env(env: dict, root: Path | None = None) -> str:
    return _store("envs", env, root)


def store_call_config(cfg: dict, root: Path | None = None) -> str:
    return _store("call_configs", cfg, root)


def load_env(sha: str, root: Path | None = None) -> dict:
    return json.loads(((root or store_root()) / "envs" / f"{sha}.json").read_text(encoding="utf-8"))


def load_call_config(sha: str, root: Path | None = None) -> dict:
    return json.loads(((root or store_root()) / "call_configs" / f"{sha}.json").read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    except OSError:
        return None


def orchestrator_identity() -> dict:
    """The host side that generates tasks and scores answers."""
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "requirements_lock_sha256": file_sha256(REPO_ROOT / "requirements.lock") or NA,
    }


def build_env(*, isolation: str, runner: dict | None, egress: dict | None, network: dict,
              resources: dict | None, generator_fingerprint: str | None,
              exemption: str | None = None) -> dict:
    """The env record. ``isolation`` is "container" or "host-exempt"; an exempt
    env must name its exemption (see framework/container.py EXEMPT_ADAPTER_CLASSES)."""
    if isolation not in ("container", "host-exempt"):
        raise ValueError(f"unknown isolation {isolation!r}")
    if isolation == "host-exempt" and not exemption:
        raise ValueError("a host-exempt env must name its exemption")
    return {
        "env_schema_version": ENV_SCHEMA_VERSION,
        "isolation": isolation,
        "exemption": exemption or NA,
        "runner": runner or NA,
        "egress": egress or NA,
        "network": network,
        "resources": resources or NA,
        "orchestrator": orchestrator_identity(),
        "generator_fingerprint": generator_fingerprint or NA,
    }


# --------------------------------------------------------------------- call_config
SHORT_TEXT = 64


def _clean_endpoint(url: str) -> str:
    """scheme://host[:port]/path only: never userinfo, query or fragment (a token can ride there)."""
    from urllib.parse import urlsplit
    url = (url or "").strip()
    if not url:
        return NA
    u = urlsplit(url)
    host = u.hostname or ""
    if u.port:
        host = f"{host}:{u.port}"
    return f"{u.scheme}://{host}{u.path}".rstrip("/")


def _safe_extra(v):
    """An unmapped request value as it may be stored: text longer than SHORT_TEXT
    characters (anywhere inside) becomes {sha256, chars}."""
    if isinstance(v, str):
        if len(v) <= SHORT_TEXT:
            return v
        return {"sha256": hashlib.sha256(v.encode("utf-8")).hexdigest(), "chars": len(v)}
    if isinstance(v, dict):
        return {str(k): _safe_extra(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe_extra(x) for x in v]
    return v
#: OpenAI-compatible body key -> call_config field.
_OPENAI_MAP = {
    "temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
    "seed": "seed", "frequency_penalty": "frequency_penalty", "presence_penalty": "presence_penalty",
    "repetition_penalty": "repetition_penalty", "logit_bias": "logit_bias", "n": "n", "stop": "stop",
    "max_tokens": "max_tokens", "max_completion_tokens": "max_completion_tokens",
    "reasoning_effort": "reasoning_effort", "verbosity": "verbosity",
    "response_format": "response_format", "tools": "tools", "tool_choice": "tool_choice",
    "parallel_tool_calls": "parallel_tool_calls", "stream": "stream", "provider": "provider_routing",
}
#: Anthropic Messages body key -> call_config field (effort and thinking handled apart).
_ANTHROPIC_MAP = {
    "temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "max_tokens": "max_tokens",
    "stop_sequences": "stop", "tools": "tools", "tool_choice": "tool_choice", "stream": "stream",
}
#: Gemini generationConfig key -> call_config field.
_GEMINI_MAP = {
    "temperature": "temperature", "topP": "top_p", "topK": "top_k", "seed": "seed",
    "maxOutputTokens": "max_tokens", "stopSequences": "stop", "candidateCount": "n",
    "presencePenalty": "presence_penalty", "frequencyPenalty": "frequency_penalty",
    "responseMimeType": "response_format", "responseSchema": "response_schema",
}

CALL_CONFIG_FIELDS = (
    # identity
    "provider", "endpoint", "api_type", "model_requested",
    # reasoning
    "reasoning_effort", "thinking_budget", "enable_thinking", "verbosity",
    # sampling
    "temperature", "top_p", "top_k", "min_p", "seed", "frequency_penalty", "presence_penalty",
    "repetition_penalty", "logit_bias", "n", "stop", "max_tokens", "max_completion_tokens",
    # output shaping
    "response_format", "response_schema", "tools", "tool_choice", "parallel_tool_calls",
    # prompt
    "prompt_template_sha256", "prompt_format", "group_size",
    # serving
    "provider_routing", "stream", "safety_settings",
    # execution
    "timeout_s", "max_attempts",
)


def build_call_config(*, entry: dict, body: dict | None, prompt_template_sha256: str | None,
                      timeout_s: int | None, group_size: int | None = None) -> dict:
    """One call setup's configuration, every field present ("n/a" when absent)."""
    cfg = {k: NA for k in CALL_CONFIG_FIELDS}
    cfg["call_config_schema_version"] = CALL_CONFIG_SCHEMA_VERSION
    cls = (entry.get("adapter_class") or "").rsplit(".", 1)[-1]
    body = dict(body or {})
    body.pop("messages_sha256", None)
    body.pop("contents_sha256", None)
    other: dict = {}
    if cls == "GeminiGenerateContentAdapter" or body.get("api") == "gemini-generate-content":
        cfg["api_type"] = "gemini-generate-content"
        cfg["provider"] = "google"
        cfg["endpoint"] = "https://generativelanguage.googleapis.com/v1beta"
        cfg["model_requested"] = body.pop("model", None) or entry.get("gemini_model") or NA
        body.pop("api", None)
        gen = dict(body.pop("generationConfig", {}) or {})
        thinking = gen.pop("thinkingConfig", None)
        if isinstance(thinking, dict):
            if "thinkingLevel" in thinking:
                cfg["reasoning_effort"] = thinking["thinkingLevel"]
            if "thinkingBudget" in thinking:
                cfg["thinking_budget"] = thinking["thinkingBudget"]
        for k, v in gen.items():
            if k in _GEMINI_MAP:
                cfg[_GEMINI_MAP[k]] = v
            else:
                other[f"generationConfig.{k}"] = v
        if "safetySettings" in body:
            cfg["safety_settings"] = body.pop("safetySettings")
    elif cls == "AnthropicApiAdapter" or body.get("api") == "anthropic-messages":
        cfg["api_type"] = "anthropic-messages"
        cfg["provider"] = "anthropic"
        cfg["endpoint"] = "https://api.anthropic.com/v1/messages"
        cfg["model_requested"] = body.pop("model", None) or entry.get("anthropic_model") or NA
        body.pop("api", None)
        out = dict(body.pop("output_config", {}) or {})
        if "effort" in out:
            cfg["reasoning_effort"] = out.pop("effort")
        if out:
            other["output_config"] = out
        thinking = body.pop("thinking", None)
        if isinstance(thinking, dict):
            if "budget_tokens" in thinking:
                cfg["thinking_budget"] = thinking["budget_tokens"]
            other["thinking"] = thinking
        elif thinking is not None:
            other["thinking"] = thinking
        for k, v in body.items():
            if k in _ANTHROPIC_MAP:
                cfg[_ANTHROPIC_MAP[k]] = v
            else:
                other[k] = v
    else:
        cfg["api_type"] = "openai-chat-completions" if body else NA
        cfg["model_requested"] = body.pop("model", None) or entry.get("openai_model") or NA
        base_env = entry.get("openai_base_url_env")
        cfg["endpoint"] = _clean_endpoint(os.environ.get(base_env, "")) if base_env else NA
        cfg["provider"] = entry.get("provider") or (entry.get("openai_api_key_env") or NA).removesuffix("_API_KEY").lower()
        ctk = body.pop("chat_template_kwargs", None)
        if isinstance(ctk, dict) and "enable_thinking" in ctk:
            cfg["enable_thinking"] = ctk.pop("enable_thinking")
            if ctk:
                other["chat_template_kwargs"] = ctk
        elif ctk is not None:
            other["chat_template_kwargs"] = ctk
        for k, v in body.items():
            if k in _OPENAI_MAP:
                cfg[_OPENAI_MAP[k]] = v
            else:
                other[k] = v
    cfg["prompt_template_sha256"] = prompt_template_sha256 or NA
    cfg["prompt_format"] = entry.get("prompt_format") or ("replace" if body else NA)
    cfg["group_size"] = group_size if group_size is not None else NA
    cfg["timeout_s"] = timeout_s if timeout_s is not None else NA
    cfg["other_body_keys"] = {k: _safe_extra(v) for k, v in other.items()} if other else NA
    return _normalise(cfg)


# --------------------------------------------------------------------- raw responses
def raw_sidecar_path(run_file: Path, *, held_out: bool, arena_id: str) -> Path:
    """Where a record's raw provider response is kept.

    Public (synthetic) tasks: beside the run file, ``<name>.raw.jsonl.gz``,
    tracked in git. Held-out tasks: the raw answer can reconstruct real article
    text (this arena's answers quote abstracts verbatim), so it goes to article
    custody, never a project directory (portfolio custody rule):
    ``$ARTICLE_REPOSITORY/ground_truth/sciencearena-raw/<arena>/<name>.raw.jsonl.gz``.
    """
    name = run_file.name.removesuffix(".jsonl") + ".raw.jsonl.gz"
    if not held_out:
        return run_file.with_name(name)
    repo = os.environ.get("ARTICLE_REPOSITORY")
    if not repo:
        raise RuntimeError("ARTICLE_REPOSITORY is not set; held-out raw responses go to article custody")
    root = Path(repo).resolve()
    if root == REPO_ROOT or REPO_ROOT in root.parents:
        raise RuntimeError(f"ARTICLE_REPOSITORY {root} is inside this repository; held-out raw "
                           "responses must go to article custody, never a project directory")
    return root / "ground_truth" / "sciencearena-raw" / arena_id / name


def append_raw(path: Path, line: dict) -> None:
    """Append one JSON line as a new gzip member (concatenated members read back as one stream)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "ab") as f:
        f.write((json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))


def read_raw(path: Path) -> list[dict]:
    """Every complete record. A member torn by a crash mid-write ends the read
    there; the records before it are returned (consult 2026-10-14)."""
    import zlib
    data = path.read_bytes()
    text = b""
    while data:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            chunk = d.decompress(data)
        except zlib.error:
            break
        if not d.eof:
            break  # torn member: drop it
        text += chunk
        data = d.unused_data
    return [json.loads(l) for l in text.decode("utf-8").splitlines() if l.strip()]
