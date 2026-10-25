"""Narrative prose for paperforge documents, written by Claude through the Max subscription.

Three guarantees, each enforced in code rather than asked for in the prompt:

1. **The model never writes a number.** Any digit in a returned block is a
   rejection. Numbers are ours (`simulate.py` -> `apa.py`), placed by the
   assembler.
2. **The model never touches a checked topic.** The prose must not mention the
   topics the arena scores (sampling, exclusions, missing data, randomisation,
   analysis model, inference criteria, which measure is primary, scale scoring,
   preregistration). A match on `FORBIDDEN` is a rejection, so the only evidence
   a document holds about a scored dimension is the catalog sentence the arena
   placed. Without this guard a model "helpfully" mentions an exclusion rule in
   the procedure and silently corrupts an "insufficient evidence" gold.
3. **Contained, subscription-only calls.** The CLI is spawned through
   `framework.hermetic.spawn_kwargs` (no tools, no settings, scratch cwd, our
   secrets stripped). The Anthropic API is never used.

Prose is generated once by a build tool and stored in a prose bank (JSON per
task); the arena generator only READS the bank, so `generate()` stays
deterministic, offline and reproducible.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass

from framework import hermetic

#: Case-insensitive patterns the prose may never contain, with the reason shown to
#: the model on retry. Digits first: the model must never write a number.
FORBIDDEN: dict[str, str] = {
    r"[0-9]": "a digit (numbers are inserted by the system)",
    r"regist": "preregistration or registration",
    r"random": "randomisation",
    # Word-bounded where a bare stem hits ordinary words ("alternative", "exclusively",
    # "detailed", "decided", "reliable effect"): those were measured false rejections.
    r"allocat|\balternat(?:e|ed|ely|ing|ion)\b|\bassign|every other (?:person|participant|one)|odd-numbered|even-numbered": "allocation to conditions",
    r"\bexclu(?:de|ded|des|ding|sion|sions)\b|attention[- ]check|careless|straight-?lin|inattentive|(?:not|lack of|without) (?:appear to )?(?:reflect )?genuine engagement|low-quality (?:response|data)|removed from (?:the )?analys": "exclusions",
    r"missing data|missing (?:value|response)|imput|listwise|incomplete (?:response|data)": "missing data",
    # Split (consult 2026-09-28, finding 4): the discussion may say "significant", but
    # never the tails or alpha, which are a scored dimension's value.
    r"(?:one|two)[- ]?(?:tailed|sided)|\balpha\b|p[- ]value|\bp\s*[<=>]": "inference criteria",
    r"significan": "significance",
    r"welch|t-test|anova|ancova|covariat|regression|linear model|control(?:led|ling)? for|adjust(?:ed|ing)? for": "the statistical model",
    r"power analysis|sample size|participants were recruited|we recruited": "sample size",
    r"\bprimary\b|\bsecondary\b|main outcome|dependent variable|outcome measure": "which measure is primary",
    r"averag|composite|sum score|mean score|reliabilit|cronbach|item-total|\bitems?\b": "scale scoring",
    r"deviat|\bosf\b|open science framework": "deviations or registries",
    r"prolific|mechanical turk|mturk|cloudresearch|participant pool|subject pool|\bpanel\b|crowdsourc|online platform": "data source",
    r"hypothes|\bH[12]\b|\bwe (?:predict|expect|anticipat)": "hypotheses",
}

#: Blocks where the model may refer to the study's hypotheses and to whether the
#: results were significant: the discussion must be able to interpret the findings.
DISCUSSION_BLOCKS = frozenset({"DISCUSSION", "CONCLUSION", "ABSTRACT_CONCLUSION"})
_DISCUSSION_ALLOWED = frozenset({"hypotheses", "significance"})


@dataclass(frozen=True)
class ProseRequest:
    """What the prose model is told. Contains no numbers."""
    task_key: str
    brief: str           # the study, in words
    blocks: dict         # BLOCK_NAME -> instruction


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def violations(block: str, text: str) -> list[str]:
    out = []
    for pat, why in FORBIDDEN.items():
        if block in DISCUSSION_BLOCKS and why in _DISCUSSION_ALLOWED:
            continue
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            out.append(f"{block}: contains {why} ({m.group(0)!r})")
    return out


def _prompt(req: ProseRequest, feedback: list[str]) -> str:
    lines = [
        "You are writing parts of a SYNTHETIC psychology journal article for a software benchmark.",
        "The study is invented. Do not reproduce or paraphrase any real article, and do not name real researchers.",
        "",
        "STUDY BRIEF:",
        req.brief,
        "",
        "HARD RULES (a violation makes the whole answer unusable):",
        "- Write NO digits at all: no numbers, years, percentages or statistics. Write words, never numerals.",
        "- Do not mention: preregistration or registration; how participants were sampled, how many there were,",
        "  or who was excluded; missing data; how participants were allocated to conditions; any statistical test,",
        "  model, covariate or significance criterion; which measure is primary or secondary; how scale scores were",
        "  computed, scale items, or reliability; deviations from any plan.",
        "- Do not name or describe where participants came from (no platform, panel, pool or crowdsourcing site).",
        "- Outside DISCUSSION, CONCLUSION and ABSTRACT_CONCLUSION, do not mention hypotheses or predictions and",
        "  do not use the words 'significant', 'predict' or 'hypothesis' at all.",
        "- Never use the words: item, items, average, composite, primary, secondary, random, assign, adjust.",
        "- Where you would cite prior work, write the marker [[CITE]] (one marker per citation, at most two in a row).",
        "  Never write an author name or a year.",
        "- Academic register, past tense for the study, British or American spelling consistently.",
        "",
        "Return every block below, each introduced by a line of the form ===BLOCK_NAME=== and nothing else:",
    ]
    for name, instr in req.blocks.items():
        lines.append(f"==={name}=== : {instr}")
    if feedback:
        lines += ["", "Your previous answer broke these rules; fix every one:"] + [f"- {f}" for f in feedback]
    return "\n".join(lines)


def parse_blocks(raw: str, expected: list[str]) -> dict[str, str]:
    parts = re.split(r"^===([A-Z_]+)===\s*$", raw, flags=re.MULTILINE)
    got: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        got[parts[i]] = parts[i + 1].strip()
    missing = [b for b in expected if not got.get(b)]
    if missing:
        raise ValueError(f"prose answer is missing blocks: {missing}")
    return {b: got[b] for b in expected}


def call_claude(prompt: str, model: str, timeout: int = 900) -> tuple[str, dict]:
    """One contained headless Claude call on the Max subscription. Returns (text, meta)."""
    binary = shutil.which("claude")
    if not binary:
        raise RuntimeError("claude CLI not on PATH")
    argv, kw = hermetic.spawn_kwargs(
        [binary, "-p", "--model", model, "--output-format", "json"],
        cwd=hermetic.hermetic_cwd(tag="paperforge"))
    r = subprocess.run(argv, input=prompt, capture_output=True, text=True,
                       encoding="utf-8", timeout=timeout, **kw)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"claude exited {r.returncode}: {(r.stderr or '')[:400]}")
    payload = json.loads(r.stdout)
    if payload.get("is_error"):
        raise RuntimeError(f"claude returned an error: {str(payload.get('result'))[:400]}")
    meta = {"model_requested": model,
            "model_returned": sorted((payload.get("modelUsage") or {}).keys()),
            "cli_version": hermetic.cli_version(binary),
            "containment": hermetic.PROFILE_VERSION,
            "output_tokens": (payload.get("usage") or {}).get("output_tokens")}
    return payload["result"], meta


def write_prose(req: ProseRequest, *, model: str = "sonnet", max_attempts: int = 3,
                caller=call_claude) -> dict:
    """Generate, validate and return a prose-bank entry. Raises after `max_attempts`.

    The entry records the prompt hash, the model the CLI reported, the attempt count
    and every rejection, so a bank is auditable after the fact.
    """
    feedback: list[str] = []
    rejections: list[list[str]] = []
    blocks: dict[str, str] = {}
    for attempt in range(1, max_attempts + 1):
        prompt = _prompt(req, feedback)
        raw, meta = caller(prompt, model)
        try:
            blocks = parse_blocks(raw, list(req.blocks))
        except ValueError as e:
            feedback = [str(e)]
            rejections.append(feedback)
            continue
        feedback = [v for name, text in blocks.items() for v in violations(name, text)]
        if not feedback:
            return {"task_key": req.task_key, "blocks": blocks, "attempts": attempt,
                    "rejections": rejections, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "generation": meta}
        rejections.append(feedback)
    # Last resort: drop the offending sentences when they are few. What the guard
    # protects is ABSENCE of a scored topic, and a dropped sentence cannot add evidence.
    if blocks:
        repaired, dropped = {}, []
        for name, text in blocks.items():
            paras = [[s for s in _SENT.split(p.strip()) if s] for p in text.split("\n\n")]
            n_all = sum(len(p) for p in paras)
            bad = [s for p in paras for s in p if violations(name, s)]
            if len(bad) > max(1, n_all // 10):
                raise RuntimeError(f"prose for {req.task_key}: {len(bad)} of {n_all} sentences in "
                                   f"{name} still break the rules after {max_attempts} attempts: {feedback[:6]}")
            dropped += [{"block": name, "sentence": s} for s in bad]
            repaired[name] = "\n\n".join(" ".join(s for s in p if s not in bad) for p in paras
                                         if any(s not in bad for s in p))
        leftover = [v for n, t in repaired.items() for v in violations(n, t)]
        if not leftover and all(repaired.values()):
            return {"task_key": req.task_key, "blocks": repaired, "attempts": max_attempts,
                    "rejections": rejections, "dropped_sentences": dropped,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "generation": meta}
    raise RuntimeError(f"prose for {req.task_key} still broke the rules after {max_attempts} "
                       f"attempts: {feedback[:6]}")
