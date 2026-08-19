"""An agentic CLI player must not be able to reach the repo it is being scored in.

WHY THIS TEST EXISTS
--------------------
Confirmed defect, 2026-08-16. ``SubprocessCliAdapter`` invoked ``claude --print
--model <id>`` with **no cwd, no environment sanitisation and no tool
restriction**. Claude Code in print mode has its full tool set by default, and
the runner's working directory is the repo — so every ``claude`` player ran
*inside the benchmark*, with read access to:

  * ``arenas/*/task_sets/*/_ground_truth.json``  — the committed answer key
  * ``arenas/*/task_sets/*/_held_out/**``        — the private corpora
  * ``arenas/*/{generator,scorer}.py``           — gold generation and tolerances
  * ``~/.claude/projects/<slug>/memory/``        — this project's own notes

All three escalating probes came back positive, the last one decisive: a file
containing an unguessable nonce was planted in the repo and the player returned
the nonce verbatim. The channel was open; no claim is made that any model used
it. See ``docs/design/cli-player-contamination.md``.

WHAT THIS TEST PINS, AND WHY IT IS SHAPED THIS WAY
--------------------------------------------------
Two layers, because each catches a different regression:

1. **The seam** (always runs, offline). Adapters are driven through their real
   ``play_task`` with ``subprocess.run`` captured, and the *actual* argv, cwd and
   env handed to the OS are asserted. A test that only exercised
   ``framework.hermetic`` in isolation would stay green while an adapter quietly
   stopped calling it — the exact "every unit is correct, the feature does
   nothing" shape this repo has shipped before. Asserting at the call site is
   what makes the guard real.

2. **The live nonce probe** (opt-in via ``SCIENCEARENA_LIVE_CLI_PROBE=1``).
   Spawns the real CLI against a real planted nonce. Costs a subscription call
   and needs network, so it cannot gate CI — but it is the only layer that can
   falsify the *premise* that these flags do what the docs say.

**The assertion is on the nonce, never on "the read looked like it failed."**
With tools denied the model FABRICATES plausible file content rather than
refusing (observed while verifying the fix). "It printed something that looks
like an error" is therefore worth nothing; only an unguessable string
distinguishes read / invented / blocked.

THE TRAP THAT COST A PROBE
--------------------------
``--tools`` is variadic. ``claude --print --tools "" "<prompt>"`` swallows the
prompt as a second tool name and dies with "Input must be provided either
through stdin or as a prompt argument". Every adapter here feeds the prompt on
stdin, which is why that is safe — and why ``test_hardened_argv_keeps_the_prompt_off_argv``
pins it.
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

from framework import hermetic
import players.adapters.llm_pdf_sections  # noqa: F401  — registers LlmCliPdfSectionsAdapter
from framework.player_adapter import build_adapter

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _capture_spawn(monkeypatch, module=subprocess) -> list[dict]:
    """Replace ``subprocess.run`` with a recorder that returns valid JSON.

    Defaults to patching the stdlib module itself: the PDF adapters do their
    ``import subprocess`` INSIDE ``play_task``, so there is no module-level
    attribute to patch and only the shared module object is reachable.
    """
    calls: list[dict] = []

    def fake_run(argv, **kwargs):
        calls.append({"argv": list(argv), **kwargs})
        return subprocess.CompletedProcess(
            argv, 0, stdout='{"label": "ok"}', stderr="",
        )

    target = getattr(module, "subprocess", module)
    monkeypatch.setattr(target, "run", fake_run)
    return calls


def _flag_values(argv: list[str], flag: str) -> list[str]:
    """Every value token following ``flag`` up to the next ``--option``."""
    out: list[str] = []
    if flag not in argv:
        return out
    for token in argv[argv.index(flag) + 1:]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


# --------------------------------------------------------------------------
# 1. the seam — what the OS actually receives
# --------------------------------------------------------------------------

def test_subprocess_cli_player_is_spawned_hermetically(monkeypatch, tmp_path):
    """The headline case: 57 registry entries invoke `claude` through this class."""
    prompt = tmp_path / "p.txt"
    prompt.write_text("{{INPUT_TEXT}}", encoding="utf-8")

    adapter = build_adapter({
        "player_id": "probe", "player_version": "v0", "player_type": "ai-model",
        "confidence_strategy": "native", "deterministic": False,
        "adapter_class": "SubprocessCliAdapter",
        "cli_command": ["claude", "--print", "--model", "claude-haiku-4-5"],
        "prompt_template_path": str(prompt),
    })

    import framework.player_adapter as mod
    calls = _capture_spawn(monkeypatch, mod)
    adapter.play_task({"input": {"text": "hi"}}, timeout_s=60)

    assert len(calls) == 1
    argv, cwd = calls[0]["argv"], calls[0].get("cwd")

    # Tools denied outright. An EMPTY ALLOWLIST, not a denylist: a denylist has
    # to name every tool, so the next tool the CLI ships is allowed by default.
    assert "--tools" in argv, f"no tool restriction in argv: {argv}"
    assert _flag_values(argv, "--tools") == [""], (
        f"expected an empty tool allowlist, got {_flag_values(argv, '--tools')}"
    )

    # The user's settings.json (effortLevel: xhigh, alwaysThinkingEnabled) and
    # the CLAUDE.md memory hierarchy are NOT part of the instrument.
    assert _flag_values(argv, "--setting-sources") == [""], (
        f"user settings/memory still load: {argv}"
    )

    # The cwd is what put the answer key in scope. It must not be the repo, and
    # must not be *under* the repo either.
    assert cwd is not None, "adapter still inherits the runner's cwd"
    resolved = Path(cwd).resolve()
    assert resolved != REPO_ROOT
    assert REPO_ROOT not in resolved.parents, f"spawned inside the repo: {resolved}"


def test_pdf_cli_player_keeps_read_but_only_inside_its_workspace(monkeypatch, tmp_path):
    """The PDF adapters hand the model a file PATH — they cannot use an empty allowlist.

    ``--tools ""`` would break them outright, so they get ``Read`` and nothing
    else, and their cwd is the temp directory holding that one PDF. Measured
    2026-08-16 on claude-code 2.1.224: with cwd set to a scratch directory,
    ``Read`` returns the file inside cwd and reports BLOCKED for an absolute path
    into the repo. Both directions were checked — a one-sided probe would not
    distinguish "blocked" from "broken".
    """
    prompt = tmp_path / "p.txt"
    prompt.write_text("{{PDF_PATH}} {{N_PAGES}}", encoding="utf-8")

    adapter = build_adapter({
        "player_id": "probe-pdf", "player_version": "v0", "player_type": "ai-model",
        "confidence_strategy": "native", "deterministic": False,
        "adapter_class": "LlmCliPdfSectionsAdapter",
        "cli_command": ["claude", "--print", "--model", "claude-haiku-4-5"],
        "prompt_template_path": str(prompt),
    })

    calls = _capture_spawn(monkeypatch)
    adapter.play_task(
        {"input": {"document_bytes_b64": _one_page_pdf_b64(), "n_pages": 1}},
        timeout_s=60,
    )

    assert len(calls) == 1
    argv, cwd = calls[0]["argv"], calls[0].get("cwd")

    assert _flag_values(argv, "--tools") == ["Read"], (
        f"PDF player should get Read and nothing else, got "
        f"{_flag_values(argv, '--tools')}"
    )
    assert _flag_values(argv, "--setting-sources") == [""]
    assert cwd is not None
    resolved = Path(cwd).resolve()
    assert REPO_ROOT not in resolved.parents and resolved != REPO_ROOT


def test_hardened_argv_never_ends_on_a_variadic_flag():
    """``--tools`` is variadic: a positional prompt after it is eaten as a tool name.

    Measured, not reasoned: ``claude --print --tools "" "<prompt>"`` exits with
    "Input must be provided either through stdin or as a prompt argument",
    because the prompt was consumed as a second tool name. Every adapter here
    feeds the prompt on stdin, but the invariant that keeps a future caller safe
    is that hardening ends on a BOOLEAN flag.
    """
    argv = hermetic.harden_argv(["claude", "--print", "--model", "x"], allow_tools=())
    assert argv[-1].startswith("--"), (
        f"hardening must end on a boolean flag, not a value: {argv[-3:]}"
    )
    assert argv[-1] in hermetic.HERMETIC_PROFILES["claude"].trailing_flags


def test_an_unknown_agentic_cli_is_refused_rather_than_silently_trusted():
    """A CLI with no verified hermetic profile must fail loudly, not run open.

    The failure mode this prevents is the one that created this whole class of
    defect: a new CLI player gets wired up, nobody asks what the subprocess
    inherits, and it runs with the repo in scope while every test stays green.
    """
    with pytest.raises(hermetic.UnhardenedCliError):
        hermetic.harden_argv(["some-future-agent", "--print"], allow_tools=())


def test_non_agentic_tools_are_left_alone():
    """`anystyle`, `pdftotext`, `Rscript` are not agents — they have no tools to deny."""
    for binary in ("anystyle", "pdftotext", "Rscript", "java"):
        assert not hermetic.is_agentic_cli(binary), binary
        assert hermetic.harden_argv([binary, "-x"], allow_tools=()) == [binary, "-x"]


def test_every_agentic_cli_in_the_registry_has_a_hermetic_profile():
    """No registry entry may name an agentic CLI we have not decided how to contain."""
    import yaml
    entries = yaml.safe_load((REPO_ROOT / "players" / "registry.yaml").read_text(encoding="utf-8"))
    players = entries["players"] if isinstance(entries, dict) else entries

    unprofiled = sorted({
        p["cli_command"][0]
        for p in players
        if isinstance(p, dict) and p.get("cli_command")
        and hermetic.is_agentic_cli(p["cli_command"][0])
        and p["cli_command"][0] not in hermetic.HERMETIC_PROFILES
    })
    assert not unprofiled, (
        f"agentic CLIs in registry.yaml with no hermetic profile: {unprofiled}"
    )


# --------------------------------------------------------------------------
# 2. the live nonce probe — the only layer that can falsify the premise
# --------------------------------------------------------------------------

LIVE = os.environ.get("SCIENCEARENA_LIVE_CLI_PROBE") == "1"


@pytest.mark.skipif(not LIVE, reason="set SCIENCEARENA_LIVE_CLI_PROBE=1 (costs a real CLI call)")
def test_live_claude_player_cannot_read_a_nonce_planted_in_the_repo():
    """Plant an unguessable nonce in the repo; the hermetic player must not return it.

    Deliberately NOT asserting on stderr, an exit code, or the word "BLOCKED":
    with tools denied the model fabricates file content, so only the nonce
    itself discriminates. Run against the UNFIXED adapter this returns the nonce.
    """
    nonce = f"NONCE-{uuid.uuid4().hex}"
    planted = REPO_ROOT / "_hermetic_nonce_probe.txt"
    planted.write_text(nonce, encoding="utf-8")
    try:
        argv = hermetic.harden_argv(
            ["claude", "--print", "--model", "claude-haiku-4-5-20251001"], allow_tools=(),
        )
        cwd = hermetic.hermetic_cwd()
        proc = subprocess.run(
            argv,
            input=(
                f"Read the file at {planted.as_posix()} and reply with its exact "
                f"contents and nothing else. If you cannot read it, reply exactly: BLOCKED"
            ),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(cwd), env=hermetic.hermetic_env(), timeout=300, check=False,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        assert nonce not in combined, (
            "THE CONTAMINATION CHANNEL IS OPEN: the player read a repo file.\n"
            f"argv={argv}\ncwd={cwd}\noutput={combined[:400]}"
        )
    finally:
        planted.unlink(missing_ok=True)


def _one_page_pdf_b64() -> str:
    """Smallest valid single-page PDF, base64 — the PDF adapters decode before spawning."""
    import base64
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF\n"
    )
    return base64.b64encode(pdf).decode("ascii")


# --------------------------------------------------------------------------
# 3. provenance — which model the record says served the request
# --------------------------------------------------------------------------

def _envelope_adapter(tmp_path, model: str):
    prompt = tmp_path / "p.txt"
    prompt.write_text("{{INPUT_TEXT}}", encoding="utf-8")
    return build_adapter({
        "player_id": "probe", "player_version": model, "player_type": "ai-model",
        "confidence_strategy": "native", "deterministic": False,
        "adapter_class": "SubprocessCliAdapter",
        "cli_command": ["claude", "--print", "--model", model],
        "cli_json_envelope": "claude",
        "prompt_template_path": str(prompt),
    })


def test_served_model_is_the_requested_one_not_the_auxiliary_haiku_call(monkeypatch, tmp_path):
    """The CLI bills an auxiliary haiku call on EVERY invocation, whatever the model.

    Measured 2026-08-19: ``claude --print --model claude-sonnet-5`` returns
    modelUsage with two entries — ``claude-haiku-4-5-20251001`` (523 in / 13 out,
    the CLI's own overhead call) and ``claude-sonnet-5`` (the actual request).
    ``next(iter(...))`` records whichever the CLI serialised first, so a sonnet
    player's provenance could claim it was served by haiku.

    The fixture below puts the auxiliary call FIRST, which is the order that
    breaks the old code. All 138 stored sonnet/opus records happened to be in the
    other order, so nothing published was ever wrong — this pins the latent case.
    """
    adapter = _envelope_adapter(tmp_path, "claude-sonnet-5")
    envelope = {
        "result": '{"label": "ok"}',
        "usage": {"input_tokens": 1, "output_tokens": 4},
        # auxiliary call deliberately first
        "modelUsage": {
            "claude-haiku-4-5-20251001": {"inputTokens": 523, "outputTokens": 13},
            "claude-sonnet-5": {"inputTokens": 9351, "outputTokens": 4},
        },
    }

    import framework.player_adapter as mod
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(envelope), stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    adapter.play_task({"input": {"text": "hi"}}, timeout_s=60)

    assert adapter.last_response_meta()["served_model"] == "claude-sonnet-5"


def test_served_model_falls_back_to_the_largest_call_when_nothing_matches(monkeypatch, tmp_path):
    """An unrecognised id must not silently attribute the score to the overhead call."""
    adapter = _envelope_adapter(tmp_path, "claude-sonnet-5")
    envelope = {
        "result": '{"label": "ok"}',
        "modelUsage": {
            "claude-haiku-4-5-20251001": {"inputTokens": 523, "outputTokens": 13},
            "some-renamed-snapshot": {"inputTokens": 9351, "outputTokens": 400},
        },
    }
    import framework.player_adapter as mod
    monkeypatch.setattr(
        mod.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout=json.dumps(envelope), stderr=""),
    )
    adapter.play_task({"input": {"text": "hi"}}, timeout_s=60)
    assert adapter.last_response_meta()["served_model"] == "some-renamed-snapshot"


def test_run_records_carry_a_containment_block_and_legacy_ones_read_uncontrolled():
    """CC3: keep and LABEL. Absence of the block is the signal, never a backfill."""
    fresh = {"provenance": {"containment": hermetic.record_containment(["claude", "--print"])}}
    assert hermetic.containment_state(fresh) == hermetic.CONTAINMENT_HERMETIC

    legacy = {"provenance": {"command": ["claude", "--print", "--model", "claude-haiku-4-5"]}}
    assert hermetic.containment_state(legacy) == hermetic.CONTAINMENT_UNCONTROLLED

    api_player = {"provenance": {"command": []}}
    assert hermetic.containment_state(api_player) == hermetic.CONTAINMENT_NOT_APPLICABLE

    # A CLI measured reading outside its workspace is uncontrolled however
    # carefully it was invoked — never "unverified", which would read as
    # "not yet checked".
    leaky = {"provenance": {"containment": hermetic.record_containment(["cursor-agent", "-p"])}}
    assert hermetic.containment_state(leaky) == hermetic.CONTAINMENT_UNCONTROLLED

    mixed = hermetic.summarise_containment([fresh, legacy])
    assert mixed["worst"] == hermetic.CONTAINMENT_UNCONTROLLED
    assert mixed["mixed"] is True


def test_a_wrapper_command_cannot_launder_an_agentic_cli():
    """`timeout -k 20 240 opencode run …` is an opencode player, not a timeout player.

    `players/registry.yaml` really ships this shape. The first version of
    `framework/hermetic.py` matched on argv[0] only, so `timeout` — correctly on
    the inert-tool allowlist — hid an agentic CLI behind it and those players
    were returned unhardened with no error at all. A wrapper must never be able
    to launder an agent.
    """
    argv = ["timeout", "-k", "20", "240", "opencode", "run", "--model", "x"]
    assert hermetic.effective_binary(argv) == "opencode"
    assert hermetic.is_agentic_cli(hermetic.effective_binary(argv))
    # It resolves to a profiled CLI, so hardening succeeds rather than raising.
    assert hermetic.harden_argv(argv) == argv  # opencode has no tool flag
    block = hermetic.record_containment(argv)
    assert block is not None and block["cli"] == "opencode"
    # No nonce probe has been run against opencode, so it must NOT read hermetic.
    assert hermetic.containment_state({"provenance": {"containment": block}}) == \
        hermetic.CONTAINMENT_UNVERIFIED

    # A wrapper in front of an INERT tool stays inert.
    assert hermetic.effective_binary(["timeout", "30", "pdftotext", "-layout"]) == "pdftotext"


def test_legacy_records_are_classified_by_adapter_class_not_by_the_command_string():
    """`provenance.command` is not a reliable argv, and two earlier rules got this wrong.

    Real values from stored records:
      * HTTP player   -> command == "OpenAIChatCompletionsAdapter" (the class name)
      * R tool        -> command == "'C:/Program Files/R/.../Rscript.exe' players/..."
      * CLI player    -> command == "claude --print --model X  # player-id"  (a STRING)

    Reading the first token and treating anything unrecognised as an agent
    labelled 120 of 127 published players `uncontrolled`. Both false, and the
    kind of over-warning that trains a reader to ignore the label.
    """
    def state(prov):
        return hermetic.containment_state({"provenance": prov})

    assert state({"adapter_class": "OpenAIChatCompletionsAdapter",
                  "command": "OpenAIChatCompletionsAdapter"}) == hermetic.CONTAINMENT_NOT_APPLICABLE
    assert state({"adapter_class": "RCliAdapter",
                  "command": "'C:/Program Files/R/R-4.4.0/bin/Rscript.exe' players/x.R"}) == \
        hermetic.CONTAINMENT_NOT_APPLICABLE
    assert state({"adapter_class": "SubprocessCliAdapter",
                  "command": "claude --print --model claude-haiku-4-5  # p"}) == \
        hermetic.CONTAINMENT_UNCONTROLLED
    # The opencode-behind-timeout player is a CLI player too.
    assert state({"adapter_class": "SubprocessCliAdapter",
                  "command": "timeout -k 20 240 opencode run --model x  # p"}) == \
        hermetic.CONTAINMENT_UNCONTROLLED
    # PDF CLI adapters count; the deterministic PDF parsers do not.
    assert state({"adapter_class": "LlmCliPdfSectionsAdapter"}) == hermetic.CONTAINMENT_UNCONTROLLED
    assert state({"adapter_class": "PdftotextSubprocessAdapter"}) == hermetic.CONTAINMENT_NOT_APPLICABLE


def test_every_cli_adapter_class_in_the_registry_is_declared():
    """A new CLI adapter class must be added to CLI_ADAPTER_CLASSES or it labels wrong.

    Missing one is silent: its players simply report `not-applicable`, which
    reads as "no agent was involved" — the most misleading answer available.
    """
    import yaml
    entries = yaml.safe_load((REPO_ROOT / "players" / "registry.yaml").read_text(encoding="utf-8"))
    players = entries["players"] if isinstance(entries, dict) else entries
    undeclared = sorted({
        p["adapter_class"] for p in players
        if isinstance(p, dict) and p.get("cli_command") and p.get("adapter_class")
        and hermetic.is_agentic_cli(hermetic.effective_binary(p["cli_command"]))
        and p["adapter_class"] not in hermetic.CLI_ADAPTER_CLASSES
    })
    assert not undeclared, (
        f"adapter classes that spawn an agentic CLI but are missing from "
        f"hermetic.CLI_ADAPTER_CLASSES: {undeclared}"
    )
