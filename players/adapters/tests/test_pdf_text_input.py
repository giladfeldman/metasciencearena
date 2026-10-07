"""The text-input and container paths added for the two PDF citation arenas (S1).

Review finding 2026-09-30: these branches shipped with no test. Each test pins one
promise a published score depends on: the model receives the paper's TEXT (not a
megabyte of base64), an extraction failure raises instead of scoring as "found
nothing", AnyStyle runs its document pipeline (`find`, not `parse`), and GROBID is
asked for the printed reference strings.
"""
from __future__ import annotations

import base64
import io
import shutil
import types

import pytest

from players.adapters import openai_compatible as oc

PDF_B64 = base64.b64encode(b"%PDF-1.4 fake bytes that are not a real document").decode()
ENVELOPE = {"task_id": "t", "input": {"document_bytes_b64": PDF_B64, "n_pages": 1,
                                      "filename": "t.pdf", "media_type": "application/pdf"}}
BASE = dict(player_id="p", player_version="v", player_type="ai-model",
            confidence_strategy="native", deterministic=False)


@pytest.fixture
def text_prompt(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("Read this:\n{{INPUT_TEXT}}\nEnd.", encoding="utf-8")
    return p


def test_api_player_receives_the_extracted_text_not_base64(text_prompt, monkeypatch):
    monkeypatch.setenv("T_BASE", "https://api.invalid/v1")
    monkeypatch.setenv("T_KEY", "k")
    monkeypatch.setattr(oc, "_pdf_to_text", lambda b64, t: "PAPER TEXT FROM PDFTOTEXT")
    sent = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"references": []}'}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent["prompt"] = json["messages"][0]["content"]
        return Resp()

    monkeypatch.setattr(oc.requests, "post", fake_post)
    ad = oc.OpenAIChatCompletionsAdapter(**BASE, openai_model="m", prompt_template_path=str(text_prompt),
                                         openai_base_url_env="T_BASE", openai_api_key_env="T_KEY",
                                         document_to_text="pdftotext")
    ad.play_task(ENVELOPE, timeout_s=10)
    assert "PAPER TEXT FROM PDFTOTEXT" in sent["prompt"]
    assert PDF_B64 not in sent["prompt"]


def test_an_unknown_document_front_end_is_refused(text_prompt):
    with pytest.raises(ValueError):
        oc.OpenAIChatCompletionsAdapter(**BASE, openai_model="m", prompt_template_path=str(text_prompt),
                                        openai_base_url_env="T_BASE", openai_api_key_env="T_KEY",
                                        document_to_text="ocr")


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler pdftotext not installed")
def test_pdftotext_failure_raises_instead_of_an_empty_answer():
    with pytest.raises(RuntimeError):
        oc._pdf_to_text(PDF_B64, 30)


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler pdftotext not installed")
def test_pdftotext_reads_a_real_pdf():
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "Smith J (2020). Anchoring revisited. Cognition 5, 1-9.")
    c.save()
    text = oc._pdf_to_text(base64.b64encode(buf.getvalue()).decode(), 30)
    assert "Anchoring revisited" in text


def test_cli_player_gets_the_same_text_and_no_tools(text_prompt, monkeypatch):
    from players.adapters import llm_pdf
    from players.adapters.llm_pdf_references import LlmPdfReferencesAdapter
    monkeypatch.setattr(oc, "_pdf_to_text", lambda b64, t: "PAPER TEXT FROM PDFTOTEXT")
    seen = {}

    def fake_spawn(argv, *, allow_tools=(), cwd=None):
        seen["allow_tools"] = tuple(allow_tools)
        return list(argv), {"cwd": cwd}

    def fake_run(argv, input=None, **kw):
        seen["prompt"] = input
        return types.SimpleNamespace(returncode=0, stdout='{"references": []}', stderr="")

    monkeypatch.setattr(llm_pdf.hermetic, "spawn_kwargs", fake_spawn)
    monkeypatch.setattr(llm_pdf.subprocess, "run", fake_run)
    ad = LlmPdfReferencesAdapter(**BASE, cli_command=["claude", "--print"],
                                 prompt_template_path=str(text_prompt), document_to_text="pdftotext")
    ad._resolved, ad._template = ["claude", "--print"], text_prompt.read_text(encoding="utf-8")
    out = ad.play_task(ENVELOPE, timeout_s=10)
    assert "PAPER TEXT FROM PDFTOTEXT" in seen["prompt"] and PDF_B64 not in seen["prompt"]
    assert seen["allow_tools"] == ()
    assert "pdftotext input" in out["player_strategy_notes"]


def test_anystyle_in_a_container_runs_find_not_parse(monkeypatch):
    from players.adapters import anystyle_references as ar
    calls = []

    def fake_run(argv, **kw):
        calls.append(list(argv))
        if argv[0] == "pdftotext":
            open(argv[-1], "w", encoding="utf-8").write("References\nSmith J. A study. 2020.\n")
        return types.SimpleNamespace(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(ar.subprocess, "run", fake_run)
    ad = ar.AnystyleReferencesAdapter(**{**BASE, "player_type": "tool", "deterministic": True},
                                      docker_container="sa-anystyle")
    ad.play_task(ENVELOPE, timeout_s=10)
    execs = [c for c in calls if c[:2] == ["docker", "exec"] and "anystyle" in c]
    assert execs and "find" in execs[0] and "parse" not in execs[0]


def test_anystyle_silence_is_an_error_unless_it_says_no_references(monkeypatch):
    from players.adapters import anystyle_references as ar

    def run_with(stderr):
        def fake_run(argv, **kw):
            if argv[0] == "pdftotext":
                open(argv[-1], "w", encoding="utf-8").write("text")
            return types.SimpleNamespace(returncode=0, stdout="", stderr=stderr)
        return fake_run

    ad = ar.AnystyleReferencesAdapter(**{**BASE, "player_type": "tool", "deterministic": True},
                                      docker_container="sa-anystyle")
    monkeypatch.setattr(ar.subprocess, "run", run_with("Analyzing t.txt ...\nno references found."))
    assert ad.play_task(ENVELOPE, timeout_s=10)["references"] == []
    monkeypatch.setattr(ar.subprocess, "run", run_with(""))
    with pytest.raises(RuntimeError):
        ad.play_task(ENVELOPE, timeout_s=10)


def test_grobid_asks_for_the_printed_reference_strings(monkeypatch):
    from players.adapters import grobid_references as gr
    sent = {}

    def fake_post(url, files=None, data=None, timeout=None):
        sent["data"] = data
        return types.SimpleNamespace(raise_for_status=lambda: None,
                                     text='<TEI xmlns="http://www.tei-c.org/ns/1.0"/>')

    monkeypatch.setattr(gr.requests, "post", fake_post)
    ad = gr.GrobidReferencesAdapter(**{**BASE, "player_type": "tool", "deterministic": True})
    ad.play_task(ENVELOPE, timeout_s=10)
    assert sent["data"].get("includeRawCitations") == "1"
