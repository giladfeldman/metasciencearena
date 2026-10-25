"""Build the PDF-universe prose bank: synthetic, digit-free article prose per topic.

Run once (and again only to ADD topics; existing entries are never rewritten):

    .venv/Scripts/python.exe -m framework.pdfforge.tools.build_prose_bank --jobs 4

Each topic gets one contained headless Claude call on the Max subscription
(`framework.paperforge.prose.call_claude`: no tools, scratch cwd, secrets stripped).
The answer is rejected and retried if any block contains a digit (numbers are ours),
a real-looking citation (author + year), or misses a block. Prior work is marked
[[CITE]], which the paper builder replaces with synthetic citations in the paper's
citation style. Output: framework/pdfforge/prose_bank.json (committed).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
from pathlib import Path

from framework.paperforge.prose import call_claude, parse_blocks

BANK = Path(__file__).resolve().parents[1] / "prose_bank.json"

TOPICS = [
    "anchoring of willingness-to-pay by arbitrary numbers", "the decoy effect in consumer choice",
    "regret anticipation and the status quo", "omission bias in vaccination decisions",
    "moral licensing after charitable giving", "the identifiable victim effect",
    "construal level and temporal discounting", "nudges toward retirement saving defaults",
    "loss aversion in effort allocation", "hindsight bias in medical judgments",
    "the endowment effect for experiential goods", "framing of risky public-health choices",
    "intellectual humility and belief updating", "gratitude expression and prosocial behaviour",
    "social comparison on image-sharing platforms", "stereotype threat in mathematics testing",
    "self-control depletion and persistence", "mindfulness training and attentional control",
    "loneliness and perceived social support in older adults", "growth mindset messages in schools",
    "procrastination and academic performance", "trust in automated advice",
    "misinformation correction and continued influence", "conspiracy beliefs and analytic thinking",
    "moral outrage and online sharing", "intergroup contact and prejudice reduction",
    "forgiveness after interpersonal transgressions", "perceived scarcity and cognitive bandwidth",
    "choice overload in assortment selection", "the planning fallacy in task estimates",
    "reactance to persuasive health messages", "power poses and felt confidence",
    "sleep restriction and emotion regulation", "music tempo and running pace",
    "colour cues and perceived product quality", "eye-gaze cues and honesty",
    "the bystander effect in online help requests", "counterfactual thinking after near misses",
    "affective forecasting of exam outcomes", "need for cognition and argument quality",
    "replication of the facial feedback effect", "the Macbeth effect of physical cleansing",
    "sunk-cost escalation in team projects", "the less-is-more effect in bundle valuation",
    "prosocial spending and happiness", "nature exposure and restorative attention",
    "the mere exposure effect for brand logos", "cognitive reflection and fraud susceptibility",
    "workplace ostracism and voice behaviour", "self-affirmation and defensive processing",
    "warmth and competence judgments of professionals", "the fluency heuristic in truth judgments",
    "goal gradient effects in loyalty programmes", "honesty oaths before self-reports",
    "the halo effect in grading", "ego-depletion replication with a sequential task",
    "dishonesty under time pressure", "fairness norms in the ultimatum game",
    "parental praise and children's persistence", "climate-change risk perception and efficacy",
]

BLOCKS = {
    "TITLE": "an article title of eight to sixteen words, no colon numbering",
    "ABSTRACT_BACKGROUND": "two sentences of background for the abstract",
    "ABSTRACT_CONCLUSION": "two sentences of conclusion for the abstract",
    "KEYWORDS": "five keywords separated by semicolons",
    "INTRO": "four paragraphs of introduction, five to seven sentences each, citing prior work with [[CITE]] markers (two to four per paragraph)",
    "THEORY": "two paragraphs developing the theoretical account, with [[CITE]] markers",
    "PROCEDURE": "two paragraphs describing the experimental procedure in words only",
    "MATERIALS": "one paragraph describing the materials and measures in words only",
    "DISCUSSION": "four paragraphs of general discussion, with [[CITE]] markers",
    "LIMITATIONS": "one paragraph of limitations and future directions",
    "CONCLUSION": "one short concluding paragraph",
    "FOOTNOTE": "one footnote sentence clarifying a terminological choice",
}

_AUTHOR_YEAR = re.compile(r"\b[A-Z][a-z]+(?: et al\.)?,? \(?\b(?:19|20)\d\d")


def _prompt(topic: str, feedback: list[str]) -> str:
    lines = [
        "You are writing parts of a SYNTHETIC psychology journal article for a software benchmark that tests PDF extraction.",
        "The study is invented. Do not reproduce or paraphrase any real article and do not name real researchers.",
        f"TOPIC: {topic}",
        "",
        "HARD RULES (a violation makes the answer unusable):",
        "- Write NO digits at all: no numbers, years, percentages or statistics. Write words, never numerals.",
        "- Never write an author name or a year. Where you would cite prior work, write the marker [[CITE]].",
        "- Academic register; past tense for the study; vary sentence length; no bullet points; no headings inside blocks.",
        "- Paragraphs inside a block are separated by one blank line.",
        "",
        "Return every block below, each introduced by a line ===BLOCK_NAME=== and nothing else:",
    ]
    lines += [f"==={k}=== : {v}" for k, v in BLOCKS.items()]
    if feedback:
        lines += ["", "Your previous answer broke these rules; fix every one:"] + [f"- {f}" for f in feedback]
    return "\n".join(lines)


def _violations(blocks: dict) -> list[str]:
    out = []
    for name, text in blocks.items():
        if re.search(r"[0-9]", text):
            out.append(f"{name}: contains a digit")
        if _AUTHOR_YEAR.search(text):
            out.append(f"{name}: contains an author-year citation; use [[CITE]]")
    return out


def build_one(topic: str, model: str) -> dict:
    feedback: list[str] = []
    for attempt in range(1, 4):
        prompt = _prompt(topic, feedback)
        raw, meta = call_claude(prompt, model)
        try:
            blocks = parse_blocks(raw, list(BLOCKS))
        except ValueError as e:
            feedback = [str(e)]
            continue
        feedback = _violations(blocks)
        if not feedback:
            return {"topic": topic, "blocks": blocks, "attempts": attempt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "generation": meta}
    raise RuntimeError(f"{topic}: still broke rules after 3 attempts: {feedback[:4]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    bank = json.loads(BANK.read_text(encoding="utf-8")) if BANK.exists() else {"entries": {}}
    todo = [t for t in TOPICS if t not in bank["entries"]][: a.limit]
    print(f"{len(bank['entries'])} in bank, {len(todo)} to build", flush=True)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(build_one, t, a.model): t for t in todo}
        for f in cf.as_completed(futs):
            t = futs[f]
            try:
                bank["entries"][t] = f.result()
                BANK.write_text(json.dumps(bank, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
                print(f"ok   {t}", flush=True)
            except Exception as e:  # noqa: BLE001 - reported, not swallowed
                print(f"FAIL {t}: {e}", flush=True)
    missing = [t for t in TOPICS if t not in bank["entries"]]
    print(f"done: {len(bank['entries'])} in bank, {len(missing)} missing", flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
