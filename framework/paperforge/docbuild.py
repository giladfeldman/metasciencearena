"""Assemble a plain-text document and remember exactly where each tagged sentence landed.

A paperforge document is rendered as the text a PDF extractor would hand a tool:
headings on their own lines, paragraphs separated by blank lines, tables as aligned
rows. Every sentence the arena writes about a scored dimension is TAGGED, and the
builder records its character span in the final text. Gold evidence is therefore a
span we placed, not a span we searched for afterwards.

Also here: synthetic references. Author names are drawn from invented syllable
combinations, journals are fictional, and DOIs use the 10.5555 prefix that Crossref
reserves for examples, so no reference can be mistaken for, or resolve to, a real work.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field


@dataclass
class Doc:
    parts: list[str] = field(default_factory=list)
    _pending: list[tuple[str, str | None]] = field(default_factory=list)
    spans: list[dict] = field(default_factory=list)
    _text: str | None = None

    def heading(self, text: str) -> "Doc":
        self.parts.append(("H", text))
        return self

    def para(self, *sentences, tag: str | None = None) -> "Doc":
        """Add a paragraph. Each sentence is a str (untagged) or (str, tag), tag a str or tuple of str."""
        items = []
        for s in sentences:
            if s is None or s == "":
                continue
            items.append(s if isinstance(s, tuple) else (s, tag))
        if items:
            self.parts.append(("P", items))
        return self

    def raw(self, text: str) -> "Doc":
        self.parts.append(("R", text))
        return self

    def render(self) -> str:
        out: list[str] = []
        pos = 0

        def emit(chunk: str):
            nonlocal pos
            out.append(chunk)
            pos += len(chunk)

        for kind, payload in self.parts:
            if out:
                emit("\n\n")
            if kind in ("H", "R"):
                emit(payload)
                continue
            for i, (sent, tag) in enumerate(payload):
                if i:
                    emit(" ")
                # A tag may be a tuple: one sentence can be evidence for several dimensions
                # (a printed "t(df) = ..." names the outcome AND the model).
                for tg in (tag if isinstance(tag, tuple) else (tag,) if tag else ()):
                    self.spans.append({"tag": tg, "start": pos, "end": pos + len(sent), "text": sent})
                emit(sent)
        self._text = "".join(out)
        for s in self.spans:
            assert self._text[s["start"]:s["end"]] == s["text"]
        return self._text


_SYLL_A = ["Ber", "Kal", "Mor", "Tas", "Lin", "Dro", "Fen", "Hal", "Vas", "Quen", "Ros", "Tal",
           "Wen", "Or", "Pel", "Sar", "Ild", "Nor", "Cas", "Mir"]
_SYLL_B = ["ding", "vane", "stow", "ric", "mont", "aker", "ley", "ovic", "tern", "holm", "ida",
           "ssen", "ard", "quist", "ello", "man", "ver", "ski", "ton", "lund"]
_INITIALS = "ABCDEFGHJKLMNPRSTVW"
_JOURNALS = ["Journal of Applied Behavioural Inquiry", "Studies in Motivation and Affect",
             "Quarterly Review of Social Cognition", "Experimental Wellbeing Research",
             "Personality and Situational Processes", "Journal of Brief Interventions",
             "Behavioural Science Letters", "Archives of Everyday Psychology"]


class References:
    """A synthetic, well-formed APA reference list driven by [[CITE]] markers."""

    def __init__(self, rng: random.Random, topic_words: list[str]):
        self.rng = rng
        self.words = topic_words
        self.entries: list[dict] = []

    def _new(self) -> dict:
        r = self.rng
        n_auth = r.choice([1, 2, 2, 3, 3, 4])
        authors = [(r.choice(_SYLL_A) + r.choice(_SYLL_B), r.choice(_INITIALS) + ".") for _ in range(n_auth)]
        year = r.randint(2004, 2023)
        a, b = r.sample(self.words, 2)   # two DISTINCT constructs: "X on X" read as machine-made
        title = r.choice([
            f"The effect of {a} on {b}: Evidence from {r.choice(['three experiments', 'a field study', 'two online studies', 'a longitudinal panel'])}",
            f"When does {a} matter? A review of {b} research",
            f"{a[0].upper() + a[1:]} and {b} in everyday life",
            f"Rethinking {a}: Mechanisms and boundary conditions",
            f"A meta-analytic review of {a} interventions",
            f"{a[0].upper() + a[1:]} as a predictor of {b}: A preregistered replication",
            f"Individual differences in {a} and their links to {b}",
        ])
        vol, iss = r.randint(8, 61), r.randint(1, 6)
        p1 = r.randint(10, 900)
        doi = f"10.5555/synth.{r.getrandbits(40):010x}"
        e = {"authors": authors, "year": year, "title": title, "journal": r.choice(_JOURNALS),
             "volume": vol, "issue": iss, "pages": f"{p1}-{p1 + r.randint(8, 30)}", "doi": doi}
        self.entries.append(e)
        return e

    @staticmethod
    def _cite(e: dict) -> str:
        a = e["authors"]
        if len(a) == 1:
            names = a[0][0]
        elif len(a) == 2:
            names = f"{a[0][0]} & {a[1][0]}"
        else:
            names = f"{a[0][0]} et al."
        return f"{names}, {e['year']}"

    def fill(self, text: str) -> str:
        """Replace runs of [[CITE]] markers with parenthetical APA citations."""
        def repl(m):
            k = m.group(0).count("[[CITE]]")
            cites = []
            for _ in range(k):
                reuse = self.entries and self.rng.random() < 0.3
                cites.append(self._cite(self.rng.choice(self.entries) if reuse else self._new()))
            cites = sorted(set(cites))
            return "(" + "; ".join(cites) + ")"
        text = re.sub(r"\[\[CITE\]\](?:\s*[,;]?\s*\[\[CITE\]\])*", repl, text)
        # A marker the model put inside its own parentheses: "(see [[CITE]])" -> "(see (A, 2011))"
        return text.replace("((", "(").replace("))", ")")

    def render(self) -> list[str]:
        out = []
        for e in sorted(self.entries, key=lambda e: (e["authors"][0][0], e["year"])):
            a = e["authors"]
            names = ", ".join(f"{s}, {i}" for s, i in a[:-1])
            names = (names + ", & " if names else "") + f"{a[-1][0]}, {a[-1][1]}"
            out.append(f"{names} ({e['year']}). {e['title']}. {e['journal']}, {e['volume']}({e['issue']}), "
                       f"{e['pages']}. https://doi.org/{e['doi']}")
        return out
