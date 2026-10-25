"""Synthetic references and in-text citations in the styles real papers use.

Every reference is invented: syllable-built surnames (plus particle and diacritic
variants, catalog B11/C05), fictional venues, and DOIs under 10.5555 (reserved by
Crossref for examples), so nothing can be mistaken for or resolve to a real work.

`CitationBook` replaces each [[CITE]] run of the prose with a citation in the paper's
citation style and keeps the exact marker text and the reference ids it points to,
which is the citation-matching gold.
"""
from __future__ import annotations

import random
import re

from .model import Reference, Run

_SYLL_A = ["Ber", "Kal", "Mor", "Tas", "Lin", "Dro", "Fen", "Hal", "Vas", "Quen", "Ros", "Tal",
           "Wen", "Or", "Pel", "Sar", "Ild", "Nor", "Cas", "Mir", "Ast", "Bel", "Cor", "Dal"]
_SYLL_B = ["ding", "vane", "stow", "ric", "mont", "aker", "ley", "ovic", "tern", "holm", "ida",
           "ssen", "ard", "quist", "ello", "man", "ver", "ski", "ton", "lund", "berg", "ini"]
_PARTICLES = ["van", "de", "van der", "von", "Ben", "van 't", "De", "Van"]
_DIACRITIC = ["Kovač", "Großmann", "Özdal", "Bartoň", "Lindqvíst", "Szőke", "Albarrán", "Nuñez",
              "Müllner", "Šťastná"]
_ORGS = ["Open Inquiry Collaboration", "Synthetic Methods Consortium", "Behavioural Data Team",
         "Panel on Reporting Standards"]
_INITIALS = "ABCDEFGHJKLMNPRSTVW"
_VENUES = ["Journal of Applied Behavioural Inquiry", "Studies in Motivation and Affect",
           "Quarterly Review of Social Cognition", "Experimental Wellbeing Research",
           "Personality and Situational Processes", "Journal of Brief Interventions",
           "Behavioural Science Letters", "Archives of Everyday Psychology",
           "Judgment and Choice Reports", "Collected Papers in Decision Research"]
_ABBREV = {"Journal of Applied Behavioural Inquiry": "J Appl Behav Inq",
           "Studies in Motivation and Affect": "Stud Motiv Affect",
           "Quarterly Review of Social Cognition": "Q Rev Soc Cogn",
           "Experimental Wellbeing Research": "Exp Wellbeing Res",
           "Personality and Situational Processes": "Pers Situat Process",
           "Journal of Brief Interventions": "J Brief Interv",
           "Behavioural Science Letters": "Behav Sci Lett",
           "Archives of Everyday Psychology": "Arch Everyday Psychol",
           "Judgment and Choice Reports": "Judgm Choice Rep",
           "Collected Papers in Decision Research": "Collect Pap Decis Res"}

AUTHOR_YEAR = {"apa", "harvard", "aom"}
NUMERIC = {"numeric", "superscript"}


def _surname(rng: random.Random, flavour: str) -> str:
    base = rng.choice(_SYLL_A) + rng.choice(_SYLL_B)
    if flavour == "particle":
        return f"{rng.choice(_PARTICLES)} {base}"
    if flavour == "diacritic":
        return rng.choice(_DIACRITIC)
    if flavour == "compound":
        return f"{base}-{rng.choice(_SYLL_A)}{rng.choice(_SYLL_B)}"
    if flavour == "camel":
        return rng.choice(["Mc", "Le", "De", "Mac"]) + base
    return base


def make_reference(rng: random.Random, idx: int, topic_words: list[str], *, flavour: str = "plain",
                   kind: str = "article") -> Reference:
    n_auth = rng.choice([1, 2, 2, 3, 3, 4, 6, 8])
    authors = []
    for k in range(n_auth):
        fl = flavour if k == 0 else rng.choice(["plain"] * 6 + ["particle", "diacritic"])
        given = rng.choice(_INITIALS) + "." + (f" {rng.choice(_INITIALS)}." if rng.random() < 0.4 else "")
        authors.append((_surname(rng, fl), given))
    if flavour == "org":
        authors = [(rng.choice(_ORGS), "")]
    year = str(rng.randint(1998, 2025))
    a, b = rng.sample(topic_words, 2) if len(topic_words) >= 2 else ("choice", "judgment")
    title = rng.choice([
        f"The effect of {a} on {b}: Evidence from three experiments",
        f"When does {a} matter? A review of {b} research",
        f"{a[0].upper() + a[1:]} and {b} in everyday life",
        f"Rethinking {a}: Mechanisms and boundary conditions",
        f"A meta-analytic review of {a} interventions",
        f"{a[0].upper() + a[1:]} as a predictor of {b}: A preregistered replication",
        f"Is {a} really about {b}?",
        f"Individual differences in {a} and their links to {b}",
    ])
    p1 = rng.randint(10, 900)
    return Reference(ref_id=f"r{idx:03d}", authors=authors, year=year, title=title,
                     venue=rng.choice(_VENUES), volume=str(rng.randint(8, 61)),
                     issue=str(rng.randint(1, 6)), fpage=str(p1), lpage=str(p1 + rng.randint(8, 30)),
                     doi=f"10.5555/synth.{rng.getrandbits(40):010x}", kind=kind)


# ------------------------------------------------------------------ reference list text
def _apa_authors(ref: Reference) -> str:
    a = ref.authors
    if not a[0][1]:
        return a[0][0]
    names = [f"{s}, {g}" for s, g in a[:20]]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def _t(title: str) -> str:
    """A title followed by its period, unless it already ends in ? or ! (APA)."""
    return title if title.endswith(("?", "!")) else title + "."


def render_reference(ref: Reference, style: str, number: int | None = None) -> str:
    a = ref.authors
    doi = f"https://doi.org/{ref.doi}" if ref.doi else ""
    pages = f"{ref.fpage}–{ref.lpage}" if ref.fpage else ""
    if style == "apa":
        s = f"{_apa_authors(ref)} ({ref.year}). {_t(ref.title)} {ref.venue}, {ref.volume}({ref.issue}), {pages}. {doi}"
    elif style == "harvard":
        names = " and ".join(f"{s} {g.replace('.', '').replace(' ', '')}" for s, g in a[:3]) + (" et al." if len(a) > 3 else "")
        s = f"{names} ({ref.year}) {_t(ref.title)} {ref.venue} {ref.volume}({ref.issue}): {pages}."
    elif style == "aom":
        names = ", ".join(f"{s}, {g}" for s, g in a[:-1]) + (", & " if len(a) > 1 else "") + f"{a[-1][0]}, {a[-1][1]}"
        s = f"{names} {ref.year}. {_t(ref.title)} {ref.venue}, {ref.volume}: {pages}."
    elif style == "vancouver":
        names = ", ".join(f"{s} {g.replace('.', '').replace(' ', '')}" for s, g in a[:6]) + (", et al" if len(a) > 6 else "")
        s = f"{number}. {names}. {_t(ref.title)} {_ABBREV.get(ref.venue, ref.venue)}. {ref.year};{ref.volume}({ref.issue}):{ref.fpage}-{ref.lpage}. doi:{ref.doi}"
    elif style == "ieee":
        names = ", ".join(f"{g} {s}" for s, g in a[:6]) + (" et al." if len(a) > 6 else "")
        s = f"[{number}] {names}, “{ref.title},” {_ABBREV.get(ref.venue, ref.venue)}, vol. {ref.volume}, no. {ref.issue}, pp. {ref.fpage}–{ref.lpage}, {ref.year}."
    elif style == "nature":
        names = ", ".join(f"{s}, {g}" for s, g in a[:5]) + (" et al." if len(a) > 5 else "")
        s = f"{number}. {names} {_t(ref.title)} {_ABBREV.get(ref.venue, ref.venue)} {ref.volume}, {ref.fpage}–{ref.lpage} ({ref.year})."
    else:
        raise ValueError(style)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------- citations
def _cite_name(ref: Reference, style: str) -> str:
    a = ref.authors
    amp = " & " if style == "apa" else " and "
    if len(a) == 1:
        return a[0][0]
    if len(a) == 2:
        return f"{a[0][0]}{amp}{a[1][0]}"
    return f"{a[0][0]} et al."


class CitationBook:
    def __init__(self, rng: random.Random, topic_words: list[str], cite_style: str, ref_style: str,
                 pool_size: int = 30):
        self.rng = rng
        self.cite_style = cite_style
        self.ref_style = ref_style
        flavours = ["plain"] * 14 + ["particle", "particle", "diacritic", "diacritic", "compound",
                                     "camel", "org"]
        self.pool = [make_reference(rng, i, topic_words, flavour=rng.choice(flavours))
                     for i in range(pool_size)]
        self._disambiguate_years()
        self.cited: list[str] = []          # ref ids in first-cited order
        self.markers: list[dict] = []       # {"marker_text", "reference_ids"}

    def _disambiguate_years(self) -> None:
        seen: dict[tuple, list[Reference]] = {}
        for r in self.pool:
            seen.setdefault((r.authors[0][0], len(r.authors) > 2, r.year), []).append(r)
        for group in seen.values():
            if len(group) > 1:
                for k, r in enumerate(sorted(group, key=lambda r: r.title)):
                    r.year = r.year + "abcdefgh"[k]

    def _number(self, ref_id: str) -> int:
        if ref_id not in self.cited:
            self.cited.append(ref_id)
        return self.cited.index(ref_id) + 1

    def cite(self, k: int, narrative: bool = False) -> list[Run]:
        """A citation of k references, as runs. Records the marker gold."""
        uncited = [r for r in self.pool if r.ref_id not in self.cited]
        refs = []
        for _ in range(k):
            if uncited and (not self.cited or self.rng.random() < 0.75):
                r = uncited.pop(self.rng.randrange(len(uncited)))
            else:
                r = self.rng.choice([x for x in self.pool if x.ref_id in self.cited] or self.pool)
            if r not in refs:
                refs.append(r)
        ids = [r.ref_id for r in refs]
        st = self.cite_style
        if st in NUMERIC:
            nums = sorted({self._number(i) for i in ids})
            txt = _compress(nums)
            self.markers.append({"marker_text": txt if st == "superscript" else f"[{txt}]", "reference_ids": ids})
            if st == "superscript":
                return [Run(txt, style="sup", cite_refs=ids)]
            return [Run(f"[{txt}]", cite_refs=ids)]
        for i in ids:
            self._number(i)
        sep = "; "
        if narrative and len(refs) == 1:
            r = refs[0]
            txt = f"{_cite_name(r, st)} ({r.year})"
        else:
            refs.sort(key=lambda r: (r.authors[0][0], r.year))
            if st == "harvard":
                body = sep.join(f"{_cite_name(r, st)} {r.year}" for r in refs)
            else:
                body = sep.join(f"{_cite_name(r, st)}, {r.year}" for r in refs)
            txt = f"({body})"
        self.markers.append({"marker_text": txt, "reference_ids": ids})
        return [Run(txt, cite_refs=ids)]

    def reference_list(self) -> list[Reference]:
        used = [r for r in self.pool if r.ref_id in self.cited]
        if self.cite_style in NUMERIC:
            used.sort(key=lambda r: self.cited.index(r.ref_id))
        else:
            used.sort(key=lambda r: (r.authors[0][0].lower(), r.year))
        for n, r in enumerate(used, 1):
            r.raw_text = render_reference(r, self.ref_style, n)
        return used


def _compress(nums: list[int]) -> str:
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ",".join(out)


STYLE_PAIRS = [("apa", "apa"), ("harvard", "harvard"), ("aom", "aom"), ("numeric", "vancouver"),
               ("numeric", "ieee"), ("superscript", "nature")]
