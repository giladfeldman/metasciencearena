"""Content-based pairing of bibliography entries, shared by the two citation arenas.

`pdf-reference-parsing-v1` needs to know which player entry is which gold entry
before it can grade fields; `pdf-citation-matching-v1` needs the same crosswalk
to translate a player's own reference ids (GROBID's `b0`, an LLM's `"12"`) into
the gold's (`R12`, `ref12`, `bib0060`). One pairing, one definition, both arenas.

Why content and not keys. The v1 reference scorer paired on DOI, else on
(first-author surname, year, title[:40]). Measured 2026-09-28 against GROBID
0.8.1 on the 30 real PMC papers, 8 papers scored exactly 0.0 with about the right
number of entries on both sides, because:

  * 12 of 30 JATS files spell names as `<string-name>`, which the gold parser
    skipped, so gold `authors` was `[]` and the surname key never matched;
  * the gold carries DOIs from `<pub-id>` that the printed PDF often does not,
    so pass 1 keyed gold on DOI and the player on surname -- two namespaces;
  * Nature-style references have no article title at all.

A key built from any single field inherits every parse difference of that field.
The pairing below scores each candidate pair on the evidence both sides DO have
-- an equal DOI, a near-equal title, or the overlap of everything printed in the
entry -- and assigns one-to-one, best pairs first.

Deliberately dependency-free (standard library only): scorers run on CI.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache

#: A pair below this similarity is not a match. 0.5 Dice on the entry's content
#: tokens means half of everything printed agrees; measured on the 30-paper PMC
#: corpus, true pairs sit far above it and the best WRONG candidate of a matched
#: entry sits below it (see tests/test_references_pairing.py for the fixtures).
MIN_PAIR_SIMILARITY = 0.5

#: A title ratio this high pairs on its own; below it the first authors must agree.
TITLE_ALONE = 0.85

#: Pairs whose first authors differ need at least this similarity (or equal DOIs).
DIFFERENT_FIRST_AUTHOR = 0.8

#: When the first authors differ, a title counts only if it is this close AND the
#: rest of the entry agrees too: "Effects of X in adults" / "... in children" by
#: different authors are 0.89 alike and are different works (Sonnet, 2026-09-30).
TITLE_ALONE_OTHER_AUTHOR = 0.95

#: A player entry pairs only with a gold entry within this margin of its own best
#: candidate. A parser that lists one reference twice otherwise got the copy
#: credited to a sibling entry (same author, year, journal) at 0.67.
PLAYER_MARGIN = 0.25

_DOI_PREFIX = re.compile(r"^((https?://)?(www\.)?(dx\.)?doi\.org/|(urn:)?doi:\s*)", re.I)
_TOKEN = re.compile(r"[a-z0-9]+")
#: Letters NFKD does not decompose; without this they were DELETED ("Weiß" -> "wei").
_TRANSLIT = str.maketrans({"ß": "ss", "ø": "o", "æ": "ae", "œ": "oe", "ł": "l", "đ": "d",
                           "ð": "d", "þ": "th", "ı": "i"})


def fold(text: str | None) -> str:
    """NFKD, strip combining marks, lowercase, transliterate, collapse whitespace."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip().lower().translate(_TRANSLIT)


def norm_doi(doi: str | None) -> str:
    from urllib.parse import unquote
    d = fold(unquote(str(doi))) if doi else ""
    d = _DOI_PREFIX.sub("", d)
    return d.rstrip(".,;)/ ")


def words(text: str | None) -> list[str]:
    """Alphanumeric word tokens of the folded text."""
    return _TOKEN.findall(fold(text))


def norm_field(value) -> str:
    """Comparison form of a scalar field: folded, punctuation-free, one space."""
    return " ".join(words(value if value is None else str(value)))


def surnames(entry: dict) -> list[str]:
    out = []
    for a in entry.get("authors") or []:
        if isinstance(a, dict):
            s = norm_field(a.get("surname"))
            if s:
                out.append(s)
    return out


def _bag(entry: dict) -> set[str]:
    """Every token printed in the entry, from the raw string AND the parsed fields.

    Both, because each side is missing something: GROBID's reconstructed string
    glues initials onto surnames ("NMeuschke"), while JATS `raw_text` glues the
    DOI onto the page number. The union gives the overlap a fair chance either way.
    """
    parts = [entry.get("raw_text"), entry.get("title"), entry.get("venue"),
             entry.get("year"), entry.get("volume"), entry.get("issue"),
             entry.get("fpage"), entry.get("lpage")]
    toks: set[str] = set()
    for p in parts:
        if p:
            toks.update(t for t in words(str(p)) if len(t) >= 2 or t.isdigit())
    toks.update(surnames(entry))
    return toks


def _dice(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


class _Profile:
    """Everything `similarity` reads from one entry, computed once."""
    __slots__ = ("doi", "bag", "nums", "title", "title_words", "title_chars", "_sm",
                 "year", "fpage", "first", "first_raw")

    def __init__(self, entry: dict):
        self.doi = norm_doi(entry.get("doi"))
        self.bag = _bag(entry)
        self.nums = _numbers(self.bag)
        self.title = norm_field(entry.get("title"))
        self.title_words = set(self.title.split())
        self.title_chars = Counter(self.title)
        self._sm = None
        self.year = norm_field(entry.get("year"))[:4]
        self.fpage = re.sub(r"\D", "", str(entry.get("fpage") or "")).lstrip("0")
        names = surnames(entry)
        self.first = names[0] if names else ""
        self.first_raw = ""
        for a in entry.get("authors") or []:
            if isinstance(a, dict) and norm_field(a.get("surname")):
                self.first_raw = str(a.get("surname"))
                break


    def matcher(self) -> SequenceMatcher:
        """A matcher whose second sequence is this entry's title, built once."""
        if self._sm is None:
            self._sm = SequenceMatcher(None, "", self.title)
        return self._sm


def _numbers(bag: set[str]) -> set[str]:
    return {t.lstrip("0") for t in bag if t.isdigit()}


def _profile_similarity(p: _Profile, g: _Profile) -> float:
    if p.doi and g.doi and p.doi == g.doi:
        # An equal DOI pairs unless both titles are present and plainly differ:
        # GROBID attached a NEIGHBOUR's DOI to an entry on real papers, pairing
        # it with the wrong work at 1.0 (measured 2026-09-30).
        if not (len(p.title_words) >= 3 and len(g.title_words) >= 3
                and SequenceMatcher(None, p.title, g.title).ratio() < MIN_PAIR_SIMILARITY):
            return 1.0
    # Character-level title similarity, so a title an extractor garbled ("Sress
    # ad memoy consolldaton") still outranks a sibling entry by the same author,
    # year and journal whose PAGES happen to match -- the bag alone picked that
    # decoy (Sonnet, 2026-09-28 consult). quick_ratio() is a cheap upper bound,
    # so the full ratio runs only on candidates.
    # The upper bound is computed from character counts cached per entry, and one
    # matcher per gold entry is reused (its second sequence is indexed once):
    # building a matcher for every pair made a 528-reference paper take 50 s.
    quick = 0.0
    if len(p.title_words) >= 3 and len(g.title_words) >= 3:
        quick = 2 * sum((p.title_chars & g.title_chars).values()) / (len(p.title) + len(g.title))
    sm = None
    if quick >= MIN_PAIR_SIMILARITY:
        sm = g.matcher()
        sm.set_seq1(p.title)
    # Without strong title agreement, a clash in year or first page is evidence
    # of a DIFFERENT work: two papers by one author in one journal volume, or a
    # sibling whose own entry the player skipped (Sonnet, 2026-09-30). A value
    # counts as clashing only if it appears NOWHERE in the other entry's printed
    # text: measured on GROBID's real output, true pairs had a year from an access
    # date, a year inside a title ("1990-2019"), a page read as the year ("1992"),
    # a PubMed id read as the first page, or "e0119" for "119" -- all printed on
    # both sides, or over 5 digits (an identifier, not a page).
    clash = (p.year.isdigit() and g.year.isdigit() and len(p.year) == len(g.year) == 4
             and p.year != g.year and p.year.lstrip("0") not in g.nums
             and g.year.lstrip("0") not in p.nums) or \
            (p.fpage and g.fpage and len(p.fpage) <= 5 and len(g.fpage) <= 5
             and p.fpage != g.fpage and p.fpage not in g.nums and g.fpage not in p.nums)
    if clash and quick < TITLE_ALONE:
        return 0.0
    ratio = sm.ratio() if sm is not None else 0.0
    if clash and ratio < TITLE_ALONE:
        return 0.0
    same_first = _same_author(p.first, g.first, p.first_raw, g.first_raw)
    best = _dice(p.bag, g.bag)
    if ratio:
        # The ratio alone is not enough evidence: two DIFFERENT works with the
        # same frame ("A meta-analysis of the effects of X on Y") reach 0.73
        # (review finding, 2026-09-30). So the title route counts only when the
        # titles are near-identical (>= TITLE_ALONE) or the first authors agree;
        # when both name a first author and the names differ, the title must be
        # closer still AND the rest of the entry must agree ("... in adults" /
        # "... in children" by two authors are 0.89 alike).
        differ = bool(p.first and g.first) and not same_first
        if differ:
            if ratio >= TITLE_ALONE_OTHER_AUTHOR and best >= MIN_PAIR_SIMILARITY:
                best = max(best, ratio)
        elif ratio >= TITLE_ALONE or (same_first and ratio >= MIN_PAIR_SIMILARITY):
            best = max(best, ratio)
    if p.first and g.first and not same_first and best < DIFFERENT_FIRST_AUTHOR:
        # Both name a first author and the names differ: that is direct evidence
        # of a different work, so only strong overall agreement may pair them.
        # Without this, two titles sharing a frame ("A meta-analysis of the
        # effects of X on Y") paired on word overlap alone (review, 2026-09-30).
        return 0.0
    return best


def _author_key(name: str) -> str:
    """Letters only, folded."""
    return re.sub(r"[^a-z]", "", fold(name))


def _umlaut_key(name: str) -> str:
    """The key with German umlaut spellings merged (Mueller = Müller). Only used
    when one side prints an umlaut: merged everywhere, "Rae" would equal "Ra"."""
    return re.sub(r"([aou])e", r"\1", _author_key(name))


def _name_parts(name: str) -> list[str]:
    """The keys of the parts of a compound surname: "van der Berg", "Smith-Jones"."""
    return [_author_key(x) for x in re.split(r"[\s\-]+", name or "") if _author_key(x)]


def _same_author(a: str, b: str, a_raw: str = "", b_raw: str = "") -> bool:
    """Whether two first-author surnames name the same person, as printed.

    Spelling variants are the same author (Sonnet, 2026-09-30: every one of these
    was treated as a DIFFERENT author, which vetoed true pairs with a garbled or
    missing title): transliteration (Weiß/Weiss, Łukasz/Lukasz, Müller/Mueller),
    punctuation (O'Brien/OBrien), and a compound surname against one of its
    parts of 3+ letters (van der Berg/Berg, Smith-Jones/Smith).

    Capital initials an extractor GLUED onto a surname ("ASmith", "XLi") are
    tolerated, judged on the raw spelling: 1-3 capitals directly followed by the
    surname proper, which starts with a capital and continues in lowercase. A
    surname that merely ends with another (Kang/Ang, Hansen/Sen) is a different
    author, and so is an ALL-CAPS "KANG" against "ANG": glued initials cannot be
    told from a longer surname there (review, attempt 3).
    """
    return _same_author_raw(a_raw or a, b_raw or b)


@lru_cache(maxsize=65536)
def _same_author_raw(a_raw: str, b_raw: str) -> bool:
    ka, kb = _author_key(a_raw), _author_key(b_raw)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    if re.search(r"[äöüÄÖÜ]", unicodedata.normalize("NFC", a_raw + b_raw)) \
            and _umlaut_key(a_raw) == _umlaut_key(b_raw):
        return True
    pa, pb = _name_parts(a_raw), _name_parts(b_raw)
    if (len(pa) > 1 or len(pb) > 1) and \
            any(len(x) >= 3 and x in pb for x in pa):
        return True
    for long_raw, short_key in ((a_raw, kb), (b_raw, ka)):
        r = re.sub(r"\s+", "", long_raw)
        for k in (1, 2, 3):
            head, tail = r[:k], r[k:]
            if head.isalpha() and head.isupper() and tail[:1].isupper() \
                    and any(c.islower() for c in tail[1:]) and _author_key(tail) == short_key:
                return True
    return False


def similarity(p: dict, g: dict) -> float:
    """How strongly player entry `p` and gold entry `g` look like the same work."""
    return _profile_similarity(_Profile(p), _Profile(g))


def pair_references(player: list[dict], gold: list[dict],
                    min_similarity: float = MIN_PAIR_SIMILARITY) -> list[tuple[int, int]]:
    """One-to-one (player_index, gold_index) pairs, highest similarity first.

    Ties break on (player index, gold index) so the result is deterministic.
    """
    gp = [_Profile(g) if isinstance(g, dict) else None for g in gold]
    cands: list[tuple[float, int, int]] = []
    for i, p in enumerate(player):
        if not isinstance(p, dict):
            continue
        pp = _Profile(p)
        mine = [(_profile_similarity(pp, g), j) for j, g in enumerate(gp) if g is not None]
        top = max((s for s, _ in mine), default=0.0)
        for s, j in mine:
            if s >= min_similarity and s >= top - PLAYER_MARGIN:
                cands.append((-s, i, j))
    cands.sort()
    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _neg, i, j in cands:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((i, j))
    pairs.sort()
    return pairs
