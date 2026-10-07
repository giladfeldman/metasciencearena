"""framework/scoring/references.py: content-based pairing of bibliography entries.

The v1 reference scorer keyed pairs on first-author surname + year + title[:40],
and 8 of 30 real papers scored exactly 0.0 for a parser that had found the
references. Each case below is one of the measured reasons it failed.
"""
from framework.scoring.references import norm_doi, pair_references, similarity


def _ref(rid, *, raw="", title=None, year=None, doi=None, surnames=(), venue=None,
         volume=None, fpage=None):
    return {"reference_id": rid, "raw_text": raw, "title": title, "year": year, "doi": doi,
            "venue": venue, "volume": volume, "fpage": fpage,
            "authors": [{"surname": s, "given_names": None} for s in surnames]}


def test_doi_forms_normalise_to_one_value():
    assert norm_doi("https://doi.org/10.1/ABC.") == "10.1/abc"
    assert norm_doi("doi: 10.1/abc") == "10.1/abc"
    assert norm_doi(None) == ""


def test_equal_doi_pairs_even_when_nothing_else_agrees():
    assert similarity(_ref("p", doi="10.5/x"), _ref("g", doi="https://doi.org/10.5/X")) == 1.0


def test_gold_without_authors_still_pairs_on_content():
    # 12 of 30 JATS files wrote names as <string-name>, which the v1 gold
    # parser dropped: gold authors [] never matched a parsed surname.
    player = [_ref("b0", raw="Sleep duration in early adolescence MAbernathy LDelacroix 2015",
                   title="Sleep duration in early adolescence and risk of anxiety",
                   year="2015", surnames=("Abernathy", "Delacroix"))]
    gold = [_ref("ref2", raw="Abernathy M, Delacroix L (2015) Sleep duration in early adolescence "
                             "and risk of anxiety. J Sleep Res",
                 title="Sleep duration in early adolescence and risk of anxiety",
                 year="2015", doi="10.0000/placeholder.one")]
    assert pair_references(player, gold) == [(0, 0)]


def test_untitled_nature_style_entry_pairs_on_printed_numbers():
    player = [_ref("b1", raw="MKovar JLund ZChao KAmin Adv. Mater 31 1811234 2018",
                   title="Adv. Mater", year="2018", surnames=("Kovar", "Lund", "Chao", "Amin"))]
    gold = [_ref("bib2", raw="2 M. Kovar, J. Lund, Z. Chao, K. Amin, Adv. Mater. 2018, 31, "
                             "1811234.10.0000/placeholder.two",
                 year="2018", venue="Adv. Mater.", volume="31", fpage="1811234",
                 doi="10.0000/placeholder.two", surnames=("Kovar", "Lund", "Chao", "Amin"))]
    assert pair_references(player, gold) == [(0, 0)]


def test_unrelated_entries_do_not_pair():
    player = [_ref("x", raw="Smith J (2018). The effect of temperature on growth. J Bot 3, 1-9.",
                   title="The effect of temperature on growth", year="2018")]
    gold = [_ref("g", raw="Nakagawa A (2019). Mechanisms of loss aversion. Cognition 12, 5-9.",
                 title="Mechanisms of loss aversion", year="2019")]
    assert pair_references(player, gold) == []


def test_pairing_is_one_to_one_and_prefers_the_better_match():
    gold = [_ref("r1", raw="Doe J (2020). Anchoring in estimation. Cognition 4, 1-9.",
                 title="Anchoring in estimation", year="2020"),
            _ref("r2", raw="Doe J (2021). Anchoring in estimation revisited. Cognition 5, 2-8.",
                 title="Anchoring in estimation revisited", year="2021")]
    player = [_ref("a", raw="Doe J (2021). Anchoring in estimation revisited. Cognition 5, 2-8.",
                   title="Anchoring in estimation revisited", year="2021"),
              _ref("b", raw="Doe J (2020). Anchoring in estimation. Cognition 4, 1-9.",
                   title="Anchoring in estimation", year="2020")]
    assert sorted(pair_references(player, gold)) == [(0, 1), (1, 0)]


def test_duplicate_gold_entries_each_take_one_player_entry():
    same = dict(raw="Weston LM (1987). Viral DNA in tissue samples. J Path Methods 12, 86-91.",
                title="Viral DNA in tissue samples", year="1987", doi="10.1/w")
    gold = [_ref("B15", **same), _ref("B39", **same)]
    player = [_ref("p1", **same), _ref("p2", **same)]
    assert pair_references(player, gold) == [(0, 0), (1, 1)]


def test_a_garbled_title_still_beats_a_sibling_whose_pages_match():
    """Sonnet, 2026-09-28 consult: same author, year, journal and issue; the
    player's title is garbled and its pages match the OTHER entry. The token bag
    alone paired it with that decoy (0.714 vs 0.571)."""
    base = dict(surnames=("Zhang",), year="2023", venue="Journal of Cell Biology", volume="45")
    g1 = _ref("g1", title="Stress and memory consolidation", fpage="101",
              raw="Zhang L (2023). Stress and memory consolidation. J Cell Biol, 45(2), 101-110.", **base)
    g2 = _ref("g2", title="Sleep deprivation and attention", fpage="111",
              raw="Zhang L (2023). Sleep deprivation and attention. J Cell Biol, 45(2), 111-120.", **base)
    p = _ref("p", title="Sress ad memoy consolldaton", fpage="111",
             raw="Zhang L (2023). Sress ad memoy consolldaton. J Cell Biol, 45(2), 111-120.", **base)
    assert pair_references([p], [g1, g2]) == [(0, 0)]


def test_same_frame_titles_by_different_authors_do_not_pair():
    """Review 2026-09-30: DIFFERENT works sharing a title frame reached 0.70-0.81
    on word overlap alone and paired. Different first authors now require 0.8."""
    for a, b in (("Effects of sleep deprivation on working memory performance in adults",
                  "Effects of sleep restriction on working memory and attention in children"),
                 ("A meta-analysis of the effects of mindfulness on depression",
                  "A meta-analysis of the effects of exercise on anxiety")):
        p = [_ref("p", raw=a, title=a, year="2020", surnames=("Abernathy",))]
        g = [_ref("g", raw=b, title=b, year="2020", surnames=("Delacroix",))]
        assert pair_references(p, g) == [], (a, b)


def test_a_near_identical_title_pairs_without_authors():
    title = "Anchoring effects in numerical estimation across cultures"
    p = [_ref("p", raw=title.lower(), title=title.lower(), year="2021")]
    g = [_ref("g", raw=title, title=title, year="2021")]
    assert pair_references(p, g) == [(0, 0)]


def test_glued_initials_do_not_break_the_first_author_check():
    """Review 2026-09-30: "ASmith" (initial glued on by an extractor) vs "Smith"
    with a garbled title lost a true pair."""
    base = dict(year="2019", venue="Cognition", volume="12", fpage="40")
    p = [_ref("p", raw="Anchrng efects in estmation ASmith Cognition 12 40 2019",
              title="Anchrng efects in estmation", surnames=("ASmith",), **base)]
    g = [_ref("g", raw="Smith A (2019). Anchoring effects in estimation. Cognition, 12, 40-52.",
              title="Anchoring effects in estimation", surnames=("Smith",), **base)]
    assert pair_references(p, g) == [(0, 0)]


def test_suffix_surnames_are_not_glued_initials():
    """Review attempt 3: "Kang" vs "Ang" (and "Hansen"/"Sen", "Yang"/"Ang") are
    different authors; only CAPITAL initials glued before a capital surname are
    the extractor artefact. Two same-frame titles must not pair across them."""
    from framework.scoring.references import _same_author
    assert not _same_author("kang", "ang", "Kang", "Ang")
    assert not _same_author("hansen", "sen", "Hansen", "Sen")
    assert _same_author("asmith", "smith", "ASmith", "Smith")
    # all-caps printing: "KANG" could be K+ANG or a surname -- no tolerance
    assert not _same_author("kang", "ang", "KANG", "ANG")
    assert _same_author("smith", "smith", "SMITH", "SMITH")
    p = [_ref("p", raw="Kang 2019 A meta-analysis of the effects of sleep on memory Cognition 12 40",
              title="A meta-analysis of the effects of sleep on memory", surnames=("Kang",),
              year="2019", venue="Cognition", volume="12", fpage="40")]
    g = [_ref("g", raw="Ang 2019 A meta-analysis of the effects of exercise on mood Psychol Sci 30 977",
              title="A meta-analysis of the effects of exercise on mood", surnames=("Ang",),
              year="2019", venue="Psychol Sci", volume="30", fpage="977")]
    assert pair_references(p, g) == []


# --- tier-2 (Sonnet) findings, 2026-09-30. Invented names and titles throughout. ---

def test_spelling_variants_of_one_surname_are_the_same_author():
    from framework.scoring.references import _same_author
    for a, b in [("Weiß", "Weiss"), ("Łukasz", "Lukasz"), ("Müller", "Mueller"), ("O'Brien", "OBrien"),
                 ("van der Berg", "Berg"), ("Smith-Jones", "Smith"), ("Sørensen", "Sorensen"),
                 ("XLi", "Li"), ("ASmith-Jones", "Smith-Jones")]:
        assert _same_author("", "", a, b), (a, b)
    # an umlaut merge applied everywhere would make these equal
    for a, b in [("Rae", "Ra"), ("Lue", "Lu"), ("D'Angelo", "Angelo")]:
        assert not _same_author("", "", a, b), (a, b)


def test_a_spelling_variant_does_not_veto_a_pair_with_a_garbled_title():
    p = [_ref("p", title="Sres nd mmry consoldtn n rts", surnames=("Weiß",), year="2019")]
    g = [_ref("g", title="Stress and memory consolidation in rats", surnames=("Weiss",), year="2019")]
    assert pair_references(p, g) == [(0, 0)]


def test_a_page_clash_without_title_agreement_is_a_different_work():
    """One author, one journal volume, different first page: the player's own
    counterpart is missing, so this must be fp + fn, not a true positive."""
    g = [_ref("g", surnames=("Rao",), year="2014", venue="Nature", volume="512", fpage="61")]
    p = [_ref("p", surnames=("Rao",), year="2014", venue="Nature", volume="512", fpage="99")]
    assert pair_references(p, g) == []
    p2 = [_ref("p", title="Stress hormones and memory in ageing", surnames=("Smith",), year="2020",
               venue="J Aging", volume="4", fpage="50")]
    g2 = [_ref("g", title="Cortisol reactivity and working memory in older adults", surnames=("Smith",),
               year="2020", venue="J Aging", volume="4", fpage="10")]
    assert pair_references(p2, g2) == []


def test_a_misread_year_that_is_printed_in_the_other_entry_is_not_a_clash():
    """GROBID read an access date / a page as the year; the value is printed in
    the gold entry too, so it is not evidence of a different work."""
    g = [_ref("g", raw="Anon. Survey report on things. Agency; 2025. Accessed 2023.", title=None,
              year="2025", venue="Agency")]
    p = [_ref("p", raw="Anon. Survey report on things. Agency; 2025. Accessed 2023.",
              title="Survey report on things", year="2023", venue="Agency")]
    assert pair_references(p, g) == [(0, 0)]


def test_a_neighbours_doi_does_not_pair_a_different_title():
    """A parser attached the NEXT entry's DOI; the same author and title win."""
    g = [_ref("g0", title="Typologies of reading fluency in children", surnames=("Ortega",), year="2018",
              volume="12", fpage="208", doi="10.0000/g0"),
         _ref("g1", title="Working memory capacity in older adults", surnames=("Lind",), year="2018",
              volume="9", fpage="315", doi="10.0000/g1")]
    p = [_ref("p", title="Working memory capacity in older adults", surnames=("Lind",), year="2018",
              volume="9", fpage="315", doi="10.0000/g0")]
    assert pair_references(p, g) == [(0, 1)]


def test_a_duplicated_player_entry_is_not_credited_to_a_sibling():
    g = [_ref("g0", surnames=("Rao",), year="2014", venue="Nature", volume="512", fpage="45"),
         _ref("g1", surnames=("Rao",), year="2014", venue="Nature", volume="512", fpage="61")]
    p = [dict(g[0]), dict(g[0])]
    assert pair_references(p, g) == [(0, 0)]
    assert pair_references([dict(g[0]), dict(g[1])], g) == [(0, 0), (1, 1)]


def test_different_first_authors_need_more_than_a_similar_title():
    p = [_ref("p", title="Effects of sleep deprivation on working memory in adults", surnames=("Ang",),
              year="2019", volume="3", fpage="5")]
    g = [_ref("g", title="Effects of sleep deprivation on working memory in children", surnames=("Kang",),
              year="2019", volume="7", fpage="90")]
    assert pair_references(p, g) == []


def test_more_doi_spellings_normalise():
    for d in ("doi.org/10.1/abc", "https://www.doi.org/10.1/abc", "urn:doi:10.1/abc",
              "https://doi.org/10.1/abc/", "https://doi.org/10.1%2Fabc"):
        assert norm_doi(d) == "10.1/abc", d
