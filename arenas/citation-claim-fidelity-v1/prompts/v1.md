You check whether sentences in a manuscript describe a cited source accurately. Your only evidence is the cited source's abstract, printed below. You do not have the full paper.

CITED SOURCE
Title: {title}
Year: {year}
Abstract:
"""
{abstract}
"""

CITING SENTENCES (each one cites the source above)
{sentences}

For EACH numbered sentence, judge only the claim it attributes to the cited source, and give one verdict:
- "consistent": the abstract supports that claim as stated. A faithful paraphrase or summary is consistent.
- "contradicts": the abstract conflicts with that claim. Examples: the opposite direction of an effect or finding; a different number; a different study design, method or measure; a different population; a claim stated more strongly, more generally or with more certainty than the abstract reports.
- "silent": the abstract neither supports nor contradicts that claim, so the abstract alone cannot settle it. The claim may still be true in the full paper.

Rules:
- Judge each sentence on its own. The sentences may describe the source differently; one verdict does not imply another.
- "evidence_quote" must be copied exactly, word for word, from the abstract, and must be the shortest span that decides the verdict. For a "silent" verdict, set "evidence_quote" to "".
- "reason" is one short sentence.

Return only this JSON object, with one item per sentence, in order:
{{"items": [{{"id": 1, "verdict": "consistent" | "contradicts" | "silent", "evidence_quote": "...", "reason": "..."}}]}}
