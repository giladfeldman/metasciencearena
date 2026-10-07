# Meta Science Arena Taxonomy

Curated registry of meta-science challenges across the empirical-science lifecycle. **First trial — to be re-evaluated with collaborators.** Auto-generated from `stages.yaml` and `leaves/*.yaml` by `scripts/render_readme.py`.

See the design spec at `docs/superpowers/specs/2026-04-29-sciencearena-taxonomy-and-contract-design.md`.

## 01 — Ideation & literature scoping

_Researcher decides what to study, scans prior work._

*(no leaves yet)*

## 02 — Preregistration & design

_Hypotheses, design, analysis plan committed before data._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `prereg-vs-paper-deviation-detection` | Preregistration vs. paper deviation detection | live | A |
| `power-analysis-reporting` | Power-analysis reporting detection & extraction | live | A |
| `prereg-extraction` | Preregistration link detection & field extraction | live | A |
| `trial-registration-consistency` | Trial registration timing and registration-vs-report consistency | design | B |
| `prereg-quality-assessment` | Preregistration quality and completeness assessment | design | A |

## 03 — Data collection & provenance

_Subjects/samples/measurements gathered._

*(no leaves yet)*

## 04 — Data sharing & documentation

_Data deposited, codebooks, licensing._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `transparency-statement-detection` | Transparency-statement detection (COI, funding, open practices) | live | A |

## 05 — Analysis & code

_Statistical analysis, reproducible code._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `code-and-repo-reproducibility-checks` | Code & repository reproducibility checks | live | A |
| `statistical-syntax-translation-to-r` | Statistical syntax translation to R | live | A |
| `code-paper-consistency` | Reported-analysis vs analysis-code consistency | design | A |

## 06 — Results reporting

_Numbers, tables, figures in the paper._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `nhst-stats-and-effect-size-extraction` | NHST stats + effect-size + CI extraction from manuscripts | live | A |
| `statistical-reporting-completeness` | Statistical reporting completeness & p-value precision | live | A |
| `summary-stat-granularity-consistency` | Summary-stat granularity consistency (GRIM / GRIMMER) | live | A |
| `summary-stat-distribution-plausibility` | Summary-stat distribution plausibility (SPRITE) | live | A |
| `effect-size-conversion` | Effect-size metric conversion (d / r / OR / eta² / f) | live | A |
| `image-duplication-detection` | Image duplication and manipulation detection in figures | design | B |
| `participant-flow-consistency` | Participant-flow and cross-table consistency in trial reports | design | B |
| `baseline-allocation-plausibility` | Allocation and baseline-data plausibility in randomised trials | design | B |
| `correlation-matrix-consistency` | Rounded correlation-matrix consistency (positive semidefiniteness) | design | A |

## 07 — Writing & claims

_Narrative, abstract, discussion._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `marginal-significance-and-spin-language` | Marginal-significance & spin / overclaim language detection | live | A |
| `text-integrity-duplication` | Text integrity (duplicated content, tortured phrases, incompatible text) | design | B |

## 08 — Citations & references

_What's cited, how, and whether correctly._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `reference-integrity-checks` | Reference integrity checks (retraction, accuracy, consistency) | live | A |
| `reference-existence-verification` | Reference existence verification (fabricated or hallucinated citations) | design | A |

## 09 — Submission & artifact handling

_Manuscript + supplements + data packaged for venue._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `pdf-text-fidelity` | PDF text fidelity (Layer 1) | live | A |
| `pdf-section-structure` | PDF section structure recognition (Layer 3) | live | A |
| `pdf-reference-parsing` | PDF reference list parsing (Layer 5) | design | A |
| `pdf-table-extraction` | PDF table extraction (Layer 4a) | live | A |
| `pdf-citation-matching` | PDF in-text citation matching (Layer 6) | design | A |
| `paper-mill-signal-detection` | Paper-mill signal detection across a batch of manuscripts | design | B |

## 10 — Peer review

_Reviews, responses, editorial decisions._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `review-flaw-detection` | Peer-review flaw detection on full manuscripts | design | B |

## 11 — Publication & indexing

_Final version, DOI, metadata in indexes._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `articlefinder-article-retrieval` | Article retrieval given DOI / title / URL | draft | A |

## 12 — Post-publication corrections

_Errata, retractions, expressions of concern._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `post-publication-notice-detection` | Post-publication notice detection (retraction, expression of concern) | design | B |

## 13 — Citation & reuse downstream

_How the paper is used by others._

*(no leaves yet)*

## 14 — Replication & robustness

_Direct + conceptual replications, multiverse, robustness checks._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `replication-target-lookup` | Replication / target DOI lookup | live | A |
| `zcurve-evidential-value` | Z-curve evidential value of a findings set | live | A |
| `p-curve-evidential-value` | P-curve evidential value of a findings set | live | A |
| `meta-analysis-publication-bias` | Meta-analysis small-study / publication bias | live | A |

## 15 — Synthesis

_Meta-analysis, systematic review, living reviews._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `rct-trustworthiness-composite` | Randomised-trial trustworthiness assessment (full INSPECT-SR) | design | B |

