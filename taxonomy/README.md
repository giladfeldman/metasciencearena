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

## 06 — Results reporting

_Numbers, tables, figures in the paper._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `nhst-stats-and-effect-size-extraction` | NHST stats + effect-size + CI extraction from manuscripts | live | A |
| `statistical-reporting-completeness` | Statistical reporting completeness & p-value precision | live | A |
| `summary-stat-granularity-consistency` | Summary-stat granularity consistency (GRIM / GRIMMER) | live | A |
| `summary-stat-distribution-plausibility` | Summary-stat distribution plausibility (SPRITE) | live | A |
| `effect-size-conversion` | Effect-size metric conversion (d / r / OR / eta² / f) | live | A |

## 07 — Writing & claims

_Narrative, abstract, discussion._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `marginal-significance-and-spin-language` | Marginal-significance & spin / overclaim language detection | live | A |

## 08 — Citations & references

_What's cited, how, and whether correctly._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `reference-integrity-checks` | Reference integrity checks (retraction, accuracy, consistency) | live | A |

## 09 — Submission & artifact handling

_Manuscript + supplements + data packaged for venue._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `pdf-text-fidelity` | PDF text fidelity (Layer 1) | live | A |
| `pdf-section-structure` | PDF section structure recognition (Layer 3) | live | A |
| `pdf-reference-parsing` | PDF reference list parsing (Layer 5) | design | A |
| `pdf-table-extraction` | PDF table extraction (Layer 4a) | live | A |
| `pdf-citation-matching` | PDF in-text citation matching (Layer 6) | design | A |

## 10 — Peer review

_Reviews, responses, editorial decisions._

*(no leaves yet)*

## 11 — Publication & indexing

_Final version, DOI, metadata in indexes._

| Leaf id | Name | Status | Scope |
|---|---|---|---|
| `articlefinder-article-retrieval` | Article retrieval given DOI / title / URL | draft | A |

## 12 — Post-publication corrections

_Errata, retractions, expressions of concern._

*(no leaves yet)*

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

*(no leaves yet)*

