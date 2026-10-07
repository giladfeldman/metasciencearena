#!/usr/bin/env Rscript
# publication-bias-v1 tool player: metafor Egger regression test + trim-and-fill.
# Reads a task envelope from stdin (a meta-analytic dataset of yi/sei studies),
# fits a random-effects model, runs Egger's regression test (regtest, model="lm")
# and trim-and-fill, and emits the arena output schema JSON on stdout:
#   bias_detected      = (Egger p < EGGER_ALPHA)
#   confidence         = mapped from the Egger p-value (further from the threshold
#                        => more confident)
#   egger_p            = the Egger regression-test p-value
#   n_missing_trimfill = trim-and-fill's imputed-missing-study count (k0)
suppressPackageStartupMessages({
  library(jsonlite)
  if (!requireNamespace("metafor", quietly = TRUE)) {
    stop("metafor is not installed. install.packages('metafor').")
  }
  library(metafor)
})

EGGER_ALPHA <- 0.10

input  <- fromJSON(readLines(con = "stdin", warn = FALSE), simplifyVector = FALSE)
studies <- input$input$studies

yi  <- vapply(studies, function(s) as.numeric(s$yi),  numeric(1))
sei <- vapply(studies, function(s) as.numeric(s$sei), numeric(1))
vi  <- sei * sei

# Random-effects model (REML). Egger's test and trim-and-fill operate on this fit.
m <- tryCatch(
  metafor::rma(yi = yi, vi = vi, method = "REML"),
  error = function(e) metafor::rma(yi = yi, vi = vi, method = "DL")
)

# Egger's regression test in its linear-model (classic Egger) form: regress the
# effect on its standard error, weighted by precision; the moderator slope is the
# small-study-effect test.
rt <- metafor::regtest(m, model = "lm")
egger_p <- as.numeric(rt$pval)

# Trim-and-fill: number of studies imputed to restore funnel symmetry.
tf <- tryCatch(metafor::trimfill(m), error = function(e) NULL)
k0 <- if (!is.null(tf) && !is.null(tf$k0)) as.integer(tf$k0) else 0L

bias_detected <- isTRUE(egger_p < EGGER_ALPHA)

# Map the p-value to a confidence: how far the p sits from the decision threshold,
# clamped to [0.5, 0.99]. A p far below alpha (clear bias) or far above (clearly
# clean) both yield high confidence in the corresponding verdict.
dist <- abs(log10(max(egger_p, 1e-12)) - log10(EGGER_ALPHA))
confidence <- max(0.5, min(0.99, 0.5 + 0.15 * dist))

out <- list(
  bias_detected      = bias_detected,
  confidence         = round(confidence, 4),
  egger_p            = round(egger_p, 6),
  n_missing_trimfill = k0
)

cat(toJSON(out, auto_unbox = TRUE, null = "null", na = "null", digits = 8))
