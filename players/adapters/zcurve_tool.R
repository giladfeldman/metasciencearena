#!/usr/bin/env Rscript
# zcurve-evidential-v1 tool player: zcurve.
# Reads a task envelope from stdin (a set of significant z-scores), fits a z-curve,
# and decides whether the set has evidential value. Emits the arena output schema
# JSON on stdout.
#
# Decision rule: fit z-curve and read the Expected Discovery Rate (EDR). A set of
# pure selected nulls has EDR ~= 0.05 (the selection floor); real underlying effects
# push EDR well above it. has_evidential_value = EDR > 0.10. confidence scales with
# the distance from that threshold.
suppressPackageStartupMessages({
  library(jsonlite)
  if (!requireNamespace("zcurve", quietly = TRUE)) {
    stop("zcurve is not installed. install.packages('zcurve').")
  }
  library(zcurve)
})

input <- fromJSON(readLines(con = "stdin", warn = FALSE), simplifyVector = FALSE)
z <- as.numeric(unlist(input$input$z_scores))
z <- z[is.finite(z) & z > 1.96]

edr <- tryCatch({
  fit <- zcurve::zcurve(z = z, method = "EM", bootstrap = FALSE)
  co <- summary(fit)$coefficients
  # EDR is reported as a row ("EDR") in the coefficients table; fall back to the
  # model object if the summary layout differs across zcurve versions.
  val <- NA
  rn <- rownames(co)
  if (!is.null(rn) && "EDR" %in% rn) {
    val <- as.numeric(co["EDR", "Estimate"])
  } else if (!is.null(fit$coefficients) && !is.null(fit$coefficients["EDR"])) {
    val <- as.numeric(fit$coefficients["EDR"])
  }
  val
}, error = function(e) NA)

if (is.na(edr)) {
  # Could not fit (e.g. too few z); fall back to a neutral low-confidence guess
  # based on the raw mean z (higher mean z => more likely real effects).
  has_ev <- mean(z) > 2.8
  conf <- 0.5
  edr_out <- NULL
} else {
  has_ev <- edr > 0.10
  conf <- max(0.5, min(1.0, 0.5 + abs(edr - 0.10) * 2))
  edr_out <- round(edr, 4)
}

out <- list(
  has_evidential_value = isTRUE(has_ev),
  confidence = round(conf, 3)
)
out$expected_discovery_rate <- if (is.null(edr_out)) NA else edr_out
cat(toJSON(out, auto_unbox = TRUE, null = "null", na = "null"))
