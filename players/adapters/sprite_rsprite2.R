#!/usr/bin/env Rscript
# sprite-plausibility-v1 tool player: rsprite2.
# Reads a task envelope from stdin; for each reported (mean, sd, n, scale) decides
# whether a sample of n integers in [scale_min, scale_max] can produce it. Emits the
# arena output schema JSON on stdout.
suppressPackageStartupMessages({
  library(jsonlite)
  if (!requireNamespace("rsprite2", quietly = TRUE)) {
    stop("rsprite2 is not installed. install.packages('rsprite2').")
  }
  library(rsprite2)
})

input <- fromJSON(readLines(con = "stdin", warn = FALSE), simplifyVector = FALSE)
stats <- input$input$statistics

records <- list()
for (s in stats) {
  mean_v <- as.numeric(s$mean)
  sd_v   <- as.numeric(s$sd)
  n_v    <- as.integer(s$n)
  lo     <- as.integer(s$scale_min)
  hi     <- as.integer(s$scale_max)
  d      <- if (!is.null(s$decimals)) as.integer(s$decimals) else 2L

  # set_parameters() validates achievability (range, GRIM, variance bounds) and
  # errors when the (mean, sd, n, range) combination is impossible. A successful
  # find_possible_distribution() proves a concrete sample exists.
  ok <- tryCatch({
    params <- rsprite2::set_parameters(
      mean = mean_v, sd = sd_v, n_obs = n_v,
      min_val = lo, max_val = hi,
      m_prec = d, sd_prec = d, restrictions_exact = NULL
    )
    res <- rsprite2::find_possible_distribution(params, seed = 1L)
    !is.null(res) && isTRUE(res$outcome == "success")
  }, error = function(e) FALSE)

  flagged <- !isTRUE(ok)
  issue_kind <- NULL
  if (flagged) {
    issue_kind <- if (mean_v < lo || mean_v > hi) "impossible_mean" else "impossible_sd"
  }

  rec <- list(stat_id = s$stat_id, flagged = flagged, confidence = 1.0)
  rec$issue_kind <- if (is.null(issue_kind)) NA else issue_kind
  records[[length(records) + 1L]] <- rec
}

cat(toJSON(list(records = records), auto_unbox = TRUE, null = "null", na = "null"))
