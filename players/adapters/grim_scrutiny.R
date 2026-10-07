#!/usr/bin/env Rscript
# grim-consistency-v1 tool player: scrutiny grim().
# Reads a task envelope from stdin, runs the GRIM granularity-consistency check
# per reported statistic, emits the arena output schema JSON on stdout.
# Handles both statistic types: stat_type="mean" (1/N grid, scrutiny::grim) and
# stat_type="percent" (100/N grid, scrutiny::grim(percent = TRUE)).
# This arena is GRIM-only, so the adapter NEVER consults grimmer()/the SD — doing
# so would raise grimmer_inconsistent false alarms against GRIM-only gold.
suppressPackageStartupMessages({
  library(jsonlite)
  if (!requireNamespace("scrutiny", quietly = TRUE)) {
    stop("scrutiny is not installed. install.packages('scrutiny').")
  }
  library(scrutiny)
})

input <- fromJSON(readLines(con = "stdin", warn = FALSE), simplifyVector = FALSE)
stats <- input$input$statistics

fmt <- function(x, d) formatC(as.numeric(x), format = "f", digits = as.integer(d))

records <- list()
for (s in stats) {
  d         <- if (!is.null(s$decimals)) s$decimals else 2
  n         <- as.integer(s$n)
  n_items   <- if (!is.null(s$n_items)) as.integer(s$n_items) else 1L
  stat_type <- if (!is.null(s$stat_type)) s$stat_type else "mean"

  # A reported PERCENTAGE of a count sits on a 100/N grid, which is the same GRIM
  # test with percent = TRUE. A reported MEAN of n_items integer items sits on a
  # 1/N grid. Both are GRIM; neither consults the SD (see header).
  if (identical(stat_type, "percent")) {
    grim_ok <- tryCatch(
      isTRUE(scrutiny::grim(x = fmt(s$percent, d), n = n, percent = TRUE)),
      error = function(e) NA
    )
    kind_when_flagged <- "grim_percent_inconsistent"
  } else {
    grim_ok <- tryCatch(
      isTRUE(scrutiny::grim(x = fmt(s$mean, d), n = n, items = n_items)),
      error = function(e) NA
    )
    kind_when_flagged <- "grim_inconsistent"
  }

  flagged <- FALSE
  issue_kind <- NULL
  if (identical(grim_ok, FALSE)) {
    flagged <- TRUE; issue_kind <- kind_when_flagged
  }

  rec <- list(
    stat_id = s$stat_id,
    flagged = flagged,
    confidence = 1.0   # deterministic arithmetic tool
  )
  rec$issue_kind <- if (is.null(issue_kind)) NA else issue_kind
  records[[length(records) + 1L]] <- rec
}

cat(toJSON(list(records = records), auto_unbox = TRUE, null = "null", na = "null"))
