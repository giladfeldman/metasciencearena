#!/usr/bin/env Rscript
# Read a task envelope from stdin, run statcheck, emit output schema JSON on stdout.
suppressPackageStartupMessages({
  library(jsonlite)
  if (!requireNamespace("statcheck", quietly = TRUE)) {
    stop("statcheck is not installed. install.packages('statcheck').")
  }
  library(statcheck)
})

input <- fromJSON(readLines(con = "stdin", warn = FALSE), simplifyVector = FALSE)
text <- input$input$text

# statcheck() prints a progress bar to stdout and a couple of messages to stderr
# during extraction. Sink stdout to nullfile while it runs so our final JSON
# is the only thing on stdout. (The previous wrapper attempted to capture
# the assignment inside capture.output(), but capture.output evaluates in its
# own frame so the binding never reached the parent scope and `results` was
# always NULL — silently producing 0 extractions.)
sink(nullfile())
results <- tryCatch(
  suppressMessages(suppressWarnings(statcheck(text))),
  error = function(e) NULL
)
sink()

extractions <- list()
if (!is.null(results) && nrow(results) > 0) {
  # statcheck >= 1.4 uses snake_case columns; fall back to old names if needed.
  has_new_api <- "test_value" %in% colnames(results)
  for (i in seq_len(nrow(results))) {
    row <- results[i, ]
    if (has_new_api) {
      val_str <- as.character(row$test_value)
      p_val   <- row$reported_p
      stat    <- row$test_type
      df1_val <- if ("df1" %in% colnames(results) && !is.na(row$df1)) row$df1 else NULL
      df2_val <- if ("df2" %in% colnames(results) && !is.na(row$df2)) row$df2 else NULL
      raw_str <- if ("raw" %in% colnames(results)) row$raw else val_str
    } else {
      val_str <- as.character(row$Value)
      p_val   <- row$Reported.P.Value
      stat    <- row$Statistic
      df1_val <- if (!is.na(row$df1)) row$df1 else NULL
      df2_val <- if (!is.na(row$df2)) row$df2 else NULL
      raw_str <- val_str
    }
    # Locate the raw match in the source text for the span.
    span_text <- if (nchar(raw_str) > 0) raw_str else val_str
    char_start <- regexpr(span_text, text, fixed = TRUE)[[1]]
    if (char_start < 1) {
      char_start <- 0L
      char_end   <- 0L
    } else {
      char_end <- char_start + nchar(span_text) - 1L
    }
    fields <- list(
      test_type = stat,
      df1 = df1_val,
      df2 = df2_val,
      value = val_str,
      p = p_val
    )
    # An R NULL inside a list serialises as `{}`, not null: a t-test (one df)
    # was publishing `"df1": {}`, an empty OBJECT where a number or null belongs.
    # Harmless to this arena's scorer, which does not read df1, but it is a
    # malformed value in a published record and any consumer that does read it
    # gets an object. Drop absent keys instead of emitting a shape for them.
    fields <- Filter(Negate(is.null), fields)

    # THE VERDICT. Until 2026-08-21 this adapter hardcoded
    # `flagged_suspicious = FALSE` on every row, so statcheck -- the canonical
    # NHST decision-error checker, whose entire purpose is this column -- scored
    # 0.000 on tier 5, the only tier that measures detection. Its own
    # `error` / `decision_error` output was computed and thrown away. That was a
    # measurement of our adapter, not of the tool.
    #
    # statcheck's two levels are kept distinct rather than merged:
    #   decision_error - the reported and recomputed p fall on opposite sides of
    #                    alpha, so the paper's stated CONCLUSION does not follow;
    #   error          - the recomputed p differs from the reported one at all.
    # Both are claims of inconsistency, so both map to "inconsistent"; the
    # narrower one is recorded in `fields` so a report can separate them.
    stc_err <- FALSE
    stc_dec <- FALSE
    if ("error" %in% colnames(results) && !is.na(row$error)) {
      stc_err <- isTRUE(as.logical(row$error))
    } else if ("Error" %in% colnames(results) && !is.na(row$Error)) {
      stc_err <- isTRUE(as.logical(row$Error))
    }
    if ("decision_error" %in% colnames(results) && !is.na(row$decision_error)) {
      stc_dec <- isTRUE(as.logical(row$decision_error))
    } else if ("DecisionError" %in% colnames(results) && !is.na(row$DecisionError)) {
      stc_dec <- isTRUE(as.logical(row$DecisionError))
    }
    fields$statcheck_error <- stc_err
    fields$statcheck_decision_error <- stc_dec
    # statcheck cannot check a row whose p was reported as an inequality, and
    # says so by leaving both flags FALSE on a computed_p it could not compare.
    # That is an abstention, not an all-clear.
    verdict <- if (stc_dec || stc_err) "inconsistent" else "ok"
    extractions[[length(extractions) + 1]] <- list(
      span = list(
        text = span_text,
        char_start = as.integer(char_start) - 1L,
        char_end   = as.integer(char_end)
      ),
      kind = "nhst_stat",
      fields = fields,
      # NOT a measurement: statcheck reports no probability, so this is a
      # constant. The stats-extraction scorer no longer ranks on calibration.
      confidence = 1.0,
      verdict = verdict,
      flagged_suspicious = (verdict != "ok")   # deprecated, kept one version
    )
  }
}

cat(toJSON(list(extractions = extractions, player_strategy_notes = "statcheck output, confidence implicit-1.0"),
           auto_unbox = TRUE, na = "null"))
