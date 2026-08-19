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
    extractions[[length(extractions) + 1]] <- list(
      span = list(
        text = span_text,
        char_start = as.integer(char_start) - 1L,
        char_end   = as.integer(char_end)
      ),
      kind = "nhst_stat",
      fields = fields,
      confidence = 1.0,
      flagged_suspicious = FALSE
    )
  }
}

cat(toJSON(list(extractions = extractions, player_strategy_notes = "statcheck output, confidence implicit-1.0"),
           auto_unbox = TRUE, na = "null"))
