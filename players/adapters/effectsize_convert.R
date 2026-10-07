#!/usr/bin/env Rscript
# effect-size-conversion-v1 tool player: effectsize / esc closed-form conversions.
# Reads a task envelope from stdin ({value, from, to, context?}), dispatches on the
# (from,to) pair to the matching effectsize function (or the textbook identity), and
# emits the arena output schema JSON on stdout: {converted, confidence:1.0}.
#
# The canonical formula set (see arenas/effect-size-conversion-v1/README.md) is matched
# function-for-function against `effectsize`, so this deterministic tool reproduces the
# arena's computed gold and scores ~1.0 (the cross-validation oracle):
#   d  -> r    effectsize::d_to_r(d, n1, n2)   (n1,n2 from context; h=4 default)
#   r  -> d    effectsize::r_to_d(r, n1, n2)
#   d  -> OR   effectsize::d_to_oddsratio(d)
#   OR -> d    effectsize::oddsratio_to_d(OR)
#   eta2 -> f  effectsize::eta2_to_f(eta2)
#   f -> eta2  effectsize::f_to_eta2(f)
#   d  -> f    f = d/2   (two equal groups; Cohen's f = d/2)
#   f  -> d    d = 2f
suppressPackageStartupMessages({
  library(jsonlite)
  if (!requireNamespace("effectsize", quietly = TRUE)) {
    stop("effectsize is not installed. install.packages('effectsize').")
  }
  library(effectsize)
})

input <- fromJSON(readLines(con = "stdin", warn = FALSE), simplifyVector = FALSE)
req   <- input$input

value <- as.numeric(req$value)
frm   <- as.character(req$from)
to    <- as.character(req$to)
ctx   <- req$context

has_groups <- !is.null(ctx) && !is.null(ctx$n1) && !is.null(ctx$n2)

converted <- if (frm == to) {
  value
} else if (frm == "d" && to == "r") {
  if (has_groups) {
    effectsize::d_to_r(value, n1 = as.numeric(ctx$n1), n2 = as.numeric(ctx$n2))
  } else {
    effectsize::d_to_r(value)
  }
} else if (frm == "r" && to == "d") {
  if (has_groups) {
    effectsize::r_to_d(value, n1 = as.numeric(ctx$n1), n2 = as.numeric(ctx$n2))
  } else {
    effectsize::r_to_d(value)
  }
} else if (frm == "d" && to == "OR") {
  effectsize::d_to_oddsratio(value)
} else if (frm == "OR" && to == "d") {
  effectsize::oddsratio_to_d(value)
} else if (frm == "eta2" && to == "f") {
  effectsize::eta2_to_f(value)
} else if (frm == "f" && to == "eta2") {
  effectsize::f_to_eta2(value)
} else if (frm == "d" && to == "f") {
  value / 2.0                       # two equal groups: Cohen's f = d/2
} else if (frm == "f" && to == "d") {
  2.0 * value
} else {
  stop(sprintf("unsupported conversion %s->%s", frm, to))
}

out <- list(
  converted  = round(as.numeric(converted), 6),
  confidence = 1.0                  # deterministic closed-form tool
)

cat(toJSON(out, auto_unbox = TRUE, null = "null", na = "null", digits = 8))
