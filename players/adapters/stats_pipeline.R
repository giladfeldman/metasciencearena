#!/usr/bin/env Rscript
# Stats-from-PDF pipelines on the R side (players/adapters/stats_pipeline.py drives it).
#
# stdin: one JSON object {"mode": ..., ...}; stdout: {"rows": [...], "versions": {...}}.
#   mode "statcheck_text"  {"text"}                      statcheck::statcheck(text)
#   mode "statcheck_pdf"   {"pdf_path"}                  statcheck::checkPDF(file) - statcheck's own PDF reader
#   mode "effectcheck"     {"pdf_path","engine","url","python","table_typer_python"}
#                                                         effectcheck::check_document(engine=)
#   mode "metacheck"       {"pdf_path","method","grobid_url","lib"}
#                                                         metacheck convert -> module_run("stat_check")
# Every mode returns the tool's OWN rows; the Python side maps them to the arena schema.
# Progress output is sunk so stdout carries only the JSON.
suppressPackageStartupMessages(library(jsonlite))
req <- fromJSON(readLines(con = "stdin", warn = FALSE), simplifyVector = TRUE)
out <- list(rows = list(), versions = list(R = paste(R.version$major, R.version$minor, sep = ".")))

rows_of <- function(df) {
  if (is.null(df) || nrow(df) == 0) return(list())
  df <- as.data.frame(df)
  lapply(seq_len(nrow(df)), function(i) as.list(df[i, , drop = FALSE]))
}

quiet <- function(expr) {
  sink(nullfile())
  on.exit(sink())
  suppressMessages(suppressWarnings(expr))
}

if (req$mode %in% c("statcheck_text", "statcheck_pdf")) {
  suppressPackageStartupMessages(library(statcheck))
  out$versions$statcheck <- as.character(packageVersion("statcheck"))
  res <- if (req$mode == "statcheck_text") {
    quiet(statcheck(req$text, messages = FALSE))
  } else {
    out$versions$pdftools <- as.character(packageVersion("pdftools"))
    quiet(checkPDF(req$pdf_path, messages = FALSE))
  }
  if (!is.null(res)) {
    keep <- intersect(c("test_type", "df1", "df2", "test_comp", "test_value", "p_comp", "reported_p",
                        "computed_p", "raw", "error", "decision_error", "one_tailed_in_txt"), names(res))
    out$rows <- rows_of(res[, keep, drop = FALSE])
  }
} else if (req$mode == "effectcheck") {
  suppressPackageStartupMessages(library(effectcheck))
  out$versions$effectcheck <- as.character(packageVersion("effectcheck"))
  args <- list(table_typer_python = req$table_typer_python)
  if (!is.null(req$url)) args$url <- req$url
  # the token arrives in the request, and no API key is sent: .Renviron must not decide
  if (!is.null(req$token)) { args$token <- req$token; args$api_key <- "" }
  if (!is.null(req$python)) args$python <- req$python
  res <- quiet(check_document(req$pdf_path, engine = req$engine, extract_args = args))
  prov <- attr(res, "extraction")
  if (!is.null(prov)) out$versions$extraction_engine_version <- as.character(prov$engine_version %||% NA)
  keep <- intersect(c("status", "p_check_status", "p_check_reason", "es_check_status", "test_type", "stat_value", "df1", "df2", "p_reported", "p_symbol",
                      "p_computed", "decision_error", "effect_reported_name", "effect_reported",
                      "ci_low", "ci_high", "raw_text", "extraction_engine_version"), names(res))
  out$rows <- rows_of(res[, keep, drop = FALSE])
} else if (req$mode == "metacheck") {
  suppressPackageStartupMessages(library(metacheck, lib.loc = req$lib))
  out$versions$metacheck <- as.character(packageVersion("metacheck", lib.loc = req$lib))
  suppressPackageStartupMessages(library(statcheck))
  out$versions$statcheck <- as.character(packageVersion("statcheck"))
  # LOCAL services only: metacheck's defaults are the public Eindhoven GROBID / hosted bibr,
  # which must never receive a held-out PDF. bibr runs only where a local bibr is set up.
  tmpd <- tempfile("metacheck_"); dir.create(tmpd)
  paper <- if (req$method == "grobid") {
    quiet(convert_grobid(req$pdf_path, save_path = tmpd, api_url = req$grobid_url))
  } else {
    stop("metacheck on bibr needs the local bibr setup (desktop); not run here")
  }
  if (is.character(paper)) paper <- quiet(read(paper))
  mr <- quiet(module_run(paper, "stat_check"))
  tab <- mr$table
  if (!is.null(tab) && nrow(tab) > 0) {
    keep <- intersect(c("test_type", "df1", "df2", "test_comp", "test_value", "p_comp", "reported_p",
                        "computed_p", "raw", "error", "decision_error"), names(tab))
    out$rows <- rows_of(tab[, keep, drop = FALSE])
  }
} else {
  stop(paste("unknown mode", req$mode))
}
cat(toJSON(out, auto_unbox = TRUE, null = "null", na = "null", digits = NA))
