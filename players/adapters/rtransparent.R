#!/usr/bin/env Rscript
# transparency-statements-v1 tool player: rtransparent (v0.2.5).
# rtransparent detects COI (rt_coi -> is_coi_pred), FUNDING (rt_fund -> is_funded_pred),
# and OPEN DATA / OPEN CODE. Its data/code module (rt_data_code) is a thin wrapper around
# oddpub::open_data_search(), so this adapter calls oddpub directly exactly as rt_data_code
# does internally. rtransparent does NOT detect materials. Its registration module
# (rt_register / rt_all) is broken in v0.2.5 (it raises "object 'index_any' not found" on
# every branch), so prereg cannot be produced and is emitted absent. Nothing is fabricated.
#
# I/O contract (mirrors players/adapters/grim_scrutiny.R):
#   stdin  : the task envelope JSON  -> input$input$text is the manuscript section.
#   stdout : JSON matching arenas/transparency-statements-v1/schemas/output.schema.json.
suppressPackageStartupMessages({
  library(jsonlite)
  if (!requireNamespace("rtransparent", quietly = TRUE)) {
    stop("rtransparent is not installed.")
  }
  library(rtransparent)
  library(oddpub)
})

input <- fromJSON(readLines(con = "stdin", warn = FALSE), simplifyVector = FALSE)
text <- input$input$text
if (is.null(text)) text <- ""
text <- as.character(text)

# rt_coi / rt_fund read a TEXT FILE PATH.
tf <- tempfile(fileext = ".txt")
writeLines(text, tf)

# --- COI ---
coi_present <- FALSE
coi_text <- ""
coi_res <- tryCatch(rtransparent::rt_coi(tf), error = function(e) NULL)
if (!is.null(coi_res) && is.data.frame(coi_res) && nrow(coi_res) >= 1) {
  coi_present <- isTRUE(as.logical(coi_res$is_coi_pred[1]))
  if (!is.null(coi_res$coi_text)) coi_text <- as.character(coi_res$coi_text[1])
}

# --- FUNDING ---
fund_present <- FALSE
fund_text <- ""
fund_res <- tryCatch(rtransparent::rt_fund(tf), error = function(e) NULL)
if (!is.null(fund_res) && is.data.frame(fund_res) && nrow(fund_res) >= 1) {
  fund_present <- isTRUE(as.logical(fund_res$is_funded_pred[1]))
  if (!is.null(fund_res$funding_text)) fund_text <- as.character(fund_res$funding_text[1])
}

# --- DATA / CODE via oddpub (rt_data_code's underlying engine) ---
sentences <- unlist(tokenizers::tokenize_sentences(text))
sentences <- tolower(sentences)
if (length(sentences) == 0) sentences <- ""
pts <- list(article = sentences)
od <- tryCatch(
  oddpub::open_data_search(pts, extract_sentences = TRUE, screen_das = "priority"),
  error = function(e) NULL
)
is_open_data <- FALSE
is_open_code <- FALSE
data_stmt <- ""
code_stmt <- ""
if (!is.null(od) && is.data.frame(od) && nrow(od) >= 1) {
  is_open_data <- isTRUE(as.logical(od$is_open_data[1]))
  is_open_code <- isTRUE(as.logical(od$is_open_code[1]))
  if (!is.null(od$open_data_statements)) data_stmt <- as.character(od$open_data_statements[1])
  if (!is.null(od$open_code_statements)) code_stmt <- as.character(od$open_code_statements[1])
}

# Pull the first URL out of a detected statement (or NULL).
first_url <- function(s) {
  if (is.null(s) || is.na(s) || !nzchar(s)) return(NULL)
  m <- regmatches(s, regexpr("https?://[^[:space:]]+", s))
  if (length(m) == 0 || !nzchar(m)) return(NULL)
  sub("[[:punct:]]+$", "", m)
}

open_field <- function(available, stmt) {
  list(
    available = isTRUE(available),
    on_request = FALSE,
    url = if (isTRUE(available)) first_url(stmt) else NULL
  )
}

# Statement is an OPTIONAL non-nullable string: include only when present + non-empty.
clean_stmt <- function(s) {
  if (is.null(s) || is.na(s)) return(NULL)
  # rt_coi/rt_fund can leave literal escape tokens (e.g. a trailing "\\r") in the
  # text; strip real control chars AND those two-char escape artifacts.
  s <- gsub("[[:cntrl:]]+", " ", s)
  s <- gsub("\\\\[rnt]", " ", s)
  s <- trimws(gsub("[[:space:]]+", " ", s))
  if (!nzchar(s)) return(NULL)
  s
}
coi_node <- list(present = coi_present)
if (coi_present) { cs <- clean_stmt(coi_text); if (!is.null(cs)) coi_node$statement <- cs }
fund_node <- list(present = fund_present)
if (fund_present) { fs <- clean_stmt(fund_text); if (!is.null(fs)) fund_node$statement <- fs }

out <- list(
  coi       = coi_node,
  funding   = fund_node,
  data      = open_field(is_open_data, data_stmt),
  code      = open_field(is_open_code, code_stmt),
  materials = list(available = FALSE, on_request = FALSE, url = NULL),
  prereg    = list(available = FALSE, url = NULL),
  confidence = 1.0   # deterministic detection tool
)

cat(toJSON(out, auto_unbox = TRUE, null = "null", na = "null"))
