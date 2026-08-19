#!/usr/bin/env Rscript
# transparency-statements-v1 tool player: oddpub (v7.2.3).
# oddpub detects OPEN DATA and OPEN CODE statements in manuscript text via
# oddpub::open_data_search(). It does NOT detect COI/funding/materials/preregistration,
# so those fields are emitted in their schema "absent" form (never invented).
#
# I/O contract (mirrors players/adapters/grim_scrutiny.R):
#   stdin  : the task envelope JSON  -> input$input$text is the manuscript section.
#   stdout : JSON matching arenas/transparency-statements-v1/schemas/output.schema.json.
suppressPackageStartupMessages({
  library(jsonlite)
  if (!requireNamespace("oddpub", quietly = TRUE)) {
    stop("oddpub is not installed.")
  }
  library(oddpub)
})

input <- fromJSON(readLines(con = "stdin", warn = FALSE), simplifyVector = FALSE)
text <- input$input$text
if (is.null(text)) text <- ""

# Tokenize into a NAMED list of one "article" -> sentence vector, the shape
# open_data_search() expects (it derives the `article` column from list names).
sentences <- unlist(tokenizers::tokenize_sentences(as.character(text)))
sentences <- tolower(sentences)
if (length(sentences) == 0) sentences <- ""
pts <- list(article = sentences)

res <- tryCatch(
  oddpub::open_data_search(pts, extract_sentences = TRUE, screen_das = "priority"),
  error = function(e) NULL
)

is_open_data <- FALSE
is_open_code <- FALSE
data_stmt <- ""
code_stmt <- ""
if (!is.null(res) && is.data.frame(res) && nrow(res) >= 1) {
  is_open_data <- isTRUE(as.logical(res$is_open_data[1]))
  is_open_code <- isTRUE(as.logical(res$is_open_code[1]))
  if (!is.null(res$open_data_statements)) data_stmt <- as.character(res$open_data_statements[1])
  if (!is.null(res$open_code_statements)) code_stmt <- as.character(res$open_code_statements[1])
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

out <- list(
  coi       = list(present = FALSE),
  funding   = list(present = FALSE),
  data      = open_field(is_open_data, data_stmt),
  code      = open_field(is_open_code, code_stmt),
  materials = list(available = FALSE, on_request = FALSE, url = NULL),
  prereg    = list(available = FALSE, url = NULL),
  confidence = 1.0   # deterministic detection tool
)

cat(toJSON(out, auto_unbox = TRUE, null = "null", na = "null"))
