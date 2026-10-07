# Capture spss2rmarkdown's real output for every SPSS source script.
#
# Run ONCE per tool version; the results are committed as version-pinned
# fixtures (user decision 2026-08-03), so the arena runs on machines without R
# or without this GitHub-only package installed.
#
#   Rscript arenas/code-translation-r-v1/tools/capture_spss2rmarkdown.R
#
# The captured code is written VERBATIM. It is not patched to satisfy the
# arena's output contract — doing so would score our edits rather than the
# tool. A converter that emits analysis code but no JSON block simply fails the
# execution gate, which is an honest result for a tool that was built to produce
# R Markdown reports for humans, not machine-checkable statistics.

suppressPackageStartupMessages(library(spss2rmarkdown))
suppressPackageStartupMessages(library(jsonlite))

# Run from the repo root, or set ARENA_DIR to the arena directory.
arena <- Sys.getenv("ARENA_DIR", "arenas/code-translation-r-v1")
if (!dir.exists(file.path(arena, "source_scripts"))) {
  stop("run from the Meta Science Arena repo root, or set ARENA_DIR")
}
src_dir <- file.path(arena, "source_scripts", "spss")
out_dir <- file.path(arena, "fixtures", "tool_outputs", "spss2rmarkdown")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tool_version <- as.character(packageVersion("spss2rmarkdown"))
cat("spss2rmarkdown version:", tool_version, "\n\n")

# The package's parse_sav() reads a .sav via haven. The arena's fixed dataset is
# a CSV, so build the same sav_info structure from it (see parse_sav's source:
# list(data, metadata, value_labels, n_obs, n_vars, path)). This gives the
# converter exactly the variable metadata it expects, so what we capture is the
# tool's real translation ability rather than an artifact of a missing argument.
csv_path <- file.path(arena, "source_scripts", "data", "wellbeing.csv")
dat <- read.csv(csv_path, stringsAsFactors = FALSE)
sav_info <- list(
  data = dat,
  metadata = data.frame(
    name = names(dat),
    label = "",
    type = vapply(dat, function(x) class(x)[1], character(1)),
    format = "",
    n_missing = vapply(dat, function(x) sum(is.na(x)), integer(1)),
    stringsAsFactors = FALSE
  ),
  value_labels = list(),
  n_obs = nrow(dat),
  n_vars = ncol(dat),
  path = csv_path
)

files <- list.files(src_dir, pattern = "[.]sps$", full.names = TRUE)
for (f in files) {
  analysis <- sub("[.]sps$", "", basename(f))
  res <- tryCatch({
    parsed <- parse_sps(f)
    converted <- convert_all_commands(parsed, sav_info)
    # convert_all_commands() returns one list per command, each with fields
    # $r_code, $packages, $analysis_type, $variables, $order. Take ONLY $r_code
    # (in $order): unlist()ing the whole structure splices metadata like
    # "Descriptive Statistics" and the variable names into the script as bare
    # symbols, which is a syntax error — that was a capture bug, not a defect in
    # the converter.
    ord <- vapply(converted, function(x) {
      o <- x$order; if (is.null(o) || !is.numeric(o)) NA_real_ else as.numeric(o)[1]
    }, numeric(1))
    if (!all(is.na(ord))) converted <- converted[order(ord, na.last = TRUE)]
    pieces <- vapply(converted, function(x) {
      code <- x$r_code
      if (is.null(code)) "" else paste(as.character(code), collapse = "\n")
    }, character(1))
    pkgs <- unique(unlist(lapply(converted, function(x) x$packages)))
    pkgs <- pkgs[nzchar(pkgs)]
    header <- if (length(pkgs)) paste0("library(", pkgs, ")", collapse = "\n") else ""
    code <- paste(c(header, pieces[nzchar(pieces)]), collapse = "\n")
    list(ok = TRUE, code = code)
  }, error = function(e) list(ok = FALSE, code = "", err = conditionMessage(e)))

  payload <- list(
    tool = "spss2rmarkdown",
    tool_version = tool_version,
    captured_from = basename(f),
    r_code = if (res$ok) res$code else "",
    note = if (res$ok) "verbatim convert_all_commands() output"
           else paste("converter error:", res$err)
  )
  writeLines(toJSON(payload, auto_unbox = TRUE, pretty = TRUE),
             file.path(out_dir, paste0(analysis, ".spss.json")))
  cat(sprintf("%-24s %s (%d chars)\n", analysis,
              if (res$ok) "OK" else "ERROR", nchar(payload$r_code)))
}
cat("\nfixtures written to", out_dir, "\n")
