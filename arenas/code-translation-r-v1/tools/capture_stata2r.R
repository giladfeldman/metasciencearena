# Capture skranz/stata2r's real output for every Stata source script.
#
# stata2r (skranz, GitHub/r-universe, part of the repbox reproduction pipeline)
# is the only actively-maintained tool in the survey and the only one with a
# real .do parser. `do_to_r(do_code)` takes do-file text and returns R.
#
# It is scoped to DATA MANIPULATION: the README states it "will not be usable to
# convert complete Stata analyses to R", and estimation results are normally
# supplied externally by repbox rather than computed. So the estimation tiers are
# expected to be partial or empty — that is a real, reportable scope limit, not
# a harness failure, and it is recorded as such.
#
#   Rscript arenas/code-translation-r-v1/tools/capture_stata2r.R
#
# Output is recorded VERBATIM and never patched.

suppressPackageStartupMessages({
  library(stata2r)
  library(jsonlite)
})

arena <- Sys.getenv("ARENA_DIR", "arenas/code-translation-r-v1")
if (!dir.exists(file.path(arena, "source_scripts"))) {
  stop("run from the Meta Science Arena repo root, or set ARENA_DIR")
}
src_dir <- file.path(arena, "source_scripts", "stata")
out_dir <- file.path(arena, "fixtures", "tool_outputs", "stata2r-skranz")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tool_version <- as.character(packageVersion("stata2r"))
cat("stata2r (skranz) version:", tool_version, "\n\n")

extract_code <- function(res) {
  # do_to_r() may return a character vector of R code, or a data frame with one
  # row per command carrying an `r_code`-like column. Handle both without
  # inventing content.
  if (is.character(res)) return(paste(res, collapse = "\n"))
  if (is.data.frame(res)) {
    for (col in c("r_code", "rcode", "code", "r")) {
      if (col %in% colnames(res)) {
        v <- as.character(res[[col]])
        return(paste(v[nzchar(trimws(v))], collapse = "\n"))
      }
    }
  }
  if (is.list(res)) {
    v <- unlist(lapply(res, function(x) if (is.character(x)) x else NULL))
    if (length(v)) return(paste(v, collapse = "\n"))
  }
  ""
}

for (f in list.files(src_dir, pattern = "[.]do$", full.names = TRUE)) {
  analysis <- sub("[.]do$", "", basename(f))
  do_code <- paste(readLines(f, warn = FALSE), collapse = "\n")
  res <- tryCatch({
    out <- do_to_r(do_code)
    list(ok = TRUE, code = extract_code(out))
  }, error = function(e) list(ok = FALSE, code = "", err = conditionMessage(e)))

  note <- if (!res$ok) paste("converter error:", res$err)
          else if (!nzchar(trimws(res$code)))
            "do_to_r() produced no R code for this script (tool is scoped to data manipulation; estimation is out of its declared scope)."
          else "verbatim do_to_r() output"

  payload <- list(
    tool = "stata2r-skranz",
    tool_version = tool_version,
    captured_from = basename(f),
    r_code = res$code,
    note = note,
    tool_gap = (!res$ok || !nzchar(trimws(res$code)))
  )
  writeLines(toJSON(payload, auto_unbox = TRUE, pretty = TRUE),
             file.path(out_dir, paste0(analysis, ".stata.json")))
  cat(sprintf("%-24s %s (%d chars)\n", analysis,
              if (res$ok) "OK" else "ERROR", nchar(payload$r_code)))
}
cat("\nfixtures written to", out_dir, "\n")
