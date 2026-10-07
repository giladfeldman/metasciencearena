# Capture SPSStoR's real output for every SPSS source script.
#
# SPSStoR (lebebr01, GitHub-only, dormant since 2021) exposes `spss_to_r(path)`,
# which takes a .sps file and returns the equivalent R. It is the only tool in
# the survey with exactly the arena's shape: file in, R code out.
#
#   Rscript arenas/code-translation-r-v1/tools/capture_spsstor.R
#
# Output is recorded VERBATIM. Nothing is patched to satisfy the arena's
# contract — the scorer harvests statistics from whatever the code produces, so
# a tool is judged on whether its numbers are right, not on its idiom.

suppressPackageStartupMessages({
  library(SPSStoR)
  library(jsonlite)
})

arena <- Sys.getenv("ARENA_DIR", "arenas/code-translation-r-v1")
if (!dir.exists(file.path(arena, "source_scripts"))) {
  stop("run from the Meta Science Arena repo root, or set ARENA_DIR")
}
src_dir <- file.path(arena, "source_scripts", "spss")
out_dir <- file.path(arena, "fixtures", "tool_outputs", "spsstor")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tool_version <- as.character(packageVersion("SPSStoR"))
cat("SPSStoR version:", tool_version, "\n\n")

for (f in list.files(src_dir, pattern = "[.]sps$", full.names = TRUE)) {
  analysis <- sub("[.]sps$", "", basename(f))
  res <- tryCatch({
    # spss_to_r() returns an object of class "rsyntax" whose print method emits
    # the translated R. Capture the PRINTED form (capture.output(print(...))),
    # not the return value: for some commands the object is length 0 and only
    # the print method produces text. Still verbatim tool output.
    obj <- spss_to_r(f)
    txt <- utils::capture.output(print(obj))
    if (!length(txt) || !any(nzchar(trimws(txt)))) {
      txt <- as.character(unlist(obj))
    }
    txt <- txt[nzchar(trimws(txt))]
    list(ok = TRUE, code = paste(txt, collapse = "\n"))
  }, error = function(e) list(ok = FALSE, code = "", err = conditionMessage(e)))

  payload <- list(
    tool = "SPSStoR",
    tool_version = tool_version,
    captured_from = basename(f),
    r_code = if (res$ok) res$code else "",
    note = if (res$ok) "verbatim spss_to_r() output (captured from stdout)"
           else paste("converter error:", res$err)
  )
  writeLines(toJSON(payload, auto_unbox = TRUE, pretty = TRUE),
             file.path(out_dir, paste0(analysis, ".spss.json")))
  cat(sprintf("%-24s %s (%d chars)\n", analysis,
              if (res$ok) "OK" else "ERROR", nchar(payload$r_code)))
}
cat("\nfixtures written to", out_dir, "\n")
