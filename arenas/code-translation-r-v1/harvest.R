# Statistic harvester for code-translation-r-v1.
#
# WHY THIS EXISTS
# ---------------
# The arena asks "is the translation ACCURATE?", not "did the translator follow
# our print contract?". An earlier version required every player to emit a JSON
# object, which scored real converters 0.00 for producing perfectly correct
# analyses in their own idiom — that measured conformance to our harness, not
# translation quality, and it made deterministic tools structurally unable to
# compete with LLMs (who simply follow instructions better).
#
# So: run the player's code verbatim, then WALK the resulting environment and
# recover the requested statistics from whatever it left behind —
#   * a jmv results object (jmv::descriptives, ttestIS, linReg, ANOVA, ...)
#   * base-R model objects (lm, htest, anova/car::Anova tables)
#   * plain numeric variables named after the statistic
#   * an explicit JSON block on stdout (still honoured — it just isn't required)
#
# Nothing here patches the player's code. If a value genuinely is not present,
# it stays missing and the translation loses the point, which is correct.

suppressWarnings(suppressMessages({
  library(jsonlite)
}))

ARENA_DATA <- Sys.getenv("ARENA_DATA")
PLAYER_FILE <- Sys.getenv("ARENA_PLAYER_FILE")
REQUIRED <- strsplit(Sys.getenv("ARENA_REQUIRED"), ",", fixed = TRUE)[[1]]
REQUIRED <- trimws(REQUIRED[nzchar(REQUIRED)])

# ---------------------------------------------------------------------------
# 1. Run the player's code in its own environment.
#
# `data` and `df` are pre-bound to the dataset: converters routinely emit an
# analysis call that assumes the frame is already loaded (spss2rmarkdown emits
# its data-load step separately). Refusing to run for that reason would score
# our capture procedure, not the translation. A player that loads the data
# itself simply overwrites these.
# ---------------------------------------------------------------------------
# Attach the converters' own runtime helpers when present. stata2r emits calls to
# its OWN exported functions (scmd_replace, scmd_keep, s2r_stata_logical, ...),
# which is legitimate translated code — the package is the runtime that code
# targets, exactly as a translation to dplyr targets dplyr. Refusing to load it
# would score a "could not find function" that only exists because the harness
# withheld the library. Silently skipped when not installed.
for (.pkg in c("restorepoint", "repboxUtils", "stata2r", "dplyr")) {
  if (requireNamespace(.pkg, quietly = TRUE)) {
    suppressWarnings(suppressMessages(
      tryCatch(attachNamespace(.pkg), error = function(e) NULL)))
  }
}

penv <- new.env(parent = globalenv())
.dat <- utils::read.csv(ARENA_DATA, stringsAsFactors = FALSE)

# Case-tolerant columns. SPSS/Stata are case-insensitive about variable names
# and SPSS conventionally reports them upper-cased, so converters legitimately
# emit `vars = c('AGE','SCORE')` for a column stored as `age`. Penalising that
# would score a naming convention, not the translation, so the frame carries
# BOTH cases. `mean_age` and `mean(AGE)` must reach the same number.
.dupe_case <- function(d) {
  extra <- list()
  for (nm in names(d)) {
    up <- toupper(nm)
    if (!identical(up, nm) && !(up %in% names(d))) extra[[up]] <- d[[nm]]
  }
  if (length(extra)) d <- cbind(d, as.data.frame(extra, stringsAsFactors = FALSE))
  d
}
.dat <- .dupe_case(.dat)

# Converters differ on what they call the data frame: spss2rmarkdown emits
# `data`, SPSStoR emits `x` ("# x is the name of your data frame"), and LLMs
# usually load it themselves. Bind all the common names — the arena measures
# whether the ANALYSIS is right, and a tool that documents its frame variable is
# not thereby wrong. Any player that loads the data itself just overwrites these.
for (.nm in c("data", "df", "x", "dat")) assign(.nm, .dat, envir = penv)

stdout_txt <- character(0)
run_err <- NULL
con <- textConnection("stdout_txt", "w", local = TRUE)
sink(con, type = "output")
ok <- tryCatch({
  # Evaluate expression-by-expression and KEEP the value of each top-level
  # expression under `.arena_valN`. Converters idiomatically end with a bare
  # `jmv::descriptives(...)` call whose result is printed, never assigned — the
  # numbers are in that returned object, so discarding unassigned values would
  # score the translation 0 for a stylistic choice.
  exprs <- parse(PLAYER_FILE)
  for (i in seq_along(exprs)) {
    v <- withVisible(eval(exprs[[i]], envir = penv))
    if (!is.null(v$value)) {
      assign(paste0(".arena_val", i), v$value, envir = penv)
    }
  }
  TRUE
}, error = function(e) { run_err <<- conditionMessage(e); FALSE })
sink(type = "output")
close(con)

if (!ok) {
  cat(toJSON(list(`__arena_error__` = paste("player code failed:", run_err)),
             auto_unbox = TRUE))
  quit(status = 0)
}

# ---------------------------------------------------------------------------
# 2. Harvest.
# ---------------------------------------------------------------------------
found <- list()
put <- function(key, value) {
  if (is.null(value) || !is.finite(suppressWarnings(as.numeric(value)[1]))) return(invisible())
  if (is.null(found[[key]])) found[[key]] <<- as.numeric(value)[1]
}

# 2a. An explicit JSON object on stdout still wins — it is the least ambiguous
#     signal a player can give, it is simply no longer mandatory.
for (line in rev(stdout_txt)) {
  line <- trimws(line)
  if (!nzchar(line) || !startsWith(line, "{")) next
  parsed <- tryCatch(fromJSON(line), error = function(e) NULL)
  if (is.list(parsed)) for (nm in names(parsed)) put(nm, parsed[[nm]])
}

# 2b. Plain variables in the player's environment named after a statistic.
for (nm in REQUIRED) {
  if (exists(nm, envir = penv, inherits = FALSE)) {
    v <- get(nm, envir = penv)
    if (is.numeric(v) && length(v) >= 1) put(nm, v[1])
  }
}
# ... and a list/vector named `out` or `results` holding them.
for (holder in c("out", "results", "res")) {
  if (exists(holder, envir = penv, inherits = FALSE)) {
    v <- get(holder, envir = penv)
    if (is.list(v) || (is.numeric(v) && !is.null(names(v)))) {
      for (nm in names(v)) put(nm, v[[nm]])
    }
  }
}

# 2c. Walk every object the player left behind and pull statistics out of the
#     structures a real translation produces.
# all.names = TRUE is REQUIRED: the captured top-level values are stored as
# `.arena_valN`, and plain ls() hides dot-prefixed names. Without it, a
# translation ending in a bare `jmv::descriptives(...)` yields nothing to harvest.
objs <- ls(envir = penv, all.names = TRUE)

flatten_jmv <- function(obj, depth = 0L) {
  # jmv results are R6 "ResultsElement"s; as.data.frame() on a Table yields
  # columns named "<VAR>[<stat>]" (e.g. "AGE[mean]"). Harvest them all,
  # case-insensitively, since converters often upper-case variable names.
  #
  # Depth-bounded, and plot/image elements are skipped: rendering a jmv plot
  # array is slow enough to blow the scorer's timeout, and an image never
  # contains a statistic we can compare.
  out <- list()
  if (depth > 3L) return(out)
  tbls <- tryCatch(names(obj), error = function(e) NULL)
  for (t in tbls) {
    if (grepl("plot|image", t, ignore.case = TRUE)) next
    el <- tryCatch(obj[[t]], error = function(e) NULL)
    if (is.null(el)) next
    if (inherits(el, "Image") || inherits(el, "Preformatted")) next
    if (inherits(el, "Table")) {
      d <- tryCatch(as.data.frame(el), error = function(e) NULL)
      if (is.data.frame(d) && nrow(d) >= 1 && ncol(d) >= 1) out[[t]] <- d
    } else if (inherits(el, "ResultsElement") || inherits(el, "Group") ||
               inherits(el, "Array")) {
      sub <- tryCatch(flatten_jmv(el, depth + 1L), error = function(e) list())
      # An Array (e.g. linReg's `models`) is indexed positionally, not by name,
      # so names() returns nothing and recursion alone misses it. Descend into
      # its elements explicitly — models[[1]]$coef is where the regression
      # coefficients live.
      if (!length(sub) && inherits(el, "Array")) {
        n <- tryCatch(el$itemCount, error = function(e) 0L)
        if (is.numeric(n) && n >= 1) {
          for (i in seq_len(min(as.integer(n), 4L))) {
            item <- tryCatch(el$get(index = i), error = function(e) NULL)
            if (is.null(item)) next
            si <- tryCatch(flatten_jmv(item, depth + 1L), error = function(e) list())
            for (k in names(si)) out[[paste(t, i, k, sep = ".")]] <- si[[k]]
          }
        }
      }
      for (k in names(sub)) out[[paste(t, k, sep = ".")]] <- sub[[k]]
    }
  }
  out
}

jmv_tables <- list()
for (o in objs) {
  v <- tryCatch(get(o, envir = penv), error = function(e) NULL)
  if (is.null(v)) next
  if (inherits(v, "ResultsElement") || inherits(v, "Group")) {
    tt <- tryCatch(flatten_jmv(v), error = function(e) list())
    for (k in names(tt)) jmv_tables[[paste(o, k, sep = ".")]] <- tt[[k]]
  }
}

# Map "<VAR>[<stat>]" cells onto requested statistic names like `mean_age`,
# `sd_score`, `n_age`. Matching is on (stat, variable) in either name order.
jmv_lookup <- function(stat, varname) {
  target_v <- toupper(varname)
  for (d in jmv_tables) {
    cn <- colnames(d)
    hit <- cn[toupper(cn) == paste0(target_v, "[", toupper(stat), "]")]
    if (length(hit)) {
      val <- suppressWarnings(as.numeric(d[[hit[1]]][1]))
      if (is.finite(val)) return(val)
    }
  }
  NULL
}

STAT_ALIASES <- list(mean = "mean", sd = "sd", n = "n", median = "median",
                     min = "min", max = "max")
for (nm in REQUIRED) {
  if (!is.null(found[[nm]])) next
  parts <- strsplit(nm, "_", fixed = TRUE)[[1]]
  if (length(parts) < 2) next
  stat <- parts[1]
  varname <- paste(parts[-1], collapse = "_")
  if (!is.null(STAT_ALIASES[[stat]])) {
    v <- jmv_lookup(STAT_ALIASES[[stat]], varname)
    if (!is.null(v)) put(nm, v)
  }
}

# jmv's inferential tables use their own column names (`stat`, `df`, `p`,
# `md`, `F`, ...), so map those onto the arena's statistic names. Without this a
# jmv-idiom translation would run, be numerically correct, and still score 0.
# jmv suffixes a table's columns by test VARIANT: a ttestIS table has
# `stat[stud]` / `df[stud]` / `p[stud]` alongside `stat[welc]` / `df[welc]` /
# `p[welc]`. Match the base name with an optional "[...]" suffix, and prefer the
# Student's (pooled) variant, since SPSS and Stata both headline the pooled test
# — which is precisely the T2 trap. Matching only bare `stat`/`df`/`p` found
# nothing and scored a numerically perfect translation 0.
jmv_col <- function(d, candidates, prefer = c("stud", "")) {
  cn <- colnames(d)
  base <- tolower(sub("\\[.*$", "", cn))
  for (c0 in candidates) {
    idx <- which(base == tolower(c0))
    if (!length(idx)) next
    for (p in prefer) {
      if (!nzchar(p)) return(cn[idx[1]])
      hit <- idx[grepl(paste0("\\[", p, "\\]$"), cn[idx], ignore.case = TRUE)]
      if (length(hit)) return(cn[hit[1]])
    }
    return(cn[idx[1]])
  }
  NULL
}
for (d in jmv_tables) {
  if (!is.data.frame(d) || nrow(d) < 1) next
  base_cn <- tolower(sub("\\[.*$", "", colnames(d)))

  # Independent-samples t-test. Require a `stat`/`t` column that is EXPLICITLY a
  # test statistic alongside a df column — jmv's assumption tables (`assum`,
  # normality, homogeneity) also carry a `p`, and treating one of those as the
  # t-test would score a Levene/Shapiro p-value as the test's p.
  if (any(base_cn %in% c("stat", "t")) && any(base_cn == "p") && any(base_cn == "df")) {
    c_stat <- jmv_col(d, c("stat", "t")); c_df <- jmv_col(d, c("df"))
    c_p <- jmv_col(d, c("p")); c_md <- jmv_col(d, c("md", "meandifference"))
    if (!is.null(c_stat)) put("t_statistic", suppressWarnings(as.numeric(d[[c_stat]][1])))
    if (!is.null(c_df))   put("df",          suppressWarnings(as.numeric(d[[c_df]][1])))
    if (!is.null(c_p))    put("p_value",     suppressWarnings(as.numeric(d[[c_p]][1])))
    if (!is.null(c_md))   put("mean_diff",   suppressWarnings(as.numeric(d[[c_md]][1])))
  }

  # ANOVA / linear-model tables: one row per term, an F column, a name column.
  c_f <- jmv_col(d, c("F", "f"))
  if (!is.null(c_f)) {
    c_name <- jmv_col(d, c("name", "term", "source"))
    labels <- if (!is.null(c_name)) as.character(d[[c_name]]) else rownames(d)
    for (i in seq_along(labels)) {
      lab <- tolower(gsub("[^A-Za-z0-9]+", "_", trimws(labels[i])))
      lab <- gsub("^_+|_+$", "", lab)
      if (!nzchar(lab)) next
      put(paste0("f_", lab), suppressWarnings(as.numeric(d[[c_f]][i])))
      c_dfcol <- jmv_col(d, c("df"))
      if (identical(lab, "residuals") && !is.null(c_dfcol)) {
        put("df_resid", suppressWarnings(as.numeric(d[[c_dfcol]][i])))
      }
    }
  }

  # Model fit: R-squared.
  c_r2 <- jmv_col(d, c("r2", "rsq", "r.squared"))
  if (!is.null(c_r2)) put("r_squared", suppressWarnings(as.numeric(d[[c_r2]][1])))

  # Regression coefficients. jmv's linReg puts these in models[[i]]$coef with
  # `term` / `est` columns (rownames are opaque hashes, so the TERM column is
  # the only usable label). Map each term onto `coef_<term>`, with jmv's
  # "Intercept" normalised to the `coef_intercept` the arena asks for.
  c_term <- jmv_col(d, c("term", "name"))
  c_est  <- jmv_col(d, c("est", "estimate", "b"))
  if (!is.null(c_term) && !is.null(c_est)) {
    terms <- as.character(d[[c_term]])
    for (i in seq_along(terms)) {
      lab <- tolower(trimws(terms[i]))
      lab <- gsub("^\\(?intercept\\)?$", "intercept", lab)
      lab <- gsub("[^a-z0-9_]", "", lab)
      if (!nzchar(lab)) next
      put(paste0("coef_", lab), suppressWarnings(as.numeric(d[[c_est]][i])))
    }
  }
}

# 2d. Base-R objects: htest (t.test), lm, and anova/car::Anova tables.
for (o in objs) {
  v <- tryCatch(get(o, envir = penv), error = function(e) NULL)
  if (is.null(v)) next

  if (inherits(v, "htest")) {
    # ONLY a t-test may fill t_statistic/df/p_value. A translation legitimately
    # runs assumption checks alongside the target analysis — SPSStoR's real
    # output calls car::leveneTest() before t.test(), and normality checks are
    # common — and every one of those is also class "htest". Harvesting the
    # first htest found would silently score Shapiro's W (0.986) as the
    # t-statistic (gold -3.147). Identify the method before believing it.
    meth <- tolower(paste(v$method, collapse = " "))
    stat_name <- tolower(paste(names(v$statistic), collapse = ""))
    is_t_test <- grepl("t-test", meth, fixed = TRUE) || identical(stat_name, "t")
    if (is_t_test) {
      put("t_statistic", unname(v$statistic))
      put("df", unname(v$parameter))
      put("p_value", v$p.value)
      if (!is.null(v$estimate) && length(v$estimate) == 2) {
        put("mean_diff", unname(v$estimate[1] - v$estimate[2]))
      }
    }
  }

  if (inherits(v, "lm")) {
    co <- tryCatch(stats::coef(v), error = function(e) NULL)
    if (!is.null(co)) {
      for (cn in names(co)) {
        key <- paste0("coef_", tolower(cn))
        key <- gsub("\\(intercept\\)", "intercept", key)
        key <- gsub("[^a-z0-9_]", "", key)
        put(key, unname(co[[cn]]))
      }
    }
    s <- tryCatch(summary(v), error = function(e) NULL)
    if (!is.null(s$r.squared)) put("r_squared", s$r.squared)
    put("resid_df", unname(v$df.residual))
    put("n_analysed", tryCatch(unname(stats::nobs(v)), error = function(e) NULL))
  }

  if (is.data.frame(v) && "F value" %in% colnames(v)) {
    rn <- rownames(v)
    for (i in seq_along(rn)) {
      key <- paste0("f_", tolower(gsub("[^A-Za-z0-9]", "_", rn[i])))
      put(key, v[i, "F value"])
      if (identical(tolower(rn[i]), "residuals") && "Df" %in% colnames(v)) {
        put("df_resid", v[i, "Df"])
      }
    }
  }
}

# An empty R list serialises as `[]`, which the Python side cannot read as a
# mapping — it would look like a harness fault when in fact the code simply ran
# and produced none of the requested statistics. Emit an explicit empty object.
if (!length(found)) {
  cat("{}")
} else {
  cat(toJSON(found, auto_unbox = TRUE, digits = 12, na = "null"))
}
