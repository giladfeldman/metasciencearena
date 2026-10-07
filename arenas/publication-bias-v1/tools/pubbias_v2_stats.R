#!/usr/bin/env Rscript
# publication-bias-v1, task set v2: every scored statistic, computed ONE way.
#
# This script is BOTH the gold computer (tools/build_gold_v2.py runs it on every
# generated dataset) and the reference tool player (`r-pubbias-suite` in
# players/registry.yaml runs it through RCliAdapter). It reads one task envelope
# as JSON on stdin and writes the v2 output schema (schemas/v2/output.schema.json)
# on stdout.
#
# Each statistic below is the method EXACTLY as players/prompts/publication_bias_v2.txt
# specifies it. The prompt and this file must change together; the arena test
# test_v2_prompt_names_every_scored_statistic pins the field list on both sides.
#
#   egger.p            classic Egger test: lm(yi ~ sei, weights = 1/vi); two-sided
#                      t-test of the sei slope on k-2 df (metafor regtest(model="lm")).
#   begg.tau / .p      Begg & Mazumdar rank correlation (metafor::ranktest): Kendall's
#                      tau between (yi - theta_FE)/sqrt(vi - 1/sum(wi)) and vi, exact p.
#   trimfill.k0 / .estimate
#                      Duval & Tweedie trim-and-fill on a REML random-effects model,
#                      estimator L0, side chosen by metafor's default; estimate = the
#                      filled model's pooled effect.
#   pet.estimate       intercept of lm(yi ~ sei,  weights = 1/vi)
#   peese.estimate     intercept of lm(yi ~ vi,   weights = 1/vi)
#   three_psm.estimate / .lrt_p
#                      metafor::selmodel(type = "stepfun", steps = .025,
#                      alternative = "greater") on the random-effects model: one
#                      cutpoint at one-sided p = .025, fitted by ML. `estimate` is the
#                      adjusted mean effect; `lrt_p` is the likelihood-ratio test of
#                      the selection model against the unadjusted model (`LRTp`).
#                      WHY NOT weightr: weightr::weightfunct is the classic 3PSM
#                      implementation, but its optimiser crashed ("non-finite
#                      finite-difference value") on 2 of the 12 v2 revealed sets,
#                      both with tau^2 near 0. On the other 10 the two agree to within
#                      0.0004 on the estimate and on the LRT p (MEASURED 2026-09-30).
#   robma.bf_bias / .mu
#                      RoBMA::RoBMA defaults (the RoBMA-PSMA ensemble), minus two
#                      machine-dependent limits (see ROBMA below): inclusion
#                      Bayes factor for the publication-bias component, and the
#                      model-averaged posterior mean of mu.
#
# REML. metafor's default Fisher scoring fails to converge on a small share of
# datasets (MEASURED 2026-09-30: 1 of the 12 v1 revealed sets). This script always
# fits with control = list(stepadj = 0.5, maxiter = 1000), metafor's documented
# remedy. Against a brute-force optimisation of the REML likelihood it agrees to
# within 2e-5 on tau^2 on all 12 v1 sets, including the one that fails by default.
#
# RoBMA is MCMC. Its seed comes from the environment variable PUBBIAS_ROBMA_SEED
# (default 1). The gold build uses a different seed from the reference player on
# purpose: the player's distance from gold is then a live measurement of MCMC noise
# rather than a tautology.
suppressPackageStartupMessages({
  library(jsonlite)
  library(metafor)
})

REML_CONTROL <- list(stepadj = 0.5, maxiter = 1000)

# ROBMA. RoBMA's defaults are NOT a pure function of (data, seed); two of them depend on the
# machine (debt d-125b6a, MEASURED 2026-10-06):
#   max_time = 60 min  autofit stops extending a sub-model's chains after 60 min of
#                      WALL CLOCK, so a loaded machine gets a shorter chain.
#   restarts = 10      a sub-model whose JAGS run errors is silently re-run with inits
#                      from seed + i; after the last failure it is silently dropped
#                      from the ensemble. Either changes the answer without a trace.
# Both are switched off: autofit extends on convergence alone (max_extend = 10 is an
# iteration count, deterministic), and a JAGS error is not retried. Whenever neither
# limit would have fired the result is identical to the defaults; whenever one would
# have, robma_fail_loudly() stops the run instead of returning a different number.
robma_autofit <- function() RoBMA::set_autofit_control(max_time = NULL, restarts = NULL)

robma_fail_loudly <- function(fit) {
  conv <- vapply(fit$models, function(m) isTRUE(m$converged), logical(1))
  errs <- unique(fit$add_info$errors)
  if (!all(conv) || length(errs) > 0) {
    stop(sprintf("RoBMA: %d of %d sub-models failed; errors: %s",
                 sum(!conv), length(conv), paste(errs, collapse = " | ")), call. = FALSE)
  }
  warns <- unique(fit$add_info$warnings)
  if (length(warns) > 0) message("RoBMA warnings: ", paste(warns, collapse = " | "))
}

# Optional evidence trail. The v2c6 deviations of 2026-10-01 left nothing to inspect
# (silent = TRUE, stderr discarded on success); with PUBBIAS_ROBMA_DIAG_DIR set, every
# fit appends one JSON line per dataset there: per-sub-model sample count, log marginal
# likelihood, convergence and errors, plus wall time. The scored output is unchanged.
robma_diag <- function(fit, task_id, seed, secs, out) {
  dir <- Sys.getenv("PUBBIAS_ROBMA_DIAG_DIR", "")
  if (!nzchar(dir)) return(invisible())
  models <- lapply(fit$models, function(m) list(
    sample    = if (is.null(m$fit$sample)) NA_integer_ else as.integer(m$fit$sample),
    logml     = if (is.null(m$marglik$logml)) NA_real_ else as.numeric(m$marglik$logml),
    converged = isTRUE(m$converged),
    errors    = as.list(m$errors)))
  line <- list(task_id = task_id, seed = seed, secs = secs, result = out,
               r_pid = Sys.getpid(), at = format(Sys.time(), tz = "UTC", usetz = TRUE),
               models = models)
  cat(toJSON(line, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n",
      file = file.path(dir, "robma_diag.jsonl"), append = TRUE, sep = "")
}

compute_v2 <- function(yi, sei, robma_seed = 1L, with_robma = TRUE, task_id = NA_character_) {
  vi <- sei * sei
  k  <- length(yi)
  out <- list()

  # --- Egger, PET, PEESE: weighted least squares with multiplicative error ------
  pet <- lm(yi ~ sei, weights = 1 / vi)
  cf  <- summary(pet)$coefficients
  out$egger <- list(p = unname(cf["sei", "Pr(>|t|)"]))
  out$pet   <- list(estimate = unname(cf["(Intercept)", "Estimate"]))
  peese <- lm(yi ~ vi, weights = 1 / vi)
  out$peese <- list(estimate = unname(coef(peese)[["(Intercept)"]]))

  # --- Begg & Mazumdar rank correlation -----------------------------------------
  rk <- metafor::ranktest(yi, sei = sei)
  out$begg <- list(tau = unname(rk$tau), p = unname(rk$pval))

  # --- Trim-and-fill on the REML random-effects model ---------------------------
  m  <- metafor::rma(yi = yi, vi = vi, method = "REML", control = REML_CONTROL)
  tf <- metafor::trimfill(m)
  out$trimfill <- list(k0 = as.integer(tf$k0), estimate = as.numeric(tf$beta[1]))

  # --- Three-parameter selection model ------------------------------------------
  sel <- metafor::selmodel(m, type = "stepfun", steps = 0.025, alternative = "greater")
  out$three_psm <- list(estimate = as.numeric(sel$beta[1]), lrt_p = as.numeric(sel$LRTp))

  # --- RoBMA --------------------------------------------------------------------
  if (with_robma) {
    suppressPackageStartupMessages(library(RoBMA))
    t0  <- Sys.time()
    fit <- RoBMA::RoBMA(d = yi, se = sei, seed = robma_seed,
                        parallel = FALSE, silent = TRUE,
                        autofit_control = robma_autofit())
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    ok <- all(vapply(fit$models, function(m) isTRUE(m$converged), logical(1)))
    sm <- if (ok) summary(fit) else NULL
    out$robma <- if (ok) list(
      bf_bias = as.numeric(sm$components["Bias", "inclusion_BF"]),
      mu      = as.numeric(sm$estimates["mu", "Mean"])
    )
    robma_diag(fit, task_id, robma_seed, secs, out$robma)
    robma_fail_loudly(fit)
  }
  out
}

if (sys.nframe() == 0L) {
  input   <- fromJSON(readLines(con = "stdin", warn = FALSE), simplifyVector = FALSE)
  studies <- input$input$studies
  yi  <- vapply(studies, function(s) as.numeric(s$yi),  numeric(1))
  sei <- vapply(studies, function(s) as.numeric(s$sei), numeric(1))
  seed <- as.integer(Sys.getenv("PUBBIAS_ROBMA_SEED", "1"))
  with_robma <- Sys.getenv("PUBBIAS_SKIP_ROBMA", "") == ""
  res <- compute_v2(yi, sei, robma_seed = seed, with_robma = with_robma,
                    task_id = if (is.null(input$task_id)) NA_character_ else input$task_id)
  cat(toJSON(res, auto_unbox = TRUE, digits = NA, null = "null", na = "null"))
}
