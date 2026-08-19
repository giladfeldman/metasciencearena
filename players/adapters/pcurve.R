#!/usr/bin/env Rscript
# p-curve-v1 tool player: a FAITHFUL implementation of the Simonsohn, Nelson &
# Simons (2014) p-curve right-skew test.
#
# Reads a task envelope from stdin (a SET of significant findings, each a test
# statistic), computes each finding's exact two-sided p-value, keeps the significant
# ones (p < .05), forms the right-skew pp-values (pp_i = p_i / .05), z-transforms them
# (z_i = qnorm(pp_i)), combines via Stouffer (Z = sum(z_i)/sqrt(k)), and emits the
# arena output schema JSON on stdout:
#   evidential_value = (right_skew_p < .05)      # full-curve right-skew test
#   right_skew_p     = pnorm(Z)
#   flatness_p       = the optional 33%-power flatness diagnostic
#   confidence       = 1.0                         # deterministic statistical tool
#
# dmetar (which wraps p-curve) is not available for R 4.4 and puniform implements
# p-uniform* (a sibling, not p-curve), so the published algorithm is implemented
# directly here using base-R distribution functions (pt/pf/pnorm/pchisq/qnorm).
suppressPackageStartupMessages({
  library(jsonlite)
})

SIG_ALPHA  <- 0.05
SKEW_ALPHA <- 0.05

input    <- fromJSON(readLines(con = "stdin", warn = FALSE), simplifyVector = FALSE)
findings <- input$input$findings

# Exact two-sided p-value for each supported statistic. p-curve uses F only with df1=1.
two_sided_p <- function(f) {
  kind <- f$type
  v    <- as.numeric(f$value)
  if (kind == "t") {
    df <- as.numeric(if (!is.null(f$df2)) f$df2 else f$df1)
    return(2 * pt(abs(v), df = df, lower.tail = FALSE))
  } else if (kind == "F") {
    df1 <- as.numeric(if (!is.null(f$df1)) f$df1 else 1)
    df2 <- as.numeric(f$df2)
    if (abs(df1 - 1) > 1e-9) stop("p-curve only accepts F with df1 == 1")
    return(pf(v, df1 = 1, df2 = df2, lower.tail = FALSE))
  } else if (kind == "z") {
    return(2 * pnorm(abs(v), lower.tail = FALSE))
  } else if (kind == "chi2") {
    return(pchisq(v, df = 1, lower.tail = FALSE))
  } else if (kind == "r") {
    n  <- as.numeric(f$n)
    df <- n - 2
    t  <- v * sqrt(df / max(1 - v * v, 1e-12))
    return(2 * pt(abs(t), df = df, lower.tail = FALSE))
  }
  stop(sprintf("unknown finding type %s", kind))
}

# pp-value of an observed p under the 33%-power alternative (flatness diagnostic),
# conditional on significance, in p-curve's standardized-normal scale. Monotone in p
# so a right-skewed (small-p) curve gives a small flatness_p.
pp_flat_33 <- function(p) {
  z_obs  <- qnorm(1 - p / 2)
  z_crit <- qnorm(0.975)
  ncp    <- z_crit - qnorm(1 / 3)
  num    <- pnorm(z_obs - ncp) - pnorm(z_crit - ncp)
  den    <- 1 - pnorm(z_crit - ncp)
  1 - (num / den)
}

ps <- c()
for (f in findings) {
  p <- two_sided_p(f)
  if (p < SIG_ALPHA) {
    ps <- c(ps, min(max(p, 1e-12), SIG_ALPHA - 1e-12))
  }
}

k <- length(ps)
if (k == 0) {
  out <- list(evidential_value = FALSE, right_skew_p = 1.0, flatness_p = 1.0,
              confidence = 1.0)
} else {
  z_skew       <- qnorm(ps / SIG_ALPHA)
  Z_skew       <- sum(z_skew) / sqrt(k)
  right_skew_p <- pnorm(Z_skew)

  pp_flat <- pmin(pmax(vapply(ps, pp_flat_33, numeric(1)), 1e-12), 1 - 1e-12)
  Z_flat  <- sum(qnorm(pp_flat)) / sqrt(k)
  flatness_p <- pnorm(Z_flat)

  out <- list(
    evidential_value = isTRUE(right_skew_p < SKEW_ALPHA),
    right_skew_p     = round(right_skew_p, 8),
    flatness_p       = round(flatness_p, 8),
    confidence       = 1.0
  )
}

cat(toJSON(out, auto_unbox = TRUE, null = "null", na = "null", digits = 10))
