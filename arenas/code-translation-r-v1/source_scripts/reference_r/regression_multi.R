# Reference translation - tier 3 OLS regression.
#
# SPSS /MISSING LISTWISE and Stata's regress both drop cases incomplete on any
# model variable; lm()'s default na.action = na.omit matches that.

df <- read.csv(Sys.getenv("ARENA_DATA"), stringsAsFactors = FALSE)

fit <- lm(score ~ age + hours, data = df)
co <- coef(fit)

out <- list(
  coef_age       = unname(co[["age"]]),
  coef_hours     = unname(co[["hours"]]),
  coef_intercept = unname(co[["(Intercept)"]]),
  r_squared      = summary(fit)$r.squared,
  resid_df       = unname(fit$df.residual)
)

cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
