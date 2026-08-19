# Reference translation - tier 6 filter-then-model pipeline.
#
# THE TRAP: SPSS SELECT IF / Stata keep drop cases PERMANENTLY, so the model is
# fitted on the filtered subset. Hoisting the model above the filter, or
# applying the filter only inside the model call, analyses a different N.
#
# Note hours has genuine NAs; `hours >= 20` is NA for those rows, and SPSS
# SELECT IF drops a case whose condition is not true. subset() does the same.

df <- read.csv(Sys.getenv("ARENA_DATA"), stringsAsFactors = FALSE)

df <- subset(df, hours >= 20)

fit <- lm(score ~ age + group, data = df)
co <- coef(fit)

out <- list(
  # N actually entering the model (after listwise deletion on age).
  n_analysed = unname(nobs(fit)),
  coef_age   = unname(co[["age"]]),
  coef_group = unname(co[["group"]]),
  r_squared  = summary(fit)$r.squared
)

cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
