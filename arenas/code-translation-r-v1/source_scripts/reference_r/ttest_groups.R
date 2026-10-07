# Reference translation - tier 2 independent-samples t-test.
#
# THE TRAP: SPSS T-TEST and Stata `ttest` headline the POOLED (equal-variance)
# test. R's t.test() defaults to var.equal = FALSE (Welch). Omitting
# var.equal = TRUE silently changes df and p.

df <- read.csv(Sys.getenv("ARENA_DATA"), stringsAsFactors = FALSE)

tt <- t.test(score ~ group, data = df, var.equal = TRUE)

m <- tapply(df$score, df$group, mean, na.rm = TRUE)

out <- list(
  t_statistic = unname(tt$statistic),
  df          = unname(tt$parameter),
  p_value     = tt$p.value,
  # group 1 minus group 2, matching the SPSS/Stata ordering.
  mean_diff   = unname(m[["1"]] - m[["2"]])
)

cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
