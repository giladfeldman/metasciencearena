# Reference translation - tier 9 SPLIT FILE.
#
# THE TRAP: SPSS `SPLIT FILE BY group` makes every following procedure run
# SEPARATELY per group until `SPLIT FILE OFF`. It is stateful and has NO R
# equivalent — the correct translation restructures the program into a grouped
# operation, which is a different SHAPE of code rather than a different
# argument. A translator that ignores the split reports one pooled mean and
# silently answers a different question.

df <- read.csv(Sys.getenv("ARENA_DATA"), stringsAsFactors = FALSE)

means <- tapply(df$score, df$group, mean, na.rm = TRUE)
ns    <- tapply(df$score, df$group, function(x) sum(!is.na(x)))

out <- list(
  mean_score_group1 = unname(means[["1"]]),
  mean_score_group2 = unname(means[["2"]]),
  n_group1          = unname(ns[["1"]]),
  n_group2          = unname(ns[["2"]])
)

cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
