# Reference translation - tier 7 pairwise vs listwise correlations.
#
# THE TRAP: SPSS CORRELATIONS defaults to PAIRWISE deletion while SPSS
# REGRESSION defaults to LISTWISE. A translator that applies one missing-data
# convention uniformly gets exactly one of them wrong. Here age and hours have
# NON-OVERLAPPING missingness, so the two rules keep different case sets and
# produce both a different r and a different N.

df <- read.csv(Sys.getenv("ARENA_DATA"), stringsAsFactors = FALSE)

vars <- df[, c("age", "hours", "score")]

# Pairwise: each cell uses every case complete on ITS OWN two variables.
r_pw <- cor(vars, use = "pairwise.complete.obs")
n_pw <- sum(stats::complete.cases(df$age, df$hours))

# Listwise: every cell uses only cases complete on ALL THREE variables.
r_lw <- cor(vars, use = "complete.obs")
n_lw <- sum(stats::complete.cases(vars))

out <- list(
  r_pairwise_age_hours = unname(r_pw["age", "hours"]),
  n_pairwise_age_hours = n_pw,
  r_listwise_age_hours = unname(r_lw["age", "hours"]),
  n_listwise_age_hours = n_lw
)

cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
