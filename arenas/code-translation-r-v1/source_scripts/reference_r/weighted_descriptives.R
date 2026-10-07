# Reference translation - tier 8 frequency-weighted descriptives.
#
# THE TRAP: SPSS `WEIGHT BY w` is STATEFUL — it applies to every procedure that
# follows until `WEIGHT OFF`, and it changes the valid N as well as the mean.
# Base R has no global weighting state: mean() ignores weights entirely, so a
# translation that drops the WEIGHT statement returns the UNWEIGHTED mean and
# looks perfectly reasonable. Here the weight is correlated with score, so the
# two differ by ~5 points.

df <- read.csv(Sys.getenv("ARENA_DATA"), stringsAsFactors = FALSE)

ok <- !is.na(df$score) & !is.na(df$w)

out <- list(
  weighted_mean_score = stats::weighted.mean(df$score[ok], df$w[ok]),
  # A frequency weight means "this row stands for w cases", so the weighted N is
  # the sum of the weights, not the number of rows.
  weighted_n = sum(df$w[ok])
)

cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
