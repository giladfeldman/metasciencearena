# Reference translation - tier 1 descriptives.
# Emits the gold statistics as a JSON object on stdout. Every reference
# translation and every player script follows this same contract, so the scorer
# compares like with like.

df <- read.csv(Sys.getenv("ARENA_DATA"), stringsAsFactors = FALSE)

out <- list(
  mean_age   = mean(df$age,   na.rm = TRUE),
  sd_age     = sd(df$age,     na.rm = TRUE),
  n_age      = sum(!is.na(df$age)),
  mean_score = mean(df$score, na.rm = TRUE),
  sd_score   = sd(df$score,   na.rm = TRUE),
  n_score    = sum(!is.na(df$score))
)

cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
