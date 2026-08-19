# Reference translation - tier 4 recode / user-missing / computed index.
#
# THE TRAPS:
#  1. SPSS `MISSING VALUES item4 (99).` makes 99 user-missing, so it is excluded
#     everywhere downstream. In R, 99 stays a literal value and inflates the
#     index unless it is converted to NA FIRST.
#  2. SPSS MEAN.3(...) requires >= 3 valid values, returning missing otherwise.
#     rowMeans(na.rm = TRUE) has no such minimum.

df <- read.csv(Sys.getenv("ARENA_DATA"), stringsAsFactors = FALSE)

df$item4[df$item4 == 99] <- NA

df$item1_high <- ifelse(df$item1 >= 4, 1, 0)

items <- df[, c("item1", "item2", "item3", "item4")]
n_valid <- rowSums(!is.na(items))
df$index <- ifelse(n_valid >= 3, rowMeans(items, na.rm = TRUE), NA)

out <- list(
  n_recoded_high  = sum(df$item1_high == 1, na.rm = TRUE),
  mean_index      = mean(df$index, na.rm = TRUE),
  n_missing_after = sum(is.na(df$item4))
)

cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
