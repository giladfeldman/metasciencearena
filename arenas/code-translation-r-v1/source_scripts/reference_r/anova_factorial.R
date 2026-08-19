# Reference translation - tier 5 two-way factorial ANOVA.
#
# THE TRAP: SPSS UNIANOVA and Stata `anova` report TYPE III sums of squares.
# R's anova(aov(...)) gives TYPE I (sequential), which differs whenever the
# cells are unbalanced — and they are here, by design.
#
# Type III also REQUIRES sum-to-zero contrasts. Calling car::Anova(type = 3) on
# a model fitted with R's default treatment contrasts yields a meaningless
# main-effect test: setting the type without the contrasts is still wrong.

df <- read.csv(Sys.getenv("ARENA_DATA"), stringsAsFactors = FALSE)
df$group <- factor(df$group)
df$condition <- factor(df$condition)

op <- options(contrasts = c("contr.sum", "contr.poly"))
fit <- lm(score ~ group * condition, data = df)
tab <- car::Anova(fit, type = 3)
options(op)

out <- list(
  f_group       = tab["group", "F value"],
  f_condition   = tab["condition", "F value"],
  f_interaction = tab["group:condition", "F value"],
  df_resid      = tab["Residuals", "Df"]
)

cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
