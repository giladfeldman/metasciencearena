* Tier 5 - two-way factorial ANOVA with interaction.
* NOTE: UNIANOVA reports TYPE III sums of squares by default. The cells are
* unbalanced, so Type I (sequential) SS would give different F values.

UNIANOVA score BY group condition
  /METHOD=SSTYPE(3)
  /INTERCEPT=INCLUDE
  /DESIGN=group condition group*condition.
