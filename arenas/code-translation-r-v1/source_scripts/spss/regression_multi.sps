* Tier 3 - OLS regression of score on age and hours.
* NOTE: /MISSING LISTWISE drops any case missing on ANY model variable.

REGRESSION
  /MISSING LISTWISE
  /STATISTICS COEFF R ANOVA
  /DEPENDENT score
  /METHOD=ENTER age hours.
