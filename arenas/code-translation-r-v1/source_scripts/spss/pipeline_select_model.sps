* Tier 6 - filter cases, then model the surviving subset.
* NOTE: SELECT IF PERMANENTLY drops cases for everything that follows, so the
* regression below is fitted on the filtered sample only. Statement order matters.

SELECT IF (hours >= 20).
EXECUTE.

REGRESSION
  /MISSING LISTWISE
  /STATISTICS COEFF R ANOVA
  /DEPENDENT score
  /METHOD=ENTER age group.
