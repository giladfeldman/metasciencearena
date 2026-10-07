* Tier 7 - correlations under PAIRWISE then LISTWISE deletion.
* NOTE: CORRELATIONS defaults to PAIRWISE, unlike REGRESSION which is LISTWISE.
* The two give different r and different N whenever missingness does not overlap.

CORRELATIONS
  /VARIABLES=age hours score
  /MISSING=PAIRWISE.

CORRELATIONS
  /VARIABLES=age hours score
  /MISSING=LISTWISE.
