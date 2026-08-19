* Tier 2 - independent-samples t-test of score by group.
* NOTE: SPSS reports the pooled (equal-variance assumed) test as its primary row.

T-TEST GROUPS=group(1 2)
  /VARIABLES=score
  /CRITERIA=CI(.95).
