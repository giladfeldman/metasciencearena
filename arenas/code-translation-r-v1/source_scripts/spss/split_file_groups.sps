* Tier 9 - split the file, then describe each group separately.
* NOTE: SPLIT FILE is STATEFUL - every procedure that follows runs once PER
* GROUP until SPLIT FILE OFF. R has no equivalent; the translation must be
* restructured into a grouped operation.

SORT CASES BY group.
SPLIT FILE BY group.

DESCRIPTIVES VARIABLES=score
  /STATISTICS=MEAN.

SPLIT FILE OFF.
