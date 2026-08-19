* Tier 8 - frequency-weighted descriptives.
* NOTE: WEIGHT BY is STATEFUL - it applies to every procedure that follows
* until WEIGHT OFF, and it changes the valid N as well as the mean.

WEIGHT BY w.

DESCRIPTIVES VARIABLES=score
  /STATISTICS=MEAN.

WEIGHT OFF.
