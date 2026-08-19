* Tier 6 - filter cases, then model the surviving subset.
* NOTE: keep permanently drops cases, so the regression is fitted on the
* filtered sample only. Statement order matters.

keep if hours >= 20

regress score age group
