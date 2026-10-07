* Tier 4 - recode, user-missing declaration, and a computed index.
* NOTE: 99 on item4 is DECLARED USER-MISSING and is therefore excluded from
* every subsequent computation. MEAN.3 requires at least 3 valid values.

MISSING VALUES item4 (99).

RECODE item1 (4 thru 5=1) (1 thru 3=0) INTO item1_high.
VARIABLE LABELS item1_high 'Item 1 endorsed (4 or 5)'.

COMPUTE index = MEAN.3(item1, item2, item3, item4).
EXECUTE.

FREQUENCIES VARIABLES=item1_high.
DESCRIPTIVES VARIABLES=index
  /STATISTICS=MEAN.
