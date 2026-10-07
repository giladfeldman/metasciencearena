* Tier 4 - recode, missing declaration, and a computed index.
* NOTE: 99 on item4 is set to missing BEFORE the index is computed. The index
* requires at least 3 non-missing items, mirroring SPSS MEAN.3.

replace item4 = . if item4 == 99

recode item1 (4/5 = 1) (1/3 = 0), generate(item1_high)
label variable item1_high "Item 1 endorsed (4 or 5)"

egen _nvalid = rownonmiss(item1 item2 item3 item4)
egen index = rowmean(item1 item2 item3 item4)
replace index = . if _nvalid < 3
drop _nvalid

tabulate item1_high
summarize index
