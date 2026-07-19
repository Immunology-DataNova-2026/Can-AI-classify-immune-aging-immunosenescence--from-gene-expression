# Control experiments

Loading GEO series (cached after the first run)...

- training samples: 68  (26 younger, 42 older)
- external samples: 23  (11 younger, 12 older)
- candidate genes: 49,372

## CONTROL 1 — random-gene null distribution

Does the ANOVA filter actually do work, or would any 25 genes score this well?

- random 25-gene panels, external AUC: mean 0.587, sd 0.122, range 0.311-0.879
- random 25-gene panels, external accuracy: mean 0.525
- **selected panel: AUC 0.811 — above 97% of random draws**
- empirical p-value: 0.030

## CONTROL 2 — how much of the signal lives in the top gene?

- top   1 genes: external acc 0.522, AUC 0.962
- top   3 genes: external acc 0.783, AUC 0.947
- top   5 genes: external acc 0.783, AUC 0.886
- top  10 genes: external acc 0.783, AUC 0.841
- top  25 genes: external acc 0.783, AUC 0.811
- top  50 genes: external acc 0.783, AUC 0.803
- top 100 genes: external acc 0.783, AUC 0.788

(Selection here is fit on the full training set, matching how the final model is fitted before external validation.)

## CONTROL 3 — what is refitting the filter inside each fold worth?

- filter refit inside each fold (correct): acc 0.882, AUC 0.932
- filter fit once on all data (leaky):     acc 0.956, AUC 0.980
- **inflation from selection leakage: +0.048 AUC**

Paste into Table 7, row 'Filter refit inside folds'.

## CONTROL 4 — empirical ROC and PR curves

- internal_cv: AUC 0.932, average precision 0.952, 16 ROC points
- external: AUC 0.811, average precision 0.834, 10 ROC points

Written to results/curves.json — plot these instead of the schematic in Figure 4.

## CONTROL 5 — bootstrap confidence intervals

- internal_cv: AUC 0.932, 95% CI [0.856, 0.986] (2000 valid resamples)
- external: AUC 0.811, 95% CI [0.592, 0.962] (2000 valid resamples)