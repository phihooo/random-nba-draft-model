# Grouped, nested evaluation of college-stat families

## Question

Do college-stat families add useful information after draft position is already
known, or are their individually significant correlations mostly shared and
redundant?

## Evaluation design

`03_grouped_nested_cv.py` uses a leakage-safe design:

1. One complete draft year is held out at a time.
2. Ridge regularization is tuned only on the remaining years with an inner
   five-fold `GroupKFold`.
3. The draft-pick value curve, missing-value imputation, feature scaling, and
   PCI standardization are all fitted inside the training fold.
4. Raw VORP is predicted directly. The stored `surplus_value` is not used,
   because its draft-slot averages were calculated from the full dataset.
5. Model differences are paired by draft year. Confidence intervals resample
   draft years, exact sign-flip tests operate on year-level MAE differences,
   and p-values are corrected within each set of family comparisons using
   Benjamini-Hochberg.

Two equal-opportunity outcomes are tested:

- `first_3yr_vorp`: mean VORP in the first three post-draft seasons, inserting
  zero for a season not played; 682 players from 15 drafts.
- `best_3yr_vorp_first5`: best three-season mean within the first five
  post-draft seasons; 596 players from the 2009–2021 drafts.

This avoids the original peak target's unequal search window, where older
players had many more seasons in which to find their best three-year stretch.

## Main results

### First three post-draft seasons

| Model | MAE | RMSE | Held-out r | MAE change vs pick | Adjusted q |
|---|---:|---:|---:|---:|---:|
| Draft pick only | 0.3824 | 0.5942 | 0.351 | — | — |
| Pick + fixed PCI | 0.3802 | 0.5807 | 0.407 | +0.0022 | 0.569 |
| Pick + possession family | 0.3789 | 0.5778 | 0.417 | +0.0035 | 0.569 |
| Pick + scoring family | 0.3874 | 0.5927 | 0.358 | −0.0049 | 0.925 |
| Pick + impact/offense family | 0.3742 | 0.5679 | 0.449 | +0.0083 | 0.512 |
| Pick + defense/size family | 0.3809 | 0.5811 | 0.402 | +0.0016 | 0.569 |
| Pick + context/age family | 0.3861 | 0.5944 | 0.350 | −0.0036 | 0.925 |
| Pick + all college stats | 0.3749 | 0.5689 | 0.449 | +0.0076 | 0.512 |

Positive MAE change means lower error. None of the additions beats the
draft-pick baseline reliably after correction. For fixed PCI, the mean
year-level MAE improvement has a 95% interval of **−0.0062 to +0.0107**, so the
data remain compatible with a small benefit, no effect, or a small cost.

### Redundancy test: remove one family from the full model

| Family restored to ablated model | Full-model MAE benefit | 95% year-clustered interval | Adjusted q |
|---|---:|---:|---:|
| Possession | +0.0043 | −0.0009 to +0.0085 | 0.092 |
| Scoring | −0.0009 | −0.0025 to +0.0007 | 0.860 |
| Impact/offense | +0.0057 | +0.0020 to +0.0094 | **0.029** |
| Defense/size | +0.0027 | −0.0009 to +0.0065 | 0.111 |
| Context/age | +0.0033 | +0.0005 to +0.0066 | 0.064 |

Only the impact/offense family has a detectable unique contribution at the 5%
FDR threshold, and its practical benefit is small. The possession result is
suggestive but does not clear the corrected threshold.

### Best three-year stretch within the first five seasons

The sensitivity result is weaker. Draft pick alone has MAE 0.5737 and RMSE
0.8551. Fixed PCI improves pooled MAE by only 0.0018; the full college model
makes pooled MAE 0.0048 worse. No add-one or leave-one-family-out comparison is
significant after correction (`q >= 0.78` for additions and `q >= 0.98` for
full-model ablations).

## Interpretation

The original single-stat tests were detecting real marginal associations, but
those associations do not imply that every stat adds new predictive
information. In this stricter evaluation:

- much of the signal is shared across correlated measures of general college
  quality;
- PCI changes held-out rankings more than it changes typical prediction error;
- the possession family has, at most, a small incremental effect beyond draft
  position;
- only broad impact/offense measures show a small unique early-career effect,
  and that result is not robust to the five-year-peak outcome.

The appropriate conclusion is therefore not that college statistics contain
no signal. It is that **most of their statistically significant signal is
redundant with draft position and with other college-stat families, and the
remaining out-of-sample gains are small and unstable across draft classes**.

Detailed values and every held-out prediction are saved in:

- `data/grouped_nested_cv_model_metrics.csv`
- `data/grouped_nested_cv_comparisons.csv`
- `data/grouped_nested_cv_oof_predictions.csv`
- `data/grouped_nested_cv_position_metrics.csv`

The position-metrics file evaluates the same pooled, leakage-safe OOF
predictions separately within PG/SG/SF/PF/C. It is a descriptive error audit;
first-NBA-season position is not passed into any predictive model.
