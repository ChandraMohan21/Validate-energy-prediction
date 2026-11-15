# Forecast quality audit

A diagnostic follow-up after reviewing results. Original benchmarks are preserved; no series is dropped and no model is selected using these test diagnostics.

## Portugal T3

All three selected Monash series match the corresponding columns in Lai's upstream processed hourly dataset for all 26,304 observations. Their 2014 processed CSVs match Monash exactly. T3 has 99.395% exact agreement with the observation 168 hours earlier over 2014. Its longest uninterrupted equality run lasts 6477 hours, from 2014-03-30 03:00:00 through 2014-12-24 23:00:00.

The strong repetition was not introduced by this project's preprocessing. All three selected series already have integer values in the published hourly archive. Exact agreement has not been independently traced back to the original 15-minute UCI meter file, and no physical explanation for T3's repetition is established. Do not call it synthetic, corrupt, or representative of ordinary customers without additional evidence. Preserve it in the primary results and report its behavior separately; do not average percentage gains when baseline MAE is zero or nearly zero.

[Monash source](https://zenodo.org/records/4656140) identifies this as a processed hourly version of [Lai's data](https://github.com/laiguokun/multivariate-time-series-data). [Original UCI documentation](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) describes meter creation zeros and daylight-saving conventions; this does not establish the cause of T3's pattern.

## Ridge and training-origin coverage

All 23 seven-day Ridge cases were reconstructed from the original choices and refit data and reproduce the saved predictions. The audit found a training-design limitation: selecting every 24th origin from midnight means all fitting/refit origins are at 00:00, while test origins span all 24 hours. The number of daily training origins was disclosed, but the resulting origin-hour coverage mismatch was not previously assessed. This can disadvantage models that use origin load features. It does not imply future-data leakage, but it limits claims about well-trained tabular alternatives.

For Q2 Tetouan Zone 1, Ridge MAE is 6005.15, versus weekly MAE 1874.87. The refitted linear model's standardized two-week average-load feature contributes an average -5112.86 to its residual correction. 100.0% of test rows have that feature outside its refit range. The total predicted mean correction is -5374.78, while the actual mean correction was +519.28.

These are algebraic contribution and distribution-shift diagnostics, not a causal attribution: correlated linear features can offset one another. Both annual calendar features and rolling load features can be outside the range seen during fitting. A limited regularization grid and partial-year fitting also constrain this Ridge setup. The present results do not establish that Chronos beats a strong, broadly tuned linear forecasting approach.

## Recommended next experiment

Keep this run as version 1. Define a version 2 before running it: cover all origin hours in training; consider all eligible hourly origins or a deterministic balanced schedule; choose any feature/regularization variants using earlier validation only. Apply the same training-origin correction to both Ridge and LightGBM. Preserve the existing test dates and report that they have already been inspected. Strong confirmatory claims need new, untouched periods or datasets. Do not silently overwrite version 1 or remove T3 because a model loses there.

Reproduce with `python src/audit_forecast_inputs.py`. The upstream gzip must be present at `data/raw/baseline_pilot/electricity_lai_original.txt.gz`. audit.json records its SHA256 and original-output hashes.
