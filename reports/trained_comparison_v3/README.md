# Matched local trained-model comparison, diagnostic v3

This experiment fits Ridge and LightGBM on hourly load available before each test period. It uses all 60 frozen cases, the same 24-hour and 168-hour horizons, and the exact test origins and targets used for Chronos-Bolt and Chronos-2. It is a retrospective diagnostic: earlier test outcomes were inspected before this protocol was designed. It cannot serve as independent confirmation that Chronos beats well-trained local models.

## Protocol

- Every model predicts a correction to weekly persistence using origin-observable load lags, recent statistics, and calendar features. No future target or weather value is used as a feature. The prior v2 day-of-year terms were removed to reduce extrapolation across quarters.
- Ridge alpha is chosen from 1, 100, and 1000. LightGBM uses 300 trees with 15 or 31 leaves. Both families choose between 28 and 84 days of recent eligible origins. A deterministic sample of at most 720 fit origins covers every origin hour.
- For Q2 through Q4, settings are chosen on the frozen validation origins using models fitted on earlier labels. Q1 has no mature labels before the frozen validation period; it instead uses an inner time split within pre-test history. Training labels end before the first inner validation origin. Each selected model is then refitted on mature history before the first test origin and held fixed across test.
- Ridge or LightGBM is selected using validation MAE. The selected seasonal reference remains separate. The original Bolt and Chronos-2 forecast archives are matched to the frozen arrays before scoring.
- Paired uncertainty uses 5,000 circular-block resamples of hourly origin losses with 168-hour and 336-hour blocks, the same rule as the main paper. Forecasts and selected settings remain fixed in bootstrap resamples. Intervals are unadjusted for multiple comparisons.

## Results

| Horizon | Selected trained lower MAE than selected seasonal | Lower MAE than Bolt | Lower MAE than Chronos-2 |
|---|---:|---:|---:|
| 24 h | 7 / 30 | 4 / 30 | 4 / 30 |
| 168 h | 4 / 30 | 4 / 30 | 4 / 30 |

All eight selected-trained wins against either Chronos model, four at each horizon, are the Q1 through Q4 Portugal T3 cases. This meter is nearly week-repeating in the published data, and weekly persistence is often exact or nearly exact. Outside T3, the selected trained model has higher test MAE than the selected seasonal reference in 19 of 26 day-ahead cases and 22 of 26 week-ahead cases. This shows the local learners are not consistently strong challengers even to the seasonal reference.

Against the validation-selected trained model, both Chronos models have clear paired support in 24 of 30 day-ahead cases and 21 of 30 week-ahead cases. The trained model has clear support in the four T3 cases at either horizon. Two cases per horizon have limited data; three additional week-ahead comparisons are unclear or block-sensitive. These counts describe this inspected benchmark and should not be used to claim general superiority over trained forecasting methods.

## Files and checks

`identity.json` pins input archive hashes, source hashes, packages, and model settings. `predictions/` contains one forecast matrix and selection record for every case. `scores.csv` contains 420 method scores. `uncertainty.json` contains 360 paired comparisons. `complete.json` records the completed 60 cases. `python src/verify_trained_v3.py` independently checked the 60 cases, all 420 scores, the 360 comparisons, forecast shapes, selected-family identity, and train/validation/test boundaries.

Reproduce the trained forecasts with `python src/trained_comparison_v3.py run` and the intervals with `python src/trained_comparison_v3.py analyze`. The commands resume only with identical code and inputs. Earlier trained-model results remain separate in their existing report directories.
