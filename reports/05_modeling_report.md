# Modeling Report: 168-hour (7-day) Load Forecasting

Code: `src/forecasting.py` (features, splits, baselines), `src/run_pipeline.py` (AutoML, training, walk-forward), `src/residual_variant.py`, `src/make_figures.py`, `src/validate_project.py`.
Results: `reports/results_corrected.json`, `reports/residual_variant_results.json`.

**Headline finding: a tuned gradient-boosted model does not beat a seasonal-naive baseline on this dataset at a 168-hour horizon.** Details and caveats below.

## Dataset

Power Consumption of Tetouan City, UCI Machine Learning Repository dataset 849. Three urban distribution networks in Tetouan, northern Morocco (Quads, Smir, Boussafou substations, operator Amendis), labelled Zone 1 to Zone 3, recorded every 10 minutes through 2017 with five weather channels. Resampled to hourly means. Source paper: Salam and El Hibaoui (2018), IEEE IRSEC.

## Task and protocol

Predict hourly consumption for each of the next 168 hours from a given forecast origin, separately per zone.

Features, all observable at the origin: horizon, value at origin, `lag24` (origin minus 24h), `lag168` and `lag336` (both anchored to the **target**, so `target - 168` and `target - 336`, which are at or before the origin because the horizon never exceeds 168), rolling mean and standard deviation over 24h and 168h at the origin, target calendar fields, and weather from 168h before the target.

Baselines, both scored on exactly the same rows as the model:

| Baseline | Definition |
|---|---|
| Seasonal naive | value at the same hour one week earlier, `y[target - 168]` |
| Persistence | value at the forecast origin, `y[origin]` |

Split, chronological with a **168-hour embargo** so no split's targets reach into the next:

| Split | Origins | Rows | Effective n (distinct target hours) | Targets end |
|---|---|---|---|---|
| Train | to 2017-09-30 | 1,044,288 | 6,383 | 2017-10-07 23:00 |
| Validation | 2017-10-08 to 2017-11-11 | 141,120 | 1,007 | 2017-11-18 23:00 |
| Test | 2017-11-19 to 2017-12-23 | 141,120 | 1,007 | 2017-12-30 23:00 |

Note the effective sample size. Each target hour is forecast from up to 168 different origins, so the raw row count overstates the evidence by roughly 140x. Any statistical claim should use the effective figure, about 1,000 per split, not 141,120.

Model selection: FLAML AutoML over LightGBM, XGBoost, Random Forest and Extra Trees, 300 seconds per zone, scored on validation only. The winning configuration is then refit on train plus validation with hyperparameters frozen, and scored once on test.

## Results

| Zone | Family chosen | Val MAPE | Test MAPE | Seasonal naive | Persistence | Beats seasonal naive |
|---|---|---|---|---|---|---|
| Zone 1 | Random Forest | 6.24% | 4.20% | **2.86%** | 24.72% | No |
| Zone 2 | Random Forest | 5.61% | 4.81% | **4.26%** | 28.60% | No |
| Zone 3 | XGBoost | 7.33% | 20.64% | **8.73%** | 28.35% | No |

The model comfortably beats persistence in every zone, but persistence is a weak baseline for a weekly-periodic series. Against the seasonal naive, which is the appropriate baseline here, it loses in all three zones.

## Walk-forward validation

Three non-overlapping 42-day windows, same frozen configuration, each fit on everything strictly before the window with the same 168-hour embargo.

| Zone | W1 (Nov 18 to Dec 30) | W2 (Oct 7 to Nov 18) | W3 (Aug 26 to Oct 7) |
|---|---|---|---|
| Zone 1 | 4.52 vs 2.86 | 6.19 vs 4.39 | 6.32 vs 5.64 |
| Zone 2 | 4.79 vs 4.26 | 5.53 vs 5.06 | 9.36 vs 9.11 |
| Zone 3 | 22.10 vs 8.73 | **9.09 vs 10.08** | 18.11 vs 10.92 |

Model beats seasonal naive in **1 of 9** combinations. That single win is Zone 3 window 2, which is the validation period the hyperparameters were selected on, so it is optimistic by construction and should not be counted as evidence of skill.

## Target framing: direct versus residual

A separate experiment, fixed LightGBM (600 trees) rather than AutoML, comparing two ways of using the same features:

- direct: predict `y`
- residual: predict `y - seasonal_naive`, then add the baseline back

The variant was chosen per zone on **validation only**, then scored once on test.

| Zone | Val direct | Val residual | Chosen | Test direct | Test residual | Test chosen | Seasonal naive |
|---|---|---|---|---|---|---|---|
| Zone 1 | 7.59% | 7.08% | residual | 3.26% | 3.53% | 3.53% | 2.86% |
| Zone 2 | 9.20% | 9.52% | direct | 4.53% | 4.86% | 4.53% | 4.26% |
| Zone 3 | 8.78% | 8.87% | direct | 24.96% | 21.45% | 24.96% | 8.73% |

Neither framing beats the baseline. Worth recording honestly: for Zone 1 and Zone 3 the validation choice selected the variant that turned out **worse** on test. Had the variant been picked on test instead, Zone 1 would read 3.26% and Zone 3 would read 21.45%. Those are not the reported numbers, because choosing on test is exactly the selection leakage this project already made once.

## Why the baseline is hard to beat here

These three series are strongly repeatable week to week. Copying last week reproduces the exact shape of the previous week including its ramps and peaks. A model that predicts the level has to reconstruct the weekly shape from calendar features, and averaging many weeks smooths it, which costs more than the model's other advantages gain.

Per-hour error on Zone 1 makes this concrete. The model loses at 22 of the 24 hours, winning only at 01:00 and 14:00 and tying at 13:00. The widest gaps are in the small hours and late evening: 04:00 (6.33% versus 2.09%), 03:00 (5.90% versus 2.21%), 23:00 (5.61% versus 1.83%) and 19:00 (5.32% versus 1.86%). Those are the most repeatable hours of the week, which is precisely where copying last week is hardest to improve on. The model comes closest during daytime hours, where week-to-week variation is larger and calendar structure carries more information. This is the signature of a model reproducing an average weekly profile rather than tracking recent conditions.

## Is the 168-hour horizon the cause?

Tested directly in `src/horizon_experiment.py`, which holds the model, the feature set and the split protocol constant and varies only the horizon. The embargo scales with the horizon. Features are restricted to those legal at every horizon tested, so horizon is not confounded with feature richness.

Model MAPE minus seasonal-naive MAPE, in percentage points. Negative means the model wins.

| Horizon | Zone 1 | Zone 2 | Zone 3 |
|---|---|---|---|
| 1 hour | +0.33 | **-0.23** | **-0.64** |
| 6 hours | +1.06 | +0.57 | +3.74 |
| 24 hours | +2.14 | +1.56 | +7.12 |
| 72 hours | +2.66 | +4.95 | +9.25 |
| 168 hours | +3.12 | +6.77 | +5.12 |

The model wins 2 of 15 horizon-zone combinations, both at one hour ahead, and by small margins. By six hours it loses everywhere, and the gap widens roughly monotonically thereafter.

So the horizon is a substantial part of the explanation, but not all of it: the model also loses at 24 hours, the standard day-ahead horizon, in all three zones.

The mechanism is visible in the baseline columns. Persistence degrades sharply with horizon, from about 6% at one hour to about 25% at a week, exactly as expected. The seasonal naive barely moves: Zone 1 records 2.77, 2.78, 2.78, 2.82 and 2.86 percent across the entire range. For a weekly-periodic series, last week's value at the target hour is almost independent of how far ahead the forecast reaches, so extending the horizon penalises the model without penalising the baseline.

Stated as a finding: **on these series the advantage of a learned model over a seasonal naive is confined to roughly one hour ahead and disappears beyond it.** This also explains why published results on this dataset that forecast one step ahead appear strong.

Caveat: this experiment used a single fixed LightGBM at every horizon rather than a per-horizon AutoML search. The tuned models reported above do better at 168 hours than this one (Zone 1 gap +1.34 rather than +3.12), so the absolute gaps here are pessimistic. The trend is the reliable part.

## Can the gap be closed? Three further experiments

### Feature ceiling (`src/ceiling_experiment.py`)

Adds information in stages to see how far each closes the gap. Stage C deliberately uses weather at the target hour, which is unavailable at forecast time, as a stand-in for a perfect weather forecast. It is an upper bound on achievable skill and is never a claimable result.

| Zone | Seasonal naive | A: base | B: + level shift | C: + oracle weather |
|---|---|---|---|---|
| Zone 1 | **2.86** | 3.43 | 3.44 | 3.49 |
| Zone 2 | **4.26** | 4.46 | 4.46 | 4.69 |
| Zone 3 | **8.73** | 25.31 | 23.41 | 25.24 |

Legal level-shift features (how far the series is currently running above or below the same point last week, plus a third seasonal lag) change nothing on Zones 1 and 2 and help Zone 3 only slightly. Perfect knowledge of future weather makes every zone marginally worse, which is what happens when added inputs carry no usable signal and only add variance.

This rules out the most plausible explanation, that the deployed design was handicapped by using lagged rather than forecast weather. It was not.

### Forecast combination (`src/combination_experiment.py`)

Blending model and baseline, `w * model + (1 - w) * seasonal_naive`, with weights chosen on validation only.

| Zone | Seasonal naive | Model alone | Equal blend (w=0.5) | Validation-tuned w |
|---|---|---|---|---|
| Zone 1 | 2.86 | 3.26 | 2.83 | 2.86 (w=0.10) |
| Zone 2 | 4.26 | 4.53 | 4.13 | 4.18 (w=0.10) |
| Zone 3 | 8.73 | 24.96 | 15.39 | 10.36 (w=0.20) |

The untuned equal blend beat the validation-tuned weights on Zones 1 and 2, consistent with the well-documented forecast-combination puzzle where simple averages often outperform estimated weights. But see the significance test below before reading anything into these margins.

### Significance (`src/significance_test.py`)

Absolute percentage errors collapsed to one value per distinct target hour (effective n = 1,007, not 141,120 rows), then Diebold-Mariano with a Newey-West variance plus a 10,000-sample paired bootstrap.

Model alone versus seasonal naive:

| Zone | Difference | 95% CI | p | Verdict |
|---|---|---|---|---|
| Zone 1 | +0.412 pp worse | [+0.27, +0.55] | 0.0008 | Significantly worse |
| Zone 2 | +0.455 pp worse | [+0.25, +0.65] | 0.0238 | Significantly worse |
| Zone 3 | +14.63 pp worse | [+13.27, +16.00] | <0.0001 | Significantly worse |

Equal blend versus seasonal naive:

| Zone | Difference | 95% CI | p | Verdict |
|---|---|---|---|---|
| Zone 1 | -0.015 pp | [-0.093, +0.063] | 0.83 | Not significant |
| Zone 2 | -0.020 pp | [-0.127, +0.086] | 0.84 | Not significant |
| Zone 3 | +5.83 pp worse | [+5.17, +6.50] | <0.0001 | Significantly worse |

**The blend's apparent 0.03 and 0.13 point advantages are not statistically distinguishable from zero.** They must not be reported as improvements.

Note also why the row-level and collapsed figures differ. Zone 2's blend looked 0.13 pp better on raw rows but is 0.02 pp different once collapsed, because target hours near the edges of the test window are forecast from fewer origins and row-level averaging silently overweights the middle of the window. The collapsed figure is the correct one.

### Conclusion of the three experiments

> The model alone is significantly worse than a seasonal naive in all three zones. Blending it with the baseline brings Zones 1 and 2 to statistical parity but never beats it. On Zone 3 both model and blend are significantly worse.

Blending produced a real effect, moving a significantly worse forecast to parity, but not a win. No further configuration was tried, because continuing would amount to searching for a variant that happens to win on this particular test window.

## Corrections applied 2026-09

This report replaces an earlier version whose numbers were invalid. An audit found:

1. **`lag168` was anchored to the origin** (`power[origin - 168]`) instead of the target. The model received one frozen value from the origin's hour of day for all 168 predictions, so it could only learn an average weekly shape and never saw last week's value at the hour being predicted. Note the weather lag was already target-anchored; the power lag should have been too.
2. **The naive baseline used the same wrong index**, which made it a strawman scoring about 25% MAPE. It was in fact measuring persistence (24.72%), not a seasonal naive (2.86%). Every "improvement over naive" figure computed against it was meaningless.
3. **No embargo between splits.** Training rows had targets reaching 168 hours into the validation period, and validation rows into the test period.
4. **Sample size overstated about 140x** by reporting raw rows rather than distinct target hours.
5. A superseded script documented an MAE objective that none of the final models used.

The previous headline, "beats naive in 9 of 9 combinations by 47 to 85 percent," was entirely an artifact of defects 1 and 2. It has been withdrawn.

A symptom was visible and misread at the time: test error was flat at 3.60 to 3.62 percent from day 1 to day 7. Forecast error grows with horizon. Flat error indicated the model was not using recent information, and it was written up as a strength.

## Limitations

- The model does not beat a seasonal naive at this horizon. Reported as the result, not worked around.
- Zone 3 is unstable: 20.64% on test against an 8.73% baseline, and 22.10% and 18.11% on two of three walk-forward windows.
- Single year of data, so seasonal effects are observed once and not confirmed as recurring.
- Hyperparameters come from a 300-second AutoML budget per zone. A longer search might narrow the gap; it was not run.
- Weather features contribute modestly. The dominant signal is each series' own lags and calendar structure.
- Untested directions that could plausibly change the conclusion: multiple seasonal lags, holiday and special-day flags, per-horizon models, and probabilistic rather than point forecasts.

## Reproducing

```bash
python src/data_cleaning.py
python src/run_pipeline.py 300
python src/residual_variant.py
python src/make_figures.py
python src/validate_project.py
```

`validate_project.py` asserts feature observability, embargo integrity, baseline definitions, effective sample size, that saved models reproduce the recorded metrics, and that every figure referenced by a report exists. It currently reports 40 passed, 0 failed.
