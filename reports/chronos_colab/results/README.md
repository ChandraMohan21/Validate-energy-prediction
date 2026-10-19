# Chronos pilot results: first run

Run on Google Colab (Tesla T4) on 19 October 2026 with `notebooks/03_chronos_pilot_colab.ipynb`. Model `amazon/chronos-bolt-base` (revision recorded in `run_metadata.json`), zero-shot, median forecast, **336-hour context**. Forecasts use only data up to each origin, on the frozen test origins of the seasonal baseline pilot (`reports/baseline_pilot/`).

Files: `comparison.csv` (16 cases x 5 baselines), `run_metadata.json`, `complete.json`, and `predictions/*.npz` holding actuals, origins, the four baseline forecasts and the Chronos forecast for paired follow-up tests.

## Summary

Eight series (Tetouan zones 1 to 3; three Portuguese meters; Victoria regional demand; one Sceaux household), 24- and 168-hour horizons, 16 cases. Gain is the percentage reduction in MAE of Chronos relative to each baseline.

| Baseline | Chronos lower MAE | Median gain |
|---|---|---|
| Daily seasonal naive | 15 of 16 | 33.2% |
| Weekly seasonal naive | 14 of 16 | 19.0% |
| Validation-selected baseline | 13 of 16 | 12.7% |

Cases where the conclusion differs between the weekly and the validation-selected baseline:

| Case | Gain vs weekly | Gain vs selected |
|---|---|---|
| Household, 24 h | 18.5% | 6.2% |
| Household, 168 h | 15.3% | 6.5% |
| Tetouan Zone 3, 168 h | 26.1% | -12.1% |

## Cautions

- **The 336-hour context handicaps Chronos.** The earlier Tetouan run used 2048 hours. For Tetouan Zone 3 at 168 hours, Chronos MAE is 681 with 336 hours and 544 with 2048 hours; with 2048 hours Chronos beats the validation-selected baseline by 10.4%. The Zone 3 reversal above may be an artefact of the short context. A 2048-hour rerun is needed before drawing conclusions.
- **Portugal T3 is a near-constant integer meter:** 75% of 2014 hours read exactly 8.0 and the year has only 12 distinct values. The weekly naive is almost exact there and Chronos loses by 200% to 360%. The series was selected by a fixed rule before scoring and is kept.
- **Descriptive only.** No significance test has been run. The eight series are related (three Tetouan zones, three meters from one archive) and are not independent replications. No threshold for a "meaningful" gain was fixed before this run.
- Familiar public datasets may overlap Chronos pretraining. No contamination-free claim is made.

## Data licences

All sources are Creative Commons Attribution 4.0: UCI Tetouan City (dataset 849), UCI Individual Household Electric Power Consumption (dataset 235), and the Monash Forecasting Archive records for Electricity Hourly (Zenodo 4656140, version 3, derived from UCI ElectricityLoadDiagrams20112014) and Elecdemand (Zenodo 4656069, version 2; Godahewa, Bergmeir, Webb, Hyndman, Montero-Manso). The Elecdemand licence was confirmed on its Zenodo page on 19 October 2026.
