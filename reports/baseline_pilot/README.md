# Seasonal baseline pilot

A baseline-only feasibility experiment. No foundation model or trained model was evaluated in this pilot.

## What was done

Forecasts were frozen at each hourly origin. Daily repeats the last observed day; weekly repeats the last observed week. The weekday hybrid uses weekly values for Saturday through Monday and daily recursion otherwise. The blend is alpha * daily + (1-alpha) * weekly, with alpha selected from a fixed grid using validation MAE only. The final rule is also selected on validation MAE, separately for each series and horizon.

The 168-hour task is primary; 24 hours is a secondary comparison. Both use identical origin sets and a 168-hour embargo. Validation and test losses count every origin/horizon pair equally. Repeated targets are not independent observations.

## Seven-day test results

MAE stays in the series' own units; compare methods within a row, not raw MAE between datasets. Positive gain means the validation-selected baseline beats weekly; negative gain means it loses on test.

| Dataset / series | Daily MAE | Weekly MAE | Hybrid MAE | Blend MAE | Validation choice | Daily weight | Gain vs weekly | Test origins |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| tetouan / zone_1 | 1150.0342 | 813.9553 | 865.5502 | 865.8478 | blend | 0.55 | -6.38% | 840 |
| tetouan / zone_2 | 1391.7113 | 974.6431 | 887.7261 | 938.3310 | blend | 0.25 | +3.73% | 840 |
| tetouan / zone_3 | 566.1666 | 921.0499 | 858.8483 | 607.3703 | blend | 0.75 | +34.06% | 840 |
| portugal / T1 | 2.8913 | 2.8230 | 2.9220 | 2.5734 | blend | 0.50 | +8.84% | 840 |
| portugal / T2 | 5.7089 | 4.2074 | 4.7810 | 4.0812 | blend | 0.30 | +3.00% | 840 |
| portugal / T3 | 0.8981 | 0.2115 | 0.4720 | 0.2115 | weekly | 0.00 | +0.00% | 840 |
| victoria / regional_demand | 0.4391 | 0.3255 | 0.3472 | 0.3171 | blend | 0.10 | +2.59% | 840 |
| household / sceaux_household | 0.5960 | 0.5448 | 0.5720 | 0.4937 | blend | 0.50 | +9.39% | 244 |

Descriptively, validation selection beats fixed weekly on 6 series, ties on 1, and loses on 1. These are correlated series, not independent significance tests.

## Tetouan in percentage-error terms

The comparison below uses the same 840 test origins as the existing seven-day study. Selection still uses validation MAE, not test MAPE. A different selection metric may select a different rule and should be declared before a future study.

| Zone | Daily MAPE | Weekly MAPE | Hybrid MAPE | Blend MAPE | Validation choice |
|---|---:|---:|---:|---:|---|
| zone_1 | 4.05% | 2.86% | 3.05% | 3.05% | blend |
| zone_2 | 6.19% | 4.26% | 3.96% | 4.12% | blend |
| zone_3 | 5.39% | 8.73% | 8.32% | 5.79% | blend |

## Sources and preparation

The year and meter-selection rules were fixed in protocol.json before scoring. Portugal uses the first three archive records, not the best-performing meters. Both Monash TSF archives start one second after their native time grid; this constant one-second offset is removed before hourly alignment, without shifting observation order. Victoria half-hourly power is averaged to hourly power; household minute power is averaged only when all 60 samples exist. Missing hours are retained as missing, with affected origins excluded. Tetouan uses the existing cleaned UCI series. Calendar timestamps otherwise follow the source, without claiming UTC alignment. Portugal retains the source's DST conventions and archive scaling.

- household: [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption). CC BY 4.0, UCI publisher. Unit: kW.
- portugal: [Monash Electricity Hourly Dataset, version 3, derived from UCI ElectricityLoadDiagrams20112014](https://zenodo.org/records/4656140). Original UCI CC BY 4.0; retain Monash attribution and version. Unit: Monash archive units; original source reports kW; hourly scaling not independently recalibrated.
- victoria: [Monash Elecdemand Dataset, version 2, from fpp2/AEMO](https://zenodo.org/records/4656069). Public Monash/fpp2 redistribution; preserve source attribution; Zenodo license field not stated in fetched page. Unit: GW.

## What this can and cannot tell us

This pilot measures baseline sensitivity, not foundation-model advantage, statistical significance, or novelty. Only one prespecified year per dataset and a few series were used. Related zones/meters are not independent datasets; do not interpret the eight series as eight independent replications. Validation selection can fail under changing patterns, and all such failures remain reported. These familiar public datasets may overlap foundation-model pretraining; no contamination-free claim is made.

The weekday hybrid is an origin-frozen adaptation of established day-ahead price practice, not a newly invented method. The daily/weekly blend is an elementary baseline inspired by reference-combination literature, not Yang's exact climatology/persistence method.

MAPE is undefined when a target is zero, so those series/horizons have null MAPE rather than an arbitrary epsilon. scores.csv also includes MAE, RMSE, WAPE and zero counts. results.json contains the full selection and split metadata. predictions/*.npz retain actuals, origins and all four test forecasts for paired follow-up analysis.

## What happens next

Use these frozen series and origins to evaluate Chronos in Colab, saving every prediction. Then evaluate clearly labelled trained models on the same data, select their settings before test, and compare paired errors with dependence-aware uncertainty. Recheck the novelty of a specific empirical finding before expanding the paper. None of those model runs was performed by this script.

## Reproduce

Run `python src/baseline_pilot.py --download` from the repository. Cached inputs permit `python src/baseline_pilot.py`. Only pandas, numpy and the Python standard library are required. Run `python -m unittest discover -s tests -p test_baseline_pilot.py` for protocol tests.
