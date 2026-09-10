# Tetouan City Load Forecasting: A Rigorous Evaluation Study

When does machine learning actually beat a classical baseline for electricity load forecasting, and when does it only appear to?

This project answers that question for the three urban distribution networks of Tetouan, Morocco, using a 168-hour horizon, five controlled experiments, formal significance testing, and a 40-assertion correctness suite. It ships with a Streamlit dashboard and a fully reproducible pipeline.

## What this project establishes

**1. A quantified boundary for where learned models help.** Holding the model, features and protocol constant and varying only the horizon, a gradient-boosted model's advantage over a seasonal naive exists at one hour ahead and disappears beyond six. At the standard 24-hour day-ahead horizon the baseline is already ahead.

**2. Why, with the obvious explanations ruled out.** An oracle experiment granting the model perfect future weather, information unavailable at forecast time, makes accuracy slightly *worse*. The limitation is not missing weather forecasts. For a weekly-periodic series, last week's value at the target hour is nearly independent of forecast distance, so extending the horizon penalises the model without penalising the baseline.

**3. That small apparent wins on this data are noise.** Blending model and baseline produced gains of 0.03 and 0.13 percentage points. Collapsing errors to one per distinct target hour, the honest effective sample of 1,007 rather than 141,120 rows, a Diebold-Mariano test returns p = 0.83 and 0.84. Not distinguishable from zero.

**4. How a plausible-looking result can be entirely an artifact.** An earlier version of this project reported beating the baseline in 9 of 9 windows by 47 to 85 percent. A self-audit found four defects that produced that number. All are documented below and now guarded by automated assertions.

## Results

Chronological train/validation/test split with a 168-hour embargo. Model family and hyperparameters chosen by FLAML AutoML on validation only, refit on train plus validation, scored once on test.

| Zone | Model | Test MAPE | Seasonal naive | Persistence |
|---|---|---|---|---|
| Zone 1 | Random Forest | 4.20% | 2.86% | 24.72% |
| Zone 2 | Random Forest | 4.81% | 4.26% | 28.60% |
| Zone 3 | XGBoost | 20.64% | 8.73% | 28.35% |

Significance, errors collapsed to one per target hour, Diebold-Mariano with Newey-West variance plus a 10,000-sample bootstrap:

| Comparison | Zone 1 | Zone 2 | Zone 3 |
|---|---|---|---|
| Model vs seasonal naive | +0.41pp, p=0.0008 | +0.46pp, p=0.024 | +14.6pp, p<0.0001 |
| Blend vs seasonal naive | -0.015pp, p=0.83 | -0.020pp, p=0.84 | +5.83pp, p<0.0001 |

**The finding: on these series a seasonal naive is the stronger forecaster at a 168-hour horizon.** The learned model is significantly behind it; blending the two reaches statistical parity but does not surpass it. The model does beat persistence by a wide margin everywhere, but persistence is the wrong benchmark for a weekly-periodic series, and mistaking one for the other is precisely what produced the original erroneous result.

Full methodology: [reports/05_modeling_report.md](reports/05_modeling_report.md).

## The five experiments

| Experiment | Question | Answer |
|---|---|---|
| [`protocol_experiment.py`](src/protocol_experiment.py) | How much of a reported score comes from evaluation choices? | A great deal. 10-minute-ahead prediction with a random split gives under 1% MAPE while trivial persistence already achieves 1.4%. Switching to a chronological split moves Zone 3 from 0.92% to 2.79%. |
| [`horizon_experiment.py`](src/horizon_experiment.py) | Is the 168-hour horizon the cause? | Partly. The model wins 2 of 15 horizon-zone combinations, both at one hour ahead. |
| [`ceiling_experiment.py`](src/ceiling_experiment.py) | Would better features close the gap? | No. Legal level-shift features change nothing; an oracle weather forecast makes it worse. |
| [`combination_experiment.py`](src/combination_experiment.py) | Does blending beat both? | Apparent small gains on two zones. |
| [`significance_test.py`](src/significance_test.py) | Are those gains real? | No. p = 0.83 and 0.84. |

### Horizon sensitivity

Model MAPE minus seasonal-naive MAPE, in percentage points. Negative means the model wins.

| Horizon | Zone 1 | Zone 2 | Zone 3 |
|---|---|---|---|
| 1 hour | +0.33 | **-0.23** | **-0.64** |
| 6 hours | +1.06 | +0.57 | +3.74 |
| 24 hours | +2.14 | +1.56 | +7.12 |
| 72 hours | +2.66 | +4.95 | +9.25 |
| 168 hours | +3.12 | +6.77 | +5.12 |

Persistence degrades from about 6% at one hour to about 25% at a week, exactly as expected. The seasonal naive barely moves at all: Zone 1 records 2.77, 2.78, 2.78, 2.82, 2.86 percent across the entire range. That asymmetry is the mechanism.

### Feature ceiling

| Zone | Seasonal naive | Base | + level shift | + oracle weather |
|---|---|---|---|---|
| Zone 1 | 2.86 | 3.43 | 3.44 | 3.49 |
| Zone 2 | 4.26 | 4.46 | 4.46 | 4.69 |
| Zone 3 | 8.73 | 25.31 | 23.41 | 25.24 |

The oracle column uses weather at the target hour, unavailable at forecast time. It is reported strictly as an upper bound and never as a claimable result.

## The self-audit

The first version of this project reported beating the naive baseline in 9 of 9 combinations by 47 to 85 percent. Four defects produced that number:

1. **The weekly lag feature was indexed off the forecast origin rather than the target.** The model received one frozen value from the origin's hour of day for all 168 predictions, so it never saw last week's value at the hour it was predicting and could only learn an average weekly shape.
2. **The naive baseline used the same wrong index**, turning it into a strawman. It was measuring persistence (24.72%) while being labelled a seasonal naive (2.86%).
3. **Splits were cut without an embargo**, so training targets reached 168 hours into the validation period and validation targets into the test period.
4. **Sample size was reported as 141,120 rows** when the effective size is about 1,007 distinct target hours, an overstatement of roughly 140x.

A symptom was visible throughout and was misread: test error was flat at 3.60 to 3.62 percent from day 1 to day 7. Forecast error grows with horizon, so flat error meant the model was ignoring recent information. It had been written up as a strength.

All four are fixed and each is now guarded by an automated assertion.

## Correctness suite

```bash
python src/validate_project.py     # 40 passed, 0 failed
```

Asserts data integrity against the pristine UCI source, feature observability at the forecast origin, embargo integrity across every split boundary, baseline definitions, effective sample size, that saved models reproduce their recorded metrics, that every figure referenced by a report exists, and that no stale labels survive anywhere in the repository.

## Dataset

[Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city), UCI Machine Learning Repository dataset 849. 52,416 rows at 10-minute resolution covering 2017. Three distribution networks (Quads, Smir, Boussafou substations, operator Amendis) plus five weather channels.

Source paper: A. Salam and A. El Hibaoui, "Comparison of Machine Learning Algorithms for the Power Consumption Prediction: Case Study of Tetouan city", IEEE IRSEC 2018.

This project began from a modified copy of the dataset in which the zones had been relabelled and the year changed. It was identified by cell-by-cell comparison against the UCI original and reverted. See [reports/01_data_cleaning_report.md](reports/01_data_cleaning_report.md).

## Running it

```bash
cd electricity-prediction
pip install -r app/requirements.txt

python src/data_cleaning.py            # UCI original to cleaned hourly data, report 01
python src/eda.py                      # report 02 and figures
python src/trend_seasonality.py        # report 03 workbook and figures
python src/outlier_detection.py        # report 04 and figures
python src/run_pipeline.py 300         # AutoML, training, walk-forward validation
python src/residual_variant.py         # direct versus residual target framing
python src/horizon_experiment.py       # horizon sensitivity
python src/ceiling_experiment.py       # feature ceiling, incl. oracle upper bound
python src/combination_experiment.py   # model plus baseline blending
python src/significance_test.py        # Diebold-Mariano and bootstrap
python src/make_deployment_forecast.py # full-year retrain and 7-day forecast
python src/make_figures.py             # modeling figures
python src/validate_project.py         # correctness suite

streamlit run app/streamlit_app.py     # dashboard at localhost:8501
```

## Repository layout

```
electricity-prediction/
├── app/                  Streamlit dashboard and requirements
├── data/
│   ├── raw/              UCI original CSV
│   └── processed/        cleaned data, detected outliers, deployment forecast
├── models/               saved models, zone_1 / zone_2 / zone_3 variants
├── notebooks/            01_modeling.ipynb, narrative walkthrough
├── paper/                research paper drafts
├── reports/              4 markdown reports, 1 workbook, 7 result JSONs, figures/
└── src/
    ├── forecasting.py            features, splits, baselines, metrics (single source of truth)
    ├── data_cleaning.py          cleaning pipeline
    ├── eda.py                    exploratory analysis
    ├── trend_seasonality.py      trend and seasonality decomposition
    ├── outlier_detection.py      MSTL residual outlier detection
    ├── run_pipeline.py           AutoML, training, walk-forward validation
    ├── residual_variant.py       direct versus residual target comparison
    ├── protocol_experiment.py    evaluation-protocol sensitivity
    ├── horizon_experiment.py     horizon sensitivity
    ├── ceiling_experiment.py     feature ceiling, incl. oracle upper bound
    ├── combination_experiment.py model plus baseline blending
    ├── significance_test.py      Diebold-Mariano and paired bootstrap
    ├── make_deployment_forecast.py  full-year retrain and 7-day forecast
    ├── make_figures.py           modeling figures
    └── validate_project.py       correctness assertions
```

## Practical implication

For an operator forecasting these networks a week ahead, the recommendation is to use the seasonal naive. It is more accurate, costs nothing to run, requires no weather feed, and needs no retraining. A learned model is worth deploying at short horizons, where this study measured its advantage, or on non-stationary series once the extrapolation problem visible in Zone 3 is addressed.

Knowing which of those situations you are in is the point of the exercise.
