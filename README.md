# Tetouan City Load Forecasting: A Rigorous Evaluation Study

When does machine learning actually beat a classical baseline for electricity load forecasting, and when does it only appear to?

This project answers that question for the three urban distribution networks of Tetouan, Morocco, at a 168-hour horizon. It compares models trained on the series, a seasonal naive baseline, and a zero-shot time series foundation model, using six controlled experiments, formal significance testing, and a 61-assertion correctness suite. It ships with a Streamlit dashboard and a fully reproducible pipeline.

## What this project establishes

**1. A zero-shot foundation model beats the seasonal naive where trained models could not.** Chronos-Bolt, never trained on this data, beats "same hour last week" in all three zones: 2.45 against 2.86, 3.36 against 4.26, and 5.17 against 8.73 percent MAPE. Every difference is significant on the honest effective sample, with p = 0.001, 0.004 and below 0.0001. It holds across the walk-forward check as well: Chronos wins all 9 zone-window combinations, every one significant, while the trained models win 1 of 9.

**2. Tuned models trained on the series lose to that same baseline.** The Random Forest and XGBoost models chosen by AutoML are significantly behind the seasonal naive in every zone. Holding model, features and protocol fixed and varying only the horizon, their advantage exists at one hour ahead and disappears beyond six.

**3. Why the trained models fall short, with the obvious explanation ruled out.** An oracle experiment granting perfect future weather, information unavailable at forecast time, makes accuracy slightly *worse*. The limitation is not missing weather forecasts.

**4. That small apparent wins on this data are noise.** Blending a trained model with the baseline produced gains of 0.03 and 0.13 percentage points. Collapsing errors to one per distinct target hour, the honest effective sample of 1,007 rather than 141,120 rows, a Diebold-Mariano test returns p = 0.83 and 0.84.

**5. How a plausible-looking result can be entirely an artifact.** An earlier version of this project reported beating the baseline in 9 of 9 windows by 47 to 85 percent. A self-audit found four defects that produced that number. All are documented below and now guarded by automated assertions.

## Results

Chronological train/validation/test split with a 168-hour embargo. Trained models were chosen by FLAML AutoML on validation only, refit on train plus validation, and scored once on test. Chronos is applied zero-shot to the same test origins, seeing only history up to each origin.

| Zone | Trained model | Trained MAPE | Seasonal naive | Chronos-Bolt, zero-shot | Persistence |
|---|---|---|---|---|---|
| Zone 1 | Random Forest | 4.20% | 2.86% | **2.45%** | 24.72% |
| Zone 2 | Random Forest | 4.81% | 4.26% | **3.36%** | 28.60% |
| Zone 3 | XGBoost | 20.64% | 8.73% | **5.17%** | 28.35% |

Significance, with errors collapsed to one per target hour, using Diebold-Mariano with Newey-West variance plus a 10,000-sample bootstrap. Negative means more accurate than the seasonal naive.

| Comparison | Zone 1 | Zone 2 | Zone 3 |
|---|---|---|---|
| Chronos-Bolt vs seasonal naive | -0.40pp, p=0.0010 | -0.72pp, p=0.0036 | -4.21pp, p<0.0001 |
| Trained LightGBM vs seasonal naive | +0.41pp, p=0.0008 | +0.46pp, p=0.024 | +14.6pp, p<0.0001 |
| Blend vs seasonal naive | -0.015pp, p=0.83 | -0.020pp, p=0.84 | +5.83pp, p<0.0001 |

**The finding: at a 168-hour horizon on these series, the seasonal naive is stronger than every model trained on the series, and a zero-shot foundation model is stronger than the seasonal naive.** The trained models do beat persistence by a wide margin, but persistence is the wrong benchmark for a weekly-periodic series. Mistaking one for the other is precisely what produced the original erroneous result.

Full methodology: [reports/05_modeling_report.md](reports/05_modeling_report.md).

## The six experiments

| Experiment | Question | Answer |
|---|---|---|
| [`protocol_experiment.py`](src/protocol_experiment.py) | How much of a reported score comes from evaluation choices? | A great deal. 10-minute-ahead prediction with a random split gives under 1% MAPE while trivial persistence already achieves 1.4%. Switching to a chronological split moves Zone 3 from 0.92% to 2.79%. |
| [`horizon_experiment.py`](src/horizon_experiment.py) | Is the 168-hour horizon the cause? | Partly. The trained model wins 2 of 15 horizon-zone combinations, both at one hour ahead. |
| [`ceiling_experiment.py`](src/ceiling_experiment.py) | Would better features close the gap? | No. Legal level-shift features change nothing, and an oracle weather forecast makes it worse. |
| [`combination_experiment.py`](src/combination_experiment.py) | Does blending beat both? | Apparent small gains on two zones. |
| [`significance_test.py`](src/significance_test.py) | Are those gains real? | No. p = 0.83 and 0.84. |
| [`foundation_baseline.py`](src/foundation_baseline.py) | Can a pretrained model with no training beat the baseline? | Yes. In all three zones on the test window, and in 9 of 9 walk-forward zone-windows, every one significant. |

### Zero-shot foundation model by day ahead

Chronos-Bolt MAPE minus seasonal-naive MAPE, in percentage points. Negative means Chronos wins.

| Day ahead | Zone 1 | Zone 2 | Zone 3 |
|---|---|---|---|
| 1 | -0.81 | -1.67 | -6.98 |
| 3 | -0.40 | -0.89 | -4.30 |
| 5 | -0.34 | -0.77 | -2.66 |
| 7 | -0.21 | -0.50 | -0.21 |

The advantage is largest one day ahead and narrows toward a week, where the forecast converges on the weekly pattern the seasonal naive copies. On the test window it stays ahead at day 7 in every zone. Zone 3 gains most. Its level moves over the test window, which a copy of last week cannot follow and a tree ensemble cannot extrapolate.

Across the same three walk-forward windows the trained models were checked on, Chronos MAPE against seasonal-naive MAPE:

| Zone | W1, 18 Nov to 30 Dec | W2, 7 Oct to 18 Nov | W3, 26 Aug to 7 Oct |
|---|---|---|---|
| Zone 1 | **2.45** vs 2.86 | **3.41** vs 4.39 | **4.39** vs 5.64 |
| Zone 2 | **3.36** vs 4.26 | **4.55** vs 5.06 | **7.32** vs 9.11 |
| Zone 3 | **5.17** vs 8.73 | **8.30** vs 10.08 | **7.13** vs 10.92 |

Chronos wins all nine, each with a Diebold-Mariano p of 0.0036 or below. The trained models won one of nine, in the window they were tuned on. Chronos has no tuning step, so every window is out of sample for it.

Scope of this result: one year of data, one foundation model at one size, and a contamination check against the published Chronos pretraining dataset list. In Zone 2 the advantage fades by days 6 and 7 in two of the three windows.

### Horizon sensitivity

Trained model MAPE minus seasonal-naive MAPE, in percentage points. Negative means the model wins.

| Horizon | Zone 1 | Zone 2 | Zone 3 |
|---|---|---|---|
| 1 hour | +0.33 | **-0.23** | **-0.64** |
| 6 hours | +1.06 | +0.57 | +3.74 |
| 24 hours | +2.14 | +1.56 | +7.12 |
| 72 hours | +2.66 | +4.95 | +9.25 |
| 168 hours | +3.12 | +6.77 | +5.12 |

Persistence degrades from about 6% at one hour to about 25% at a week, exactly as expected. The seasonal naive barely moves at all: Zone 1 records 2.77, 2.78, 2.78, 2.82 and 2.86 percent across the entire range. That asymmetry is the mechanism.

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
python src/validate_project.py     # 61 passed, 0 failed
```

Asserts data integrity against the pristine UCI source, feature observability at the forecast origin, embargo integrity across every split boundary, baseline definitions, effective sample size, that saved models reproduce their recorded metrics, that the foundation model runs used the canonical test origins and the recorded walk-forward windows with a context window ending at each origin, that every figure referenced by a report exists, and that no stale labels survive anywhere in the repository.

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
python src/foundation_baseline.py      # Chronos-Bolt zero-shot, needs torch and several GB free
python src/foundation_baseline.py --walkforward  # same, across the three walk-forward windows
python src/make_deployment_forecast.py # full-year retrain and 7-day forecast
python src/make_figures.py             # modeling figures
python src/validate_project.py         # correctness suite

streamlit run app/streamlit_app.py     # dashboard at localhost:8501
```

The foundation model step needs PyTorch and `chronos-forecasting`. Without a few GB of free memory, open [notebooks/02_foundation_baseline_colab.ipynb](notebooks/02_foundation_baseline_colab.ipynb) in Google Colab instead. It clones this repository and runs the same script on a free GPU.

## Repository layout

```
electricity-prediction/
├── app/                  Streamlit dashboard and requirements
├── data/
│   ├── raw/              UCI original CSV
│   └── processed/        cleaned data, detected outliers, deployment forecast
├── models/               saved models, zone_1 / zone_2 / zone_3 variants
├── notebooks/            01_modeling.ipynb walkthrough, 02_foundation_baseline_colab.ipynb
├── paper/                research paper drafts
├── reports/              4 markdown reports, 1 workbook, 9 result JSONs, figures/
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
    ├── foundation_baseline.py    Chronos-Bolt zero-shot baseline
    ├── make_deployment_forecast.py  full-year retrain and 7-day forecast
    ├── make_figures.py           modeling figures
    └── validate_project.py       correctness assertions
```

## Practical implication

For an operator forecasting these networks a week ahead, a zero-shot foundation model was the most accurate option tested. It needs no training, no feature engineering and no weather feed. The seasonal naive is the strongest cheap fallback and beat every model trained here. Training a bespoke tree ensemble does not pay at this horizon on these series.

Its advantage held in every walk-forward window tested. The open question before operational use is whether it holds in other years and on other networks. Knowing which result generalises is the point of the exercise.
