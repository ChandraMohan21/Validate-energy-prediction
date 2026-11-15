# Seasonal Baselines and Time-Series Foundation Models for Electricity Load

How much of a foundation model's reported advantage depends on the seasonal baseline it is compared against? This project keeps the model forecasts fixed, changes only the reference baseline, and measures how far the conclusion moves.

## Main study: seasonal-baseline sensitivity

### Setup

- **Data:** 8 hourly electricity-load series from 4 public datasets: Tetouan city zones 1-3 (UCI), Portugal meters T1-T3 (Monash / UCI ElectricityLoadDiagrams), Victoria regional demand (Monash / AEMO), and one French household (UCI).
- **Cases:** 60 frozen test cases, 30 day-ahead (24 h) and 30 week-ahead (168 h). They come from four calendar windows per series, each with a 42-day validation block and a 42-day test block. The household series has complete data only in two windows.
- **Foundation models:** Chronos-Bolt and Chronos-2, zero-shot and target-only, with context capped at 2,048 hours for both so the comparison is matched.
- **Seasonal baselines:** previous day, previous week, a weekday hybrid (weekly values for Saturday to Monday, daily otherwise), and a daily-weekly blend. For each case the rule and blend weight are chosen on the validation block, never on test.
- **Trained comparators:** Ridge and LightGBM fitted per series on history available before each test period.
- **Uncertainty:** paired circular-block bootstrap with 5,000 resamples at 168-hour and 336-hour blocks. A result counts as clear only when both block lengths agree.

### Results

Cases where the foundation model has clear paired support over the reference baseline:

| Horizon | Model | vs fixed weekly rule | vs validation-selected rule |
|---|---|---:|---:|
| Week-ahead (168 h) | Chronos-Bolt | 21 / 30 | 14 / 30 |
| Week-ahead (168 h) | Chronos-2 | 18 / 30 | 15 / 30 |
| Day-ahead (24 h) | Chronos-Bolt | 23 / 30 | 22 / 30 |
| Day-ahead (24 h) | Chronos-2 | 23 / 30 | 24 / 30 |

- Replacing the fixed weekly rule with the validation-selected rule removes 7 of Bolt's week-ahead wins and 3 of Chronos-2's. The day-ahead conclusion barely moves.
- Chronos-2 has lower test MAE than Chronos-Bolt in 29 of 30 day-ahead and 24 of 30 week-ahead cases (clear paired support in 15 and 12).
- The validation-selected Ridge or LightGBM model beats the selected seasonal rule in only 7 of 30 day-ahead and 4 of 30 week-ahead cases. Its wins over either Chronos model all come from one nearly week-repeating meter (Portugal T3).

### Limits

The eight series are related, not independent replications. The four-window follow-up was designed after the pilot results were seen, so it is not preregistered. Intervals are not corrected for multiple comparisons. These public datasets may overlap the foundation models' pretraining data.

## Earlier study: Tetouan city load at a 168-hour horizon

The project started as a single-city study. Zero-shot Chronos-Bolt beat the same-hour-last-week baseline in all three Tetouan zones (MAPE 2.45% vs 2.86%, 3.36% vs 4.26%, 5.17% vs 8.73%), while AutoML-selected Random Forest and XGBoost models lost to that baseline. A self-audit found four defects behind an earlier, too-good result; all are documented and guarded by tests. Full write-up: [reports/00_tetouan_study.md](reports/00_tetouan_study.md) and [reports/05_modeling_report.md](reports/05_modeling_report.md).

## Repository map

| Path | Contents |
|---|---|
| `src/baseline_pilot.py` | Seasonal baselines and validation-based rule selection |
| `src/prepare_expanded_chronos.py` | Four calendar windows and the 60 frozen cases |
| `src/chronos2_study.py` | Chronos-2 vs Chronos-Bolt comparison and paired intervals |
| `src/trained_comparison_v3.py` | Ridge and LightGBM comparators |
| `src/chronos_uncertainty.py` | Paired circular-block bootstrap |
| `notebooks/04-06_*.ipynb` | Colab notebooks for Chronos inference (T4 GPU) |
| `reports/` | Protocols, results JSON and a README per experiment |
| `tests/` | Unit tests for each experiment |
| `app/streamlit_app.py` | Dashboard for the Tetouan study |

## Reproduce

```bash
pip install pandas numpy
python src/baseline_pilot.py --download          # seasonal baselines
python src/chronos2_study.py compare --archive chronos2_results.zip
python src/trained_comparison_v3.py analyze
python -m unittest discover -s tests
```

Chronos inference runs in Google Colab with the notebooks in `notebooks/`. The returned result archives are included in this repository.

## Data sources

- Tetouan: UCI Power Consumption of Tetouan City (CC BY 4.0)
- Portugal: Monash Electricity Hourly Dataset v3, derived from UCI ElectricityLoadDiagrams20112014 (CC BY 4.0)
- Victoria: Monash Elecdemand Dataset v2, from fpp2 / AEMO
- Household: UCI Individual Household Electric Power Consumption (CC BY 4.0)
