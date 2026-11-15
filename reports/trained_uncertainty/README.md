# Uncertainty: Chronos versus trained models

Exploratory paired circular-block bootstrap: 5,000 resamples using 168- and 336-hour blocks. All saved predictions are held fixed. Positive difference means lower Chronos MAE. Calendar gaps remain missing. No new model training is performed. No multiplicity adjustment is applied, and short windows provide few long blocks. These results are conditional sensitivity evidence, not universal model rankings.

Q1 is excluded under the original frozen split. Household periods receive limited-data labels. The selected trained family was chosen by validation MAE before refitting, never by test error; it duplicates one family rather than providing an independent comparison.

## Counts

| Horizon | Comparator | Assessment | Count |
|---|---|---|---:|
| 24 | lightgbm | chronos_supported_both_lengths | 17 |
| 24 | lightgbm | limited_data | 2 |
| 24 | lightgbm | trained_supported_both_lengths | 3 |
| 24 | lightgbm | unclear_or_block_sensitive | 1 |
| 24 | ridge | chronos_supported_both_lengths | 18 |
| 24 | ridge | limited_data | 2 |
| 24 | ridge | trained_supported_both_lengths | 3 |
| 24 | selected_trained | chronos_supported_both_lengths | 17 |
| 24 | selected_trained | limited_data | 2 |
| 24 | selected_trained | trained_supported_both_lengths | 3 |
| 24 | selected_trained | unclear_or_block_sensitive | 1 |
| 168 | lightgbm | chronos_supported_both_lengths | 11 |
| 168 | lightgbm | limited_data | 2 |
| 168 | lightgbm | trained_supported_both_lengths | 3 |
| 168 | lightgbm | unclear_or_block_sensitive | 7 |
| 168 | ridge | chronos_supported_both_lengths | 18 |
| 168 | ridge | limited_data | 2 |
| 168 | ridge | trained_supported_both_lengths | 3 |
| 168 | selected_trained | chronos_supported_both_lengths | 14 |
| 168 | selected_trained | limited_data | 2 |
| 168 | selected_trained | trained_supported_both_lengths | 3 |
| 168 | selected_trained | unclear_or_block_sensitive | 4 |

## Seven-day comparisons with LightGBM

95% intervals below are gain percentages at the primary 168-hour block length. Assessment considers both lengths. Positive gain favors Chronos.

| Period | Series | Observed gain | Approximate gain interval | Assessment |
|---|---|---:|---|---|
| Q2 | tetouan/zone_1 | +8.12% | [+2.15%, +13.95%] | chronos_supported_both_lengths |
| Q2 | tetouan/zone_2 | +2.24% | [-2.77%, +6.30%] | unclear_or_block_sensitive |
| Q2 | tetouan/zone_3 | +7.96% | [-0.46%, +14.39%] | unclear_or_block_sensitive |
| Q2 | portugal/T1 | +48.42% | [+39.05%, +58.58%] | chronos_supported_both_lengths |
| Q2 | portugal/T2 | +6.37% | [-0.51%, +13.18%] | unclear_or_block_sensitive |
| Q2 | portugal/T3 | undefined (zero comparator MAE) | undefined | trained_supported_both_lengths |
| Q2 | victoria/regional_demand | +65.86% | [+50.95%, +75.68%] | chronos_supported_both_lengths |
| Q3 | tetouan/zone_1 | +14.15% | [-6.65%, +24.67%] | unclear_or_block_sensitive |
| Q3 | tetouan/zone_2 | +9.12% | [-4.26%, +19.12%] | unclear_or_block_sensitive |
| Q3 | tetouan/zone_3 | +26.54% | [+11.71%, +45.91%] | chronos_supported_both_lengths |
| Q3 | portugal/T1 | +43.76% | [+5.23%, +74.54%] | chronos_supported_both_lengths |
| Q3 | portugal/T2 | +17.56% | [-9.76%, +40.14%] | unclear_or_block_sensitive |
| Q3 | portugal/T3 | undefined (zero comparator MAE) | undefined | trained_supported_both_lengths |
| Q3 | victoria/regional_demand | +7.94% | [-2.64%, +16.43%] | unclear_or_block_sensitive |
| Q3 | household/sceaux_household | +3.95% | [+1.50%, +6.38%] | limited_data |
| Q4 | tetouan/zone_1 | +24.03% | [+11.93%, +37.44%] | chronos_supported_both_lengths |
| Q4 | tetouan/zone_2 | +30.23% | [+21.08%, +37.35%] | chronos_supported_both_lengths |
| Q4 | tetouan/zone_3 | +64.79% | [+51.37%, +77.08%] | chronos_supported_both_lengths |
| Q4 | portugal/T1 | +47.21% | [+2.37%, +69.42%] | chronos_supported_both_lengths |
| Q4 | portugal/T2 | +10.67% | [+4.17%, +17.39%] | chronos_supported_both_lengths |
| Q4 | portugal/T3 | -203.84% | undefined | trained_supported_both_lengths |
| Q4 | victoria/regional_demand | +65.28% | [+28.20%, +76.79%] | chronos_supported_both_lengths |
| Q4 | household/sceaux_household | -5.64% | [-9.59%, -1.90%] | limited_data |

Reproduce: `python src/trained_uncertainty.py`. Full intervals for both lengths, all comparators and all 46 cases are in results.json and summary.csv.

Method reference: [arch time-series bootstrap documentation](https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html).
