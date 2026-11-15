# Trained-model comparison

Completed 46 cases. Full run: True. Excluded 14 cases with insufficient earlier training data.

Ridge and LightGBM predict residuals from weekly naive using load/calendar features. Two settings per family are chosen on earlier validation MAE. Each final model is refitted using mature historical targets only, then remains fixed over the test period. Training uses daily origins; all methods are evaluated on the frozen hourly test origins. Q1 remains excluded rather than moving split boundaries. Every prediction and the training/validation boundaries are saved.

These are descriptive errors, not significance tests. This follow-up was designed after seeing Chronos results. It is a compact trained-model reference, not an exhaustive search. Do not compare raw MAE across different units.

## Seven-day MAE

| Period | Series | Ridge | LightGBM | Chronos | Selected seasonal baseline |
|---|---|---:|---:|---:|---:|
| Q2 | portugal/T1 | 22.79 | 12.77 | 6.588 | 8.512 |
| Q2 | portugal/T2 | 20.9 | 6.896 | 6.456 | 7.806 |
| Q2 | portugal/T3 | 0.04655 | 0 | 0.4422 | 0 |
| Q2 | tetouan/zone_1 | 6005 | 1954 | 1795 | 1741 |
| Q2 | tetouan/zone_2 | 1773 | 1335 | 1306 | 1331 |
| Q2 | tetouan/zone_3 | 3187 | 1151 | 1059 | 1022 |
| Q2 | victoria/regional_demand | 0.7297 | 0.4641 | 0.1584 | 0.1858 |
| Q3 | household/sceaux_household | 0.4347 | 0.4205 | 0.4039 | 0.4274 |
| Q3 | portugal/T1 | 17.58 | 23.33 | 13.12 | 14.59 |
| Q3 | portugal/T2 | 16.42 | 11 | 9.071 | 10.3 |
| Q3 | portugal/T3 | 0.03588 | 0 | 0.4422 | 0 |
| Q3 | tetouan/zone_1 | 3527 | 1916 | 1645 | 1765 |
| Q3 | tetouan/zone_2 | 2669 | 1855 | 1686 | 1969 |
| Q3 | tetouan/zone_3 | 2737 | 2252 | 1655 | 1664 |
| Q3 | victoria/regional_demand | 0.7926 | 0.211 | 0.1942 | 0.2293 |
| Q4 | household/sceaux_household | 0.4678 | 0.4368 | 0.4614 | 0.4937 |
| Q4 | portugal/T1 | 3.111 | 4.337 | 2.29 | 2.573 |
| Q4 | portugal/T2 | 5.888 | 4.011 | 3.583 | 4.081 |
| Q4 | portugal/T3 | 0.2258 | 0.2115 | 0.6427 | 0.2115 |
| Q4 | tetouan/zone_1 | 1318 | 930.2 | 706.6 | 865.8 |
| Q4 | tetouan/zone_2 | 3275 | 1130 | 788.4 | 938.3 |
| Q4 | tetouan/zone_3 | 1208 | 1935 | 681.1 | 607.4 |
| Q4 | victoria/regional_demand | 0.3535 | 0.8025 | 0.2786 | 0.3171 |

Reproduce: `python src/trained_comparison.py`. Existing outputs resume only with the same code, protocol, inputs and package versions.
