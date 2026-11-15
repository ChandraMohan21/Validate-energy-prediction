# Version 2: hourly training origins

Completed 46 cases. Full run: True. Excluded 14 Q1 cases under the frozen split.

Every eligible hourly origin is used for fitting and refitting. All included cases cover all 24 origin hours. Model families, features, hyperparameter grids and validation/test origins are unchanged from v1. Settings are chosen on validation MAE only, then refitted before the first test origin. Both versions and every prediction are preserved.

This is a diagnostic correction after inspecting v1 test results. Test periods are not untouched confirmation data. V2 corrects hour coverage but retains the limited grids and partial-year/extrapolation limitations. No v2 uncertainty calculations are performed by the training script. Do not reuse the v1 uncertainty intervals for v2.

## Seven-day MAE

| Period | Series | Ridge v1 | Ridge v2 | LightGBM v1 | LightGBM v2 | Chronos | Seasonal choice |
|---|---|---:|---:|---:|---:|---:|---:|
| Q2 | portugal/T1 | 22.79 | 23.55 | 12.77 | 13.53 | 6.588 | 8.512 |
| Q2 | portugal/T2 | 20.9 | 22.49 | 6.896 | 6.91 | 6.456 | 7.806 |
| Q2 | portugal/T3 | 0.04655 | 0.05109 | 0 | 0 | 0.4422 | 0 |
| Q2 | tetouan/zone_1 | 6005 | 4937 | 1954 | 1834 | 1795 | 1741 |
| Q2 | tetouan/zone_2 | 1773 | 1380 | 1335 | 1336 | 1306 | 1331 |
| Q2 | tetouan/zone_3 | 3187 | 1313 | 1151 | 1150 | 1059 | 1022 |
| Q2 | victoria/regional_demand | 0.7297 | 0.457 | 0.4641 | 0.3177 | 0.1584 | 0.1858 |
| Q3 | household/sceaux_household | 0.4347 | 0.4813 | 0.4205 | 0.4252 | 0.4039 | 0.4274 |
| Q3 | portugal/T1 | 17.58 | 15.73 | 23.33 | 23.66 | 13.12 | 14.59 |
| Q3 | portugal/T2 | 16.42 | 17.15 | 11 | 10.17 | 9.071 | 10.3 |
| Q3 | portugal/T3 | 0.03588 | 0.03606 | 0 | 0 | 0.4422 | 0 |
| Q3 | tetouan/zone_1 | 3527 | 3184 | 1916 | 1836 | 1645 | 1765 |
| Q3 | tetouan/zone_2 | 2669 | 3368 | 1855 | 1766 | 1686 | 1969 |
| Q3 | tetouan/zone_3 | 2737 | 4319 | 2252 | 2153 | 1655 | 1664 |
| Q3 | victoria/regional_demand | 0.7926 | 0.7897 | 0.211 | 0.2015 | 0.1942 | 0.2293 |
| Q4 | household/sceaux_household | 0.4678 | 0.464 | 0.4368 | 0.4177 | 0.4614 | 0.4937 |
| Q4 | portugal/T1 | 3.111 | 3.018 | 4.337 | 3.87 | 2.29 | 2.573 |
| Q4 | portugal/T2 | 5.888 | 4.755 | 4.011 | 4.046 | 3.583 | 4.081 |
| Q4 | portugal/T3 | 0.2258 | 0.2258 | 0.2115 | 0.2115 | 0.6427 | 0.2115 |
| Q4 | tetouan/zone_1 | 1318 | 783.5 | 930.2 | 1018 | 706.6 | 865.8 |
| Q4 | tetouan/zone_2 | 3275 | 1376 | 1130 | 1149 | 788.4 | 938.3 |
| Q4 | tetouan/zone_3 | 1208 | 694.3 | 1935 | 2003 | 681.1 | 607.4 |
| Q4 | victoria/regional_demand | 0.3535 | 0.3006 | 0.8025 | 0.6851 | 0.2786 | 0.3171 |

Reproduce with `python src/trained_comparison_v2.py`. The run resumes saved cases only if the code, v1 results, inputs and package versions match.
