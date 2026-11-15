# Chronos with a 2048-hour history

Same 60 cases as the 336-hour run; only the history length differs. Protocol written before this run: `../protocol.json`. Q1 history is shorter than 2048 hours. Q4 repeats the original test period. Exploratory; no multiple-comparison correction.

## Sanity check: Q4 Tetouan, 168 hours

| Zone | 2048-hour MAE now | Earlier 2048-hour run |
|---|---:|---:|
| zone_1 | 696.0 | 695.7 |
| zone_2 | 756.3 | 756.0 |
| zone_3 | 543.8 | 543.9 |

## How often the claim holds

Clear = both bootstrap block lengths favour Chronos. Meaningful = clear and at least 5% lower MAE.

| History | Horizon | Cases | Clear vs weekly | Clear vs selected | Meaningful vs weekly | Meaningful vs selected | Weakened by fair baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| 336 h | 24 h | 30 | 24 | 20 | 24 | 20 | 4 |
| 336 h | 168 h | 30 | 18 | 16 | 18 | 15 | 6 |
| 2048 h | 24 h | 30 | 23 | 22 | 23 | 22 | 2 |
| 2048 h | 168 h | 30 | 21 | 14 | 21 | 14 | 8 |

Sensitivity at 2% and 10% and the exact weakened cases are in `summary.json`.

## Seven-day MAE against the validation-selected baseline

| Window | Series | Selected | Chronos 336 h | Chronos 2048 h | Gain 336 h | Gain 2048 h |
|---|---|---:|---:|---:|---:|---:|
| Q1 | portugal/T1 | 4.348 | 3.888 | 3.807 | +10.6% | +12.4% |
| Q1 | portugal/T2 | 6.286 | 4.258 | 4.172 | +32.3% | +33.6% |
| Q1 | portugal/T3 | 0.007937 | 0.4449 | 0.3241 | -5505.7% | -3984.1% |
| Q1 | tetouan/zone_1 | 970.5 | 922 | 889.6 | +5.0% | +8.3% |
| Q1 | tetouan/zone_2 | 716.5 | 692.5 | 670.9 | +3.3% | +6.4% |
| Q1 | tetouan/zone_3 | 572.3 | 564 | 537.1 | +1.4% | +6.2% |
| Q1 | victoria/regional_demand | 0.2587 | 0.1972 | 0.2112 | +23.8% | +18.4% |
| Q2 | portugal/T1 | 8.512 | 6.588 | 5.845 | +22.6% | +31.3% |
| Q2 | portugal/T2 | 7.806 | 6.456 | 6.561 | +17.3% | +16.0% |
| Q2 | portugal/T3 | 0 | 0.4422 | 0.3473 | undefined | undefined |
| Q2 | tetouan/zone_1 | 1741 | 1795 | 1697 | -3.1% | +2.5% |
| Q2 | tetouan/zone_2 | 1331 | 1306 | 1301 | +1.9% | +2.3% |
| Q2 | tetouan/zone_3 | 1022 | 1059 | 989.1 | -3.7% | +3.2% |
| Q2 | victoria/regional_demand | 0.1858 | 0.1584 | 0.1418 | +14.7% | +23.7% |
| Q3 | household/sceaux_household | 0.4274 | 0.4039 | 0.4123 | +5.5% | +3.5% |
| Q3 | portugal/T1 | 14.59 | 13.12 | 13.36 | +10.1% | +8.4% |
| Q3 | portugal/T2 | 10.3 | 9.071 | 9.52 | +12.0% | +7.6% |
| Q3 | portugal/T3 | 0 | 0.4422 | 0.3469 | undefined | undefined |
| Q3 | tetouan/zone_1 | 1765 | 1645 | 1610 | +6.8% | +8.8% |
| Q3 | tetouan/zone_2 | 1969 | 1686 | 1666 | +14.4% | +15.4% |
| Q3 | tetouan/zone_3 | 1664 | 1655 | 1457 | +0.5% | +12.4% |
| Q3 | victoria/regional_demand | 0.2293 | 0.1942 | 0.2023 | +15.3% | +11.8% |
| Q4 | household/sceaux_household | 0.4937 | 0.4614 | 0.4242 | +6.5% | +14.1% |
| Q4 | portugal/T1 | 2.573 | 2.29 | 2.264 | +11.0% | +12.0% |
| Q4 | portugal/T2 | 4.081 | 3.583 | 3.468 | +12.2% | +15.0% |
| Q4 | portugal/T3 | 0.2115 | 0.6427 | 0.552 | -203.8% | -161.0% |
| Q4 | tetouan/zone_1 | 865.8 | 706.6 | 696 | +18.4% | +19.6% |
| Q4 | tetouan/zone_2 | 938.3 | 788.4 | 756.3 | +16.0% | +19.4% |
| Q4 | tetouan/zone_3 | 607.4 | 681.1 | 543.8 | -12.1% | +10.5% |
| Q4 | victoria/regional_demand | 0.3171 | 0.2786 | 0.271 | +12.1% | +14.5% |

Reproduce with `python src/long_context_study.py compare`.
