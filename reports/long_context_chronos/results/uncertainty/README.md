# Chronos paired uncertainty follow-up

Exploratory circular-block bootstrap of origin-level MAE, 5,000 resamples at each of 168 and 336 calendar hours. Positive differences favor Chronos. Missing calendar hours remain gaps. Intervals are conditional on frozen baseline choices and model predictions. No familywise or false-discovery correction is applied.

The windows are short: a complete window contains only five week-long blocks. Circular resampling wraps the end to the start and assumes approximately stationary losses within a window. Treat intervals as approximate sensitivity evidence, not definitive proof. Sparse household windows receive a limited-data label. Q4 repeats the pilot.

## Seven-day comparisons against validation-selected baselines

Difference intervals use each series' original units; do not compare their magnitudes across datasets.

| Window | Series | Gain | 168h difference interval | 336h difference interval | Assessment |
|---|---|---:|---|---|---|
| Q1 | portugal/T1 | +12.45% | [0.4045, 0.6823] | [0.4052, 0.6642] | chronos_supported_both_lengths |
| Q1 | portugal/T2 | +33.64% | [0.7616, 4.092] | [0.8067, 3.876] | chronos_supported_both_lengths |
| Q1 | portugal/T3 | -3984.09% | [-0.3218, -0.3086] | [-0.3217, -0.3099] | baseline_supported_both_lengths |
| Q1 | tetouan/zone_1 | +8.33% | [23.59, 152.4] | [29.17, 143.9] | chronos_supported_both_lengths |
| Q1 | tetouan/zone_2 | +6.36% | [-6.56, 123.9] | [-4.562, 111.5] | unclear_or_block_sensitive |
| Q1 | tetouan/zone_3 | +6.15% | [6.221, 62.65] | [5.583, 59.31] | chronos_supported_both_lengths |
| Q1 | victoria/regional_demand | +18.36% | [0.01042, 0.08541] | [0.01097, 0.086] | chronos_supported_both_lengths |
| Q2 | portugal/T1 | +31.33% | [0.9402, 5.117] | [0.9824, 4.805] | chronos_supported_both_lengths |
| Q2 | portugal/T2 | +15.95% | [0.1718, 2.512] | [0.1196, 2.519] | chronos_supported_both_lengths |
| Q2 | portugal/T3 | undefined (zero baseline) | [-0.3475, -0.3471] | [-0.3475, -0.3471] | baseline_supported_both_lengths |
| Q2 | tetouan/zone_1 | +2.54% | [-32.39, 113.3] | [-37.86, 118] | unclear_or_block_sensitive |
| Q2 | tetouan/zone_2 | +2.26% | [-38.38, 101.6] | [-32.8, 100.9] | unclear_or_block_sensitive |
| Q2 | tetouan/zone_3 | +3.23% | [-54.48, 131.5] | [-50.51, 129.1] | unclear_or_block_sensitive |
| Q2 | victoria/regional_demand | +23.68% | [0.02883, 0.06136] | [0.03207, 0.05591] | chronos_supported_both_lengths |
| Q3 | household/sceaux_household | +3.53% | [-0.01834, 0.04875] | not estimable (short calendar span) | limited_data |
| Q3 | portugal/T1 | +8.44% | [-0.3909, 2.518] | [0.04674, 2.285] | unclear_or_block_sensitive |
| Q3 | portugal/T2 | +7.60% | [-0.3456, 2.101] | [-0.1252, 1.779] | unclear_or_block_sensitive |
| Q3 | portugal/T3 | undefined (zero baseline) | [-0.3469, -0.3469] | [-0.3469, -0.3469] | baseline_supported_both_lengths |
| Q3 | tetouan/zone_1 | +8.82% | [-64.12, 409.2] | [-67.23, 406.9] | unclear_or_block_sensitive |
| Q3 | tetouan/zone_2 | +15.39% | [74.3, 579.5] | [66.85, 576.5] | chronos_supported_both_lengths |
| Q3 | tetouan/zone_3 | +12.45% | [-9.866, 458.1] | [49.1, 401.4] | unclear_or_block_sensitive |
| Q3 | victoria/regional_demand | +11.77% | [-0.005062, 0.06197] | [0.003306, 0.04999] | unclear_or_block_sensitive |
| Q4 | household/sceaux_household | +14.06% | [0.04611, 0.1045] | [0.04612, 0.1026] | limited_data |
| Q4 | portugal/T1 | +12.03% | [0.09887, 0.4941] | [0.1204, 0.4788] | chronos_supported_both_lengths |
| Q4 | portugal/T2 | +15.02% | [0.2239, 0.9516] | [0.2871, 0.92] | chronos_supported_both_lengths |
| Q4 | portugal/T3 | -160.95% | [-0.3469, -0.33] | [-0.3469, -0.3316] | baseline_supported_both_lengths |
| Q4 | tetouan/zone_1 | +19.62% | [73.89, 267.1] | [98.01, 232.7] | chronos_supported_both_lengths |
| Q4 | tetouan/zone_2 | +19.40% | [96.9, 272.8] | [116.7, 240.2] | chronos_supported_both_lengths |
| Q4 | tetouan/zone_3 | +10.46% | [21.84, 108.9] | [33.91, 96.66] | chronos_supported_both_lengths |
| Q4 | victoria/regional_demand | +14.54% | [-0.0004698, 0.1059] | [-0.00176, 0.1035] | unclear_or_block_sensitive |

Method reference: [arch time-series bootstrap documentation](https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html). This script implements the index resampling using NumPy and does not require arch.

Reproduce with `python src/chronos_uncertainty.py`. No new model inference is performed.
