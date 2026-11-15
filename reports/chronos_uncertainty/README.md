# Chronos paired uncertainty follow-up

Exploratory circular-block bootstrap of origin-level MAE, 5,000 resamples at each of 168 and 336 calendar hours. Positive differences favor Chronos. Missing calendar hours remain gaps. Intervals are conditional on frozen baseline choices and model predictions. No familywise or false-discovery correction is applied.

The windows are short: a complete window contains only five week-long blocks. Circular resampling wraps the end to the start and assumes approximately stationary losses within a window. Treat intervals as approximate sensitivity evidence, not definitive proof. Sparse household windows receive a limited-data label. Q4 repeats the pilot.

## Seven-day comparisons against validation-selected baselines

Difference intervals use each series' original units; do not compare their magnitudes across datasets.

| Window | Series | Gain | 168h difference interval | 336h difference interval | Assessment |
|---|---|---:|---|---|---|
| Q1 | portugal/T1 | +10.57% | [0.3498, 0.5962] | [0.3596, 0.5674] | chronos_supported_both_lengths |
| Q1 | portugal/T2 | +32.26% | [0.6848, 3.898] | [0.7098, 3.697] | chronos_supported_both_lengths |
| Q1 | portugal/T3 | -5505.67% | [-0.4423, -0.4263] | [-0.4423, -0.4281] | baseline_supported_both_lengths |
| Q1 | tetouan/zone_1 | +4.99% | [0.6905, 103.7] | [13.49, 90.14] | chronos_supported_both_lengths |
| Q1 | tetouan/zone_2 | +3.34% | [-22.61, 82.65] | [-12.96, 69.97] | unclear_or_block_sensitive |
| Q1 | tetouan/zone_3 | +1.45% | [-8.506, 25.83] | [-8.591, 25.56] | unclear_or_block_sensitive |
| Q1 | victoria/regional_demand | +23.76% | [0.0303, 0.09236] | [0.02933, 0.09495] | chronos_supported_both_lengths |
| Q2 | portugal/T1 | +22.60% | [0.8534, 3.429] | [0.9783, 3.115] | chronos_supported_both_lengths |
| Q2 | portugal/T2 | +17.29% | [0.3378, 2.565] | [0.2834, 2.547] | chronos_supported_both_lengths |
| Q2 | portugal/T3 | undefined (zero baseline) | [-0.4422, -0.4422] | [-0.4422, -0.4422] | baseline_supported_both_lengths |
| Q2 | tetouan/zone_1 | -3.12% | [-137.4, 35.46] | [-139.5, 34.39] | unclear_or_block_sensitive |
| Q2 | tetouan/zone_2 | +1.95% | [-63.92, 122] | [-65.89, 127.3] | unclear_or_block_sensitive |
| Q2 | tetouan/zone_3 | -3.66% | [-84.3, 6.335] | [-79.01, 4.406] | unclear_or_block_sensitive |
| Q2 | victoria/regional_demand | +14.73% | [0.01314, 0.04271] | [0.01219, 0.04375] | chronos_supported_both_lengths |
| Q3 | household/sceaux_household | +5.49% | [-0.006304, 0.05331] | not estimable (short calendar span) | limited_data |
| Q3 | portugal/T1 | +10.05% | [0.3806, 2.439] | [0.6934, 2.242] | chronos_supported_both_lengths |
| Q3 | portugal/T2 | +11.96% | [0.1814, 2.609] | [0.3003, 2.355] | chronos_supported_both_lengths |
| Q3 | portugal/T3 | undefined (zero baseline) | [-0.4422, -0.4422] | [-0.4422, -0.4422] | baseline_supported_both_lengths |
| Q3 | tetouan/zone_1 | +6.80% | [-23.4, 319.1] | [-22.14, 312.9] | unclear_or_block_sensitive |
| Q3 | tetouan/zone_2 | +14.37% | [98.62, 521.6] | [89, 508.1] | chronos_supported_both_lengths |
| Q3 | tetouan/zone_3 | +0.54% | [-108.9, 130.9] | [-87.74, 109.9] | unclear_or_block_sensitive |
| Q3 | victoria/regional_demand | +15.29% | [0.00386, 0.06742] | [0.01594, 0.0549] | chronos_supported_both_lengths |
| Q4 | household/sceaux_household | +6.54% | [0.01655, 0.06264] | [0.01644, 0.06267] | limited_data |
| Q4 | portugal/T1 | +11.03% | [0.159, 0.4008] | [0.165, 0.3858] | chronos_supported_both_lengths |
| Q4 | portugal/T2 | +12.20% | [0.1973, 0.7727] | [0.2542, 0.7388] | chronos_supported_both_lengths |
| Q4 | portugal/T3 | -203.84% | [-0.4422, -0.4134] | [-0.4422, -0.4161] | baseline_supported_both_lengths |
| Q4 | tetouan/zone_1 | +18.39% | [74.85, 244.1] | [81.67, 228.2] | chronos_supported_both_lengths |
| Q4 | tetouan/zone_2 | +15.98% | [95.73, 220.2] | [96.62, 213.6] | chronos_supported_both_lengths |
| Q4 | tetouan/zone_3 | -12.13% | [-241.3, 57.82] | [-237.6, 60.54] | unclear_or_block_sensitive |
| Q4 | victoria/regional_demand | +12.12% | [0.003307, 0.07851] | [0.001817, 0.07981] | chronos_supported_both_lengths |

Method reference: [arch time-series bootstrap documentation](https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html). This script implements the index resampling using NumPy and does not require arch.

Reproduce with `python src/chronos_uncertainty.py`. No new model inference is performed.
