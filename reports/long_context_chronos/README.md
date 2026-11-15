# Chronos with a 2048-hour history

The expanded run (`reports/expanded_chronos/`) gave Chronos-Bolt only 336 hours of history. Its standard history is 2048 hours. On the Q4 Tetouan Zone 3 seven-day case, the earlier 2048-hour run scored MAE 544 against 681 with 336 hours, enough to turn a loss against the validation-selected baseline (607) into a win. This rerun repeats all 60 expanded cases with up to 2048 hours of history and changes nothing else: same model commit, test origins, baselines, blend weights and bootstrap settings.

`protocol.json` fixes the question, the decision rules and the "meaningful improvement" threshold (5%) before any 2048-hour forecast is run. `history_coverage.csv` shows the history each case receives: Q1 starts in February, so its history is 1,153 to 1,992 hours; the two household windows contain short gaps (at most 72 of 2048 hours), passed to Chronos as missing values.

## Run

1. `python src/long_context_study.py prepare` (already done; rebuilds the ZIP and notebook).
2. Upload `notebooks/05_long_context_chronos_colab.ipynb` to Google Colab, choose a T4 GPU, run the cells in order and upload `long_context_chronos_inputs.zip` when asked.
3. Put the downloaded `long_context_chronos_results.zip` in the project folder.
4. `python src/long_context_study.py compare` writes `results/`: 336 vs 2048 MAE per case, the paired bootstrap, and how often the fair baseline changes the claim under each history length.

The comparison step was dry-run with the 336-hour results relabelled as 2048, which reproduced the 336-hour counts exactly.
