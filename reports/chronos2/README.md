# Chronos-2 frozen-case experiment

Prepared 21 October 2026. No Chronos-2 inference or results yet.

This tests whether the seasonal-baseline finding carries over from Chronos-Bolt
to Chronos-2. Same 60 frozen cases, origin-inclusive history up to 2048 hours,
24/168-hour horizons, validation-selected rules and blend weights. Q1 history is
shorter; older household gaps remain NaN. Target-only, zero-shot, float32 median,
separate univariate task per origin, cross_learning=False. Chronos-2 can use more
history, but this comparison deliberately caps both models at 2048 hours.

Model: amazon/chronos-2, commit 29ec3766d36d6f73f0696f85560a422f50e8498c.
Direct package pins are in requirements-colab.txt; pip freeze records the resolved
environment in the output ZIP. Official API and model sources are in protocol.json.

## Run yourself in Colab

1. Upload notebooks/06_chronos2_colab.ipynb to https://colab.research.google.com/.
2. Select Runtime > Change runtime type > T4 GPU.
3. Run setup and GPU check cells. The isolated virtualenv avoids Colab package conflicts.
4. At the upload prompt choose reports/chronos2/chronos2_inputs.zip only.
5. Run the short preflight. It checks both horizons, ragged/gapped histories,
   finite median output and prediction agreement alone vs in a batch.
   If it fails, stop and return the traceback; do not launch the long cell.
6. Leave RESTORE=False for a fresh run. Run the long cell with BATCH_SIZE=8.
   It prints window/case counts and per-batch origin progress with elapsed time/ETA.
7. Run the last cell and download chronos2_results.zip. All four windows must say
   complete, with 14/14/16/16 cases. Put the ZIP in electricity-prediction/ and
   return it with the notebook output, including preflight and package errors if any.

## Interruptions and memory

Finished cases save atomically as NPZ files. An interrupted case restarts, while
completed cases resume. In the same runtime rerun the long cell. Before disconnecting,
run the download cell to save partial results. A lost runtime loses undownloaded files.
In a new runtime repeat setup/input upload, set RESTORE=True, upload the partial
chronos2_results.zip, then rerun the long cell. Configuration fingerprints guard reuse.
For GPU memory failures use a new output folder, reduce BATCH_SIZE consistently,
and do not mix old cases with a changed batch size. Return the error if unsure.

## Files saved

ZIP contains Q1/ through Q4/: per-origin forecast matrices in NPZ (rows are
test_origins, columns are forecast hours 1..horizon), hourly_time for timestamps,
actuals, every frozen baseline, selected rule, history lengths/missing counts,
configuration fingerprint, run_metadata.json, comparison.csv and complete.json.
The legacy NPZ key 'chronos' stores Chronos-2; model identity is explicit in metadata.
Top-level protocol and pip freeze preserve experiment/environment provenance.
Earlier Chronos-Bolt results and notebooks are unchanged.

## After output is returned

From electricity-prediction/:

    python src/chronos2_study.py compare --archive chronos2_results.zip

This is local analysis only, no model inference. It checks all 60 frozen cases,
actuals, origins, baseline arrays, choices, model/version identity and matched
history, then writes reports/chronos2/results/ only. Reports MAE/RMSE/MAPE for
both models and all seasonal baselines; paired Chronos-2 intervals versus daily,
weekly, hybrid, blend, selected and Bolt. Also recomputes Bolt weekly/selected
intervals using the same existing functions, seeds and block lengths.

5,000 paired circular bootstrap resamples at 168/336 calendar hours; retain gaps.
Clear requires agreement at both lengths and sufficient valid origins. Meaningful
requires clear and >=5% MAE gain; 2%/10% sensitivity. Zero-baseline gains remain
undefined. Summary counts weakened/strengthened claims for each model separately.
No pooling of errors in different units. Intervals are exploratory, without multiple
comparison correction. Q1 shorter history, Q4 reuse, related meters, sparse household
windows and possible pretraining overlap remain caveats. Both models belong to the
Chronos family. Do not interpret outcomes until the user returns actual forecasts.

Preparation can be reproduced with `python src/chronos2_study.py prepare`.
This does not run model inference. No commit or push was performed.
