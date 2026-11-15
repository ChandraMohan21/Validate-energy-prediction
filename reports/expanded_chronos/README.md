# Expanded date evaluation

Four calendar windows per series, ending March, June, September and December.
The same eight series and selected years are retained. No new years are downloaded.
Each window uses the preceding 42-day validation block to select its baseline,
then a 42-day test block with only complete forecasts and no missing context.
The 168-hour validation embargo prevents validation targets entering the test period.
See `window_schedule.csv` for exact dates and `exclusions.json` for excluded windows.
The French household has no complete validation/test setup in Q1 and Q2 under
the original missing-data rule. It remains included in Q3 and Q4. There are 60
series/horizon/window cases in total, including the 16 original Q4 cases.
Different years and
hemispheres mean quarter labels are not identical seasons across datasets.

Baseline calculations are complete. New Chronos inference is pending in Colab.
Q4 repeats the original test period; only Q1-Q3 add previously untested periods.
This follow-up was designed after viewing pilot results, not a preregistered study.
No uncertainty or significance calculations have been performed by this script.

Upload `notebooks/04_expanded_chronos_colab.ipynb` into Google Colab, choose T4 GPU,
run cells in order and upload `expanded_chronos_inputs.zip` when prompted.
Return `expanded_chronos_results.zip` to the workspace for paired analysis.
The fixed model commit matches the original Chronos run. Forecasting package
versions, including torch 2.11.0, are pinned in an isolated virtual environment.
Colab's preinstalled Gradio and Diffusers are not imported into that environment.
The setup cell checks dependency consistency and the next cell checks GPU access.
Initial setup downloads PyTorch and may take several minutes. The existing
input ZIP is unchanged; only the notebook needs to be uploaded again.
Every prediction is saved, and completed cases can resume within the same runtime.

Rebuild locally with `python src/prepare_expanded_chronos.py`.
Original pilot files and the returned original results ZIP are preserved.
