# Chronos Colab run

1. Open Google Colab: https://colab.research.google.com/
2. File > Upload notebook: choose `notebooks/03_chronos_pilot_colab.ipynb`.
3. Runtime > Change runtime type > T4 GPU.
4. Run the notebook cells in order. When prompted, upload `chronos_pilot_inputs.zip` from this folder.
5. Download `chronos_pilot_results.zip` from the final cell and put it in this workspace.

The notebook is prepared and its inputs checked locally. No new real Chronos inference has been executed locally.
The ZIP contains only the eight processed public series, baseline outputs/protocol and two runner modules.
No repository publishing is needed. Rebuild after source changes with `python src/prepare_chronos_colab.py`.

The fixed context is 336 hours, unlike the older Tetouan-only 2048-hour experiment.
The older scores must not be substituted for this run. The 168-hour task uses Bolt's built-in extension
beyond its native 64-step output. Float32 and small batches support T4 hardware.
The package is pinned; full resolved versions and model commit are recorded during the real run.
No significance, no contamination-free evaluation and no novelty claim follows from setup checks.
