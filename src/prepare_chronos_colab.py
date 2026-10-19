"""Build a portable Colab input bundle without publishing or cloning the repository."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    result["id"] = hashlib.sha256(source.encode()).hexdigest()[:12]
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build():
    report_path = ROOT / "reports/baseline_pilot/results.json"
    report = json.loads(report_path.read_text())
    protocol = report["metadata"]["protocol"]
    paths = {report_path, ROOT / "reports/baseline_pilot/protocol.json",
             ROOT / "reports/baseline_pilot/README.md", ROOT / "src/chronos_pilot.py",
             ROOT / "src/baseline_pilot.py"}
    for record in report["results"]:
        year = protocol["years"][record["dataset"]]
        paths.add(ROOT / record["prediction_file"])
        paths.add(ROOT / f"data/processed/baseline_pilot/{record['dataset']}_{record['series']}_{year}.csv")
    manifest = {"purpose": "Frozen baseline cases for zero-shot Chronos-Bolt comparison",
                "files": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(paths)}}
    destination = ROOT / "reports/chronos_colab"
    destination.mkdir(parents=True, exist_ok=True)
    bundle = destination / "chronos_pilot_inputs.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    cells = [cell("markdown", """
# Chronos-Bolt versus frozen seasonal baselines

Eight series across Tetouan, Portugal, a French household and Victoria. Forecasts:
24 and 168 hours, with the exact test origins from the completed baseline pilot.
No fine-tuning or test-based model selection. The model receives the last 336 hourly
observations ending at each origin. Missing contexts are not filled.

**Before running:** Runtime > Change runtime type > T4 GPU. Then run cells in order.
Upload `chronos_pilot_inputs.zip` when prompted. This bundle contains the local
experiment inputs and code, so no GitHub push or repository clone is needed.

The model is Chronos-Bolt base, chosen to continue the existing project. It is not
the newest Chronos model. The 0.5 quantile is used because MAE is the primary metric.
Bolt has a native 64-step horizon; the 168-hour task uses its built-in long-horizon
extension, which can degrade accuracy. There is no guarantee these public datasets
were absent from pretraining. This notebook does not establish statistical significance.

References: [official code](https://github.com/amazon-science/chronos-forecasting),
[model configuration](https://huggingface.co/amazon/chronos-bolt-base/blob/main/config.json).
"""), cell("code", """
%pip install -q "chronos-forecasting==2.2.2" "transformers>=4.48,<5" "numpy>=1.26,<3" "pandas>=2.2,<3"
"""), cell("code", """
import torch
assert torch.cuda.is_available(), "Choose a GPU runtime, then rerun."
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
"""), cell("markdown", """
## Upload the frozen inputs

Choose the supplied input ZIP. Its checksum and every extracted input will be
checked. This uploads public electricity data and experiment files to your Colab
runtime. No credentials or private project files are included.
"""), cell("code", f'''
from google.colab import files
from pathlib import Path, PurePosixPath
import hashlib, zipfile

ROOT = Path("/content/chronos_pilot")
ROOT.mkdir(exist_ok=True)
uploaded = files.upload()
assert len(uploaded) == 1, "Select only chronos_pilot_inputs.zip"
filename, payload = next(iter(uploaded.items()))
assert hashlib.sha256(payload).hexdigest() == "{digest}", "Wrong or changed bundle"
with zipfile.ZipFile(filename) as archive:
    for member in archive.infolist():
        parts = PurePosixPath(member.filename)
        assert not parts.is_absolute() and ".." not in parts.parts and "\\\\" not in member.filename
        assert (ROOT / member.filename).resolve().is_relative_to(ROOT.resolve())
    archive.extractall(ROOT)
del uploaded, payload
print("Inputs ready:", ROOT)
'''), cell("code", """
import subprocess, sys
subprocess.run([sys.executable, str(ROOT / "src/chronos_pilot.py"),
                "--root", str(ROOT), "--check-only"], check=True)
"""), cell("markdown", """
## Run the forecasts

The first run downloads model weights. Progress is printed per batch. Completed
series/horizon files are reused if you rerun with the same configuration.
Progress in `/content` survives rerunning cells, but not deletion of the runtime.
If interrupted, use the download cell to save completed files before disconnecting.

The default batch size is 8 because the 168-step extension increases GPU memory
use. If CUDA runs out of memory, restart the runtime and use batch size 2 with a
new output directory. Do not change model/context after looking at test scores.
"""), cell("code", """
OUTPUT = ROOT / "reports/chronos_pilot"
subprocess.run([sys.executable, "-u", str(ROOT / "src/chronos_pilot.py"),
                "--root", str(ROOT), "--output", str(OUTPUT), "--batch-size", "8"], check=True)
"""), cell("code", """
import pandas as pd
table = pd.read_csv(OUTPUT / "comparison.csv")
display(table.loc[table.baseline == "selected", ["dataset", "series", "horizon_hours",
    "baseline_rule", "chronos_MAE", "baseline_MAE", "chronos_MAE_gain_pct"]])
print("Positive gain means lower MAE for Chronos. These are descriptive results.")
"""), cell("markdown", """
## Download results

Download the ZIP, retain it locally, and return it to the workspace for analysis.
It includes every prediction, actuals, origins, baseline predictions, model commit,
package versions and comparison scores. `complete.json` indicates all 16 cases
finished. Without that marker, results are partial.

Next analysis will compare paired errors while accounting for time dependence.
Trained LightGBM/XGBoost models are a separate later experiment.
"""), cell("code", """
import shutil
assert OUTPUT.exists(), "No results yet"
archive_path = shutil.make_archive("/content/chronos_pilot_results", "zip", OUTPUT)
files.download(archive_path)
""")]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "03_chronos_pilot_colab.ipynb"},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks/03_chronos_pilot_colab.ipynb"
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    (destination / "README.md").write_text("""# Chronos Colab run

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
""", encoding="utf-8")
    print(f"Notebook: {path}\nBundle: {bundle} ({bundle.stat().st_size:,} bytes)\nSHA256: {digest}")


if __name__ == "__main__":
    build()
