"""Prepare quarterly follow-up windows; preserve the original pilot and its results."""
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_pilot as baseline
from chronos_pilot import load_cases
from prepare_chronos_colab import cell

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "reports/expanded_chronos"
REVISION = "5d9f166d69f47aef3401367a7b842e78fe97b121"


def export_window(label, month, original, protocol):
    root = DEST / "inputs" / label
    output = root / "reports/baseline_pilot"
    output.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(exist_ok=True)
    for name in ("baseline_pilot.py", "chronos_pilot.py"):
        shutil.copyfile(ROOT / "src" / name, root / "src" / name)
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    unique = {(r["dataset"], r["series"]): r for r in original["results"]}
    records, exclusions = [], []
    old_root, old_out = baseline.ROOT, baseline.OUT
    try:
        baseline.ROOT, baseline.OUT = root, output
        for (dataset, name), record in unique.items():
            year = protocol["years"][dataset]
            relative = f"data/processed/baseline_pilot/{dataset}_{name}_{year}.csv"
            source = ROOT / relative
            if baseline.file_hash(source) != record["processed_sha256"]:
                raise ValueError(f"Original processed file changed: {source}")
            series = pd.read_csv(source, index_col="Datetime", parse_dates=True, float_precision="round_trip")["load"]
            end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
            # Tetouan genuinely ends on December 30. Never invent a final day.
            series = series.loc[:min(end, series.index.max())]
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            series.to_csv(path, index_label="Datetime")
            try:
                baseline.origin_indices(series, protocol)
            except ValueError as error:
                if "No complete validation or test origins" not in str(error):
                    raise
                exclusions.append({"dataset": dataset, "series": name, "window": label,
                                   "reason": str(error), "horizons_hours": protocol["horizons_hours"]})
                continue
            records.extend(baseline.evaluate({"dataset": dataset, "series": name,
                "unit": record["unit"], "data": series, "path": path}, protocol))
    finally:
        baseline.ROOT, baseline.OUT = old_root, old_out
    metadata = {"protocol": protocol, "protocol_sha256": baseline.file_hash(protocol_path),
                "window": label, "cutoff_month": month, "source": "Original pilot processed files",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "Baselines computed; Chronos inference pending", "exclusions": exclusions}
    (output / "results.json").write_text(json.dumps({"metadata": metadata, "results": records}, indent=2, allow_nan=False), encoding="utf-8")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {"purpose": "Quarterly follow-up with frozen earlier-validation selection",
                "files": {p.relative_to(root).as_posix(): baseline.file_hash(p) for p in files}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    load_cases(root)
    return records, exclusions


def build_notebook(digest):
    cells = [cell("markdown", """
# Chronos follow-up across four periods

Eight series, four datasets, four periods ending in March, June, September and
December of each dataset's selected year. Each period has 42-day validation and
test calendar blocks, with a 168-hour validation embargo. Complete test forecasts
must finish by the period endpoint, giving up to 840 hourly origins per window.
The French household is excluded in Q1 and Q2: gaps prevent complete validation/test
setups under the original missing-data rule. It remains included in Q3 and Q4.
Quarter labels are calendar
labels, not identical seasons across countries.

Daily/weekly/blend/hybrid definitions, 336-hour context, 0.5 quantile and model ID
are unchanged. Blend weights and baseline choice are selected afresh using only
the earlier validation period for each window. No tuning uses that window's test
answers. The model commit is fixed to the completed pilot's commit.

This is a follow-up designed after inspecting the original pilot. December
repeats the original evaluation period, so do not count it as new evidence.
Earlier windows cover more seasonal conditions, not additional years.
Pretraining overlap is not excluded. Forecasts overlap within a window, so
uncertainty analysis must still account for time dependence.

**Run:** choose Runtime > Change runtime type > T4 GPU. Run cells in order.
Upload `expanded_chronos_inputs.zip` when prompted. No GitHub push is needed.
Forecasting packages are installed in an isolated environment so Colab's Gradio
and Diffusers dependencies cannot conflict. Initial setup downloads PyTorch and
can take several minutes. The notebook's own environment is left unchanged.
"""), cell("code", '''
import os, subprocess, sys
from pathlib import Path

ENV = Path("/content/chronos_forecast_env")
PYTHON = str(ENV / "bin/python")
subprocess.run([sys.executable, "-m", "venv", str(ENV)], check=True)
# Do not inherit external Python package paths into the isolated process.
RUN_ENV = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "PYTHONHOME")}
RUN_ENV["PYTHONNOUSERSITE"] = "1"
subprocess.run([PYTHON, "-m", "pip", "install", "--quiet",
                "torch==2.11.0", "chronos-forecasting==2.2.2",
                "transformers==4.57.6", "numpy==2.1.3", "pandas==2.2.3"],
               env=RUN_ENV, check=True)
subprocess.run([PYTHON, "-m", "pip", "check"], env=RUN_ENV, check=True)
'''), cell("code", '''
subprocess.run([PYTHON, "-c", "import torch; from chronos import BaseChronosPipeline; "
                "assert torch.cuda.is_available(), 'Select a GPU runtime first'; "
                "print(torch.__version__, torch.cuda.get_device_name(0))"],
               env=RUN_ENV, check=True)
'''), cell("code", f'''
from google.colab import files
from pathlib import Path, PurePosixPath
import hashlib, zipfile, subprocess, sys, json

ROOT = Path("/content/expanded_chronos")
ROOT.mkdir(exist_ok=True)
uploaded = files.upload()
assert len(uploaded) == 1, "Choose only expanded_chronos_inputs.zip"
filename, payload = next(iter(uploaded.items()))
assert hashlib.sha256(payload).hexdigest() == "{digest}", "Wrong or changed ZIP"
with zipfile.ZipFile(filename) as archive:
    for member in archive.infolist():
        parts = PurePosixPath(member.filename)
        assert not parts.is_absolute() and ".." not in parts.parts and "\\\\" not in member.filename
        assert (ROOT / member.filename).resolve().is_relative_to(ROOT.resolve())
    archive.extractall(ROOT)
del uploaded, payload
for label in ("Q1", "Q2", "Q3", "Q4"):
    base = ROOT / label
    subprocess.run([PYTHON, str(base / "src/chronos_pilot.py"),
                    "--root", str(base), "--check-only"], env=RUN_ENV, check=True)
'''), cell("markdown", """
## Run all four periods

60 series/horizon/window cases in total. The model is loaded once per window.
Completed cases are saved under `results/Q1` etc. Rerunning this cell reuses
completed cases with an identical configuration. Runtime deletion removes local
progress, so download partial results before disconnecting if necessary.

The 168-hour task still uses Bolt's built-in extension beyond its native 64 steps.
No observed future values enter the model. Float32 and batch size 8 support T4.
December is rerun to keep outputs self-contained and comparable; the earlier
December ZIP remains preserved in the local project.
"""), cell("code", f'''
OUTPUT = ROOT / "results"
for label in ("Q1", "Q2", "Q3", "Q4"):
    base = ROOT / label
    subprocess.run([PYTHON, "-u", str(base / "src/chronos_pilot.py"),
                    "--root", str(base), "--output", str(OUTPUT / label),
                    "--batch-size", "8", "--model-revision", "{REVISION}"], env=RUN_ENV, check=True)
'''), cell("code", '''
import csv
rows = []
for label in ("Q1", "Q2", "Q3", "Q4"):
    assert (OUTPUT / label / "complete.json").exists(), f"Incomplete window: {label}"
    with (OUTPUT / label / "comparison.csv").open(newline="") as stream:
        rows.extend({"window": label, **row} for row in csv.DictReader(stream))
with (OUTPUT / "all_windows_comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    if row["baseline"] == "selected" and row["horizon_hours"] == "168":
        print(row["window"], row["dataset"], row["series"], row["baseline_rule"],
              "Chronos gain:", round(float(row["chronos_MAE_gain_pct"]), 2), "%")
print("Positive means lower Chronos MAE. These are descriptive comparisons, not significance tests.")
'''), cell("code", '''
import shutil
assert OUTPUT.exists(), "No results yet"
result_zip = shutil.make_archive("/content/expanded_chronos_results", "zip", OUTPUT)
files.download(result_zip)
''')]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks/04_expanded_chronos_colab.ipynb"
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return path


def build():
    DEST.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT / "reports/baseline_pilot/results.json").read_text())
    protocol = original["metadata"]["protocol"]
    # Save follow-up rules before computing any new test metrics.
    plan = {"status": "Follow-up after inspecting the original pilot, not preregistration",
            "windows": {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12},
            "window_endpoint": "Last available hourly observation through calendar quarter end",
            "years": protocol["years"], "validation_days": 42, "test_days": 42,
            "embargo_hours": 168, "horizons": [24, 168], "context_hours": 336,
            "selection": "Choose baseline and blend weights separately on preceding validation MAE only",
            "model_id": "amazon/chronos-bolt-base", "model_revision": REVISION,
            "repeated_evidence": "Q4 repeats the original test period and is not independent new evidence",
            "missing_window_policy": "Exclude and report any series/window with no complete validation or test origins; do not change gap rules"}
    (DEST / "protocol.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    rows, excluded = [], []
    for label, month in plan["windows"].items():
        print(f"Preparing {label}", flush=True)
        records, omissions = export_window(label, month, original, protocol)
        excluded.extend(omissions)
        for r in records:
            rows.append({"window": label, **{k: r[k] for k in (
                "dataset", "series", "horizon_hours", "selected_on_validation", "blend_daily_weight",
                "validation_origin_first", "validation_target_last", "test_origin_first", "test_target_last")},
                "validation_origins": r["split"]["validation_origins"], "test_origins": r["split"]["test_origins"]})
    pd.DataFrame(rows).to_csv(DEST / "window_schedule.csv", index=False)
    (DEST / "exclusions.json").write_text(json.dumps(excluded, indent=2), encoding="utf-8")
    bundle = DEST / "expanded_chronos_inputs.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((DEST / "inputs").rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                archive.write(p, p.relative_to(DEST / "inputs").as_posix())
    notebook = build_notebook(baseline.file_hash(bundle))
    (DEST / "README.md").write_text("""# Expanded date evaluation

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
Every prediction is saved, and completed cases can resume within the same runtime.

Rebuild locally with `python src/prepare_expanded_chronos.py`.
Original pilot files and the returned original results ZIP are preserved.
""", encoding="utf-8")
    print(f"Verified {len(rows)} cases.\nNotebook: {notebook}\nBundle: {bundle} ({bundle.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
