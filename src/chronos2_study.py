"""Prepare Chronos-2 Colab files, or compare returned forecasts without model inference."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_pilot import file_hash, score
from chronos2_frozen import MODEL_ID, PACKAGES, REVISION
from chronos_long_context import CONTEXT, long_contexts
from chronos_pilot import load_cases
from chronos_uncertainty import calendar_losses, paired_interval
from long_context_study import claim_table, summarize

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "reports/chronos2"
INPUTS = ROOT / "reports/expanded_chronos/expanded_chronos_inputs.zip"
BOLT = ROOT / "long_context_chronos_results.zip"
BUNDLE = DEST / "chronos2_inputs.zip"
NOTEBOOK = ROOT / "notebooks/06_chronos2_colab.ipynb"
WINDOWS = ("Q1", "Q2", "Q3", "Q4")
PROTOCOL = {
    "prepared_date": "2026-10-21",
    "status": "Planned before Chronos-2 inference; earlier baseline and Chronos-Bolt results inspected. "
              "Exploratory follow-up, not an independent confirmation or a preregistration.",
    "question": "Does seasonal baseline choice also change the margin and uncertainty for Chronos-2, "
                "and how does Chronos-2 compare with Chronos-Bolt at matched history?",
    "model_id": MODEL_ID, "model_revision": REVISION, "packages": PACKAGES,
    "cases": 60, "horizons_hours": [24, 168], "context_hours": CONTEXT,
    "history": "Same origin-inclusive up-to-2048-hour history as Bolt; Q1 shorter; NaN gaps retained; "
               "latest 336 hours complete. No interpolation or future observations.",
    "inference": "Zero-shot, target-only, float32, median (0.5 quantile); each origin a separate "
                 "univariate task; cross_learning=False; context_length=2048; native horizon prediction.",
    "comparators": ["daily", "weekly", "hybrid", "blend", "selected", "chronos_bolt_2048"],
    "selection": "Frozen validation-MAE baseline rule and blend weight. No test-based reselection or tuning.",
    "metrics": "Reuse baseline_pilot.score for MAE, RMSE, MAPE; MAE primary. "
               "Per-case percentage gain undefined when comparator MAE is zero; no pooled raw-unit errors.",
    "uncertainty": "Reuse chronos_uncertainty.calendar_losses and paired_interval: origin-level horizon MAE, "
                   "paired circular blocks, 5000 resamples, 168 and 336 calendar hours, 95% intervals, "
                   "seed 20261019 plus first eight SHA256 hex digits of window/dataset/series/horizon plus block.",
    "clear": "Both block lengths favour the same method, with at least two blocks of valid origins "
             "at each length; otherwise unclear/block-sensitive or limited_data.",
    "meaningful": "Clear and at least 5% lower MAE; 2% and 10% sensitivity, as in the Bolt comparison.",
    "reporting": "Report all cases including Portugal T3 zero-error baselines; Q1 separately because "
                 "history is shorter. Count weakened and strengthened clear wins vs weekly/selected "
                 "for each model. Include paired Chronos-2 vs Bolt intervals, with positive favouring Chronos-2.",
    "limits": "No multiplicity correction. Short windows, related meters, Q4 repeats pilot, "
              "household sparse, cases already inspected, pretraining overlap not ruled out. "
              "Two models from the same family do not establish cross-family generality.",
    "sources": [
        "https://huggingface.co/amazon/chronos-2/tree/" + REVISION,
        "https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos2/pipeline.py"],
}


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    coverage = []
    with zipfile.ZipFile(INPUTS) as source:
        for window in WINDOWS:
            folder = ROOT / "reports/expanded_chronos/inputs" / window
            for name in source.namelist():
                if name.startswith(window + "/") and not name.endswith("/"):
                    if (folder / name[len(window) + 1:]).read_bytes() != source.read(name):
                        raise ValueError(f"Frozen input differs: {name}")
            for record, series, saved in load_cases(folder):
                histories = long_contexts(series.to_numpy(), saved["test_origins"])
                coverage.append({"window": window, "dataset": record["dataset"],
                    "series": record["series"], "horizon_hours": record["horizon_hours"],
                    "test_origins": len(histories), "min_history_hours": min(map(len, histories)),
                    "max_history_hours": max(map(len, histories)),
                    "max_missing_hours": max(int(np.isnan(h).sum()) for h in histories)})
        if len(coverage) != 60:
            raise ValueError("Expected exactly 60 cases")
        protocol = {**PROTOCOL, "frozen_inputs_sha256": file_hash(INPUTS),
                    "bolt_results_sha256": file_hash(BOLT),
                    "runner_sha256": file_hash(ROOT / "src/chronos2_frozen.py"),
                    "history_helper_sha256": file_hash(ROOT / "src/chronos_long_context.py")}
        protocol_file = DEST / "protocol.json"
        if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
            raise ValueError("Prepared protocol changed. Preserve the old experiment before changing it.")
        protocol_file.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame(coverage).to_csv(DEST / "history_coverage.csv", index=False)
        requirements = DEST / "requirements-colab.txt"
        requirements.write_text("\n".join(f"{p}=={v}" for p, v in PACKAGES.items()) + "\n")
        with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in source.namelist():
                bundle.writestr(name, source.read(name))
            for name in ("chronos2_frozen.py", "chronos_long_context.py"):
                bundle.write(ROOT / "src" / name, name)
            bundle.write(protocol_file, "chronos2_protocol.json")
            bundle.write(requirements, requirements.name)
    build_notebook(file_hash(BUNDLE))
    (DEST / "README.md").write_text(README, encoding="utf-8")
    print(f"Prepared {len(coverage)} cases. No model inference run.\n{NOTEBOOK}\n{BUNDLE}")


def build_notebook(digest):
    def cell(kind, source):
        item = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        item["id"] = hashlib.sha256(source.encode()).hexdigest()[:12]
        return item

    # Reuse the isolated environment setup that fixed the previous Colab setup failure.
    old = json.loads((ROOT / "notebooks/05_long_context_chronos_colab.ipynb").read_text())
    setup = old["cells"][1]["source"]
    if isinstance(setup, list):
        setup = "".join(setup)
    setup = setup.replace("chronos_forecast_env", "chronos2_forecast_env").replace(
        "chronos_env_bootstrap", "chronos2_env_bootstrap")
    setup += '\nwith open("/content/chronos2_packages.txt", "w") as stream:\n    subprocess.run([PYTHON, "-m", "pip", "freeze"], stdout=stream, env=RUN_ENV, check=True)\n'
    cells = [cell("markdown", """# Chronos-2 on the same 60 frozen cases
Select **Runtime > Change runtime type > T4 GPU**, then run cells in order.
Upload `chronos2_inputs.zip`. The short preflight must pass before the long cell.
Same 24/168-hour targets, up to 2048 hours of history, frozen seasonal baselines.
No fine-tuning, covariates, or cross-learning between origins. Earlier results stay separate.
This notebook does not interpret the results. Download the ZIP and return it for comparison.
"""), cell("code", setup), cell("code", """
subprocess.run([PYTHON, "-c", "import torch; from chronos import Chronos2Pipeline; "
                "assert torch.cuda.is_available(), 'Select a GPU runtime'; "
                "print(torch.__version__, torch.cuda.get_device_name(0))"], env=RUN_ENV, check=True)
"""), cell("code", f'''
from google.colab import files
from pathlib import PurePosixPath
import hashlib, zipfile, shutil
ROOT = Path("/content/chronos2_frozen")
ROOT.mkdir(exist_ok=True)
def unpack(payload, destination):
    import io
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename)
            assert not parts.is_absolute() and ".." not in parts.parts and "\\\\" not in member.filename
            assert (destination / member.filename).resolve().is_relative_to(destination.resolve())
        archive.extractall(destination)
uploaded = files.upload()
assert len(uploaded) == 1, "Choose only chronos2_inputs.zip"
payload = next(iter(uploaded.values()))
assert hashlib.sha256(payload).hexdigest() == "{digest}", "Wrong or changed inputs ZIP"
unpack(payload, ROOT)
del uploaded, payload
RUNNER = str(ROOT / "chronos2_frozen.py")
OUTPUT = ROOT / "results"
OUTPUT.mkdir(exist_ok=True)
shutil.copyfile("/content/chronos2_packages.txt", OUTPUT / "packages.txt")
shutil.copyfile(ROOT / "chronos2_protocol.json", OUTPUT / "chronos2_protocol.json")
for label in ("Q1", "Q2", "Q3", "Q4"):
    subprocess.run([PYTHON, RUNNER, "--root", str(ROOT / label), "--check-only"], env=RUN_ENV, check=True)
'''), cell("code", '''
# Small synthetic check only: both horizons, ragged/gapped inputs, batch isolation.
subprocess.run([PYTHON, "-u", RUNNER, "--root", str(ROOT / "Q1"), "--smoke-only"],
               env=RUN_ENV, check=True)
'''), cell("markdown", '''## Optional: restore completed cases after a disconnected runtime
Skip the next cell for a fresh run. To restore, upload a previously downloaded
`chronos2_results.zip`. The runner checks configuration and frozen arrays before reuse.
Only completed cases resume. An interrupted case restarts from its first origin.
'''), cell("code", '''
RESTORE = False  # Change to True only when restoring a saved results ZIP.
if RESTORE:
    uploaded = files.upload()
    assert len(uploaded) == 1, "Choose only chronos2_results.zip"
    unpack(next(iter(uploaded.values())), OUTPUT)
    del uploaded
'''), cell("markdown", '''## Long run: 60 cases
Progress shows window, case, origin count, elapsed time and estimated time remaining.
Keep batch size 8 when resuming. If GPU memory fails, use a new output folder and
reduce the batch size consistently for all periods. No automatic configuration changes.
The ZIP download cell also works after an interruption while the runtime remains available.
'''), cell("code", '''
BATCH_SIZE = 8
for number, label in enumerate(("Q1", "Q2", "Q3", "Q4"), 1):
    print(f"Window {number}/4: {label}", flush=True)
    subprocess.run([PYTHON, "-u", RUNNER, "--root", str(ROOT / label),
                    "--output", str(OUTPUT / label), "--batch-size", str(BATCH_SIZE)],
                   env=RUN_ENV, check=True)
print("Forecast generation finished. Run the download cell next.")
'''), cell("code", '''
import json
for label, expected in (("Q1", 14), ("Q2", 14), ("Q3", 16), ("Q4", 16)):
    marker = OUTPUT / label / "complete.json"
    if marker.exists():
        assert json.loads(marker.read_text())["cases"] == expected
        print(f"{label}: complete ({expected} cases)")
    else:
        print(f"{label}: incomplete; this ZIP is a partial checkpoint")
result_zip = shutil.make_archive("/content/chronos2_results", "zip", OUTPUT)
files.download(result_zip)
''')]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def checked_forecasts(archive, inputs, protocol, model_id, revision):
    """Validate every case before comparison; a relabelled/partial run must fail."""
    from chronos_long_context import REVISION as BOLT_REVISION
    cases = {}
    for window, expected in zip(WINDOWS, (14, 14, 16, 16)):
        identity = json.loads(archive.read(window + "/run_metadata.json"))["identity"]
        if (identity["model_id"], identity["model_revision"], identity["context_hours"], identity["quantile"]) != (
                model_id, revision, CONTEXT, 0.5):
            raise ValueError("Unexpected model, revision, history or quantile")
        if identity["manifest_sha256"] != hashlib.sha256(inputs.read(window + "/manifest.json")).hexdigest():
            raise ValueError("Input manifest differs")
        if revision != BOLT_REVISION:
            if identity.get("cross_learning") is not False or identity.get("covariates") is not False:
                raise ValueError("Chronos-2 must be target-only with isolated origins")
            for key in ("runner_sha256", "history_helper_sha256"):
                if identity[key] != protocol[key]:
                    raise ValueError("Runner differs from the prepared protocol")
            if any(identity["packages"][p].split("+")[0] != v for p, v in PACKAGES.items()):
                raise ValueError("Package version differs")
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        names = sorted(n for n in archive.namelist() if n.startswith(window + "/") and n.endswith(".npz"))
        frozen_names = sorted(n for n in inputs.namelist() if n.startswith(
            window + "/reports/baseline_pilot/predictions/") and n.endswith(".npz"))
        if len(names) != expected or [n.split("/")[-1] for n in names] != [n.split("/")[-1] for n in frozen_names]:
            raise ValueError("Frozen case list differs")
        if json.loads(archive.read(window + "/complete.json"))["cases"] != expected:
            raise ValueError("Incomplete window")
        for name, frozen_name in zip(names, frozen_names):
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as saved, np.load(
                    io.BytesIO(inputs.read(frozen_name)), allow_pickle=False) as frozen:
                for key in frozen.files:
                    np.testing.assert_array_equal(saved[key], frozen[key])
                if str(saved["run_fingerprint"]) != fingerprint:
                    raise ValueError("Forecast fingerprint differs")
                if saved["chronos"].shape != frozen["actual"].shape or not np.isfinite(saved["chronos"]).all():
                    raise ValueError("Invalid forecasts")
                cases[name] = {k: saved[k] for k in saved.files}
    if len(cases) != 60:
        raise ValueError("Expected 60 cases")
    return cases


def compare(archive_path):
    from chronos_long_context import MODEL_ID as BOLT_ID, REVISION as BOLT_REVISION
    protocol = json.loads((DEST / "protocol.json").read_text())
    for key, value in PROTOCOL.items():
        if protocol[key] != value:
            raise ValueError("Protocol changed")
    if file_hash(INPUTS) != protocol["frozen_inputs_sha256"] or file_hash(BOLT) != protocol["bolt_results_sha256"]:
        raise ValueError("Frozen inputs or prior Bolt results changed")
    with zipfile.ZipFile(archive_path) as archive, zipfile.ZipFile(INPUTS) as inputs, zipfile.ZipFile(BOLT) as bolt:
        if json.loads(archive.read("chronos2_protocol.json")) != protocol:
            raise ValueError("Returned protocol differs")
        new_cases = checked_forecasts(archive, inputs, protocol, MODEL_ID, REVISION)
        old_cases = checked_forecasts(bolt, inputs, protocol, BOLT_ID, BOLT_REVISION)
    rows, uncertainty = [], []
    for name, new in new_cases.items():
        old = old_cases[name]
        for key in ("context_hours", "context_missing_hours"):
            np.testing.assert_array_equal(new[key], old[key])
        window, basename = name.split("/")
        dataset, rest = basename.split("_", 1)
        series, horizon = rest.rsplit("_h", 1)
        horizon = int(horizon[:-4])
        label = f"{window}/{dataset}/{series}/{horizon}"
        selected = str(new["selected_on_validation"])
        predictions = {k: new[k] for k in ("daily", "weekly", "hybrid", "blend")}
        predictions.update(selected=new[selected], chronos_bolt_2048=old["chronos"], chronos2=new["chronos"])
        metrics = {k: score(new["actual"], value) for k, value in predictions.items()}
        rows.append({"window": window, "dataset": dataset, "series": series, "horizon_hours": horizon,
            "selected_rule": selected, "test_origins": len(new["test_origins"]),
            **{f"{model}_{metric}": value for model, scores in metrics.items() for metric, value in scores.items()}})
        seed = 20261019 + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
        for model in ("chronos2", "chronos_bolt_2048"):
            comparators = PROTOCOL["comparators"] if model == "chronos2" else ["weekly", "selected"]
            for comparator in comparators:
                grid = calendar_losses(new["test_origins"], new["actual"], predictions[model], predictions[comparator])
                means = np.nanmean(grid, axis=0)
                intervals = [paired_interval(grid, block, 5000, seed + block) for block in (168, 336)]
                directions = {i["interval_direction"] for i in intervals}
                assessment = ("limited_data" if not all(i["at_least_two_blocks_of_valid_origins"] for i in intervals)
                    else "chronos_supported_both_lengths" if directions == {"chronos"}
                    else "baseline_supported_both_lengths" if directions == {"baseline"}
                    else "unclear_or_block_sensitive")
                uncertainty.append({"window": window, "dataset": dataset, "series": series,
                    "horizon_hours": horizon, "model": model, "comparator": comparator,
                    "model_MAE": float(means[0]), "comparator_MAE": float(means[1]),
                    "difference": float(means[1] - means[0]),
                    "gain_pct": float(100*(1-means[0]/means[1])) if means[1] else None,
                    "intervals": intervals, "assessment": assessment})
        print(f"Compared {label}", flush=True)
    out = DEST / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    report = {"protocol": protocol, "results_zip_sha256": file_hash(archive_path),
              "label_note": "chronos_supported means the row's model wins; baseline_supported means its comparator wins",
              "results": uncertainty}
    (out / "uncertainty.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    summary = {f"{model}_h{horizon}": summarize(claim_table([
        r for r in uncertainty if r["model"] == model and r["comparator"] in ("weekly", "selected")]), horizon)
        for model in ("chronos2", "chronos_bolt_2048") for horizon in (24, 168)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved paired comparisons and uncertainty to {out}. Interpret only with the returned run metadata.")


README = """# Chronos-2 frozen-case experiment

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
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "compare"))
    parser.add_argument("--archive", type=Path, default=ROOT / "chronos2_results.zip")
    args = parser.parse_args()
    prepare() if args.step == "prepare" else compare(args.archive)
