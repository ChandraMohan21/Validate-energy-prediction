"""Rerun the expanded Chronos cases with a 2048-hour history and compare with the 336-hour run.

    python src/long_context_study.py prepare   # inputs ZIP, protocol, Colab notebook
    python src/long_context_study.py compare   # after long_context_chronos_results.zip returns
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline_pilot import file_hash, score
from chronos_long_context import CONTEXT, REVISION, long_contexts
from chronos_pilot import load_cases

DEST = ROOT / "reports/long_context_chronos"
EXPANDED = ROOT / "reports/expanded_chronos"
INPUTS = EXPANDED / "expanded_chronos_inputs.zip"
SHORT_RESULTS = ROOT / "expanded_chronos_results.zip"
SHORT_UNCERTAINTY = ROOT / "reports/chronos_uncertainty/results.json"
LONG_RESULTS = ROOT / "long_context_chronos_results.zip"
BUNDLE = DEST / "long_context_chronos_inputs.zip"
NOTEBOOK = ROOT / "notebooks/05_long_context_chronos_colab.ipynb"
WINDOWS = ("Q1", "Q2", "Q3", "Q4")
# Earlier 2048-hour Tetouan run on the Q4 test window (reports/foundation_baseline_results.json).
EARLIER_Q4_MAE = {"zone_1": 695.7, "zone_2": 756.0, "zone_3": 543.9}
THRESHOLDS = (2, 5, 10)

PROTOCOL = {
    "status": "Written on 2026-10-20 before any 2048-hour forecast was run. The 336-hour results "
              "were already seen, so this is a planned follow-up, not an untouched confirmation.",
    "question": "Do the 336-hour conclusions hold when Chronos-Bolt gets up to its standard "
                "2048-hour history, in particular the cases where the validation-selected baseline "
                "turns a clear Chronos win into an unclear result?",
    "change": "History length 336 -> up to 2048 hours. Model commit, cases, test origins, "
              "baselines, blend weights, 0.5 quantile and bootstrap settings are unchanged.",
    "model_revision": REVISION,
    "context_policy": "Up to 2048 hours ending at the origin; all available history when the series "
                      "starts later (Q1); missing hours passed as NaN, which Chronos-Bolt masks.",
    "clear_result": "Paired circular-block bootstrap from src/chronos_uncertainty.py (5,000 resamples, "
                    "168- and 336-hour blocks, same seeds). Clear means both 95% intervals favour the "
                    "same method; otherwise unclear; household windows with too few blocks are limited.",
    "claim_change": "Chronos clearly beats weekly naive but not the validation-selected baseline, or the reverse.",
    "meaningful_improvement": "A clear Chronos win with an MAE gain of at least 5% against the comparator. "
                              "2% and 10% are reported as sensitivity checks.",
    "sanity_check": "Q4 Tetouan at 168 hours should be close to the earlier 2048-hour run on the same "
                    "test window: MAE about 696, 756 and 544 for zones 1 to 3.",
    "reporting": "Report both history lengths side by side and keep the 336-hour results. Report Q1 "
                 "separately because its history is shorter than 2048 hours. Q4 repeats the original "
                 "test period and is not new evidence. No multiple-comparison correction; exploratory.",
}


def case_name(basename):
    dataset, rest = basename.split("_", 1)
    series, horizon = rest.rsplit("_h", 1)
    return dataset, series, int(horizon[:-4])


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    coverage = []
    with zipfile.ZipFile(INPUTS) as source:
        for window in WINDOWS:
            folder = EXPANDED / "inputs" / window
            if (folder / "manifest.json").read_bytes() != source.read(window + "/manifest.json"):
                raise ValueError(f"Local {window} inputs differ from the uploaded ZIP")
            for record, series, saved in load_cases(folder):
                histories = long_contexts(series.to_numpy(), saved["test_origins"])
                lengths = np.array([len(h) for h in histories])
                coverage.append({"window": window, "dataset": record["dataset"], "series": record["series"],
                    "horizon_hours": record["horizon_hours"], "test_origins": len(lengths),
                    "min_history_hours": int(lengths.min()), "max_history_hours": int(lengths.max()),
                    "max_missing_hours": int(max(np.isnan(h).sum() for h in histories))})
        (DEST / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
        pd.DataFrame(coverage).to_csv(DEST / "history_coverage.csv", index=False)
        with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for member in source.infolist():
                bundle.writestr(member, source.read(member))
            bundle.write(ROOT / "src/chronos_long_context.py", "chronos_long_context.py")
    build_notebook(file_hash(BUNDLE))
    frame = pd.DataFrame(coverage)
    short = frame[frame.min_history_hours < CONTEXT]
    print(f"{len(frame)} cases; {len(short)} have under {CONTEXT} hours of history somewhere "
          f"(windows: {sorted(short.window.unique())}); {int((frame.max_missing_hours > 0).sum())} contain gaps")
    print(f"Bundle: {BUNDLE} ({BUNDLE.stat().st_size:,} bytes)\nNotebook: {NOTEBOOK}")


def build_notebook(digest):
    def cell(kind, text):
        body = {"cell_type": kind, "metadata": {}, "source": text.strip("\n")}
        return {**body, "execution_count": None, "outputs": []} if kind == "code" else body
    cells = [cell("markdown", """
# Chronos with a 2048-hour history

The earlier run gave Chronos only 336 hours (two weeks) of history. Chronos-Bolt
normally reads up to 2048 hours (about 12 weeks). This notebook repeats the same
60 forecast cases with the longer history. Nothing else changes: same model
commit, same test hours, same baselines.

Q1 starts in February, so its history is shorter than 2048 hours; it uses all
available history. Missing hours are passed as gaps, which Chronos ignores.

**Run:** Runtime > Change runtime type > T4 GPU. Run the cells in order.
Upload `long_context_chronos_inputs.zip` when asked. At the end, download
`long_context_chronos_results.zip` and put it in the project folder.
"""), cell("code", """
import os, subprocess, sys
from pathlib import Path

ENV = Path("/content/chronos_forecast_env")
PYTHON = str(ENV / "bin/python")
BOOTSTRAP = Path("/content/chronos_env_bootstrap")
# Isolated environment, so Colab's own packages cannot conflict.
RUN_ENV = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "PYTHONHOME")}
RUN_ENV["PYTHONNOUSERSITE"] = "1"
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
                "--target", str(BOOTSTRAP), "virtualenv"], env=RUN_ENV, check=True)
subprocess.run([sys.executable, "-m", "virtualenv", str(ENV)],
               env={**RUN_ENV, "PYTHONPATH": str(BOOTSTRAP)}, check=True)
subprocess.run([PYTHON, "-m", "pip", "install", "--quiet",
                "torch==2.11.0", "chronos-forecasting==2.2.2",
                "transformers==4.57.6", "numpy==2.1.3", "pandas==2.2.3"],
               env=RUN_ENV, check=True)
subprocess.run([PYTHON, "-m", "pip", "check"], env=RUN_ENV, check=True)
print("Setup complete")
"""), cell("code", """
subprocess.run([PYTHON, "-c", "import torch; from chronos import BaseChronosPipeline; "
                "assert torch.cuda.is_available(), 'Select a GPU runtime first'; "
                "print(torch.__version__, torch.cuda.get_device_name(0))"],
               env=RUN_ENV, check=True)
"""), cell("code", f"""
from google.colab import files
from pathlib import PurePosixPath
import hashlib, zipfile

ROOT = Path("/content/long_context_chronos")
ROOT.mkdir(exist_ok=True)
uploaded = files.upload()
assert len(uploaded) == 1, "Choose only long_context_chronos_inputs.zip"
filename, payload = next(iter(uploaded.items()))
assert hashlib.sha256(payload).hexdigest() == "{digest}", "Wrong or changed ZIP"
with zipfile.ZipFile(filename) as archive:
    for member in archive.infolist():
        parts = PurePosixPath(member.filename)
        assert not parts.is_absolute() and ".." not in parts.parts and "\\\\" not in member.filename
        assert (ROOT / member.filename).resolve().is_relative_to(ROOT.resolve())
    archive.extractall(ROOT)
del uploaded, payload
RUNNER = str(ROOT / "chronos_long_context.py")
for label in ("Q1", "Q2", "Q3", "Q4"):
    subprocess.run([PYTHON, RUNNER, "--root", str(ROOT / label), "--check-only"], env=RUN_ENV, check=True)
"""), cell("markdown", """
## Run all four periods

60 cases. Finished cases are saved under `results/Q1` etc. If the run stops,
run this cell again: finished cases are reused. Downloading partial results
before disconnecting keeps them.
"""), cell("code", """
OUTPUT = ROOT / "results"
for label in ("Q1", "Q2", "Q3", "Q4"):
    subprocess.run([PYTHON, "-u", RUNNER, "--root", str(ROOT / label),
                    "--output", str(OUTPUT / label), "--batch-size", "32"], env=RUN_ENV, check=True)
"""), cell("code", """
import csv
print("SEVEN-DAY RESULTS, 2048-hour history (positive = Chronos has lower error)")
for label in ("Q1", "Q2", "Q3", "Q4"):
    assert (OUTPUT / label / "complete.json").exists(), f"Incomplete window: {label}"
    with (OUTPUT / label / "comparison.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["baseline"] == "selected" and row["horizon_hours"] == "168":
                gain = row["chronos_MAE_gain_pct"].strip()
                gain = f"{float(gain):+.2f}%" if gain else "undefined (baseline MAE is zero)"
                print(f"{label} | {row['dataset']} | {row['series']} | {row['baseline_rule']} | "
                      f"Chronos MAE {float(row['chronos_MAE']):.6g} | gain {gain}")
print("Descriptive only. The uncertainty check runs locally.")
"""), cell("code", """
import shutil
result_zip = shutil.make_archive("/content/long_context_chronos_results", "zip", OUTPUT)
files.download(result_zip)
""")]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def claim_table(results):
    """Per case and comparator: assessment and gain, keyed by (window, dataset, series, horizon)."""
    table = {}
    for r in results:
        key = (r["window"], r["dataset"], r["series"], r["horizon_hours"])
        table.setdefault(key, {})[r["comparator"]] = (r["assessment"], r["gain_pct"])
    return table


def summarize(table, horizon):
    clear = "chronos_supported_both_lengths"
    rows = {k: v for k, v in table.items() if k[3] == horizon}
    out = {"cases": len(rows)}
    for comparator in ("weekly", "selected"):
        out[f"clear_vs_{comparator}"] = sum(v[comparator][0] == clear for v in rows.values())
        for t in THRESHOLDS:
            out[f"meaningful{t}_vs_{comparator}"] = sum(
                v[comparator][0] == clear and v[comparator][1] is not None and v[comparator][1] >= t
                for v in rows.values())
    out["weakened"] = sorted(k for k, v in rows.items() if v["weekly"][0] == clear and v["selected"][0] != clear)
    out["strengthened"] = sorted(k for k, v in rows.items() if v["weekly"][0] != clear and v["selected"][0] == clear)
    return out


def compare(archive_path):
    import chronos_uncertainty
    if json.loads((DEST / "protocol.json").read_text()) != PROTOCOL:
        raise ValueError("Protocol changed after preparation")
    out = DEST / "results"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(archive_path) as long, zipfile.ZipFile(SHORT_RESULTS) as short:
        for window in WINDOWS:
            identity = json.loads(long.read(window + "/run_metadata.json"))["identity"]
            if identity["context_hours"] != CONTEXT or identity["model_revision"] != REVISION:
                raise ValueError(f"{window}: not the planned 2048-hour run")
            names = sorted(n for n in short.namelist() if n.startswith(window + "/") and n.endswith(".npz"))
            if names != sorted(n for n in long.namelist() if n.startswith(window + "/") and n.endswith(".npz")):
                raise ValueError(f"{window}: case lists differ")
            for name in names:
                with np.load(io.BytesIO(short.read(name))) as a, np.load(io.BytesIO(long.read(name))) as b:
                    for key in ("test_origins", "actual", "weekly", "blend", "daily", "hybrid"):
                        np.testing.assert_array_equal(a[key], b[key])
                    rule = str(b["selected_on_validation"])
                    mae = {"weekly": score(b["actual"], b["weekly"])["MAE"],
                           "selected": score(b["actual"], b[rule])["MAE"],
                           "chronos_336": score(a["actual"], a["chronos"])["MAE"],
                           "chronos_2048": score(b["actual"], b["chronos"])["MAE"]}
                    dataset, series, horizon = case_name(name.split("/")[-1])
                    rows.append({"window": window, "dataset": dataset, "series": series,
                        "horizon_hours": horizon, "selected_rule": rule,
                        **{f"MAE_{k}": v for k, v in mae.items()},
                        "gain_vs_selected_336_pct": 100 * (1 - mae["chronos_336"] / mae["selected"]) if mae["selected"] else None,
                        "gain_vs_selected_2048_pct": 100 * (1 - mae["chronos_2048"] / mae["selected"]) if mae["selected"] else None,
                        "min_history_hours": int(b["context_hours"].min()),
                        "max_missing_hours": int(b["context_missing_hours"].max())})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "comparison_336_vs_2048.csv", index=False)
    chronos_uncertainty.analyze(archive_path, INPUTS, out / "uncertainty")
    long_table = claim_table(json.loads((out / "uncertainty/results.json").read_text())["results"])
    short_table = claim_table(json.loads(SHORT_UNCERTAINTY.read_text())["results"])
    summary = {f"{label}_h{h}": summarize(table, h)
               for label, table in (("336", short_table), ("2048", long_table)) for h in (24, 168)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=list), encoding="utf-8")
    write_report(frame, summary, out)
    print((out / "README.md").read_text(encoding="utf-8"))


def write_report(frame, summary, out):
    check = frame[(frame.window == "Q4") & (frame.dataset == "tetouan") & (frame.horizon_hours == 168)]
    lines = ["# Chronos with a 2048-hour history", "",
        "Same 60 cases as the 336-hour run; only the history length differs. Protocol written before "
        "this run: `../protocol.json`. Q1 history is shorter than 2048 hours. Q4 repeats the original "
        "test period. Exploratory; no multiple-comparison correction.", "",
        "## Sanity check: Q4 Tetouan, 168 hours", "",
        "| Zone | 2048-hour MAE now | Earlier 2048-hour run |", "|---|---:|---:|"]
    for _, r in check.iterrows():
        lines.append(f"| {r['series']} | {r['MAE_chronos_2048']:.1f} | {EARLIER_Q4_MAE[r['series']]:.1f} |")
    lines += ["", "## How often the claim holds", "",
        "Clear = both bootstrap block lengths favour Chronos. Meaningful = clear and at least 5% lower MAE.", "",
        "| History | Horizon | Cases | Clear vs weekly | Clear vs selected | Meaningful vs weekly | Meaningful vs selected | Weakened by fair baseline |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, s in summary.items():
        history, horizon = key.split("_h")
        lines.append(f"| {history} h | {horizon} h | {s['cases']} | {s['clear_vs_weekly']} | {s['clear_vs_selected']} | "
                     f"{s['meaningful5_vs_weekly']} | {s['meaningful5_vs_selected']} | {len(s['weakened'])} |")
    lines += ["", "Sensitivity at 2% and 10% and the exact weakened cases are in `summary.json`.", "",
        "## Seven-day MAE against the validation-selected baseline", "",
        "| Window | Series | Selected | Chronos 336 h | Chronos 2048 h | Gain 336 h | Gain 2048 h |",
        "|---|---|---:|---:|---:|---:|---:|"]
    fmt = lambda g: "undefined" if pd.isna(g) else f"{g:+.1f}%"
    for _, r in frame[frame.horizon_hours == 168].iterrows():
        lines.append(f"| {r['window']} | {r['dataset']}/{r['series']} | {r['MAE_selected']:.4g} | "
                     f"{r['MAE_chronos_336']:.4g} | {r['MAE_chronos_2048']:.4g} | "
                     f"{fmt(r['gain_vs_selected_336_pct'])} | {fmt(r['gain_vs_selected_2048_pct'])} |")
    lines += ["", "Reproduce with `python src/long_context_study.py compare`.", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("prepare", "compare"))
    parser.add_argument("--archive", type=Path, default=LONG_RESULTS)
    args = parser.parse_args()
    prepare() if args.step == "prepare" else compare(args.archive)
