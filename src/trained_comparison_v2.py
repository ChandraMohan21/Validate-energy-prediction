"""Version 2: all eligible hourly training origins, with the v1 features and grids."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_pilot import file_hash, score
from chronos_pilot import load_cases
from trained_comparison import eligible_training, run_case

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/trained_comparison_v2"


def coverage(series, origins):
    hours, counts = np.unique(series.index[origins].hour, return_counts=True)
    return {str(int(hour)): int(count) for hour, count in zip(hours, counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Optional smoke-run case limit; output remains marked partial")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    v1_path = ROOT / "reports/trained_comparison/results.json"
    v1 = json.loads(v1_path.read_text())
    protocol = {**v1["protocol"], "version": 2,
        "study_status": "Diagnostic follow-up after inspecting v1 test errors; test periods are not untouched",
        "training_origin_stride_hours": 1, "minimum_training_origins": 168,
        "change_from_v1": "Use every eligible hourly training/refit origin instead of midnight-only daily origins; features, model settings grids, test cases and MAE selection remain unchanged",
        "training_budget": "All eligible hourly origins; duplicate target hours across forecast horizons are training rows, not independent statistical evidence",
        "refit_boundary": "All eligible hourly origins with targets through 168h ending by validation_cutoff, before first test origin",
        "caveats": [*v1["protocol"]["caveats"], "Corrects origin-hour coverage but does not resolve all model specification or extrapolation limitations"]}
    protocol_path = OUT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    identity = {"protocol_sha256": file_hash(protocol_path), "runner_sha256": file_hash(Path(__file__)),
        "training_module_sha256": file_hash(Path(__file__).with_name("trained_comparison.py")),
        "v1_results_sha256": file_hash(v1_path),
        "v1_scores_sha256": file_hash(ROOT / "reports/trained_comparison/scores.csv"),
        "chronos_zip_sha256": file_hash(ROOT / "expanded_chronos_results.zip"),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "lightgbm")}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    metadata_path = OUT / "identity.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != identity:
        raise ValueError("Existing v2 configuration differs; preserve outputs and use another version")
    metadata_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    rows, details, exclusions = [], [], []
    started = time.monotonic()
    with zipfile.ZipFile(ROOT / "expanded_chronos_results.zip") as chronos:
        for window in ("Q1", "Q2", "Q3", "Q4"):
            root = ROOT / "reports/expanded_chronos/inputs" / window
            for record, series, saved in load_cases(root):
                base = {"window": window, "dataset": record["dataset"], "series": record["series"], "horizon_hours": record["horizon_hours"]}
                train = eligible_training(series, record["split"]["train_cutoff"], stride=1)
                refit = eligible_training(series, record["split"]["validation_cutoff"], stride=1)
                if len(train) < protocol["minimum_training_origins"]:
                    exclusions.append({**base, "training_origins": len(train), "reason": "Insufficient earlier training under frozen split"})
                    continue
                if args.limit is not None and len(details) >= args.limit:
                    continue
                train_hours, refit_hours = coverage(series, train), coverage(series, refit)
                if len(train_hours) != 24 or len(refit_hours) != 24:
                    raise ValueError("Eligible case does not cover all 24 origin hours")
                name = f"{record['dataset']}_{record['series']}_h{record['horizon_hours']}.npz"
                with np.load(io.BytesIO(chronos.read(window + "/" + name)), allow_pickle=False) as existing:
                    for key in saved:
                        np.testing.assert_array_equal(saved[key], existing[key])
                    chronos_prediction = existing["chronos"]
                destination = OUT / "predictions" / window / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                detail_path = destination.with_suffix(".json")
                if destination.exists() and detail_path.exists():
                    detail = json.loads(detail_path.read_text())
                    if detail["fingerprint"] != fingerprint:
                        raise ValueError("Cached output differs from current v2 setup")
                    with np.load(destination, allow_pickle=False) as cache:
                        predictions = {key: cache[key] for key in ("ridge", "lightgbm", "selected_trained")}
                        np.testing.assert_array_equal(cache["actual"], saved["actual"])
                else:
                    print(f"Training v2 {window}/{name}: {len(train)} fitting, {len(refit)} refit origins; all 24 hours", flush=True)
                    predictions, choices, selected = run_case(series, record, saved, train, refit, protocol["models"])
                    detail = {**base, "fingerprint": fingerprint, "choices": choices, "selected_trained": selected,
                        "train_origins": len(train), "refit_origins": len(refit),
                        "train_origin_hour_counts": train_hours, "refit_origin_hour_counts": refit_hours,
                        "train_last_target": str(series.index[train[-1] + 168]),
                        "validation_first_origin": str(series.index[saved["validation_origins"][0]]),
                        "refit_last_target": str(series.index[refit[-1] + 168]),
                        "test_first_origin": str(series.index[saved["test_origins"][0]])}
                    if not (pd.Timestamp(detail["train_last_target"]) < pd.Timestamp(detail["validation_first_origin"])
                            and pd.Timestamp(detail["refit_last_target"]) < pd.Timestamp(detail["test_first_origin"])):
                        raise AssertionError("Fitting labels cross evaluation boundaries")
                    temporary = destination.with_suffix(".part")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, **saved, **predictions, chronos=chronos_prediction)
                    temporary.replace(destination)
                    detail_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
                details.append(detail)
                for method, prediction in {**predictions, "chronos": chronos_prediction,
                    "selected_baseline": saved[record["selected_on_validation"]], "weekly": saved["weekly"]}.items():
                    rows.append({**base, "method": method, "unit": record["unit"], **score(saved["actual"], prediction)})
                pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
                print(f"Completed v2 {len(details)} cases; elapsed {time.monotonic()-started:.0f}s", flush=True)
    result = {"identity": identity, "protocol": protocol, "complete": args.limit is None,
        "completed_cases": len(details), "exclusions": exclusions, "cases": details}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    write_report(result, rows)


def write_report(result, rows):
    frame = pd.DataFrame(rows)
    old = pd.read_csv(ROOT / "reports/trained_comparison/scores.csv")
    changes = frame[frame.method.isin(["ridge", "lightgbm"])].merge(
        old[old.method.isin(["ridge", "lightgbm"])], on=["window", "dataset", "series", "horizon_hours", "method"],
        suffixes=("_v2", "_v1"), validate="one_to_one")
    changes["v2_MAE_reduction_vs_v1_pct"] = np.where(changes.MAE_v1 > 0, 100*(1-changes.MAE_v2/changes.MAE_v1), np.nan)
    changes.to_csv(OUT / "v1_v2_comparison.csv", index=False)
    lines = ["# Version 2: hourly training origins", "",
        f"Completed {result['completed_cases']} cases. Full run: {result['complete']}. Excluded {len(result['exclusions'])} Q1 cases under the frozen split.", "",
        "Every eligible hourly origin is used for fitting and refitting. All included cases cover all 24 origin hours. Model families, features, hyperparameter grids and validation/test origins are unchanged from v1. Settings are chosen on validation MAE only, then refitted before the first test origin. Both versions and every prediction are preserved.", "",
        "This is a diagnostic correction after inspecting v1 test results. Test periods are not untouched confirmation data. V2 corrects hour coverage but retains the limited grids and partial-year/extrapolation limitations. No v2 uncertainty calculations are performed by the training script. Do not reuse the v1 uncertainty intervals for v2.", "",
        "## Seven-day MAE", "", "| Period | Series | Ridge v1 | Ridge v2 | LightGBM v1 | LightGBM v2 | Chronos | Seasonal choice |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    if not frame.empty:
        seven = frame[frame.horizon_hours == 168].pivot(index=["window", "dataset", "series"], columns="method", values="MAE")
        previous = old[old.horizon_hours == 168].pivot(index=["window", "dataset", "series"], columns="method", values="MAE")
        for index, row in seven.iterrows():
            prior = previous.loc[index]
            lines.append(f"| {index[0]} | {index[1]}/{index[2]} | {prior['ridge']:.4g} | {row['ridge']:.4g} | {prior['lightgbm']:.4g} | {row['lightgbm']:.4g} | {row['chronos']:.4g} | {row['selected_baseline']:.4g} |")
    lines += ["", "Reproduce with `python src/trained_comparison_v2.py`. The run resumes saved cases only if the code, v1 results, inputs and package versions match.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved v2: {OUT}")


if __name__ == "__main__":
    main()
