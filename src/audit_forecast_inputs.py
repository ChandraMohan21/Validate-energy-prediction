"""Diagnose Portugal repetition and Ridge failures without changing benchmark outputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_pilot import file_hash, tsf_records
from trained_comparison import eligible_training, features, model_for

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/forecast_quality_audit"


def longest_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    if not len(starts):
        return None
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i])


def source_audit():
    archive = ROOT / "data/raw/baseline_pilot/electricity_hourly.zip"
    upstream = ROOT / "data/raw/baseline_pilot/electricity_lai_original.txt.gz"
    original = pd.read_csv(upstream, header=None, usecols=[0, 1, 2])
    rows = []
    for i, (meta, series, _) in enumerate(tsf_records(archive, 3)):
        np.testing.assert_array_equal(series.to_numpy(), original[i].to_numpy())
        selected = series.loc["2014"]
        processed = pd.read_csv(ROOT / f"data/processed/baseline_pilot/portugal_{meta['series_name']}_2014.csv",
            index_col="Datetime", parse_dates=True, float_precision="round_trip")["load"]
        np.testing.assert_array_equal(selected.to_numpy(), processed.to_numpy())
        difference = selected.to_numpy()[168:] - selected.to_numpy()[:-168]
        run = longest_run(difference == 0)
        yearly = []
        for year, values in series.groupby(series.index.year):
            lag = values.to_numpy()[168:] - values.to_numpy()[:-168]
            yearly.append({"year": int(year), "weekly_exact_fraction": float(np.mean(lag == 0)),
                "mean": float(values.mean()), "minimum": float(values.min()), "maximum": float(values.max())})
        rows.append({"series": meta["series_name"], "upstream_column_1based": i+1,
            "full_upstream_match": True, "processed_2014_match": True,
            "hours": len(selected), "integer_fraction": float(np.mean(selected == np.round(selected))),
            "weekly_exact_fraction": float(np.mean(difference == 0)),
            "weekly_lag_MAE": float(np.mean(np.abs(difference))),
            "longest_exact_lag168_run_hours": run[1]-run[0] if run else 0,
            "run_start": str(selected.index[run[0]+168]) if run else None,
            "run_end": str(selected.index[run[1]+167]) if run else None,
            "yearly": yearly})
    return {"monash_sha256": file_hash(archive), "upstream_sha256": file_hash(upstream),
        "upstream_url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz",
        "monash_page": "https://zenodo.org/records/4656140",
        "original_uci_page": "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
        "limitation": "Verified against Lai's processed hourly data, not the original 15-minute UCI meter file; T labels do not establish UCI meter identifiers",
        "series": rows}


def ridge_audit():
    fitted = json.loads((ROOT / "reports/trained_comparison/results.json").read_text())
    diagnostics = []
    feature_rows = []
    for details in fitted["cases"]:
        if details["horizon_hours"] != 168:
            continue
        window, dataset, name = details["window"], details["dataset"], details["series"]
        root = ROOT / "reports/expanded_chronos/inputs" / window
        report = json.loads((root / "reports/baseline_pilot/results.json").read_text())
        record = next(r for r in report["results"] if r["dataset"] == dataset and r["series"] == name and r["horizon_hours"] == 168)
        year = report["metadata"]["protocol"]["years"][dataset]
        series = pd.read_csv(root / f"data/processed/baseline_pilot/{dataset}_{name}_{year}.csv",
            index_col="Datetime", parse_dates=True, float_precision="round_trip")["load"]
        refit = eligible_training(series, record["split"]["validation_cutoff"])
        train = eligible_training(series, record["split"]["train_cutoff"])
        x, weekly, targets = features(series, refit, 168)
        model = model_for("ridge", details["choices"]["ridge"]["chosen_config"])
        model.fit(x, series.to_numpy()[targets].ravel() - weekly)
        with np.load(ROOT / f"reports/trained_comparison/predictions/{window}/{dataset}_{name}_h168.npz", allow_pickle=False) as saved:
            test_x, test_weekly, _ = features(series, saved["test_origins"], 168)
            prediction = model.predict(test_x) + test_weekly
            np.testing.assert_allclose(prediction, saved["ridge"].ravel(), rtol=1e-12, atol=1e-9)
            scaler, regression = model.steps[0][1], model.steps[1][1]
            contributions = scaler.transform(test_x) * regression.coef_
            base = {"window": window, "dataset": dataset, "series": name}
            for j, column in enumerate(x.columns):
                feature_rows.append({**base, "feature": column, "coefficient_standardized": float(regression.coef_[j]),
                    "refit_min": float(x[column].min()), "refit_max": float(x[column].max()),
                    "test_min": float(test_x[column].min()), "test_max": float(test_x[column].max()),
                    "test_outside_refit_range_fraction": float(np.mean((test_x[column] < x[column].min()) | (test_x[column] > x[column].max()))),
                    "mean_contribution": float(contributions[:, j].mean()),
                    "mean_absolute_contribution": float(np.abs(contributions[:, j]).mean())})
            diagnostics.append({**base, "ridge_reproduced": True,
                "ridge_test_MAE": float(np.mean(np.abs(prediction - saved["actual"].ravel()))),
                "weekly_test_MAE": float(np.mean(np.abs(test_weekly - saved["actual"].ravel()))),
                "ridge_validation_MAE": details["choices"]["ridge"]["validation_MAE"],
                "selected_baseline_validation_MAE": record["validation"][record["selected_on_validation"]]["MAE"],
                "training_origin_hours": sorted(int(v) for v in np.unique(series.index[train].hour)),
                "refit_origin_hours": sorted(int(v) for v in np.unique(series.index[refit].hour)),
                "test_origin_hours": sorted(int(v) for v in np.unique(series.index[saved["test_origins"]].hour)),
                "ridge_predicted_mean_weekly_correction": float((prediction - test_weekly).mean()),
                "actual_mean_weekly_correction": float((saved["actual"].ravel() - test_weekly).mean())})
        print(f"Audited Ridge {window}/{dataset}/{name}", flush=True)
    return diagnostics, feature_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = source_audit()
    diagnostics, feature_rows = ridge_audit()
    original_paths = [ROOT / "reports/trained_comparison/results.json", ROOT / "reports/trained_comparison/scores.csv",
                      ROOT / "expanded_chronos_results.zip"]
    result = {"status": "Post-result diagnostic audit; no benchmark predictions, scores, selection rules or exclusions changed",
        "source": source, "ridge_diagnostics": diagnostics,
        "original_files_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in original_paths}}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(feature_rows).to_csv(OUT / "ridge_feature_diagnostics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "ridge_case_diagnostics.csv", index=False)
    t3 = next(r for r in source["series"] if r["series"] == "T3")
    q2 = next(r for r in diagnostics if r["window"] == "Q2" and r["dataset"] == "tetouan" and r["series"] == "zone_1")
    contribution = next(r for r in feature_rows if r["window"] == "Q2" and r["dataset"] == "tetouan" and r["series"] == "zone_1" and r["feature"] == "context_mean_336")
    lines = ["# Forecast quality audit", "", "A diagnostic follow-up after reviewing results. Original benchmarks are preserved; no series is dropped and no model is selected using these test diagnostics.", "",
        "## Portugal T3", "",
        f"All three selected Monash series match the corresponding columns in Lai's upstream processed hourly dataset for all 26,304 observations. Their 2014 processed CSVs match Monash exactly. T3 has {100*t3['weekly_exact_fraction']:.3f}% exact agreement with the observation 168 hours earlier over 2014. Its longest uninterrupted equality run lasts {t3['longest_exact_lag168_run_hours']} hours, from {t3['run_start']} through {t3['run_end']}.", "",
        "The strong repetition was not introduced by this project's preprocessing. All three selected series already have integer values in the published hourly archive. Exact agreement has not been independently traced back to the original 15-minute UCI meter file, and no physical explanation for T3's repetition is established. Do not call it synthetic, corrupt, or representative of ordinary customers without additional evidence. Preserve it in the primary results and report its behavior separately; do not average percentage gains when baseline MAE is zero or nearly zero.", "",
        "[Monash source](https://zenodo.org/records/4656140) identifies this as a processed hourly version of [Lai's data](https://github.com/laiguokun/multivariate-time-series-data). [Original UCI documentation](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) describes meter creation zeros and daylight-saving conventions; this does not establish the cause of T3's pattern.", "",
        "## Ridge and training-origin coverage", "",
        "All 23 seven-day Ridge cases were reconstructed from the original choices and refit data and reproduce the saved predictions. The audit found a training-design limitation: selecting every 24th origin from midnight means all fitting/refit origins are at 00:00, while test origins span all 24 hours. The number of daily training origins was disclosed, but the resulting origin-hour coverage mismatch was not previously assessed. This can disadvantage models that use origin load features. It does not imply future-data leakage, but it limits claims about well-trained tabular alternatives.", "",
        f"For Q2 Tetouan Zone 1, Ridge MAE is {q2['ridge_test_MAE']:.2f}, versus weekly MAE {q2['weekly_test_MAE']:.2f}. The refitted linear model's standardized two-week average-load feature contributes an average {contribution['mean_contribution']:+.2f} to its residual correction. {100*contribution['test_outside_refit_range_fraction']:.1f}% of test rows have that feature outside its refit range. The total predicted mean correction is {q2['ridge_predicted_mean_weekly_correction']:+.2f}, while the actual mean correction was {q2['actual_mean_weekly_correction']:+.2f}.", "",
        "These are algebraic contribution and distribution-shift diagnostics, not a causal attribution: correlated linear features can offset one another. Both annual calendar features and rolling load features can be outside the range seen during fitting. A limited regularization grid and partial-year fitting also constrain this Ridge setup. The present results do not establish that Chronos beats a strong, broadly tuned linear forecasting approach.", "",
        "## Recommended next experiment", "",
        "Keep this run as version 1. Define a version 2 before running it: cover all origin hours in training; consider all eligible hourly origins or a deterministic balanced schedule; choose any feature/regularization variants using earlier validation only. Apply the same training-origin correction to both Ridge and LightGBM. Preserve the existing test dates and report that they have already been inspected. Strong confirmatory claims need new, untouched periods or datasets. Do not silently overwrite version 1 or remove T3 because a model loses there.", "",
        "Reproduce with `python src/audit_forecast_inputs.py`. The upstream gzip must be present at `data/raw/baseline_pilot/electricity_lai_original.txt.gz`. audit.json records its SHA256 and original-output hashes.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved audit: {OUT}")


if __name__ == "__main__":
    main()
