"""Ridge and LightGBM follow-up on frozen Chronos cases, without test-based tuning."""
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

from baseline_pilot import file_hash, origin_indices, reference_predictions, score
from chronos_pilot import load_cases

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/trained_comparison"


def eligible_training(series, cutoff, stride=24):
    """History and label windows fully observed; every label ends by cutoff."""
    values = series.to_numpy()
    origins = np.arange(336, len(series) - 168)
    missing = np.r_[0, np.cumsum(~np.isfinite(values))]
    good = ((missing[origins + 1] - missing[origins - 335] == 0)
            & (missing[origins + 169] - missing[origins + 1] == 0))
    cutoff_position = series.index.searchsorted(pd.Timestamp(cutoff), side="right") - 1
    return origins[good & (origins + 168 <= cutoff_position) & (origins % stride == 0)]


def features(series, origins, horizon):
    """Calendar plus load features from only the last 336 observed hours."""
    if not 1 <= horizon <= 168:
        raise ValueError("Horizon must be 1..168")
    values = series.to_numpy()
    origins = np.asarray(origins, int)
    if not len(origins) or (origins < 335).any():
        raise ValueError("Insufficient context")
    steps = np.arange(1, horizon + 1)
    target = origins[:, None] + steps
    daily = origins[:, None] + (steps - 1) % 24 - 23
    weekly = target - 168
    previous_week = target - 336
    for source in (daily, weekly, previous_week):
        if not ((source >= 0) & (source <= origins[:, None])).all():
            raise AssertionError("Unobserved source value")
    context = values[origins[:, None] + np.arange(-335, 1)]
    if not np.isfinite(context).all():
        raise ValueError("Missing context")
    shape = target.shape
    data = {
        "horizon": np.broadcast_to(steps / 168, shape),
        "daily": values[daily], "weekly": values[weekly],
        "previous_week": values[previous_week],
        "daily_minus_weekly": values[daily] - values[weekly],
        "weekly_change": values[weekly] - values[previous_week],
    }
    for lag in (0, 1, 23, 24, 167, 335):
        data[f"origin_lag_{lag}"] = np.broadcast_to(values[origins - lag, None], shape)
    for width in (24, 168, 336):
        for name, statistic in (("mean", np.mean), ("std", np.std)):
            data[f"context_{name}_{width}"] = np.broadcast_to(statistic(context[:, -width:], axis=1)[:, None], shape)
    times = series.index[target.ravel()]
    for name, numbers, period in (("hour", times.hour, 24), ("weekday", times.dayofweek, 7), ("dayofyear", times.dayofyear, 365.25)):
        data[name + "_sin"] = np.sin(2 * np.pi * numbers.to_numpy().reshape(shape) / period)
        data[name + "_cos"] = np.cos(2 * np.pi * numbers.to_numpy().reshape(shape) / period)
    data["weekend"] = (times.dayofweek.to_numpy().reshape(shape) >= 5).astype(float)
    frame = pd.DataFrame({key: value.ravel() for key, value in data.items()})
    return frame, values[weekly].ravel(), target


def model_for(family, value):
    if family == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=value, solver="cholesky"))
    if family == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=value,
            min_child_samples=30, objective="regression_l1", reg_lambda=1.0,
            random_state=42, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)
    raise ValueError(family)


def fit_predict(family, config, x, residual, future_x, weekly):
    model = model_for(family, config)
    model.fit(x, residual)
    # Neither clipping nor any other postprocessing is chosen from test results.
    return model.predict(future_x) + weekly


def run_case(series, record, saved, train, refit, grids):
    horizon = record["horizon_hours"]
    values = series.to_numpy()
    tx, tw, tt = features(series, train, horizon)
    vx, vw, _ = features(series, saved["validation_origins"], horizon)
    residual = values[tt].ravel() - tw
    choices = {}
    # Finish every choice using validation before building the test feature frame.
    for family, configurations in grids.items():
        trials = []
        for config in configurations:
            prediction = fit_predict(family, config, tx, residual, vx, vw)
            loss = float(np.mean(np.abs(prediction - saved["validation_actual"].ravel())))
            trials.append({"config": config, "validation_MAE": loss})
        best = min(trials, key=lambda trial: trial["validation_MAE"])
        choices[family] = {"chosen_config": best["config"], "validation_MAE": best["validation_MAE"], "trials": trials}
    del tx, vx
    rx, rw, rt = features(series, refit, horizon)
    test_x, test_weekly, _ = features(series, saved["test_origins"], horizon)
    residual = values[rt].ravel() - rw
    predictions = {}
    for family in grids:
        predictions[family] = fit_predict(family, choices[family]["chosen_config"], rx, residual,
            test_x, test_weekly).reshape(saved["actual"].shape)
        if not np.isfinite(predictions[family]).all():
            raise ValueError("Nonfinite trained predictions")
    selected = min(choices, key=lambda name: choices[name]["validation_MAE"])
    predictions["selected_trained"] = predictions[selected]
    return predictions, choices, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Optional number of eligible cases for a smoke run")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"created_date": "2026-10-20", "study_status": "Follow-up after observing Chronos results",
        "models": {"ridge": [1.0, 100.0], "lightgbm": [15, 31]},
        "ridge_grid_parameter": "alpha", "lightgbm_grid_parameter": "num_leaves",
        "training_origin_stride_hours": 24, "minimum_training_origins": 7,
        "test_and_validation_origins": "Exact frozen hourly origins from expanded Chronos cases",
        "history_hours": 336, "horizons_hours": [24, 168],
        "training_boundary": "All targets through 168h end by train_cutoff, before validation origins",
        "refit_boundary": "After tuning, refit on all eligible daily origins with targets through 168h ending by validation_cutoff, before first test origin",
        "target": "Residual from target-aligned weekly naive; add weekly prediction back at inference",
        "features": "Origin load lags and 24/168/336h statistics, daily/weekly/prior-week target-aligned loads, differences, horizon, cyclic hour/weekday/year and weekend",
        "selection": "Validation MAE only; Ridge scaler fitted only on training; newly fitted on refit data",
        "training_budget": "Daily training origins reduce duplicated labels and compute; evaluation remains hourly; no weather inputs",
        "fixed_lightgbm": {"n_estimators": 200, "learning_rate": 0.05, "min_child_samples": 30, "objective": "regression_l1", "reg_lambda": 1.0, "random_state": 42},
        "missing_data": "No filling; require complete 336h history and 168h future during training; report excluded cases",
        "caveats": ["Q1 has no earlier fitting data under frozen split; do not move its boundaries", "Later windows may train on earlier test-period observations once those observations are historical", "Not an exhaustive model search or proof about all trained models", "Trained-model uncertainty not calculated by this script"]}
    protocol_path = OUT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    identity = {"protocol_sha256": file_hash(protocol_path), "script_sha256": file_hash(Path(__file__)),
        "chronos_zip_sha256": file_hash(ROOT / "expanded_chronos_results.zip"),
        "inputs_zip_sha256": file_hash(ROOT / "reports/expanded_chronos/expanded_chronos_inputs.zip"),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "lightgbm")}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    rows, exclusions, metadata = [], [], []
    started = time.monotonic()
    finished_cases = 0
    with zipfile.ZipFile(ROOT / "expanded_chronos_results.zip") as chronos:
        for window in ("Q1", "Q2", "Q3", "Q4"):
            root = ROOT / "reports/expanded_chronos/inputs" / window
            cases = load_cases(root)
            for record, series, saved in cases:
                split = record["split"]
                train = eligible_training(series, split["train_cutoff"])
                refit = eligible_training(series, split["validation_cutoff"])
                base = {"window": window, "dataset": record["dataset"], "series": record["series"], "horizon_hours": record["horizon_hours"]}
                if len(train) < protocol["minimum_training_origins"]:
                    exclusions.append({**base, "training_origins": len(train), "reason": "Insufficient earlier training origins under frozen split"})
                    continue
                if args.limit is not None and finished_cases >= args.limit:
                    continue
                name = f"{record['dataset']}_{record['series']}_h{record['horizon_hours']}.npz"
                with np.load(io.BytesIO(chronos.read(window + "/" + name)), allow_pickle=False) as existing:
                    for key in saved:
                        np.testing.assert_array_equal(saved[key], existing[key])
                    chronos_prediction = existing["chronos"]
                destination = OUT / "predictions" / window / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                details_path = destination.with_suffix(".json")
                if destination.exists() and details_path.exists():
                    details = json.loads(details_path.read_text())
                    if details["fingerprint"] != fingerprint:
                        raise ValueError("Cached trained output configuration differs; use a separate output for changed experiments")
                    with np.load(destination, allow_pickle=False) as cache:
                        predictions = {k: cache[k] for k in ("ridge", "lightgbm", "selected_trained")}
                        np.testing.assert_array_equal(cache["actual"], saved["actual"])
                else:
                    print(f"Training {window}/{name} ({len(train)} fitting / {len(refit)} refit origins)", flush=True)
                    predictions, choices, selected = run_case(series, record, saved, train, refit, protocol["models"])
                    details = {**base, "fingerprint": fingerprint, "choices": choices, "selected_trained": selected,
                        "train_origins": len(train), "refit_origins": len(refit),
                        "train_last_target": str(series.index[train[-1] + 168]),
                        "validation_first_origin": str(series.index[saved["validation_origins"][0]]),
                        "refit_last_target": str(series.index[refit[-1] + 168]),
                        "test_first_origin": str(series.index[saved["test_origins"][0]])}
                    assert pd.Timestamp(details["train_last_target"]) < pd.Timestamp(details["validation_first_origin"])
                    assert pd.Timestamp(details["refit_last_target"]) < pd.Timestamp(details["test_first_origin"])
                    temp = destination.with_suffix(".part")
                    with temp.open("wb") as stream:
                        np.savez_compressed(stream, **saved, **predictions, chronos=chronos_prediction)
                    temp.replace(destination)
                    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
                metadata.append(details)
                for method, prediction in {**predictions, "chronos": chronos_prediction,
                    "selected_baseline": saved[record["selected_on_validation"]], "weekly": saved["weekly"]}.items():
                    rows.append({**base, "method": method, "unit": record["unit"], **score(saved["actual"], prediction)})
                finished_cases += 1
                pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
                print(f"Completed {finished_cases} cases; elapsed {time.monotonic()-started:.0f}s", flush=True)
    result = {"identity": identity, "protocol": protocol, "complete": args.limit is None,
        "completed_cases": finished_cases, "exclusions": exclusions, "cases": metadata}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    report(rows, result)


def report(rows, result):
    frame = pd.DataFrame(rows)
    lines = ["# Trained-model comparison", "", f"Completed {result['completed_cases']} cases. Full run: {result['complete']}. Excluded {len(result['exclusions'])} cases with insufficient earlier training data.", "",
        "Ridge and LightGBM predict residuals from weekly naive using load/calendar features. Two settings per family are chosen on earlier validation MAE. Each final model is refitted using mature historical targets only, then remains fixed over the test period. Training uses daily origins; all methods are evaluated on the frozen hourly test origins. Q1 remains excluded rather than moving split boundaries. Every prediction and the training/validation boundaries are saved.", "",
        "These are descriptive errors, not significance tests. This follow-up was designed after seeing Chronos results. It is a compact trained-model reference, not an exhaustive search. Do not compare raw MAE across different units.", "",
        "## Seven-day MAE", "", "| Period | Series | Ridge | LightGBM | Chronos | Selected seasonal baseline |", "|---|---|---:|---:|---:|---:|"]
    if not frame.empty:
        seven = frame[frame.horizon_hours == 168].pivot(index=["window", "dataset", "series"], columns="method", values="MAE")
        for index, row in seven.iterrows():
            lines.append(f"| {index[0]} | {index[1]}/{index[2]} | {row['ridge']:.4g} | {row['lightgbm']:.4g} | {row['chronos']:.4g} | {row['selected_baseline']:.4g} |")
    lines += ["", "Reproduce: `python src/trained_comparison.py`. Existing outputs resume only with the same code, protocol, inputs and package versions.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    if not frame.empty:
        print(frame.pivot_table(index=["horizon_hours", "method"], values="MAE", aggfunc="count").to_string())


if __name__ == "__main__":
    main()
