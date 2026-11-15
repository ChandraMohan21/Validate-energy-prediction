"""Matched, test-independent local Ridge/LightGBM comparison for the 60 frozen cases.

Run a short check with ``python src/trained_comparison_v3.py run --limit 1``.
Run or resume the full training with ``python src/trained_comparison_v3.py run``.
After the full run, use ``python src/trained_comparison_v3.py analyze``.
All files are separate from the earlier trained-model experiments.
"""

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
from chronos_uncertainty import calendar_losses, paired_interval
from trained_comparison import eligible_training, features


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "trained_comparison_v3"
INPUTS = ROOT / "reports" / "expanded_chronos" / "expanded_chronos_inputs.zip"
BOLT = ROOT / "long_context_chronos_results.zip"
CHRONOS2 = ROOT / "chronos2_results.zip"
WINDOWS = ("Q1", "Q2", "Q3", "Q4")
LOOKBACK_DAYS = (28, 84)
RIDGE_ALPHAS = (1.0, 100.0, 1000.0)
LIGHTGBM_LEAVES = (15, 31)
MAX_FIT_ORIGINS = 720
MIN_FIT_ORIGINS = 168
INNER_VALIDATION_ORIGINS = 168
FEATURE_DROP = ("dayofyear_sin", "dayofyear_cos")


def case_key(window, record):
    return f"{window}/{record['dataset']}_{record['series']}_h{record['horizon_hours']}.npz"


def split_origins(series, record, saved):
    """Q1 uses an inner time split; later cases use the frozen validation set."""
    horizon = int(record["horizon_hours"])
    if record["window"] == "Q1":
        pool = eligible_training(series, record["split"]["validation_cutoff"], stride=1)
        if len(pool) < MIN_FIT_ORIGINS + horizon + INNER_VALIDATION_ORIGINS:
            raise ValueError("Q1 has too little mature history for the inner split")
        validation = pool[-INNER_VALIDATION_ORIGINS:]
        training = pool[pool + horizon < validation[0]]
        selection_period = "last 168 eligible pre-test origins; earlier labels end before its first origin"
    else:
        training = eligible_training(series, record["split"]["train_cutoff"], stride=1)
        validation = saved["validation_origins"]
        selection_period = "frozen validation origins"
    refit = eligible_training(series, record["split"]["validation_cutoff"], stride=1)
    if len(training) < MIN_FIT_ORIGINS or len(refit) < MIN_FIT_ORIGINS:
        raise ValueError("Insufficient mature training labels")
    if series.index[training[-1] + horizon] >= series.index[validation[0]]:
        raise AssertionError("A training label reaches the validation period")
    if series.index[refit[-1] + horizon] >= series.index[saved["test_origins"][0]]:
        raise AssertionError("A refit label reaches the test period")
    if record["window"] != "Q1":
        np.testing.assert_array_equal(validation, saved["validation_origins"])
    return training, validation, refit, selection_period


def recent_sample(series, origins, days):
    end = series.index[int(origins[-1])]
    recent = origins[series.index[origins] >= end - pd.Timedelta(days=days)]
    if len(recent) < MIN_FIT_ORIGINS:
        recent = origins
    if len(recent) <= MAX_FIT_ORIGINS:
        return recent
    # Sample across time within each origin hour, preserving all 24 hours.
    picked = []
    for hour in range(24):
        group = recent[series.index[recent].hour == hour]
        quota = MAX_FIT_ORIGINS // 24
        if len(group):
            picked.extend(group[np.linspace(0, len(group) - 1, min(quota, len(group)), dtype=int)])
    result = np.unique(np.asarray(picked, dtype=int))
    if len(result) < MIN_FIT_ORIGINS:
        raise ValueError("Training sample became too small")
    return result


def design(series, origins, horizon):
    frame, weekly, target = features(series, origins, horizon)
    frame = frame.drop(columns=list(FEATURE_DROP))
    return frame, weekly, target


def make_model(family, setting):
    if family == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=float(setting)))
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=int(setting),
        min_child_samples=40, objective="regression_l1", reg_lambda=1.0,
        random_state=20261023, n_jobs=2, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )


def train_predict(family, setting, series, origins, horizon, predict_x, predict_weekly):
    fit_x, fit_weekly, fit_target = design(series, origins, horizon)
    residual = series.to_numpy()[fit_target].ravel() - fit_weekly
    if not np.isfinite(residual).all():
        raise ValueError("Nonfinite training target")
    model = make_model(family, setting)
    model.fit(fit_x, residual)
    prediction = model.predict(predict_x) + predict_weekly
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite trained prediction")
    return prediction


def fit_case(series, record, saved):
    training, validation, refit, selection_period = split_origins(series, record, saved)
    horizon = int(record["horizon_hours"])
    validation_x, validation_weekly, validation_target = design(series, validation, horizon)
    validation_actual = series.to_numpy()[validation_target].ravel()
    choices = {}
    for family, settings in (("ridge", RIDGE_ALPHAS), ("lightgbm", LIGHTGBM_LEAVES)):
        trials = []
        for days in LOOKBACK_DAYS:
            sampled = recent_sample(series, training, days)
            for setting in settings:
                prediction = train_predict(family, setting, series, sampled, horizon, validation_x, validation_weekly)
                trials.append({"lookback_days": days, "setting": setting,
                               "fit_origins": len(sampled),
                               "validation_MAE": float(np.mean(np.abs(prediction - validation_actual)))})
        best = min(trials, key=lambda row: row["validation_MAE"])
        choices[family] = {"best": best, "trials": trials}
    test_x, test_weekly, _ = design(series, saved["test_origins"], horizon)
    predictions = {}
    for family, choice in choices.items():
        best = choice["best"]
        sampled = recent_sample(series, refit, best["lookback_days"])
        prediction = train_predict(family, best["setting"], series, sampled, horizon, test_x, test_weekly)
        predictions[family] = prediction.reshape(saved["actual"].shape)
        choice["refit_origins"] = len(sampled)
    selected = min(choices, key=lambda family: choices[family]["best"]["validation_MAE"])
    detail = {"selection_period": selection_period, "train_origins": len(training),
              "validation_origins": len(validation), "refit_eligible_origins": len(refit),
              "train_last_target": str(series.index[training[-1] + horizon]),
              "validation_first_origin": str(series.index[validation[0]]),
              "refit_last_target": str(series.index[refit[-1] + horizon]),
              "test_first_origin": str(series.index[saved["test_origins"][0]]),
              "choices": choices, "selected_trained": selected}
    return predictions, detail


def identity():
    packages = {name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "lightgbm")}
    return {"script_sha256": file_hash(Path(__file__)),
            "feature_script_sha256": file_hash(ROOT / "src" / "trained_comparison.py"),
            "frozen_inputs_sha256": file_hash(INPUTS),
            "bolt_archive_sha256": file_hash(BOLT),
            "chronos2_archive_sha256": file_hash(CHRONOS2),
            "packages": packages,
            "settings": {"lookback_days": list(LOOKBACK_DAYS), "ridge_alphas": list(RIDGE_ALPHAS),
                         "lightgbm_leaves": list(LIGHTGBM_LEAVES),
                         "max_fit_origins": MAX_FIT_ORIGINS,
                         "q1_inner_validation_origins": INNER_VALIDATION_ORIGINS,
                         "features_removed": list(FEATURE_DROP)}}


def frozen_models(archives, window, record, saved):
    name = case_key(window, record)
    forecasts = {}
    for label, archive in archives.items():
        with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as result:
            for key in saved:
                np.testing.assert_array_equal(result[key], saved[key])
            forecast = result["chronos"]
            if forecast.shape != saved["actual"].shape or not np.isfinite(forecast).all():
                raise ValueError(f"Invalid {label} forecast: {name}")
            forecasts[label] = forecast
    return forecasts


def run(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    current = identity()
    identity_file = OUT / "identity.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != current:
        raise ValueError("Existing v3 output belongs to different code or inputs")
    identity_file.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    fingerprint = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
    started = time.monotonic()
    completed = 0
    with zipfile.ZipFile(BOLT) as bolt, zipfile.ZipFile(CHRONOS2) as chronos2:
        archives = {"bolt": bolt, "chronos2": chronos2}
        for window in WINDOWS:
            for record, series, saved in load_cases(ROOT / "reports" / "expanded_chronos" / "inputs" / window):
                record = {**record, "window": window}
                name = case_key(window, record)
                destination = OUT / "predictions" / name
                detail_file = destination.with_suffix(".json")
                destination.parent.mkdir(parents=True, exist_ok=True)
                frozen = frozen_models(archives, window, record, saved)
                if destination.exists() and detail_file.exists():
                    detail = json.loads(detail_file.read_text())
                    if detail["fingerprint"] != fingerprint:
                        raise ValueError(f"Cached configuration differs: {name}")
                    with np.load(destination, allow_pickle=False) as cache:
                        np.testing.assert_array_equal(cache["actual"], saved["actual"])
                        np.testing.assert_array_equal(cache["test_origins"], saved["test_origins"])
                        for label in frozen:
                            np.testing.assert_array_equal(cache[label], frozen[label])
                    print(f"Verified saved case {name}", flush=True)
                else:
                    if limit is not None and completed >= limit:
                        print(f"Stopped after {completed} cases as requested", flush=True)
                        return
                    print(f"Training {name} ({completed+1}/60)", flush=True)
                    predictions, detail = fit_case(series, record, saved)
                    detail = {"fingerprint": fingerprint, "window": window,
                              "dataset": record["dataset"], "series": record["series"],
                              "horizon_hours": record["horizon_hours"], **detail}
                    temporary = destination.with_suffix(".part")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, actual=saved["actual"],
                            test_origins=saved["test_origins"],
                            selected_seasonal=saved[str(saved["selected_on_validation"])],
                            weekly=saved["weekly"], selected_trained=predictions[detail["selected_trained"]],
                            **predictions, **frozen)
                    temporary.replace(destination)
                    detail_file.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
                    print(f"Saved {name}; elapsed {time.monotonic()-started:.0f}s", flush=True)
                completed += 1
    if completed != 60:
        raise AssertionError(f"Expected 60 cases, found {completed}")
    (OUT / "complete.json").write_text(json.dumps({"cases": completed, "fingerprint": fingerprint}, indent=2) + "\n")
    summarize()


def summarize():
    rows = []
    for detail_file in sorted((OUT / "predictions").glob("Q*/*.json")):
        detail = json.loads(detail_file.read_text())
        with np.load(detail_file.with_suffix(".npz"), allow_pickle=False) as saved:
            for method in ("ridge", "lightgbm", "selected_trained", "selected_seasonal", "weekly", "bolt", "chronos2"):
                rows.append({**{key: detail[key] for key in ("window", "dataset", "series", "horizon_hours")},
                             "method": method, "MAE": score(saved["actual"], saved[method])["MAE"]})
    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)


def analyze():
    completion = OUT / "complete.json"
    if not completion.exists() or json.loads(completion.read_text())["cases"] != 60:
        raise ValueError("Finish all 60 trained cases before uncertainty analysis")
    results = []
    for detail_file in sorted((OUT / "predictions").glob("Q*/*.json")):
        detail = json.loads(detail_file.read_text())
        label = case_key(detail["window"], detail)
        seed = 20261023 + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
        with np.load(detail_file.with_suffix(".npz"), allow_pickle=False) as saved:
            for model in ("bolt", "chronos2"):
                for comparator in ("ridge", "lightgbm", "selected_trained"):
                    grid = calendar_losses(saved["test_origins"], saved["actual"], saved[model], saved[comparator])
                    intervals = [paired_interval(grid, block, 5000, seed + block) for block in (168, 336)]
                    directions = {row["interval_direction"] for row in intervals}
                    if not all(row["at_least_two_blocks_of_valid_origins"] for row in intervals):
                        assessment = "limited_data"
                    elif directions == {"chronos"}:
                        assessment = "foundation_supported_both_lengths"
                    elif directions == {"baseline"}:
                        assessment = "trained_supported_both_lengths"
                    else:
                        assessment = "unclear_or_block_sensitive"
                    results.append({**{key: detail[key] for key in ("window", "dataset", "series", "horizon_hours")},
                                    "model": model, "comparator": comparator,
                                    "model_MAE": float(np.nanmean(grid[:, 0])),
                                    "trained_MAE": float(np.nanmean(grid[:, 1])),
                                    "assessment": assessment, "intervals": intervals})
        print(f"Analyzed {label}", flush=True)
    report = {"method": "Paired circular block bootstrap, 5000 resamples per block length; fixed forecasts",
              "blocks_hours": [168, 336], "multiple_testing_correction": False,
              "results": results}
    (OUT / "uncertainty.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(results).groupby(["horizon_hours", "model", "comparator", "assessment"]).size().to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--limit", type=int, help="Train at most this many new cases for a short check")
    commands.add_parser("analyze")
    args = parser.parse_args()
    if args.command == "run":
        run(args.limit)
    else:
        analyze()


if __name__ == "__main__":
    main()
