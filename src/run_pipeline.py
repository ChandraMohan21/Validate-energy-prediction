"""
Full modeling pipeline on the corrected feature set.

Stages, per zone:
  1. FLAML AutoML over LightGBM / XGBoost / Random Forest / Extra Trees, scored on
     the validation split only (never on test).
  2. Train the winning configuration on train, score on val and test.
  3. Refit the same frozen configuration on train + validation, score on test.
     This is the headline number.
  4. Compare against the seasonal-naive and persistence baselines on the same rows.
  5. Multi-window walk-forward validation, three non-overlapping 42-day windows,
     each fit on everything strictly before the window with the same 168h embargo.

Splits carry a 168-hour embargo so no split's targets reach into the next one.
See src/forecasting.py for the corrections this replaces.

Run: python src/run_pipeline.py
"""

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting import (ZONES, HORIZON, EMBARGO, TRAIN_CUTOFF, VAL_CUTOFF, FEATURES,
                         load_hourly, build_supervised_table, split, metrics,
                         baseline_metrics, effective_n, describe_split)

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
RESULTS = Path("reports/results_corrected.json")
TIME_BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 300

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def automl_search(train, val, budget):
    """Return (family name, the fitted sklearn-API estimator FLAML selected)."""
    from flaml import AutoML
    a = AutoML()
    a.fit(
        X_train=train[FEATURES], y_train=train["y"].values,
        X_val=val[FEATURES], y_val=val["y"].values,
        task="regression", metric="mape", time_budget=budget,
        estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
        eval_method="holdout", seed=42, verbose=0,
    )
    return a.best_estimator, a.model.estimator


def fresh(estimator):
    """Unfitted copy of an estimator with identical hyperparameters.

    Cloning FLAML's own fitted estimator avoids re-deriving parameters from
    `best_config`, whose names are FLAML's (e.g. `max_leaves`) and do not always
    match the underlying sklearn constructor (`max_leaf_nodes`).
    """
    from sklearn.base import clone
    return clone(estimator)


def main():
    hdf = load_hourly()
    out = {"time_budget_s": TIME_BUDGET, "zones": {}, "multi_window": []}

    for col, label in ZONES.items():
        slug = label.replace(" ", "_").lower()
        t0 = time.time()
        print(f"\n{'='*60}\n{label}\n{'='*60}", flush=True)

        data = build_supervised_table(hdf, col)
        train, val, test = split(data)
        if col == list(ZONES)[0]:
            out["split_description"] = json.loads(
                describe_split(train, val, test).to_json(orient="records", date_format="iso"))

        best_est, best_model = automl_search(train, val, TIME_BUDGET)
        best_cfg = best_model.get_params()
        print(f"  AutoML picked: {best_est}", flush=True)
        print(f"  config: {best_cfg}", flush=True)

        # stage 2: train-only fit
        m = fresh(best_model)
        m.fit(train[FEATURES], train["y"])
        val_m = metrics(val["y"], m.predict(val[FEATURES]))
        test_m = metrics(test["y"], m.predict(test[FEATURES]))
        joblib.dump(m, MODELS_DIR / f"{slug}_final_automl.joblib")

        # stage 3: refit on train+val, frozen config
        trval = pd.concat([train, val])
        mr = fresh(best_model)
        mr.fit(trval[FEATURES], trval["y"])
        test_r = metrics(test["y"], mr.predict(test[FEATURES]))
        joblib.dump(mr, MODELS_DIR / f"{slug}_final_refit_trainval.joblib")

        base_test = baseline_metrics(test)
        base_val = baseline_metrics(val)

        out["zones"][label] = {
            "model_family": best_est,
            "config": {k: (float(v) if isinstance(v, (np.floating,)) else
                           int(v) if isinstance(v, (np.integer,)) else v)
                       for k, v in best_cfg.items() if isinstance(v, (int, float, str, bool, type(None), np.integer, np.floating))},
            "val": val_m, "test_train_only": test_m, "test_refit": test_r,
            "baselines_test": base_test, "baselines_val": base_val,
            "effective_n_test": effective_n(test),
            "rows_test": int(len(test)),
            "beats_seasonal_naive_refit": bool(test_r["MAPE"] < base_test["seasonal_naive"]["MAPE"]),
        }
        print(f"  val MAPE            {val_m['MAPE']:.2f}%", flush=True)
        print(f"  test (train-only)   {test_m['MAPE']:.2f}%", flush=True)
        print(f"  test (refit)        {test_r['MAPE']:.2f}%", flush=True)
        print(f"  seasonal naive      {base_test['seasonal_naive']['MAPE']:.2f}%", flush=True)
        print(f"  persistence         {base_test['persistence']['MAPE']:.2f}%", flush=True)
        print(f"  beats naive: {out['zones'][label]['beats_seasonal_naive_refit']}"
              f"   ({time.time()-t0:.0f}s)", flush=True)

        # stage 5: multi-window walk-forward, same frozen config, with embargo
        max_t = hdf.index.max()
        for i in range(3):
            start = max_t - pd.Timedelta(days=42 * (i + 1))
            end = max_t - pd.Timedelta(days=42 * i)
            w_tr = data[data["origin_time"] <= start - EMBARGO]
            w_te = data[(data["origin_time"] > start) & (data["origin_time"] <= end)]
            if len(w_te) == 0 or len(w_tr) == 0:
                continue
            wm = fresh(best_model)
            wm.fit(w_tr[FEATURES], w_tr["y"])
            wmm = metrics(w_te["y"], wm.predict(w_te[FEATURES]))
            wb = baseline_metrics(w_te)
            out["multi_window"].append({
                "zone": label, "window": i + 1,
                "period": f"{start.date()} to {end.date()}",
                "model_MAPE": wmm["MAPE"],
                "seasonal_naive_MAPE": wb["seasonal_naive"]["MAPE"],
                "persistence_MAPE": wb["persistence"]["MAPE"],
                "beats_seasonal_naive": bool(wmm["MAPE"] < wb["seasonal_naive"]["MAPE"]),
                "effective_n": effective_n(w_te),
            })
            print(f"    W{i+1} {start.date()}..{end.date()}  model {wmm['MAPE']:.2f}%  "
                  f"snaive {wb['seasonal_naive']['MAPE']:.2f}%  "
                  f"beats={wmm['MAPE'] < wb['seasonal_naive']['MAPE']}", flush=True)
            del w_tr, w_te, wm
            gc.collect()

        del data, train, val, test, trval, m, mr
        gc.collect()

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {RESULTS}")

    wins = sum(r["beats_seasonal_naive"] for r in out["multi_window"])
    print(f"Multi-window: model beats seasonal naive in {wins}/{len(out['multi_window'])} combinations")


if __name__ == "__main__":
    main()
