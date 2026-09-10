"""
Compare two ways of using the same model on the same corrected features.

  direct    : model predicts y
  residual  : model predicts (y - seasonal_naive), prediction = seasonal_naive + output

The seasonal naive is a strong baseline on this data (about 3-9% MAPE depending on
zone). When a baseline is that strong, predicting the level directly forces the model
to rediscover the whole weekly shape before it can add anything. Modelling the
deviation from the baseline starts it at the baseline and asks only for the
correction, which is the standard framing in load forecasting.

Both variants use identical features, identical splits with the 168h embargo, and
identical hyperparameters, so the only thing that differs is the target.

Run: python src/residual_variant.py
"""

import gc
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting import (ZONES, FEATURES, load_hourly, build_supervised_table,
                         split, metrics, baseline_metrics, effective_n)

OUT = Path("reports/residual_variant_results.json")

PARAMS = dict(n_estimators=600, num_leaves=63, learning_rate=0.05,
              min_child_samples=40, colsample_bytree=0.8, subsample=0.9,
              subsample_freq=1, reg_lambda=1.0,
              random_state=42, verbosity=-1, n_jobs=-1)


def main():
    hdf = load_hourly()
    rows = []
    for col, label in ZONES.items():
        data = build_supervised_table(hdf, col)
        train, val, test = split(data)
        trval = pd.concat([train, val])
        base_val = baseline_metrics(val)
        base_test = baseline_metrics(test)

        # --- selection stage: fit on train, compare variants on VALIDATION only ---
        dv = lgb.LGBMRegressor(**PARAMS).fit(train[FEATURES], train["y"])
        val_direct = metrics(val["y"], dv.predict(val[FEATURES]))["MAPE"]

        rv = lgb.LGBMRegressor(**PARAMS).fit(train[FEATURES], train["y"] - train["seasonal_naive"])
        val_residual = metrics(val["y"], val["seasonal_naive"].values + rv.predict(val[FEATURES]))["MAPE"]

        chosen = "residual" if val_residual < val_direct else "direct"

        # --- reporting stage: refit both on train+val, score once on TEST ---
        d = lgb.LGBMRegressor(**PARAMS).fit(trval[FEATURES], trval["y"])
        m_direct = metrics(test["y"], d.predict(test[FEATURES]))

        r = lgb.LGBMRegressor(**PARAMS).fit(trval[FEATURES], trval["y"] - trval["seasonal_naive"])
        m_res = metrics(test["y"], test["seasonal_naive"].values + r.predict(test[FEATURES]))

        test_chosen = m_res["MAPE"] if chosen == "residual" else m_direct["MAPE"]

        rows.append({
            "zone": label,
            "val_direct": val_direct, "val_residual": val_residual,
            "val_seasonal_naive": base_val["seasonal_naive"]["MAPE"],
            "chosen_on_validation": chosen,
            "test_direct": m_direct["MAPE"], "test_residual": m_res["MAPE"],
            "test_chosen": test_chosen,
            "test_seasonal_naive": base_test["seasonal_naive"]["MAPE"],
            "test_persistence": base_test["persistence"]["MAPE"],
            "chosen_beats_naive": bool(test_chosen < base_test["seasonal_naive"]["MAPE"]),
            "effective_n": effective_n(test),
        })
        print(f"{label:8s} VAL direct {val_direct:6.2f}% residual {val_residual:6.2f}% "
              f"snaive {base_val['seasonal_naive']['MAPE']:6.2f}% -> chose {chosen}", flush=True)
        print(f"{'':8s} TEST direct {m_direct['MAPE']:6.2f}% residual {m_res['MAPE']:6.2f}% "
              f"snaive {base_test['seasonal_naive']['MAPE']:6.2f}% -> chosen {test_chosen:6.2f}% "
              f"beats naive: {rows[-1]['chosen_beats_naive']}", flush=True)
        del data, train, val, test, trval, d, r, dv, rv
        gc.collect()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
