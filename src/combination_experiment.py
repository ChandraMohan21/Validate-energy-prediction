"""
Forecast combination: can blending the model with the seasonal naive beat both?

Standard forecast combination. If two forecasts make partly uncorrelated errors, a
weighted blend can beat either alone even when one is clearly worse.

    prediction = w * model + (1 - w) * seasonal_naive

Three variants, all with the weight chosen on VALIDATION only and scored once on test:

  equal        w fixed at 0.5, no fitting at all
  global w     one w per zone, grid searched on validation
  per-horizon  a separate w for each day-offset 1..7, grid searched on validation

An oracle row is also reported: the best w had it been chosen on test. That is a
ceiling for reference and is never a claimable result, since choosing on test is
exactly the selection leakage this project committed once already.

Run: python src/combination_experiment.py
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
                         split, metrics)

OUT = Path("reports/combination_experiment_results.json")
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)

PARAMS = dict(n_estimators=600, num_leaves=63, learning_rate=0.05,
              min_child_samples=40, colsample_bytree=0.8, subsample=0.9,
              subsample_freq=1, reg_lambda=1.0,
              random_state=42, verbosity=-1, n_jobs=-1)


def mape(a, p):
    a = np.asarray(a, float)
    return float(np.mean(np.abs((a - np.asarray(p, float)) / a)) * 100)


def best_w(y, model_pred, snaive):
    scores = [(mape(y, w * model_pred + (1 - w) * snaive), w) for w in GRID]
    s, w = min(scores)
    return w, s


def main():
    hdf = load_hourly()
    rows = []
    for col, label in ZONES.items():
        data = build_supervised_table(hdf, col)
        train, val, test = split(data)
        trval = pd.concat([train, val])

        # selection stage: fit on train, choose weights on validation
        m_sel = lgb.LGBMRegressor(**PARAMS).fit(train[FEATURES], train["y"])
        v_pred = m_sel.predict(val[FEATURES])
        v_sn = val["seasonal_naive"].values
        w_global, _ = best_w(val["y"].values, v_pred, v_sn)

        w_by_day = {}
        v_day = np.ceil(val["horizon"].values / 24).astype(int)
        for d in range(1, 8):
            k = v_day == d
            if k.sum():
                w_by_day[d], _ = best_w(val["y"].values[k], v_pred[k], v_sn[k])
        del m_sel, v_pred
        gc.collect()

        # reporting stage: refit on train+val, score once on test
        m = lgb.LGBMRegressor(**PARAMS).fit(trval[FEATURES], trval["y"])
        t_pred = m.predict(test[FEATURES])
        t_y = test["y"].values
        t_sn = test["seasonal_naive"].values
        t_day = np.ceil(test["horizon"].values / 24).astype(int)

        sn_score = mape(t_y, t_sn)
        model_score = mape(t_y, t_pred)
        equal_score = mape(t_y, 0.5 * t_pred + 0.5 * t_sn)
        global_score = mape(t_y, w_global * t_pred + (1 - w_global) * t_sn)

        per_h = t_sn.copy().astype(float)
        for d, w in w_by_day.items():
            k = t_day == d
            per_h[k] = w * t_pred[k] + (1 - w) * t_sn[k]
        per_h_score = mape(t_y, per_h)

        w_oracle, oracle_score = best_w(t_y, t_pred, t_sn)   # ceiling, not claimable

        rows.append({
            "zone": label,
            "seasonal_naive": sn_score, "model_alone": model_score,
            "equal_blend": equal_score,
            "w_global_from_val": w_global, "global_blend": global_score,
            "w_by_day_from_val": w_by_day, "per_horizon_blend": per_h_score,
            "ORACLE_w_on_test": w_oracle, "ORACLE_blend": oracle_score,
            "best_legit": min(equal_score, global_score, per_h_score),
            "beats_naive": bool(min(equal_score, global_score, per_h_score) < sn_score),
        })
        print(f"{label}", flush=True)
        print(f"  seasonal naive      {sn_score:6.2f}%", flush=True)
        print(f"  model alone         {model_score:6.2f}%", flush=True)
        print(f"  equal blend (0.5)   {equal_score:6.2f}%", flush=True)
        print(f"  global w={w_global:<4.2f}       {global_score:6.2f}%", flush=True)
        print(f"  per-horizon w       {per_h_score:6.2f}%   w by day: {w_by_day}", flush=True)
        print(f"  [ORACLE w={w_oracle:<4.2f}]     {oracle_score:6.2f}%  (not claimable)", flush=True)
        print(f"  BEATS NAIVE: {rows[-1]['beats_naive']}\n", flush=True)
        del data, train, val, test, trval, m, t_pred
        gc.collect()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    df = pd.DataFrame(rows)
    print(df[["zone", "seasonal_naive", "model_alone", "equal_blend",
              "global_blend", "per_horizon_blend", "beats_naive"]].round(2).to_string(index=False))
    print(f"\nBlend beats seasonal naive in {df['beats_naive'].sum()}/{len(df)} zones")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
