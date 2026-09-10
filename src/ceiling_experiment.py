"""
Why does the model not beat the seasonal naive, and what would it take?

To beat "same hour last week" a model must predict how this week differs from last
week. This script adds, in stages, the information that could support that, and
measures how far each stage closes the gap.

  A  base            the deployed feature set: lagged weather only
  B  + level shift   legal, deployable. How far the series is currently running
                     above or below the same point last week, measured at the origin:
                     y[origin] - y[origin-168], plus the same for the 24h rolling mean,
                     plus a third seasonal lag at 504h.
  C  + oracle weather ORACLE ONLY, NOT DEPLOYABLE. Actual temperature and humidity at
                     the target hour, standing in for a perfect weather forecast, and
                     their difference from the same hour last week. This is an upper
                     bound on achievable skill, not a result that can be claimed.

Stage C uses information unavailable at forecast time. It is reported strictly as a
ceiling, never as model performance.

Run: python src/ceiling_experiment.py
"""

import gc
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting import ZONES, HORIZON, EMBARGO, TRAIN_CUTOFF, VAL_CUTOFF, load_hourly, metrics

OUT = Path("reports/ceiling_experiment_results.json")

PARAMS = dict(n_estimators=600, num_leaves=63, learning_rate=0.05,
              min_child_samples=40, colsample_bytree=0.8, subsample=0.9,
              subsample_freq=1, reg_lambda=1.0,
              random_state=42, verbosity=-1, n_jobs=-1)

BASE = ["horizon", "origin_value", "lag24", "lag168", "lag336",
        "roll_mean_24", "roll_std_24", "roll_mean_168", "roll_std_168",
        "target_hour", "target_dayofweek", "target_month", "target_is_weekend",
        "weather_temp_lag168", "weather_humidity_lag168"]
SHIFT = ["level_shift", "roll24_shift", "lag504"]
ORACLE = ["temp_target", "hum_target", "temp_delta_vs_lastweek", "hum_delta_vs_lastweek"]


def build(hdf, col):
    power = hdf[col].values.astype(np.float32)
    temp = hdf["Temperature"].values.astype(np.float32)
    hum = hdf["Humidity"].values.astype(np.float32)
    idx = hdf.index
    hours = idx.hour.values.astype(np.int16)
    dows = idx.dayofweek.values.astype(np.int16)
    months = idx.month.values.astype(np.int16)
    n = len(hdf)

    rm24 = hdf[col].rolling(24).mean().values.astype(np.float32)
    rs24 = hdf[col].rolling(24).std().values.astype(np.float32)
    rm168 = hdf[col].rolling(168).mean().values.astype(np.float32)
    rs168 = hdf[col].rolling(168).std().values.astype(np.float32)

    origins = np.arange(504, n - HORIZON)
    hz = np.arange(1, HORIZON + 1, dtype=np.int16)
    og, hg = np.meshgrid(origins, hz, indexing="ij")
    oi, ha = og.ravel(), hg.ravel()
    ti = oi + ha
    del og, hg
    gc.collect()

    return pd.DataFrame({
        "horizon": ha,
        "origin_value": power[oi], "lag24": power[oi - 24],
        "lag168": power[ti - 168], "lag336": power[ti - 336], "lag504": power[ti - 504],
        "roll_mean_24": rm24[oi], "roll_std_24": rs24[oi],
        "roll_mean_168": rm168[oi], "roll_std_168": rs168[oi],
        "target_hour": hours[ti], "target_dayofweek": dows[ti], "target_month": months[ti],
        "target_is_weekend": (dows[ti] >= 5).astype(np.int16),
        "weather_temp_lag168": temp[ti - 168], "weather_humidity_lag168": hum[ti - 168],
        # legal level-shift signals, all measured at or before the origin
        "level_shift": power[oi] - power[oi - 168],
        "roll24_shift": rm24[oi] - rm24[oi - 168],
        # ORACLE, unavailable at forecast time
        "temp_target": temp[ti], "hum_target": hum[ti],
        "temp_delta_vs_lastweek": temp[ti] - temp[ti - 168],
        "hum_delta_vs_lastweek": hum[ti] - hum[ti - 168],
        "seasonal_naive": power[ti - 168],
        "origin_time": idx[oi], "target_time": idx[ti],
        "y": power[ti],
    }).dropna()


def main():
    hdf = load_hourly()
    rows = []
    for col, label in ZONES.items():
        d = build(hdf, col)
        tr = d[d["origin_time"] <= TRAIN_CUTOFF - EMBARGO]
        va = d[(d["origin_time"] > TRAIN_CUTOFF) & (d["origin_time"] <= VAL_CUTOFF - EMBARGO)]
        te = d[d["origin_time"] > VAL_CUTOFF]
        trval = pd.concat([tr, va])
        sn = metrics(te["y"], te["seasonal_naive"])["MAPE"]

        stages = {"A base": BASE, "B +level shift": BASE + SHIFT,
                  "C +oracle weather": BASE + SHIFT + ORACLE}
        out = {"zone": label, "seasonal_naive": sn}
        for name, feats in stages.items():
            m = lgb.LGBMRegressor(**PARAMS).fit(trval[feats], trval["y"])
            s = metrics(te["y"], m.predict(te[feats]))["MAPE"]
            out[name] = s
            out[f"{name} gap_pp"] = s - sn
            print(f"{label}  {name:20s} {s:6.2f}%   snaive {sn:6.2f}%   "
                  f"gap {s - sn:+6.2f}pp   wins={s < sn}", flush=True)
            del m
            gc.collect()
        rows.append(out)
        print(flush=True)
        del d, tr, va, te, trval
        gc.collect()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows)
    print(df[["zone", "seasonal_naive", "A base", "B +level shift", "C +oracle weather"]].round(2).to_string(index=False))
    print(f"\nSaved -> {OUT}")
    print("\nStage C uses weather at the target hour, which is not available at forecast")
    print("time. It is an upper bound on achievable skill, not a deployable result.")


if __name__ == "__main__":
    main()
