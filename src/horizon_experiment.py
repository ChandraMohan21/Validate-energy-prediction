"""
Is the 168-hour horizon the reason the model loses to the seasonal naive?

Holds the model, the feature set and the split protocol constant and varies only the
forecast horizon. Any change in the model-versus-baseline gap is therefore attributable
to horizon length, not to modelling choices.

Feature set is held identical across horizons, restricted to what is legal at every
horizon tested: lag24 anchored to the origin, lag168 and lag336 anchored to the target.
Giving shorter horizons the extra lags they could legally use would confound horizon
with feature richness.

The embargo scales with the horizon, so no split's targets ever reach into the next.

Run: python src/horizon_experiment.py
"""

import gc
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting import ZONES, TRAIN_CUTOFF, VAL_CUTOFF, FEATURES, load_hourly, metrics

OUT = Path("reports/horizon_experiment_results.json")
HORIZONS = [1, 6, 24, 72, 168]

PARAMS = dict(n_estimators=400, num_leaves=63, learning_rate=0.05,
              min_child_samples=40, colsample_bytree=0.8, subsample=0.9,
              subsample_freq=1, reg_lambda=1.0,
              random_state=42, verbosity=-1, n_jobs=-1)


def build(hdf, col, horizon):
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

    origins = np.arange(336, n - horizon)
    hz = np.arange(1, horizon + 1, dtype=np.int32)
    og, hg = np.meshgrid(origins, hz, indexing="ij")
    oi, ha = og.ravel(), hg.ravel()
    ti = oi + ha
    del og, hg
    gc.collect()

    return pd.DataFrame({
        "horizon": ha,
        "origin_value": power[oi],
        "lag24": power[oi - 24],
        "lag168": power[ti - 168],
        "lag336": power[ti - 336],
        "roll_mean_24": rm24[oi], "roll_std_24": rs24[oi],
        "roll_mean_168": rm168[oi], "roll_std_168": rs168[oi],
        "target_hour": hours[ti], "target_dayofweek": dows[ti],
        "target_month": months[ti],
        "target_is_weekend": (dows[ti] >= 5).astype(np.int16),
        "weather_temp_lag168": temp[ti - 168],
        "weather_humidity_lag168": hum[ti - 168],
        "seasonal_naive": power[ti - 168],
        "persistence": power[oi],
        "origin_time": idx[oi],
        "target_time": idx[ti],
        "y": power[ti],
    }).dropna()


def main():
    hdf = load_hourly()
    rows = []
    for horizon in HORIZONS:
        emb = pd.Timedelta(hours=horizon)
        for col, label in ZONES.items():
            d = build(hdf, col, horizon)
            tr = d[d["origin_time"] <= TRAIN_CUTOFF - emb]
            te = d[d["origin_time"] > VAL_CUTOFF]
            assert tr["target_time"].max() <= TRAIN_CUTOFF

            m = lgb.LGBMRegressor(**PARAMS).fit(tr[FEATURES], tr["y"])
            mm = metrics(te["y"], m.predict(te[FEATURES]))["MAPE"]
            sn = metrics(te["y"], te["seasonal_naive"])["MAPE"]
            pe = metrics(te["y"], te["persistence"])["MAPE"]

            rows.append({"horizon_h": horizon, "zone": label,
                         "model": mm, "seasonal_naive": sn, "persistence": pe,
                         "gap_pp": mm - sn, "model_wins": bool(mm < sn),
                         "effective_n": int(te["target_time"].nunique())})
            print(f"h={horizon:4d}  {label}  model {mm:6.2f}%  snaive {sn:6.2f}%  "
                  f"pers {pe:6.2f}%  gap {mm-sn:+6.2f}pp  wins={mm < sn}", flush=True)
            del d, tr, te, m
            gc.collect()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows)
    print("\nModel minus seasonal naive, percentage points (negative means model wins):")
    print(df.pivot(index="horizon_h", columns="zone", values="gap_pp").round(2).to_string())
    print(f"\nModel wins {df['model_wins'].sum()}/{len(df)} horizon-zone combinations")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
