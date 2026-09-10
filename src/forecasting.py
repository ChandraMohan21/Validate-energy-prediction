"""
Core forecasting utilities: features, splits, baselines, metrics.

Single source of truth for the 168-hour-ahead load forecasting setup, shared by
every script and by notebooks/01_modeling.ipynb.

Corrections applied 2026-09 after an audit found the original implementation had
three defects:

1. `lag168` was indexed off the forecast ORIGIN (`power[origin - 168]`) instead of
   off the TARGET (`power[target - 168]`). Because the horizon never exceeds 168,
   `target - 168` is always at or before the origin, so it is fully observable at
   forecast time. The original version handed the model one frozen morning value
   for all 168 predictions, so the model could only learn an average weekly shape.
   Note the weather lag was already target-anchored; the power lag should have been
   too.

2. The naive baseline used that same wrong index, making it a strawman (about 25%
   MAPE instead of the roughly 3% a correct seasonal naive achieves). Every
   "improvement over naive" figure computed against it was meaningless.

3. Splits were cut on origin time with no embargo, so training rows had targets
   reaching 168 hours into the validation period, and validation rows had targets
   reaching into the test period. `split()` below enforces a 168-hour embargo.

Also reported here: the effective sample size. A test set of N origins yields
N * 168 rows but covers only about N + 168 distinct target hours, so raw row counts
overstate the evidence by roughly 140x. Use `effective_n` for any statistical claim.
"""

import gc

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ZONES = {
    "PowerConsumption_Zone1": "Zone 1",
    "PowerConsumption_Zone2": "Zone 2",
    "PowerConsumption_Zone3": "Zone 3",
}

HORIZON = 168                                        # hours (7 days)
EMBARGO = pd.Timedelta(hours=HORIZON)                # gap enforced between splits
TRAIN_CUTOFF = pd.Timestamp("2017-10-07 23:00:00")
VAL_CUTOFF = pd.Timestamp("2017-11-18 23:00:00")

FEATURES = [
    "horizon", "origin_value", "lag24", "lag168", "lag336",
    "roll_mean_24", "roll_std_24", "roll_mean_168", "roll_std_168",
    "target_hour", "target_dayofweek", "target_month", "target_is_weekend",
    "weather_temp_lag168", "weather_humidity_lag168",
]


def load_hourly(path="data/processed/cleaned_energy_data.csv"):
    df = pd.read_csv(path, parse_dates=["Datetime"]).set_index("Datetime")
    return df[list(ZONES.keys()) + ["Temperature", "Humidity"]].resample("h").mean()


def build_supervised_table(hdf, power_col):
    """One row per (origin, horizon) pair. Every feature is observable at the origin."""
    power = hdf[power_col].values.astype(np.float32)
    temp = hdf["Temperature"].values.astype(np.float32)
    hum = hdf["Humidity"].values.astype(np.float32)
    idx = hdf.index
    hours = idx.hour.values.astype(np.int16)
    dows = idx.dayofweek.values.astype(np.int16)
    months = idx.month.values.astype(np.int16)
    n = len(hdf)

    rm24 = hdf[power_col].rolling(24).mean().values.astype(np.float32)
    rs24 = hdf[power_col].rolling(24).std().values.astype(np.float32)
    rm168 = hdf[power_col].rolling(168).mean().values.astype(np.float32)
    rs168 = hdf[power_col].rolling(168).std().values.astype(np.float32)

    # origins need 336h of history behind them for lag336 on the earliest target
    origins = np.arange(336, n - HORIZON)
    hz = np.arange(1, HORIZON + 1, dtype=np.int16)
    og, hg = np.meshgrid(origins, hz, indexing="ij")
    oi = og.ravel()
    ha = hg.ravel()
    ti = oi + ha
    del og, hg
    gc.collect()

    data = pd.DataFrame({
        "horizon": ha,
        "origin_value": power[oi],
        "lag24": power[oi - 24],            # origin-anchored: target-24 would be unobservable for h>24
        "lag168": power[ti - 168],          # target-anchored, always observable since h <= 168
        "lag336": power[ti - 336],          # target-anchored, always observable since h <= 168
        "roll_mean_24": rm24[oi],
        "roll_std_24": rs24[oi],
        "roll_mean_168": rm168[oi],
        "roll_std_168": rs168[oi],
        "target_hour": hours[ti],
        "target_dayofweek": dows[ti],
        "target_month": months[ti],
        "target_is_weekend": (dows[ti] >= 5).astype(np.int16),
        "weather_temp_lag168": temp[ti - 168],
        "weather_humidity_lag168": hum[ti - 168],
        # baselines, carried alongside so every evaluation uses the same definitions
        "seasonal_naive": power[ti - 168],
        "persistence": power[oi],
        "origin_time": idx[oi],
        "target_time": idx[ti],
        "y": power[ti],
    })
    return data.dropna()


def split(data, train_cutoff=TRAIN_CUTOFF, val_cutoff=VAL_CUTOFF, embargo=EMBARGO):
    """Chronological split with an embargo so no split's targets enter the next split.

    train targets end at train_cutoff, val targets end at val_cutoff.
    """
    t_end = train_cutoff - embargo
    v_end = val_cutoff - embargo
    train = data[data["origin_time"] <= t_end]
    val = data[(data["origin_time"] > train_cutoff) & (data["origin_time"] <= v_end)]
    test = data[data["origin_time"] > val_cutoff]
    return train, val, test


def effective_n(frame):
    """Distinct target hours covered, the honest sample size for statistical claims."""
    return int(frame["target_time"].nunique())


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def baseline_metrics(frame):
    return {
        "seasonal_naive": metrics(frame["y"], frame["seasonal_naive"]),
        "persistence": metrics(frame["y"], frame["persistence"]),
    }


def describe_split(train, val, test):
    rows = []
    for name, f in [("train", train), ("val", val), ("test", test)]:
        rows.append({
            "split": name,
            "rows": len(f),
            "origins": int(f["origin_time"].nunique()),
            "effective_n (distinct target hours)": effective_n(f),
            "origin_first": f["origin_time"].min(),
            "origin_last": f["origin_time"].max(),
            "target_first": f["target_time"].min(),
            "target_last": f["target_time"].max(),
        })
    return pd.DataFrame(rows)
