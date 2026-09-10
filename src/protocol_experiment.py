"""
Protocol sensitivity experiment on the Tetouan City dataset.

Question: published results on this dataset range from ~1% to ~5% MAPE, while this
project reports 3.6-17.3%. Are those numbers comparable?

Method: hold the MODEL constant (one LightGBM, same feature family) and vary only
the evaluation protocol. Any change in the score is therefore caused by the protocol,
not by the model. Baselines are reported at every setting so the difficulty of each
task is visible.

Settings
  A  10-min resolution, predict t+1 (10 minutes ahead), RANDOM split
  B  10-min resolution, predict t+1 (10 minutes ahead), chronological split
  C  hourly, predict t+1h, chronological split
  D  hourly, predict t+24h, chronological split
  E  hourly, predict t+168h (7 days, this project's task), chronological split

Baselines
  persistence     = repeat the most recent observed value
  seasonal naive  = value at the same time one week earlier

Run: python src/protocol_experiment.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/cleaned_energy_data.csv")
OUT_JSON = Path("reports/protocol_experiment_results.json")

ZONES = {
    "PowerConsumption_Zone1": "Zone 1",
    "PowerConsumption_Zone2": "Zone 2",
    "PowerConsumption_Zone3": "Zone 3",
}

LGB_PARAMS = dict(
    n_estimators=300, num_leaves=31, learning_rate=0.05,
    min_child_samples=20, random_state=42, verbosity=-1, n_jobs=-1,
)


def mape(a, p):
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs((a - p) / a)) * 100)


def build_table(s, temp, hum, horizon, day_steps, week_steps):
    """Supervised table: all features observable at origin t, target at t+horizon."""
    n = len(s)
    start = week_steps
    origins = np.arange(start, n - horizon)
    tgt = origins + horizon

    idx = s.index
    roll_day = s.rolling(day_steps).mean().values
    v = s.values

    df = pd.DataFrame({
        "origin_value": v[origins],
        "lag_1": v[origins - 1],
        "lag_day": v[origins - day_steps],
        "lag_week": v[origins - week_steps],
        "roll_mean_day": roll_day[origins],
        "target_hour": idx[tgt].hour,
        "target_dayofweek": idx[tgt].dayofweek,
        "target_month": idx[tgt].month,
        "target_is_weekend": (idx[tgt].dayofweek >= 5).astype(int),
        "temp_origin": temp[origins],
        "hum_origin": hum[origins],
        # baselines
        "_persistence": v[origins],
        "_seasonal_naive": v[tgt - week_steps],
        "y": v[tgt],
    })
    return df.dropna().reset_index(drop=True)


FEATS = ["origin_value", "lag_1", "lag_day", "lag_week", "roll_mean_day",
         "target_hour", "target_dayofweek", "target_month", "target_is_weekend",
         "temp_origin", "hum_origin"]


def run(df, split):
    if split == "random":
        tr, te = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    else:
        cut = int(len(df) * 0.8)
        tr, te = df.iloc[:cut], df.iloc[cut:]
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(tr[FEATS], tr["y"])
    pred = m.predict(te[FEATS])
    return {
        "model": mape(te["y"], pred),
        "persistence": mape(te["y"], te["_persistence"]),
        "seasonal_naive": mape(te["y"], te["_seasonal_naive"]),
        "n_test": int(len(te)),
    }


def main():
    raw = pd.read_csv(DATA_PATH, parse_dates=["Datetime"]).set_index("Datetime")
    hourly = raw[list(ZONES.keys()) + ["Temperature", "Humidity"]].resample("h").mean()

    settings = [
        ("A", "10-min", 1, "random", 144, 1008),
        ("B", "10-min", 1, "chrono", 144, 1008),
        ("C", "hourly", 1, "chrono", 24, 168),
        ("D", "hourly", 24, "chrono", 24, 168),
        ("E", "hourly", 168, "chrono", 24, 168),
    ]

    results = {}
    for code, res, hz, split, day_steps, week_steps in settings:
        src = raw if res == "10-min" else hourly
        temp = src["Temperature"].values
        hum = src["Humidity"].values
        per_zone = {}
        for col, label in ZONES.items():
            tbl = build_table(src[col], temp, hum, hz, day_steps, week_steps)
            per_zone[label] = run(tbl, split)
            del tbl
        results[code] = {
            "resolution": res, "horizon_steps": hz, "split": split,
            "horizon_human": ("10 minutes" if res == "10-min" else
                              ("1 hour" if hz == 1 else f"{hz} hours")),
            "zones": per_zone,
        }
        line = " | ".join(
            f"{z}: model {v['model']:.2f}% pers {v['persistence']:.2f}% snaive {v['seasonal_naive']:.2f}%"
            for z, v in per_zone.items())
        print(f"[{code}] {res} h={hz} {split:6s} -> {line}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
