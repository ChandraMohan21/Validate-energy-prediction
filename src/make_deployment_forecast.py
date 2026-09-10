"""
Produce the deployment forecast: the 168 hours after the dataset's last timestamp.

Takes the configuration selected on validation (cloned from the saved refit model so
hyperparameters are identical), retrains it on the full year, and forecasts forward.
The seasonal-naive forecast for the same hours is emitted alongside it, because on
this dataset the naive baseline is the stronger of the two and the dashboard should
not show the model alone as if it were the better choice.

No actuals exist for these hours. This is a demonstration of the deployment path,
not an evaluation.

Run: python src/make_deployment_forecast.py
"""

import gc
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting import ZONES, HORIZON, FEATURES, load_hourly, build_supervised_table

MODELS_DIR = Path("models")
OUT = Path("data/processed/future_7day_forecast.csv")


def future_features(hdf, col):
    """Feature rows for the 168 hours after the last observation.

    Every input is already observed at the forecast origin: lags and rolling stats
    come from the tail of the data, calendar from the future timestamps, and both
    the 168h and 336h lags plus the weather lag resolve to indices at or before the
    origin because the horizon never exceeds 168.
    """
    power = hdf[col].values.astype(np.float32)
    temp = hdf["Temperature"].values.astype(np.float32)
    hum = hdf["Humidity"].values.astype(np.float32)
    n = len(hdf)
    i = n - 1
    hz = np.arange(1, HORIZON + 1)
    times = hdf.index[-1] + pd.to_timedelta(hz, unit="h")
    tgt = i + hz                      # virtual target index
    lag168_idx = tgt - 168            # <= i for all h <= 168
    lag336_idx = tgt - 336

    X = pd.DataFrame({
        "horizon": hz,
        "origin_value": power[i],
        "lag24": power[i - 24],
        "lag168": power[lag168_idx],
        "lag336": power[lag336_idx],
        "roll_mean_24": power[i - 23:i + 1].mean(),
        "roll_std_24": power[i - 23:i + 1].std(ddof=1),
        "roll_mean_168": power[i - 167:i + 1].mean(),
        "roll_std_168": power[i - 167:i + 1].std(ddof=1),
        "target_hour": times.hour,
        "target_dayofweek": times.dayofweek,
        "target_month": times.month,
        "target_is_weekend": (times.dayofweek >= 5).astype(int),
        "weather_temp_lag168": temp[lag168_idx],
        "weather_humidity_lag168": hum[lag168_idx],
    }, index=times)
    seasonal_naive = pd.Series(power[lag168_idx], index=times)
    return X, seasonal_naive


def main():
    hdf = load_hourly()
    cols = {}
    for col, label in ZONES.items():
        slug = label.replace(" ", "_").lower()
        ref = MODELS_DIR / f"{slug}_final_refit_trainval.joblib"
        if not ref.exists():
            print(f"missing {ref}, run src/run_pipeline.py first")
            return
        m = clone(joblib.load(ref))

        data = build_supervised_table(hdf, col)
        m.fit(data[FEATURES], data["y"])
        joblib.dump(m, MODELS_DIR / f"{slug}_deploy_fullyear.joblib")

        X, snaive = future_features(hdf, col)
        cols[label] = pd.Series(m.predict(X[FEATURES]), index=X.index)
        cols[f"{label} seasonal naive"] = snaive
        print(f"{label}: model mean {cols[label].mean():,.0f} kW, "
              f"seasonal naive mean {snaive.mean():,.0f} kW")
        del data, m, X
        gc.collect()

    out = pd.DataFrame(cols)
    out.to_csv(OUT, index_label="Datetime")
    print(f"\n{out.index[0]} to {out.index[-1]}")
    print(f"Saved -> {OUT}")

    # figure: last 7 observed days, then model and seasonal-naive forecasts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#E6E6E6", "legend.frameon": False})
    colors = {"Zone 1": "#4477AA", "Zone 2": "#228833", "Zone 3": "#EE6677"}
    ctx = hdf[list(ZONES.keys())].iloc[-168:].rename(columns=ZONES)
    fig, ax = plt.subplots(figsize=(13, 5))
    for label in ZONES.values():
        ax.plot(ctx.index, ctx[label], color=colors[label], linewidth=1.4, label=f"{label} observed")
        ax.plot(out.index, out[label], color=colors[label], linewidth=1.4, linestyle="--")
        ax.plot(out.index, out[f"{label} seasonal naive"], color=colors[label],
                linewidth=1.0, linestyle=":", alpha=0.75)
    ax.axvline(ctx.index[-1], color="#888888", linewidth=1)
    ax.set_ylabel("kW")
    ax.set_title("Next 7 days, deployment forecast (solid observed, dashed model, dotted seasonal naive)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    figp = Path("reports/figures/modeling/final_future_7day_forecast.png")
    figp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figp)
    plt.close()
    print(f"Saved -> {figp}")


if __name__ == "__main__":
    main()
