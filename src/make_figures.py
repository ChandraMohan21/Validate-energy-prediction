"""
Generate the modeling figures from the saved models and the corrected feature set.

Reads models/zone_*_final_refit_trainval.joblib (the headline configuration) and
plots against BOTH baselines, seasonal naive and persistence, so no figure can
imply an improvement that is measured against a strawman.

Run: python src/make_figures.py
"""

import gc
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting import (ZONES, FEATURES, load_hourly, build_supervised_table,
                         split, metrics, baseline_metrics)

MODELS_DIR = Path("models")
FIG_DIR = Path("reports/figures/modeling")
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_MODEL, C_SNAIVE, C_PERS, C_ACTUAL = "#4477AA", "#EE6677", "#BBBBBB", "#222222"
ZONE_COLORS = {"Zone 1": "#4477AA", "Zone 2": "#228833", "Zone 3": "#EE6677"}

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})


def main():
    hdf = load_hourly()
    res = {}
    for col, label in ZONES.items():
        slug = label.replace(" ", "_").lower()
        path = MODELS_DIR / f"{slug}_final_refit_trainval.joblib"
        if not path.exists():
            print(f"missing {path}, run src/run_pipeline.py first")
            return
        m = joblib.load(path)
        data = build_supervised_table(hdf, col)
        _, _, test = split(data)
        test = test.copy()
        test["y_pred"] = m.predict(test[FEATURES])
        imp = pd.Series(m.feature_importances_, index=FEATURES, dtype=float)
        res[label] = {
            "type": type(m).__name__, "test": test, "imp": imp / imp.sum(),
            "m": metrics(test["y"], test["y_pred"]), "b": baseline_metrics(test),
        }
        print(f"{label}: model {res[label]['m']['MAPE']:.2f}%  "
              f"snaive {res[label]['b']['seasonal_naive']['MAPE']:.2f}%")
        del data, m
        gc.collect()

    labels = list(res)

    # 1. headline: model vs both baselines
    x = np.arange(len(labels)); w = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    series = [("Final model", [res[l]["m"]["MAPE"] for l in labels], C_MODEL, -w),
              ("Seasonal naive (same hour last week)", [res[l]["b"]["seasonal_naive"]["MAPE"] for l in labels], C_SNAIVE, 0),
              ("Persistence (value at forecast time)", [res[l]["b"]["persistence"]["MAPE"] for l in labels], C_PERS, w)]
    for name, vals, colr, off in series:
        bars = ax.bar(x + off, vals, w, color=colr, label=name)
        for r in bars:
            ax.annotate(f"{r.get_height():.1f}", xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n({res[l]['type']})" for l in labels])
    ax.set_ylabel("Test MAPE (%)")
    ax.set_title("168-hour-ahead forecast error vs baselines (untouched test window)")
    ax.legend(loc="upper left", fontsize=8.5); ax.grid(axis="x", visible=False)
    plt.tight_layout(); plt.savefig(FIG_DIR / "final_headline_model_vs_naive.png"); plt.close()

    # 2. error vs horizon, model and seasonal naive
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, l in zip(axes, labels):
        t = res[l]["test"]
        d = np.ceil(t["horizon"].values / 24).astype(int)
        by = pd.DataFrame({
            "d": d,
            "model": np.abs(t["y"].values - t["y_pred"].values) / t["y"].values * 100,
            "snaive": np.abs(t["y"].values - t["seasonal_naive"].values) / t["y"].values * 100,
        }).groupby("d").mean()
        ax.plot(by.index, by["model"], marker="o", markersize=5, linewidth=2, color=C_MODEL, label="Final model")
        ax.plot(by.index, by["snaive"], marker="o", markersize=5, linewidth=2, color=C_SNAIVE, linestyle="--", label="Seasonal naive")
        ax.set_title(f"{l} ({res[l]['type']})")
        ax.set_xlabel("Days ahead"); ax.set_ylabel("MAPE (%)")
        ax.set_xticks(range(1, 8)); ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(axis="x", visible=False)
    plt.suptitle("Forecast error vs horizon, test window", y=1.02)
    plt.tight_layout(); plt.savefig(FIG_DIR / "final_error_vs_horizon.png"); plt.close()

    # 3. sample 7-day forecast
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    for ax, l in zip(axes, labels):
        t = res[l]["test"]
        o = t["origin_time"].iloc[len(t) // 2]
        wk = t[t["origin_time"] == o].sort_values("horizon")
        ax.plot(wk["target_time"], wk["y"], color=C_ACTUAL, linewidth=1.4, label="Actual")
        ax.plot(wk["target_time"], wk["y_pred"], color=C_MODEL, linestyle="--", linewidth=1.4, label="Model")
        ax.plot(wk["target_time"], wk["seasonal_naive"], color=C_SNAIVE, linestyle=":", linewidth=1.2, label="Seasonal naive")
        ax.set_title(f"{l} ({res[l]['type']}), forecast made at {o}")
        ax.set_ylabel("kW"); ax.legend(fontsize=8); ax.grid(axis="x", visible=False)
    plt.tight_layout(); plt.savefig(FIG_DIR / "final_sample_forecasts.png"); plt.close()

    # 4. predicted vs actual
    rng = np.random.RandomState(42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, l in zip(axes, labels):
        t = res[l]["test"]
        i = rng.choice(len(t), size=min(4000, len(t)), replace=False)
        a = t["y"].values[i]; p = t["y_pred"].values[i]
        lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=C_ACTUAL, linewidth=1, linestyle="--", zorder=3, label="Perfect")
        ax.scatter(a, p, s=4, alpha=0.15, color=C_MODEL, edgecolors="none", zorder=2)
        ax.set_title(f"{l}, MAPE {res[l]['m']['MAPE']:.1f}%")
        ax.set_xlabel("Actual (kW)"); ax.set_ylabel("Predicted (kW)")
        ax.set_aspect("equal", adjustable="box"); ax.legend(fontsize=8, loc="upper left")
    plt.suptitle("Predicted vs actual, test window (4,000-point sample per zone)", y=1.02)
    plt.tight_layout(); plt.savefig(FIG_DIR / "final_predicted_vs_actual.png"); plt.close()

    # 5. feature importance
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, l in zip(axes, labels):
        imp = res[l]["imp"].sort_values()
        ax.barh(imp.index, imp.values, color=C_MODEL, height=0.65)
        for yp, v in enumerate(imp.values):
            ax.annotate(f"{v:.2f}", xy=(v, yp), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color="#333333")
        ax.set_title(f"{l} ({res[l]['type']})"); ax.set_xlabel("Normalized importance")
        ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8)
    plt.suptitle("Feature importance per zone", y=1.02)
    plt.tight_layout(); plt.savefig(FIG_DIR / "final_feature_importance.png"); plt.close()

    print(f"\nFigures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
