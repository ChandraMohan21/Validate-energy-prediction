"""
Time-series-based outlier detection for Zone 1 / Zone 2 / Zone 3 power consumption.

Why this differs from the earlier global-IQR flag (data cleaning step):
the earlier flag applied IQR directly to raw values, which conflates real
anomalies with legitimate seasonal peaks (e.g. Zone 3's July/August surge).
Here, each zone is decomposed via MSTL (trend + daily seasonal[24h] +
weekly seasonal[168h] + residual), and outliers are flagged on the RESIDUAL
only: i.e. what's left after removing the expected trend/seasonal pattern.
A point flagged here is unusual *given the time of day, day of week, and
season*: not just unusual in absolute terms.

Method: IQR (1.5x Tukey fence) on the residual, used alone. Residuals were
checked for normality (D'Agostino-Pearson) and found strongly heavy-tailed
(excess kurtosis ~9-10, p~0 for all 3 zones) -- Z-score's normal assumption
doesn't hold here, and its mean/std are prone to masking (extreme values
inflate std, hiding themselves and others). IQR's quartile-based bounds are
robust to that, so it's used as the sole method rather than cross-checked
against Z-score.

Run: python src/outlier_detection.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.tsa.seasonal import MSTL

DATA_PATH = Path("data/processed/cleaned_energy_data.csv")
FIG_DIR = Path("reports/figures/outliers")
REPORT_PATH = Path("reports/04_outlier_detection_report.md")
OUT_CSV = Path("data/processed/detected_outliers_timeseries.csv")

ZONES = {
    "PowerConsumption_Zone1": "Zone 1",
    "PowerConsumption_Zone2": "Zone 2",
    "PowerConsumption_Zone3": "Zone 3",
}
NAIVE_GLOBAL_IQR_COUNTS = {"Zone 1": 0, "Zone 2": 7, "Zone 3": 1191}

FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_hourly():
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"]).set_index("Datetime")
    return df[list(ZONES.keys())].resample("h").mean()


def detect(series):
    mstl = MSTL(series, periods=[24, 24 * 7]).fit()
    resid = mstl.resid

    q1, q3 = resid.quantile(0.25), resid.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_flag = (resid < lo) | (resid > hi)

    return {
        "trend": mstl.trend,
        "seasonal_daily": mstl.seasonal["seasonal_24"],
        "seasonal_weekly": mstl.seasonal["seasonal_168"],
        "resid": resid,
        "iqr_flag": iqr_flag,
        "bounds": (lo, hi),
    }


def plot_outliers(series, result, label):
    flagged = series[result["iqr_flag"]]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(series.index, series.values, linewidth=0.4, color="steelblue", label="Hourly mean")
    ax.scatter(flagged.index, flagged.values, color="crimson", s=14, zorder=5, label=f"IQR outliers (n={len(flagged)})")
    ax.set_title(f"{label}: time-series outliers (residual-based, IQR 1.5x)")
    ax.set_ylabel("kW")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{label.replace(' ', '_').lower()}_outliers.png", dpi=110)
    plt.close()


def plot_histograms(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, label) in zip(axes, ZONES.items()):
        resid = results[col]["resid"]
        skew = stats.skew(resid)
        kurt = stats.kurtosis(resid)

        ax.hist(resid, bins=80, density=True, color="steelblue", alpha=0.7, edgecolor="white", label="Residual")

        x = np.linspace(resid.min(), resid.max(), 300)
        normal_pdf = stats.norm.pdf(x, resid.mean(), resid.std())
        ax.plot(x, normal_pdf, color="crimson", linewidth=2, label="Normal fit (same mean/std)")

        lo, hi = results[col]["bounds"]
        ax.axvline(lo, color="black", linestyle="--", linewidth=1, label="IQR bounds")
        ax.axvline(hi, color="black", linestyle="--", linewidth=1)

        ax.set_title(f"{label}\nskew={skew:.2f}, excess kurtosis={kurt:.2f}", fontsize=10)
        ax.set_xlabel("Residual (kW)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    plt.suptitle("Residual distribution vs normal fit (heavy tails = not bell-shaped)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "residual_histograms.png", dpi=110)
    plt.close()


def plot_boxplots(results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (col, label) in zip(axes, ZONES.items()):
        resid = results[col]["resid"]
        bp = ax.boxplot(resid.values, vert=True, widths=0.5, patch_artist=True,
                         flierprops=dict(marker="o", markerfacecolor="crimson", markersize=4,
                                          markeredgecolor="crimson", alpha=0.6),
                         boxprops=dict(facecolor="steelblue", alpha=0.5),
                         medianprops=dict(color="black", linewidth=1.5))
        lo, hi = results[col]["bounds"]
        n_out = int(results[col]["iqr_flag"].sum())
        ax.set_title(f"{label}\nIQR bounds=({lo:.0f}, {hi:.0f}), n_outliers={n_out}", fontsize=10)
        ax.set_ylabel("Residual (kW)")
        ax.set_xticks([])
    plt.suptitle("Residual distributions with IQR outlier bounds (box plot)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "residual_boxplots.png", dpi=110)
    plt.close()


def main():
    hdf = load_hourly()
    all_flags = []
    results = {}
    report_lines = ["# Time-Series Outlier Detection Report", "",
                     "Method: MSTL decomposition (daily period=24h, weekly period=168h) per zone, "
                     "then outliers flagged on the **residual** (value with trend + daily + weekly "
                     "seasonality removed) using IQR (1.5x Tukey fence) alone. Residuals were checked "
                     "for normality and found strongly heavy-tailed (excess kurtosis ~9-10, "
                     "D'Agostino-Pearson p~0 for all 3 zones), so Z-score (which assumes normality "
                     "and is prone to masking by extreme values inflating std) was dropped in favor "
                     "of IQR alone.", ""]

    summary_rows = []
    for col, label in ZONES.items():
        result = detect(hdf[col])
        results[col] = result
        plot_outliers(hdf[col], result, label)

        flagged_idx = hdf.index[result["iqr_flag"]]
        n_flagged = len(flagged_idx)
        pct = 100 * n_flagged / len(hdf)

        by_month = flagged_idx.month.value_counts().sort_index()
        by_hour = flagged_idx.hour.value_counts().sort_index()

        summary_rows.append((label, n_flagged, pct, NAIVE_GLOBAL_IQR_COUNTS[label]))

        report_lines += [
            f"## {label}",
            f"- Flagged (residual-based IQR): {n_flagged} hours ({pct:.2f}%) vs naive global-IQR-on-raw-value: {NAIVE_GLOBAL_IQR_COUNTS[label]} rows (10-min resolution)",
            f"- Residual IQR bounds: ({result['bounds'][0]:.1f}, {result['bounds'][1]:.1f})",
            f"- Flagged by month: {by_month.to_dict()}",
            f"- Flagged by hour: {by_hour.to_dict()}",
            f"![{label} outliers](figures/outliers/{label.replace(' ', '_').lower()}_outliers.png)",
            "",
        ]

        for ts in flagged_idx:
            all_flags.append({
                "Datetime": ts, "Zone": label, "Value": hdf[col].loc[ts],
                "Trend": result["trend"].loc[ts], "Residual": result["resid"].loc[ts],
            })

    plot_boxplots(results)
    plot_histograms(results)

    report_lines += ["## Residual box plots (IQR outlier bounds)", "",
                      "![Residual box plots](figures/outliers/residual_boxplots.png)", "",
                      "## Residual histograms (vs normal fit)", "",
                      "![Residual histograms](figures/outliers/residual_histograms.png)", "",
                      "## Summary (residual-based IQR vs naive global-IQR)", "",
                      "| Zone | Time-series flagged (IQR on residual) | % of hours | Naive global-IQR flagged (10-min rows) |",
                      "|---|---|---|---|"]
    for label, n, pct, naive in summary_rows:
        report_lines.append(f"| {label} | {n} | {pct:.2f}% | {naive} |")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    out_df = pd.DataFrame(all_flags).sort_values(["Zone", "Datetime"])
    out_df.to_csv(OUT_CSV, index=False)

    print(f"Report -> {REPORT_PATH}")
    print(f"Flagged points CSV -> {OUT_CSV}")
    for label, n, pct, naive in summary_rows:
        print(f"{label}: time-series flagged={n} ({pct:.2f}%) | naive global-IQR={naive}")


if __name__ == "__main__":
    main()
