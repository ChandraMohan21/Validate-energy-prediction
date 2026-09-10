"""
Exploratory data analysis for the Tetouan City power-consumption dataset (UCI 849).

Covers both possible modeling framings (decision deferred until after this EDA):
  A. Time-series structure  -> supports a forecasting framing
  B. Weather/time feature relationships -> supports a conditions-based prediction framing
  C. Cross-zone comparison

Run: python src/eda.py
Outputs: reports/figures/*.png and reports/02_eda_report.md
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

DATA_PATH = Path("data/processed/cleaned_energy_data.csv")
FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/02_eda_report.md")

ZONES = {
    "PowerConsumption_Zone1": "Zone 1",
    "PowerConsumption_Zone2": "Zone 2",
    "PowerConsumption_Zone3": "Zone 3",
}
WEATHER_COLS = ["Temperature", "Humidity", "WindSpeed", "GeneralDiffuseFlows", "DiffuseFlows"]

sns.set_style("whitegrid")
FIG_DIR.mkdir(parents=True, exist_ok=True)

findings = {}


def load():
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
    df = df.set_index("Datetime")
    return df


def hourly(df):
    return df[list(ZONES.keys()) + WEATHER_COLS].resample("h").mean()


# --- A. Time series structure ---

def plot_full_year(df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, (col, label) in zip(axes, ZONES.items()):
        ax.plot(df.index, df[col], linewidth=0.4, color="steelblue")
        ax.set_ylabel("kW")
        ax.set_title(f"{label}: full year")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_full_year_timeseries.png", dpi=110)
    plt.close()


def seasonal_decomposition(hdf):
    results = {}
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    for i, (col, label) in enumerate(ZONES.items()):
        result = seasonal_decompose(hdf[col], model="additive", period=24)
        axes[i, 0].plot(result.trend, color="darkorange")
        axes[i, 0].set_title(f"{label}: Trend")
        axes[i, 1].plot(result.seasonal.iloc[:24 * 7], color="seagreen")
        axes[i, 1].set_title(f"{label}: Daily Seasonality (1 wk sample)")
        axes[i, 2].plot(result.resid, color="gray", linewidth=0.3)
        axes[i, 2].set_title(f"{label}: Residual")
        results[label] = {
            "resid_std": float(np.nanstd(result.resid)),
            "seasonal_amplitude": float(result.seasonal.max() - result.seasonal.min()),
        }
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_seasonal_decomposition.png", dpi=110)
    plt.close()
    return results


def adf_tests(hdf):
    results = {}
    for col, label in ZONES.items():
        stat, pvalue, *_ = adfuller(hdf[col].dropna())
        results[label] = {"adf_stat": float(stat), "p_value": float(pvalue), "stationary": bool(pvalue < 0.05)}
    return results


def acf_pacf_plots(hdf):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for i, (col, label) in enumerate(ZONES.items()):
        plot_acf(hdf[col].dropna(), lags=168, ax=axes[i, 0])
        axes[i, 0].set_title(f"{label}: ACF (lags=168h/7d)")
        plot_pacf(hdf[col].dropna(), lags=48, ax=axes[i, 1], method="ywm")
        axes[i, 1].set_title(f"{label}: PACF (lags=48h)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_acf_pacf.png", dpi=110)
    plt.close()


def daily_weekly_profiles(hdf):
    hdf = hdf.copy()
    hdf["hour"] = hdf.index.hour
    hdf["is_weekend"] = hdf.index.dayofweek >= 5
    hdf["month"] = hdf.index.month

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for col, label in ZONES.items():
        profile = hdf.groupby(["hour", "is_weekend"])[col].mean().unstack()
        axes[0].plot(profile.index, profile[False], label=f"{label} weekday")
        axes[1].plot(profile.index, profile[True], label=f"{label} weekend")
    axes[0].set_title("Avg hourly profile: weekdays")
    axes[1].set_title("Avg hourly profile: weekends")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    monthly = hdf.groupby("month")[list(ZONES.keys())].mean()
    monthly.columns = list(ZONES.values())
    monthly.plot(ax=axes[2])
    axes[2].set_title("Avg monthly profile")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_daily_weekly_monthly_profiles.png", dpi=110)
    plt.close()

    peak_hours = {}
    for col, label in ZONES.items():
        p = hdf.groupby("hour")[col].mean()
        peak_hours[label] = {"peak_hour": int(p.idxmax()), "trough_hour": int(p.idxmin())}
    return peak_hours


# --- B. Weather relationships ---

def weather_correlation(df):
    corr = df[list(ZONES.keys()) + WEATHER_COLS].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix: power & weather")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_weather_correlation_heatmap.png", dpi=110)
    plt.close()

    top_corr = {}
    for col, label in ZONES.items():
        weather_corr = corr.loc[col, WEATHER_COLS].abs().sort_values(ascending=False)
        top_corr[label] = {w: round(corr.loc[col, w], 3) for w in weather_corr.index}
    return top_corr


def weather_scatter(df):
    fig, axes = plt.subplots(len(ZONES), len(WEATHER_COLS), figsize=(20, 10))
    sample = df.sample(min(5000, len(df)), random_state=42)
    for i, (col, label) in enumerate(ZONES.items()):
        for j, w in enumerate(WEATHER_COLS):
            axes[i, j].scatter(sample[w], sample[col], s=2, alpha=0.3, color="teal")
            axes[i, j].set_xlabel(w, fontsize=8)
            axes[i, j].set_ylabel(label, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_weather_scatter.png", dpi=110)
    plt.close()


def distributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, label) in zip(axes, ZONES.items()):
        sns.histplot(df[col], kde=True, ax=ax, color="slateblue")
        ax.set_title(label)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_power_distributions.png", dpi=110)
    plt.close()


def outlier_location(df):
    flag_col = "PowerConsumption_Zone3_outlier"
    flagged = df[df[flag_col]]
    result = {
        "total_flagged": int(len(flagged)),
        "by_month": flagged.index.month.value_counts().sort_index().to_dict(),
        "by_hour": flagged.index.hour.value_counts().sort_index().to_dict(),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    flagged.index.month.value_counts().sort_index().plot(kind="bar", ax=axes[0], color="crimson")
    axes[0].set_title("Zone 3 flagged outliers by month")
    flagged.index.hour.value_counts().sort_index().plot(kind="bar", ax=axes[1], color="crimson")
    axes[1].set_title("Zone 3 flagged outliers by hour-of-day")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_zone3_outlier_location.png", dpi=110)
    plt.close()
    return result


# --- C. Cross-zone comparison ---

def cross_zone_correlation(df):
    corr = df[list(ZONES.keys())].corr()
    corr.index = corr.columns = list(ZONES.values())
    return corr.round(3)


def write_report(f):
    lines = ["# EDA Report", "", f"Generated by `src/eda.py` from `{DATA_PATH}`.", ""]

    lines += ["## A. Time-series structure", ""]
    lines += ["### Seasonal decomposition (daily period=24h)"]
    for label, s in f["decomposition"].items():
        lines.append(f"- {label}: daily seasonal amplitude = {s['seasonal_amplitude']:.1f} kW, residual std = {s['resid_std']:.1f}")
    lines += ["", "### Stationarity (Augmented Dickey-Fuller test, hourly series)"]
    for label, s in f["adf"].items():
        verdict = "stationary" if s["stationary"] else "NOT stationary"
        lines.append(f"- {label}: ADF stat={s['adf_stat']:.2f}, p-value={s['p_value']:.4f} -> {verdict}")
    lines += ["", "### Daily peak/trough hours"]
    for label, s in f["peaks"].items():
        lines.append(f"- {label}: peak hour={s['peak_hour']}:00, trough hour={s['trough_hour']}:00")
    lines += ["", "![Full year](figures/01_full_year_timeseries.png)",
              "![Decomposition](figures/02_seasonal_decomposition.png)",
              "![ACF/PACF](figures/03_acf_pacf.png)",
              "![Profiles](figures/04_daily_weekly_monthly_profiles.png)", ""]

    lines += ["## B. Weather relationships", ""]
    for label, corrs in f["weather_corr"].items():
        top3 = list(corrs.items())[:3]
        lines.append(f"- {label} top correlated weather vars: " + ", ".join(f"{k}={v}" for k, v in top3))
    lines += ["", "![Weather correlation](figures/05_weather_correlation_heatmap.png)",
              "![Weather scatter](figures/06_weather_scatter.png)",
              "![Distributions](figures/07_power_distributions.png)", ""]

    lines += ["## Zone 3 outlier location (from the flagged column, informational only)", ""]
    o = f["outliers"]
    lines.append(f"- Total flagged: {o['total_flagged']} ({100*o['total_flagged']/f['n_rows']:.2f}% of rows)")
    lines.append(f"- By month: {o['by_month']}")
    lines.append(f"- By hour: {o['by_hour']}")
    lines += ["", "![Zone 3 outlier location](figures/08_zone3_outlier_location.png)", ""]

    lines += ["## C. Cross-zone correlation (power vs power)", ""]
    lines.append(f["cross_zone_corr"].to_markdown())

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    df = load()
    hdf = hourly(df)

    plot_full_year(df)
    findings["decomposition"] = seasonal_decomposition(hdf)
    findings["adf"] = adf_tests(hdf)
    acf_pacf_plots(hdf)
    findings["peaks"] = daily_weekly_profiles(hdf)
    findings["weather_corr"] = weather_correlation(df)
    weather_scatter(df)
    distributions(df)
    findings["outliers"] = outlier_location(df)
    findings["cross_zone_corr"] = cross_zone_correlation(df)
    findings["n_rows"] = len(df)

    write_report(findings)
    print("EDA complete.")
    print("ADF results:", findings["adf"])
    print("Peaks:", findings["peaks"])
    print("Weather corr:", findings["weather_corr"])
    print("Cross-zone corr:\n", findings["cross_zone_corr"])
    print("Outliers:", findings["outliers"])
