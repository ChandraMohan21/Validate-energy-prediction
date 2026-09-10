"""
Is the blend's advantage over the seasonal naive statistically real?

The test set has 141,120 rows but only about 1,007 distinct target hours, because each
hour is forecast from up to 168 different origins. Treating the rows as independent
would overstate the evidence roughly 140-fold. Errors are therefore first collapsed to
one value per target hour, and every test runs on that collapsed series.

Two tests, both paired, on absolute percentage error per target hour:

  Diebold-Mariano   the standard test for comparing forecast accuracy, with a
                    Newey-West variance to allow for autocorrelation between
                    neighbouring hours
  paired bootstrap  10,000 resamples over target hours, giving a confidence
                    interval on the MAPE difference

Null hypothesis: the two forecasts are equally accurate. A small p-value means the
observed difference is unlikely to be chance.

Run: python src/significance_test.py
"""

import gc
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting import ZONES, FEATURES, load_hourly, build_supervised_table, split

OUT = Path("reports/significance_test_results.json")
PARAMS = dict(n_estimators=600, num_leaves=63, learning_rate=0.05,
              min_child_samples=40, colsample_bytree=0.8, subsample=0.9,
              subsample_freq=1, reg_lambda=1.0,
              random_state=42, verbosity=-1, n_jobs=-1)
RNG = np.random.default_rng(42)


def newey_west_var(d, lag=None):
    n = len(d)
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    d = d - d.mean()
    g0 = np.dot(d, d) / n
    v = g0
    for k in range(1, lag + 1):
        gk = np.dot(d[k:], d[:-k]) / n
        v += 2 * (1 - k / (lag + 1)) * gk
    return max(v, 1e-12), lag


def diebold_mariano(loss_a, loss_b):
    """H0: equal accuracy. Negative stat means loss_a < loss_b, i.e. a is better."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    var, lag = newey_west_var(d)
    stat = d.mean() / np.sqrt(var / n)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p), int(lag)


def paired_bootstrap(loss_a, loss_b, n_boot=10000):
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float((means >= 0).mean())


def main():
    hdf = load_hourly()
    rows = []
    for col, label in ZONES.items():
        data = build_supervised_table(hdf, col)
        train, val, test = split(data)
        trval = pd.concat([train, val])

        m = lgb.LGBMRegressor(**PARAMS).fit(trval[FEATURES], trval["y"])
        t = test[["target_time", "y", "seasonal_naive"]].copy()
        t["model"] = m.predict(test[FEATURES])
        t["blend"] = 0.5 * t["model"] + 0.5 * t["seasonal_naive"]

        # collapse to one absolute percentage error per distinct target hour
        for c in ["model", "seasonal_naive", "blend"]:
            t[f"ape_{c}"] = (t[c] - t["y"]).abs() / t["y"] * 100
        g = t.groupby("target_time")[["ape_model", "ape_seasonal_naive", "ape_blend"]].mean()
        n_eff = len(g)

        out = {"zone": label, "effective_n": int(n_eff),
               "MAPE_seasonal_naive": float(g["ape_seasonal_naive"].mean()),
               "MAPE_model": float(g["ape_model"].mean()),
               "MAPE_blend": float(g["ape_blend"].mean())}

        for name, colname in [("blend_vs_naive", "ape_blend"), ("model_vs_naive", "ape_model")]:
            dm_stat, dm_p, lag = diebold_mariano(g[colname].values, g["ape_seasonal_naive"].values)
            mean_d, lo, hi, p_worse = paired_bootstrap(g[colname].values, g["ape_seasonal_naive"].values)
            out[name] = {"mean_diff_pp": mean_d, "ci95": [lo, hi],
                         "dm_stat": dm_stat, "dm_p": dm_p, "nw_lag": lag,
                         "significant_at_05": bool(dm_p < 0.05),
                         "ci_excludes_zero": bool(lo > 0 or hi < 0)}

        rows.append(out)
        b = out["blend_vs_naive"]
        print(f"{label}  (effective n = {n_eff})", flush=True)
        print(f"  seasonal naive {out['MAPE_seasonal_naive']:6.3f}%   "
              f"blend {out['MAPE_blend']:6.3f}%", flush=True)
        print(f"  blend minus naive: {b['mean_diff_pp']:+.4f} pp   "
              f"95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]", flush=True)
        print(f"  Diebold-Mariano p = {b['dm_p']:.4f}  "
              f"-> {'SIGNIFICANT' if b['significant_at_05'] else 'NOT significant'} at 0.05\n",
              flush=True)
        del data, train, val, test, trval, m, t, g
        gc.collect()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
