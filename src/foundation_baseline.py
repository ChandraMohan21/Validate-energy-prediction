"""
Zero-shot foundation model baseline: does a pretrained time series model beat
"same hour last week"?

Every model in this project so far was trained on Tetouan data alone. A time series
foundation model is trained once, by someone else, on millions of unrelated series
(energy, traffic, retail, weather, web), then applied to a new series with no
training and no feature engineering at all. Chronos-Bolt from Amazon is used here.

The question this answers is the one a reviewer asks about the main result. Tree
ensembles losing to a seasonal naive is a known story; whether a model with broad
pretraining does any better at a 168-hour horizon is not.

Protocol, identical to every other experiment in this repository:

  origins       the same origins as the experiment being compared against
  context       up to CONTEXT hours of history ending AT the origin, so nothing
                after the forecast origin is ever visible
  horizon       1..168 hours ahead, one forecast per origin
  point forecast the 0.5 quantile, which is the MAPE-appropriate summary
  baselines     seasonal_naive = y[target - 168], persistence = y[origin]

Two modes:

  default        the single test window, origin_time strictly after VAL_CUTOFF,
                 matching forecasting.split(). Writes foundation_baseline_results.json.
  --walkforward  the three 42-day walk-forward windows of run_pipeline.py stage 5.
                 Writes foundation_walkforward_results.json. Chronos has no training
                 or selection step, so every window is out of sample for it, including
                 window 2, where the tree models' hyperparameters were chosen.

No training happens. There is no fitting stage, so there is no split to leak across;
the model simply never saw this series before. Tetouan does not appear in the
published Chronos pretraining corpus (autogluon/chronos_datasets).

torch and chronos are imported lazily, so validate_project.py can import the origin
and context logic below on a machine without them.

Run: python src/foundation_baseline.py [--walkforward] [model_id]
Needs a few GB of free memory; notebooks/02_foundation_baseline_colab.ipynb runs it
on Google Colab.
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting import ZONES, HORIZON, VAL_CUTOFF, load_hourly, metrics

MODEL_ID = "amazon/chronos-bolt-base"
CONTEXT = 2048          # Chronos-Bolt maximum, about 12 weeks of hourly history
BATCH = 16
WINDOW_DAYS = 42        # walk-forward window length, as in run_pipeline.py
N_WINDOWS = 3
OUT = Path("reports/foundation_baseline_results.json")
WF_OUT = Path("reports/foundation_walkforward_results.json")
ML_RESULTS = Path("reports/results_corrected.json")


def candidate_origins(n):
    """Origins with 336h of history behind them, as in build_supervised_table."""
    return np.arange(336, n - HORIZON)


def test_origin_indices(idx, n):
    """Positions of the test origins, matching forecasting.split() exactly."""
    origins = candidate_origins(n)
    return origins[idx[origins] > VAL_CUTOFF]


def window_origin_indices(idx, n, i):
    """Origins of walk-forward window i + 1, matching run_pipeline.py stage 5."""
    max_t = idx.max()
    start = max_t - pd.Timedelta(days=WINDOW_DAYS * (i + 1))
    end = max_t - pd.Timedelta(days=WINDOW_DAYS * i)
    origins = candidate_origins(n)
    t = idx[origins]
    return origins[(t > start) & (t <= end)], start, end


def context_slice(origin):
    """History handed to the model for one origin. The last index is the origin itself."""
    return slice(max(0, origin - CONTEXT + 1), origin + 1)


def load_pipeline(model_id):
    import torch
    from chronos import BaseChronosPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"  device: {device}", flush=True)
    return BaseChronosPipeline.from_pretrained(
        model_id, device_map=device, torch_dtype=dtype
    )


def forecast(pipeline, power, origins):
    """One 168-step forecast per origin. Context ends at the origin."""
    import torch
    batch = 64 if torch.cuda.is_available() else BATCH
    preds = np.empty((len(origins), HORIZON), dtype=np.float32)
    t0 = time.time()
    for s in range(0, len(origins), batch):
        chunk = origins[s:s + batch]
        ctx = [torch.tensor(power[context_slice(o)], dtype=torch.float32) for o in chunk]
        # passed positionally: the argument is `context` in chronos 1.x, `inputs` in 2.x
        q, _ = pipeline.predict_quantiles(
            ctx, prediction_length=HORIZON, quantile_levels=[0.5]
        )
        preds[s:s + len(chunk)] = q[:, :, 0].to(torch.float32).cpu().numpy()
        if (s // batch) % 5 == 0:
            done = s + len(chunk)
            print(f"    {done}/{len(origins)} origins  ({time.time() - t0:.0f}s)", flush=True)
    return preds


def build_frame(origins, preds, power, idx):
    oi = np.repeat(origins, HORIZON)
    ha = np.tile(np.arange(1, HORIZON + 1, dtype=np.int16), len(origins))
    ti = oi + ha
    return pd.DataFrame({
        "origin_time": idx[oi],
        "target_time": idx[ti],
        "horizon": ha,
        "y": power[ti],
        "chronos": preds.ravel(),
        "seasonal_naive": power[ti - 168],
        "persistence": power[oi],
    })


def score(f):
    """Point metrics, per-day-ahead gaps, and significance on the effective sample."""
    from significance_test import diebold_mariano, paired_bootstrap

    out = {
        "n_origins": int(f["origin_time"].nunique()),
        "rows": int(len(f)),
        "effective_n": int(f["target_time"].nunique()),
        "chronos": metrics(f["y"], f["chronos"]),
        "seasonal_naive": metrics(f["y"], f["seasonal_naive"]),
        "persistence": metrics(f["y"], f["persistence"]),
    }

    day = np.ceil(f["horizon"].values / 24).astype(int)
    out["by_day_ahead"] = {
        str(d): {
            "chronos_MAPE": metrics(f["y"].values[day == d], f["chronos"].values[day == d])["MAPE"],
            "seasonal_naive_MAPE": metrics(f["y"].values[day == d],
                                           f["seasonal_naive"].values[day == d])["MAPE"],
        }
        for d in range(1, 8)
    }

    # one error per distinct target hour, the honest sample for a test
    ape = pd.DataFrame({
        "target_time": f["target_time"],
        "chronos": (f["chronos"] - f["y"]).abs() / f["y"] * 100,
        "seasonal_naive": (f["seasonal_naive"] - f["y"]).abs() / f["y"] * 100,
    })
    g = ape.groupby("target_time")[["chronos", "seasonal_naive"]].mean()
    stat, p, lag = diebold_mariano(g["chronos"].values, g["seasonal_naive"].values)
    mean_d, lo, hi, _ = paired_bootstrap(g["chronos"].values, g["seasonal_naive"].values)
    out["chronos_vs_seasonal_naive"] = {
        "mean_diff_pp": mean_d, "ci95": [lo, hi],
        "dm_stat": stat, "dm_p": p, "nw_lag": lag,
        "significant_at_05": bool(p < 0.05),
        "chronos_wins": bool(mean_d < 0),
    }
    return out


def main():
    args = sys.argv[1:]
    walkforward = "--walkforward" in args
    args = [a for a in args if a != "--walkforward"]
    model_id = args[0] if args else MODEL_ID

    mode = "walk-forward windows" if walkforward else "single test window"
    print(f"Loading {model_id} (zero-shot, no training on this data), {mode}\n", flush=True)
    pipeline = load_pipeline(model_id)

    ml, wf_ml = {}, {}
    if ML_RESULTS.exists():
        res = json.loads(ML_RESULTS.read_text(encoding="utf-8"))
        ml = res.get("zones", {})
        wf_ml = {(r["zone"], r["window"]): r for r in res.get("multi_window", [])}

    hdf = load_hourly()
    idx, n = hdf.index, len(hdf)
    rows = []

    for col, label in ZONES.items():
        power = hdf[col].values.astype(np.float32)
        if walkforward:
            jobs = []
            for i in range(N_WINDOWS):
                origins, start, end = window_origin_indices(idx, n, i)
                jobs.append((i + 1, f"{start.date()} to {end.date()}", origins))
        else:
            jobs = [(None, None, test_origin_indices(idx, n))]

        for window, period, origins in jobs:
            print(label if window is None else f"{label}  window {window}  {period}", flush=True)
            preds = forecast(pipeline, power, origins)
            f = build_frame(origins, preds, power, idx)

            out = {"zone": label, "model_id": model_id,
                   "context_length": CONTEXT, "zero_shot": True}
            if window is not None:
                out["window"] = window
                out["period"] = period
            out.update(score(f))

            family = ml.get(label, {}).get("model_family")
            if window is None and label in ml:
                out["trained_ml_model"] = {"family": family,
                                           "MAPE": ml[label]["test_refit"]["MAPE"]}
            elif window is not None and (label, window) in wf_ml:
                out["trained_ml_model"] = {"family": family,
                                           "MAPE": wf_ml[(label, window)]["model_MAPE"]}

            rows.append(out)
            c = out["chronos_vs_seasonal_naive"]
            trained = out.get("trained_ml_model", {}).get("MAPE", float("nan"))
            print(f"  chronos {out['chronos']['MAPE']:6.2f}%   "
                  f"seasonal naive {out['seasonal_naive']['MAPE']:6.2f}%   "
                  f"trained {trained:6.2f}%", flush=True)
            print(f"  diff {c['mean_diff_pp']:+.3f}pp   p = {c['dm_p']:.4f}   "
                  f"chronos wins: {c['chronos_wins']}\n", flush=True)
            del preds, f
            gc.collect()
        del power
        gc.collect()

    dest = WF_OUT if walkforward else OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("=" * 70)
    summary = pd.DataFrame([{
        "zone": r["zone"],
        "window": r.get("window", "test"),
        "seasonal_naive": r["seasonal_naive"]["MAPE"],
        "chronos_zeroshot": r["chronos"]["MAPE"],
        "trained_ml": r.get("trained_ml_model", {}).get("MAPE", float("nan")),
        "p": r["chronos_vs_seasonal_naive"]["dm_p"],
        "chronos_wins": r["chronos_vs_seasonal_naive"]["chronos_wins"],
    } for r in rows])
    print(summary.round(4).to_string(index=False))
    wins = int(summary["chronos_wins"].sum())
    print(f"\nChronos beats the seasonal naive in {wins} of {len(summary)}")
    print(f"Saved -> {dest}")


if __name__ == "__main__":
    main()
