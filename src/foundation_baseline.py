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

  origins       the same test origins as forecasting.split(), i.e. origin_time
                strictly after VAL_CUTOFF
  context       up to CONTEXT hours of history ending AT the origin, so nothing
                after the forecast origin is ever visible
  horizon       1..168 hours ahead, one forecast per origin
  point forecast the 0.5 quantile, which is the MAPE-appropriate summary
  baselines     seasonal_naive = y[target - 168], persistence = y[origin]

No training happens. There is no fitting stage, so there is no split to leak across;
the model simply never saw this series before. Tetouan does not appear in the
published Chronos pretraining corpus (autogluon/chronos_datasets).

torch and chronos are imported lazily, so validate_project.py can import the origin
and context logic below on a machine without them.

Run: python src/foundation_baseline.py [model_id]
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
OUT = Path("reports/foundation_baseline_results.json")
ML_RESULTS = Path("reports/results_corrected.json")


def test_origin_indices(idx, n):
    """Positions of the test origins, matching forecasting.split() exactly."""
    origins = np.arange(336, n - HORIZON)
    return origins[idx[origins] > VAL_CUTOFF]


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


def forecast_zone(pipeline, power, idx, n):
    """One 168-step forecast per test origin. Context ends at the origin."""
    import torch
    test_origins = test_origin_indices(idx, n)

    batch = 64 if torch.cuda.is_available() else BATCH
    preds = np.empty((len(test_origins), HORIZON), dtype=np.float32)
    t0 = time.time()
    for s in range(0, len(test_origins), batch):
        chunk = test_origins[s:s + batch]
        ctx = [torch.tensor(power[context_slice(o)], dtype=torch.float32) for o in chunk]
        # passed positionally: the argument is `context` in chronos 1.x, `inputs` in 2.x
        q, _ = pipeline.predict_quantiles(
            ctx, prediction_length=HORIZON, quantile_levels=[0.5]
        )
        preds[s:s + len(chunk)] = q[:, :, 0].to(torch.float32).cpu().numpy()
        if (s // batch) % 5 == 0:
            done = s + len(chunk)
            print(f"    {done}/{len(test_origins)} origins  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return test_origins, preds


def build_frame(test_origins, preds, power, idx):
    oi = np.repeat(test_origins, HORIZON)
    ha = np.tile(np.arange(1, HORIZON + 1, dtype=np.int16), len(test_origins))
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


def main():
    from significance_test import diebold_mariano, paired_bootstrap

    model_id = sys.argv[1] if len(sys.argv) > 1 else MODEL_ID
    print(f"Loading {model_id} (zero-shot, no training on this data)\n", flush=True)
    pipeline = load_pipeline(model_id)

    ml = {}
    if ML_RESULTS.exists():
        ml = json.loads(ML_RESULTS.read_text(encoding="utf-8")).get("zones", {})

    hdf = load_hourly()
    idx = hdf.index
    n = len(hdf)
    rows = []

    for col, label in ZONES.items():
        print(f"{label}", flush=True)
        power = hdf[col].values.astype(np.float32)
        test_origins, preds = forecast_zone(pipeline, power, idx, n)
        f = build_frame(test_origins, preds, power, idx)

        out = {
            "zone": label,
            "model_id": model_id,
            "context_length": CONTEXT,
            "zero_shot": True,
            "n_origins": int(len(test_origins)),
            "rows": int(len(f)),
            "effective_n": int(f["target_time"].nunique()),
            "chronos": metrics(f["y"], f["chronos"]),
            "seasonal_naive": metrics(f["y"], f["seasonal_naive"]),
            "persistence": metrics(f["y"], f["persistence"]),
        }
        if label in ml:
            out["trained_ml_model"] = {
                "family": ml[label]["model_family"],
                "MAPE": ml[label]["test_refit"]["MAPE"],
            }

        # per-day-ahead breakdown, the horizon story
        day = np.ceil(f["horizon"].values / 24).astype(int)
        by_day = {}
        for d in range(1, 8):
            k = day == d
            by_day[str(d)] = {
                "chronos_MAPE": metrics(f["y"].values[k], f["chronos"].values[k])["MAPE"],
                "seasonal_naive_MAPE": metrics(f["y"].values[k],
                                               f["seasonal_naive"].values[k])["MAPE"],
            }
        out["by_day_ahead"] = by_day

        # significance on the honest sample: one error per distinct target hour
        for c in ["chronos", "seasonal_naive"]:
            f[f"ape_{c}"] = (f[c] - f["y"]).abs() / f["y"] * 100
        g = f.groupby("target_time")[["ape_chronos", "ape_seasonal_naive"]].mean()
        stat, p, lag = diebold_mariano(g["ape_chronos"].values,
                                       g["ape_seasonal_naive"].values)
        mean_d, lo, hi, _ = paired_bootstrap(g["ape_chronos"].values,
                                             g["ape_seasonal_naive"].values)
        out["chronos_vs_seasonal_naive"] = {
            "mean_diff_pp": mean_d, "ci95": [lo, hi],
            "dm_stat": stat, "dm_p": p, "nw_lag": lag,
            "significant_at_05": bool(p < 0.05),
            "chronos_wins": bool(mean_d < 0),
        }

        rows.append(out)
        print(f"  chronos        {out['chronos']['MAPE']:6.2f}%", flush=True)
        print(f"  seasonal naive {out['seasonal_naive']['MAPE']:6.2f}%", flush=True)
        if "trained_ml_model" in out:
            print(f"  trained {out['trained_ml_model']['family']:<4s}   "
                  f"{out['trained_ml_model']['MAPE']:6.2f}%", flush=True)
        print(f"  diff {mean_d:+.3f}pp   p = {p:.4f}   "
              f"chronos wins: {out['chronos_vs_seasonal_naive']['chronos_wins']}\n",
              flush=True)

        del power, preds, f, g
        gc.collect()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("=" * 62)
    summary = pd.DataFrame([{
        "zone": r["zone"],
        "seasonal_naive": r["seasonal_naive"]["MAPE"],
        "chronos_zeroshot": r["chronos"]["MAPE"],
        "trained_ml": r.get("trained_ml_model", {}).get("MAPE", float("nan")),
        "chronos_wins": r["chronos_vs_seasonal_naive"]["chronos_wins"],
    } for r in rows])
    print(summary.round(2).to_string(index=False))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
