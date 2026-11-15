"""Target-only Chronos-2 on the existing frozen cases; GPU inference is for Colab only."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chronos_long_context import CONTEXT, long_contexts

MODEL_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
PACKAGES = {"torch": "2.11.0", "chronos-forecasting": "2.2.2",
            "transformers": "4.57.6", "numpy": "2.1.3", "pandas": "2.2.3"}


def medians(pipeline, histories, horizon, batch_size):
    """Each 1-D array is a separate univariate task, never a shared multivariate task."""
    if horizon not in (24, 168):
        raise ValueError("Only the frozen 24- and 168-hour horizons are allowed")
    quantiles, _ = pipeline.predict_quantiles(
        histories, prediction_length=horizon, quantile_levels=[0.5],
        context_length=CONTEXT, batch_size=batch_size, cross_learning=False,
        limit_prediction_length=True)
    if not isinstance(quantiles, list) or len(quantiles) != len(histories):
        raise ValueError("Expected one Chronos-2 tensor per univariate task")
    output = []
    for tensor in quantiles:
        if tuple(tensor.shape) != (1, horizon, 1):
            raise ValueError(f"Unexpected Chronos-2 quantile shape: {tensor.shape}")
        output.append(tensor[0, :, 0].detach().float().cpu().numpy())
    result = np.stack(output)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite Chronos-2 forecasts")
    return result


def preflight(pipeline):
    """Small synthetic GPU check, including a 72-hour gap outside the latest 336 hours."""
    t = np.arange(2048, dtype=np.float32)
    full = 10 + np.sin(t * (2 * np.pi / 168))
    gapped = full.copy()
    gapped[100:172] = np.nan
    short = full[-1153:].copy()
    for horizon in (24, 168):
        together = medians(pipeline, [full, short, gapped], horizon, 3)
        alone = medians(pipeline, [full], horizon, 1)
        np.testing.assert_allclose(together[:1], alone, rtol=1e-4, atol=1e-4,
                                   err_msg="Batch isolation check failed")
        print(f"Preflight passed: h={horizon}, ragged/gapped histories, finite medians, batch isolation",
              flush=True)


def forecast(pipeline, values, origins, horizon, batch_size):
    predictions, lengths, missing = [], [], []
    began = time.monotonic()
    for start in range(0, len(origins), batch_size):
        histories = long_contexts(values, origins[start:start + batch_size])
        prediction = medians(pipeline, histories, horizon, batch_size)
        predictions.append(prediction)
        lengths.extend(map(len, histories))
        missing.extend(int(np.isnan(h).sum()) for h in histories)
        done = min(start + batch_size, len(origins))
        elapsed = time.monotonic() - began
        eta = elapsed * (len(origins) - done) / done
        print(f"  {done}/{len(origins)} origins ({100*done/len(origins):.1f}%), "
              f"elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min", flush=True)
    return np.concatenate(predictions), np.asarray(lengths), np.asarray(missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from baseline_pilot import file_hash
    from chronos_pilot import comparison, load_cases
    cases = load_cases(root)
    for record, series, saved in cases:
        if record["horizon_hours"] not in (24, 168):
            raise ValueError("Unexpected horizon")
        long_contexts(series.to_numpy(), saved["test_origins"])
    print(f"Verified {len(cases)} frozen series/horizon cases", flush=True)
    if args.check_only:
        return

    # Imported only after --check-only exits. No local torch import is needed.
    import torch
    from chronos import Chronos2Pipeline
    installed = {p: importlib.metadata.version(p) for p in PACKAGES}
    if any(installed[p].split("+")[0] != version for p, version in PACKAGES.items()):
        raise ValueError(f"Use the notebook's pinned environment: {installed}")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime")
    torch.manual_seed(20261021)
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID, revision=REVISION, device_map="cuda", torch_dtype=torch.float32)
    pipeline.model.eval()
    if pipeline.model_context_length < CONTEXT or pipeline.model_prediction_length < 168:
        raise RuntimeError("Pinned model does not support the planned context/horizon")
    preflight(pipeline)
    if args.smoke_only:
        return

    output = args.output or root / "chronos2_results"
    output.mkdir(parents=True, exist_ok=True)
    identity = {"model_id": MODEL_ID, "model_revision": REVISION, "context_hours": CONTEXT,
        "context_policy": "Up to 2048 hours ending at origin; shorter at start; gaps kept as NaN",
        "quantile": 0.5, "dtype": "float32", "batch_size": args.batch_size,
        "cross_learning": False, "covariates": False, "seed": 20261021,
        "manifest_sha256": file_hash(root / "manifest.json"),
        "runner_sha256": file_hash(Path(__file__)),
        "history_helper_sha256": file_hash(Path(__file__).with_name("chronos_long_context.py")),
        "packages": installed, "long_horizon_policy": "Native direct prediction; no observed future values"}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    metadata = output / "run_metadata.json"
    if metadata.exists():
        if json.loads(metadata.read_text())["identity"] != identity:
            raise ValueError("Output belongs to another configuration; choose a separate output folder")
    else:
        metadata.write_text(json.dumps({"identity": identity, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0), "started_utc": datetime.now(timezone.utc).isoformat(),
            "caveats": ["Exploratory follow-up on inspected cases", "Pretraining overlap not ruled out"]}, indent=2))
    all_rows = []
    for number, (record, series, saved) in enumerate(cases, 1):
        name = f"{record['dataset']}_{record['series']}_h{record['horizon_hours']}"
        path = output / f"{name}.npz"
        print(f"Case {number}/{len(cases)}: {name}", flush=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                if str(cached["run_fingerprint"]) != fingerprint:
                    raise ValueError("Cached forecast belongs to another configuration")
                for key, value in saved.items():
                    np.testing.assert_array_equal(cached[key], value)
                predicted = cached["chronos"]
            print("  Reused completed case", flush=True)
        else:
            predicted, lengths, missing = forecast(pipeline, series.to_numpy(), saved["test_origins"],
                                                   record["horizon_hours"], args.batch_size)
            temporary = path.with_suffix(".part")
            with temporary.open("wb") as stream:
                # 'chronos' is the existing comparison schema, identified as Chronos-2 by metadata.
                np.savez_compressed(stream, **saved, chronos=predicted, run_fingerprint=fingerprint,
                                    context_hours=lengths, context_missing_hours=missing)
            temporary.replace(path)
        all_rows.extend(comparison(record, saved, predicted))
        pd.DataFrame(all_rows).to_csv(output / "comparison.csv", index=False)
    (output / "complete.json").write_text(json.dumps({"cases": len(cases),
        "prediction_files": len(cases), "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
