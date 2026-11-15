"""Chronos-Bolt on the frozen expanded cases with up to 2048 hours of history.

Only the history length differs from the 336-hour run: same model commit, cases,
0.5 quantile, baselines and output layout. When the series starts less than 2048
hours before an origin (Q1), all available history is used. Missing hours inside
the history stay NaN, which Chronos-Bolt masks; the latest 336 hours are complete
for every frozen case.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ID = "amazon/chronos-bolt-base"
REVISION = "5d9f166d69f47aef3401367a7b842e78fe97b121"
CONTEXT = 2048
COMPLETE_HOURS = 336


def long_contexts(values, origins, length=CONTEXT):
    """History ending at each origin: up to `length` hours, shorter near the series start."""
    values = np.asarray(values, dtype=np.float32)
    batch = []
    for origin in np.asarray(origins, dtype=int):
        if origin < COMPLETE_HOURS - 1:
            raise ValueError("Insufficient context")
        history = values[max(0, origin - length + 1):origin + 1]
        if not np.isfinite(history[-COMPLETE_HOURS:]).all():
            raise ValueError("Latest 336 hours must be complete")
        batch.append(history)
    return batch


def forecast(pipeline, values, origins, horizon, batch_size):
    import torch
    predictions, lengths, missing = [], [], []
    for start in range(0, len(origins), batch_size):
        batch = long_contexts(values, origins[start:start + batch_size])
        lengths += [len(h) for h in batch]
        missing += [int(np.isnan(h).sum()) for h in batch]
        with torch.inference_mode():
            quantiles, _ = pipeline.predict_quantiles(
                [torch.from_numpy(h) for h in batch], prediction_length=horizon,
                quantile_levels=[0.5], limit_prediction_length=False)
        prediction = quantiles[:, :, 0].float().cpu().numpy()
        if prediction.shape != (len(batch), horizon) or not np.isfinite(prediction).all():
            raise ValueError("Unexpected model output")
        predictions.append(prediction)
        print(f"  {min(start + batch_size, len(origins))}/{len(origins)} origins", flush=True)
    return np.concatenate(predictions), np.array(lengths), np.array(missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="One window folder from the inputs ZIP")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    root = args.root.resolve()
    # Use the window's frozen copies of the case loader and scoring code.
    sys.path.insert(0, str(root / "src"))
    from baseline_pilot import file_hash
    from chronos_pilot import comparison, load_cases
    cases = load_cases(root)
    for _, series, saved in cases:
        long_contexts(series.to_numpy(), saved["test_origins"])
    print(f"Verified {len(cases)} series/horizon cases", flush=True)
    if args.check_only:
        return

    import torch
    from chronos import BaseChronosPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before running inference")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID, revision=REVISION, device_map="cuda", torch_dtype=torch.float32)
    pipeline.model.eval()
    # Fail early if this package version rejects ragged or gapped histories.
    probe = [torch.ones(100), torch.tensor([1.0, float("nan")] * 30)]
    with torch.inference_mode():
        check, _ = pipeline.predict_quantiles(probe, prediction_length=24, quantile_levels=[0.5])
    if check.shape[:2] != (2, 24) or not torch.isfinite(check).all():
        raise RuntimeError("Chronos did not accept variable-length histories with gaps")

    output = args.output or root / "results"
    output.mkdir(parents=True, exist_ok=True)
    identity = {"model_id": MODEL_ID, "model_revision": REVISION, "context_hours": CONTEXT,
                "context_policy": "Up to 2048 hours; shorter near series start; missing hours passed as NaN",
                "quantile": 0.5, "dtype": "float32", "batch_size": args.batch_size,
                "manifest_sha256": file_hash(root / "manifest.json"),
                "runner_sha256": file_hash(Path(__file__)),
                "long_horizon_policy": "Chronos-Bolt built-in extension beyond native 64 steps; no observed future values",
                "packages": {p: importlib.metadata.version(p) for p in ("torch", "chronos-forecasting", "transformers", "numpy", "pandas")}}
    metadata_file = output / "run_metadata.json"
    if metadata_file.exists():
        if json.loads(metadata_file.read_text())["identity"] != identity:
            raise ValueError("Output belongs to a different run configuration. Choose a new output directory.")
    else:
        metadata_file.write_text(json.dumps({"identity": identity, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0), "started_utc": datetime.now(timezone.utc).isoformat(),
            "caveats": ["No fine-tuning; pretraining overlap not ruled out",
                        "Same frozen cases as the 336-hour run; only history length differs"]}, indent=2))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    all_rows = []
    for record, series, saved in cases:
        name = f"{record['dataset']}_{record['series']}_h{record['horizon_hours']}"
        path = output / f"{name}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                if str(cached["run_fingerprint"]) != fingerprint:
                    raise ValueError("Cached predictions belong to another configuration")
                np.testing.assert_array_equal(cached["test_origins"], saved["test_origins"])
                predicted = cached["chronos"]
            print(f"Resuming: loaded {name}", flush=True)
        else:
            print(f"Forecasting {name}", flush=True)
            predicted, lengths, missing = forecast(pipeline, series.to_numpy(), saved["test_origins"],
                                                   record["horizon_hours"], args.batch_size)
            temp = path.with_suffix(".part")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, **saved, chronos=predicted, run_fingerprint=fingerprint,
                                    context_hours=lengths, context_missing_hours=missing)
            temp.replace(path)
        all_rows.extend(comparison(record, saved, predicted))
        pd.DataFrame(all_rows).to_csv(output / "comparison.csv", index=False)
    (output / "complete.json").write_text(json.dumps({"cases": len(cases), "prediction_files": len(cases),
        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"Complete: {output}")


if __name__ == "__main__":
    main()
