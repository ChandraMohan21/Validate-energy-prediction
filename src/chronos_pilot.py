"""Evaluate Chronos-Bolt on frozen pilot cases. Torch is imported only for inference."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_pilot import file_hash, origin_indices, reference_predictions, score

MODEL_ID = "amazon/chronos-bolt-base"
CONTEXT = 336


def contexts(values, origins):
    indices = np.asarray(origins)[:, None] + np.arange(-CONTEXT + 1, 1)
    if (indices < 0).any():
        raise ValueError("Insufficient context")
    result = np.asarray(values)[indices].astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Context contains missing values")
    return result


def load_cases(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Input checksum mismatch: {name}")
    protocol = json.loads((root / "reports/baseline_pilot/protocol.json").read_text())
    report = json.loads((root / "reports/baseline_pilot/results.json").read_text())
    if file_hash(root / "reports/baseline_pilot/protocol.json") != report["metadata"]["protocol_sha256"]:
        raise ValueError("Pilot protocol changed")
    cases = []
    for record in report["results"]:
        year = protocol["years"][record["dataset"]]
        path = root / f"data/processed/baseline_pilot/{record['dataset']}_{record['series']}_{year}.csv"
        if file_hash(path) != record["processed_sha256"]:
            raise ValueError("Processed input differs from baseline experiment")
        series = pd.read_csv(path, index_col="Datetime", parse_dates=True, float_precision="round_trip")["load"]
        with np.load(root / record["prediction_file"], allow_pickle=False) as archive:
            saved = {k: archive[k] for k in archive.files}
        val, test, _ = origin_indices(series, protocol)
        np.testing.assert_array_equal(test, saved["test_origins"])
        np.testing.assert_array_equal(val, saved["validation_origins"])
        np.testing.assert_array_equal(series.index.to_numpy(), saved["hourly_time"])
        predictions, actual, _ = reference_predictions(series, test, record["horizon_hours"])
        np.testing.assert_array_equal(actual, saved["actual"])
        for name, prediction in predictions.items():
            np.testing.assert_array_equal(prediction, saved[name])
        alpha = record["blend_daily_weight"]
        np.testing.assert_array_equal(alpha * predictions["daily"] + (1-alpha) * predictions["weekly"], saved["blend"])
        if str(saved["selected_on_validation"]) != record["selected_on_validation"]:
            raise ValueError("Baseline selection changed")
        contexts(series.to_numpy(), test)
        cases.append((record, series, saved))
    return cases


def forecast(pipeline, values, origins, horizon, batch_size):
    import torch
    predictions = []
    for start in range(0, len(origins), batch_size):
        batch = contexts(values, origins[start:start + batch_size])
        with torch.inference_mode():
            quantiles, _ = pipeline.predict_quantiles(
                torch.from_numpy(batch), prediction_length=horizon,
                quantile_levels=[0.5], limit_prediction_length=False)
        prediction = quantiles[:, :, 0].float().cpu().numpy()
        if prediction.shape != (len(batch), horizon) or not np.isfinite(prediction).all():
            raise ValueError("Unexpected model output")
        predictions.append(prediction)
        print(f"  {min(start + batch_size, len(origins))}/{len(origins)} origins", flush=True)
    return np.concatenate(predictions)


def comparison(record, saved, predicted):
    actual = saved["actual"]
    if predicted.shape != actual.shape or not np.isfinite(predicted).all():
        raise ValueError("Prediction shape/finiteness mismatch")
    metrics = score(actual, predicted)
    rows = []
    for baseline in ("daily", "weekly", "hybrid", "blend", "selected"):
        name = record["selected_on_validation"] if baseline == "selected" else baseline
        reference = score(actual, saved[name])
        rows.append({"dataset": record["dataset"], "series": record["series"],
                     "horizon_hours": record["horizon_hours"], "unit": record["unit"],
                     "baseline": baseline, "baseline_rule": name,
                     "test_origins": len(saved["test_origins"]),
                     "chronos_MAE": metrics["MAE"], "baseline_MAE": reference["MAE"],
                     "chronos_MAE_gain_pct": 100*(1-metrics["MAE"]/reference["MAE"]) if reference["MAE"] else None,
                     **{f"chronos_{key}": value for key, value in metrics.items() if key != "MAE"}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    root = args.root.resolve()
    cases = load_cases(root)
    print(f"Verified {len(cases)} series/horizon cases", flush=True)
    if args.check_only:
        return

    import torch
    from chronos import BaseChronosPipeline
    from huggingface_hub import HfApi
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before running inference")
    revision = HfApi().model_info(MODEL_ID).sha
    # float32 avoids unsupported bfloat16 on Colab T4; small batches limit long-horizon expansion.
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID, revision=revision, device_map="cuda", torch_dtype=torch.float32)
    pipeline.model.eval()
    output = args.output or root / "reports/chronos_pilot"
    output.mkdir(parents=True, exist_ok=True)
    identity = {"model_id": MODEL_ID, "model_revision": revision, "context_hours": CONTEXT,
                "quantile": 0.5, "dtype": "float32", "batch_size": args.batch_size,
                "manifest_sha256": file_hash(root / "manifest.json"),
                "runner_sha256": file_hash(Path(__file__)),
                "long_horizon_policy": "Chronos-Bolt built-in extension beyond native 64 steps; no observed future values",
                "packages": {p: importlib.metadata.version(p) for p in ("torch", "chronos-forecasting", "transformers", "numpy", "pandas")}}
    metadata_file = output / "run_metadata.json"
    if metadata_file.exists():
        old = json.loads(metadata_file.read_text())
        if old["identity"] != identity:
            raise ValueError("Output belongs to a different run configuration. Choose a new output directory.")
    else:
        metadata_file.write_text(json.dumps({"identity": identity, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0), "started_utc": datetime.now(timezone.utc).isoformat(),
            "caveats": ["No fine-tuning; pretraining overlap not ruled out", "Overlapping forecasts are dependent; no significance claim", "336-hour context differs from older 2048-hour Tetouan run"]}, indent=2))
    all_rows = []
    for record, series, saved in cases:
        name = f"{record['dataset']}_{record['series']}_h{record['horizon_hours']}"
        path = output / f"{name}.npz"
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                if str(cached["run_fingerprint"]) != fingerprint:
                    raise ValueError("Cached predictions belong to another configuration")
                np.testing.assert_array_equal(cached["test_origins"], saved["test_origins"])
                np.testing.assert_array_equal(cached["actual"], saved["actual"])
                predicted = cached["chronos"]
            print(f"Resuming: loaded {name}", flush=True)
        else:
            print(f"Forecasting {name}", flush=True)
            predicted = forecast(pipeline, series.to_numpy(), saved["test_origins"], record["horizon_hours"], args.batch_size)
            # Atomic replace prevents an interrupted write from looking complete.
            temp = path.with_suffix(".part")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, **saved, chronos=predicted, run_fingerprint=fingerprint)
            temp.replace(path)
        all_rows.extend(comparison(record, saved, predicted))
        pd.DataFrame(all_rows).to_csv(output / "comparison.csv", index=False)
    (output / "complete.json").write_text(json.dumps({"cases": len(cases), "prediction_files": len(cases),
        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(pd.DataFrame(all_rows).query("baseline == 'selected'").to_string(index=False))
    print(f"Complete: {output}")


if __name__ == "__main__":
    main()
