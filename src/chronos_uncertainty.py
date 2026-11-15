"""Exploratory paired circular-block bootstrap of frozen Chronos forecast losses."""
import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def calendar_losses(origins, actual, chronos, baseline):
    """One mean loss per origin; retain missing calendar hours, never close gaps."""
    origins = np.asarray(origins, dtype=int)
    if len(origins) < 2 or not np.all(np.diff(origins) > 0):
        raise ValueError("Need increasing unique origins")
    if actual.shape != chronos.shape or actual.shape != baseline.shape:
        raise ValueError("Forecast shape mismatch")
    if not all(np.isfinite(a).all() for a in (actual, chronos, baseline)):
        raise ValueError("Nonfinite forecasts")
    grid = np.full((origins[-1] - origins[0] + 1, 2), np.nan)
    grid[origins - origins[0], 0] = np.abs(chronos - actual).mean(axis=1)
    grid[origins - origins[0], 1] = np.abs(baseline - actual).mean(axis=1)
    return grid


def circular_indices(n, block, repetitions, seed):
    if not 1 <= block <= n:
        raise ValueError("Block must fit calendar grid")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n / block))))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(repetitions, -1)[:, :n]


def paired_interval(grid, block, repetitions, seed):
    if block > len(grid):
        return {"block_hours": block, "difference_ci95": [None, None],
                "gain_pct_ci95": None, "valid_replicates": 0,
                "zero_baseline_replicates": 0, "at_least_two_blocks_of_valid_origins": False,
                "interval_direction": "insufficient_calendar_span"}
    indices = circular_indices(len(grid), block, repetitions, seed)
    valid = np.isfinite(grid).all(axis=1)
    counts = valid[indices].sum(axis=1)
    keep = counts > 0
    sampled = np.nan_to_num(grid)[indices[keep]].sum(axis=1) / counts[keep, None]
    # Positive difference means Chronos has lower mean absolute error.
    differences = sampled[:, 1] - sampled[:, 0]
    lo, hi = np.percentile(differences, [2.5, 97.5])
    positive_denominator = sampled[:, 1] > 0
    # Never publish a conditional ratio interval after silently dropping zero baselines.
    gain_interval = None
    if positive_denominator.all():
        gain_interval = np.percentile(100 * (1 - sampled[:, 0] / sampled[:, 1]), [2.5, 97.5]).tolist()
    enough = int(valid.sum()) >= 2 * block
    direction = "chronos" if lo > 0 else "baseline" if hi < 0 else "unclear"
    return {"block_hours": block, "difference_ci95": [float(lo), float(hi)],
            "gain_pct_ci95": gain_interval, "valid_replicates": int(keep.sum()),
            "zero_baseline_replicates": int((~positive_denominator).sum()),
            "at_least_two_blocks_of_valid_origins": enough,
            "interval_direction": direction}


def analyze(archive_path, inputs_path, output, repetitions=5000, blocks=(168, 336)):
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"method": "Paired circular block bootstrap, percentile intervals",
        "repetitions": repetitions, "seed": 20261019, "block_hours": list(blocks),
        "primary_block_hours": 168, "loss": "Mean absolute error over horizon per forecast origin",
        "difference": "baseline MAE minus Chronos MAE; positive favors Chronos",
        "calendar_gaps": "Retain invalid origins as missing hourly positions; resample common missingness mask",
        "selection": "Conditional on original validation-selected rule and weight; not refitted within bootstrap",
        "interpretation": "Exploratory per-comparison 95% intervals; no multiplicity correction or population-generalization claim",
        "sensitivity": "Report both lengths without selecting the more favorable interval",
        "sparse_guard": "Flag intervals with fewer than two blocks worth of valid origins; do not call those supported",
        "limitations": ["Only about five 168-hour blocks or 2.5 336-hour blocks per complete window",
            "Circular wrapping joins end and start; assumes approximately stationary paired loss process within each window",
            "Missingness-mask resampling is exploratory, not proof gaps are ignorable",
            "Does not quantify pretraining overlap, model/selection uncertainty or future-year performance",
            "Q4 repeats original pilot; windows and related meters are not independent replications"],
        "reference": "https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html",
        "results_zip_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "inputs_zip_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest()}
    # Method settings recorded before inspecting bootstrap outcomes.
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results = []
    with zipfile.ZipFile(archive_path) as archive, zipfile.ZipFile(inputs_path) as inputs:
        for window in ("Q1", "Q2", "Q3", "Q4"):
            complete = json.loads(archive.read(window + "/complete.json"))
            identity = json.loads(archive.read(window + "/run_metadata.json"))["identity"]
            assert identity["manifest_sha256"] == hashlib.sha256(inputs.read(window + "/manifest.json")).hexdigest()
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            names = sorted(n for n in archive.namelist() if n.startswith(window + "/") and n.endswith(".npz"))
            assert len(names) == complete["cases"]
            for name in names:
                basename = name.split("/")[-1]
                dataset, rest = basename.split("_", 1)
                series, horizon = rest.rsplit("_h", 1)
                horizon = int(horizon[:-4])
                with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as saved, np.load(io.BytesIO(inputs.read(window + "/reports/baseline_pilot/predictions/" + basename)), allow_pickle=False) as frozen:
                    for key in frozen.files:
                        np.testing.assert_array_equal(saved[key], frozen[key])
                    assert str(saved["run_fingerprint"]) == fingerprint
                    for comparator in ("weekly", "selected"):
                        rule = str(saved["selected_on_validation"]) if comparator == "selected" else "weekly"
                        grid = calendar_losses(saved["test_origins"], saved["actual"], saved["chronos"], saved[rule])
                        means = np.nanmean(grid, axis=0)
                        label = f"{window}/{dataset}/{series}/{horizon}"
                        seed = 20261019 + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
                        intervals = [paired_interval(grid, block, repetitions, seed + block) for block in blocks]
                        directions = {r["interval_direction"] for r in intervals}
                        if not all(r["at_least_two_blocks_of_valid_origins"] for r in intervals):
                            conclusion = "limited_data"
                        elif directions == {"chronos"}:
                            conclusion = "chronos_supported_both_lengths"
                        elif directions == {"baseline"}:
                            conclusion = "baseline_supported_both_lengths"
                        else:
                            conclusion = "unclear_or_block_sensitive"
                        record = {"window": window, "dataset": dataset, "series": series,
                            "horizon_hours": horizon, "comparator": comparator, "baseline_rule": rule,
                            "test_origins": len(saved["test_origins"]), "calendar_origin_hours": len(grid),
                            "chronos_MAE": float(means[0]), "baseline_MAE": float(means[1]),
                            "difference": float(means[1] - means[0]),
                            "gain_pct": float(100*(1-means[0]/means[1])) if means[1] else None,
                            "intervals": intervals, "assessment": conclusion}
                        results.append(record)
                print(f"Analyzed {label}", flush=True)
    (output / "results.json").write_text(json.dumps({"protocol": protocol,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "results": results}, indent=2, allow_nan=False), encoding="utf-8")
    rows = [{**{k: v for k, v in r.items() if k != "intervals"},
             **{f"block_{i['block_hours']}_difference_{bound}": i["difference_ci95"][j]
                for i in r["intervals"] for j, bound in enumerate(("lo", "hi"))}}
            for r in results]
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    lines = ["# Chronos paired uncertainty follow-up", "",
        "Exploratory circular-block bootstrap of origin-level MAE, 5,000 resamples at each of 168 and 336 calendar hours. Positive differences favor Chronos. Missing calendar hours remain gaps. Intervals are conditional on frozen baseline choices and model predictions. No familywise or false-discovery correction is applied.", "",
        "The windows are short: a complete window contains only five week-long blocks. Circular resampling wraps the end to the start and assumes approximately stationary losses within a window. Treat intervals as approximate sensitivity evidence, not definitive proof. Sparse household windows receive a limited-data label. Q4 repeats the pilot.", "",
        "## Seven-day comparisons against validation-selected baselines", "",
        "Difference intervals use each series' original units; do not compare their magnitudes across datasets.", "",
        "| Window | Series | Gain | 168h difference interval | 336h difference interval | Assessment |",
        "|---|---|---:|---|---|---|"]
    for r in results:
        if r["horizon_hours"] != 168 or r["comparator"] != "selected":
            continue
        gain = "undefined (zero baseline)" if r["gain_pct"] is None else f"{r['gain_pct']:+.2f}%"
        ci = ["not estimable (short calendar span)" if i["difference_ci95"][0] is None
              else f"[{i['difference_ci95'][0]:.4g}, {i['difference_ci95'][1]:.4g}]" for i in r["intervals"]]
        lines.append(f"| {r['window']} | {r['dataset']}/{r['series']} | {gain} | {ci[0]} | {ci[1]} | {r['assessment']} |")
    lines += ["", "Method reference: [arch time-series bootstrap documentation](https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html). This script implements the index resampling using NumPy and does not require arch.", "",
        "Reproduce with `python src/chronos_uncertainty.py`. No new model inference is performed.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(pd.DataFrame(rows).query("comparator == 'selected'").groupby(["horizon_hours", "assessment"]).size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5000)
    args = parser.parse_args()
    if args.repetitions < 100:
        parser.error("Use at least 100 resamples")
    analyze(ROOT / "expanded_chronos_results.zip", ROOT / "reports/expanded_chronos/expanded_chronos_inputs.zip",
            ROOT / "reports/chronos_uncertainty", args.repetitions)
