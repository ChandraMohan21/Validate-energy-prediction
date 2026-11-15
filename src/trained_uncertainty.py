"""Paired block-bootstrap sensitivity checks for the frozen trained-model follow-up."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_pilot import file_hash
from chronos_uncertainty import calendar_losses, paired_interval

ROOT = Path(__file__).resolve().parents[1]
TRAINED = ROOT / "reports/trained_comparison"
OUT = ROOT / "reports/trained_uncertainty"


def assessment(intervals):
    if not all(row["at_least_two_blocks_of_valid_origins"] for row in intervals):
        return "limited_data"
    directions = {row["interval_direction"] for row in intervals}
    if directions == {"chronos"}:
        return "chronos_supported_both_lengths"
    if directions == {"baseline"}:
        return "trained_supported_both_lengths"
    return "unclear_or_block_sensitive"


def case_results(details, saved, repetitions=5000):
    label = "/".join(str(details[key]) for key in ("window", "dataset", "series", "horizon_hours"))
    seed = 20261020 + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
    results, cache = [], {}
    for comparator in ("ridge", "lightgbm", "selected_trained"):
        rule = details["selected_trained"] if comparator == "selected_trained" else comparator
        if rule not in cache:
            grid = calendar_losses(saved["test_origins"], saved["actual"], saved["chronos"], saved[rule])
            means = np.nanmean(grid, axis=0)
            intervals = [paired_interval(grid, block, repetitions, seed + block) for block in (168, 336)]
            cache[rule] = {"chronos_MAE": float(means[0]), "trained_MAE": float(means[1]),
                "difference": float(means[1] - means[0]),
                "gain_pct": float(100 * (1 - means[0]/means[1])) if means[1] else None,
                "test_origins": len(saved["test_origins"]), "calendar_origin_hours": len(grid),
                "intervals": intervals, "assessment": assessment(intervals)}
        results.append({**{key: details[key] for key in ("window", "dataset", "series", "horizon_hours")},
                        "comparator": comparator, "trained_rule": rule, **cache[rule]})
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fitted = json.loads((TRAINED / "results.json").read_text())
    if not fitted["complete"] or len(fitted["cases"]) != fitted["completed_cases"]:
        raise ValueError("Trained comparison is incomplete")
    fingerprint = hashlib.sha256(json.dumps(fitted["identity"], sort_keys=True).encode()).hexdigest()
    protocol = {"created_date": "2026-10-20", "method": "Paired circular block bootstrap percentile intervals",
        "repetitions": 5000, "block_hours": [168, 336], "primary_block_hours": 168,
        "seed": 20261020, "comparators": ["ridge", "lightgbm", "selected_trained"],
        "loss": "Mean absolute error per forecast origin, averaged across all horizon steps",
        "positive_difference": "Trained MAE minus Chronos MAE; positive favors Chronos",
        "missingness": "Preserve calendar gaps, resample the same blocks and missingness mask for both models",
        "selection": "Conditional on fitted models and validation-selected settings/family; do not retrain or reselect in bootstrap",
        "assessment": "Support only if both block lengths give same nonzero direction and at least two blocks of valid origins exist; otherwise unclear or limited data",
        "limitations": ["Exploratory per-comparison 95% intervals without multiplicity correction",
            "Few long blocks per short window; circular wrap and within-window approximate stationarity assumed",
            "Intervals do not capture training/selection uncertainty, pretraining contamination or new-year generalization",
            "Q1 unavailable for trained comparison; sparse household windows flagged; Q4 repeats pilot",
            "Not an exhaustive search of Ridge or LightGBM configurations"],
        "trained_results_sha256": file_hash(TRAINED / "results.json"),
        "trained_scores_sha256": file_hash(TRAINED / "scores.csv"),
        "script_sha256": file_hash(Path(__file__)),
        "bootstrap_module_sha256": file_hash(Path(__file__).with_name("chronos_uncertainty.py")),
        "reference": "https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html"}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    scores = pd.read_csv(TRAINED / "scores.csv")
    records = []
    source_hashes = {}
    for detail in fitted["cases"]:
        name = f"{detail['dataset']}_{detail['series']}_h{detail['horizon_hours']}.npz"
        source = TRAINED / "predictions" / detail["window"] / name
        if detail["fingerprint"] != fingerprint:
            raise ValueError("Training configuration mismatch")
        source_hashes[str(source.relative_to(ROOT)).replace("\\", "/")] = file_hash(source)
        with np.load(source, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["selected_trained"], saved[detail["selected_trained"]])
            results = case_results(detail, saved)
        for row in results:
            common = scores[(scores.window == row["window"]) & (scores.dataset == row["dataset"])
                & (scores.series == row["series"]) & (scores.horizon_hours == row["horizon_hours"])]
            np.testing.assert_allclose(row["chronos_MAE"], common.loc[common.method == "chronos", "MAE"].item(), rtol=1e-12)
            np.testing.assert_allclose(row["trained_MAE"], common.loc[common.method == row["comparator"], "MAE"].item(), rtol=1e-12)
        records.extend(results)
        print(f"Analyzed {detail['window']}/{name}", flush=True)
    (OUT / "results.json").write_text(json.dumps({"protocol": protocol,
        "prediction_sha256": source_hashes, "results": records}, indent=2, allow_nan=False), encoding="utf-8")
    rows = [{**{key: value for key, value in row.items() if key != "intervals"},
             **{f"block_{interval['block_hours']}_difference_{bound}": interval["difference_ci95"][i]
                for interval in row["intervals"] for i, bound in enumerate(("lo", "hi"))}}
            for row in records]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "summary.csv", index=False)
    counts = frame.groupby(["horizon_hours", "comparator", "assessment"]).size().rename("comparisons").reset_index()
    counts.to_csv(OUT / "counts.csv", index=False)
    lines = ["# Uncertainty: Chronos versus trained models", "",
        "Exploratory paired circular-block bootstrap: 5,000 resamples using 168- and 336-hour blocks. All saved predictions are held fixed. Positive difference means lower Chronos MAE. Calendar gaps remain missing. No new model training is performed. No multiplicity adjustment is applied, and short windows provide few long blocks. These results are conditional sensitivity evidence, not universal model rankings.", "",
        "Q1 is excluded under the original frozen split. Household periods receive limited-data labels. The selected trained family was chosen by validation MAE before refitting, never by test error; it duplicates one family rather than providing an independent comparison.", "",
        "## Counts", "", "| Horizon | Comparator | Assessment | Count |", "|---|---|---|---:|"]
    for _, row in counts.iterrows():
        lines.append(f"| {row.horizon_hours} | {row.comparator} | {row.assessment} | {row.comparisons} |")
    lines += ["", "## Seven-day comparisons with LightGBM", "",
        "95% intervals below are gain percentages at the primary 168-hour block length. Assessment considers both lengths. Positive gain favors Chronos.", "",
        "| Period | Series | Observed gain | Approximate gain interval | Assessment |", "|---|---|---:|---|---|"]
    for row in records:
        if row["horizon_hours"] != 168 or row["comparator"] != "lightgbm":
            continue
        interval = row["intervals"][0]["gain_pct_ci95"]
        gain = "undefined (zero comparator MAE)" if row["gain_pct"] is None else f"{row['gain_pct']:+.2f}%"
        ci = "undefined" if interval is None else f"[{interval[0]:+.2f}%, {interval[1]:+.2f}%]"
        lines.append(f"| {row['window']} | {row['dataset']}/{row['series']} | {gain} | {ci} | {row['assessment']} |")
    lines += ["", "Reproduce: `python src/trained_uncertainty.py`. Full intervals for both lengths, all comparators and all 46 cases are in results.json and summary.csv.", "",
        "Method reference: [arch time-series bootstrap documentation](https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html).", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
