"""Recalculate v2 scores and verify matched cases, hour coverage and frozen v1 files."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_pilot import file_hash, score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/trained_comparison_v2"


def main():
    result = json.loads((OUT / "results.json").read_text())
    if not result["complete"]:
        raise ValueError("V2 training is not complete")
    v1 = json.loads((ROOT / "reports/trained_comparison/results.json").read_text())
    expected = {(r["window"], r["dataset"], r["series"], r["horizon_hours"]) for r in v1["cases"]}
    actual_cases = {(r["window"], r["dataset"], r["series"], r["horizon_hours"]) for r in result["cases"]}
    assert actual_cases == expected
    identity = result["identity"]
    assert identity["v1_results_sha256"] == file_hash(ROOT / "reports/trained_comparison/results.json")
    assert identity["v1_scores_sha256"] == file_hash(ROOT / "reports/trained_comparison/scores.csv")
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    scores = pd.read_csv(OUT / "scores.csv")
    assert len(scores) == 6 * len(expected)
    with zipfile.ZipFile(ROOT / "expanded_chronos_results.zip") as chronos:
        for details in result["cases"]:
            window, dataset, series, horizon = (details[k] for k in ("window", "dataset", "series", "horizon_hours"))
            assert details["fingerprint"] == fingerprint
            assert set(details["train_origin_hour_counts"]) == {str(i) for i in range(24)}
            assert set(details["refit_origin_hour_counts"]) == {str(i) for i in range(24)}
            assert sum(details["train_origin_hour_counts"].values()) == details["train_origins"]
            assert sum(details["refit_origin_hour_counts"].values()) == details["refit_origins"]
            assert pd.Timestamp(details["train_last_target"]) < pd.Timestamp(details["validation_first_origin"])
            assert pd.Timestamp(details["refit_last_target"]) < pd.Timestamp(details["test_first_origin"])
            for family in ("ridge", "lightgbm"):
                choice = details["choices"][family]
                assert choice["validation_MAE"] == min(t["validation_MAE"] for t in choice["trials"])
            selected = min(details["choices"], key=lambda family: details["choices"][family]["validation_MAE"])
            assert details["selected_trained"] == selected
            name = f"{dataset}_{series}_h{horizon}.npz"
            with np.load(OUT / "predictions" / window / name, allow_pickle=False) as saved, np.load(io.BytesIO(chronos.read(window + "/" + name)), allow_pickle=False) as original:
                for key in original.files:
                    if key != "run_fingerprint":
                        np.testing.assert_array_equal(saved[key], original[key])
                np.testing.assert_array_equal(saved["selected_trained"], saved[selected])
                subset = scores[(scores.window == window) & (scores.dataset == dataset)
                    & (scores.series == series) & (scores.horizon_hours == horizon)]
                for _, row in subset.iterrows():
                    method = str(saved["selected_on_validation"]) if row.method == "selected_baseline" else row.method
                    assert saved[method].shape == saved["actual"].shape
                    assert np.isfinite(saved[method]).all()
                    for metric, value in score(saved["actual"], saved[method]).items():
                        if value is None:
                            assert pd.isna(row[metric])
                        else:
                            np.testing.assert_allclose(row[metric], value, rtol=1e-12, atol=1e-12)
    summary = {"verified_cases": len(expected), "verified_score_rows": len(scores),
        "v1_hashes_preserved": True, "all_origin_hours_present": True,
        "time_boundaries": "passed", "selection": "validation only",
        "original_chronos_and_baseline_arrays": "exact match", "score_recalculation": "passed"}
    (OUT / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for horizon in (24, 168):
        table = scores[scores.horizon_hours == horizon].pivot(index=["window", "dataset", "series"], columns="method", values="MAE")
        print(f"\nHorizon {horizon}")
        for method in ("ridge", "lightgbm", "selected_trained", "selected_baseline"):
            ties = np.isclose(table.chronos, table[method], rtol=1e-9, atol=1e-10)
            wins = int(((table.chronos < table[method]) & ~ties).sum())
            losses = int(((table.chronos > table[method]) & ~ties).sum())
            print(f"Chronos vs {method}: {wins} wins, {losses} losses, {int(ties.sum())} ties")


if __name__ == "__main__":
    main()
