"""Independent consistency checks for the 60-case trained comparison v3."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_pilot import score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "trained_comparison_v3"
KEYS = ("window", "dataset", "series", "horizon_hours")


def main():
    details = sorted((OUT / "predictions").glob("Q*/*.json"))
    if len(details) != 60 or json.loads((OUT / "complete.json").read_text())["cases"] != 60:
        raise AssertionError("Expected 60 complete cases")
    scores = pd.read_csv(OUT / "scores.csv")
    if len(scores) != 60 * 7:
        raise AssertionError("Expected seven scores for each case")
    checks = 0
    for detail_file in details:
        detail = json.loads(detail_file.read_text())
        with np.load(detail_file.with_suffix(".npz"), allow_pickle=False) as saved:
            actual = saved["actual"]
            origins = saved["test_origins"]
            if actual.shape[0] != len(origins) or actual.shape[1] != detail["horizon_hours"]:
                raise AssertionError(f"Wrong target shape: {detail_file}")
            if not np.all(np.diff(origins) > 0):
                raise AssertionError(f"Test origins out of order: {detail_file}")
            if not (pd.Timestamp(detail["train_last_target"]) < pd.Timestamp(detail["validation_first_origin"])
                    and pd.Timestamp(detail["refit_last_target"]) < pd.Timestamp(detail["test_first_origin"])):
                raise AssertionError(f"Labels cross a boundary: {detail_file}")
            np.testing.assert_array_equal(saved["selected_trained"], saved[detail["selected_trained"]])
            for method in ("ridge", "lightgbm", "selected_trained", "selected_seasonal", "weekly", "bolt", "chronos2"):
                prediction = saved[method]
                if prediction.shape != actual.shape or not np.isfinite(prediction).all():
                    raise AssertionError(f"Bad prediction: {detail_file} {method}")
                match = scores.loc[
                    (scores.window == detail["window"]) &
                    (scores.dataset == detail["dataset"]) &
                    (scores.series == detail["series"]) &
                    (scores.horizon_hours == detail["horizon_hours"]) &
                    (scores.method == method), "MAE"]
                if len(match) != 1:
                    raise AssertionError(f"Missing or duplicated score: {detail_file} {method}")
                np.testing.assert_allclose(score(actual, prediction)["MAE"], match.item(), rtol=1e-11, atol=1e-12)
                checks += 1
    uncertainty = json.loads((OUT / "uncertainty.json").read_text())["results"]
    if len(uncertainty) != 60 * 2 * 3:
        raise AssertionError("Expected both models against three trained choices for every case")
    frame = pd.DataFrame(uncertainty)
    if frame.duplicated(list(KEYS) + ["model", "comparator"]).any():
        raise AssertionError("Duplicated uncertainty comparison")
    print(f"Verified {len(details)} cases, {checks} scores, {len(frame)} paired comparisons.")


if __name__ == "__main__":
    main()
