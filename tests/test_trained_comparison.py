import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline_pilot import reference_predictions
from trained_comparison import eligible_training, features, run_case


class TrainedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series(100 + 10*np.sin(np.arange(4000)*2*np.pi/24),
            index=pd.date_range("2017-01-01", periods=4000, freq="h"))

    def test_future_values_cannot_change_features(self):
        before, weekly, _ = features(self.series, [700], 168)
        changed = self.series.copy()
        changed.iloc[701:] = -100000
        after, weekly_after, _ = features(changed, [700], 168)
        pd.testing.assert_frame_equal(before, after)
        np.testing.assert_array_equal(weekly, weekly_after)

    def test_lag_features_match_frozen_baselines(self):
        x, weekly, _ = features(self.series, [700], 168)
        pred, _, _ = reference_predictions(self.series, [700], 168)
        np.testing.assert_array_equal(x.daily, pred["daily"].ravel())
        np.testing.assert_array_equal(weekly, pred["weekly"].ravel())
        self.assertEqual(x.origin_lag_335.iloc[0], self.series.iloc[365])

    def test_training_labels_mature_before_cutoff_and_gaps_excluded(self):
        train = eligible_training(self.series, self.series.index[1600])
        self.assertTrue((train+168 <= 1600).all())
        self.assertTrue((train % 24 == 0).all())
        changed = self.series.copy()
        changed.iloc[900] = np.nan
        clean = eligible_training(changed, changed.index[1600])
        self.assertTrue(all(not o-335 <= 900 <= o+168 for o in clean))
        self.assertLess(len(clean), len(train))

    def test_first_quarter_cannot_fit_before_frozen_validation(self):
        self.assertEqual(len(eligible_training(self.series, "2017-01-06 23:00")), 0)

    def test_real_estimators_produce_aligned_finite_predictions(self):
        validation = np.arange(1600, 1605)
        test = np.arange(2000, 2003)
        _, vy, _ = reference_predictions(self.series, validation, 24)
        _, ty, _ = reference_predictions(self.series, test, 24)
        train = eligible_training(self.series, self.series.index[1500])
        refit = eligible_training(self.series, self.series.index[1900])
        predictions, choices, selected = run_case(self.series, {"horizon_hours": 24},
            {"validation_origins": validation, "test_origins": test,
             "validation_actual": vy, "actual": ty}, train, refit,
            {"ridge": [1., 100.], "lightgbm": [15, 31]})
        self.assertIn(selected, ("ridge", "lightgbm"))
        for array in predictions.values():
            self.assertEqual(array.shape, ty.shape)
            self.assertTrue(np.isfinite(array).all())
        for detail in choices.values():
            self.assertEqual(detail["validation_MAE"], min(t["validation_MAE"] for t in detail["trials"]))


if __name__ == "__main__":
    unittest.main()
