"""Regression tests for origin availability, recursion and selection boundaries."""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline_pilot import PROTOCOL, choose_on_validation, origin_indices, reference_predictions, score, tsf_records


class BaselineProtocolTests(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series(np.arange(2000, dtype=float) + 1,
                                index=pd.date_range("2026-01-01", periods=2000, freq="h"))
        self.origin = 500

    def test_daily_repeats_last_observed_day(self):
        p, _, _ = reference_predictions(self.series, [self.origin], 168)
        expected = self.series.iloc[self.origin - 23:self.origin + 1].to_numpy()
        np.testing.assert_array_equal(p["daily"][0], np.tile(expected, 7))

    def test_weekly_uses_same_target_hour_last_week(self):
        p, _, _ = reference_predictions(self.series, [self.origin], 168)
        expected = self.series.iloc[self.origin - 167:self.origin + 1].to_numpy()
        np.testing.assert_array_equal(p["weekly"][0], expected)

    def test_future_values_cannot_change_any_prediction(self):
        before, _, _ = reference_predictions(self.series, [self.origin], 168)
        changed = self.series.copy()
        changed.iloc[self.origin + 1:] = -100000
        after, _, _ = reference_predictions(changed, [self.origin], 168)
        for method in before:
            np.testing.assert_array_equal(before[method], after[method])

    def test_hybrid_matches_independent_recursive_implementation(self):
        p, _, _ = reference_predictions(self.series, [self.origin], 168)
        history = self.series.iloc[:self.origin + 1].to_list()
        expected = []
        for h in range(1, 169):
            t = self.origin + h
            lag = 168 if self.series.index[t].dayofweek in [0, 5, 6] else 24
            value = history[t - lag]
            history.append(value)
            expected.append(value)
        np.testing.assert_array_equal(p["hybrid"][0], expected)

    def test_validation_alone_selects_blend(self):
        protocol = json.loads(PROTOCOL.read_text())
        pred = {"daily": np.array([[0., 0.]]), "weekly": np.array([[10., 10.]]),
                "hybrid": np.array([[0., 10.]])}
        alpha, selected, _ = choose_on_validation(np.array([[3., 3.]]), pred, protocol)
        self.assertAlmostEqual(alpha, .7)
        self.assertEqual(selected, "blend")

    def test_zero_target_does_not_get_epsilon_mape(self):
        result = score(np.array([0., 2.]), np.array([1., 2.]))
        self.assertIsNone(result["MAPE_pct"])
        self.assertEqual(result["zero_target_rows"], 1)

    def test_embargo_and_missing_windows(self):
        protocol = json.loads(PROTOCOL.read_text())
        series = pd.Series(np.ones(365 * 24), index=pd.date_range("2026-01-01", periods=365 * 24, freq="h"))
        val, test, info = origin_indices(series, protocol)
        self.assertEqual(len(val), 840)
        self.assertEqual(len(test), 840)
        self.assertGreaterEqual(test[0] - val[-1], 169)
        series.iloc[test[0] + 10] = np.nan
        _, filtered, _ = origin_indices(series, protocol)
        self.assertNotIn(test[0], filtered)

    def test_archive_one_second_offset_aligns_without_losing_values(self):
        content = ("@attribute series_name string\n@attribute start_timestamp date\n"
                   "@frequency hourly\n@data\nT1:2014-01-01 00-00-01:1,2,3\n")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("sample.tsf", content)
            _, series, _ = next(tsf_records(path, 1))
        self.assertEqual(series.index[0], pd.Timestamp("2014-01-01 00:00"))
        np.testing.assert_array_equal(series.to_numpy(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
