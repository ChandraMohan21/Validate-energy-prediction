import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trained_uncertainty import assessment, case_results


class TrainedUncertaintyTests(unittest.TestCase):
    def test_support_requires_both_lengths_and_enough_data(self):
        supported = {"interval_direction": "chronos", "at_least_two_blocks_of_valid_origins": True}
        self.assertEqual(assessment([supported, supported]), "chronos_supported_both_lengths")
        self.assertEqual(assessment([supported, {**supported, "interval_direction": "unclear"}]), "unclear_or_block_sensitive")
        self.assertEqual(assessment([supported, {**supported, "at_least_two_blocks_of_valid_origins": False}]), "limited_data")

    def test_selected_family_identical_uncertainty_and_known_errors(self):
        actual = np.zeros((840, 2))
        saved = {"actual": actual, "test_origins": np.arange(500, 1340),
            "chronos": np.ones_like(actual), "ridge": np.full_like(actual, 3.), "lightgbm": np.full_like(actual, 2.)}
        details = {"window": "Q2", "dataset": "synthetic", "series": "one", "horizon_hours": 2, "selected_trained": "lightgbm"}
        rows = case_results(details, saved, repetitions=200)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["intervals"], rows[2]["intervals"])
        self.assertEqual(rows[0]["difference"], 2.)
        self.assertEqual(rows[1]["gain_pct"], 50.)
        self.assertTrue(all(row["assessment"] == "chronos_supported_both_lengths" for row in rows))


if __name__ == "__main__":
    unittest.main()
