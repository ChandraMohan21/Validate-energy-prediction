import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trained_comparison import eligible_training
from trained_comparison_v2 import coverage


class HourlyTrainingTests(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series(np.ones(4000), index=pd.date_range("2017-01-01", periods=4000, freq="h"))

    def test_hourly_selection_covers_all_hours_and_contains_v1(self):
        old = eligible_training(self.series, self.series.index[1600], stride=24)
        new = eligible_training(self.series, self.series.index[1600], stride=1)
        self.assertTrue(set(old).issubset(new))
        self.assertEqual(set(coverage(self.series, new)), {str(i) for i in range(24)})
        self.assertEqual(set(coverage(self.series, old)), {"0"})
        self.assertTrue((new+168 <= 1600).all())

    def test_missing_hour_does_not_get_filled_by_hourly_selection(self):
        self.series.iloc[900] = np.nan
        new = eligible_training(self.series, self.series.index[1800], stride=1)
        self.assertTrue(all(not o-335 <= 900 <= o+168 for o in new))


if __name__ == "__main__":
    unittest.main()
