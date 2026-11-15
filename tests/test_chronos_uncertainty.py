import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chronos_uncertainty import calendar_losses, circular_indices, paired_interval


class UncertaintyTests(unittest.TestCase):
    def test_calendar_gaps_and_original_mae_preserved(self):
        actual = np.zeros((3, 2))
        grid = calendar_losses([5, 6, 10], actual, np.array([[1, 3], [2, 4], [3, 5]]), np.full((3, 2), 6))
        self.assertEqual(len(grid), 6)
        self.assertTrue(np.isnan(grid[2:5]).all())
        np.testing.assert_array_equal(np.nanmean(grid, axis=0), [3, 6])

    def test_common_indices_keep_constant_pair_difference(self):
        x = np.arange(840, dtype=float)
        result = paired_interval(np.column_stack([x, x + 5]), 168, 300, 42)
        np.testing.assert_allclose(result["difference_ci95"], [5, 5], atol=1e-10)
        self.assertEqual(result["interval_direction"], "chronos")

    def test_zero_baseline_ratio_not_fabricated(self):
        result = paired_interval(np.column_stack([np.ones(840), np.zeros(840)]), 168, 300, 42)
        self.assertEqual(result["difference_ci95"], [-1, -1])
        self.assertIsNone(result["gain_pct_ci95"])
        self.assertEqual(result["zero_baseline_replicates"], 300)
        self.assertEqual(result["interval_direction"], "baseline")

    def test_indices_are_consecutive_within_blocks_and_reproducible(self):
        indices = circular_indices(840, 168, 20, 42)
        np.testing.assert_array_equal(indices, circular_indices(840, 168, 20, 42))
        blocks = indices.reshape(20, 5, 168)
        self.assertTrue((np.diff(blocks, axis=2) % 840 == 1).all())

    def test_long_blocks_widen_interval_for_persistent_errors(self):
        differences = np.repeat([-1., 1., -1., 1., -1.], 168)
        grid = np.column_stack([np.full(840, 2.), 2 + differences])
        iid = paired_interval(grid, 1, 1000, 42)["difference_ci95"]
        block = paired_interval(grid, 168, 1000, 42)["difference_ci95"]
        self.assertGreater(block[1] - block[0], 3 * (iid[1] - iid[0]))

    def test_sparse_origins_flagged(self):
        grid = np.full((840, 2), np.nan)
        grid[::4] = [1., 2.]
        result = paired_interval(grid, 168, 300, 42)
        self.assertFalse(result["at_least_two_blocks_of_valid_origins"])

    def test_calendar_span_shorter_than_block_not_resampled(self):
        result = paired_interval(np.ones((239, 2)), 336, 300, 42)
        self.assertEqual(result["difference_ci95"], [None, None])
        self.assertEqual(result["valid_replicates"], 0)


if __name__ == "__main__":
    unittest.main()
