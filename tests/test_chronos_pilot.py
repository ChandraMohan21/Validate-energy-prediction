"""Verify portable inputs, origin-frozen contexts and model-output comparisons."""
import contextlib
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chronos_pilot import CONTEXT, comparison, contexts, forecast, load_cases


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, item):
        return FakeTensor(self.values[item])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class ChronosPilotTests(unittest.TestCase):
    def test_context_ends_at_origin_and_ignores_future(self):
        values = np.arange(1000, dtype=float)
        before = contexts(values, [500])
        self.assertEqual(before.shape, (1, CONTEXT))
        self.assertEqual(before[0, 0], 165)
        self.assertEqual(before[0, -1], 500)
        values[501:] = -123456
        np.testing.assert_array_equal(before, contexts(values, [500]))

    def test_missing_context_rejected(self):
        values = np.ones(1000)
        values[499] = np.nan
        with self.assertRaises(ValueError):
            contexts(values, [500])

    def test_batched_forecast_keeps_origin_order_and_horizon(self):
        class Pipeline:
            def predict_quantiles(self, batch, prediction_length, quantile_levels, limit_prediction_length):
                assert quantile_levels == [0.5]
                assert limit_prediction_length is False
                result = np.repeat(batch[:, -1, None, None], prediction_length, axis=1)
                return FakeTensor(result), None
        torch = types.SimpleNamespace(from_numpy=lambda x: x, inference_mode=contextlib.nullcontext)
        with patch.dict(sys.modules, {"torch": torch}):
            predictions = forecast(Pipeline(), np.arange(1000), np.array([500, 502, 504]), 168, 2)
        np.testing.assert_array_equal(predictions[:, 0], [500, 502, 504])
        self.assertEqual(predictions.shape, (3, 168))

    def test_all_exported_cases_exactly_match_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(ROOT / "reports/chronos_colab/chronos_pilot_inputs.zip") as archive:
                archive.extractall(root)
            cases = load_cases(root)
            check = subprocess.run([sys.executable, str(root / "src/chronos_pilot.py"),
                                    "--root", str(root), "--check-only"],
                                   capture_output=True, text=True, check=True)
            self.assertIn("Verified 16", check.stdout)
            self.assertEqual(len(cases), 16)
            self.assertEqual(len({r["dataset"] for r, _, _ in cases}), 4)
            for record, series, saved in cases:
                # Identity forecast is only a scoring check, never a model result.
                rows = comparison(record, saved, saved["actual"].copy())
                self.assertEqual(len(rows), 5)
                self.assertTrue(all(row["chronos_MAE"] == 0 for row in rows))
                selected = rows[-1]
                expected = record["test"][record["selected_on_validation"]]["MAE"]
                self.assertAlmostEqual(selected["baseline_MAE"], expected, places=10)
            path = root / "src/baseline_pilot.py"
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_cases(root)

    def test_wrong_prediction_shape_rejected(self):
        with self.assertRaises(ValueError):
            comparison({}, {"actual": np.ones((2, 24))}, np.ones((2, 1)))

    def test_notebook_valid_and_code_parses(self):
        import ast
        import nbformat
        notebook = nbformat.read(ROOT / "notebooks/03_chronos_pilot_colab.ipynb", as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                source = "\n".join(line for line in cell.source.splitlines() if not line.startswith("%"))
                ast.parse(source)
        self.assertTrue(all(not cell.get("outputs") for cell in notebook.cells))


if __name__ == "__main__":
    unittest.main()
