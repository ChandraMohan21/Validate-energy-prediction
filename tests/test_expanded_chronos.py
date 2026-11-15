"""Check follow-up windows, earlier-only selection and preservation of the pilot."""
import ast
import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline_pilot import choose_on_validation, reference_predictions
from chronos_pilot import load_cases


class ExpandedChronosTests(unittest.TestCase):
    def test_all_windows_and_earlier_only_selection(self):
        total = 0
        previous_end = {}
        for label in ("Q1", "Q2", "Q3", "Q4"):
            root = ROOT / "reports/expanded_chronos/inputs" / label
            protocol = json.loads((root / "reports/baseline_pilot/protocol.json").read_text())
            cases = load_cases(root)
            self.assertEqual(len(cases), 14 if label in ("Q1", "Q2") else 16)
            total += len(cases)
            for record, series, saved in cases:
                self.assertLess(pd.Timestamp(record["validation_target_last"]), pd.Timestamp(record["test_origin_first"]))
                self.assertLessEqual(pd.Timestamp(record["test_target_last"]), series.index.max())
                key = (record["dataset"], record["series"], record["horizon_hours"])
                if key in previous_end:
                    self.assertGreater(pd.Timestamp(record["test_origin_first"]), previous_end[key])
                previous_end[key] = pd.Timestamp(record["test_target_last"])
                predictions, actual, _ = reference_predictions(series, saved["validation_origins"], record["horizon_hours"])
                alpha, selected, _ = choose_on_validation(actual, predictions, protocol)
                self.assertEqual(alpha, record["blend_daily_weight"])
                self.assertEqual(selected, record["selected_on_validation"])
        self.assertEqual(total, 60)

    def test_q4_repeats_original_baselines(self):
        root = ROOT / "reports/expanded_chronos/inputs/Q4"
        for record, _, saved in load_cases(root):
            with np.load(ROOT / record["prediction_file"], allow_pickle=False) as original:
                for name in original.files:
                    np.testing.assert_array_equal(saved[name], original[name])

    def test_exclusions_are_reported_and_outputs_not_claimed(self):
        destination = ROOT / "reports/expanded_chronos"
        excluded = json.loads((destination / "exclusions.json").read_text())
        self.assertEqual({(r["dataset"], r["window"]) for r in excluded}, {("household", "Q1"), ("household", "Q2")})
        with zipfile.ZipFile(destination / "expanded_chronos_inputs.zip") as archive:
            self.assertFalse(any("__pycache__" in n or "complete.json" in n for n in archive.namelist()))
            for label in ("Q1", "Q2", "Q3", "Q4"):
                manifest = json.loads(archive.read(label + "/manifest.json"))
                for name, digest in manifest["files"].items():
                    import hashlib
                    self.assertEqual(hashlib.sha256(archive.read(label + "/" + name)).hexdigest(), digest)

    def test_notebook_valid_and_upload_digest_matches(self):
        import hashlib
        notebook = nbformat.read(ROOT / "notebooks/04_expanded_chronos_colab.ipynb", as_version=4)
        nbformat.validate(notebook)
        digest = hashlib.sha256((ROOT / "reports/expanded_chronos/expanded_chronos_inputs.zip").read_bytes()).hexdigest()
        self.assertTrue(any(digest in cell.source for cell in notebook.cells))
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse("\n".join(line for line in cell.source.splitlines() if not line.startswith("%")))

    def test_forecasts_use_isolated_python_and_dependency_check(self):
        notebook = nbformat.read(ROOT / "notebooks/04_expanded_chronos_colab.ipynb", as_version=4)
        code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        self.assertNotIn("%pip", code)
        self.assertIn('"-m", "venv"', code)
        self.assertIn('[PYTHON, "-m", "pip", "check"]', code)
        self.assertIn('"PYTHONNOUSERSITE"', code)
        for cell in notebook.cells:
            if cell.cell_type == "code" and '"src/chronos_pilot.py"' in cell.source:
                self.assertIn('[PYTHON,', cell.source)
                self.assertIn('env=RUN_ENV', cell.source)


if __name__ == "__main__":
    unittest.main()
