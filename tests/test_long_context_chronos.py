"""Check the 2048-hour rerun: history rule, unchanged cases, notebook and claim counting."""
import ast
import hashlib
import json
import sys
import unittest
import zipfile
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chronos_long_context import long_contexts
from long_context_study import BUNDLE, INPUTS, NOTEBOOK, PROTOCOL, summarize


class LongContextTests(unittest.TestCase):
    def test_history_is_capped_shortened_at_start_and_keeps_gaps(self):
        values = np.arange(3000, dtype=float)
        values[30] = np.nan
        short, full = long_contexts(values, [400, 2999])
        self.assertEqual(len(short), 401)
        self.assertEqual(len(full), 2048)
        self.assertEqual(full[-1], 2999)
        self.assertTrue(np.isnan(short[30]))
        values[2990] = np.nan
        with self.assertRaises(ValueError):
            long_contexts(values, [2999])
        with self.assertRaises(ValueError):
            long_contexts(values, [334])

    def test_bundle_keeps_every_frozen_input_and_adds_only_the_runner(self):
        with zipfile.ZipFile(INPUTS) as old, zipfile.ZipFile(BUNDLE) as new:
            self.assertEqual(set(new.namelist()) - set(old.namelist()), {"chronos_long_context.py"})
            for name in old.namelist():
                self.assertEqual(old.read(name), new.read(name), name)
            self.assertEqual(new.read("chronos_long_context.py"),
                             (ROOT / "src/chronos_long_context.py").read_bytes())

    def test_protocol_saved_before_run_and_notebook_matches_bundle(self):
        saved = json.loads((ROOT / "reports/long_context_chronos/protocol.json").read_text())
        self.assertEqual(saved, PROTOCOL)
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
        self.assertTrue(any(digest in cell.source for cell in notebook.cells))
        code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        ast.parse(code)
        self.assertIn('[PYTHON, "-m", "pip", "check"]', code)
        self.assertIn("chronos-forecasting==2.2.2", code)

    def test_claim_counting(self):
        clear, unclear = "chronos_supported_both_lengths", "unclear_or_block_sensitive"
        table = {("Q2", "a", "x", 168): {"weekly": (clear, 20.0), "selected": (unclear, 3.0)},
                 ("Q3", "a", "x", 168): {"weekly": (clear, 8.0), "selected": (clear, 4.0)},
                 ("Q3", "a", "x", 24): {"weekly": (unclear, 1.0), "selected": (clear, 6.0)}}
        result = summarize(table, 168)
        self.assertEqual((result["cases"], result["clear_vs_weekly"], result["clear_vs_selected"]), (2, 2, 1))
        self.assertEqual((result["meaningful5_vs_weekly"], result["meaningful5_vs_selected"]), (2, 0))
        self.assertEqual(result["weakened"], [("Q2", "a", "x", 168)])
        self.assertEqual(summarize(table, 24)["strengthened"], [("Q3", "a", "x", 24)])


if __name__ == "__main__":
    unittest.main()
