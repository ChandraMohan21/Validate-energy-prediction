"""Chronos-2 preparation checks; no torch imports or model downloads."""
import ast
import hashlib
import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chronos2_frozen import MODEL_ID, PACKAGES, REVISION, medians
from chronos2_study import BOLT, BUNDLE, DEST, INPUTS, NOTEBOOK, checked_forecasts
from chronos_long_context import MODEL_ID as BOLT_ID, REVISION as BOLT_REVISION, long_contexts


class Tensor:
    """Minimal tensor stand-in for checking the list/variate/time/quantile adapter."""
    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape = self.data.shape

    def __getitem__(self, key):
        return Tensor(self.data[key])

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class Pipeline:
    def predict_quantiles(self, histories, **kwargs):
        self.histories, self.kwargs = histories, kwargs
        h = kwargs["prediction_length"]
        return [Tensor(np.full((1, h, 1), a[-1])) for a in histories], None


class Chronos2Tests(unittest.TestCase):
    def test_adapter_is_univariate_isolated_and_median(self):
        pipeline = Pipeline()
        histories = long_contexts(np.arange(3000), [1200, 2500])
        result = medians(pipeline, histories, 168, 8)
        self.assertEqual(result.shape, (2, 168))
        np.testing.assert_array_equal(result[:, 0], [1200, 2500])
        self.assertTrue(all(a.ndim == 1 and len(a) <= 2048 for a in pipeline.histories))
        self.assertEqual(pipeline.kwargs, dict(prediction_length=168, quantile_levels=[0.5],
            context_length=2048, batch_size=8, cross_learning=False, limit_prediction_length=True))
        # Altering observations after the origin must not alter that origin's input.
        values = np.arange(3000, dtype=float)
        before = long_contexts(values, [1200])[0]
        values[1201:] = -100000
        np.testing.assert_array_equal(before, long_contexts(values, [1200])[0])

    def test_adapter_rejects_bolt_shape_multivariate_and_nonfinite(self):
        class Bad:
            def __init__(self, output):
                self.output = output
            def predict_quantiles(self, *args, **kwargs):
                return self.output, None
        for output in (Tensor(np.zeros((1, 24, 1))), [],
                       [Tensor(np.zeros((2, 24, 1)))], [Tensor(np.full((1, 24, 1), np.nan))]):
            with self.assertRaises(ValueError):
                medians(Bad(output), [np.ones(400)], 24, 1)
        with self.assertRaises(ValueError):
            medians(Pipeline(), [np.ones(400)], 25, 1)

    def test_bundle_preserves_all_frozen_bytes_and_pins(self):
        with zipfile.ZipFile(INPUTS) as old, zipfile.ZipFile(BUNDLE) as new:
            self.assertEqual(set(new.namelist()) - set(old.namelist()), {
                "chronos2_frozen.py", "chronos_long_context.py", "chronos2_protocol.json", "requirements-colab.txt"})
            for name in old.namelist():
                self.assertEqual(old.read(name), new.read(name), name)
            for name in ("chronos2_frozen.py", "chronos_long_context.py"):
                self.assertEqual(new.read(name), (ROOT / "src" / name).read_bytes())
            protocol = json.loads(new.read("chronos2_protocol.json"))
            self.assertEqual(protocol["model_revision"], REVISION)
            self.assertEqual(protocol["packages"], PACKAGES)
            self.assertEqual(protocol["bolt_results_sha256"], hashlib.sha256(BOLT.read_bytes()).hexdigest())

    def test_notebook_parses_validates_and_references_exact_bundle(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
        ast.parse(code)
        self.assertIn(hashlib.sha256(BUNDLE.read_bytes()).hexdigest(), code)
        self.assertIn("--smoke-only", code)
        self.assertLess(code.index("--smoke-only"), code.index("BATCH_SIZE = 8"))
        self.assertIn("virtualenv", code)
        self.assertIn('"pip", "check"', code)
        for p, version in PACKAGES.items():
            self.assertIn(f"{p}=={version}", code)
        self.assertTrue(all(c.execution_count is None and not c.outputs
                            for c in notebook.cells if c.cell_type == "code"))

    def test_real_bolt_alignment_and_wrong_model_rejection(self):
        protocol = json.loads((DEST / "protocol.json").read_text())
        with zipfile.ZipFile(BOLT) as archive, zipfile.ZipFile(INPUTS) as inputs:
            result = checked_forecasts(archive, inputs, protocol, BOLT_ID, BOLT_REVISION)
            self.assertEqual(len(result), 60)
            with self.assertRaises(ValueError):
                checked_forecasts(archive, inputs, protocol, MODEL_ID, REVISION)

    def test_returned_chronos2_identity_and_frozen_actuals_are_checked(self):
        # Use old forecasts strictly as a validator fixture, never as Chronos-2 results.
        protocol = json.loads((DEST / "protocol.json").read_text())
        with zipfile.ZipFile(BOLT) as archive:
            payloads = {name: archive.read(name) for name in archive.namelist()}
        for window in ("Q1", "Q2", "Q3", "Q4"):
            name = window + "/run_metadata.json"
            metadata = json.loads(payloads[name])
            identity = metadata["identity"]
            identity.update(model_id=MODEL_ID, model_revision=REVISION, cross_learning=False,
                covariates=False, packages=PACKAGES,
                runner_sha256=protocol["runner_sha256"], history_helper_sha256=protocol["history_helper_sha256"])
            payloads[name] = json.dumps(metadata).encode()
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            for case in [n for n in payloads if n.startswith(window + "/") and n.endswith(".npz")]:
                with np.load(io.BytesIO(payloads[case]), allow_pickle=False) as saved:
                    arrays = {k: saved[k] for k in saved.files}
                arrays["run_fingerprint"] = fingerprint
                stream = io.BytesIO()
                np.savez_compressed(stream, **arrays)
                payloads[case] = stream.getvalue()

        class Archive:
            def read(self, name):
                return payloads[name]
            def namelist(self):
                return list(payloads)
        with zipfile.ZipFile(INPUTS) as inputs:
            self.assertEqual(len(checked_forecasts(Archive(), inputs, protocol, MODEL_ID, REVISION)), 60)
            case = next(n for n in payloads if n.startswith("Q1/") and n.endswith(".npz"))
            original = payloads[case]
            with np.load(io.BytesIO(original), allow_pickle=False) as saved:
                arrays = {k: saved[k] for k in saved.files}
            arrays["actual"] = arrays["actual"].copy()
            arrays["actual"][0, 0] += 1
            stream = io.BytesIO()
            np.savez_compressed(stream, **arrays)
            payloads[case] = stream.getvalue()
            with self.assertRaises(AssertionError):
                checked_forecasts(Archive(), inputs, protocol, MODEL_ID, REVISION)
            payloads[case] = original
            metadata = json.loads(payloads["Q1/run_metadata.json"])
            metadata["identity"]["cross_learning"] = True
            payloads["Q1/run_metadata.json"] = json.dumps(metadata).encode()
            with self.assertRaisesRegex(ValueError, "isolated origins"):
                checked_forecasts(Archive(), inputs, protocol, MODEL_ID, REVISION)


if __name__ == "__main__":
    unittest.main()
