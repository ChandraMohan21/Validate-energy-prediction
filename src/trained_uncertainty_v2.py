"""Apply the unchanged trained-model bootstrap method to version 2 predictions."""
import json
from pathlib import Path

from baseline_pilot import file_hash
import trained_uncertainty as analysis

ROOT = Path(__file__).resolve().parents[1]


def main():
    analysis.TRAINED = ROOT / "reports/trained_comparison_v2"
    analysis.OUT = ROOT / "reports/trained_uncertainty_v2"
    # The base analysis records the actual v2 input hashes and validates its scores.
    analysis.main()
    report = analysis.OUT / "README.md"
    text = report.read_text(encoding="utf-8")
    text = text.replace("# Uncertainty: Chronos versus trained models", "# Version 2 uncertainty: Chronos versus hourly-trained models", 1)
    text = text.replace("python src/trained_uncertainty.py", "python src/trained_uncertainty_v2.py")
    text += "\nThis analysis uses v2 hourly-training predictions. V1 models and uncertainty outputs remain preserved. The test periods were inspected during v1, so this remains a diagnostic follow-up.\n"
    report.write_text(text, encoding="utf-8")
    (analysis.OUT / "wrapper_provenance.json").write_text(json.dumps({
        "wrapper_sha256": file_hash(Path(__file__)),
        "analysis_module_sha256": file_hash(Path(analysis.__file__)),
        "trained_input_directory": "reports/trained_comparison_v2",
        "training_coverage": "All eligible hourly origins, all 24 origin hours represented",
        "method_changes_from_v1": "None; same 5000 resamples and 168/336-hour blocks"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
