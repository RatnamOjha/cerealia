"""Repeat protocol v1 in a temporary directory and compare stable artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
STABLE_OUTPUTS = [
    "classifier_runs.csv", "classifier_summary.csv", "predictions.csv",
    "legacy_holdout_predictions.csv", "legacy_predictions.csv",
    "classification_report.csv", "confusion_matrices_long.csv",
    "confusion_matrix_0pct.csv", "confusion_matrix_0pct_normalized.csv",
    "confusion_matrix_20pct.csv", "confusion_matrix_20pct_normalized.csv",
    "confusion_matrix_50pct.csv", "confusion_matrix_50pct_normalized.csv",
    "confusion_summary.csv", "paired_tests.csv", "ranking_ablation_by_region.csv",
    "ranking_ablation_candidates.csv", "ranking_ablation_summary.csv",
    "weight_sensitivity.csv", "ranking_stability.csv", "integrity_checks.csv",
    "regional_case_studies.json", "reproduction_metrics.json",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    committed_manifest = json.loads((RESULTS / "manifest.json").read_text())
    expected = committed_manifest["artifact_files_sha256"]
    with tempfile.TemporaryDirectory(prefix="cerealia-repro-") as temp:
        env = os.environ.copy()
        env["CEREALIA_RESEARCH_RESULTS_DIR"] = temp
        subprocess.run(
            [sys.executable, "research/evaluate.py", "--seeds", "42", "43", "44", "45", "46"],
            cwd=ROOT,
            env=env,
            check=True,
        )
        mismatches = []
        for relative in STABLE_OUTPUTS:
            expected_sha = expected.get(relative)
            actual_sha = digest(Path(temp) / relative)
            if expected_sha != actual_sha:
                mismatches.append({"file": relative, "expected": expected_sha, "actual": actual_sha})
        if mismatches:
            raise SystemExit("Reproducibility mismatch:\n" + json.dumps(mismatches, indent=2))
    print(f"Reproducibility verified: {len(STABLE_OUTPUTS)} stable artifacts match.")


if __name__ == "__main__":
    main()
