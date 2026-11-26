"""Recompute research artifact integrity from committed raw observations."""
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import unittest
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RESULTS = ROOT / "results"
SEEDS = [42, 43, 44, 45, 46]
NOISES = [0, 10, 20, 30, 50]
BASELINE = "Extra Trees (no augmentation)"
AUGMENTED = "Extra Trees (augmentation)"
MODEL_ORDER = [
    "Random Forest (no augmentation)", "Random Forest (augmentation)",
    BASELINE, AUGMENTED, "HistGradientBoosting (augmentation)",
    "Gaussian Naive Bayes",
]


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def holm_adjust(raw):
    order = sorted(range(len(raw)), key=raw.__getitem__)
    adjusted = [1.0] * len(raw)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (len(raw) - position) * raw[index]))
        adjusted[index] = running
    return adjusted


class ResearchIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((RESULTS / "manifest.json").read_text())
        cls.preds = pd.read_csv(RESULTS / "predictions.csv")
        cls.runs = pd.read_csv(RESULTS / "classifier_runs.csv")
        cls.ranking = pd.read_csv(RESULTS / "ranking_ablation_by_region.csv")
        cls.candidates = pd.read_csv(RESULTS / "ranking_ablation_candidates.csv")

    def test_manifest_and_all_artifact_checksums(self):
        expected_manifest = (RESULTS / "manifest.sha256").read_text().split()[0]
        self.assertEqual(file_sha(RESULTS / "manifest.json"), expected_manifest)
        self.assertEqual(self.manifest["seeds"], SEEDS)
        self.assertEqual(self.manifest["noise_levels_pct"], NOISES)
        self.assertEqual(self.manifest["checks"]["result_combinations"], 150)
        self.assertEqual(self.manifest["checks"]["prediction_rows"], 66_000)
        self.assertEqual(self.manifest["dependencies"]["python"], platform.python_version())
        for package, distribution in [("numpy", "numpy"), ("pandas", "pandas"),
                                      ("scikit_learn", "scikit-learn"), ("scipy", "scipy"),
                                      ("matplotlib", "matplotlib"), ("joblib", "joblib")]:
            self.assertEqual(self.manifest["dependencies"][package], version(distribution))
        for relative, expected in self.manifest["artifact_files_sha256"].items():
            path = RESULTS / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(file_sha(path), expected, relative)
        for relative, expected in self.manifest["dataset_sha256"].items():
            path = REPO / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(file_sha(path), expected, relative)

    def test_protocol_has_exactly_five_matched_splits(self):
        self.assertEqual(len(self.runs), 6 * 5 * 5)
        self.assertEqual(set(self.runs.seed), set(SEEDS))
        self.assertEqual(set(self.runs.noise_pct), set(NOISES))
        self.assertEqual(set(self.runs.model), set(MODEL_ORDER))
        self.assertFalse(self.runs.duplicated(["seed", "model", "noise_pct"]).any())
        self.assertEqual(len(self.preds), 66_000)
        for (seed, noise), group in self.preds.groupby(["seed", "noise_pct"]):
            sample_sets = [set(g.sample_id) for _, g in group.groupby("model")]
            self.assertEqual(len(sample_sets), 6)
            self.assertTrue(all(ids == sample_sets[0] for ids in sample_sets[1:]))
        for row in self.runs.itertuples():
            k = round(row.accuracy * 440)
            lower, upper = wilson(k, 440)
            self.assertAlmostEqual(row.accuracy_wilson_low, lower, places=10)
            self.assertAlmostEqual(row.accuracy_wilson_high, upper, places=10)

        rejected = subprocess.run(
            [sys.executable, str(ROOT / "evaluate.py"), "--seeds", "42"],
            cwd=REPO, capture_output=True, text=True,
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("requires exactly", rejected.stderr)

    def test_confusion_matrices_and_classification_metrics_recompute(self):
        for noise in [0, 20, 50]:
            sample = self.preds[(self.preds.seed == 42) &
                                (self.preds.model == AUGMENTED) &
                                (self.preds.noise_pct == noise)]
            labels = sorted(sample.y_true.unique())
            actual = confusion_matrix(sample.y_true, sample.y_pred, labels=labels)
            saved = pd.read_csv(RESULTS / f"confusion_matrix_{noise}pct.csv", index_col=0)
            self.assertEqual(saved.index.tolist(), labels)
            np.testing.assert_array_equal(actual, saved.to_numpy())
            self.assertEqual(int(actual.sum()), 440)
            self.assertTrue((actual.sum(axis=1) == 20).all())
            normalized = pd.read_csv(RESULTS / f"confusion_matrix_{noise}pct_normalized.csv", index_col=0)
            np.testing.assert_allclose(normalized.to_numpy().sum(axis=1), np.ones(22), atol=1e-12)

            row = pd.read_csv(RESULTS / "confusion_summary.csv").set_index("noise_pct").loc[noise]
            y, pred = sample.y_true, sample.y_pred
            self.assertEqual(row.samples, 440)
            self.assertEqual(row.correct, int(np.trace(actual)))
            self.assertEqual(row.incorrect, 440 - int(np.trace(actual)))
            self.assertAlmostEqual(row.accuracy, accuracy_score(y, pred))
            self.assertAlmostEqual(row.macro_precision, precision_score(y, pred, labels=labels, average="macro", zero_division=0))
            self.assertAlmostEqual(row.macro_recall, recall_score(y, pred, labels=labels, average="macro", zero_division=0))
            self.assertAlmostEqual(row.macro_f1, f1_score(y, pred, labels=labels, average="macro", zero_division=0))
            self.assertAlmostEqual(row.weighted_f1, f1_score(y, pred, labels=labels, average="weighted", zero_division=0))

    def test_paired_statistics_and_holm_adjustment_recompute(self):
        paired = pd.read_csv(RESULTS / "paired_tests.csv").sort_values("noise_pct")
        raw_p = []
        for row in paired.itertuples():
            sub = self.preds[(self.preds.seed == 42) & (self.preds.noise_pct == row.noise_pct)]
            wide = sub.pivot(index="sample_id", columns="model", values=["y_true", "y_pred"])
            truth = wide[("y_true", BASELINE)].to_numpy()
            baseline_ok = wide[("y_pred", BASELINE)].to_numpy() == truth
            augmented_ok = wide[("y_pred", AUGMENTED)].to_numpy() == truth
            counts = (int((baseline_ok & augmented_ok).sum()),
                      int((baseline_ok & ~augmented_ok).sum()),
                      int((~baseline_ok & augmented_ok).sum()),
                      int((~baseline_ok & ~augmented_ok).sum()))
            self.assertEqual(counts, (row.both_correct, row.baseline_only_correct,
                                      row.augmented_only_correct, row.both_incorrect))
            p = binomtest(counts[2], counts[1] + counts[2], .5).pvalue if counts[1] + counts[2] else 1.0
            raw_p.append(p)
            self.assertAlmostEqual(row.raw_p, p, places=14)
            self.assertAlmostEqual(row.baseline_accuracy, baseline_ok.mean())
            self.assertAlmostEqual(row.augmented_accuracy, augmented_ok.mean())
        for row, adjusted in zip(paired.itertuples(), holm_adjust(raw_p)):
            self.assertAlmostEqual(row.holm_adjusted_p, adjusted, places=14)

    def test_ranking_candidate_sets_and_fallback_are_consistent(self):
        primary_names = ["Fitness only", "Return term only", "Fitness + return",
                         "Fitness + return + water", "Full Cerealia"]
        primary = self.ranking[self.ranking.variant == "primary"]
        candidates = self.candidates[self.candidates.configuration.isin(primary_names)]
        self.assertEqual(primary.state_id.nunique(), 36)
        self.assertEqual(len(primary), 36 * len(primary_names))
        for state_id, state_rows in primary.groupby("state_id"):
            universes = []
            for config in primary_names:
                summary = state_rows[state_rows.configuration == config].iloc[0]
                group = candidates[(candidates.state_id == state_id) &
                                   (candidates.configuration == config)].sort_values("rank")
                self.assertEqual(group["rank"].tolist(), list(range(1, len(group) + 1)))
                self.assertEqual(len(group), int(summary.eligible_candidates))
                self.assertEqual(group.iloc[0].candidate_crop, summary.top1)
                self.assertEqual("|".join(group.candidate_crop.iloc[:3]), summary.top3)
                self.assertEqual("|".join(group.candidate_crop.iloc[:6]), summary.top6)
                self.assertTrue((group.fitness_gate_fallback == summary.fitness_gate_fallback).all())
                universes.append(set(group.candidate_crop))
            self.assertTrue(all(universe == universes[0] for universe in universes[1:]))
            fallback_values = state_rows.fitness_gate_fallback.unique()
            self.assertEqual(len(fallback_values), 1)
        ladakh = primary[(primary.state_name == "Ladakh")]
        self.assertTrue(ladakh.fitness_gate_fallback.all())


if __name__ == "__main__":
    unittest.main()
