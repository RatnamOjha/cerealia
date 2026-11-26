"""Post-run integrity checks for saved research artifacts."""
import json
from pathlib import Path
import unittest

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


class ResearchIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((RESULTS / "manifest.json").read_text())

    def test_split_and_prediction_counts(self):
        runs = pd.read_csv(RESULTS / "classifier_runs.csv")
        preds = pd.read_csv(RESULTS / "predictions.csv")
        self.assertEqual(len(runs), 150)
        self.assertEqual(len(preds), 5 * 6 * 5 * 440)
        self.assertEqual(runs.groupby(["seed", "model", "noise_pct"]).size().min(), 1)
        for (seed, noise), group in preds.groupby(["seed", "noise_pct"]):
            sample_sets = [set(g.sample_id) for _, g in group.groupby("model")]
            self.assertTrue(all(ids == sample_sets[0] for ids in sample_sets[1:]))

    def test_confusion_matrix_and_metric_integrity(self):
        for noise in [0, 20, 50]:
            cm = pd.read_csv(RESULTS / f"confusion_matrix_{noise}pct.csv", index_col=0)
            self.assertEqual(int(cm.to_numpy().sum()), 440)
            self.assertEqual(int(cm.to_numpy().trace()), int(pd.read_csv(RESULTS / "confusion_summary.csv").set_index("noise_pct").loc[noise, "correct"]))
            self.assertTrue((cm.sum(axis=1) == 20).all())
            normalized = pd.read_csv(RESULTS / f"confusion_matrix_{noise}pct_normalized.csv", index_col=0)
            self.assertTrue(((normalized.sum(axis=1) - 1).abs() < 1e-10).all())
        preds = pd.read_csv(RESULTS / "predictions.csv")
        sample = preds[(preds.seed == 42) & (preds.model == "Extra Trees (augmentation)") & (preds.noise_pct == 20)]
        actual = (sample.y_true == sample.y_pred).mean()
        recorded = pd.read_csv(RESULTS / "confusion_summary.csv").set_index("noise_pct").loc[20, "accuracy"]
        self.assertAlmostEqual(actual, recorded)

    def test_statistical_tests_are_recomputable(self):
        paired = pd.read_csv(RESULTS / "paired_tests.csv")
        self.assertEqual(len(paired), 5)
        self.assertTrue((paired["both_correct"] + paired["baseline_only_correct"] + paired["augmented_only_correct"] + paired["both_incorrect"] == 440).all())
        self.assertTrue(((paired.holm_adjusted_p >= paired.raw_p) & (paired.holm_adjusted_p <= 1)).all())
        for row in paired.itertuples():
            discordant = row.baseline_only_correct + row.augmented_only_correct
            expected = binomtest(row.augmented_only_correct, discordant, .5).pvalue if discordant else 1.0
            self.assertAlmostEqual(row.raw_p, expected)

    def test_regional_coverage_and_figures(self):
        ranks = pd.read_csv(RESULTS / "ranking_ablation_by_region.csv")
        self.assertEqual(ranks[ranks.variant == "primary"].state_id.nunique(), 36)
        self.assertEqual(ranks[ranks.variant == "primary"].groupby("configuration").state_id.nunique().min(), 36)
        self.assertGreaterEqual(self.manifest["checks"]["figures"], 8)


if __name__ == "__main__":
    unittest.main()
