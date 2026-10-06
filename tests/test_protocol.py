"""Tests for temporal boundaries, cohort isolation, resampling, and metrics."""
import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import FEATURES, load_data, select_partitions, make_pipeline, score_predictions


class ProtocolTests(unittest.TestCase):
    def frame(self):
        rows = []
        for company in range(4):
            for year in range(2000, 2005):
                row = {feature: float(company + 1) for feature in FEATURES}
                row.update(company_name=f'C_{company}', year=year, X8=500.0,
                           status_label='failed' if company < 2 else 'alive',
                           target=int(company < 2 and year in (2001, 2004)))
                rows.append(row)
        return pd.DataFrame(rows)

    def test_calendar_separation(self):
        train, valid = select_partitions(self.frame(), 2002, 2004)
        self.assertLess(train.year.max(), valid.year.min())
        self.assertFalse(set(train.year) & set(valid.year))

    def test_future_values_do_not_change_cohort(self):
        frame = self.frame()
        old_train, old_valid = select_partitions(frame, 2002, 2004)
        frame.loc[frame.year > 2002, 'X8'] = 1e9
        new_train, new_valid = select_partitions(frame, 2002, 2004)
        pd.testing.assert_frame_equal(old_train, new_train)
        self.assertEqual(set(old_valid.company_name), set(new_valid.company_name))

    def test_resampling_only_during_fit(self):
        rng = np.random.default_rng(42)
        features = pd.DataFrame(rng.normal(size=(40, 18)), columns=FEATURES)
        labels = pd.Series([0] * 32 + [1] * 8)
        model = make_pipeline({'n_estimators': 5, 'min_child_samples': 2}, 8)
        model.fit(features, labels)
        self.assertEqual(model.named_steps['scale'].n_samples_seen_, 40)
        before = features.iloc[:7].copy()
        predicted = model.predict_proba(before)
        self.assertEqual(predicted.shape, (7, 2))
        pd.testing.assert_frame_equal(before, features.iloc[:7])
        np.testing.assert_allclose(predicted.sum(axis=1), 1.0)

    def test_auc_uses_scores_and_counts_match(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.3, 0.2, 0.4])
        result = score_predictions(labels, probabilities)
        self.assertEqual(result['roc_auc'], roc_auc_score(labels, probabilities))
        self.assertEqual(result['roc_auc'], 0.75)
        self.assertEqual(result['tp'], 0)
        self.assertEqual(result['fn'], 2)
        self.assertEqual(result['tn'], 2)

    def test_single_class_rejected(self):
        frame = self.frame()
        frame['target'] = 0
        with self.assertRaises(ValueError):
            select_partitions(frame, 2002, 2004)

    def test_terminal_label_is_independent_of_input_order(self):
        frame = self.frame().drop(columns='target').sample(frac=1, random_state=11)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.csv'
            frame.to_csv(path, index=False)
            loaded = load_data(path)
        positives = loaded.loc[loaded.target == 1]
        self.assertEqual(len(positives), 2)
        self.assertEqual(set(positives.year), {2004})
        self.assertEqual(set(positives.company_name), {'C_0', 'C_1'})

    def test_duplicate_company_year_rejected(self):
        frame = self.frame().drop(columns='target')
        frame = pd.concat([frame, frame.iloc[[0]]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.csv'
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_data(path)


if __name__ == '__main__':
    unittest.main()
