"""Tests for Part B: Segmentation, RFM, PCA, Leakage Prevention, and Model Tuning."""

import unittest
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from src.common.config import CLEAN_DATA_DIR, OUTPUTS_DIR
from src.part_b.rfm_segmentation import build_rfm
from src.part_b.features import build_lapse_dataset, split_lapse_dataset, CUTOFF_DATE, LABEL_END_DATE
from src.part_b.models import oversample_minority_training


class TestPartBCustomerIntelligence(unittest.TestCase):

    def setUp(self):
        self.orders = pd.read_csv(CLEAN_DATA_DIR / "orders.csv")

    def test_rfm_summary_metrics(self):
        """RFM calculates recency, frequency, and monetary metrics on completed orders."""
        rfm = build_rfm(self.orders)
        self.assertEqual(len(rfm), 1175)
        self.assertEqual(rfm["recency"].median(), 19)
        self.assertEqual(rfm["recency"].min(), 0)
        self.assertEqual(rfm["recency"].max(), 180)
        self.assertAlmostEqual(rfm["frequency"].mean(), 9.46, places=2)
        self.assertEqual(rfm["frequency"].max(), 33)
        self.assertAlmostEqual(rfm["monetary"].mean(), 2552.06, places=2)
        self.assertAlmostEqual(rfm["monetary"].max(), 14766.55, places=2)

    def test_leakage_free_dataset_construction(self):
        """Features must use orders strictly before 2025-09-01; labels use September only."""
        X, y = build_lapse_dataset(CLEAN_DATA_DIR)
        self.assertEqual(len(X), 1156)
        self.assertEqual(int(y.sum()), 443)
        self.assertAlmostEqual(float(y.mean()), 0.3832, places=3)

        # Confirm target columns are excluded from feature set
        forbidden = {"lapsed", "is_premium", "segment_id", "segment_name", "customer_id"}
        self.assertTrue(forbidden.isdisjoint(set(X.columns)))
        self.assertFalse(X.isna().any().any())

    def test_stratified_train_test_split(self):
        """Train/test split is strictly disjoint and stratified."""
        X, y = build_lapse_dataset(CLEAN_DATA_DIR)
        X_train, X_test, y_train, y_test = split_lapse_dataset(X, y)

        self.assertEqual(len(X_train), 924)
        self.assertEqual(len(X_test), 232)
        self.assertEqual(set(X_train.index).intersection(set(X_test.index)), set())
        self.assertEqual(y_train.value_counts().to_dict(), {0: 570, 1: 354})
        self.assertEqual(y_test.value_counts().to_dict(), {0: 143, 1: 89})

    def test_oversampling_training_only(self):
        """Oversampling adds only minority training rows without affecting test set."""
        X, y = build_lapse_dataset(CLEAN_DATA_DIR)
        X_train, X_test, y_train, y_test = split_lapse_dataset(X, y)

        X_bal, y_bal = oversample_minority_training(X_train, y_train)
        self.assertEqual(y_bal.value_counts().to_dict(), {0: 570, 1: 570})
        self.assertTrue(set(X_bal.index).issubset(set(X_train.index)))
        self.assertTrue(set(X_bal.index).isdisjoint(set(X_test.index)))

    def test_model_reload_and_retention_list(self):
        """Saved model joblib loads cleanly and produces identical test predictions."""
        model_path = OUTPUTS_DIR / "lapse_model.joblib"
        self.assertTrue(model_path.is_file())

        model = joblib.load(model_path)
        X, y = build_lapse_dataset(CLEAN_DATA_DIR)
        _, X_test, _, _ = split_lapse_dataset(X, y)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        self.assertEqual(len(preds), len(X_test))
        self.assertEqual(probs.shape, (len(X_test), 2))

        retention_df = pd.read_csv(OUTPUTS_DIR / "retention_list.csv")
        self.assertEqual(len(retention_df), 10)
        self.assertTrue(retention_df["lapse_probability"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
