"""Tests for Part A: Data cleaning edge cases, missing values, duplicates, and database schema."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.common.config import RAW_DATA_DIR, CLEAN_DATA_DIR, DB_PATH
from src.part_a.cleaner import clean_customers, clean_products, clean_orders
from src.part_a.audit import audit_raw_data
from src.part_a.database import get_db_connection


class TestPartADataPipeline(unittest.TestCase):

    def setUp(self):
        self.customers_raw = pd.read_csv(RAW_DATA_DIR / "customers.csv", keep_default_na=False)
        self.products_raw = pd.read_csv(RAW_DATA_DIR / "products.csv", keep_default_na=False)
        self.orders_raw = pd.read_csv(RAW_DATA_DIR / "orders.csv", keep_default_na=False)

    def test_raw_audit_metrics(self):
        """Audit accurately identifies exact raw counts, missing values, and anomalies."""
        metrics = audit_raw_data(RAW_DATA_DIR)
        self.assertEqual(metrics["raw_orders"], 13084)
        self.assertEqual(metrics["exact_duplicates"], 119)
        self.assertEqual(metrics["pending_rows"], 656)
        self.assertEqual(metrics["distinct_order_ids"], 12309)
        self.assertEqual(metrics["non_numeric_amounts"], 120)
        self.assertEqual(metrics["negative_delivery_times"], 106)
        self.assertEqual(metrics["blank_delivery_times"], 2541)
        self.assertEqual(metrics["blank_ages"], 61)
        self.assertEqual(metrics["invalid_ages"], 6)

    def test_customer_cleaning_edge_cases(self):
        """Customer cleaning standardizes cities, maps genders, and handles ages."""
        clean_df, _ = clean_customers(self.customers_raw)
        self.assertEqual(len(clean_df), 1200)
        self.assertFalse(clean_df["customer_id"].duplicated().any())
        self.assertEqual(set(clean_df["gender"].unique()), {"male", "female", "unknown"})
        self.assertTrue(clean_df["age"].between(10, 100).all())
        self.assertEqual(clean_df["age"].isna().sum(), 0)

        # Expected city counts
        counts = clean_df["city"].value_counts().to_dict()
        self.assertEqual(counts["Pune"], 227)
        self.assertEqual(counts["Delhi"], 210)
        self.assertEqual(counts["Bengaluru"], 197)

    def test_order_deduplication_and_latest_snapshot(self):
        """Duplicate orders are removed and latest status is preserved."""
        products_clean, _ = clean_products(self.products_raw)
        orders_clean, _ = clean_orders(self.orders_raw, products_clean)

        self.assertEqual(len(orders_clean), 12309)
        self.assertFalse(orders_clean["order_id"].duplicated().any())
        self.assertTrue(set(orders_clean["status"].unique()).issubset({"completed", "cancelled"}))

    def test_order_amount_recovery_and_revenue(self):
        """Missing or non-numeric amounts are recovered accurately from catalog price."""
        products_clean, _ = clean_products(self.products_raw)
        orders_clean, _ = clean_orders(self.orders_raw, products_clean)

        self.assertEqual(orders_clean["order_amount"].isna().sum(), 0)
        completed = orders_clean[orders_clean["status"] == "completed"]
        total_rev = round(float(completed["order_amount"].sum()), 2)
        self.assertEqual(total_rev, 2998668.80)

    def test_delivery_minutes_imputation(self):
        """Completed orders have positive imputed delivery minutes; cancelled orders are NaN."""
        products_clean, _ = clean_products(self.products_raw)
        orders_clean, _ = clean_orders(self.orders_raw, products_clean)

        completed = orders_clean[orders_clean["status"] == "completed"]
        cancelled = orders_clean[orders_clean["status"] == "cancelled"]

        self.assertEqual(completed["delivery_minutes"].isna().sum(), 0)
        self.assertTrue((completed["delivery_minutes"] >= 0).all())
        self.assertEqual(cancelled["delivery_minutes"].notna().sum(), 0)

    def test_database_foreign_key_and_constraints(self):
        """Database enforces foreign key constraints and valid ranges."""
        with get_db_connection(DB_PATH, read_only=True) as conn:
            cust_count = conn.execute("SELECT count(*) FROM customers").fetchone()[0]
            prod_count = conn.execute("SELECT count(*) FROM products").fetchone()[0]
            order_count = conn.execute("SELECT count(*) FROM orders").fetchone()[0]

            self.assertEqual(cust_count, 1200)
            self.assertEqual(prod_count, 40)
            self.assertEqual(order_count, 12309)


if __name__ == "__main__":
    unittest.main()
