"""Raw data audit module for Instamart datasets."""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from src.common.config import RAW_DATA_DIR


def audit_raw_data(raw_dir: Path = RAW_DATA_DIR) -> Dict[str, Any]:
    """Perform comprehensive audit on raw CSV exports and return key metrics."""
    cust_path = raw_dir / "customers.csv"
    orders_path = raw_dir / "orders.csv"
    prod_path = raw_dir / "products.csv"

    customers_raw = pd.read_csv(cust_path, keep_default_na=False)
    orders_raw = pd.read_csv(orders_path, keep_default_na=False)
    products_raw = pd.read_csv(prod_path, keep_default_na=False)

    # Order metrics
    raw_orders = len(orders_raw)
    exact_duplicates = len(orders_raw) - len(orders_raw.drop_duplicates())
    pending_rows = int((orders_raw["status"] == "pending").sum())
    distinct_order_ids = int(orders_raw["order_id"].nunique())

    # Numeric conversions
    order_amounts_numeric = pd.to_numeric(orders_raw["order_amount"], errors="coerce")
    non_numeric_amounts = int(order_amounts_numeric.isna().sum())

    delivery_numeric = pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce")
    blank_delivery_times = int((orders_raw["delivery_minutes"].str.strip() == "").sum())
    negative_delivery_times = int((delivery_numeric < 0).sum())

    # Customer metrics
    age_numeric = pd.to_numeric(customers_raw["age"], errors="coerce")
    blank_ages = int((customers_raw["age"].str.strip() == "").sum())
    invalid_ages = int((age_numeric.notna() & ((age_numeric < 10) | (age_numeric > 100))).sum())

    raw_city_spellings = int(customers_raw["city"].nunique())
    blank_genders = int((customers_raw["gender"].str.strip() == "").sum())
    nonblank_gender_spellings = int(
        customers_raw.loc[customers_raw["gender"].str.strip() != "", "gender"].nunique()
    )

    metrics = {
        "raw_orders": raw_orders,
        "exact_duplicates": exact_duplicates,
        "pending_rows": pending_rows,
        "distinct_order_ids": distinct_order_ids,
        "non_numeric_amounts": non_numeric_amounts,
        "negative_delivery_times": negative_delivery_times,
        "blank_delivery_times": blank_delivery_times,
        "blank_ages": blank_ages,
        "invalid_ages": invalid_ages,
        "raw_city_spellings": raw_city_spellings,
        "nonblank_gender_spellings": nonblank_gender_spellings,
        "blank_genders": blank_genders,
    }

    return metrics
