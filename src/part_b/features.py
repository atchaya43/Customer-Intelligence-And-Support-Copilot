"""Point-in-time leakage-free feature engineering and cohort dataset generation for lapse prediction."""

from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.common.config import CLEAN_DATA_DIR

CUTOFF_DATE = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END_DATE = pd.Timestamp("2025-10-01 00:00:00")

NUMERIC_FEATURES: List[str] = [
    "recency",
    "frequency",
    "monetary",
    "average_discount",
    "cancellation_rate",
    "average_delivery",
    "age",
    "tenure",
]

CATEGORICAL_FEATURES: List[str] = [
    "favourite_category",
    "city",
    "gender",
]

ALL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_lapse_dataset(clean_dir: Path = CLEAN_DATA_DIR) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct leakage-free feature matrix from orders before cutoff and target labels in evaluation window."""
    customers = pd.read_csv(clean_dir / "customers.csv")
    products = pd.read_csv(clean_dir / "products.csv")
    orders = pd.read_csv(clean_dir / "orders.csv")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # Historical cohort: orders strictly before cutoff
    history = orders[orders["order_ts"] < CUTOFF_DATE].copy()
    completed = history[history["status"] == "completed"]

    # Compute RFM & behavioral features from completed orders
    features = completed.groupby("customer_id", sort=True).agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_discount=("discount_pct", "mean"),
        average_delivery=("delivery_minutes", "mean"),
    )
    features["recency"] = (CUTOFF_DATE - features.pop("last_order")).dt.days
    features["monetary"] = features["monetary"].round(2)

    # Cancellation rate across all historical attempts
    cancellation = (history["status"] == "cancelled").groupby(history["customer_id"]).mean()
    features["cancellation_rate"] = cancellation.reindex(features.index).fillna(0.0)

    # Favourite product category
    merged_prod = completed.merge(products[["product_id", "category"]], on="product_id")
    category_counts = merged_prod.groupby(["customer_id", "category"]).size().reset_index(name="order_count")
    fav_cat = category_counts.sort_values(
        ["customer_id", "order_count", "category"],
        ascending=[True, False, True],
    ).drop_duplicates("customer_id").set_index("customer_id")["category"]
    features["favourite_category"] = fav_cat.reindex(features.index)

    # Join customer demographic and tenure features
    cust_lookup = customers.set_index("customer_id")
    features["city"] = cust_lookup.loc[features.index, "city"]
    features["age"] = cust_lookup.loc[features.index, "age"]
    features["gender"] = cust_lookup.loc[features.index, "gender"]

    signup_dates = pd.to_datetime(cust_lookup.loc[features.index, "signup_date"])
    features["tenure"] = (CUTOFF_DATE - signup_dates).dt.days

    # Order features according to schema
    X = features[ALL_FEATURES].sort_index()

    # Target label: lapsed=1 if NO completed orders in September 2025; 0 otherwise
    sept_completed = orders[
        (orders["status"] == "completed")
        & (orders["order_ts"] >= CUTOFF_DATE)
        & (orders["order_ts"] < LABEL_END_DATE)
    ]["customer_id"].unique()

    y = pd.Series(
        (~X.index.isin(sept_completed)).astype(int),
        index=X.index,
        name="lapsed",
    )

    return X, y


def split_lapse_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified train/test split on cohort dataset."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
