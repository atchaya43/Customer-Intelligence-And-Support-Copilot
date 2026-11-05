"""Data cleaning pipeline for customers, products, and orders."""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from src.common.config import RAW_DATA_DIR, CLEAN_DATA_DIR


def clean_customers(customers_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean customers: standardize city/gender, handle invalid ages, impute with city median."""
    df = customers_raw.copy(deep=True)
    stats: Dict[str, Any] = {}

    # City standardization: strip whitespace and title-case
    df["city"] = df["city"].astype(str).str.strip().str.title()
    if df["city"].isna().any() or (df["city"] == "").any():
        raise ValueError("Customer cities cannot be empty or null")

    # Gender standardization
    gender_map = {
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female",
    }
    df["gender"] = df["gender"].astype(str).str.strip().str.lower().map(gender_map).fillna("unknown")

    # Age cleaning: numeric conversion, values outside [10, 100] converted to NaN
    age_numeric = pd.to_numeric(df["age"], errors="coerce")
    invalid_mask = age_numeric.notna() & ~age_numeric.between(10, 100)
    age_clean = age_numeric.mask(invalid_mask)

    # Impute missing ages with city median
    city_medians = df.groupby("city")[age_clean.name].transform(lambda s: age_clean.loc[s.index].median())
    df["age"] = age_clean.fillna(city_medians)
    if df["age"].isna().any():
        raise ValueError("Age imputation failed: some cities had no valid ages")

    # Signup date
    df["signup_date"] = pd.to_datetime(df["signup_date"], format="%Y-%m-%d", errors="raise")
    df["signup_date"] = df["signup_date"].dt.strftime("%Y-%m-%d")

    # Key constraints
    if df["customer_id"].isna().any() or df["customer_id"].duplicated().any():
        raise ValueError("Customer identifiers must be non-null and unique")

    df = df.sort_values("customer_id").reset_index(drop=True)
    return df, stats


def clean_products(products_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean products: ensure types and constraints."""
    df = products_raw.copy(deep=True)
    stats: Dict[str, Any] = {}

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["product_name"] = df["product_name"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="raise").astype(float)

    if (df["price_inr"] < 0).any():
        raise ValueError("Product prices cannot be negative")
    if df["product_id"].duplicated().any():
        raise ValueError("Product identifiers must be unique")

    df = df.sort_values("product_id").reset_index(drop=True)
    return df, stats


def clean_orders(orders_raw: pd.DataFrame, products_clean: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean orders: deduplicate snapshots, recover prices, handle delivery times."""
    df = orders_raw.copy(deep=True)
    stats: Dict[str, Any] = {}

    # Basic validations
    if df[["order_id", "customer_id", "product_id"]].isna().any().any():
        raise ValueError("Order, customer, and product IDs must not be missing")

    # Order timestamp
    df["order_ts"] = pd.to_datetime(df["order_ts"], format="%Y-%m-%d %H:%M:%S", errors="raise")

    # Deduplication: remove exact duplicate rows
    df = df.drop_duplicates().copy()

    # Multiple snapshots per order_id: sort by order_ts and keep latest
    df = df.sort_values("order_ts", kind="stable").drop_duplicates("order_id", keep="last").copy()

    # Status check
    if not df["status"].isin(["completed", "cancelled"]).all():
        raise ValueError("All latest order statuses must be completed or cancelled")

    # Recover order amount using product catalogue
    prod_lookup = products_clean.set_index("product_id")
    amounts_numeric = pd.to_numeric(df["order_amount"], errors="coerce")
    catalog_prices = df["product_id"].map(prod_lookup["price_inr"])
    quantities = pd.to_numeric(df["quantity"], errors="raise")
    discounts = pd.to_numeric(df["discount_pct"], errors="raise")

    recovered_amounts = (catalog_prices * quantities * (1.0 - discounts / 100.0)).round(2)
    df["order_amount"] = amounts_numeric.fillna(recovered_amounts)

    # Delivery minutes handling
    delivery_numeric = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    delivery_numeric = delivery_numeric.mask(delivery_numeric < 0)  # negative to NaN
    df["delivery_minutes"] = delivery_numeric

    # Impute missing completed orders with product category median (completed only)
    completed_mask = df["status"] == "completed"
    prod_categories = df["product_id"].map(prod_lookup["category"])
    cat_medians = df.loc[completed_mask].groupby(prod_categories[completed_mask])["delivery_minutes"].transform("median")

    df.loc[completed_mask, "delivery_minutes"] = df.loc[completed_mask, "delivery_minutes"].fillna(cat_medians)
    df.loc[~completed_mask, "delivery_minutes"] = np.nan

    # Format timestamp back to string
    df["order_ts"] = df["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df = df.sort_values("order_id").reset_index(drop=True)
    return df, stats


def clean_all_data(
    raw_dir: Path = RAW_DATA_DIR,
    clean_dir: Path = CLEAN_DATA_DIR,
) -> Dict[str, pd.DataFrame]:
    """Execute complete cleaning pipeline and save clean CSVs."""
    clean_dir.mkdir(parents=True, exist_ok=True)

    cust_raw = pd.read_csv(raw_dir / "customers.csv", keep_default_na=False)
    prod_raw = pd.read_csv(raw_dir / "products.csv", keep_default_na=False)
    orders_raw = pd.read_csv(raw_dir / "orders.csv", keep_default_na=False)

    customers_clean, _ = clean_customers(cust_raw)
    products_clean, _ = clean_products(prod_raw)
    orders_clean, _ = clean_orders(orders_raw, products_clean)

    # Save clean datasets
    customers_clean.to_csv(clean_dir / "customers.csv", index=False)
    products_clean.to_csv(clean_dir / "products.csv", index=False)
    orders_clean.to_csv(clean_dir / "orders.csv", index=False)

    return {
        "customers": customers_clean,
        "products": products_clean,
        "orders": orders_clean,
    }
