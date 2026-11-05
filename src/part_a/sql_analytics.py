"""SQL query execution, pandas cross-checks, and checkpoint validation."""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from src.common.config import DB_PATH, SQL_DIR, CLEAN_DATA_DIR
from src.common.db import query_to_dataframe, get_db_connection
from src.part_a.audit import audit_raw_data

# Reference checkpoints for Part A validation
REFERENCE_METRICS = {
    "raw_orders": 13084,
    "exact_duplicates": 119,
    "pending_rows": 656,
    "distinct_order_ids": 12309,
    "non_numeric_amounts": 120,
    "negative_delivery_times": 106,
    "blank_delivery_times": 2541,
    "blank_ages": 61,
    "invalid_ages": 6,
    "raw_city_spellings": 24,
    "nonblank_gender_spellings": 10,
    "blank_genders": 30,
}

REFERENCE_CITY_COUNTS = {
    "Pune": 227,
    "Delhi": 210,
    "Bengaluru": 197,
    "Chennai": 191,
    "Hyderabad": 191,
    "Mumbai": 184,
}

REFERENCE_GENDER_COUNTS = {
    "female": 610,
    "male": 560,
    "unknown": 30,
}

REFERENCE_REVENUE = 2998668.80
REFERENCE_SQL_ROW_COUNTS = {"Q1": 30, "Q2": 30, "Q3": 11, "Q4": 25, "Q5": 550, "Q6": 1175, "Q7": 183}

REFERENCE_Q6_LEADERS = {
    "Bengaluru": ("C0082", 12856.20),
    "Chennai": ("C0930", 10151.40),
    "Delhi": ("C0291", 13742.60),
    "Hyderabad": ("C0743", 12730.05),
    "Mumbai": ("C0693", 14766.55),
    "Pune": ("C0426", 10377.10),
}

REFERENCE_CITY_AGGREGATES = {
    "Bengaluru": (523643.40, 276.47, 1894),
    "Mumbai": (479058.50, 297.00, 1613),
}


def load_sql_queries(sql_file: Path = SQL_DIR / "queries.sql") -> Dict[str, str]:
    """Parse queries.sql into individual named queries Q1 through Q7."""
    text = sql_file.read_text(encoding="utf-8")
    queries: Dict[str, str] = {}
    current_key = None
    buffer = []

    for line in text.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("-- Q") and ":" in trimmed:
            if current_key and buffer:
                queries[current_key] = "\n".join(buffer).strip()
                buffer = []
            current_key = trimmed.split(":")[0].replace("--", "").strip()
        elif current_key:
            buffer.append(line)

    if current_key and buffer:
        queries[current_key] = "\n".join(buffer).strip()

    return queries


def run_pandas_q1(orders: pd.DataFrame) -> pd.DataFrame:
    """Pandas cross-check for Q1."""
    completed = orders[(orders["status"] == "completed") & (orders["order_amount"] > 500)]
    grouped = completed.groupby("customer_id")["order_amount"].sum().reset_index()
    filtered = grouped[grouped["order_amount"] > 8000].copy()
    filtered["qualifying_spend"] = filtered["order_amount"].round(2)
    return filtered[["customer_id", "qualifying_spend"]].sort_values(
        by=["qualifying_spend", "customer_id"], ascending=[False, True]
    ).reset_index(drop=True)


def run_pandas_q2(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Pandas cross-check for Q2."""
    completed = orders[orders["status"] == "completed"].merge(customers, on="customer_id").merge(products, on="product_id")
    grouped = completed.groupby(["city", "category"])["order_amount"].sum().round(2).reset_index()
    grouped.rename(columns={"order_amount": "revenue"}, inplace=True)
    return grouped.sort_values(by=["revenue", "city", "category"], ascending=[False, True, True]).reset_index(drop=True)


def run_pandas_q3(products: pd.DataFrame) -> pd.DataFrame:
    """Pandas cross-check for Q3."""
    avg_price = products["price_inr"].mean()
    above = products[products["price_inr"] > avg_price].copy()
    return above[["product_id", "product_name", "category", "price_inr"]].sort_values(
        by=["price_inr", "product_id"], ascending=[False, True]
    ).reset_index(drop=True)


def run_pandas_q4(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Pandas cross-check for Q4."""
    completed_cust_ids = set(orders[orders["status"] == "completed"]["customer_id"])
    inactive = customers[~customers["customer_id"].isin(completed_cust_ids)].copy()
    return inactive[["customer_id", "city"]].sort_values("customer_id").reset_index(drop=True)


def run_pandas_q5(orders: pd.DataFrame) -> pd.DataFrame:
    """Pandas cross-check for Q5."""
    completed = orders[orders["status"] == "completed"]
    grouped = completed.groupby("customer_id")["order_amount"].sum().round(2).reset_index()
    grouped.rename(columns={"order_amount": "total_spend"}, inplace=True)
    avg_spend = grouped["total_spend"].mean()
    above = grouped[grouped["total_spend"] > avg_spend].copy()
    above["average_spend"] = avg_spend
    return above.sort_values(by=["total_spend", "customer_id"], ascending=[False, True]).reset_index(drop=True)


def run_pandas_q6(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Pandas cross-check for Q6."""
    completed = orders[orders["status"] == "completed"].merge(customers, on="customer_id")
    grouped = completed.groupby(["city", "customer_id"])["order_amount"].sum().round(2).reset_index()
    grouped.rename(columns={"order_amount": "total_spend"}, inplace=True)
    grouped = grouped.sort_values(by=["city", "total_spend"], ascending=[True, False]).reset_index(drop=True)
    grouped["spend_rank"] = grouped.groupby("city")["total_spend"].rank(method="min", ascending=False).astype(int)
    grouped["dense_rank"] = grouped.groupby("city")["total_spend"].rank(method="dense", ascending=False).astype(int)
    grouped["row_number"] = grouped.groupby("city").cumcount() + 1
    return grouped.sort_values(by=["city", "row_number"]).reset_index(drop=True)


def run_pandas_q7(orders: pd.DataFrame) -> pd.DataFrame:
    """Pandas cross-check for Q7."""
    completed = orders[orders["status"] == "completed"].copy()
    completed["day"] = completed["order_ts"].str.slice(0, 10)
    daily = completed.groupby("day")["order_amount"].sum().round(2).reset_index()
    daily.rename(columns={"order_amount": "revenue"}, inplace=True)
    daily = daily.sort_values("day").reset_index(drop=True)
    daily["cumulative_revenue"] = daily["revenue"].cumsum().round(2)
    return daily


def run_sql_analytics(db_path: Path = DB_PATH) -> Dict[str, pd.DataFrame]:
    """Execute all 7 SQL queries and cross-check against pandas implementations."""
    queries = load_sql_queries()
    results = {}
    for q_name, query in sorted(queries.items()):
        df = query_to_dataframe(query, db_path=db_path)
        results[q_name] = df
    return results


def verify_checkpoints(db_path: Path = DB_PATH, clean_dir: Path = CLEAN_DATA_DIR) -> Dict[str, Any]:
    """Validate all Part A requirements against authoritative reference metrics."""
    audit_metrics = audit_raw_data()
    for key, expected_val in REFERENCE_METRICS.items():
        actual_val = audit_metrics[key]
        assert actual_val == expected_val, f"Audit mismatch for {key}: expected {expected_val}, got {actual_val}"

    # Load clean data
    cust_df = pd.read_csv(clean_dir / "customers.csv")
    prod_df = pd.read_csv(clean_dir / "products.csv")
    orders_df = pd.read_csv(clean_dir / "orders.csv")

    # City counts
    actual_cities = cust_df["city"].value_counts().to_dict()
    for city, count in REFERENCE_CITY_COUNTS.items():
        assert actual_cities.get(city) == count, f"City count mismatch for {city}: expected {count}, got {actual_cities.get(city)}"

    # Gender counts
    actual_genders = cust_df["gender"].value_counts().to_dict()
    for gender, count in REFERENCE_GENDER_COUNTS.items():
        assert actual_genders.get(gender) == count, f"Gender count mismatch for {gender}: expected {count}, got {actual_genders.get(gender)}"

    # Total completed revenue
    completed_orders = orders_df[orders_df["status"] == "completed"]
    actual_rev = round(float(completed_orders["order_amount"].sum()), 2)
    assert actual_rev == REFERENCE_REVENUE, f"Revenue mismatch: expected {REFERENCE_REVENUE}, got {actual_rev}"

    # SQL query checks
    sql_results = run_sql_analytics(db_path=db_path)
    for q_name, expected_rows in REFERENCE_SQL_ROW_COUNTS.items():
        actual_rows = len(sql_results[q_name])
        assert actual_rows == expected_rows, f"Row count mismatch for {q_name}: expected {expected_rows}, got {actual_rows}"

    # Cross-checks with Pandas
    p_q1 = run_pandas_q1(orders_df)
    assert len(p_q1) == len(sql_results["Q1"])
    np.testing.assert_allclose(p_q1["qualifying_spend"], sql_results["Q1"]["qualifying_spend"])

    p_q2 = run_pandas_q2(cust_df, prod_df, orders_df)
    assert len(p_q2) == len(sql_results["Q2"])

    p_q3 = run_pandas_q3(prod_df)
    assert len(p_q3) == len(sql_results["Q3"])

    p_q4 = run_pandas_q4(cust_df, orders_df)
    assert len(p_q4) == len(sql_results["Q4"])

    p_q5 = run_pandas_q5(orders_df)
    assert len(p_q5) == len(sql_results["Q5"])

    p_q6 = run_pandas_q6(cust_df, orders_df)
    assert len(p_q6) == len(sql_results["Q6"])

    p_q7 = run_pandas_q7(orders_df)
    assert len(p_q7) == len(sql_results["Q7"])
    np.testing.assert_allclose(p_q7["cumulative_revenue"], sql_results["Q7"]["cumulative_revenue"])

    # Q6 leaders
    q6_df = sql_results["Q6"]
    q6_top = q6_df[q6_df["spend_rank"] == 1]
    for city, (expected_cust, expected_spend) in REFERENCE_Q6_LEADERS.items():
        row = q6_top[q6_top["city"] == city].iloc[0]
        assert row["customer_id"] == expected_cust, f"Q6 leader customer mismatch for {city}: expected {expected_cust}, got {row['customer_id']}"
        assert abs(row["total_spend"] - expected_spend) < 1e-2, f"Q6 leader spend mismatch for {city}: expected {expected_spend}, got {row['total_spend']}"

    # City aggregates
    for city, (exp_rev, exp_avg, exp_cnt) in REFERENCE_CITY_AGGREGATES.items():
        city_orders = completed_orders.merge(cust_df[cust_df["city"] == city], on="customer_id")
        actual_city_rev = round(float(city_orders["order_amount"].sum()), 2)
        actual_city_cnt = len(city_orders)
        actual_city_avg = round(float(city_orders["order_amount"].mean()), 2)
        assert actual_city_rev == exp_rev, f"City aggregate revenue mismatch for {city}: expected {exp_rev}, got {actual_city_rev}"
        assert actual_city_cnt == exp_cnt, f"City aggregate count mismatch for {city}: expected {exp_cnt}, got {actual_city_cnt}"
        assert actual_city_avg == exp_avg, f"City aggregate average mismatch for {city}: expected {exp_avg}, got {actual_city_avg}"

    return {
        "status": "PASS",
        "audit_metrics": audit_metrics,
        "clean_customers": len(cust_df),
        "clean_products": len(prod_df),
        "clean_orders": len(orders_df),
        "completed_revenue": actual_rev,
        "sql_row_counts": {k: len(v) for k, v in sql_results.items()},
    }
