"""Database creation and data ingestion module for Instamart SQLite storage."""

from pathlib import Path
import sqlite3
import pandas as pd
from src.common.config import DB_PATH, CLEAN_DATA_DIR
from src.common.db import get_db_connection

DDL_SCHEMA = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;

CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY NOT NULL,
    city TEXT NOT NULL,
    signup_date TEXT NOT NULL,
    age REAL NOT NULL CHECK(age BETWEEN 10 AND 100),
    gender TEXT NOT NULL CHECK(gender IN ('male', 'female', 'unknown'))
);

CREATE TABLE products (
    product_id TEXT PRIMARY KEY NOT NULL,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    price_inr REAL NOT NULL CHECK(price_inr >= 0)
);

CREATE TABLE orders (
    order_id TEXT PRIMARY KEY NOT NULL,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    order_ts TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('completed', 'cancelled')),
    discount_pct REAL NOT NULL CHECK(discount_pct BETWEEN 0 AND 100),
    order_amount REAL NOT NULL CHECK(order_amount >= 0),
    delivery_minutes REAL CHECK(delivery_minutes IS NULL OR delivery_minutes >= 0)
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_product ON orders(product_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_order_ts ON orders(order_ts);
"""


def create_database(db_path: Path = DB_PATH) -> None:
    """Create the SQLite database schema with constraints and indexes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(DDL_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def load_clean_data_to_db(clean_dir: Path = CLEAN_DATA_DIR, db_path: Path = DB_PATH) -> None:
    """Load clean CSV files into the SQLite database."""
    customers_df = pd.read_csv(clean_dir / "customers.csv")
    products_df = pd.read_csv(clean_dir / "products.csv")
    orders_df = pd.read_csv(clean_dir / "orders.csv")

    conn = get_db_connection(db_path, read_only=False)
    try:
        customers_df.to_sql("customers", conn, if_exists="append", index=False)
        products_df.to_sql("products", conn, if_exists="append", index=False)
        orders_df.to_sql("orders", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()
