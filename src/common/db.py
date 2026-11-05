"""SQLite database management utilities."""

import sqlite3
from pathlib import Path
from typing import Optional
import pandas as pd
from .config import DB_PATH


def get_db_connection(db_path: Optional[Path] = None, read_only: bool = False) -> sqlite3.Connection:
    """Create a connection to the SQLite database with strict foreign keys enabled."""
    path = db_path or DB_PATH
    if read_only:
        uri_path = f"file:{path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri_path, uri=True)
    else:
        conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def execute_script(sql_script: str, db_path: Optional[Path] = None) -> None:
    """Execute a multi-statement SQL script against the database."""
    with get_db_connection(db_path, read_only=False) as conn:
        conn.executescript(sql_script)


def query_to_dataframe(query: str, params=(), db_path: Optional[Path] = None) -> pd.DataFrame:
    """Execute a SQL query and return the resulting pandas DataFrame."""
    with get_db_connection(db_path, read_only=True) as conn:
        return pd.read_sql_query(query, conn, params=params)
