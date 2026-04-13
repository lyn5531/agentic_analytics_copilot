from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


DB_PATH = "agentic_analytics.duckdb"


FORBIDDEN_SQL_PATTERNS = [
    r"\bINSERT\b",
    r"\bUPDATE\b",
    r"\bDELETE\b",
    r"\bDROP\b",
    r"\bALTER\b",
    r"\bCREATE\b",
    r"\bTRUNCATE\b",
    r"\bREPLACE\b",
]


def get_connection(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    if not Path(db_path).exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")
    return duckdb.connect(db_path)


def validate_sql(sql: str) -> None:
    """
    Basic SQL safety validation for MVP.
    Only allow SELECT / WITH...SELECT style queries.
    """
    normalized = sql.strip().upper()

    if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
        raise ValueError("Only SELECT or WITH ... SELECT queries are allowed.")

    for pattern in FORBIDDEN_SQL_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            raise ValueError(f"Forbidden SQL detected: {pattern}")


def run_sql(sql: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Validate and run SQL against DuckDB.
    """
    validate_sql(sql)
    conn = get_connection(db_path)
    try:
        df = conn.execute(sql).df()
        return df
    finally:
        conn.close()


if __name__ == "__main__":
    test_sql = """
    SELECT
        metric_date,
        SUM(gmv) AS total_gmv
    FROM agg_daily_metrics
    GROUP BY metric_date
    ORDER BY metric_date
    LIMIT 10
    """
    result = run_sql(test_sql)
    print(result)
