from __future__ import annotations

from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd


DB_PATH = "agentic_analytics.duckdb"


def get_connection(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Create or connect to a DuckDB database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)
    return conn


def create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all core tables for the MVP."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS dim_users (
        user_id VARCHAR PRIMARY KEY,
        register_date DATE,
        register_datetime TIMESTAMP,
        country VARCHAR,
        city VARCHAR,
        region VARCHAR,
        device_type VARCHAR,
        acquisition_channel VARCHAR,
        campaign_id VARCHAR,
        user_segment VARCHAR,
        is_premium BOOLEAN
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS fact_events (
        event_id VARCHAR PRIMARY KEY,
        user_id VARCHAR,
        event_time TIMESTAMP,
        event_date DATE,
        event_name VARCHAR,
        session_id VARCHAR,
        product_id VARCHAR,
        category VARCHAR,
        platform VARCHAR,
        device_type VARCHAR,
        country VARCHAR,
        city VARCHAR,
        acquisition_channel VARCHAR,
        event_value DOUBLE
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS fact_orders (
        order_id VARCHAR PRIMARY KEY,
        user_id VARCHAR,
        order_time TIMESTAMP,
        order_date DATE,
        amount DECIMAL(12, 2),
        currency VARCHAR,
        status VARCHAR,
        payment_method VARCHAR,
        coupon_used BOOLEAN,
        discount_amount DECIMAL(12, 2),
        product_category VARCHAR,
        country VARCHAR,
        city VARCHAR,
        acquisition_channel VARCHAR
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS agg_daily_metrics (
        metric_date DATE,
        region VARCHAR,
        country VARCHAR,
        city VARCHAR,
        acquisition_channel VARCHAR,
        new_users INTEGER,
        dau INTEGER,
        orders_paid INTEGER,
        purchasers INTEGER,
        gmv DECIMAL(14, 2),
        avg_order_value DECIMAL(12, 2)
    );
    """)


def clear_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Delete all existing data."""
    for table in ["agg_daily_metrics", "fact_orders", "fact_events", "dim_users"]:
        conn.execute(f"DELETE FROM {table};")


def write_dataframe(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    df: pd.DataFrame,
) -> None:
    """
    Insert a pandas DataFrame into a DuckDB table.
    """
    if df.empty:
        return
    conn.register("tmp_df", df)
    conn.execute(f"INSERT INTO {table_name} SELECT * FROM tmp_df")
    conn.unregister("tmp_df")


def write_all(
    conn: duckdb.DuckDBPyConnection,
    tables: Dict[str, pd.DataFrame],
) -> None:
    """Write multiple DataFrames into corresponding tables."""
    for table_name, df in tables.items():
        write_dataframe(conn, table_name, df)


def build_agg_daily_metrics(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Rebuild agg_daily_metrics from base tables.
    This is useful if you regenerate users/events/orders and want the aggregate refreshed.
    """
    conn.execute("DELETE FROM agg_daily_metrics;")

    conn.execute("""
    INSERT INTO agg_daily_metrics
    WITH new_users AS (
        SELECT
            register_date AS metric_date,
            region,
            country,
            city,
            device_type,
            acquisition_channel,
            COUNT(DISTINCT user_id) AS new_users
        FROM dim_users
        GROUP BY 1,2,3,4,5,6
    ),
    dau AS (
        SELECT
            event_date AS metric_date,
            region,
            country,
            city,
            device_type,
            acquisition_channel,
            COUNT(DISTINCT user_id) AS dau
        FROM (
            SELECT
                fe.event_date,
                du.region,
                fe.country,
                fe.city,
                fe.device_type,
                fe.acquisition_channel,
                fe.user_id
            FROM fact_events fe
            LEFT JOIN dim_users du
                ON fe.user_id = du.user_id
            WHERE fe.event_name IN (
                'app_open', 'view_homepage', 'view_product',
                'add_to_cart', 'begin_checkout', 'purchase'
            )
        ) t
        GROUP BY 1,2,3,4,5,6
    ),
    paid_orders AS (
        SELECT
            order_date AS metric_date,
            CASE
                WHEN country IN ('Canada', 'United States', 'Mexico') THEN 'North America'
                WHEN country IN ('Japan', 'China', 'Singapore') THEN 'APAC'
                ELSE 'Europe'
            END AS region,
            country,
            city,
            NULL::VARCHAR AS device_type,
            acquisition_channel,
            COUNT(*) FILTER (WHERE status = 'paid') AS orders_paid,
            COUNT(DISTINCT user_id) FILTER (WHERE status = 'paid') AS purchasers,
            COALESCE(SUM(amount) FILTER (WHERE status = 'paid'), 0) AS gmv
        FROM fact_orders
        GROUP BY 1,2,3,4,6
    ),
    all_keys AS (
        SELECT metric_date, region, country, city, device_type, acquisition_channel FROM new_users
        UNION
        SELECT metric_date, region, country, city, device_type, acquisition_channel FROM dau
        UNION
        SELECT metric_date, region, country, city, device_type, acquisition_channel FROM paid_orders
    )
    SELECT
        k.metric_date,
        k.region,
        k.country,
        k.city,
        k.device_type,
        k.acquisition_channel,
        COALESCE(n.new_users, 0) AS new_users,
        COALESCE(d.dau, 0) AS dau,
        COALESCE(p.orders_paid, 0) AS orders_paid,
        COALESCE(p.purchasers, 0) AS purchasers,
        COALESCE(p.gmv, 0) AS gmv,
        CASE
            WHEN COALESCE(p.orders_paid, 0) > 0
            THEN ROUND(p.gmv / p.orders_paid, 2)
            ELSE 0
        END AS avg_order_value
    FROM all_keys k
    LEFT JOIN new_users n
        ON k.metric_date = n.metric_date
       AND k.region = n.region
       AND k.country = n.country
       AND k.city = n.city
       AND k.device_type = n.device_type
       AND k.acquisition_channel = n.acquisition_channel
    LEFT JOIN dau d
        ON k.metric_date = d.metric_date
       AND k.region = d.region
       AND k.country = d.country
       AND k.city = d.city
       AND k.device_type = d.device_type
       AND k.acquisition_channel = d.acquisition_channel
    LEFT JOIN paid_orders p
        ON k.metric_date = p.metric_date
       AND k.region = p.region
       AND k.country = p.country
       AND k.city = p.city
       AND k.acquisition_channel = p.acquisition_channel
    ;
    """)


if __name__ == "__main__":
    conn = get_connection()
    create_tables(conn)
    print("DuckDB tables created successfully.")
