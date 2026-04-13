from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from db_setup import (
    get_connection,
    create_tables,
    clear_tables,
    write_all,
    build_agg_daily_metrics,
)

random.seed(42)
np.random.seed(42)


START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 3, 31)

CITIES = [
    ("Toronto", "Canada", "North America"),
    ("Vancouver", "Canada", "North America"),
    ("New York", "United States", "North America"),
    ("Tokyo", "Japan", "APAC"),
    ("Singapore", "Singapore", "APAC"),
]

DEVICE_TYPES = ["ios", "android", "web"]
CHANNELS = ["facebook_ads", "google_ads", "organic", "referral"]
PRODUCT_CATEGORIES = ["electronics", "fashion", "home", "beauty"]
PAYMENT_METHODS = ["credit_card", "paypal", "apple_pay"]


ANDROID_CONVERSION_DROP_DATE = datetime(2026, 2, 10).date()
GMV_DROP_DATE = datetime(2026, 2, 18).date()   # 某天 GMV 异常下跌
TORONTO_PROMO_START = datetime(2026, 3, 8).date()
TORONTO_PROMO_END = datetime(2026, 3, 14).date()


def daterange(start_dt: datetime, end_dt: datetime):
    current = start_dt
    while current <= end_dt:
        yield current
        current += timedelta(days=1)


def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5


def get_daily_base_new_users(dt: datetime) -> int:
    """
    周末流量波动：
    - 周末整体流量略高
    - 每月中旬略高
    """
    base = 180
    if is_weekend(dt):
        base = int(base * 1.20)
    if 10 <= dt.day <= 18:
        base = int(base * 1.08)
    return base


def channel_weights(dt: datetime) -> Dict[str, float]:
    """
    Facebook Ads 在 2 月中下旬开始放量，但质量变差。
    """
    weights = {
        "facebook_ads": 0.28,
        "google_ads": 0.26,
        "organic": 0.30,
        "referral": 0.16,
    }
    if dt.date() >= datetime(2026, 2, 15).date():
        weights["facebook_ads"] += 0.08
        weights["organic"] -= 0.04
        weights["google_ads"] -= 0.02
        weights["referral"] -= 0.02
    return weights


def device_weights() -> Dict[str, float]:
    return {
        "ios": 0.40,
        "android": 0.38,
        "web": 0.22,
    }


def city_weights() -> Dict[str, float]:
    return {
        "Toronto": 0.26,
        "Vancouver": 0.14,
        "New York": 0.24,
        "Tokyo": 0.22,
        "Singapore": 0.14,
    }


def weighted_choice(weight_map: Dict[str, float]) -> str:
    keys = list(weight_map.keys())
    weights = np.array(list(weight_map.values()), dtype=float)
    weights = weights / weights.sum()
    return np.random.choice(keys, p=weights)


def make_user_id(idx: int) -> str:
    return f"U{idx:07d}"


def make_event_id(idx: int) -> str:
    return f"E{idx:09d}"


def make_order_id(idx: int) -> str:
    return f"O{idx:09d}"


def make_session_id(idx: int) -> str:
    return f"S{idx:08d}"


def random_time_on_day(day: datetime, hour_start=8, hour_end=22) -> datetime:
    hour = random.randint(hour_start, hour_end)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime(day.year, day.month, day.day, hour, minute, second)


def generate_users() -> pd.DataFrame:
    users: List[Dict] = []
    user_counter = 1

    for dt in daterange(START_DATE, END_DATE):
        n_new_users = np.random.poisson(get_daily_base_new_users(dt))

        for _ in range(n_new_users):
            city = weighted_choice(city_weights())
            city_info = next(x for x in CITIES if x[0] == city)
            city_name, country, region = city_info

            device = weighted_choice(device_weights())
            channel = weighted_choice(channel_weights(dt))

            users.append({
                "user_id": make_user_id(user_counter),
                "register_date": dt.date(),
                "register_datetime": random_time_on_day(dt),
                "country": country,
                "city": city_name,
                "region": region,
                "device_type": device,
                "acquisition_channel": channel,
                "campaign_id": f"CMP_{dt.strftime('%Y%m')}_{channel.upper()}",
                "user_segment": "new_user",
                "is_premium": bool(np.random.rand() < 0.12),
            })
            user_counter += 1

    return pd.DataFrame(users)


def funnel_probabilities(user_row: pd.Series, current_date: datetime) -> Tuple[float, float, float]:
    """
    漏斗逻辑：
    app_open -> view_product -> add_to_cart -> purchase
    
    这里返回:
    - p_view_product
    - p_add_to_cart_given_view
    - p_purchase_given_cart
    """
    channel = user_row["acquisition_channel"]
    device = user_row["device_type"]
    city = user_row["city"]

    p_view = 0.68
    p_cart_given_view = 0.26
    p_purchase_given_cart = 0.42

    # 周末用户更活跃，view 概率更高
    if is_weekend(current_date):
        p_view += 0.05

    # Facebook Ads 低质量流量：view 还可以，但后链路转化差
    if channel == "facebook_ads" and current_date.date() >= datetime(2026, 2, 15).date():
        p_cart_given_view -= 0.04
        p_purchase_given_cart -= 0.08

    # Android 自 2/10 起支付转化变差
    if device == "android" and current_date.date() >= ANDROID_CONVERSION_DROP_DATE:
        p_purchase_given_cart -= 0.12

    # Toronto 活动周购买意愿提高
    if city == "Toronto" and TORONTO_PROMO_START <= current_date.date() <= TORONTO_PROMO_END:
        p_cart_given_view += 0.05
        p_purchase_given_cart += 0.08

    # 某天 GMV 异常下跌：支付大幅受损
    if current_date.date() == GMV_DROP_DATE:
        p_purchase_given_cart *= 0.35

    return (
        max(min(p_view, 0.95), 0.01),
        max(min(p_cart_given_view, 0.95), 0.01),
        max(min(p_purchase_given_cart, 0.95), 0.01),
    )


def generate_events_and_orders(users_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    events: List[Dict] = []
    orders: List[Dict] = []

    event_counter = 1
    order_counter = 1
    session_counter = 1

    for _, user in users_df.iterrows():
        reg_date = pd.to_datetime(user["register_date"]).to_pydatetime()

        # 用户注册后 0~10 天内可能活跃若干次
        n_active_days = np.random.randint(1, 5)
        active_offsets = sorted(np.random.choice(range(0, 10), size=n_active_days, replace=False))

        for offset in active_offsets:
            current_day = reg_date + timedelta(days=int(offset))
            if current_day > END_DATE:
                continue

            p_view, p_cart, p_purchase = funnel_probabilities(user, current_day)
            session_id = make_session_id(session_counter)
            session_counter += 1

            base_time = random_time_on_day(current_day)

            # app_open
            events.append({
                "event_id": make_event_id(event_counter),
                "user_id": user["user_id"],
                "event_time": base_time,
                "event_date": current_day.date(),
                "event_name": "app_open",
                "session_id": session_id,
                "product_id": None,
                "category": None,
                "platform": "app" if user["device_type"] != "web" else "web",
                "device_type": user["device_type"],
                "country": user["country"],
                "city": user["city"],
                "acquisition_channel": user["acquisition_channel"],
                "event_value": None,
            })
            event_counter += 1

            # 首页浏览
            if np.random.rand() < 0.88:
                events.append({
                    "event_id": make_event_id(event_counter),
                    "user_id": user["user_id"],
                    "event_time": base_time + timedelta(minutes=1),
                    "event_date": current_day.date(),
                    "event_name": "view_homepage",
                    "session_id": session_id,
                    "product_id": None,
                    "category": None,
                    "platform": "app" if user["device_type"] != "web" else "web",
                    "device_type": user["device_type"],
                    "country": user["country"],
                    "city": user["city"],
                    "acquisition_channel": user["acquisition_channel"],
                    "event_value": None,
                })
                event_counter += 1

            # 商品浏览
            viewed = np.random.rand() < p_view
            if viewed:
                category = random.choice(PRODUCT_CATEGORIES)
                product_id = f"P{random.randint(100, 999)}"
                events.append({
                    "event_id": make_event_id(event_counter),
                    "user_id": user["user_id"],
                    "event_time": base_time + timedelta(minutes=3),
                    "event_date": current_day.date(),
                    "event_name": "view_product",
                    "session_id": session_id,
                    "product_id": product_id,
                    "category": category,
                    "platform": "app" if user["device_type"] != "web" else "web",
                    "device_type": user["device_type"],
                    "country": user["country"],
                    "city": user["city"],
                    "acquisition_channel": user["acquisition_channel"],
                    "event_value": None,
                })
                event_counter += 1

            # 加购
            added_to_cart = viewed and (np.random.rand() < p_cart)
            if added_to_cart:
                events.append({
                    "event_id": make_event_id(event_counter),
                    "user_id": user["user_id"],
                    "event_time": base_time + timedelta(minutes=7),
                    "event_date": current_day.date(),
                    "event_name": "add_to_cart",
                    "session_id": session_id,
                    "product_id": product_id,
                    "category": category,
                    "platform": "app" if user["device_type"] != "web" else "web",
                    "device_type": user["device_type"],
                    "country": user["country"],
                    "city": user["city"],
                    "acquisition_channel": user["acquisition_channel"],
                    "event_value": None,
                })
                event_counter += 1

                events.append({
                    "event_id": make_event_id(event_counter),
                    "user_id": user["user_id"],
                    "event_time": base_time + timedelta(minutes=10),
                    "event_date": current_day.date(),
                    "event_name": "begin_checkout",
                    "session_id": session_id,
                    "product_id": product_id,
                    "category": category,
                    "platform": "app" if user["device_type"] != "web" else "web",
                    "device_type": user["device_type"],
                    "country": user["country"],
                    "city": user["city"],
                    "acquisition_channel": user["acquisition_channel"],
                    "event_value": None,
                })
                event_counter += 1

            # 购买
            purchased = added_to_cart and (np.random.rand() < p_purchase)
            if purchased:
                price_base = {
                    "electronics": 160,
                    "fashion": 85,
                    "home": 110,
                    "beauty": 60,
                }[category]

                amount = max(15, np.random.normal(price_base, price_base * 0.25))
                coupon_used = bool(np.random.rand() < 0.25)
                discount_amount = round(float(amount * np.random.uniform(0.05, 0.18)), 2) if coupon_used else 0.0
                paid_amount = round(float(amount - discount_amount), 2)

                # Toronto 活动周：优惠使用上升，但订单量明显抬高
                if user["city"] == "Toronto" and TORONTO_PROMO_START <= current_day.date() <= TORONTO_PROMO_END:
                    if np.random.rand() < 0.45:
                        coupon_used = True
                        discount_amount = round(float(amount * np.random.uniform(0.08, 0.20)), 2)
                        paid_amount = round(float(amount - discount_amount), 2)

                events.append({
                    "event_id": make_event_id(event_counter),
                    "user_id": user["user_id"],
                    "event_time": base_time + timedelta(minutes=14),
                    "event_date": current_day.date(),
                    "event_name": "purchase",
                    "session_id": session_id,
                    "product_id": product_id,
                    "category": category,
                    "platform": "app" if user["device_type"] != "web" else "web",
                    "device_type": user["device_type"],
                    "country": user["country"],
                    "city": user["city"],
                    "acquisition_channel": user["acquisition_channel"],
                    "event_value": paid_amount,
                })
                event_counter += 1

                orders.append({
                    "order_id": make_order_id(order_counter),
                    "user_id": user["user_id"],
                    "order_time": base_time + timedelta(minutes=15),
                    "order_date": current_day.date(),
                    "amount": paid_amount,
                    "currency": "CAD",
                    "status": "paid",
                    "payment_method": random.choice(PAYMENT_METHODS),
                    "coupon_used": coupon_used,
                    "discount_amount": discount_amount,
                    "product_category": category,
                    "country": user["country"],
                    "city": user["city"],
                    "acquisition_channel": user["acquisition_channel"],
                })
                order_counter += 1

            # 少量未支付/取消订单
            elif added_to_cart and np.random.rand() < 0.18:
                amount = round(float(max(15, np.random.normal(100, 25))), 2)
                orders.append({
                    "order_id": make_order_id(order_counter),
                    "user_id": user["user_id"],
                    "order_time": base_time + timedelta(minutes=15),
                    "order_date": current_day.date(),
                    "amount": amount,
                    "currency": "CAD",
                    "status": random.choice(["created", "cancelled"]),
                    "payment_method": random.choice(PAYMENT_METHODS),
                    "coupon_used": False,
                    "discount_amount": 0.0,
                    "product_category": category,
                    "country": user["country"],
                    "city": user["city"],
                    "acquisition_channel": user["acquisition_channel"],
                })
                order_counter += 1

    events_df = pd.DataFrame(events)
    orders_df = pd.DataFrame(orders)
    return events_df, orders_df


def build_agg_from_pandas(
    users_df: pd.DataFrame,
    events_df: pd.DataFrame,
    orders_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build agg_daily_metrics at a consistent grain:
    metric_date + region + country + city + acquisition_channel

    IMPORTANT:
    device_type is intentionally excluded from agg_daily_metrics in MVP,
    because fact_orders does not reliably carry device_type at the same grain.
    This avoids duplicated GMV / order metrics after merging.
    """

    # 1) new users
    users_daily = (
        users_df.groupby(
            ["register_date", "region", "country", "city", "acquisition_channel"],
            dropna=False
        )["user_id"]
        .nunique()
        .reset_index(name="new_users")
        .rename(columns={"register_date": "metric_date"})
    )

    # 2) DAU from core events
    active_events = events_df[
        events_df["event_name"].isin(
            ["app_open", "view_homepage", "view_product", "add_to_cart", "begin_checkout", "purchase"]
        )
    ].copy()

    if not active_events.empty:
        user_region = users_df[["user_id", "region"]].drop_duplicates()
        active_events = active_events.merge(user_region, on="user_id", how="left")

        dau_daily = (
            active_events.groupby(
                ["event_date", "region", "country", "city", "acquisition_channel"],
                dropna=False
            )["user_id"]
            .nunique()
            .reset_index(name="dau")
            .rename(columns={"event_date": "metric_date"})
        )
    else:
        dau_daily = pd.DataFrame(columns=[
            "metric_date", "region", "country", "city", "acquisition_channel", "dau"
        ])

    # 3) paid order metrics
    paid_orders = orders_df[orders_df["status"] == "paid"].copy()

    if not paid_orders.empty:
        paid_orders["region"] = paid_orders["country"].map({
            "Canada": "North America",
            "United States": "North America",
            "Mexico": "North America",
            "Japan": "APAC",
            "China": "APAC",
            "Singapore": "APAC",
        }).fillna("Europe")

        orders_daily = (
            paid_orders.groupby(
                ["order_date", "region", "country", "city", "acquisition_channel"],
                dropna=False
            )
            .agg(
                orders_paid=("order_id", "count"),
                purchasers=("user_id", "nunique"),
                gmv=("amount", "sum"),
            )
            .reset_index()
            .rename(columns={"order_date": "metric_date"})
        )
    else:
        orders_daily = pd.DataFrame(columns=[
            "metric_date", "region", "country", "city", "acquisition_channel",
            "orders_paid", "purchasers", "gmv"
        ])

    # 4) union key space
    all_keys = pd.concat([
        users_daily[["metric_date", "region", "country", "city", "acquisition_channel"]],
        dau_daily[["metric_date", "region", "country", "city", "acquisition_channel"]],
        orders_daily[["metric_date", "region", "country", "city", "acquisition_channel"]],
    ], ignore_index=True).drop_duplicates()

    # 5) merge all metrics on the SAME grain
    merged = all_keys.merge(
        users_daily,
        on=["metric_date", "region", "country", "city", "acquisition_channel"],
        how="left",
    ).merge(
        dau_daily,
        on=["metric_date", "region", "country", "city", "acquisition_channel"],
        how="left",
    ).merge(
        orders_daily,
        on=["metric_date", "region", "country", "city", "acquisition_channel"],
        how="left",
    )

    # 6) fill nulls
    merged["new_users"] = merged["new_users"].fillna(0).astype(int)
    merged["dau"] = merged["dau"].fillna(0).astype(int)
    merged["orders_paid"] = merged["orders_paid"].fillna(0).astype(int)
    merged["purchasers"] = merged["purchasers"].fillna(0).astype(int)
    merged["gmv"] = merged["gmv"].fillna(0.0).round(2)

    merged["avg_order_value"] = np.where(
        merged["orders_paid"] > 0,
        (merged["gmv"] / merged["orders_paid"]).round(2),
        0.0,
    )

    merged = merged[[
        "metric_date",
        "region",
        "country",
        "city",
        "acquisition_channel",
        "new_users",
        "dau",
        "orders_paid",
        "purchasers",
        "gmv",
        "avg_order_value",
    ]].copy()

    merged = merged.sort_values(
        by=["metric_date", "region", "country", "city", "acquisition_channel"]
    ).reset_index(drop=True)

    return merged


def seed_database(db_path: str = "agentic_analytics.duckdb") -> None:
    conn = get_connection(db_path)
    create_tables(conn)
    clear_tables(conn)

    print("Generating dim_users...")
    users_df = generate_users()
    print(f"dim_users rows: {len(users_df):,}")

    print("Generating fact_events and fact_orders...")
    events_df, orders_df = generate_events_and_orders(users_df)
    print(f"fact_events rows: {len(events_df):,}")
    print(f"fact_orders rows: {len(orders_df):,}")

    print("Generating agg_daily_metrics...")
    agg_df = build_agg_from_pandas(users_df, events_df, orders_df)
    print(f"agg_daily_metrics rows: {len(agg_df):,}")

    write_all(conn, {
        "dim_users": users_df,
        "fact_events": events_df,
        "fact_orders": orders_df,
        "agg_daily_metrics": agg_df,
    })

    print("Seeding completed.")

    # 简单 sanity checks
    print("\nSample checks:")

    print("\n[Check 1] Lowest GMV days from fact_orders:")
    print(conn.execute("""
        SELECT
            order_date,
            ROUND(SUM(amount), 2) AS gmv
        FROM fact_orders
        WHERE status = 'paid'
        GROUP BY 1
        ORDER BY gmv ASC
        LIMIT 5
    """).df())

    print("\n[Check 2] Toronto metrics sample:")
    print(conn.execute("""
        SELECT
            metric_date,
            city,
            acquisition_channel,
            new_users,
            dau,
            orders_paid,
            purchasers,
            gmv
        FROM agg_daily_metrics
        WHERE city = 'Toronto'
        ORDER BY metric_date DESC, acquisition_channel
        LIMIT 12
    """).df())

    print("\n[Check 3] Duplicate key check on agg_daily_metrics:")
    print(conn.execute("""
        SELECT
            metric_date,
            region,
            country,
            city,
            acquisition_channel,
            COUNT(*) AS row_count
        FROM agg_daily_metrics
        GROUP BY 1,2,3,4,5
        HAVING COUNT(*) > 1
        ORDER BY row_count DESC
        LIMIT 10
    """).df())

    print("\n[Check 4] Total GMV consistency:")
    print(conn.execute("""
        SELECT
            (SELECT ROUND(SUM(amount), 2) FROM fact_orders WHERE status = 'paid') AS orders_gmv,
            (SELECT ROUND(SUM(gmv), 2) FROM agg_daily_metrics) AS agg_gmv
    """).df())


if __name__ == "__main__":
    seed_database()
