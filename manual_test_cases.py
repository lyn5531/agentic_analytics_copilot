from sql_runner import run_sql


TEST_CASES = [
    {
        "name": "yesterday_gmv",
        "question": "昨天总 GMV 是多少？",
        "sql": """
        SELECT
            SUM(gmv) AS total_gmv
        FROM agg_daily_metrics
        WHERE metric_date = DATE '2026-03-31'
        """
    },
    {
        "name": "march_new_users_by_channel",
        "question": "按 acquisition_channel 拆解上个月新增注册用户数",
        "sql": """
        SELECT
            acquisition_channel,
            SUM(new_users) AS total_new_users
        FROM agg_daily_metrics
        WHERE metric_date BETWEEN DATE '2026-03-01' AND DATE '2026-03-31'
        GROUP BY acquisition_channel
        ORDER BY total_new_users DESC
        """
    },
    {
        "name": "7d_funnel",
        "question": "统计过去 7 天从 view_product 到 add_to_cart 再到 purchase 的转化率漏斗",
        "sql": """
        WITH funnel_users AS (
          SELECT
            event_name,
            COUNT(DISTINCT user_id) AS users
          FROM fact_events
          WHERE event_date BETWEEN DATE '2026-03-25' AND DATE '2026-03-31'
            AND event_name IN ('view_product', 'add_to_cart', 'purchase')
          GROUP BY event_name
        ),
        pivoted AS (
          SELECT
            MAX(CASE WHEN event_name = 'view_product' THEN users END) AS view_product_users,
            MAX(CASE WHEN event_name = 'add_to_cart' THEN users END) AS add_to_cart_users,
            MAX(CASE WHEN event_name = 'purchase' THEN users END) AS purchase_users
          FROM funnel_users
        )
        SELECT
          view_product_users,
          add_to_cart_users,
          purchase_users,
          ROUND(add_to_cart_users * 1.0 / NULLIF(view_product_users, 0), 4) AS view_to_cart_conversion,
          ROUND(purchase_users * 1.0 / NULLIF(add_to_cart_users, 0), 4) AS cart_to_purchase_conversion
        FROM pivoted
        """
    },
]


def main() -> None:
    for case in TEST_CASES:
        print(f"\n=== TEST: {case['name']} ===")
        print(f"Question: {case['question']}")
        df = run_sql(case["sql"])
        print(df)


if __name__ == "__main__":
    main()
