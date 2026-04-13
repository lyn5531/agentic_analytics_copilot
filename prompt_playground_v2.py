from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from sql_runner import run_sql


BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt_v1.txt"
SEMANTIC_CONTEXT_PATH = BASE_DIR / "semantic_context.yml"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_yaml_as_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


def build_instructions() -> str:
    """
    Combine the system prompt and semantic context into one instruction block.
    """
    system_prompt = load_text(SYSTEM_PROMPT_PATH)
    semantic_context = load_yaml_as_text(SEMANTIC_CONTEXT_PATH)

    return f"""
{system_prompt}

========================
SEMANTIC CONTEXT
========================
{semantic_context}
""".strip()


def extract_sql(text: str) -> str:
    """
    Extract SQL from a markdown sql code fence if present.
    Fallback: try to detect a raw SELECT / WITH query block.
    """
    sql_block_pattern = r"```sql\s*(.*?)```"
    match = re.search(sql_block_pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    generic_block_pattern = r"```(.*?)```"
    generic_match = re.search(generic_block_pattern, text, flags=re.DOTALL)
    if generic_match:
        candidate = generic_match.group(1).strip()
        if candidate.upper().startswith("SELECT") or candidate.upper().startswith("WITH"):
            return candidate

    raw_sql_match = re.search(
        r"((SELECT|WITH)\s+.*)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if raw_sql_match:
        return raw_sql_match.group(1).strip()

    return ""


def get_client_and_model() -> tuple[OpenAI, str]:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Please set it in your .env file.")

    client = OpenAI(api_key=api_key)
    return client, model


def get_data_reference_dates() -> dict[str, str]:
    """
    Read the maximum available dates from the database so the model resolves
    relative time expressions against the actual data window, not the machine date.
    """
    metric_df = run_sql("SELECT MAX(metric_date) AS max_metric_date FROM agg_daily_metrics")
    event_df = run_sql("SELECT MAX(event_date) AS max_event_date FROM fact_events")
    order_df = run_sql("SELECT MAX(order_date) AS max_order_date FROM fact_orders")

    max_metric_date = str(metric_df.iloc[0]["max_metric_date"])
    max_event_date = str(event_df.iloc[0]["max_event_date"])
    max_order_date = str(order_df.iloc[0]["max_order_date"])

    return {
        "max_metric_date": max_metric_date,
        "max_event_date": max_event_date,
        "max_order_date": max_order_date,
    }


def ask_model_for_sql(question: str) -> str:
    """
    First-pass SQL generation.

    Output policy:
    - Either return ONE clarification question in plain text
    - Or return ONE SQL code block only
    """
    client, model = get_client_and_model()
    instructions = build_instructions()
    ref_dates = get_data_reference_dates()

    user_input = f"""
User question:
{question}

Available data reference dates:
- max_metric_date: {ref_dates['max_metric_date']}
- max_event_date: {ref_dates['max_event_date']}
- max_order_date: {ref_dates['max_order_date']}

Time resolution policy:
- For KPI / aggregate-table queries, resolve relative dates using max_metric_date.
- For funnel / event queries, resolve relative dates using max_event_date.
- For order / transaction queries, resolve relative dates using max_order_date.
- Do not use the machine's current date.
- Do not assume data exists beyond these reference dates.

You must return EXACTLY ONE of the following:

Option A:
One single clarification question in plain text, with no heading and no extra text.

Use Option A only if the request is truly ambiguous and SQL cannot be generated safely.

Option B:
One single SQL code block only, in exactly this format:

```sql
SELECT ...```

Rules:
Do NOT output headings such as [PLAN], [SQL], [BUSINESS_EXPLANATION], or [CLARIFICATION].
Do NOT explain your reasoning.
Do NOT include commentary before or after the SQL code block.
Resolve relative time expressions like “昨天”, “上周”, “上个月”, “过去7天” into explicit date ranges before writing SQL.
If the question is answerable with the current schema and semantic context, do NOT ask a clarification question.
Prefer agg_daily_metrics for macro KPI and dimension breakdown queries.
Use fact_events for funnel analysis.
Use fact_orders for order-specific queries.
Only generate read-only SQL.
When using agg_daily_metrics, never reference base-level identifiers such as user_id, order_id, or event_id.
When using agg_daily_metrics, use aggregated columns directly, such as:
SUM(new_users)
SUM(dau)
SUM(orders_paid)
SUM(gmv)

Return only Option A or Option B.
""".strip()
    
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
        max_output_tokens=900,
    )

    return response.output_text.strip()

def ask_model_to_fix_sql(
    question: str,
    failed_sql: str,
    error_message: str,
    ) -> str:
    """
    Second-pass SQL repair.
    The model receives:
    - original question
    - failed SQL
    - database error
    and must return ONLY a corrected SQL code block.
    """
    client, model = get_client_and_model()
    instructions = build_instructions()
    ref_dates = get_data_reference_dates()
    repair_input = f"""
Original user question:
{question}

Available data reference dates:

max_metric_date: {ref_dates['max_metric_date']}
max_event_date: {ref_dates['max_event_date']}
max_order_date: {ref_dates['max_order_date']}

Time resolution policy:

For KPI / aggregate-table queries, resolve relative dates using max_metric_date.
For funnel / event queries, resolve relative dates using max_event_date.
For order / transaction queries, resolve relative dates using max_order_date.
Do not use the machine's current date.
Do not assume data exists beyond these reference dates.

The following SQL failed:
{failed_sql}
Database error:
{error_message}

Task:
Please fix the SQL so that it correctly answers the user's question using the available schema and semantic context.
Output requirements:

Return exactly one SQL code block only
Do NOT include explanation
Do NOT include headings
Do NOT include markdown text outside the SQL code block

Important repair guidance:

If a selected table is already aggregated, do NOT reference base-level identifiers like user_id, order_id, or event_id unless they really exist in that table.
For agg_daily_metrics, use aggregated columns directly, such as:
SUM(new_users)
SUM(dau)
SUM(orders_paid)
SUM(gmv)
Preserve the user's business intent.
Only generate read-only SQL.
If the previous SQL used a time range outside the available data window, correct it using the appropriate reference date.

Return only the corrected SQL code block.
""".strip()
    
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=repair_input,
        max_output_tokens=900,
    )

    return response.output_text.strip()

def execute_with_one_retry(question: str, sql: str) -> None:
    """
    Execute SQL once.
    If it fails, ask the model to repair SQL and retry once.
    """
    print("\n[2] Extracted SQL:")
    print(sql)

    try:
        df = run_sql(sql)
        print("\n=== QUERY RESULT ===")
        print(df)
        return
    except Exception as e:
        error_message = str(e)
        print("\n=== SQL EXECUTION ERROR (FIRST ATTEMPT) ===")
        print(error_message)

    print("\n[3] Asking model to repair SQL...")
    repaired_output = ask_model_to_fix_sql(
        question=question,
        failed_sql=sql,
        error_message=error_message,
    )

    print("\n=== RAW REPAIR OUTPUT ===")
    print(repaired_output)

    repaired_sql = extract_sql(repaired_output)

    if not repaired_sql or not (
        repaired_sql.upper().startswith("SELECT")
        or repaired_sql.upper().startswith("WITH")
    ):
        print("\nNo executable repaired SQL extracted.")
        return

    print("\n[4] Repaired SQL:")
    print(repaired_sql)

    try:
        df = run_sql(repaired_sql)
        print("\n=== QUERY RESULT AFTER REPAIR ===")
        print(df)
    except Exception as e:
        print("\n=== SQL EXECUTION ERROR (SECOND ATTEMPT) ===")
        print(str(e))

def main() -> None:
    print("=== Agentic Analytics Copilot Playground ===")
    question = input("Enter your analytics question: ").strip()
    if not question:
        print("No question entered.")
        return

    print("\n[1] Asking model...")
    model_output = ask_model_for_sql(question)

    print("\n=== RAW MODEL OUTPUT ===")
    print(model_output)

    sql = extract_sql(model_output)

    if sql and (sql.upper().startswith("SELECT") or sql.upper().startswith("WITH")):
        execute_with_one_retry(question, sql)
    else:
        print("\nNo executable SQL extracted.")
        print("Model likely returned a clarification question:")
        print(model_output)

if __name__ == "__main__":
    main()


