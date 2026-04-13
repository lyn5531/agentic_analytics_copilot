from __future__ import annotations

import os
import re
from datetime import date
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


def ask_model_for_sql(question: str) -> str:
    """
    Ask OpenAI model to generate SQL using the Responses API.

    Output policy:
    - Either return ONE clarification question in plain text
    - Or return ONE SQL code block only
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Please set it in your .env file.")

    client = OpenAI(api_key=api_key)
    instructions = build_instructions()

    today_str = str(date.today())

    user_input = f"""
User question:
{question}

Current reference date:
{today_str}

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
Use the current reference date above when interpreting relative time expressions.
If the question is answerable with the current schema and semantic context, do NOT ask a clarification question.
Prefer agg_daily_metrics for macro KPI and dimension breakdown queries.
Use fact_events for funnel analysis.
Only generate read-only SQL.

Return only Option A or Option B.
""".strip()
    
    response = client.responses.create(
    model=model,
    instructions=instructions,
    input=user_input,
    max_output_tokens=800,
    )

    return response.output_text.strip()

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
        print("\n[2] Extracted SQL:")
        print(sql)

        try:
            df = run_sql(sql)
            print("\n=== QUERY RESULT ===")
            print(df)
        except Exception as e:
            print("\n=== SQL EXECUTION ERROR ===")
            print(str(e))
    else:
        print("\nNo executable SQL extracted.")
        print("Model likely returned a clarification question:")
        print(model_output)

if __name__ == "__main__":
    main()
