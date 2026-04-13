# Agentic Analytics Copilot

An AI analytics copilot prototype for growth and operations use cases.  
This project converts natural-language business questions into SQL queries, chart-ready outputs, and business analysis results.

## Project Overview

This prototype focuses on three MVP scenarios:
- KPI queries
- Dimension breakdown analysis
- Basic funnel analysis

Example questions:
- What was yesterday's GMV?
- Break down last month's new users by acquisition channel
- Show the 7-day funnel from `view_product` to `add_to_cart` to `purchase`

## Current Components

- `db_setup.py`: create DuckDB tables
- `seed_data.py`: generate mock business data with realistic patterns
- `semantic_context.yml`: semantic layer for metrics, joins, enums, and time rules
- `system_prompt_v1.txt`: prompt template for Text-to-SQL generation
- `sql_runner.py`: SQL validation and execution
- `prompt_playground.py`: OpenAI-powered SQL generation playground
- `prompt_playground_v2.py`: SQL generation with one retry/repair loop
- `manual_test_cases.py`: manual test queries
- `api_smoke_test.py`: OpenAI API connectivity test

## Mock Data Design

The generated data includes:
- weekend traffic fluctuations
- funnel drop-off
- channel quality differences
- GMV anomaly day
- Toronto promotion uplift scenario

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
