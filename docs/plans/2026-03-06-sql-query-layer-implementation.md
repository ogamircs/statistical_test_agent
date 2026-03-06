# SQL Query Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a hybrid SQLite-backed natural-language query layer for both raw uploaded data and computed A/B analysis results.

**Architecture:** Preserve the current pandas/Spark analysis pipeline and mirror raw data plus typed analysis outputs into SQLite. Add a read-only NL-to-SQL service, a new agent tool, and tests for persistence, validation, and end-to-end question answering.

**Tech Stack:** Python, sqlite3, pandas/Spark adapters, existing agent tools, OpenAI/LangChain model calls, pytest

---

### Task 1: Define the SQLite persistence contract

**Files:**
- Create: `src/query_store.py`
- Test: `tests/test_query_store.py`

**Step 1: Write the failing test**

Cover:

- creating a query-store database
- writing a raw dataframe into `raw_data`
- writing analysis segment rows into `analysis_segment_results`
- writing summary rows into `analysis_summary`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_query_store.py -q`

**Step 3: Write minimal implementation**

Implement a small `SQLiteQueryStore` with:

- database path creation
- `save_raw_dataframe()`
- `save_segment_results()`
- `save_summary()`
- `list_tables()` / `get_schema()`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_query_store.py -q`

### Task 2: Persist raw data and analysis outputs

**Files:**
- Modify: `src/agent.py`
- Modify: `src/agent_tools.py`
- Modify: `src/statistics/models.py`
- Test: `tests/test_agent.py`

**Step 1: Write the failing test**

Add agent/tool tests proving that:

- loading a CSV persists raw data to SQLite
- running analysis persists summary and segment results

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py -k query_store -q`

**Step 3: Write minimal implementation**

Add a query-store instance to the active session agent and populate it after:

- load
- auto-analyze
- full analysis
- configure-and-analyze

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent.py -k query_store -q`

### Task 3: Add safe SQL validation and execution

**Files:**
- Create: `src/sql_query_service.py`
- Test: `tests/test_sql_query_service.py`

**Step 1: Write the failing test**

Cover:

- safe `SELECT` statements are accepted
- destructive SQL is rejected
- missing-table references are rejected
- row limits are enforced

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sql_query_service.py -q`

**Step 3: Write minimal implementation**

Implement:

- SQL statement validation
- schema exposure helpers
- read-only query execution
- compact tabular result formatting

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sql_query_service.py -q`

### Task 4: Add NL-to-SQL prompting

**Files:**
- Modify: `src/sql_query_service.py`
- Test: `tests/test_sql_query_service.py`

**Step 1: Write the failing test**

Mock the model layer and verify:

- raw-data questions map to `raw_data`
- analysis-result questions map to `analysis_segment_results` / `analysis_summary`
- invalid model SQL is rejected cleanly

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sql_query_service.py -k nl -q`

**Step 3: Write minimal implementation**

Add a bounded prompt template with:

- schema
- allowed SQL rules
- example mappings
- final SQL-only output requirement

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sql_query_service.py -k nl -q`

### Task 5: Add agent tool and routing

**Files:**
- Modify: `src/agent_tools.py`
- Modify: `src/agent.py`
- Modify: `src/agent_reporting.py`
- Test: `tests/test_agent.py`

**Step 1: Write the failing test**

Cover:

- asking a question before data is loaded returns a clean error
- raw-data question returns a formatted answer
- analysis-result question returns a formatted answer

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py -k sql_query -q`

**Step 3: Write minimal implementation**

Add a new tool such as `answer_data_question` and update the agent prompt so it uses the new tool for natural-language questions about loaded data/results.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent.py -k sql_query -q`

### Task 6: End-to-end verification

**Files:**
- Test: `tests/test_agent.py`
- Test: `tests/test_query_store.py`
- Test: `tests/test_sql_query_service.py`

**Step 1: Run focused suites**

Run:

- `pytest tests/test_query_store.py -q`
- `pytest tests/test_sql_query_service.py -q`
- `pytest tests/test_agent.py -k "query_store or sql_query" -q`

**Step 2: Run full regression suite**

Run: `pytest -q`

**Step 3: Manual app check**

Verify in the running app that:

- upload persists raw data
- analysis persists result tables
- questions like "what is the total effect size for Premium?" return a clean answer and compact table
