# SQL Query Layer Design

## Goal

Add a natural-language question-answering layer so users can ask about both uploaded data and computed A/B analysis results, such as:

- "What is the total effect size for Premium?"
- "How many treatment users are in Premium?"
- "Which segment has the highest Bayesian probability?"

## Recommended Approach

Use a hybrid SQLite query layer:

- Keep the current pandas/Spark statistical analysis pipeline intact.
- Persist both raw uploaded data and computed analysis outputs into SQLite.
- Add a read-only NL-to-SQL service that queries SQLite and returns formatted answers.

This is the least invasive way to add durable, queryable state without rewriting the existing analyzer around SQL.

## Why Not The Other Options

### Raw-data SQL only

This would answer row-level questions, but analysis questions would still need a separate custom tool path. That creates an awkward split between "questions about data" and "questions about results."

### SQLite-first system of record

This would push ingestion and analysis orchestration into SQL too early. The current repo is built around pandas/Spark analyzers, not SQL-native inference, so this would add more migration risk than user value.

## Proposed Architecture

### 1. Session Query Store

Add a small persistence layer responsible for one SQLite database per active app session or uploaded dataset. That layer should:

- create a database file in a managed directory such as `output/query_store/`
- create stable table names for raw rows and analysis results
- expose schema metadata to the agent/query service

### 2. Persisted Tables

At minimum, store:

- `raw_data`
  - full uploaded dataset
- `analysis_segment_results`
  - one row per segment using canonical result fields
- `analysis_summary`
  - one row per run with overall summary values
- `analysis_segment_failures`
  - skipped/failed segments, when present

Optional later extensions:

- `analysis_diagnostics`
- `analysis_recommendations`
- `analysis_query_audit`

### 3. NL-to-SQL Query Service

Add a service that:

- receives the user question
- inspects the available SQLite schema
- builds a bounded prompt with:
  - allowed tables and columns
  - read-only SQL rules
  - example questions
- asks the model for SQL
- validates the SQL before execution
- executes the SQL with row limits
- formats the answer and result table for the user

### 4. Agent Integration

Add a dedicated tool for natural-language data questions. The agent should use it when:

- data has already been loaded
- the user is asking an informational question about the dataset or computed results
- the question is not better answered by the existing analysis/report tools directly

The tool should fail clearly when no data has been loaded yet.

## Data Flow

1. User uploads CSV.
2. Existing load path reads the file into pandas or Spark.
3. Raw data is mirrored into SQLite.
4. User runs analysis.
5. Segment results and summary outputs are mirrored into SQLite.
6. User asks a natural-language question.
7. Query service generates safe SQL and executes it against SQLite.
8. UI returns a concise explanation plus a small formatted table.

## Safety Constraints

The query layer should be explicitly constrained:

- read-only SQL only
- only `SELECT` and safe CTEs
- no DDL or DML
- no cross-database access
- bounded `LIMIT`
- schema allowlist
- user-facing error when SQL validation fails

This is important because the model will be generating SQL dynamically.

## UX Expectations

The feature should feel like a follow-on to analysis, not a separate product mode.

Good examples:

- "What is the total effect size for the Premium segment?"
- "Which segment has the lowest p-value?"
- "How many customers are in treatment vs control for Standard?"
- "Show me the top 5 rows where post_effect > 100"

Response shape:

- one-sentence answer first
- compact markdown table second
- note whether the answer came from raw data or analysis results

## Testing Strategy

Add coverage for:

- persistence of raw data into SQLite
- persistence of typed summary/result models into SQLite
- SQL validation rejecting unsafe statements
- NL question to SQL generation via mocked model output
- agent tool routing for query questions
- end-to-end happy path for raw-data and analysis-result questions

## Success Criteria

The design is successful when:

- users can ask questions about both raw data and computed analysis
- the statistical engine remains unchanged in its core responsibilities
- SQL execution is read-only and safe
- the answers are durable across turns within the active session
