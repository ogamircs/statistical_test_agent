# TODO — Improvement Backlog

Prioritized backlog for the A/B Testing Analysis Agent. Items are grouped into five buckets ordered by leverage on user outcomes. Bucket 1 (statistical correctness) takes precedence over later buckets — a polished agent that misreports a significance call is worse than a rougher agent that calls it correctly.

For background see [architecture.md](architecture.md), [development.md](development.md), and [testing.md](testing.md).

Effort tags: **(S)** ≤ ½ day, **(M)** 1–3 days, **(L)** > 3 days.

> **Status (2026-04-19):** All 35 items shipped across waves 1–6. Each item links to the commit on `codex/agent-runtime-ui-polish` via `git log --grep "TODO.md #N"`. Use this file as a historical record; new ideas should be opened as fresh GitHub issues.

---

## Bucket 1 — Statistical correctness

Highest leverage. These items affect what the agent tells users about their experiments.

1. [x] **SRM guardrail enforcement** — `diagnostics.py` already runs a chi-square SRM check, but the result is buried in `result.diagnostics["experiment_quality"]["srm"]`. Promote to a top-level `inference_guardrail_triggered` flag on `ABTestResult` and have the reporter block "statistically significant" claims when SRM `p < 0.05`. _Files:_ `src/statistics/diagnostics.py`, `src/statistics/models.py`, `src/agent_reporting.py`. **(M)**
2. [x] **Ratio metrics with delta-method SE** — no ratio metric family today (revenue/user, CTR as ratio of sums). Add `MetricFamily.RATIO` with delta-method confidence intervals. _Files:_ `src/statistics/model_families.py`, `src/statistics/statsmodels_engine.py`. **(L)**
3. [x] **CUPED variance reduction** — pre-period data is only used for AA tests and DiD. Add a CUPED option to `CovariateResolver` (regress Y on pre-Y, analyze residuals). _Files:_ `src/statistics/covariate_resolver.py`. **(M)**
4. [x] **Sequential alpha-spending: enforce, not advise** — O'Brien–Fleming / Pocock decisions are advisory only. When `SequentialConfig` is active, the sequential bound should *be* the significance threshold rather than a side recommendation. _Files:_ `src/statistics/sequential_config.py`, `src/statistics/analyzer.py`. **(M)**
5. [x] **Surface diagnostics violations in chat** — normality, Levene, and outlier-sensitivity results are computed and silently dropped. Add a "Diagnostics" section to `render_full_analysis_output` listing failed checks with one-line guidance. _Files:_ `src/agent_reporting.py`. **(S)**
6. [x] **Render Bayesian results in summaries** — `prob_treatment_better` and credible intervals are populated on `ABTestResult` but never shown to the user. _Files:_ `src/agent_reporting.py`. **(S)**
7. [x] **NaN-drop accounting** — `dropna()` in `data_manager` / `segment_preparer` silently removes rows. Add a `rows_dropped` field to `ABTestResult` and surface a warning when the drop exceeds 5%. _Files:_ `src/statistics/data_manager.py`, `src/statistics/segment_preparer.py`, `src/statistics/models.py`. **(S)**
8. [x] **Duplicate / repeated-measures detection** — warn when `nunique(customer_id) < len(df)` or when the same customer appears in both arms (independence violation). _Files:_ `src/statistics/data_manager.py`, `src/statistics/diagnostics.py`. **(S)**
9. [x] **Achieved-MDE field** — `power_analysis` computes required-N for a target power but never the inverse: the smallest detectable effect at the current sample. Add `achieved_mde` to `ABTestResult`. _Files:_ `src/statistics/power_analysis.py`, `src/statistics/models.py`. **(S)**
10. [x] **Configurable random seed** — `np.random.seed(42)` is hard-coded inside bootstrap, AA, and Bayesian routines. Hoist to `ABTestAnalyzer(seed=...)` and thread through. _Files:_ `src/statistics/analyzer.py`, `src/statistics/bayesian.py`, `src/statistics/experiment_design.py`, `src/statistics/pyspark_analyzer.py`. **(S)**

---

## Bucket 2 — pandas / Spark parity

The Spark backend currently lags pandas in covariates, diagnostics, and data querying. Every Spark-touching change in this bucket must land with a corresponding case in `tests/test_parity_pandas_spark.py`.

11. [x] **Port covariate adjustment to Spark** — Spark backend has no covariate support today. Use Spark MLlib OLS, or driver-side numpy collect for small-K covariate sets. _Files:_ `src/statistics/pyspark_analyzer.py` (reuse `src/statistics/covariate_resolver.py`). **(L)**
12. [x] **Real Spark diagnostics** — Spark currently returns an empty diagnostics payload. Implement at least normality (KS on a sample), variance ratio, and SRM. _Files:_ `src/statistics/pyspark_analyzer.py`. **(M)**
13. [x] **Spark `query_data` via Spark SQL** — README marks this unsupported. Add a minimal Spark SQL execution path so chat-side data questions work on big files. _Files:_ `src/statistics/pyspark_analyzer.py`, `src/sql_query_service.py`, `src/tooling/analysis.py`. **(M)**
14. [x] **Spark seed correctness** — driver-side `np.random.seed(42)` is not respected per executor, so distributed Monte Carlo isn't reproducible. Use `pyspark.sql.functions.rand(seed)` or broadcast a seeded `Generator`. _Files:_ `src/statistics/pyspark_analyzer.py`. **(S)**
15. [x] **Parity-test reminder** — every Bucket 2 change adds a case to `tests/test_parity_pandas_spark.py` before merge. Tracked here as one rolling reminder rather than per-feature items.

---

## Bucket 3 — Production readiness, observability, deployment

16. [x] **Session-store GC** — `output/query_store/session-*.sqlite` accumulates indefinitely (a recent local audit found 43 MB across 739 files). Add a startup sweep that deletes files older than N days and caps per-session DB size. _Files:_ `src/agent_session.py`, new `src/query_store_gc.py`. **(S)**
17. [x] **Persist + restore chat history on `on_chat_resume`** — the current handler in `app.py` instantiates a fresh `ABTestingAgent` on reconnect, dropping history. Persist history alongside the session SQLite and rehydrate on resume. _Files:_ `app.py`, `src/agent_session.py`, `src/query_store.py`. **(M)**
18. [x] **SQL query timeout + audit log** — `SQLiteQueryStore.execute_query` has no wall-clock timeout. Add a 5 s limit and write executed SQL to an audit table. (Read-only mode already landed in Wave 3.) _Files:_ `src/query_store.py`. **(S)**
19. [x] **Externalize config** — model name (`gpt-5.2`), temperature, `FILE_SIZE_THRESHOLD_MB` (2.0), and the default SQL `LIMIT 20` are scattered constants. Introduce `src/config.py` with a `Config.from_env()` factory and validation in `app.py`. _Files:_ `src/config.py` (new), `src/agent.py`, `src/agent_runtime.py`, `src/sql_query_service.py`, `app.py`. **(M)**
20. [x] **Structured logging + token / latency metrics** — adopt `structlog` (or a stdlib JSON formatter), hook `ChatOpenAI` callbacks to capture `prompt_tokens` / `completion_tokens`, wrap each tool with timing. Optional LangSmith via `LANGCHAIN_TRACING_V2=true`. _Files:_ `src/agent.py`, `src/tooling/common.py`. **(M)**
21. [x] **Dockerfile + deploy doc** — no container today. Multi-stage `uv`-based Dockerfile and a `docs/deployment.md` covering Railway/Fly. _Files:_ `Dockerfile` (new), `docs/deployment.md` (new). **(M)**
22. [x] **Chainlit auth + upload caps** — UI is currently open. Add token-based auth (`@cl.password_auth_callback` or header bearer) and cap upload size in `.chainlit/config.toml`. _Files:_ `app.py`, `.chainlit/config.toml`. **(S)**

---

## Bucket 4 — Tooling, prompts, eval harness

23. [x] **Externalize system prompt + version** — currently inline in `src/agent.py` (lines ~233–280). Move to `src/prompts/system.md`, load via `importlib.resources`, and log a `PROMPT_VERSION` constant on each agent run. _Files:_ `src/prompts/system.md` (new), `src/agent.py`. **(S)**
24. [x] **Golden-task eval harness** — `tests/test_agent.py` only exercises a mocked graph. Add `tests/eval/test_golden_tasks.py` with 10–20 fixtures asserting the expected tool sequence per user message. Run nightly, not on every PR. _Files:_ `tests/eval/test_golden_tasks.py` (new), `.github/workflows/ci.yml`. **(M)**
25. [x] **Power-planning tool** — `power_analysis.py` exposes `calculate_required_sample_size` but no LangChain tool wraps it. Add `plan_sample_size` so users can ask "how many users do I need to detect a 5% lift at 80% power?". _Files:_ `src/tooling/analysis.py`, `src/agent_tools.py`. **(S)**
26. [x] **Centralized tool-result truncation** — current ad-hoc `head(20)` / `unique[:20]`. Add `truncate_for_llm(payload, max_bytes=8_000)` in `src/tooling/common.py` with a consistent "truncated to N of M" suffix. _Files:_ `src/tooling/common.py`, `src/tooling/*.py`. **(S)**

---

## Bucket 5 — Code quality, tests, deps

27. [x] **Decompose `analyzer.run_ab_test` (~312 lines)** — split into `_run_single_segment`, `_handle_segment_failure`, `_merge_results`. _Files:_ `src/statistics/analyzer.py`. **(M)**
28. [x] **Decompose `pyspark_analyzer.run_ab_test` (~172 lines) + `detect_columns` (~88 lines)** — extract `SparkColumnDetector` and a Spark result serializer. _Files:_ `src/statistics/pyspark_analyzer.py`. **(M)**
29. [x] **Tighten typing on `ABTestingAgent` property pass-throughs** — replace `Any` with `ABAnalyzerProtocol`, `ABTestResult`, `ABTestSummary`. _Files:_ `src/agent.py`, `src/agent_session.py`. **(S)**
30. [x] **Add ruff + mypy config** — no lint or type-check today. Add `[tool.ruff]` (line-length 100, py39 target) and `[tool.mypy]` (`check_untyped_defs = true`); wire both into CI. _Files:_ `pyproject.toml`, `.github/workflows/ci.yml`. **(S)**
31. [x] **Tighten dependency floors** — `langchain>=0.1.0` and `langgraph>=0.0.1` are too loose. Bump to known-good ranges with upper caps. _Files:_ `pyproject.toml`. **(S)**
32. [x] **Direct unit tests for orphaned modules** — `agent_tools.py`, `tooling/common.py`, `chart_builders.py`, `engine_helpers.py`, `experiment_design.py`, `label_inference.py`, `model_families.py`, `power_analysis.py`, `summary_builder.py`. **(M cumulative)**
33. [x] **Replace `print` in the `src/agent.py` CLI loop** with `logger.info`. **(S)**
34. [x] **Log file-size silent-swallow** — `AgentRuntime.get_file_size_mb` swallows `Exception` and returns `0.0`. Add a `logger.warning`. _Files:_ `src/agent_runtime.py`. **(S)**
35. [x] **Remove redundant `pyspark_analyzer` BC property setters** — Wave 3 left tuple-backed properties with setters that just re-pack tuples. Make them read-only and document the tuple contract. _Files:_ `src/statistics/pyspark_analyzer.py`. **(S)**

---

## How to pick a task

- Prefer Bucket 1 unless something in Bucket 3 is actively breaking production.
- Never ship a Spark-touching change without a new case in `tests/test_parity_pandas_spark.py`.
- Bucket 5 items are good "between bigger work" picks — small, isolated, no design surface.
