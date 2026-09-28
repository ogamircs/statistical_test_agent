# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, etc.) working in this repository. Human contributors should start with [README.md](README.md) and [docs/development.md](docs/development.md).

## What this project is

A conversational A/B-test analysis agent. Users upload a CSV in a Chainlit chat UI; a LangGraph ReAct agent (OpenAI model, default `gpt-5.2`) calls Python tools that run frequentist, Bayesian, CUPED/covariate-adjusted, segmented, sequential, and ratio-metric analyses, then renders markdown reports and Plotly charts.

All analysis is pure Python: CSVs are loaded into pandas and analyzed in memory by `src/statistics/analyzer.py` (statsmodels, scipy, numpy). There is no distributed backend.

**Statistical correctness comes first.** A polished answer that misreports a significance call is worse than a rough one that gets it right.

## Commands

Always use the project venv via `uv run` (or `./.venv/bin/...`).

```bash
uv sync --extra dev                                       # install
uv run chainlit run app.py --host 127.0.0.1 --port 8010   # run the UI
uv run python scripts/generate_sample_data.py             # regenerate data/sample_ab_data*.csv

# The CI gates. Run all of them before calling a change done:
uv run ruff check .
uv run mypy src app.py
uv run pytest -q -ra --cov=src --cov-fail-under=78
```

Live LLM evals (`tests/eval/test_live_tools.py`) run only with `STATAGENT_RUN_LIVE_EVAL=1` and a real `OPENAI_API_KEY`. Every other test must stay offline. Mock the LLM.

## Code map

```text
app.py                      Chainlit entry: upload handling, auth hook, chat start/resume
src/agent.py                ABTestingAgent: LLM construction, LangGraph agent, run loop, history
src/agent_runtime.py        CSV loading (path confinement + pandas analyzer)
src/agent_session.py        Per-session state (loaded data, results, chat history, query store)
src/agent_tools.py          Tool registry facade -> src/tooling/{loading,analysis,visualization}.py
src/tooling/common.py       ToolContext / AgentProtocol shared by all tools
src/agent_reporting.py      Markdown report + structured error rendering (what the LLM and user see)
src/prompts/system.md       System prompt (versioned via PROMPT_VERSION in src/prompts/__init__.py)
src/config.py               Config dataclass, all knobs loaded from STATAGENT_* env vars
src/query_store*.py         Per-session SQLite (raw data, chat history, audit), read-only query conn, GC
src/sql_query_service.py    NL -> SQL planner for answer_data_question
src/data_paths.py           CSV path confinement (allowed roots only, no URLs)
src/auth.py                 Optional password auth (STATAGENT_AUTH_USERNAME/PASSWORD)
src/statistics/
  analyzer.py               pandas facade: load, auto-configure, run_ab_test, segmented/full analysis
  analyzer_protocol.py      The analyzer interface the agent and tools depend on
  models.py                 ABTestResult and the canonical typed result schema
  statsmodels_engine.py     Inference facade (t-test, proportion test, SRM, sequential)
  model_families.py         GLM family selection (Gaussian/Binomial/Poisson/heavy-tail) + DiD
  diagnostics.py            SRM, duplicate units, normality/variance checks, guardrails
  power_analysis.py, bayesian.py, ratio_metric.py, experiment_design.py, sequential_config.py
  summary_builder.py        Typed summary + recommendations
  visualizer.py, charts_*.py, chart_builders.py, chart_catalog.py   Plotly charts
tests/                      pytest suite; tests/eval/ holds golden routing tasks + gated live evals
docs/TODO.md                The prioritized backlog. Read it before starting non-trivial work.
```

## Rules and invariants

1. **Pure Python.** Keep the stack pandas/statsmodels/scipy/numpy. Do not add JVM or distributed backends (PySpark was removed deliberately). Put reusable statistical math in small pure functions (e.g. `engine_helpers.py`, `statsmodels_engine.py`) with direct unit tests.
2. **Never degrade silently.** When a statistical fallback fires (model-fit fallback, test exception, power/MDE sentinel), append to `ABTestResult.statistical_warnings` so it reaches the report. Do not return plausible-looking defaults.
3. **Report on consistent scales.** Effect sizes, CIs, and chart error bars must use the same scale (see TODO #36).
4. **Tool contract.** Tools are built in `src/tooling/` and receive a `ToolContext`. Prefer `StructuredTool` with typed args over hand-parsed JSON or comma-split strings. The schema is what the LLM sees. Tool errors go through `render_tool_error` with a stable error code.
5. **Prompt changes.** When you materially edit `src/prompts/system.md`, bump `PROMPT_VERSION` and update the tests that pin it. Keep golden tasks in `tests/eval/golden_tasks.py` in sync with tool names.
6. **Untrusted data.** CSV column names and values are data, not instructions. Never interpolate them into prompts unless they are delimited.
7. **Security guardrails.** CSV loading stays confined to allowed roots (`src/data_paths.py`). SQL runs on a read-only SQLite connection (`mode=ro`). Do not weaken either.
8. **Configuration.** New knobs go on `Config` in `src/config.py` with a `STATAGENT_*` env var, validation in `Config.validate()`, and a row in `docs/deployment.md`. Do not scatter module-level constants.
9. **Types.** mypy is blocking. The `[[tool.mypy.overrides]]` list in `pyproject.toml` is a burn-down list. Remove modules from it as you fix them, and never add to it.
10. **Markdown files are gitignored by default** (`*.md` in `.gitignore`). To track a new doc, add an explicit `!path` allowlist entry.

## Workflow conventions

- Reference backlog items in commits and PRs as `TODO.md #N` (e.g. `fix(stats): ... (TODO.md #42)`). Numbering is global and never reused. Mark items `[x]` in `docs/TODO.md` when shipped.
- Commit style: conventional prefixes (`fix:`, `feat:`, `ci:`, `docs:`, `refactor:`), with an optional scope.
- Add a focused regression test with every bug fix. Put tests in the feature-named suite, not in PR-history files like `test_pr_review_fixes.py` (TODO #67).
- `pyproject.toml` + `uv.lock` are the dependency source of truth. `requirements.txt` is a compatibility shim only.
- Do not commit screenshots, generated `output/`, `data/sample_ab_data_large.csv`, or `.env`.

## Gotchas

- The installed package is literally named `src` (TODO #66). Import as `from src.statistics import ...`.
- `load_dotenv()` runs in both `app.py` and `src/agent.py`. Tests that depend on env vars should pass an explicit `Config` or mapping instead of mutating `os.environ` globally.
- Chainlit uploads land in `.files/`. Session SQLite stores live under the query-store directory and are garbage-collected by `src/query_store_gc.py`.
- The browser layer (`public/custom.js`, `public/custom.css`) is intentionally thin. Keep analysis logic in Python.
