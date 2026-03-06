# Statistical Test Agent Improvement Backlog

Last updated: 2026-03-06

This backlog was derived from the original codebase review and is now recorded as completed work plus the small amount of environment follow-up that remains outside the codebase itself.

## Completed

### P1

- [x] Unify the pandas/Spark analyzer contract behind one active backend.
  Outcome: agent tools now route through the active backend, and unsupported Spark-only gaps are explicitly blocked instead of silently falling back to pandas-only behavior.

- [x] Fix auto-configuration label detection so treatment/control guesses are not based on fragile substring matches.
  Outcome: label inference now uses shared confidence-based heuristics rather than brittle substring matches like `"a"` and `"b"`.

- [x] Stop silently skipping failed segments during segmented analysis.
  Outcome: segment failures are now surfaced in structured summaries and reports instead of being printed and dropped.

- [x] Add end-to-end tests for the agent/tool/backend flow.
  Outcome: the agent/tool/reporting path now has direct regression coverage, including backend-aware tool behavior.

### P2

- [x] Decompose `run_ab_test` into smaller analysis stages.
  Outcome: the orchestration flow in `src/statistics/analyzer.py` is staged through smaller helpers instead of one oversized method.

- [x] Break `StatsmodelsABTestEngine` into model-family and diagnostics modules.
  Outcome: inference, diagnostics, Bayesian logic, and power analysis now live in narrower modules with the engine acting as a facade.

- [x] Replace ad hoc summary dictionaries with typed report models.
  Outcome: typed summary/report structures now back the reporting layer while preserving compatibility where needed.

- [x] Refactor chart generation into reusable chart builders with stronger tests.
  Outcome: chart construction is split into reusable builders and visualization tests assert more semantics than simple figure existence.

- [x] Add structured logging instead of `print`-based diagnostics.
  Outcome: runtime paths now use logging for backend/tool/segment context, with user-facing errors staying stable.

### P3

- [x] Modernize packaging and dependency management.
  Outcome: `pyproject.toml` is now canonical, `requirements.txt` is a compatibility shim, optional Spark support is modeled as an extra, and `uv.lock` is generated for reproducible installs.

- [x] Tighten repository hygiene around generated artifacts.
  Outcome: generated `output/` artifacts are ignored and stale test-result/docs content was refreshed.

- [x] Add explicit backend capability documentation.
  Outcome: `README.md` and `LARGE_FILE_SUPPORT.md` now document parity targets, unsupported Spark helpers, and fallback behavior explicitly.

- [x] Improve CI signal for optional Spark support.
  Outcome: GitHub Actions now includes a dedicated Spark job that installs the Spark extra and runs Spark-specific test suites.

## Operational Follow-Up

- [ ] Verify the Spark test path locally on a machine with both Java and `pyspark` installed.
  Why: the code and CI path are in place, but this local workstation currently does not have a Java runtime, so Spark tests cannot run here.
  Validation target: `pytest -q tests/test_pyspark_analyzer.py tests/test_parity_pandas_spark.py -ra`
