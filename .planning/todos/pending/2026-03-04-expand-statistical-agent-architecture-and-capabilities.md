---
created: 2026-03-04T08:01:55.080Z
title: Expand statistical agent architecture and capabilities
area: general
files:
  - src/agent.py
  - src/statistics/analyzer.py
  - src/statistics/pyspark_analyzer.py
  - src/statistics/models.py
  - tests/test_analyzer_comprehensive.py
  - tests/test_pyspark_analyzer.py
---

## Problem

The project still has major expansion gaps despite the modular analyzer refactor: backend parity between pandas and PySpark is incomplete, agent orchestration remains monolithic with duplicated reporting logic, and statistical guardrails (multiple testing controls, SRM checks, stronger diagnostics) are not fully integrated into the product workflow. This makes long-term feature delivery and reliability harder than necessary.

## Solution

Use the generated backlog as a phased roadmap and execute high-priority tasks first:
1. Normalize a canonical result schema across pandas and Spark paths.
2. Refactor `src/agent.py` into tool registration and report rendering modules.
3. Add backend parity tests and CI enforcement.
4. Add statistical robustness features (multiple testing correction, SRM, assumption diagnostics) and new model families through statsmodels-based components.
5. Expand capabilities for sequential analysis and production observability.
