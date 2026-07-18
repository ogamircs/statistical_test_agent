"""Shared gate for Spark-dependent tests (TODO.md #62).

Spark tests must skip cleanly on machines without a working Java/Spark
runtime (local dev), but the dedicated Spark CI job sets
``STATAGENT_REQUIRE_SPARK=1`` so the same unavailability becomes a hard
failure — the job can never go green without executing real Spark.
"""

from __future__ import annotations

import os
from typing import NoReturn

import pytest

REQUIRE_SPARK_ENV = "STATAGENT_REQUIRE_SPARK"


def spark_required() -> bool:
    """True when Spark availability is mandatory (the Spark CI job)."""
    return os.environ.get(REQUIRE_SPARK_ENV) == "1"


def skip_or_fail(reason: str) -> NoReturn:
    """Skip with ``reason`` locally; hard-fail when Spark is required."""
    if spark_required():
        pytest.fail(f"{REQUIRE_SPARK_ENV}=1 requires a working Spark runtime. {reason}")
    pytest.skip(reason, allow_module_level=True)
