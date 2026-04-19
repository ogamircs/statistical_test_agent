"""Golden-task fixtures for the agent's tool-routing eval harness.

Each GoldenTask is a single user utterance plus the tool the agent
SHOULD call. The static prompt-coverage check uses these to assert
that the system prompt mentions every expected tool. The optional
live runner (skipped without OPENAI_API_KEY) actually invokes the
LLM and asserts the tool sequence.

Bumping the prompt? Add a task here for each new capability or
behavior change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GoldenTask:
    name: str
    user_message: str
    expected_tools: List[str]
    keywords: List[str]


GOLDEN_TASKS: List[GoldenTask] = [
    GoldenTask(
        name="auto_analyze_via_keyword",
        user_message="Load /tmp/ab.csv and best guess auto analyze it.",
        expected_tools=["load_and_auto_analyze"],
        keywords=["best guess", "auto"],
    ),
    GoldenTask(
        name="manual_load_then_configure",
        user_message="Here's my CSV /tmp/ab.csv — show me the columns first before analyzing.",
        expected_tools=["load_csv", "configure_and_analyze"],
        keywords=["show me the columns"],
    ),
    GoldenTask(
        name="plan_sample_size_continuous",
        user_message=(
            "Before I run an experiment: how many users per arm to detect a "
            "0.5 mean lift on a baseline mean of 10 with std 2 at 80% power?"
        ),
        expected_tools=["plan_sample_size"],
        keywords=["how many users", "power"],
    ),
    GoldenTask(
        name="plan_sample_size_proportion",
        user_message=(
            "What sample size do I need to detect a 2 percentage point lift "
            "from a 10 percent baseline conversion at 80 percent power?"
        ),
        expected_tools=["plan_sample_size"],
        keywords=["sample size", "baseline"],
    ),
    GoldenTask(
        name="answer_data_question_after_load",
        user_message=(
            "How many customers are in each segment after loading the file?"
        ),
        expected_tools=["answer_data_question"],
        keywords=["how many customers"],
    ),
    GoldenTask(
        name="dashboard_request",
        user_message="Show me the full dashboard for the analysis.",
        expected_tools=["generate_charts"],
        keywords=["dashboard"],
    ),
]
