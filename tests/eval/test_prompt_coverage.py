"""Static (no-LLM) eval: every golden task's expected tool must be
mentioned in the system prompt so the agent has a chance to route to it.
"""

from __future__ import annotations

import pytest

from src.prompts import load_system_prompt
from tests.eval.golden_tasks import GOLDEN_TASKS

PROMPT = load_system_prompt()
KNOWN_AGENT_TOOLS = {
    "load_and_auto_analyze",
    "load_csv",
    "configure_and_analyze",
    "answer_data_question",
    "plan_sample_size",
    "auto_configure_and_analyze",
    "generate_charts",
    "show_distribution_chart",
}


@pytest.mark.parametrize(
    "task",
    GOLDEN_TASKS,
    ids=[task.name for task in GOLDEN_TASKS],
)
def test_expected_tools_documented_in_prompt(task) -> None:
    for tool in task.expected_tools:
        if tool not in KNOWN_AGENT_TOOLS:
            pytest.skip(
                f"Tool '{tool}' is not yet in the registered tool set — "
                "skip this row until the tool ships."
            )
        assert tool in PROMPT, (
            f"system prompt does not mention `{tool}`; LLM cannot route "
            f"task '{task.name}' without it"
        )


def test_every_golden_task_message_is_unique() -> None:
    seen = set()
    for task in GOLDEN_TASKS:
        assert task.user_message not in seen, f"duplicate utterance: {task.name}"
        seen.add(task.user_message)
