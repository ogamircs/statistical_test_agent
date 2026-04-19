"""System prompt loader.

Loads the canonical agent system prompt from a sibling Markdown file via
``importlib.resources`` and exposes a stable ``PROMPT_VERSION`` that
should be bumped whenever ``system.md`` is materially edited so logs and
eval harnesses can trace which prompt the agent was running on.
"""

from __future__ import annotations

from importlib import resources

PROMPT_VERSION = "2026-04-19.2"


def load_system_prompt() -> str:
    """Read the system prompt as a string."""
    return resources.files(__package__).joinpath("system.md").read_text(encoding="utf-8")
