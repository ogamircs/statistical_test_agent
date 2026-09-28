"""Centralized runtime configuration for the conversational agent.

All knobs that previously lived as scattered constants — LLM model name,
temperature, default SQL row limit — collapse
into one ``Config`` dataclass loadable from environment variables. The
defaults preserve current behavior so existing callers can adopt
``Config()`` lazily without behavior change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_DEFAULT_LLM_MODEL = "gpt-5.2"
_DEFAULT_LLM_TEMPERATURE = 0.0
_DEFAULT_SQL_ROW_LIMIT = 20
_DEFAULT_QUERY_TIMEOUT_SECONDS = 5.0
_DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 120.0
_DEFAULT_LLM_MAX_RETRIES = 2
# LangGraph's own default; one ReAct step (model call or tool call) per unit.
_DEFAULT_AGENT_RECURSION_LIMIT = 25
# Prior chat messages (human + AI) re-sent to the model each turn. The full
# history stays persisted; only the model-bound window is bounded.
_DEFAULT_MAX_HISTORY_MESSAGES = 40
# Largest CSV accepted by the web upload endpoint.
_DEFAULT_MAX_UPLOAD_MB = 50.0
# Per-session SQLite stores (session-<id>.sqlite); the UI lists them as history.
_DEFAULT_QUERY_STORE_DIR = "output/query_store"
# HMAC key for UI bearer tokens; shorter keys are rejected by validate().
MIN_AUTH_SECRET_LENGTH = 16


@dataclass(frozen=True)
class Config:
    """Resolved runtime configuration."""

    llm_model: str = _DEFAULT_LLM_MODEL
    llm_temperature: float = _DEFAULT_LLM_TEMPERATURE
    sql_default_row_limit: int = _DEFAULT_SQL_ROW_LIMIT
    query_timeout_seconds: float = _DEFAULT_QUERY_TIMEOUT_SECONDS
    llm_request_timeout_seconds: float = _DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
    llm_max_retries: int = _DEFAULT_LLM_MAX_RETRIES
    agent_recursion_limit: int = _DEFAULT_AGENT_RECURSION_LIMIT
    max_history_messages: int = _DEFAULT_MAX_HISTORY_MESSAGES
    max_upload_mb: float = _DEFAULT_MAX_UPLOAD_MB
    query_store_dir: str = _DEFAULT_QUERY_STORE_DIR
    # Refuse to start unless password auth is configured (TODO.md #49).
    require_auth: bool = False
    # Optional password auth for the web UI; enabled only when both are set.
    auth_username: str | None = field(default=None, repr=False)
    auth_password: str | None = field(default=None, repr=False)
    # Token signing key; None means a random per-app key (tokens die on restart).
    auth_secret: str | None = field(default=None, repr=False)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_username) and bool(self.auth_password)

    @classmethod
    def from_env(cls, environ: dict | None = None) -> "Config":
        """Build a Config from environment variables (or a provided mapping)."""
        env = environ if environ is not None else os.environ

        return cls(
            llm_model=env.get("STATAGENT_LLM_MODEL", _DEFAULT_LLM_MODEL),
            llm_temperature=_coerce_float(
                env.get("STATAGENT_LLM_TEMPERATURE"), _DEFAULT_LLM_TEMPERATURE
            ),
            sql_default_row_limit=_coerce_int(
                env.get("STATAGENT_SQL_ROW_LIMIT"), _DEFAULT_SQL_ROW_LIMIT
            ),
            query_timeout_seconds=_coerce_float(
                env.get("STATAGENT_QUERY_TIMEOUT_SECONDS"),
                _DEFAULT_QUERY_TIMEOUT_SECONDS,
            ),
            llm_request_timeout_seconds=_coerce_float(
                env.get("STATAGENT_LLM_TIMEOUT_SECONDS"),
                _DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
            ),
            llm_max_retries=_coerce_int(
                env.get("STATAGENT_LLM_MAX_RETRIES"), _DEFAULT_LLM_MAX_RETRIES
            ),
            agent_recursion_limit=_coerce_int(
                env.get("STATAGENT_AGENT_RECURSION_LIMIT"),
                _DEFAULT_AGENT_RECURSION_LIMIT,
            ),
            max_history_messages=_coerce_int(
                env.get("STATAGENT_MAX_HISTORY_MESSAGES"),
                _DEFAULT_MAX_HISTORY_MESSAGES,
            ),
            max_upload_mb=_coerce_float(
                env.get("STATAGENT_MAX_UPLOAD_MB"), _DEFAULT_MAX_UPLOAD_MB
            ),
            query_store_dir=env.get("STATAGENT_QUERY_STORE_DIR") or _DEFAULT_QUERY_STORE_DIR,
            require_auth=_parse_bool(env.get("STATAGENT_REQUIRE_AUTH"), "STATAGENT_REQUIRE_AUTH"),
            auth_username=env.get("STATAGENT_AUTH_USERNAME") or None,
            auth_password=env.get("STATAGENT_AUTH_PASSWORD") or None,
            auth_secret=env.get("STATAGENT_AUTH_SECRET") or None,
        )

    def validate_security(self) -> None:
        """Raise ValueError for invalid security settings.

        Separate from ``validate`` because callers that fall back to defaults
        on a bad numeric knob must never fall back on these: an invalid
        security setting has to stop the app, not open it.
        """
        if self.require_auth and not self.auth_enabled:
            raise ValueError(
                "STATAGENT_REQUIRE_AUTH is set but password auth is not configured: set both "
                "STATAGENT_AUTH_USERNAME and STATAGENT_AUTH_PASSWORD (and STATAGENT_AUTH_SECRET "
                "so tokens survive restarts and work across replicas)."
            )
        if self.auth_secret is not None and len(self.auth_secret) < MIN_AUTH_SECRET_LENGTH:
            raise ValueError(
                f"STATAGENT_AUTH_SECRET must be at least {MIN_AUTH_SECRET_LENGTH} characters"
            )

    def validate(self) -> None:
        """Raise ValueError when a knob is out of range (security checks included)."""
        self.validate_security()
        if not self.llm_model:
            raise ValueError("Config.llm_model must be a non-empty string")
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            raise ValueError("Config.llm_temperature must be in [0, 2]")
        if self.sql_default_row_limit <= 0:
            raise ValueError("Config.sql_default_row_limit must be > 0")
        if self.query_timeout_seconds <= 0:
            raise ValueError("Config.query_timeout_seconds must be > 0")
        if self.llm_request_timeout_seconds <= 0:
            raise ValueError("Config.llm_request_timeout_seconds must be > 0")
        if self.llm_max_retries < 0:
            raise ValueError("Config.llm_max_retries must be >= 0")
        if self.agent_recursion_limit < 2:
            raise ValueError("Config.agent_recursion_limit must be >= 2")
        if self.max_history_messages < 1:
            raise ValueError("Config.max_history_messages must be >= 1")
        if self.max_upload_mb <= 0:
            raise ValueError("Config.max_upload_mb must be > 0")
        if not self.query_store_dir.strip():
            raise ValueError("Config.query_store_dir must be a non-empty path")


def _coerce_float(value: object, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        if isinstance(value, (int, float)):
            return int(value)
        return int(str(value))
    except (TypeError, ValueError):
        return default


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def _parse_bool(value: object, name: str) -> bool:
    """Strict boolean parsing for safety knobs: unknown values raise.

    A typo such as ``STATAGENT_REQUIRE_AUTH=tru`` must stop the app rather
    than silently disable the guard it was meant to enable.
    """
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise ValueError(
        f"{name}={value!r} is not a boolean; use one of 1/true/yes/on or 0/false/no/off"
    )
