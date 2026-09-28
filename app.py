"""Web entrypoint for the A/B Testing Agent.

Serves the JSON/SSE API used by the React UI in ``frontend/`` and, when
``frontend/dist`` has been built, the UI itself.

Run with ``uvicorn app:app`` (or ``python app.py``).
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.api import create_app
from src.auth import is_auth_enabled
from src.config import Config
from src.observability import configure_json_logging
from src.query_store_gc import run_startup_gc

load_dotenv()
# uvicorn only configures its own loggers; without a root handler every
# src.* log (tool timings, token usage, failures) was silently dropped.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
configure_json_logging()
logger = logging.getLogger(__name__)

_STARTUP_CONFIG = Config.from_env()
try:
    _STARTUP_CONFIG.validate()
    logger.info(
        "Startup config: model=%s temperature=%s sql_row_limit=%d query_timeout_s=%.1f "
        "max_upload_mb=%.0f",
        _STARTUP_CONFIG.llm_model,
        _STARTUP_CONFIG.llm_temperature,
        _STARTUP_CONFIG.sql_default_row_limit,
        _STARTUP_CONFIG.query_timeout_seconds,
        _STARTUP_CONFIG.max_upload_mb,
    )
except ValueError:
    logger.exception("Startup Config validation failed; continuing with defaults")
    # Keep the security-relevant knob: falling back to defaults must not
    # silently turn a require-auth deployment into an open one.
    _STARTUP_CONFIG = Config(require_auth=_STARTUP_CONFIG.require_auth)

run_startup_gc(Path(_STARTUP_CONFIG.query_store_dir))

if is_auth_enabled():
    logger.info("Password auth ENABLED via STATAGENT_AUTH_* env vars")
else:
    logger.info(
        "Password auth disabled (set STATAGENT_AUTH_USERNAME and STATAGENT_AUTH_PASSWORD to enable)"
    )

app = create_app(config=_STARTUP_CONFIG)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )
