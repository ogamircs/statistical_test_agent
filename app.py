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
    _STARTUP_CONFIG = Config()

run_startup_gc(Path("output") / "query_store")

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
