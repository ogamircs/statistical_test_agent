"""Optional password-based auth for the web UI.

Auth activates only when ``Config.auth_username`` and ``Config.auth_password``
(``STATAGENT_AUTH_USERNAME`` / ``STATAGENT_AUTH_PASSWORD``) are both set. With
neither set, the UI runs unauthenticated (the dev-time default). Callers pass
the resolved ``Config`` so an injected configuration fully controls auth.
"""

from __future__ import annotations

import hmac
from typing import Optional

from src.config import Config


def is_auth_enabled(config: Config) -> bool:
    """True when both a username and a password are configured."""
    return config.auth_enabled


def verify_credentials(config: Config, username: str, password: str) -> Optional[str]:
    """Return the username on a successful match, otherwise None.

    Uses ``hmac.compare_digest`` so timing leaks between matched and
    mismatched values are avoided. When auth is not enabled this function
    always returns None so callers can treat "auth off" as "no logged-in user".
    """
    expected_username = config.auth_username
    expected_password = config.auth_password
    if not expected_username or not expected_password:
        return None

    if not isinstance(username, str) or not isinstance(password, str):
        return None

    username_match = hmac.compare_digest(username.encode("utf-8"), expected_username.encode("utf-8"))
    password_match = hmac.compare_digest(password.encode("utf-8"), expected_password.encode("utf-8"))
    if username_match and password_match:
        return expected_username
    return None
