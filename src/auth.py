"""Optional password-based auth for the Chainlit UI.

Auth activates only when both ``STATAGENT_AUTH_USERNAME`` and
``STATAGENT_AUTH_PASSWORD`` env vars are set. With neither set, the UI
runs unauthenticated (the existing dev-time default).
"""

from __future__ import annotations

import hmac
import os
from typing import Optional

_USERNAME_ENV = "STATAGENT_AUTH_USERNAME"
_PASSWORD_ENV = "STATAGENT_AUTH_PASSWORD"


def is_auth_enabled() -> bool:
    """True when both username and password env vars are configured."""
    return bool(os.environ.get(_USERNAME_ENV)) and bool(os.environ.get(_PASSWORD_ENV))


def verify_credentials(username: str, password: str) -> Optional[str]:
    """Return the username on a successful match, otherwise None.

    Uses ``hmac.compare_digest`` so timing leaks between matched and
    mismatched values are avoided. When auth is not enabled this
    function always returns None so callers can treat "auth off" as
    "no logged-in user".
    """
    expected_username = os.environ.get(_USERNAME_ENV)
    expected_password = os.environ.get(_PASSWORD_ENV)
    if not expected_username or not expected_password:
        return None

    if not isinstance(username, str) or not isinstance(password, str):
        return None

    username_match = hmac.compare_digest(username.encode("utf-8"), expected_username.encode("utf-8"))
    password_match = hmac.compare_digest(password.encode("utf-8"), expected_password.encode("utf-8"))
    if username_match and password_match:
        return expected_username
    return None
