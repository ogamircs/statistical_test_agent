"""Bearer tokens for the optional password auth (see ``src.auth``).

A token is ``<base64url(username:expiry)>.<hex hmac-sha256>`` signed with
``STATAGENT_AUTH_SECRET``. When that variable is unset a random secret is
generated per process, so tokens stop working after a restart.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Optional

_SECRET_ENV = "STATAGENT_AUTH_SECRET"
DEFAULT_TOKEN_TTL_SECONDS = 12 * 60 * 60

_process_secret = secrets.token_bytes(32)


def _secret() -> bytes:
    configured = os.environ.get(_SECRET_ENV)
    return configured.encode("utf-8") if configured else _process_secret


def _sign(payload: bytes) -> str:
    return hmac.new(_secret(), payload, hashlib.sha256).hexdigest()


def issue_token(username: str, ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS) -> str:
    expiry = int(time.time()) + ttl_seconds
    payload = f"{username}:{expiry}".encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"{encoded}.{_sign(payload)}"


def verify_token(token: str) -> Optional[str]:
    """Return the username for a valid, unexpired token, otherwise None."""
    try:
        encoded, signature = token.split(".", 1)
        padded = encoded + "=" * (-len(encoded) % 4)
        payload = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        return None
    if not hmac.compare_digest(signature, _sign(payload)):
        return None
    try:
        username, expiry_text = payload.decode("utf-8").rsplit(":", 1)
        expiry = int(expiry_text)
    except (UnicodeDecodeError, ValueError):
        return None
    if expiry < time.time():
        return None
    return username
