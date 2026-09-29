"""Bearer tokens for the optional password auth (see ``src.auth``).

A token is ``<base64url(username:expiry)>.<hex hmac-sha256>`` signed with
``Config.auth_secret`` (``STATAGENT_AUTH_SECRET``). When no secret is
configured each ``TokenSigner`` generates a random key, so tokens stop
working after a restart and are not shared across replicas.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from typing import Optional

DEFAULT_TOKEN_TTL_SECONDS = 12 * 60 * 60


class TokenSigner:
    """Issues and verifies tokens with one signing key."""

    def __init__(self, secret: Optional[str] = None) -> None:
        self._key = secret.encode("utf-8") if secret else secrets.token_bytes(32)

    def _sign(self, payload: bytes) -> str:
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def issue(self, username: str, ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS) -> str:
        expiry = int(time.time()) + ttl_seconds
        payload = f"{username}:{expiry}".encode("utf-8")
        encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        return f"{encoded}.{self._sign(payload)}"

    def verify(self, token: str) -> Optional[str]:
        """Return the username for a valid, unexpired token, otherwise None."""
        try:
            encoded, signature = token.split(".", 1)
            padded = encoded + "=" * (-len(encoded) % 4)
            payload = base64.urlsafe_b64decode(padded.encode("ascii"))
        except (ValueError, UnicodeEncodeError):
            return None
        if not hmac.compare_digest(signature, self._sign(payload)):
            return None
        try:
            username, expiry_text = payload.decode("utf-8").rsplit(":", 1)
            expiry = int(expiry_text)
        except (UnicodeDecodeError, ValueError):
            return None
        if expiry < time.time():
            return None
        return username
