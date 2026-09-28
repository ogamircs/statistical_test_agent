"""Confinement of user/LLM-supplied data file paths.

The conversational agent lets the LLM choose the ``filepath`` passed to the
loading tools, and ``pd.read_csv`` accepts both arbitrary local paths and
URLs. Every load therefore goes through :func:`resolve_data_path`, which
rejects URL schemes and any path outside an explicitly allowed root.

Allowed roots by default: ``<cwd>/data`` (bundled sample data), ``<cwd>/.uploads``
(web UI upload storage, see ``UPLOADS_DIRNAME``), and the system temp directory.
Additional roots come from ``Config.data_roots`` (``STATAGENT_DATA_ROOTS``,
``os.pathsep``-separated) and are passed in by the caller; this module never
reads the environment, so an injected ``Config`` fully controls confinement.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Union
from urllib.parse import urlsplit

# Directory (relative to the working directory) where the web API stores
# uploaded CSVs, one sub-directory per session.
UPLOADS_DIRNAME = ".uploads"


class DataPathNotAllowedError(ValueError):
    """Raised when a requested data path falls outside the allowed roots.

    Exposes ``code``/``user_message`` so ``classify_agent_error`` renders it
    with a stable error code instead of a generic failure.
    """

    code = "DATA_PATH_NOT_ALLOWED"

    def __init__(self, message: str):
        super().__init__(message)
        self.user_message = message


def default_data_roots(extra_roots: Sequence[Union[str, Path]] = ()) -> List[Path]:
    """Roots a data file may be loaded from, resolved to real paths."""
    cwd = Path.cwd()
    roots = [cwd / "data", cwd / UPLOADS_DIRNAME, Path(tempfile.gettempdir())]
    roots.extend(Path(entry) for entry in extra_roots if str(entry).strip())
    return [root.expanduser().resolve() for root in roots]


def resolve_data_path(
    filepath: str,
    allowed_roots: Optional[Sequence[Path]] = None,
) -> Path:
    """Validate ``filepath`` and return its resolved form.

    Raises:
        DataPathNotAllowedError: for URLs, paths outside every allowed root,
            or paths with hidden components below the matching root.
    """
    candidate = str(filepath).strip()
    if not candidate:
        raise DataPathNotAllowedError("Empty file path.")

    scheme = urlsplit(candidate).scheme
    if len(scheme) > 1:  # single letters pass through for Windows drive paths
        raise DataPathNotAllowedError(
            f"URLs are not allowed for data loading (got scheme '{scheme}'). "
            "Provide a local file path inside an allowed data directory."
        )

    path = Path(candidate).expanduser().resolve()
    if allowed_roots is None:
        roots = default_data_roots()
    else:
        roots = [Path(root).expanduser().resolve() for root in allowed_roots]

    for root in roots:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        # Block dotfiles/dot-directories below the root (e.g. `.env`), while
        # allowed roots themselves may be hidden (e.g. `.uploads`).
        if any(part.startswith(".") for part in relative.parts):
            continue
        return path

    allowed = ", ".join(str(root) for root in roots)
    raise DataPathNotAllowedError(
        f"File path '{candidate}' is outside the allowed data directories ({allowed}). "
        "Upload the file through the chat or place it in an allowed directory."
    )
