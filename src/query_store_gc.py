"""Garbage-collect stale per-session SQLite query stores.

Each chat session produces an `output/query_store/session-<uuid>.sqlite` file.
Without cleanup these accumulate without bound. This module sweeps stale files
on app/session startup.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List


logger = logging.getLogger(__name__)

_DEFAULT_MAX_AGE_DAYS = 7
_DEFAULT_SIZE_WARN_MB = 50.0

_GC_RAN: set[str] = set()


def cleanup_query_stores(
    directory: Path | str,
    max_age_days: int = _DEFAULT_MAX_AGE_DAYS,
    size_warn_mb: float = _DEFAULT_SIZE_WARN_MB,
) -> Dict[str, List[str]]:
    """Delete `*.sqlite` files older than max_age_days; flag oversized ones.

    Returns a dict with two keys:
      - "deleted": absolute paths of removed files
      - "oversized": absolute paths of files exceeding size_warn_mb (kept)
    """
    deleted: List[str] = []
    oversized: List[str] = []
    target = Path(directory)
    if not target.exists() or not target.is_dir():
        return {"deleted": deleted, "oversized": oversized}

    cutoff = time.time() - max_age_days * 86400
    size_threshold = size_warn_mb * 1024 * 1024

    for entry in target.iterdir():
        if not entry.is_file() or entry.suffix != ".sqlite":
            continue
        try:
            stat = entry.stat()
        except OSError as exc:
            logger.warning("cleanup_query_stores: stat failed for %s: %s", entry, exc)
            continue

        if stat.st_size > size_threshold:
            logger.warning(
                "Query store %s exceeds %.1f MB (%.1f MB); investigate.",
                entry,
                size_warn_mb,
                stat.st_size / (1024 * 1024),
            )
            oversized.append(str(entry))
            continue

        if stat.st_mtime < cutoff:
            try:
                entry.unlink()
            except OSError as exc:
                logger.warning("cleanup_query_stores: failed to delete %s: %s", entry, exc)
                continue
            logger.info("cleanup_query_stores: deleted stale %s", entry)
            deleted.append(str(entry))

    return {"deleted": deleted, "oversized": oversized}


def run_startup_gc(
    directory: Path | str,
    max_age_days: int = _DEFAULT_MAX_AGE_DAYS,
    size_warn_mb: float = _DEFAULT_SIZE_WARN_MB,
) -> Dict[str, List[str]]:
    """Once-per-process wrapper. Safe — never raises."""
    key = str(Path(directory).resolve())
    if key in _GC_RAN:
        return {"deleted": [], "oversized": []}
    _GC_RAN.add(key)
    try:
        return cleanup_query_stores(directory, max_age_days, size_warn_mb)
    except Exception:  # defensive: GC must never break startup
        logger.exception("cleanup_query_stores failed unexpectedly for %s", directory)
        return {"deleted": [], "oversized": []}


def reset_gc_state() -> None:
    """Reset the once-per-process guard. Test helper."""
    _GC_RAN.clear()
