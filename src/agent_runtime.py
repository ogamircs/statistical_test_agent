"""Data-loading runtime for the conversational agent."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

from src.data_paths import resolve_data_path
from src.statistics.analyzer_protocol import ABAnalyzerProtocol

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Own the active analyzer and CSV loading."""

    def __init__(self, *, analyzer: ABAnalyzerProtocol) -> None:
        self.analyzer: ABAnalyzerProtocol = analyzer

    def get_file_size_mb(self, filepath: str) -> float:
        """Get file size in megabytes; warn on OS errors and return 0.0."""
        try:
            return os.path.getsize(filepath) / (1024 * 1024)
        except OSError as exc:
            logger.warning("get_file_size_mb failed for %s: %s", filepath, exc)
            return 0.0

    def get_active_analyzer(self) -> ABAnalyzerProtocol:
        return self.analyzer

    @staticmethod
    def normalize_shape(info: Dict[str, Any]) -> Tuple[int, int]:
        """Normalize load_data metadata to (rows, columns)."""
        shape = info.get("shape")
        if isinstance(shape, (tuple, list)) and len(shape) >= 2:
            return int(shape[0]), int(shape[1])

        row_count = info.get("row_count")
        columns = info.get("columns")
        if row_count is not None and columns is not None:
            return int(row_count), len(columns)

        raise KeyError("shape")

    def load_data(self, filepath: str) -> Tuple[ABAnalyzerProtocol, Dict[str, Any], float]:
        """
        Load a CSV into the pandas analyzer.

        Returns:
            (analyzer, info, file_size_mb)

        Raises:
            DataPathNotAllowedError: when the path is a URL or falls outside
                the allowed data roots (see src.data_paths).
        """
        filepath = str(resolve_data_path(filepath))
        file_size_mb = self.get_file_size_mb(filepath)
        logger.info("Starting data load (file=%s, size_mb=%.2f)", filepath, file_size_mb)
        info = self.analyzer.load_data(filepath)
        logger.info("Data load completed")
        return self.analyzer, info, file_size_mb
