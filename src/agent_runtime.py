"""Backend selection and data-loading runtime for the conversational agent."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

from src.statistics.analyzer_protocol import ABAnalyzerProtocol


logger = logging.getLogger(__name__)


class AgentRuntime:
    """Own the active backend selection and CSV loading fallback behavior."""

    def __init__(
        self,
        *,
        analyzer: ABAnalyzerProtocol,
        spark_factory: Optional[Callable[[], ABAnalyzerProtocol]],
        spark_available: Callable[[], bool],
        file_size_threshold_mb: float = 2.0,
    ) -> None:
        self.analyzer: ABAnalyzerProtocol = analyzer
        self.spark_analyzer: Optional[ABAnalyzerProtocol] = None
        self.using_spark: bool = False
        self.spark_factory = spark_factory
        self.spark_available = spark_available
        self.file_size_threshold_mb = file_size_threshold_mb

    def get_file_size_mb(self, filepath: str) -> float:
        """Get file size in megabytes; warn on OS errors and return 0.0."""
        try:
            return os.path.getsize(filepath) / (1024 * 1024)
        except OSError as exc:
            logger.warning("get_file_size_mb failed for %s: %s", filepath, exc)
            return 0.0

    def should_use_spark(self, filepath: str) -> bool:
        """Determine if Spark should be used based on file size and availability."""
        if not self.spark_available():
            return False
        file_size_mb = self.get_file_size_mb(filepath)
        should_use = file_size_mb > self.file_size_threshold_mb
        logger.info(
            "Backend selection evaluated (file=%s, size_mb=%.2f, threshold_mb=%.2f, use_spark=%s)",
            filepath,
            file_size_mb,
            self.file_size_threshold_mb,
            should_use,
        )
        return should_use

    def get_active_analyzer(self) -> ABAnalyzerProtocol:
        if self.using_spark and self.spark_analyzer is not None:
            return self.spark_analyzer
        return self.analyzer

    def init_spark_analyzer(self) -> ABAnalyzerProtocol:
        if self.spark_analyzer is not None:
            return self.spark_analyzer
        if not self.spark_available() or self.spark_factory is None:
            raise RuntimeError("PySpark is not available in this environment")
        logger.info("Initializing PySpark analyzer")
        self.spark_analyzer = self.spark_factory()
        return self.spark_analyzer

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

    def load_data_with_backend(self, filepath: str):
        """
        Load data using Spark when appropriate, with automatic pandas fallback.

        Returns:
            (analyzer, info, backend_name, file_size_mb, spark_selected, fallback_note)
        """
        file_size_mb = self.get_file_size_mb(filepath)
        spark_selected = self.should_use_spark(filepath)
        fallback_note = None
        logger.info(
            "Starting data load (file=%s, size_mb=%.2f, spark_selected=%s)",
            filepath,
            file_size_mb,
            spark_selected,
        )

        if spark_selected:
            try:
                analyzer = self.init_spark_analyzer()
            except Exception as error:
                self.using_spark = False
                fallback_note = f"PySpark initialization failed: {error}. Falling back to pandas."
                logger.warning("PySpark initialization failed; using pandas fallback", exc_info=error)
            else:
                try:
                    info = analyzer.load_data(filepath, format="csv")
                    self.using_spark = True
                    logger.info("Data load completed with spark backend")
                    return analyzer, info, "spark", file_size_mb, spark_selected, fallback_note
                except Exception as error:
                    self.using_spark = False
                    fallback_note = f"PySpark backend failed while loading data: {error}. Falling back to pandas."
                    logger.warning("PySpark load failed; using pandas fallback", exc_info=error)

        analyzer = self.analyzer
        info = analyzer.load_data(filepath)
        self.using_spark = False
        logger.info("Data load completed with pandas backend")
        return analyzer, info, "pandas", file_size_mb, spark_selected, fallback_note
