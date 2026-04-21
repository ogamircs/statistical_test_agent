"""
Statistics Module

Provides statistical analysis and visualization for A/B testing:
- models: Data structures for test results
- analyzer: High-level analysis facade
- data_manager: Data loading and schema inference
- statsmodels_engine: Inferential computation engine
- summary_builder: Summary and recommendation builder
- visualizer: Chart and dashboard generation
"""

from .analyzer import ABTestAnalyzer
from .data_manager import ABTestDataManager
from .models import ABTestResult
from .statsmodels_engine import StatsmodelsABTestEngine
from .summary_builder import ABTestSummaryBuilder
from .visualizer import ABTestVisualizer

__all__ = [
    "ABTestResult",
    "ABTestAnalyzer",
    "ABTestDataManager",
    "StatsmodelsABTestEngine",
    "ABTestSummaryBuilder",
    "ABTestVisualizer",
]
