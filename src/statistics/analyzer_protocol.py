"""Protocol defining the shared public interface for A/B test analyzer backends."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, runtime_checkable

from typing import Protocol


@runtime_checkable
class ABAnalyzerProtocol(Protocol):
    """Structural interface satisfied by both ABTestAnalyzer and PySparkABTestAnalyzer."""

    def load_data(self, filepath: str, **kwargs: Any) -> Dict[str, Any]: ...

    def detect_columns(self) -> Dict[str, Any]: ...

    def auto_configure(self) -> Dict[str, Any]: ...

    def set_column_mapping(self, mapping: Dict[str, str]) -> None: ...

    def set_group_labels(self, treatment_label: str, control_label: str) -> None: ...

    def run_ab_test(
        self,
        segment_filter: Optional[str] = None,
        sequential_config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any: ...

    def run_segmented_analysis(
        self,
        sequential_config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Any]: ...

    def generate_summary(self, results: list) -> Any: ...
