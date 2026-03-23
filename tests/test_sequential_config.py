"""Tests for the sequential_config extraction from analyzer.py."""

import pytest

from src.statistics.sequential_config import (
    SequentialConfig,
    resolve_sequential_config,
    evaluate_sequential_decision,
    _coerce_int,
    _coerce_float,
)
from src.statistics.statsmodels_engine import StatsmodelsABTestEngine


@pytest.fixture
def engine():
    return StatsmodelsABTestEngine(significance_level=0.05, power_threshold=0.8)


class TestResolveSequentialConfig:
    """Test config resolution from None, True, dict, and edge cases."""

    def test_none_returns_disabled(self, engine):
        config = resolve_sequential_config(None, {}, engine)
        assert config.enabled is False

    def test_false_returns_disabled(self, engine):
        config = resolve_sequential_config(False, {}, engine)
        assert config.enabled is False

    def test_true_returns_enabled_defaults(self, engine):
        config = resolve_sequential_config(True, {}, engine)
        assert config.enabled is True
        assert config.look_index == 1
        assert config.max_looks == 1
        assert config.spending_method == engine.DEFAULT_SEQUENTIAL_METHOD

    def test_dict_with_settings(self, engine):
        raw = {
            "look_index": 3,
            "max_looks": 5,
            "spending_method": "pocock",
        }
        config = resolve_sequential_config(raw, {}, engine)
        assert config.enabled is True
        assert config.look_index == 3
        assert config.max_looks == 5
        assert config.spending_method == "pocock"

    def test_dict_enabled_false(self, engine):
        raw = {"enabled": False, "look_index": 3}
        config = resolve_sequential_config(raw, {}, engine)
        assert config.enabled is False

    def test_fallback_to_column_mapping(self, engine):
        column_mapping = {"sequential": {"look_index": 2, "max_looks": 4}}
        config = resolve_sequential_config(None, column_mapping, engine)
        assert config.enabled is True
        assert config.look_index == 2
        assert config.max_looks == 4

    def test_look_index_clamped_to_max_looks(self, engine):
        raw = {"look_index": 10, "max_looks": 3}
        config = resolve_sequential_config(raw, {}, engine)
        assert config.look_index == 3

    def test_unsupported_type_returns_disabled(self, engine):
        config = resolve_sequential_config("invalid", {}, engine)
        assert config.enabled is False

    def test_frozen_dataclass(self, engine):
        config = resolve_sequential_config(True, {}, engine)
        with pytest.raises(AttributeError):
            config.enabled = False


class TestEvaluateSequentialDecision:
    """Test the full decision evaluation flow."""

    def test_disabled_returns_not_requested(self, engine):
        result = evaluate_sequential_decision(
            sequential_config=None,
            column_mapping={},
            stats_engine=engine,
            p_value=0.03,
            effect_size=1.5,
            confidence_interval=(0.5, 2.5),
        )
        assert result["decision"] == "not_requested"
        assert result["enabled"] is False

    def test_enabled_returns_decision(self, engine):
        result = evaluate_sequential_decision(
            sequential_config={"look_index": 2, "max_looks": 5},
            column_mapping={},
            stats_engine=engine,
            p_value=0.03,
            effect_size=1.5,
            confidence_interval=(0.5, 2.5),
        )
        assert result["enabled"] is True
        assert result["decision"] in ("stop_reject", "stop_futility", "continue")
        assert result["look_index"] == 2
        assert result["max_looks"] == 5


class TestCoercionHelpers:
    """Test _coerce_int and _coerce_float edge cases."""

    def test_coerce_int_valid(self):
        assert _coerce_int(5, default=1) == 5

    def test_coerce_int_string(self):
        assert _coerce_int("3", default=1) == 3

    def test_coerce_int_invalid(self):
        assert _coerce_int("abc", default=7) == 7

    def test_coerce_int_none(self):
        assert _coerce_int(None, default=2) == 2

    def test_coerce_int_minimum(self):
        assert _coerce_int(-5, default=1, minimum=1) == 1

    def test_coerce_float_valid(self):
        assert _coerce_float(0.5, default=0.1) == 0.5

    def test_coerce_float_clamp_high(self):
        assert _coerce_float(2.0, default=0.5, maximum=1.0) == 1.0

    def test_coerce_float_clamp_low(self):
        assert _coerce_float(-0.5, default=0.5, minimum=0.0) == 0.0

    def test_coerce_float_invalid(self):
        assert _coerce_float("nope", default=0.75) == 0.75
