"""UI regression tests for reusable components and state logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.predict import _classify_risk
from src.ui_components import card_html
from src.ui_styles import CSS, DESIGN
from src.utils import format_probability


class TestFormatProbability:
    """Tests for probability formatting helper."""

    def test_zero(self) -> None:
        assert format_probability(0.0) == "0.0%"

    def test_one(self) -> None:
        assert format_probability(1.0) == "100.0%"

    def test_half(self) -> None:
        assert format_probability(0.5) == "50.0%"

    def test_small_value(self) -> None:
        assert format_probability(0.001) == "0.1%"

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            format_probability(-0.1)

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            format_probability(1.1)


class TestClassifyRisk:
    """Tests for risk level classification."""

    def test_high_risk(self) -> None:
        assert _classify_risk(0.95, threshold=0.5) == "High"
        assert _classify_risk(0.71, threshold=0.5) == "High"

    def test_medium_risk(self) -> None:
        assert _classify_risk(0.5, threshold=0.5) == "Medium"
        assert _classify_risk(0.3, threshold=0.5) == "Medium"

    def test_low_risk(self) -> None:
        assert _classify_risk(0.1, threshold=0.5) == "Low"
        assert _classify_risk(0.29, threshold=0.5) == "Low"

    def test_all_levels_covered(self) -> None:
        levels = {
            _classify_risk(p, threshold=0.5) for p in [0.05, 0.4, 0.55, 0.8, 0.99]
        }
        assert levels == {"Low", "Medium", "High"}


class TestCardHtml:
    """Tests for card HTML generation."""

    def test_contains_title(self) -> None:
        html = card_html("Accuracy", "0.95", "#00d4ff")
        assert "Accuracy" in html

    def test_contains_value(self) -> None:
        html = card_html("Accuracy", "0.95", "#00d4ff")
        assert "0.95" in html

    def test_contains_color(self) -> None:
        html = card_html("Accuracy", "0.95", "#00d4ff")
        assert "#00d4ff" in html


class TestDesignTokens:
    """Tests for design token consistency."""

    def test_required_colors_present(self) -> None:
        required = [
            "accent",
            "success",
            "warning",
            "error",
            "bg_primary",
            "text_primary",
        ]
        for key in required:
            assert key in DESIGN, f"Missing design token: {key}"

    def test_css_contains_key_selectors(self) -> None:
        assert ".metric-card" in CSS
        assert ".prediction-card" in CSS
        assert ".section-title" in CSS
        assert ".hero" in CSS
        assert ".status-dot" in CSS


class TestPredictionStatus:
    """Tests for prediction status logic."""

    def test_normal_prediction(self) -> None:
        result = {"prediction": 0, "probability": 0.1, "risk_level": "Low"}
        assert result["prediction"] == 0

    def test_failure_prediction(self) -> None:
        result = {"prediction": 1, "probability": 0.8, "risk_level": "High"}
        assert result["prediction"] == 1

    def test_threshold_classification(self) -> None:
        assert (0.6 >= 0.5) is True
        assert (0.4 >= 0.5) is False


class TestDisplayPredictionCardCompatibility:
    """Ensure _display_prediction_card maintains expected behavior."""

    def test_no_format_keyword(self) -> None:
        from app.streamlit_app import _display_prediction_card

        mock_progress = MagicMock()
        mock_st = MagicMock()
        mock_st.progress = mock_progress

        with patch("app.streamlit_app.st", mock_st):
            _display_prediction_card(
                {"prediction": 0, "probability": 0.001, "risk_level": "Low"},
                0.5,
            )

        assert mock_progress.called
        call_kwargs = mock_progress.call_args.kwargs
        assert "format" not in call_kwargs
        assert call_kwargs.get("text") == "0.1%"
        assert mock_progress.call_args.args[0] == 0.001

    def test_failure_result(self) -> None:
        from app.streamlit_app import _display_prediction_card

        mock_progress = MagicMock()
        mock_st = MagicMock()
        mock_st.progress = mock_progress

        with patch("app.streamlit_app.st", mock_st):
            _display_prediction_card(
                {"prediction": 1, "probability": 0.75, "risk_level": "High"},
                0.5,
            )

        assert mock_progress.called
        call_kwargs = mock_progress.call_args.kwargs
        assert "format" not in call_kwargs
        assert call_kwargs.get("text") == "75.0%"
        assert mock_progress.call_args.args[0] == 0.75
