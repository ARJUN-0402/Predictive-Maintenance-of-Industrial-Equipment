"""Regression tests for centralized HTML rendering architecture."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from src.ui_components import (
    card_html,
    command_hero,
    feature_contribution_bars,
    metric_editorial_row,
    page_header,
    prediction_panel,
    render_html,
    render_status_dot,
    render_metric_card,
    render_metric_row,
    render_info_card,
    render_action_card,
    render_highlight,
    risk_badge,
    risk_scale,
    section_header,
    section_title,
    status_indicator,
    system_alert,
    telemetry_row,
    technical_metadata,
)


class TestRenderHtml:
    """Centralized HTML renderer must always use the trusted Streamlit API."""

    def test_calls_markdown_with_unsafe_html(self) -> None:
        mock_st = MagicMock()
        with patch("src.ui_components.st", mock_st):
            render_html("<div>test</div>")

        mock_st.markdown.assert_called_once_with(
            "<div>test</div>",
            unsafe_allow_html=True,
        )

    def test_does_not_call_write(self) -> None:
        mock_st = MagicMock()
        with patch("src.ui_components.st", mock_st):
            render_html("<div>test</div>")

        mock_st.write.assert_not_called()

    def test_does_not_call_text(self) -> None:
        mock_st = MagicMock()
        with patch("src.ui_components.st", mock_st):
            render_html("<div>test</div>")

        mock_st.text.assert_not_called()

    def test_rejects_non_string_input(self) -> None:
        with pytest.raises(TypeError, match="render_html expected str"):
            render_html(123)

    def test_rejects_none_input(self) -> None:
        with pytest.raises(TypeError, match="render_html expected str"):
            render_html(None)

    def test_rejects_list_input(self) -> None:
        with pytest.raises(TypeError, match="render_html expected str"):
            render_html(["<div>", "</div>"])


class TestComponentsUseCentralizedRenderer:
    """Every HTML-producing component must route through render_html."""

    def _collect_st_markdown_calls(self, func, *args, **kwargs):
        mock_st = MagicMock()
        with patch("src.ui_components.st", mock_st):
            func(*args, **kwargs)
        return [call.args for call in mock_st.markdown.call_args_list]

    def test_page_header_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(page_header, "Title", "Subtitle")
        assert len(calls) == 1
        html = calls[0][0]
        assert "<div" in html
        assert "page-identity" in html

    def test_section_header_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(section_header, "Section")
        assert len(calls) == 1
        assert calls[0] == ('<div class="section-label">Section</div>',)

    def test_section_title_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(section_title, "Title")
        assert len(calls) == 1
        assert calls[0] == ('<div class="section-title">Title</div>',)

    def test_telemetry_row_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(
            telemetry_row,
            [("Temp", "298.0", "K", "ready")],
        )
        assert len(calls) == 1
        html = calls[0][0]
        assert "telemetry-item" in html
        assert "telemetry-grid" in html

    def test_metric_editorial_row_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(
            metric_editorial_row,
            [("Accuracy", "0.95", "#00d4ff")],
        )
        assert len(calls) == 1
        html = calls[0][0]
        assert "metric-editorial-item" in html
        assert "metric-editorial-row" in html

    def test_technical_metadata_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(
            technical_metadata,
            [("Engine", "XGBoost")],
        )
        assert len(calls) == 1
        html = calls[0][0]
        assert "meta-item" in html
        assert "meta-strip" in html

    def test_prediction_panel_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(
            prediction_panel,
            {"prediction": 1, "probability": 0.75, "risk_level": "High"},
            0.5,
        )
        assert len(calls) == 1
        html = calls[0][0]
        assert "prediction-panel" in html

    def test_feature_contribution_bars_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(
            feature_contribution_bars,
            [("feature_a", 0.5)],
        )
        assert len(calls) == 1
        html = calls[0][0]
        assert "feature-row" in html

    def test_status_indicator_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(status_indicator, "ready", "OK")
        assert len(calls) == 1
        html = calls[0][0]
        assert "<span" in html

    def test_risk_badge_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(risk_badge, "HIGH")
        assert len(calls) == 1
        html = calls[0][0]
        assert "HIGH" in html

    def test_risk_scale_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(risk_scale, 0.7, 0.5)
        assert len(calls) == 1
        html = calls[0][0]
        assert "risk-scale" in html

    def test_system_alert_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(system_alert, "Alert", "error")
        assert len(calls) == 1
        html = calls[0][0]
        assert "system-alert" in html
        assert "Alert" in html

    def test_command_hero_uses_render_html(self) -> None:
        calls = self._collect_st_markdown_calls(
            command_hero, "Hero", "Subtitle", "<span>right</span>"
        )
        assert len(calls) == 1
        html = calls[0][0]
        assert "command-hero" in html


class TestComponentReturnValues:
    """Self-rendering components must return None to enforce Architecture A."""

    def test_page_header_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = page_header("Title", "Subtitle")
        assert result is None

    def test_section_header_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = section_header("Section")
        assert result is None

    def test_section_title_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = section_title("Title")
        assert result is None

    def test_telemetry_row_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = telemetry_row([("Temp", "298.0", "K", "ready")])
        assert result is None

    def test_metric_editorial_row_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = metric_editorial_row([("Accuracy", "0.95", "#00d4ff")])
        assert result is None

    def test_technical_metadata_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = technical_metadata([("Engine", "XGBoost")])
        assert result is None

    def test_prediction_panel_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = prediction_panel({"prediction": 1, "probability": 0.75, "risk_level": "High"}, 0.5)
        assert result is None

    def test_feature_contribution_bars_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = feature_contribution_bars([("feature_a", 0.5)])
        assert result is None

    def test_status_indicator_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = status_indicator("ready", "OK")
        assert result is None

    def test_risk_badge_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = risk_badge("HIGH")
        assert result is None

    def test_risk_scale_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = risk_scale(0.7, 0.5)
        assert result is None

    def test_system_alert_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = system_alert("Alert", "error")
        assert result is None

    def test_command_hero_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = command_hero("Hero", "Subtitle", "<span>right</span>")
        assert result is None

    def test_render_metric_card_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = render_metric_card("Accuracy", "0.95", "#00d4ff")
        assert result is None

    def test_render_metric_row_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = render_metric_row([("Accuracy", "0.95", "#00d4ff")])
        assert result is None

    def test_render_info_card_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = render_info_card("Title", "Content")
        assert result is None

    def test_render_action_card_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = render_action_card("Title", "Content")
        assert result is None

    def test_render_highlight_returns_none(self) -> None:
        with patch("src.ui_components.st", MagicMock()):
            result = render_highlight("Text")
        assert result is None


class TestHtmlCompleteness:
    """Multi-item components must include every item in a single render call."""

    def test_telemetry_row_includes_all_five_production_items(self) -> None:
        items = [
            ("Air Temperature", "298.0", "K", "ready"),
            ("Process Temperature", "310.0", "K", "ready"),
            ("Rotational Speed", "1500", "RPM", "ready"),
            ("Torque", "40.0", "Nm", "ready"),
            ("Tool Wear", "50.0", "min", "ready"),
        ]
        calls = self._collect_st_markdown_calls(telemetry_row, items)
        assert len(calls) == 1
        html = calls[0][0]
        for label, value, unit, _ in items:
            assert label in html, f"Missing telemetry label: {label}"
            assert value in html, f"Missing telemetry value: {value}"
            assert unit in html, f"Missing telemetry unit: {unit}"

    def test_telemetry_row_html_is_balanced_tags(self) -> None:
        items = [
            ("Air Temperature", "298.0", "K", "ready"),
            ("Process Temperature", "310.0", "K", "ready"),
            ("Rotational Speed", "1500", "RPM", "ready"),
            ("Torque", "40.0", "Nm", "ready"),
            ("Tool Wear", "50.0", "min", "ready"),
        ]
        calls = self._collect_st_markdown_calls(telemetry_row, items)
        html = calls[0][0]
        assert html.count("<div") == html.count("</div>")
        assert html.count("<span") == html.count("</span>")

    def test_metric_editorial_row_includes_all_items(self) -> None:
        metrics = [
            ("Accuracy", "0.95", "#00d4ff"),
            ("ROC-AUC", "0.92", "#ffab00"),
            ("F1 Score", "0.89", "#00d4ff"),
            ("Models", "3", "#8b949e"),
        ]
        calls = self._collect_st_markdown_calls(metric_editorial_row, metrics)
        assert len(calls) == 1
        html = calls[0][0]
        for label, value, _ in metrics:
            assert label in html
            assert value in html

    def test_technical_metadata_includes_all_items(self) -> None:
        items = [
            ("Inference Engine", "XGBoost"),
            ("Explainability", "SHAP + LIME"),
            ("Dataset", "AI4I 2020"),
            ("Processed Rows", "10000"),
        ]
        calls = self._collect_st_markdown_calls(technical_metadata, items)
        assert len(calls) == 1
        html = calls[0][0]
        for label, value in items:
            assert label in html
            assert value in html

    def test_feature_contribution_bars_includes_all_features(self) -> None:
        features = [
            ("air_temperature", 0.5),
            ("process_temperature", -0.3),
            ("rotational_speed", 0.8),
        ]
        calls = self._collect_st_markdown_calls(feature_contribution_bars, features)
        assert len(calls) == 1
        html = calls[0][0]
        for name, _ in features:
            display = name.replace("_", " ").title()
            assert display in html

    def _collect_st_markdown_calls(self, func, *args, **kwargs):
        mock_st = MagicMock()
        with patch("src.ui_components.st", mock_st):
            func(*args, **kwargs)
        return [call.args for call in mock_st.markdown.call_args_list]


class TestRawHtmlNotRenderedAsPlainText:
    """Trusted application HTML must never pass through plain-text outputs."""

    def test_no_st_write_used_for_html_in_streamlit_app(self) -> None:
        import app.streamlit_app as app

        source = inspect.getsource(app)
        lines = source.splitlines()
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "st.write(" in stripped:
                assert "Loading and preprocessing" in stripped, (
                    f"st.write on line {i} appears to render non-text content: {stripped}"
                )


class TestReturnedHtmlStrings:
    """HTML-returning helpers should document their contract."""

    def test_render_status_dot_returns_html_string(self) -> None:
        html = render_status_dot("ready")
        assert isinstance(html, str)
        assert "<span" in html

    def test_card_html_returns_html_string(self) -> None:
        html = card_html("Accuracy", "0.95", "#00d4ff")
        assert isinstance(html, str)
        assert "Accuracy" in html
        assert "0.95" in html
