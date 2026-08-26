"""Regression tests for centralized HTML rendering architecture."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from src.ui_components import (
    card_html,
    command_hero,
    feature_contribution_bars,
    metric_editorial_row,
    page_header,
    prediction_panel,
    render_html,
    render_status_dot,
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
                # The one legitimate st.write is a loading message, not HTML
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
