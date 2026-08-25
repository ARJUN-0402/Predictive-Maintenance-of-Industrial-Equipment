"""Reusable UI helper functions for the industrial AI Streamlit application."""

from __future__ import annotations

from typing import Sequence

import streamlit as st

from src.ui_styles import DESIGN


def render_page_header(title: str, subtitle: str | None = None) -> None:
    """Render a consistent page header with optional subtitle."""
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def render_section_header(title: str) -> None:
    """Render a section header with the accent style."""
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, color: str | None = None) -> None:
    """Render a single metric card."""
    accent = color or DESIGN["accent"]
    st.markdown(
        card_html(title, value, accent),
        unsafe_allow_html=True,
    )


def render_metric_row(
    metrics: Sequence[tuple[str, str, str | None]],
    columns: int = 4,
) -> None:
    """Render a row of metric cards.

    Args:
        metrics: Sequence of (title, value, color) tuples.
        columns: Number of columns per row.
    """
    for i in range(0, len(metrics), columns):
        chunk = metrics[i : i + columns]
        cols = st.columns(len(chunk))
        for col, (title, value, color) in zip(cols, chunk):
            with col:
                render_metric_card(title, value, color)


def render_info_card(title: str, content: str) -> None:
    """Render an informational card with title and body text."""
    st.markdown(
        f"""
        <div class="info-card">
            <div class="title">{title}</div>
            <div class="text">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_card(title: str, content: str) -> None:
    """Render a recommended action card."""
    st.markdown(
        f"""
        <div class="action-box">
            <div class="title">{title}</div>
            <div class="text">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_dot(status: str = "ready") -> str:
    """Return HTML for a status dot."""
    css_class = f"status-{status}"
    return f'<span class="status-dot {css_class}"></span>'


def render_highlight(text: str) -> None:
    """Render a highlighted feature/checklist item."""
    st.markdown(
        f"<div class='highlight-item'>"
        f"<span class='icon'>✓</span>{text}</div>",
        unsafe_allow_html=True,
    )


def render_prediction_card(result: dict, threshold: float) -> None:
    """Render the prediction result card with probability and risk."""
    pred = result["prediction"]
    prob = result["probability"]
    risk = result["risk_level"]

    label = "FAILURE" if pred == 1 else "NORMAL"
    color = DESIGN["error"] if pred == 1 else DESIGN["success"]

    st.markdown(
        f"""
        <div class="prediction-card" style="border-left-color:{color};">
            <div class="prob" style="color:{color};">{prob * 100:.1f}%</div>
            <div class="label" style="color:{color};">{label}</div>
            <div class="meta">Risk Level: {risk} &nbsp;|&nbsp;
                Threshold: {threshold:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(prob, text=f"{prob:.1%}")

    if pred == 1:
        st.info(
            f"Model indicates elevated failure risk ({prob * 100:.1f}%) "
            f"above the decision threshold ({threshold:.2f})."
        )
        render_action_card(
            "Recommended Action",
            "Schedule inspection or preventive maintenance as soon as "
            "practical. This is a model indication, not a guarantee of failure.",
        )
    else:
        st.info(
            f"Model indicates normal operation ({prob * 100:.1f}%) "
            f"below the decision threshold ({threshold:.2f})."
        )
        render_action_card(
            "Recommended Action",
            "Continue monitoring according to the normal maintenance "
            "schedule. This is a model indication, not a guarantee of "
            "continued operation.",
        )


def card_html(title: str, value: str, color: str) -> str:
    """Generate HTML for a metric card."""
    return (
        f'<div style="background:#141A22;border-radius:12px;padding:18px 16px;'
        f"text-align:center;border:1px solid #1E2630;"
        f"border-left:4px solid {color};'>"
        f'<div style="font-size:0.7rem;color:#6b6b7b;text-transform:uppercase;'
        f"letter-spacing:0.8px;margin-bottom:6px;\">"
        f"{title}</div>"
        f'<div style="font-size:1.6rem;font-weight:700;color:{color};'
        f"line-height:1.2;\">"
        f"{value}</div></div>"
    )


__all__ = [
    "render_page_header",
    "render_section_header",
    "render_metric_card",
    "render_metric_row",
    "render_info_card",
    "render_action_card",
    "render_status_dot",
    "render_highlight",
    "render_prediction_card",
    "card_html",
]
