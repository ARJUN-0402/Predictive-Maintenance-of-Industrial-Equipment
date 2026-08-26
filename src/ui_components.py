"""Reusable UI components for the industrial AI command center."""

from __future__ import annotations

from typing import Sequence

import streamlit as st

from src.ui_styles import DESIGN

# ---------------------------------------------------------------------------
# Navigation registry — single source of truth for page routing
# ---------------------------------------------------------------------------
# Each item is (sidebar_label, page_id, button_key). ``page_id`` is the value
# stored in ``st.session_state.page`` and used to dispatch to the page render
# function. ``button_key`` is the unique Streamlit key for the nav button.
NAV_GROUPS: list[dict[str, list[tuple[str, str, str]]]] = [
    {"label": "Overview", "items": [("Dashboard", "Home", "nav_dashboard")]},
    {
        "label": "Intelligence",
        "items": [
            ("Predict", "Predict Failure", "nav_predict"),
            ("Explain", "Explain Prediction", "nav_explain"),
        ],
    },
    {
        "label": "Analytics",
        "items": [
            ("Dataset", "Dataset Overview", "nav_dataset"),
            ("EDA", "Exploratory Data Analysis", "nav_eda"),
        ],
    },
    {
        "label": "Evaluation",
        "items": [
            ("Performance", "Performance Metrics", "nav_performance"),
            ("Threshold", "Threshold Optimization", "nav_threshold"),
        ],
    },
    {
        "label": "Reporting",
        "items": [
            ("Reports", "Reports & Downloads", "nav_reports"),
        ],
    },
    {
        "label": "System",
        "items": [
            ("Model Info", "Model Information", "nav_model_info"),
            ("Model Training", "Model Training", "nav_training"),
        ],
    },
]

DEFAULT_PAGE: str = "Home"

VALID_PAGES: set[str] = {
    page_id for group in NAV_GROUPS for _, page_id, _ in group["items"]
}


def navigate_to_page(page_id: str) -> None:
    """Set the current page and trigger a rerun.

    This is the only supported mechanism for changing pages. It updates the
    single source of truth (``st.session_state.page``) and requests a rerun so
    the new page renders immediately. Unknown page IDs are ignored so that
    corrupted session state cannot route the app to a non-existent page.
    """
    if page_id in VALID_PAGES:
        st.session_state["page"] = page_id
        st.rerun()


# ---------------------------------------------------------------------------
# Core identity
# ---------------------------------------------------------------------------
def render_html(html: str) -> None:
    """Render trusted application HTML through the centralized renderer.

    All static UI components must use this helper instead of calling
    ``st.markdown`` or ``st.write`` directly. This guarantees that trusted
    markup is rendered with the correct Streamlit API and prevents raw HTML
    from accidentally being displayed as plain text.

    Raises ``TypeError`` if ``html`` is not a ``str`` so that contract
    violations are caught immediately instead of falling through to Streamlit's
    plain-text renderer.
    """
    if not isinstance(html, str):
        raise TypeError(
            f"render_html expected str, got {type(html).__name__}: {html!r}"
        )
    st.html(html)


def page_header(title: str, subtitle: str) -> None:
    render_html(
        f"""
        <div class="page-identity">
            <div class="page-identity-left">
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <div class="page-identity-right">
                <span class="identity-meta">Inference Engine</span>
                <span class="identity-value">XGBoost</span>
                <span class="identity-meta">Explainability</span>
                <span class="identity-value">SHAP + LIME</span>
                <div style="margin-top: 0.5rem;">
                    <span class="status-online">
                        <span class="status-dot"></span>
                        SYSTEM ONLINE
                    </span>
                </div>
            </div>
        </div>
        """,
    )
    return None


# ---------------------------------------------------------------------------
# Section headers
# ---------------------------------------------------------------------------
def section_header(title: str) -> None:
    render_html(f'<div class="section-label">{title}</div>')
    return None


def section_title(title: str) -> None:
    render_html(f'<div class="section-title">{title}</div>')
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metric_mega(value: str, unit: str = "") -> None:
    unit_html = f'<span style="font-size:0.4em;font-weight:600;color:#8b949e;margin-left:0.25rem;">{unit}</span>' if unit else ""
    render_html(f'<div class="metric-mega">{value}{unit_html}</div>')
    return None


def metric_large(value: str, color: str | None = None) -> None:
    c = color or DESIGN["text_primary"]
    render_html(
        f'<div class="metric-large" style="color:{c};">{value}</div>',
    )
    return None


def metric_editorial_row(metrics: Sequence[tuple[str, str, str | None]]) -> None:
    if not metrics:
        return None
    cells = []
    for label, value, color in metrics:
        c = color or DESIGN["text_primary"]
        cells.append(
            f"""
            <div class="metric-editorial-item">
                <div class="metric-editorial-value" style="color:{c};">{value}</div>
                <div class="metric-editorial-label">{label}</div>
            </div>
            """
        )
    render_html(f'<div class="metric-editorial-row">{"".join(cells)}</div>')
    return None


# ---------------------------------------------------------------------------
# Status & risk
# ---------------------------------------------------------------------------
def status_indicator(status: str = "ready", label: str = "") -> None:
    color = {
        "ready": DESIGN["success"],
        "warn": DESIGN["warning"],
        "error": DESIGN["error"],
    }.get(status, DESIGN["text_muted"])

    label_html = f'<span style="margin-left:0.4rem;font-size:0.75rem;font-weight:600;color:#e6edf3;">{label}</span>' if label else ""
    render_html(
        f"""
        <span style="display:inline-flex;align-items:center;gap:0.4rem;">
            <span style="width:6px;height:6px;border-radius:50%;background-color:{color};"></span>
            {label_html}
        </span>
        """,
    )
    return None


def risk_badge(risk: str) -> None:
    color = {
        "LOW": DESIGN["success"],
        "MEDIUM": DESIGN["warning"],
        "HIGH": DESIGN["error"],
    }.get(risk.upper(), DESIGN["text_secondary"])
    render_html(
        f"""
        <span style="display:inline-flex;align-items:center;gap:0.35rem;padding:0.25rem 0.6rem;
        background-color:{color}15;border:1px solid {color}30;border-radius:3px;
        font-size:0.7rem;font-weight:700;letter-spacing:0.08em;color:{color};">
            {risk.upper()}
        </span>
        """,
    )
    return None


def risk_scale(probability: float, threshold: float = 0.5) -> None:
    pct = max(0.0, min(1.0, probability))
    marker_left = pct * 100
    render_html(
        f"""
        <div style="position:relative;padding:0.5rem 0 1.5rem 0;">
            <div class="risk-scale">
                <div class="risk-marker" style="left:{marker_left}%;" data-value="{pct:.1%}"></div>
            </div>
            <div class="risk-scale-labels">
                <span>Low</span>
                <span>Medium</span>
                <span>High</span>
            </div>
        </div>
        """,
    )
    return None


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
def telemetry_row(items: Sequence[tuple[str, str, str, str]]) -> None:
    """items: (label, value, unit, status) where status is 'ready'|'warn'|'error'"""
    color_map = {"ready": DESIGN["success"], "warn": DESIGN["warning"], "error": DESIGN["error"]}
    cells = []
    for label, value, unit, status in items:
        c = color_map.get(status, DESIGN["text_secondary"])
        cells.append(
            f"""
            <div class="telemetry-item">
                <div class="telemetry-label">{label}</div>
                <div class="telemetry-value">
                    <span>{value}<span class="telemetry-unit">{unit}</span></span>
                    <span class="telemetry-status" style="background-color:{c};"></span>
                </div>
            </div>
            """
        )
    render_html(f'<div class="telemetry-grid">{"".join(cells)}</div>')
    return None


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def prediction_panel(result: dict, threshold: float) -> None:
    pred = result["prediction"]
    prob = result["probability"]
    risk = result["risk_level"]

    label = "FAILURE" if pred == 1 else "NORMAL"
    color = DESIGN["error"] if pred == 1 else DESIGN["success"]
    pct = f"{prob * 100:.1f}%"

    decision = (
        f"Model indicates elevated failure risk ({pct}) above the decision threshold ({threshold:.2f}). "
        "Schedule inspection or preventive maintenance. This is a model indication, not a guarantee of failure."
        if pred == 1
        else f"Model indicates normal operation ({pct}) below the decision threshold ({threshold:.2f}). "
        "Continue monitoring according to the normal maintenance schedule."
    )

    render_html(
        f"""
        <div class="prediction-panel">
            <div class="prediction-value" style="color:{color};">{pct}</div>
            <div class="prediction-label" style="color:{color};">{label}</div>
            <div class="prediction-meta">Risk: {risk} &nbsp;|&nbsp; Threshold: {threshold:.2f}</div>
            <div class="prediction-decision">{decision}</div>
        </div>
        """,
    )
    return None


# ---------------------------------------------------------------------------
# Feature contributions
# ---------------------------------------------------------------------------
def feature_contribution_bars(
    features: Sequence[tuple[str, float]],
    top_n: int = 8,
) -> None:
    rows = []
    for name, val in features[:top_n]:
        display_name = name.replace("_", " ").title()
        if len(display_name) > 28:
            display_name = display_name[:25] + "..."
        direction = "positive" if val >= 0 else "negative"
        width = min(abs(val) * 400, 100)
        color = DESIGN["accent"] if val >= 0 else DESIGN["success"]
        sign = "+" if val >= 0 else ""
        rows.append(
            f"""
            <div class="feature-row">
                <div class="feature-name">{display_name}</div>
                <div class="feature-bar-track">
                    <div class="feature-bar-fill {direction}" style="width:{width}%;background-color:{color};"></div>
                </div>
                <div class="feature-value {direction}">{sign}{val:.4f}</div>
            </div>
            """
        )
    render_html("".join(rows))
    return None


# ---------------------------------------------------------------------------
# Technical metadata
# ---------------------------------------------------------------------------
def technical_metadata(items: Sequence[tuple[str, str]]) -> None:
    cells = []
    for label, value in items:
        cells.append(
            f"""
            <div class="meta-item">
                <span class="identity-meta">{label}</span>
                <span class="identity-value">{value}</span>
            </div>
            """
        )
    render_html(f'<div class="meta-strip">{"".join(cells)}</div>')
    return None


# ---------------------------------------------------------------------------
# Sidebar nav rail
# ---------------------------------------------------------------------------
def nav_rail_item(
    label: str,
    page_id: str,
    button_key: str,
    active: bool = False,
    primary: bool = False,
) -> None:
    """Render a single sidebar navigation item as a real Streamlit button.

    The active page is shown as a disabled (highlighted) button. Inactive pages
    are clickable and trigger navigation via :func:`navigate_to_page`, which
    updates ``st.session_state.page`` and reruns the app.
    """
    if active:
        st.button(
            label,
            key=button_key,
            disabled=True,
            use_container_width=True,
        )
        return None

    if primary:
        render_html('<div class="nav-rail-primary">')

    if st.button(label, key=button_key, use_container_width=True):
        navigate_to_page(page_id)

    if primary:
        render_html("</div>")
    return None


# ---------------------------------------------------------------------------
# Compact alerts
# ---------------------------------------------------------------------------
def system_alert(message: str, level: str = "error") -> None:
    render_html(
        f'<div class="system-alert {level}">{message}</div>',
    )
    return None


# ---------------------------------------------------------------------------
# Command hero
# ---------------------------------------------------------------------------
def command_hero(title: str, subtitle: str, right_content: str = "") -> None:
    render_html(
        f"""
        <div class="command-hero">
            <div class="command-hero-grid">
                <div class="command-primary">
                    <div class="hero-title">{title}</div>
                    <div class="hero-subtitle">{subtitle}</div>
                </div>
                <div class="command-secondary">
                    {right_content}
                </div>
            </div>
        </div>
        """,
    )
    return None


# ---------------------------------------------------------------------------
# Instrumentation inputs
# ---------------------------------------------------------------------------
def instrument_input(label: str, widget_html: str) -> None:
    render_html(
        f"""
        <div class="instrument-item">
            <span class="instrument-name">{label}</span>
            {widget_html}
        </div>
        """,
    )
    return None


# ---------------------------------------------------------------------------
# Report center
# ---------------------------------------------------------------------------
def report_row(title: str, meta: str, download_fn) -> None:
    render_html(
        f"""
        <div class="report-item">
            <div>
                <div class="report-title">{title}</div>
                <div class="report-meta">{meta}</div>
            </div>
        </div>
        """,
    )
    download_fn()
    return None


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------
def render_page_header(title: str, subtitle: str | None = None) -> None:
    page_header(title, subtitle or "")
    return None


def render_section_header(title: str) -> None:
    section_header(title)
    return None


def render_metric_card(title: str, value: str, color: str | None = None) -> None:
    c = color or DESIGN["text_primary"]
    render_html(
        f"""
        <div style="padding:1rem 0;border-bottom:1px solid #21262d;">
            <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#484f58;margin-bottom:0.35rem;">{title}</div>
            <div style="font-size:1.5rem;font-weight:800;color:{c};letter-spacing:-0.02em;">{value}</div>
        </div>
        """,
    )
    return None


def render_metric_row(
    metrics: Sequence[tuple[str, str, str | None]],
    columns: int = 4,
) -> None:
    for i in range(0, len(metrics), columns):
        chunk = metrics[i : i + columns]
        cols = st.columns(len(chunk))
        for col, (title, value, color) in zip(cols, chunk):
            with col:
                render_metric_card(title, value, color)
    return None


def render_status_dot(status: str = "ready") -> str:
    color = {"ready": DESIGN["success"], "warn": DESIGN["warning"], "error": DESIGN["error"]}.get(
        status, DESIGN["text_muted"]
    )
    return f'<span style="width:6px;height:6px;border-radius:50%;background-color:{color};display:inline-block;"></span>'


def card_html(title: str, value: str, color: str) -> str:
    return (
        f'<div style="padding:1rem 0;border-bottom:1px solid #21262d;">'
        f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#484f58;margin-bottom:0.35rem;">{title}</div>'
        f'<div style="font-size:1.5rem;font-weight:800;color:{color};letter-spacing:-0.02em;">{value}</div></div>'
    )


def render_info_card(title: str, content: str) -> None:
    render_html(
        f"""
        <div style="padding:0.75rem 0;border-bottom:1px solid #161c24;">
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#00d4ff;margin-bottom:0.35rem;">{title}</div>
            <div style="font-size:0.8rem;color:#8b949e;line-height:1.5;">{content}</div>
        </div>
        """,
    )
    return None


def render_action_card(title: str, content: str) -> None:
    render_html(
        f"""
        <div style="padding:0.75rem 0;border-bottom:1px solid #161c24;">
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#00d4ff;margin-bottom:0.35rem;">{title}</div>
            <div style="font-size:0.8rem;color:#8b949e;line-height:1.5;">{content}</div>
        </div>
        """,
    )
    return None


def render_highlight(text: str) -> None:
    render_html(
        f"<div style='display:flex;align-items:center;gap:0.5rem;padding:0.35rem 0;"
        f"font-size:0.85rem;color:#e6edf3;'><span style='color:#00c853;font-weight:700;'>✓</span>{text}</div>",
    )
    return None


def render_prediction_card(result: dict, threshold: float) -> None:
    prediction_panel(result, threshold)
    return None


__all__ = [
    "NAV_GROUPS",
    "DEFAULT_PAGE",
    "VALID_PAGES",
    "navigate_to_page",
    "page_header",
    "section_header",
    "section_title",
    "metric_mega",
    "metric_large",
    "metric_editorial_row",
    "status_indicator",
    "risk_badge",
    "risk_scale",
    "telemetry_row",
    "prediction_panel",
    "feature_contribution_bars",
    "technical_metadata",
    "nav_rail_item",
    "system_alert",
    "command_hero",
    "render_page_header",
    "render_section_header",
    "render_metric_card",
    "render_metric_row",
    "render_status_dot",
    "card_html",
    "render_info_card",
    "render_action_card",
    "render_highlight",
    "render_prediction_card",
]
