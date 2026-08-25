"""Centralized UI styles and design tokens for the industrial AI theme."""

DESIGN = {
    "bg_primary": "#0B0F14",
    "bg_secondary": "#11161D",
    "bg_tertiary": "#171C24",
    "bg_card": "#141A22",
    "accent": "#00d4ff",
    "accent_dim": "#00a8cc",
    "success": "#00c853",
    "warning": "#ffab00",
    "error": "#ff4b4b",
    "text_primary": "#e0e0e0",
    "text_secondary": "#8b8b9b",
    "text_muted": "#6b6b7b",
    "border": "#1E2630",
    "border_dim": "#171C24",
}

CSS = """
<style>
/* === Industrial AI Design System === */

/* Global app background */
.stApp {
    background-color: #0B0F14;
}

/* Main content padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

/* Typography */
h1, h2, h3 {
    color: #e0e0e0;
    font-weight: 600;
}
h1 { font-size: 1.8rem; }
h2 { font-size: 1.4rem; }
h3 { font-size: 1.15rem; }

/* === Metric Cards === */
.metric-card {
    background-color: #141A22;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    border: 1px solid #1E2630;
    border-left: 4px solid #00d4ff;
    transition: border-color 0.2s ease;
}
.metric-card .label {
    font-size: 0.7rem;
    color: #6b6b7b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #00d4ff;
    line-height: 1.2;
}

/* === Section Headers === */
.section-title {
    color: #00d4ff;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 28px;
    margin-bottom: 12px;
}

/* === Hero Banner === */
.hero {
    background: linear-gradient(135deg, #11161D 0%, #171C24 100%);
    border-radius: 16px;
    padding: 36px 32px;
    margin-bottom: 24px;
    border: 1px solid #1E2630;
}
.hero h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.hero p {
    color: #8b8b9b;
    font-size: 1rem;
    line-height: 1.6;
    margin-bottom: 0;
}

/* === Status Dots === */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}
.status-ready { background-color: #00c853; }
.status-warn { background-color: #ffab00; }
.status-error { background-color: #ff4b4b; }

/* === Prediction Card === */
.prediction-card {
    background-color: #141A22;
    border-radius: 14px;
    padding: 32px 24px;
    margin: 16px 0;
    border: 1px solid #1E2630;
    border-left: 5px solid #00d4ff;
    text-align: center;
}
.prediction-card .prob {
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1.1;
}
.prediction-card .label {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e0e0e0;
    margin-top: 8px;
}
.prediction-card .meta {
    font-size: 0.85rem;
    color: #6b6b7b;
    margin-top: 10px;
}

/* === Info Cards === */
.info-card {
    background-color: #141A22;
    border-radius: 10px;
    padding: 18px;
    margin-top: 12px;
    border: 1px solid #1E2630;
}
.info-card .title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.info-card .text {
    font-size: 0.85rem;
    color: #8b8b9b;
    line-height: 1.5;
}

/* === Action Box === */
.action-box {
    background-color: #141A22;
    border-radius: 10px;
    padding: 18px;
    margin-top: 12px;
    border: 1px solid #1E2630;
}
.action-box .title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #00d4ff;
    margin-bottom: 8px;
}
.action-box .text {
    font-size: 0.82rem;
    color: #8b8b9b;
    line-height: 1.5;
}

/* === Highlight Items === */
.highlight-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    color: #e0e0e0;
    font-size: 0.9rem;
}
.highlight-item .icon {
    color: #00c853;
    font-weight: bold;
}

/* === Architecture Flow === */
.arch-node {
    background-color: #141A22;
    border: 1px solid #1E2630;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    margin: 4px;
}
.arch-node .title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #00d4ff;
}
.arch-node .desc {
    font-size: 0.7rem;
    color: #6b6b7b;
    margin-top: 4px;
}
.arch-arrow {
    text-align: center;
    color: #00d4ff;
    font-size: 18px;
    margin: 6px 0;
}

/* === Sidebar === */
.sidebar-section {
    margin-bottom: 14px;
}
.sidebar-section .section-header {
    font-size: 0.65rem;
    font-weight: 700;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
    padding-left: 4px;
}

/* === Dataframe === */
.stDataFrame {
    background-color: #141A22;
    color: #e0e0e0;
}

/* === Tabs === */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #141A22;
    color: #8b8b9b;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background-color: #00d4ff;
    color: #0B0F14;
}

/* === Buttons === */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
}

/* === Alerts === */
.stAlert {
    border-radius: 10px;
}

/* === Expander === */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #e0e0e0;
}

/* === Horizontal Rule === */
hr {
    border-color: #1E2630;
}
</style>
"""
