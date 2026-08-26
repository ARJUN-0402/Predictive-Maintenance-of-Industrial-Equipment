"""Centralized design tokens and injected CSS for the industrial AI command center."""

DESIGN = {
    "bg_primary": "#080B0F",
    "bg_secondary": "#0D1117",
    "bg_tertiary": "#121820",
    "bg_elevated": "#161C24",
    "accent": "#00d4ff",
    "accent_dim": "#007a99",
    "success": "#00c853",
    "warning": "#ffab00",
    "error": "#ff4b4b",
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "text_muted": "#484f58",
    "border": "#21262d",
    "border_dim": "#161c24",
    "font_mono": "SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace",
    "font_sans": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
}

CSS = """
<style>
/* === Reset & Base === */
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
.stApp {
    background-color: #080B0F;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: #e6edf3;
}

/* Engineering grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

.block-container {
    padding-top: 0.75rem;
    padding-bottom: 2.5rem;
    max-width: 1400px;
    position: relative;
    z-index: 1;
}

/* === Typography Scale === */
.hero-title {
    font-size: clamp(2.5rem, 5vw, 4.5rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.05;
    color: #e6edf3;
    margin: 0;
}
.hero-subtitle {
    font-size: 1rem;
    color: #8b949e;
    line-height: 1.5;
    margin-top: 0.5rem;
    font-weight: 400;
}
.metric-mega {
    font-size: clamp(3rem, 6vw, 5.5rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    color: #00d4ff;
}
.metric-large {
    font-size: clamp(2rem, 3vw, 2.75rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.metric-medium {
    font-size: clamp(1.25rem, 2vw, 1.5rem);
    font-weight: 600;
    line-height: 1.2;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #00d4ff;
    margin-bottom: 0.75rem;
    display: block;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 0.75rem;
    margin-top: 1.75rem;
    letter-spacing: -0.01em;
}
.technical-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #484f58;
    display: block;
    margin-bottom: 0.25rem;
}
.technical-value {
    font-size: 0.85rem;
    font-weight: 600;
    color: #e6edf3;
    font-family: SFMono-Regular, Consolas, monospace;
}

/* === Page Identity Header === */
.page-identity {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 1.25rem 0 1.5rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1.5rem;
}
.page-identity-left h1 {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0;
    color: #e6edf3;
}
.page-identity-left p {
    font-size: 0.75rem;
    color: #8b949e;
    margin: 0.25rem 0 0 0;
}
.page-identity-right {
    text-align: right;
}
.identity-meta {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #484f58;
    display: block;
    margin-bottom: 0.35rem;
}
.identity-value {
    font-size: 0.8rem;
    font-weight: 600;
    color: #e6edf3;
    font-family: SFMono-Regular, Consolas, monospace;
    display: block;
    margin-bottom: 0.35rem;
}
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #00c853;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #00c853;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,200,83,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 4px rgba(0,200,83,0); }
}

/* === Sidebar Navigation Rail === */
[data-testid="stSidebar"] {
    background-color: #0D1117 !important;
    border-right: 1px solid #21262d !important;
    padding-top: 1rem !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
    max-width: none !important;
}
.sidebar-identity {
    padding: 0 1rem 0.75rem 1rem;
    border-bottom: 1px solid #21262d;
    margin-bottom: 0.5rem;
}
.sidebar-brand {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #e6edf3;
    line-height: 1.3;
    margin: 0;
}
.sidebar-sub {
    font-size: 0.65rem;
    color: #00d4ff;
    letter-spacing: 0.04em;
    margin-top: 0.15rem;
    display: block;
}
.sidebar-status {
    padding: 0.5rem 1rem;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #00c853;
    border-bottom: 1px solid #21262d;
}
.sidebar-section {
    margin-bottom: 0.25rem;
}
.sidebar-section-title {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #484f58;
    padding: 0.6rem 1rem 0.25rem 1rem;
}
.sidebar-nav-item {
    display: block;
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: #8b949e;
    text-decoration: none;
    border-left: 2px solid transparent;
    cursor: pointer;
    transition: all 0.15s ease;
    background: none;
    border-top: none;
    border-right: none;
    border-bottom: none;
    width: 100%;
    text-align: left;
}
.sidebar-nav-item:hover {
    color: #e6edf3;
    background-color: rgba(0,212,255,0.04);
}
.sidebar-nav-item.active {
    color: #00d4ff;
    border-left-color: #00d4ff;
    background-color: rgba(0,212,255,0.06);
    font-weight: 600;
}
.sidebar-nav-primary {
    color: #00d4ff;
    font-weight: 600;
}

/* === Hero / Command Center === */
.command-hero {
    padding: 2rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1.5rem;
}
.command-hero-grid {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 2rem;
    align-items: end;
}
.command-primary {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}
.command-secondary {
    text-align: right;
    padding-bottom: 0.5rem;
}
.command-risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.75rem;
    background-color: rgba(0,200,83,0.08);
    border: 1px solid rgba(0,200,83,0.2);
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #00c853;
    margin-top: 0.75rem;
    text-transform: uppercase;
}

/* === Telemetry Panel === */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0;
    border-top: 1px solid #21262d;
    border-left: 1px solid #21262d;
}
.telemetry-item {
    padding: 1rem 1.25rem;
    border-right: 1px solid #21262d;
    border-bottom: 1px solid #21262d;
}
.telemetry-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #484f58;
    margin-bottom: 0.35rem;
}
.telemetry-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e6edf3;
    font-family: SFMono-Regular, Consolas, monospace;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.telemetry-unit {
    font-size: 0.75rem;
    color: #8b949e;
    font-weight: 400;
    margin-left: 0.5rem;
}
.telemetry-status {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #00c853;
    flex-shrink: 0;
}

/* === Risk Scale === */
.risk-scale {
    position: relative;
    height: 6px;
    background: linear-gradient(90deg, #00c853 0%, #ffab00 50%, #ff4b4b 100%);
    border-radius: 3px;
    margin: 0.75rem 0;
}
.risk-scale-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #484f58;
    text-transform: uppercase;
    margin-top: 0.35rem;
}
.risk-marker {
    position: absolute;
    top: -4px;
    width: 2px;
    height: 14px;
    background-color: #e6edf3;
    border-radius: 1px;
    transform: translateX(-50%);
}
.risk-marker::after {
    content: attr(data-value);
    position: absolute;
    top: -18px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.65rem;
    font-weight: 700;
    color: #e6edf3;
    font-family: SFMono-Regular, Consolas, monospace;
    white-space: nowrap;
}

/* === Prediction Panel === */
.prediction-panel {
    padding: 1.5rem;
    background-color: #0D1117;
    border: 1px solid #21262d;
    margin: 1rem 0;
}
.prediction-value {
    font-size: clamp(3.5rem, 7vw, 6rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    color: #00c853;
    margin-bottom: 0.25rem;
}
.prediction-label {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #e6edf3;
}
.prediction-meta {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 0.5rem;
    font-family: SFMono-Regular, Consolas, monospace;
}
.prediction-decision {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #21262d;
    font-size: 0.85rem;
    color: #8b949e;
    line-height: 1.5;
}

/* === Feature Contribution === */
.feature-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #161c24;
}
.feature-row:last-child { border-bottom: none; }
.feature-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: #e6edf3;
    width: 160px;
    flex-shrink: 0;
    letter-spacing: -0.01em;
}
.feature-bar-track {
    flex: 1;
    height: 8px;
    background-color: #161c24;
    position: relative;
    overflow: hidden;
}
.feature-bar-fill {
    position: absolute;
    top: 0;
    height: 100%;
    background-color: #00d4ff;
    transition: width 0.4s ease;
}
.feature-bar-fill.negative {
    background-color: #00c853;
}
.feature-value {
    font-size: 0.8rem;
    font-weight: 700;
    font-family: SFMono-Regular, Consolas, monospace;
    width: 70px;
    text-align: right;
    flex-shrink: 0;
}
.feature-value.positive { color: #00d4ff; }
.feature-value.negative { color: #00c853; }

/* === Compact Inputs / Instrumentation === */
.instrument-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
.instrument-group {
    margin-bottom: 1.25rem;
}
.instrument-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #00d4ff;
    margin-bottom: 0.5rem;
    display: block;
}
.instrument-item {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid #161c24;
}
.instrument-item:last-child { border-bottom: none; }
.instrument-name {
    font-size: 0.75rem;
    color: #8b949e;
    width: 140px;
    flex-shrink: 0;
}
.instrument-number {
    font-size: 0.95rem;
    font-weight: 700;
    color: #e6edf3;
    font-family: SFMono-Regular, Consolas, monospace;
}
.instrument-unit {
    font-size: 0.75rem;
    color: #484f58;
    width: 50px;
    flex-shrink: 0;
}

/* === Action Control === */
.action-control {
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.7rem 1.5rem;
    background-color: #00d4ff;
    color: #080B0F;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
}
.action-control:hover {
    background-color: #33ddff;
    transform: translateY(-1px);
}
.action-control-arrow {
    font-size: 1.1rem;
    transition: transform 0.2s ease;
}
.action-control:hover .action-control-arrow {
    transform: translateX(3px);
}

/* === Report Center === */
.report-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 0;
    border-bottom: 1px solid #21262d;
}
.report-item:last-child { border-bottom: none; }
.report-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #e6edf3;
}
.report-meta {
    font-size: 0.7rem;
    color: #8b949e;
    margin-top: 0.15rem;
}
.report-action {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #00d4ff;
    text-transform: uppercase;
    white-space: nowrap;
}

/* === Metric Editorial Row === */
.metric-editorial-row {
    display: flex;
    gap: 2rem;
    padding: 1rem 0;
    border-bottom: 1px solid #21262d;
}
.metric-editorial-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}
.metric-editorial-value {
    font-size: clamp(1.75rem, 3vw, 2.25rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #e6edf3;
    line-height: 1.1;
}
.metric-editorial-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #484f58;
}

/* === Technical Metadata Strip === */
.meta-strip {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    padding: 0.5rem 0;
    border-top: 1px solid #161c24;
    border-bottom: 1px solid #161c24;
    margin: 1rem 0;
}
.meta-item {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
}

/* === Compact Alert === */
.system-alert {
    padding: 0.6rem 1rem;
    background-color: rgba(255,75,75,0.06);
    border: 1px solid rgba(255,75,75,0.15);
    border-left: 2px solid #ff4b4b;
    font-size: 0.8rem;
    color: #ff9b9b;
    margin-bottom: 0.75rem;
}
.system-alert.warning {
    background-color: rgba(255,171,0,0.06);
    border-color: rgba(255,171,0,0.15);
    border-left-color: #ffab00;
    color: #ffcc80;
}
.system-alert.success {
    background-color: rgba(0,200,83,0.06);
    border-color: rgba(0,200,83,0.15);
    border-left-color: #00c853;
    color: #69f0ae;
}

/* === Plotly chart overrides === */
.js-plotly-plot .plotly .modebar {
    top: 4px !important;
    right: 4px !important;
}
.js-plotly-plot .plotly .modebar-btn {
    opacity: 0.3 !important;
}
.js-plotly-plot .plotly .modebar-btn:hover {
    opacity: 1 !important;
}

/* === Streamlit widget overrides === */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 0.6rem 1rem;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom-color: #00d4ff !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1rem;
}

.stButton>button {
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.02em;
    transition: all 0.15s ease;
    border: 1px solid transparent;
}
.stButton>button:hover {
    transform: translateY(-1px);
}

.stSlider {
    padding-top: 0.5rem;
}

.stSelectbox {
    margin-bottom: 0.5rem;
}

.stAlert {
    border-radius: 4px;
    font-size: 0.85rem;
}

/* === Dataframe === */
.stDataFrame {
    font-size: 0.8rem;
}
.stDataFrame [data-testid="stDataFrameData"] {
    font-family: SFMono-Regular, Consolas, monospace;
    font-size: 0.75rem;
}

/* === Expander === */
.streamlit-expanderHeader {
    font-size: 0.8rem;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.01em;
}

/* === Horizontal Rule === */
hr {
    border-color: #21262d;
    margin: 1.5rem 0;
}

/* === Chart container === */
[data-testid="stPlotlyChart"] {
    border: 1px solid #21262d;
    border-radius: 4px;
    overflow: hidden;
}

/* === Download button text === */
[data-testid="stDownloadButton"] button {
    font-size: 0.75rem !important;
}

/* === Responsive === */
@media (max-width: 768px) {
    .command-hero-grid {
        grid-template-columns: 1fr;
    }
    .command-secondary {
        text-align: left;
    }
    .telemetry-grid {
        grid-template-columns: 1fr 1fr;
    }
    .metric-editorial-row {
        flex-wrap: wrap;
        gap: 1rem;
    }
    .meta-strip {
        gap: 1rem;
    }
}

/* === Legacy CSS class names (backward compatibility) === */
.metric-card { padding: 1rem 0; border-bottom: 1px solid #21262d; }
.metric-card .label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #484f58; margin-bottom: 0.35rem; }
.metric-card .value { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }
.prediction-card { padding: 1.5rem; background-color: #0D1117; border: 1px solid #21262d; margin: 1rem 0; }
.prediction-card .prob { font-size: clamp(3.5rem, 7vw, 6rem); font-weight: 800; letter-spacing: -0.04em; line-height: 1; }
.prediction-card .label { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.prediction-card .meta { font-size: 0.75rem; color: #8b949e; margin-top: 0.5rem; }
.section-title { font-size: 1.1rem; font-weight: 600; color: #e6edf3; margin-bottom: 0.75rem; margin-top: 1.75rem; letter-spacing: -0.01em; }
.hero { padding: 2rem 0; border-bottom: 1px solid #21262d; margin-bottom: 1.5rem; }
.status-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; }
.status-ready { background-color: #00c853; }
.status-warn { background-color: #ffab00; }
.status-error { background-color: #ff4b4b; }
.action-box { padding: 0.75rem 0; border-bottom: 1px solid #161c24; }
.action-box .title { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #00d4ff; margin-bottom: 0.35rem; }
.action-box .text { font-size: 0.8rem; color: #8b949e; line-height: 1.5; }
</style>
"""
