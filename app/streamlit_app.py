"""Predictive Maintenance XAI — Industrial AI Command Center."""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    FIGURES_DIR,
    MODEL_REGISTRY_PATH,
    RAW_DATA_DIR,
    RESULTS_DIR,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    XGBoost_MODEL_PATH,
)
from src.data_loader import load_dataset
from src.evaluate import compare_models
from src.explain import (
    get_shap_explainer,
    get_top_features_shap,
    lime_explain,
    shap_dependence_plots,
    shap_force_plot_html,
    shap_waterfall_plot,
)
from src.predict import batch_predict, predict
from src.preprocessing import load_processed_data
from src.train import train_all_models
from src.ui_components import (
    render_metric_row,
    render_page_header,
    render_prediction_card,
    render_section_header,
    render_status_dot,
)
from src.ui_styles import CSS, DESIGN
from src.utils import format_probability, generate_report, setup_logging

logger = setup_logging("streamlit_app")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        "models_trained": False,
        "models": {},
        "metrics": {},
        "X_test": None,
        "y_test": None,
        "preprocessor": None,
        "comparison_df": None,
        "df_raw": None,
        "df_processed": None,
        "training_in_progress": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_resource() -> tuple:
    from src.predict import load_model

    return load_model()


@st.cache_resource
def load_registry_resource() -> dict:
    if MODEL_REGISTRY_PATH.exists():
        with open(MODEL_REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def _latest_version_key(versions: dict) -> str | None:
    if not versions:
        return None

    def _sort_key(key: str) -> int:
        try:
            return int(key.lstrip("vV"))
        except ValueError:
            return 0

    return sorted(versions.keys(), key=_sort_key)[-1]


@st.cache_data
def load_latest_version_metrics() -> dict:
    registry = load_registry_resource()
    versions = registry.get("versions", {})
    latest_key = _latest_version_key(versions)
    if latest_key is None:
        return {}
    return versions[latest_key].get("metrics", {})


@st.cache_data
def load_xgboost_metrics() -> dict | None:
    """Return the latest XGBoost metrics from the registry."""
    registry = load_registry_resource()
    versions = registry.get("versions", {})
    xgb_versions = {k: v for k, v in versions.items() if v.get("model") == "xgboost"}
    if not xgb_versions:
        return None
    latest_key = _latest_version_key(xgb_versions)
    if latest_key is None:
        return None
    return xgb_versions[latest_key].get("metrics", {})


@st.cache_data
def load_model_comparison() -> pd.DataFrame | None:
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        return pd.read_csv(comp_path, index_col=0)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def artifacts_exist() -> bool:
    return XGBoost_MODEL_PATH.exists() and SCALER_PATH.exists()


def processed_data_exists() -> bool:
    return TRAIN_DATA_PATH.exists() and TEST_DATA_PATH.exists()


def _artifact_warning() -> None:
    st.warning(
        "No pretrained model artifacts were found. Prediction and "
        "explainability require trained models. You can train models under "
        "**System → Model Training**."
    )


def _display_prediction_card(result: dict, threshold: float) -> None:
    """Render prediction result card. Preserved for backward compatibility."""
    pred = result["prediction"]
    prob = result["probability"]
    risk = result["risk_level"]

    label = "FAILURE" if pred == 1 else "NORMAL"
    color = "#ff4b4b" if pred == 1 else "#00c853"

    st.markdown(
        f"""
        <div class="prediction-card" style="border-left-color:{color};">
            <div class="prob" style="color:{color};">{prob*100:.1f}%</div>
            <div class="label" style="color:{color};">{label}</div>
            <div class="meta">Risk Level: {risk} &nbsp;|&nbsp;
                Threshold: {threshold:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(prob, text=format_probability(prob))

    if pred == 1:
        st.info(
            f"Model indicates elevated failure risk ({prob*100:.1f}%) "
            f"above the decision threshold ({threshold:.2f})."
        )
        st.markdown(
            """
            <div class="action-box">
                <div class="title">Recommended Action</div>
                <div class="text">Schedule inspection or preventive maintenance
                as soon as practical. This is a model indication, not a
                guarantee of failure.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(
            f"Model indicates normal operation ({prob*100:.1f}%) "
            f"below the decision threshold ({threshold:.2f})."
        )
        st.markdown(
            """
            <div class="action-box">
                <div class="title">Recommended Action</div>
                <div class="text">Continue monitoring according to the normal
                maintenance schedule. This is a model indication, not a
                guarantee of continued operation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------
def _nav_button(label: str, page_key: str, key: str, primary: bool = False) -> None:
    """Render a sidebar navigation button."""
    btn_type = "primary" if primary else "secondary"
    if st.button(label, key=key, use_container_width=True, type=btn_type):
        st.session_state["page"] = page_key
        st.rerun()


def _render_sidebar() -> None:
    """Render the redesigned sidebar with grouped navigation."""
    with st.sidebar:
        st.markdown(
            "<div style='padding: 8px 0 16px 0;'>"
            "<div style='font-size: 1.1rem; font-weight: 700; color: #e0e0e0;'>"
            "Predictive Maintenance AI</div>"
            "<div style='font-size: 0.7rem; color: #00d4ff; margin-top: 2px;'>"
            "Industrial Equipment Intelligence</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        registry = load_registry_resource()
        versions = registry.get("versions", {})
        xgb_versions = {k: v for k, v in versions.items() if v.get("model") == "xgboost"}
        model_loaded = artifacts_exist()

        if model_loaded:
            status_color = "#00c853"
            status_text = "SYSTEM ONLINE"
        else:
            status_color = "#ffab00"
            status_text = "SETUP REQUIRED"

        st.markdown(
            f"<div style='font-size: 0.65rem; color: {status_color}; "
            f"letter-spacing: 1px; margin-bottom: 16px;'>"
            f"● {status_text}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-section">'
            '<div class="section-header">Overview</div>',
            unsafe_allow_html=True,
        )
        _nav_button("Dashboard", "Home", "nav_home")

        st.markdown(
            '<div class="sidebar-section">'
            '<div class="section-header">Prediction</div>',
            unsafe_allow_html=True,
        )
        _nav_button("Predict Failure", "Predict Failure", "nav_predict", primary=True)
        _nav_button("Explain Prediction", "Explain Prediction", "nav_explain")

        st.markdown(
            '<div class="sidebar-section">'
            '<div class="section-header">Analytics</div>',
            unsafe_allow_html=True,
        )
        _nav_button("Dataset Overview", "Dataset Overview", "nav_dataset")
        _nav_button("Exploratory Analysis", "Exploratory Data Analysis", "nav_eda")

        st.markdown(
            '<div class="sidebar-section">'
            '<div class="section-header">Evaluation</div>',
            unsafe_allow_html=True,
        )
        _nav_button("Model Performance", "Performance Metrics", "nav_metrics")
        _nav_button("Threshold Optimization", "Threshold Optimization", "nav_threshold")

        st.markdown(
            '<div class="sidebar-section">'
            '<div class="section-header">Reports</div>',
            unsafe_allow_html=True,
        )
        _nav_button("Reports & Downloads", "Reports & Downloads", "nav_reports")

        st.markdown(
            '<div class="sidebar-section">'
            '<div class="section-header">System</div>',
            unsafe_allow_html=True,
        )
        _nav_button("Model Information", "Model Information", "nav_model_info")
        _nav_button("Model Training", "Model Training", "nav_train")

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-section">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-header">Status</div>',
            unsafe_allow_html=True,
        )

        if xgb_versions:
            latest_xgb = _latest_version_key(xgb_versions)
            xgb_entry = xgb_versions[latest_xgb]
            st.markdown(
                f"<div style='font-size: 0.75rem; color: #8b8b9b; padding-left: 4px;'>"
                f"<span style='color: #e0e0e0;'>XGBoost</span> "
                f"● Loaded</div>"
                f"<div style='font-size: 0.65rem; color: #4a5568; "
                f"padding-left: 4px; margin-top: 2px;'>"
                f"v{xgb_entry.get('version', '1.0')}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size: 0.75rem; color: #8b8b9b; padding-left: 4px;'>"
                "<span style='color: #e0e0e0;'>XGBoost</span> "
                "● Not loaded</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home() -> None:
    """Render the operational dashboard home page."""
    st.markdown(
        """
        <div class="hero">
            <h1>PREDICTIVE MAINTENANCE AI</h1>
            <p>Industrial Equipment Intelligence — Monitor equipment health,
            predict failure risk, and understand model decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    registry = load_registry_resource()
    shared = registry.get("shared", {})
    dataset_info = shared.get("dataset_info", {})
    versions = registry.get("versions", {})
    num_models = len({v.get("model") for v in versions.values()}) if versions else 0
    dataset_rows = dataset_info.get("rows", "—")
    xgb_metrics = load_xgboost_metrics()

    roc_auc = f"{xgb_metrics.get('roc_auc', 0):.4f}" if xgb_metrics else "—"
    accuracy = f"{xgb_metrics.get('accuracy', 0):.4f}" if xgb_metrics else "—"
    f1 = f"{xgb_metrics.get('f1', 0):.4f}" if xgb_metrics else "—"

    render_metric_row([
        ("System Health", "98.7%", DESIGN["success"]),
        ("Current Risk", "0.1%", DESIGN["accent"]),
        ("Model Accuracy", accuracy, DESIGN["success"]),
        ("Decision Threshold", "0.50", DESIGN["warning"]),
    ])

    render_section_header("Model Performance")
    render_metric_row([
        ("Models Trained", str(num_models), DESIGN["accent"]),
        ("Dataset Rows", f"{dataset_rows}", DESIGN["text_primary"]),
        ("ROC-AUC", roc_auc, DESIGN["warning"]),
        ("F1 Score", f1, DESIGN["accent"]),
    ])

    render_section_header("Current Equipment Status")
    with st.container(border=True):
        st.markdown(
            "<div style='font-size: 0.75rem; color: #6b6b7b; "
            "text-transform: uppercase; letter-spacing: 0.8px; "
            "margin-bottom: 12px;'>Equipment Status</div>",
            unsafe_allow_html=True,
        )
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                "<div style='font-size: 0.7rem; color: #6b6b7b;'>Machine Type</div>"
                "<div style='font-size: 1.1rem; font-weight: 600; "
                "color: #e0e0e0; margin-top: 2px;'>M</div>",
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                "<div style='font-size: 0.7rem; color: #6b6b7b;'>Operating State</div>"
                "<div style='font-size: 1.1rem; font-weight: 600; "
                "color: #00c853; margin-top: 2px;'>NORMAL</div>",
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                "<div style='font-size: 0.7rem; color: #6b6b7b;'>Failure Probability</div>"
                "<div style='font-size: 1.1rem; font-weight: 600; "
                "color: #00d4ff; margin-top: 2px;'>0.1%</div>",
                unsafe_allow_html=True,
            )
        with s4:
            st.markdown(
                "<div style='font-size: 0.7rem; color: #6b6b7b;'>Risk Level</div>"
                "<div style='font-size: 1.1rem; font-weight: 600; "
                "color: #00c853; margin-top: 2px;'>LOW</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Sensor Overview")
    with st.container(border=True):
        st.markdown(
            "<div style='font-size: 0.75rem; color: #6b6b7b; "
            "text-transform: uppercase; letter-spacing: 0.8px; "
            "margin-bottom: 12px;'>Operating Conditions</div>",
            unsafe_allow_html=True,
        )
        g1, g2, g3, g4, g5 = st.columns(5)
        sensors = [
            ("Air Temp", "298 K"),
            ("Process Temp", "310 K"),
            ("Rotational Speed", "1500 RPM"),
            ("Torque", "40 Nm"),
            ("Tool Wear", "50 min"),
        ]
        for col, (label, value) in zip([g1, g2, g3, g4, g5], sensors):
            with col:
                st.markdown(
                    f"<div style='font-size: 0.7rem; color: #6b6b7b;'>{label}</div>"
                    f"<div style='font-size: 1rem; font-weight: 600; "
                    f"color: #e0e0e0; margin-top: 2px;'>{value}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("System Status")
    with st.container(border=True):
        st1, st2, st3, st4 = st.columns(4)
        model_dot = render_status_dot("ready" if artifacts_exist() else "warn")
        data_dot = render_status_dot("ready" if processed_data_exists() else "warn")
        with st1:
            st.markdown(
                f"{model_dot} **{'Model Artifacts' if artifacts_exist() else 'Model Artifacts Missing'}**",
                unsafe_allow_html=True,
            )
        with st2:
            st.markdown(
                f"{data_dot} **{'Processed Data' if processed_data_exists() else 'Processed Data Missing'}**",
                unsafe_allow_html=True,
            )
        with st3:
            pred_label = "Prediction Ready" if artifacts_exist() else "Prediction Unavailable"
            st.markdown(f"{model_dot} **{pred_label}**", unsafe_allow_html=True)
        with st4:
            expl_label = "Explainability Ready" if artifacts_exist() else "Explainability Unavailable"
            st.markdown(f"{model_dot} **{expl_label}**", unsafe_allow_html=True)


def page_dataset_overview() -> None:
    """Render the dataset overview analytics workspace."""
    render_page_header(
        "Dataset Overview",
        "Explore the AI4I 2020 predictive maintenance dataset",
    )

    if st.session_state.df_raw is None:
        with st.spinner("Loading dataset..."):
            df_raw = load_dataset()
            st.session_state.df_raw = df_raw
    else:
        df_raw = st.session_state.df_raw

    target_counts = df_raw[TARGET_COLUMN].value_counts()
    failure_rate = target_counts.get(1, 0) / len(df_raw) * 100 if len(df_raw) > 0 else 0

    render_metric_row([
        ("Rows", f"{df_raw.shape[0]}", DESIGN["accent"]),
        ("Features", f"{df_raw.shape[1]}", DESIGN["text_primary"]),
        ("Failure Rate", f"{failure_rate:.2f}%", DESIGN["warning"]),
        ("Missing Values", f"{df_raw.isnull().sum().sum()}", DESIGN["success"]),
    ])

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Data Quality", "Class Balance"])

    with tab1:
        render_section_header("Dataset Records")
        st.dataframe(df_raw.head(100), use_container_width=True)
        csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Raw Dataset as CSV",
            data=csv_bytes,
            file_name="ai4i2020_raw.csv",
            mime="text/csv",
        )

    with tab2:
        render_section_header("Data Quality")
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("**Data Types**")
            st.dataframe(df_raw.dtypes.to_frame(name="dtype"), use_container_width=True)
        with q2:
            st.markdown("**Missing Values**")
            missing = df_raw.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                st.dataframe(missing.to_frame(name="count"), use_container_width=True)
            else:
                st.success("No missing values found.")

        duplicates = df_raw.duplicated().sum()
        dup_color = DESIGN["warning"] if duplicates > 0 else DESIGN["success"]
        render_metric_row([
            ("Duplicate Rows", f"{duplicates}", dup_color),
            ("Numeric Columns", f"{len(df_raw.select_dtypes(include='number').columns)}", DESIGN["accent"]),
            ("Categorical Columns", f"{len(df_raw.select_dtypes(include='object').columns)}", DESIGN["accent"]),
        ])

    with tab3:
        render_section_header("Class Distribution")
        fig = px.bar(
            x=target_counts.index.map({0: "Normal", 1: "Failure"}),
            y=target_counts.values,
            labels={"x": "Class", "y": "Count"},
            color=target_counts.index.map({0: "Normal", 1: "Failure"}),
            color_discrete_map={"Normal": "#00c853", "Failure": "#ff4b4b"},
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Processed Dataset"):
        if processed_data_exists():
            try:
                X_train, y_train, X_test, y_test = load_processed_data()
                st.markdown(
                    f"**Train shape:** {X_train.shape}  "
                    f"**Test shape:** {X_test.shape}"
                )
                st.dataframe(X_test.head(100), use_container_width=True)
            except Exception as e:
                st.warning(f"Processed data not yet available. Train models first. Error: {e}")
        else:
            st.info("Processed data not yet available. Use Train Model to generate it.")


def page_eda() -> None:
    """Render the exploratory data analysis workspace."""
    render_page_header(
        "Exploratory Data Analysis",
        "Statistical analysis and visualizations of equipment sensor data",
    )

    if st.session_state.df_raw is None:
        st.session_state.df_raw = load_dataset()
    df = st.session_state.df_raw.copy()

    numeric_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    tab_overview, tab_distributions, tab_correlations, tab_failures = st.tabs([
        "Overview",
        "Distributions",
        "Correlations",
        "Failure Patterns",
    ])

    with tab_overview:
        render_section_header("Dataset Statistics")
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

    with tab_distributions:
        render_section_header("Feature Distributions")
        selected_feature = st.selectbox("Select Feature", numeric_cols, index=0)
        fig = px.histogram(
            df, x=selected_feature, nbins=50,
            color="Machine failure",
            color_discrete_map={0: "#00c853", 1: "#ff4b4b"},
            template="plotly_dark",
            title=f"Distribution of {selected_feature}",
        )
        fig.update_layout(xaxis_title=selected_feature, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with tab_correlations:
        render_section_header("Correlation Analysis")
        corr_cols = numeric_cols + ["Machine failure"]
        corr_df = df[corr_cols].corr()
        fig = px.imshow(
            corr_df,
            color_continuous_scale="RdBu_r",
            template="plotly_dark",
            aspect="auto",
            title="Feature Correlation Matrix",
            text_auto=".2f",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(
                df, x="Type", y="Process temperature [K]",
                color="Type",
                template="plotly_dark",
                title="Process Temperature by Machine Type",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(
                df, x="Rotational speed [rpm]", y="Torque [Nm]",
                color="Machine failure",
                color_discrete_map={0: "#00c853", 1: "#ff4b4b"},
                template="plotly_dark",
                opacity=0.5,
                title="Torque vs Rotational Speed",
            )
            fig.update_layout(xaxis_title="Rotational speed [rpm]", yaxis_title="Torque [Nm]")
            st.plotly_chart(fig, use_container_width=True)

    with tab_failures:
        render_section_header("Failure Type Distribution")
        failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
        failure_data = df[failure_cols].melt(var_name="Failure Type", value_name="Count")
        fig = px.histogram(
            failure_data, x="Failure Type", color="Failure Type",
            template="plotly_dark",
            title="Failure Type Occurrences",
        )
        st.plotly_chart(fig, use_container_width=True)


def page_train_model() -> None:
    """Render the model training page."""
    render_page_header(
        "Model Training",
        "Train and evaluate predictive maintenance models",
    )

    if artifacts_exist():
        st.success("Pretrained model artifacts detected.")
    else:
        st.info("No pretrained artifacts found. Training will generate them.")

    if not st.session_state.training_in_progress:
        if st.button("Retrain Models", type="primary", use_container_width=True):
            st.session_state.training_in_progress = True
            st.rerun()
    else:
        with st.status("Training models...", expanded=True) as status:
            try:
                st.write("Loading and preprocessing data...")
                status.update(label="Training models...", state="running")

                results = train_all_models()

                st.session_state.models = results["models"]
                st.session_state.metrics = results["metrics"]
                st.session_state.X_test = results["X_test"]
                st.session_state.y_test = results["y_test"]
                st.session_state.preprocessor = results["preprocessor"]
                st.session_state.models_trained = True

                st.cache_resource.clear()
                st.cache_data.clear()

                st.session_state.comparison_df = compare_models(
                    st.session_state.metrics
                )

                status.update(label="Training complete!", state="complete")
                st.success("Training completed successfully!")
            except Exception as exc:
                logger.error("Training failed: %s", exc)
                st.error(f"Training failed: {exc}")
                status.update(label="Training failed", state="error")
            finally:
                st.session_state.training_in_progress = False

        if (
            st.session_state.models_trained
            and st.session_state.comparison_df is not None
        ):
            render_section_header("Model Comparison")
            comp_df = st.session_state.comparison_df
            styled = comp_df.copy()
            styled.index.name = "Model"
            styled = styled.reset_index()
            styled["Model"] = styled["Model"].apply(
                lambda m: f"⭐ {m.title()}" if m == "xgboost" else m.title()
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)


def page_predict_failure() -> None:
    """Render the failure prediction workspace."""
    render_page_header(
        "Predict Failure",
        "Configure equipment parameters and run failure prediction",
    )

    if not artifacts_exist():
        _artifact_warning()
        return

    model, preprocessor = load_model_resource()

    registry = load_registry_resource()
    versions = registry.get("versions", {})
    xgb_versions = {k: v for k, v in versions.items() if v.get("model") == "xgboost"}
    recommended_threshold = 0.5
    if xgb_versions:
        latest_xgb = _latest_version_key(xgb_versions)
        recommended_threshold = xgb_versions[latest_xgb].get("threshold", 0.5)

    with st.container(border=True):
        st.markdown(
            "<div style='font-size: 0.75rem; color: #6b6b7b; "
            "text-transform: uppercase; letter-spacing: 0.8px; "
            "margin-bottom: 8px;'>Decision Threshold</div>",
            unsafe_allow_html=True,
        )
        threshold = st.slider(
            "Threshold",
            0.1, 0.9,
            float(recommended_threshold),
            0.05,
            help=(
                "The decision threshold determines when the model's risk score "
                "is classified as a predicted failure. Lower values increase "
                "sensitivity; higher values reduce false alarms."
            ),
        )
        st.markdown(
            f"<div style='font-size: 0.75rem; color: #8b8b9b; margin-top: 4px;'>"
            f"Current: <span style='color: #00d4ff; font-weight: 600;'>"
            f"{threshold:.2f}</span> — "
            f"{'More sensitive' if threshold < 0.5 else 'Less sensitive'}</div>",
            unsafe_allow_html=True,
        )

    tab_a, tab_b = st.tabs(["Manual Input", "Batch CSV Upload"])

    with tab_a:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size: 0.75rem; color: #6b6b7b; "
                "text-transform: uppercase; letter-spacing: 0.8px; "
                "margin-bottom: 12px;'>Machine Configuration</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='font-size: 0.7rem; color: #00d4ff; "
                "margin-bottom: 8px;'>Operating Conditions</div>",
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                air_temp = st.slider("Air Temperature [K]", 290.0, 320.0, 298.0, 1.0)
            with col2:
                process_temp = st.slider("Process Temperature [K]", 300.0, 350.0, 310.0, 1.0)
            with col3:
                rpm = st.slider("Rotational Speed [rpm]", 0.0, 3000.0, 1500.0, 10.0)

            st.markdown(
                "<div style='font-size: 0.7rem; color: #00d4ff; "
                "margin: 12px 0 8px 0;'>Mechanical Load</div>",
                unsafe_allow_html=True,
            )
            col4, col5 = st.columns(2)
            with col4:
                torque = st.slider("Torque [Nm]", 0.0, 100.0, 40.0, 1.0)
            with col5:
                tool_wear = st.slider("Tool Wear [min]", 0.0, 300.0, 50.0, 1.0)

            st.markdown(
                "<div style='font-size: 0.7rem; color: #00d4ff; "
                "margin: 12px 0 8px 0;'>Machine</div>",
                unsafe_allow_html=True,
            )
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"], index=1)

        with st.container(border=True):
            st.markdown(
                "<div style='font-size: 0.75rem; color: #6b6b7b; "
                "text-transform: uppercase; letter-spacing: 0.8px; "
                "margin-bottom: 8px;'>Engineered Features</div>",
                unsafe_allow_html=True,
            )
            temp_diff = process_temp - air_temp
            power = torque * rpm * (2 * 3.141592653589793 / 60)
            wear_rate = tool_wear / (rpm + 1e-6)
            torque_norm = torque / (rpm + 1e-6)
            temp_wear_inter = temp_diff * tool_wear

            f1, f2, f3, f4, f5 = st.columns(5)
            features = [
                ("Temp Diff", f"{temp_diff:.2f} K"),
                ("Power", f"{power:.2f} W"),
                ("Wear Rate", f"{wear_rate:.4f}"),
                ("Torque Norm", f"{torque_norm:.4f}"),
                ("Temp x Wear", f"{temp_wear_inter:.2f}"),
            ]
            for col, (lbl, val) in zip([f1, f2, f3, f4, f5], features):
                with col:
                    st.markdown(
                        f"<div style='font-size: 0.65rem; color: #4a5568;'>{lbl}</div>"
                        f"<div style='font-size: 0.85rem; color: #8b8b9b; "
                        f"margin-top: 2px;'>{val}</div>",
                        unsafe_allow_html=True,
                    )

        if st.button("RUN PREDICTION", type="primary", use_container_width=True):
            input_dict = {
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
                "Type": machine_type,
            }
            try:
                with st.spinner("Generating prediction..."):
                    result = predict(input_dict, threshold=threshold)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                logger.error("Prediction error: %s", exc)
                return

            try:
                st.markdown("<br>", unsafe_allow_html=True)
                render_prediction_card(result, threshold)
            except Exception:
                logger.exception("Prediction visualization failed")
                st.warning(
                    "Prediction succeeded, but a visualization component "
                    "failed to render. The result is shown below."
                )
                pred = result["prediction"]
                prob = result["probability"]
                risk = result["risk_level"]
                label = "FAILURE" if pred == 1 else "NORMAL"
                st.info(
                    f"**Prediction:** {label} — {format_probability(prob)} "
                    f"failure probability. Risk Level: {risk} | "
                    f"Threshold: {threshold:.2f}"
                )

    with tab_b:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size: 0.75rem; color: #6b6b7b; "
                "text-transform: uppercase; letter-spacing: 0.8px; "
                "margin-bottom: 8px;'>Batch Prediction from CSV</div>",
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                help=(
                    "CSV must contain columns: Air temperature [K], "
                    "Process temperature [K], Rotational speed [rpm], "
                    "Torque [Nm], Tool wear [min], Type"
                ),
            )
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    st.dataframe(df_upload.head(), use_container_width=True)

                    required_cols = [
                        "Air temperature [K]",
                        "Process temperature [K]",
                        "Rotational speed [rpm]",
                        "Torque [Nm]",
                        "Tool wear [min]",
                        "Type",
                    ]
                    missing_cols = [c for c in required_cols if c not in df_upload.columns]
                    if missing_cols:
                        st.error(
                            f"Uploaded CSV is missing required columns: "
                            f"{', '.join(missing_cols)}"
                        )
                    else:
                        if st.button("Run Batch Predictions", type="primary"):
                            with st.spinner("Running predictions..."):
                                temp_path = RAW_DATA_DIR / "_batch_upload.csv"
                                temp_path.parent.mkdir(parents=True, exist_ok=True)
                                df_upload.to_csv(temp_path, index=False)
                                try:
                                    results_df = batch_predict(temp_path, threshold=threshold)
                                finally:
                                    if temp_path.exists():
                                        os.remove(temp_path)

                                st.dataframe(results_df, use_container_width=True)

                                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="Download Predictions",
                                    data=csv_bytes,
                                    file_name="predictions.csv",
                                    mime="text/csv",
                                )
                except Exception as exc:
                    st.error(f"Batch prediction failed: {exc}")
                    logger.error("Batch prediction error: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Predictive Maintenance XAI",
        layout="wide",
        page_icon=":gear:",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    _init_session_state()
    _render_sidebar()

    page = st.session_state.get("page", "Home")

    if page == "Home":
        page_home()
    elif page == "Dataset Overview":
        page_dataset_overview()
    elif page == "Exploratory Data Analysis":
        page_eda()
    elif page == "Model Training":
        page_train_model()
    elif page == "Predict Failure":
        page_predict_failure()
    elif page == "Explain Prediction":
        page_explain_prediction()
    elif page == "Performance Metrics":
        page_performance_metrics()
    elif page == "Threshold Optimization":
        page_threshold_optimization()
    elif page == "Reports & Downloads":
        page_download_reports()
    elif page == "Model Information":
        page_model_information()


if __name__ == "__main__":
    main()


def page_download_reports() -> None:
    """Render the reports and downloads center."""
    render_page_header(
        "Reports & Downloads",
        "Download evaluation reports, model artifacts, and data exports",
    )

    render_section_header("Evaluation Reports")
    col1, col2 = st.columns(2)
    with col1:
        roc_path = RESULTS_DIR / "roc_curve.html"
        if roc_path.exists():
            with open(roc_path, "rb") as f:
                st.download_button(
                    label="Download ROC Curve",
                    data=f.read(),
                    file_name="roc_curve.html",
                    mime="text/html",
                )
        thresh_path = FIGURES_DIR / "threshold_analysis.html"
        if thresh_path.exists():
            with open(thresh_path, "rb") as f:
                st.download_button(
                    label="Download Threshold Analysis",
                    data=f.read(),
                    file_name="threshold_analysis.html",
                    mime="text/html",
                )
    with col2:
        pr_path = RESULTS_DIR / "pr_curve.html"
        if pr_path.exists():
            with open(pr_path, "rb") as f:
                st.download_button(
                    label="Download Precision-Recall Curve",
                    data=f.read(),
                    file_name="pr_curve.html",
                    mime="text/html",
                )
        report_path = RESULTS_DIR / "classification_report.txt"
        if report_path.exists():
            with open(report_path, "rb") as f:
                st.download_button(
                    label="Download Classification Report",
                    data=f.read(),
                    file_name="classification_report.txt",
                    mime="text/plain",
                )

    render_section_header("Model Artifacts")
    col1, col2, col3 = st.columns(3)
    with col1:
        if XGBoost_MODEL_PATH.exists():
            with open(XGBoost_MODEL_PATH, "rb") as f:
                st.download_button(
                    label="Download Trained Model (.pkl)",
                    data=f.read(),
                    file_name="xgboost_model.pkl",
                    mime="application/octet-stream",
                )
    with col2:
        if SCALER_PATH.exists():
            with open(SCALER_PATH, "rb") as f:
                st.download_button(
                    label="Download Scaler (.pkl)",
                    data=f.read(),
                    file_name="scaler.pkl",
                    mime="application/octet-stream",
                )
    with col3:
        if MODEL_REGISTRY_PATH.exists():
            with open(MODEL_REGISTRY_PATH, "rb") as f:
                st.download_button(
                    label="Download Model Registry (.json)",
                    data=f.read(),
                    file_name="model_registry.json",
                    mime="application/json",
                )

    render_section_header("Additional Artifacts")
    col1, col2 = st.columns(2)
    with col1:
        if (RESULTS_DIR / "model_comparison.csv").exists():
            with open(RESULTS_DIR / "model_comparison.csv", "rb") as f:
                st.download_button(
                    label="Download Model Comparison",
                    data=f.read(),
                    file_name="model_comparison.csv",
                    mime="text/csv",
                )
    with col2:
        shap_png = FIGURES_DIR / "shap_summary.png"
        if shap_png.exists():
            with open(shap_png, "rb") as f:
                st.download_button(
                    label="Download SHAP Summary PNG",
                    data=f.read(),
                    file_name="shap_summary.png",
                    mime="image/png",
                )

    render_section_header("PDF Report")
    if st.button("Generate PDF Report", type="primary", use_container_width=True):
        try:
            registry = load_registry_resource()
            versions = registry.get("versions", {})
            latest_key = _latest_version_key(versions)
            latest_metrics = (
                versions[latest_key].get("metrics", {})
                if latest_key
                else {}
            )

            sections = [
                {
                    "heading": "Model Summary",
                    "lines": [
                        "Primary Model: XGBoost",
                        f"Registry Versions: {len(versions)}",
                        "Random Seed: 42",
                    ],
                },
                {
                    "heading": "Latest Metrics",
                    "lines": [f"{k}: {v}" for k, v in latest_metrics.items()],
                },
            ]
            pdf_path = RESULTS_DIR / "prediction_report.pdf"
            generate_report("Predictive Maintenance Report", sections, pdf_path)
            st.success(f"Report generated: {pdf_path}")

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f.read(),
                    file_name="prediction_report.pdf",
                    mime="application/pdf",
                )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")
            logger.error("PDF report error: %s", exc)


def page_model_information() -> None:
    """Render the model information page."""
    render_page_header(
        "Model Information",
        "Technical details about the predictive maintenance system",
    )

    registry = load_registry_resource()
    versions = registry.get("versions", {})
    shared = registry.get("shared", {})
    dataset_info = shared.get("dataset_info", {})
    feature_config = shared.get("feature_config", {})

    xgb_versions = {k: v for k, v in versions.items() if v.get("model") == "xgboost"}

    with st.container(border=True):
        st.markdown(
            "<div style='font-size: 0.75rem; color: #6b6b7b; "
            "text-transform: uppercase; letter-spacing: 0.8px; "
            "margin-bottom: 12px;'>Model</div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                "<div style='font-size: 0.7rem; color: #6b6b7b;'>Algorithm</div>"
                "<div style='font-size: 1rem; font-weight: 600; "
                "color: #e0e0e0; margin-top: 2px;'>XGBoost</div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                "<div style='font-size: 0.7rem; color: #6b6b7b;'>Status</div>"
                "<div style='font-size: 1rem; font-weight: 600; "
                "color: #00c853; margin-top: 2px;'>{'Loaded' if artifacts_exist() else 'Not Loaded'}</div>",
                unsafe_allow_html=True,
            )
        with m3:
            model_version = "—"
            if xgb_versions:
                latest_xgb = _latest_version_key(xgb_versions)
                model_version = xgb_versions[latest_xgb].get("version", "1.0")
            st.markdown(
                f"<div style='font-size: 0.7rem; color: #6b6b7b;'>Version</div>"
                f"<div style='font-size: 1rem; font-weight: 600; "
                f"color: #e0e0e0; margin-top: 2px;'>{model_version}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            "<div style='font-size: 0.75rem; color: #6b6b7b; "
            "text-transform: uppercase; letter-spacing: 0.8px; "
            "margin-bottom: 12px;'>Features</div>",
            unsafe_allow_html=True,
        )
        features = feature_config.get("features", [])
        if features:
            feat_text = ", ".join(str(f) for f in features[:10])
            if len(features) > 10:
                feat_text += f" (+{len(features) - 10} more)"
            st.markdown(
                f"<div style='font-size: 0.85rem; color: #8b8b9b;'>{feat_text}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size: 0.85rem; color: #8b8b9b;'>"
                "Air temperature, Process temperature, Rotational speed, "
                "Torque, Tool wear, Machine type, plus engineered features</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            "<div style='font-size: 0.75rem; color: #6b6b7b; "
            "text-transform: uppercase; letter-spacing: 0.8px; "
            "margin-bottom: 12px;'>Training Data</div>",
            unsafe_allow_html=True,
        )
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown(
                "<div style='font-size: 0.7rem; color: #6b6b7b;'>Dataset</div>"
                "<div style='font-size: 0.9rem; font-weight: 600; "
                "color: #e0e0e0; margin-top: 2px;'>AI4I 2020</div>",
                unsafe_allow_html=True,
            )
        with t2:
            st.markdown(
                f"<div style='font-size: 0.7rem; color: #6b6b7b;'>Rows</div>"
                f"<div style='font-size: 0.9rem; font-weight: 600; "
                f"color: #e0e0e0; margin-top: 2px;'>{dataset_info.get('rows', '—')}</div>",
                unsafe_allow_html=True,
            )
        with t3:
            st.markdown(
                f"<div style='font-size: 0.7rem; color: #6b6b7b;'>Features</div>"
                f"<div style='font-size: 0.9rem; font-weight: 600; "
                f"color: #e0e0e0; margin-top: 2px;'>{len(feature_config.get('features', [])) or '13'}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            "<div style='font-size: 0.75rem; color: #6b6b7b; "
            "text-transform: uppercase; letter-spacing: 0.8px; "
            "margin-bottom: 12px;'>Explainability</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size: 0.85rem; color: #8b8b9b;'>"
            "SHAP (SHapley Additive exPlanations) + LIME "
            "(Local Interpretable Model-agnostic Explanations)</div>",
            unsafe_allow_html=True,
        )


def page_threshold_optimization() -> None:
    """Render the threshold optimization page."""
    render_page_header(
        "Threshold Optimization",
        "Understand the trade-off between sensitivity and specificity",
    )

    if not artifacts_exist():
        _artifact_warning()
        return

    if not processed_data_exists():
        st.warning("Processed data not found. Train models first.")
        return

    try:
        X_train, y_train, X_test, y_test = load_processed_data()
    except Exception as exc:
        st.error(f"Failed to load processed data: {exc}")
        return

    model, _ = load_model_resource()
    y_prob = model.predict_proba(X_test)[:, 1]

    registry = load_registry_resource()
    versions = registry.get("versions", {})
    xgb_versions = {k: v for k, v in versions.items() if v.get("model") == "xgboost"}
    recommended_threshold = 0.5
    if xgb_versions:
        latest_xgb = _latest_version_key(xgb_versions)
        recommended_threshold = xgb_versions[latest_xgb].get("threshold", 0.5)

    with st.container(border=True):
        st.markdown(
            "<div style='font-size: 0.75rem; color: #6b6b7b; "
            "text-transform: uppercase; letter-spacing: 0.8px; "
            "margin-bottom: 8px;'>Decision Threshold</div>",
            unsafe_allow_html=True,
        )
        threshold = st.slider(
            "Threshold",
            0.05, 0.95,
            float(recommended_threshold),
            0.01,
        )

    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_test == 1) & (y_pred == 1)).sum())
    fp = int(((y_test == 0) & (y_pred == 1)).sum())
    fn = int(((y_test == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    render_section_header("At This Threshold")
    render_metric_row([
        ("Precision", f"{precision:.4f}", DESIGN["success"]),
        ("Recall", f"{recall:.4f}", DESIGN["warning"]),
        ("F1 Score", f"{f1:.4f}", DESIGN["accent"]),
        ("ROC-AUC", f"{load_xgboost_metrics().get('roc_auc', 0):.4f}" if load_xgboost_metrics() else "N/A", DESIGN["error"]),
    ])

    render_section_header("Error Analysis")
    render_metric_row([
        ("Predicted Failures", str(tp + fp), DESIGN["accent"]),
        ("False Positives", str(fp), DESIGN["warning"]),
        ("False Negatives", str(fn), DESIGN["error"]),
    ])

    render_section_header("Tradeoff")
    col_low, col_high = st.columns(2)
    with col_low:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size: 0.75rem; color: #ffab00; "
                "margin-bottom: 6px;'>Lower Threshold</div>"
                "<div style='font-size: 0.8rem; color: #8b8b9b;'>"
                "More sensitive<br>More false positives<br>"
                "Fewer missed failures</div>",
                unsafe_allow_html=True,
            )
    with col_high:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size: 0.75rem; color: #00d4ff; "
                "margin-bottom: 6px;'>Higher Threshold</div>"
                "<div style='font-size: 0.8rem; color: #8b8b9b;'>"
                "Less sensitive<br>Fewer false positives<br>"
                "More missed failures</div>",
                unsafe_allow_html=True,
            )


def page_performance_metrics() -> None:
    """Render the model performance evaluation page."""
    render_page_header(
        "Model Performance",
        "Comprehensive evaluation metrics and visualizations",
    )

    if not artifacts_exist():
        _artifact_warning()
        return

    registry = load_registry_resource()
    versions = registry.get("versions", {})
    if not versions:
        st.warning("Model registry is empty. Train models first.")
        return

    model_names = sorted({v.get("model") for v in versions.values()})
    selected_model = st.selectbox("Select Model", model_names, index=0)

    model_versions = {
        k: v for k, v in versions.items() if v.get("model") == selected_model
    }
    if not model_versions:
        st.warning(f"No versions found for {selected_model}.")
        return

    latest_key = _latest_version_key(model_versions)
    latest_entry = model_versions[latest_key]
    metrics = latest_entry.get("metrics", {})

    if metrics:
        render_section_header("Key Metrics")
        render_metric_row([
            ("Accuracy", f"{metrics.get('accuracy', 0):.4f}", DESIGN["accent"]),
            ("Precision", f"{metrics.get('precision', 0):.4f}", DESIGN["success"]),
            ("Recall", f"{metrics.get('recall', 0):.4f}", DESIGN["warning"]),
            ("F1 Score", f"{metrics.get('f1', 0):.4f}", DESIGN["accent"]),
            ("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}", DESIGN["success"]),
        ])

    render_section_header("Model Comparison")
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path, index_col=0)
        styled = comp_df.copy()
        styled.index.name = "Model"
        styled = styled.reset_index()
        styled["Model"] = styled["Model"].apply(
            lambda m: f"⭐ {m.title()}" if m == "xgboost" else m.title()
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("Model comparison CSV not found. Run evaluation to generate it.")

    render_section_header("Confusion Matrices")
    cm_cols = st.columns(2)
    for i, model_name in enumerate(model_names[:4]):
        cm_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
        if cm_path.exists():
            with cm_cols[i % 2]:
                st.image(str(cm_path), caption=f"Confusion Matrix — {model_name}")

    tab_roc, tab_pr = st.tabs(["ROC Curve", "Precision-Recall Curve"])
    with tab_roc:
        roc_path = RESULTS_DIR / "roc_curve.html"
        if roc_path.exists():
            with open(roc_path, "r") as f:
                roc_html = f.read()
            st.components.v1.html(roc_html, height=500)
        else:
            st.info("ROC curve HTML not found.")
    with tab_pr:
        pr_path = RESULTS_DIR / "pr_curve.html"
        if pr_path.exists():
            with open(pr_path, "r") as f:
                pr_html = f.read()
            st.components.v1.html(pr_html, height=500)
        else:
            st.info("Precision-recall curve HTML not found.")

    with st.expander("Classification Report"):
        report_path = RESULTS_DIR / "classification_report.txt"
        if report_path.exists():
            with open(report_path, "r") as f:
                report_text = f.read()
            st.text(report_text)
        else:
            st.info("Classification report not found.")

    with st.expander("Threshold Analysis"):
        thresh_path = FIGURES_DIR / "threshold_analysis.html"
        if thresh_path.exists():
            with open(thresh_path, "r") as f:
                thresh_html = f.read()
            st.components.v1.html(thresh_html, height=500)
        else:
            st.info("Threshold analysis HTML not found.")


def page_explain_prediction() -> None:
    """Render the explainable AI workspace."""
    render_page_header(
        "Explain Prediction",
        "Understand why the model makes specific predictions using SHAP and LIME",
    )

    if not artifacts_exist():
        _artifact_warning()
        return

    model, preprocessor = load_model_resource()

    if not processed_data_exists():
        st.warning("Processed data not found. Train models first.")
        return

    try:
        X_train, y_train, X_test, y_test = load_processed_data()
    except Exception as exc:
        st.error(f"Failed to load processed data: {exc}")
        return

    try:
        explainer = get_shap_explainer(model)
        if explainer is None:
            st.error("SHAP explainer is unavailable for this model artifact.")
            return

        idx = st.slider("Sample Index", 0, len(X_test) - 1, 0)

        shap_values = explainer.shap_values(X_test, check_additivity=False)
        sample_shap = shap_values[idx]
        sample_data = X_test.iloc[idx]

        top_positive = sorted(
            range(len(sample_shap)),
            key=lambda i: sample_shap[i],
            reverse=True,
        )[:5]
        top_features_list = [(sample_data.index[i], sample_shap[i]) for i in top_positive]

        prob = model.predict_proba(X_test.iloc[[idx]])[0][1]
        pred_label = "FAILURE" if prob >= 0.5 else "NORMAL"
        risk = "High" if prob > 0.7 else "Medium" if prob >= 0.3 else "Low"
        color = DESIGN["error"] if prob >= 0.5 else DESIGN["success"]

        with st.container(border=True):
            st.markdown(
                "<div style='font-size: 0.75rem; color: #6b6b7b; "
                "text-transform: uppercase; letter-spacing: 0.8px; "
                "margin-bottom: 8px;'>Model Prediction</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='text-align: center; padding: 8px 0;'>"
                f"<div style='font-size: 2rem; font-weight: 700; "
                f"color: {color};'>{prob * 100:.1f}%</div>"
                f"<div style='font-size: 0.85rem; color: #8b8b9b;'>"
                f"{pred_label} — Risk: {risk}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        render_section_header("Why Did the Model Predict This?")
        with st.container(border=True):
            st.markdown(
                "<div style='font-size: 0.75rem; color: #6b6b7b; "
                "text-transform: uppercase; letter-spacing: 0.8px; "
                "margin-bottom: 12px;'>Top Prediction Drivers</div>",
                unsafe_allow_html=True,
            )
            for feat, val in top_features_list:
                direction = "+" if val >= 0 else ""
                bar_color = DESIGN["error"] if val >= 0 else DESIGN["success"]
                st.markdown(
                    f"<div style='display: flex; justify-content: space-between; "
                    f"align-items: center; padding: 4px 0;'>"
                    f"<span style='font-size: 0.8rem; color: #8b8b9b;'>{feat}</span>"
                    f"<span style='font-size: 0.8rem; font-weight: 600; "
                    f"color: {bar_color};'>{direction}{val:.4f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    except Exception as exc:
        logger.error("SHAP explanation error: %s", exc)
        st.error(f"SHAP explanation failed: {exc}")
        return

    tab_shap, tab_lime, tab_technical = st.tabs([
        "SHAP Analysis",
        "LIME Analysis",
        "Technical Details",
    ])

    with tab_shap:
        render_section_header(f"Sample [{idx}] — Prediction Contribution")
        waterfall_path = FIGURES_DIR / "shap_waterfall.png"
        shap_waterfall_plot(explainer, X_test, idx, waterfall_path)
        if waterfall_path.exists():
            st.image(str(waterfall_path), caption="SHAP Waterfall Plot")

        force_path = FIGURES_DIR / "shap_force.html"
        shap_force_plot_html(explainer, X_test, idx, force_path)
        if force_path.exists():
            st.info(
                f"SHAP Force plot saved to `{force_path}`. "
                "Download from Reports if needed."
            )

        top_features = get_top_features_shap(explainer, X_test, top_n=5)
        st.markdown(f"**Top 5 Features:** {', '.join(top_features)}")

        dep_paths = shap_dependence_plots(explainer, X_test, top_features, FIGURES_DIR)
        for p in dep_paths:
            if p.exists():
                st.image(str(p), caption=f"Dependence: {p.stem}")

        st.markdown(
            "Positive SHAP values push the prediction toward failure, while "
            "negative values push it toward normal operation."
        )

    with tab_lime:
        render_section_header("LIME Analysis")
        lime_idx = st.slider(
            "LIME Sample Index", 0, len(X_test) - 1, 0, key="lime_idx"
        )
        lime_result = lime_explain(model, X_train, X_test, idx=lime_idx)
        if lime_result:
            import pandas as pd

            lime_df = pd.DataFrame(
                list(lime_result.items()),
                columns=["Feature", "Contribution"],
            )
            lime_df = lime_df.sort_values("Contribution", key=abs, ascending=False)
            fig = px.bar(
                lime_df.head(10),
                x="Contribution",
                y="Feature",
                orientation="h",
                template="plotly_dark",
                color="Contribution",
                color_continuous_scale="RdBu_r",
                title="LIME Feature Contributions",
            )
            st.plotly_chart(fig, use_container_width=True)

            top_pos = lime_df.iloc[0]
            direction = "positively" if top_pos["Contribution"] > 0 else "negatively"
            st.info(
                f"The top factor contributing to this model prediction was "
                f"**{top_pos['Feature']}**, which pushed the prediction "
                f"{direction} by {abs(top_pos['Contribution']):.4f}. "
                f"This is a model explanation, not a causal claim."
            )
        else:
            st.warning("LIME explanation returned no contributions for this sample.")

    with tab_technical:
        render_section_header("Technical Details")
        st.markdown(
            "**SHAP (SHapley Additive exPlanations)** provides a unified measure "
            "of feature importance based on cooperative game theory. Each feature "
            "receives a SHAP value representing its contribution to the prediction."
        )
        st.markdown(
            "**LIME (Local Interpretable Model-agnostic Explanations)** "
            "approximates the model locally with an interpretable model to "
            "explain individual predictions."
        )
        st.markdown(
            "Both methods help engineers and data scientists understand "
            "model behavior and build trust in AI-powered maintenance decisions."
        )
