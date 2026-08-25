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
from src.utils import card_html, format_probability, generate_report, setup_logging

logger = setup_logging("streamlit_app")


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def _get_css() -> str:
    return """
<style>
    .stApp { background-color: #0e1117; }
    .metric-card { background-color: #1c1e26; border-radius: 12px;
        padding: 16px; margin: 8px; text-align: center;
        border-left: 4px solid #00d4ff; }
    .metric-card .label { font-size: 12px; color: #8b8b9b; margin-bottom: 4px; }
    .metric-card .value { font-size: 28px; font-weight: bold; color: #00d4ff; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { color: #e0e0e0; }
    .stDataFrame { background-color: #1c1e26; color: #e0e0e0; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1c1e26; color: #e0e0e0;
        border-radius: 8px; }
    .stTabs [aria-selected="true"] { background-color: #00d4ff; color: #0e1117; }

    /* Hero */
    .hero { background: linear-gradient(135deg, #1c1e26 0%, #0e1117 100%);
        border-radius: 16px; padding: 32px; margin-bottom: 24px;
        border: 1px solid #2a2d3a; }
    .hero h1 { color: #ffffff; font-size: 2.2rem; font-weight: 700;
        margin-bottom: 8px; }
    .hero p { color: #b0b0b0; font-size: 1.05rem; line-height: 1.6;
        margin-bottom: 20px; }

    /* Section title */
    .section-title { color: #00d4ff; font-size: 1.05rem; font-weight: 600;
        margin-top: 24px; margin-bottom: 12px; text-transform: uppercase;
        letter-spacing: 0.5px; }

    /* Architecture */
    .arch-node { background-color: #1c1e26; border: 1px solid #2a2d3a;
        border-radius: 10px; padding: 14px; text-align: center; margin: 6px; }
    .arch-node .title { font-size: 13px; font-weight: 600; color: #00d4ff; }
    .arch-node .desc { font-size: 11px; color: #8b8b9b; margin-top: 4px; }
    .arch-arrow { text-align: center; color: #00d4ff; font-size: 20px; margin: 4px 0; }

    /* Highlights */
    .highlight-item { display: flex; align-items: center; gap: 8px;
        padding: 6px 0; color: #e0e0e0; font-size: 0.95rem; }
    .highlight-item .icon { color: #00c853; font-weight: bold; }

    /* Status dots */
    .status-dot { display: inline-block; width: 8px; height: 8px;
        border-radius: 50%; margin-right: 6px; }
    .status-ready { background-color: #00c853; }
    .status-warn { background-color: #ffab00; }
    .status-error { background-color: #ff4b4b; }

    /* Prediction card */
    .prediction-card { background-color: #1c1e26; border-radius: 14px;
        padding: 24px; margin: 16px 0; border-left: 5px solid #00d4ff;
        text-align: center; }
    .prediction-card .prob { font-size: 3rem; font-weight: 700; }
    .prediction-card .label { font-size: 1.2rem; font-weight: 600;
        color: #e0e0e0; margin-top: 4px; }
    .prediction-card .meta { font-size: 0.9rem; color: #8b8b9b; margin-top: 8px; }

    /* Action box */
    .action-box { background-color: #1c1e26; border-radius: 10px; padding: 16px;
        margin-top: 12px; border: 1px solid #2a2d3a; }
    .action-box .title { font-size: 0.9rem; font-weight: 600; color: #00d4ff;
        margin-bottom: 6px; }
    .action-box .text { font-size: 0.85rem; color: #b0b0b0; line-height: 1.5; }

    /* Sidebar section */
    .sidebar-section { margin-bottom: 16px; }
    .sidebar-section .section-header { font-size: 0.7rem; font-weight: 700;
        color: #6b6b7b; text-transform: uppercase; letter-spacing: 0.8px;
        margin-bottom: 6px; padding-left: 4px; }
    .stAlert { border-radius: 10px; }
</style>
"""


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
        "**Administration → Model Training**."
    )


def _render_architecture() -> None:
    """Render the ML architecture flow using native Streamlit components."""
    steps = [
        ("Raw Dataset", "AI4I 2020 dataset with sensor readings"),
        ("Data Loading", "Load and validate the raw CSV data"),
        ("Preprocessing", "Clean, encode, scale, and handle missing values"),
        ("Feature Engineering", "Create derived features like temperature diff, power, etc."),
        ("Model Prediction", "XGBoost classifier predicts failure probability"),
        ("Failure Probability", "Model outputs probability of equipment failure"),
        ("Decision Threshold", "Compare probability to threshold to decide risk"),
        ("Risk Classification", "Classify as Low, Medium, or High risk"),
        ("SHAP Explanation", "Global and local explainability with SHAP values"),
        ("LIME Explanation", "Local interpretable model-agnostic explanations"),
    ]

    st.markdown("### ML Architecture & Workflow")

    for i, (title, desc) in enumerate(steps):
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)

        if i < len(steps) - 1:
            st.markdown(
                "<div style='text-align: center; color: #00d4ff; "
                "font-size: 24px; margin: 8px 0;'>↓</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home() -> None:
    # Hero section
    st.markdown(
        """
        <div class="hero">
            <h1>Predictive Maintenance of Industrial Equipment</h1>
            <p>ML-powered machine failure prediction with Explainable AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Primary CTA
    if st.button(
        "⚡ Try Failure Prediction Now",
        type="primary",
        use_container_width=True,
    ):
        st.session_state["page"] = "Predict Failure"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Description
    st.markdown(
        """
        This application predicts industrial equipment failure using sensor data
        from the AI4I 2020 dataset. An XGBoost classifier produces failure
        probabilities, which are compared against an adjustable decision threshold
        to classify risk. Every prediction is explained using SHAP and LIME so you
        can understand *why* the model flags a machine as high-risk.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Metric cards row
    registry = load_registry_resource()
    shared = registry.get("shared", {})
    dataset_info = shared.get("dataset_info", {})
    versions = registry.get("versions", {})
    num_models = len({v.get("model") for v in versions.values()}) if versions else 0
    dataset_rows = dataset_info.get("rows", "Available after training")
    xgb_metrics = load_xgboost_metrics()

    # Prepare metrics
    roc_auc = (
        f"{xgb_metrics.get('roc_auc', 0):.4f}"
        if xgb_metrics
        else "Available after training"
    )
    precision = (
        f"{xgb_metrics.get('precision', 0):.4f}"
        if xgb_metrics
        else "Available after training"
    )
    recall = (
        f"{xgb_metrics.get('recall', 0):.4f}"
        if xgb_metrics
        else "Available after training"
    )
    f1 = (
        f"{xgb_metrics.get('f1', 0):.4f}"
        if xgb_metrics
        else "Available after training"
    )
    explainability = (
        "SHAP + LIME" if artifacts_exist() else "Available after training"
    )

    # Create 4 columns for the metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            card_html("Models", str(num_models), "#00d4ff"),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            card_html("Dataset Size", f"{dataset_rows} rows", "#00c853"),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            card_html("ROC-AUC", roc_auc, "#ffab00"),
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            card_html("Explainability", explainability, "#00d4ff"),
            unsafe_allow_html=True,
        )

    # Second row of metric cards
    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.markdown(
            card_html("Precision", precision, "#00c853"),
            unsafe_allow_html=True,
        )
    with m6:
        st.markdown(
            card_html("Recall", recall, "#ffab00"),
            unsafe_allow_html=True,
        )
    with m7:
        st.markdown(
            card_html("F1 Score", f1, "#ff4b4b"),
            unsafe_allow_html=True,
        )
    with m8:
        # Primary model indicator
        primary_model = "XGBoost" if artifacts_exist() else "Available after training"
        st.markdown(
            card_html("Primary Model", primary_model, "#00d4ff"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture section
    st.markdown(
        '<div class="section-title">System Architecture</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        _render_architecture()

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown(
        '<div class="section-title">How It Works</div>',
        unsafe_allow_html=True,
    )
    steps = [
        ("01", "Sensor Data",
         "Industrial equipment measurements are loaded from the AI4I 2020 dataset."),
        ("02", "Preprocessing",
         "Data is cleaned, encoded, scaled, and transformed for modeling."),
        ("03", "Feature Engineering",
         "Derived features capture physical relationships in the sensor readings."),
        ("04", "Failure Prediction",
         "The trained XGBoost model produces a failure probability."),
        ("05", "Risk Classification",
         "Probability is compared to decision threshold to classify risk level."),
        ("06", "Explainable AI",
         "SHAP and LIME provide interpretable explanations for each prediction."),
    ]
    for num, title, desc in steps:
        c1, c2 = st.columns([1, 5])
        with c1:
            st.markdown(
                f"<div style='font-size:1.4rem;font-weight:700;color:#00d4ff;'>{num}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(f"**{title}**  {desc}")
        st.markdown(
            "<hr style='margin:8px 0;border-color:#2a2d3a;'>",
            unsafe_allow_html=True,
        )

    # Project highlights
    st.markdown(
        '<div class="section-title">Key Features</div>',
        unsafe_allow_html=True,
    )
    highlights = [
        "End-to-end predictive maintenance pipeline",
        "Production-ready ML with XGBoost",
        "Interactive prediction with threshold adjustment",
        "Comprehensive explainability (SHAP & LIME)",
        "Batch prediction for industrial IoT datasets",
        "Automated model evaluation and comparison",
        "Professional reporting and model management",
        "Deployment-ready with Docker and CI/CD",
    ]
    cols = st.columns(2)
    for i, item in enumerate(highlights):
        with cols[i % 2]:
            st.markdown(
                f"<div class='highlight-item'><span class='icon'>✓</span>{item}</div>",
                unsafe_allow_html=True,
            )

    # System status
    st.markdown(
        '<div class="section-title">System Status</div>',
        unsafe_allow_html=True,
    )
    status1, status2, status3, status4 = st.columns(4)
    with status1:
        dot = (
            '<span class="status-dot status-ready"></span>'
            if artifacts_exist()
            else '<span class="status-dot status-warn"></span>'
        )
        label = "Model Artifacts" if artifacts_exist() else "Model Artifacts Missing"
        st.markdown(f"{dot} **{label}**", unsafe_allow_html=True)
    with status2:
        dot = (
            '<span class="status-dot status-ready"></span>'
            if processed_data_exists()
            else '<span class="status-dot status-warn"></span>'
        )
        label = "Processed Data" if processed_data_exists() else "Processed Data Missing"
        st.markdown(f"{dot} **{label}**", unsafe_allow_html=True)
    with status3:
        dot = (
            '<span class="status-dot status-ready"></span>'
            if artifacts_exist()
            else '<span class="status-dot status-warn"></span>'
        )
        label = "Prediction Ready" if artifacts_exist() else "Prediction Unavailable"
        st.markdown(f"{dot} **{label}**", unsafe_allow_html=True)
    with status4:
        dot = (
            '<span class="status-dot status-ready"></span>'
            if artifacts_exist()
            else '<span class="status-dot status-warn"></span>'
        )
        label = "Explainability Ready" if artifacts_exist() else "Explainability Unavailable"
        st.markdown(f"{dot} **{label}**", unsafe_allow_html=True)


def page_dataset_overview() -> None:
    st.title("Dataset Overview")

    if st.session_state.df_raw is None:
        with st.spinner("Loading dataset..."):
            df_raw = load_dataset()
            st.session_state.df_raw = df_raw
    else:
        df_raw = st.session_state.df_raw

    tab1, tab2, tab3 = st.tabs(
        ["Raw Data", "Data Types & Missing", "Class Balance"]
    )
    with tab1:
        st.markdown(
            f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns"
        )
        st.dataframe(df_raw.head(100), use_container_width=True)
        csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Raw Dataset as CSV",
            data=csv_bytes,
            file_name="ai4i2020_raw.csv",
            mime="text/csv",
        )
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Types**")
            st.dataframe(
                df_raw.dtypes.to_frame(name="dtype"), use_container_width=True
            )
        with col2:
            st.markdown("**Missing Values**")
            missing = df_raw.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                st.dataframe(
                    missing.to_frame(name="count"), use_container_width=True
                )
            else:
                st.success("No missing values found.")
    with tab3:
        target_counts = df_raw[TARGET_COLUMN].value_counts()
        fig = px.bar(
            x=target_counts.index.map({0: "Normal", 1: "Failure"}),
            y=target_counts.values,
            labels={"x": "Class", "y": "Count"},
            color=target_counts.index.map({0: "Normal", 1: "Failure"}),
            color_discrete_map={"Normal": "#00c853", "Failure": "#ff4b4b"},
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Processed Dataset")
    if processed_data_exists():
        try:
            X_train, y_train, X_test, y_test = load_processed_data()
            st.markdown(
                f"**Train shape:** {X_train.shape}  "
                f"**Test shape:** {X_test.shape}"
            )
            st.dataframe(X_test.head(100), use_container_width=True)
        except Exception as e:
            st.warning(
                f"Processed data not yet available. Train models first. Error: {e}"
            )
    else:
        st.info("Processed data not yet available. Use Train Model to generate it.")


def page_eda() -> None:
    st.title("Exploratory Data Analysis")

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

    st.markdown("#### Correlation Heatmap")
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

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Box Plots by Machine Type")
        fig = px.box(
            df, x="Type", y="Process temperature [K]",
            color="Type",
            template="plotly_dark",
            title="Process Temperature by Machine Type",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Torque vs RPM")
        fig = px.scatter(
            df, x="Rotational speed [rpm]", y="Torque [Nm]",
            color="Machine failure",
            color_discrete_map={0: "#00c853", 1: "#ff4b4b"},
            template="plotly_dark",
            opacity=0.5,
            title="Torque vs Rotational Speed",
        )
        fig.update_layout(
            xaxis_title="Rotational speed [rpm]", yaxis_title="Torque [Nm]"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Failure Type Distribution")
    failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    failure_data = df[failure_cols].melt(
        var_name="Failure Type", value_name="Count"
    )
    fig = px.histogram(
        failure_data, x="Failure Type", color="Failure Type",
        template="plotly_dark",
        title="Failure Type Occurrences",
    )
    st.plotly_chart(fig, use_container_width=True)


def page_train_model() -> None:
    st.title("Model Training")

    if artifacts_exist():
        st.success("Pretrained model artifacts detected.")
    else:
        st.info("No pretrained artifacts found. Training will generate them.")

    if not st.session_state.training_in_progress:
        if st.button("🔄 Retrain Models", type="primary", use_container_width=True):
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
            st.markdown("#### Model Comparison")
            comp_df = st.session_state.comparison_df
            styled = comp_df.copy()
            styled.index.name = "Model"
            styled = styled.reset_index()
            styled["Model"] = styled["Model"].apply(
                lambda m: f"⭐ {m.title()}" if m == "xgboost" else m.title()
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)


def _display_prediction_card(result: dict, threshold: float) -> None:
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


def page_predict_failure() -> None:
    st.title("Predict Failure")

    if not artifacts_exist():
        _artifact_warning()
        return

    model, preprocessor = load_model_resource()

    registry = load_registry_resource()
    versions = registry.get("versions", {})
    xgb_versions = {
        k: v for k, v in versions.items() if v.get("model") == "xgboost"
    }
    recommended_threshold = 0.5
    if xgb_versions:
        latest_xgb = _latest_version_key(xgb_versions)
        recommended_threshold = xgb_versions[latest_xgb].get("threshold", 0.5)

    threshold = st.slider(
        "Decision Threshold",
        0.1,
        0.9,
        float(recommended_threshold),
        0.05,
        help=(
            "The decision threshold determines when the model's risk score "
            "is classified as a predicted failure. The recommended value is "
            "loaded from the project's threshold optimization artifacts."
        ),
    )

    tab_a, tab_b = st.tabs(["Manual Input", "Batch CSV Upload"])

    with tab_a:
        st.markdown("#### Manual Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            air_temp = st.slider(
                "Air Temperature [K]", 290.0, 320.0, 298.0, 1.0
            )
            process_temp = st.slider(
                "Process Temperature [K]", 300.0, 350.0, 310.0, 1.0
            )
            rpm = st.slider(
                "Rotational Speed [rpm]", 0.0, 3000.0, 1500.0, 10.0
            )
        with col2:
            torque = st.slider("Torque [Nm]", 0.0, 100.0, 40.0, 1.0)
            tool_wear = st.slider("Tool Wear [min]", 0.0, 300.0, 50.0, 1.0)
            machine_type = st.selectbox(
                "Machine Type", ["L", "M", "H"], index=1
            )
        with col3:
            temp_diff = process_temp - air_temp
            power = torque * rpm * (2 * 3.141592653589793 / 60)
            wear_rate = tool_wear / (rpm + 1e-6)
            torque_norm = torque / (rpm + 1e-6)
            temp_wear_inter = temp_diff * tool_wear

        st.markdown("**Derived Features:**")
        st.write(
            f"Temperature Diff: {temp_diff:.2f} K | Power: {power:.2f} W | "
            f"Wear Rate: {wear_rate:.6f} | Torque Normalized: {torque_norm:.6f} | "
            f"Temp-Wear Interaction: {temp_wear_inter:.2f}"
        )

        if st.button("Predict", type="primary", use_container_width=True):
            input_dict = {
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
                "Type": machine_type,
            }
            try:
                result = predict(input_dict, threshold=threshold)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                logger.error("Prediction error: %s", exc)
                return

            try:
                _display_prediction_card(result, threshold)
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
        st.markdown("#### Batch Prediction from CSV")
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
                missing_cols = [
                    c for c in required_cols if c not in df_upload.columns
                ]
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
                                results_df = batch_predict(
                                    temp_path, threshold=threshold
                                )
                            finally:
                                if temp_path.exists():
                                    os.remove(temp_path)

                            st.dataframe(
                                results_df, use_container_width=True
                            )

                            csv_bytes = results_df.to_csv(
                                index=False
                            ).encode("utf-8")
                            st.download_button(
                                label="Download Predictions",
                                data=csv_bytes,
                                file_name="predictions.csv",
                                mime="text/csv",
                            )
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")
                logger.error("Batch prediction error: %s", exc)


def page_explain_prediction() -> None:
    st.title("Explain Prediction")

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

    st.markdown("#### SHAP Analysis")

    try:
        explainer = get_shap_explainer(model)
        if explainer is None:
            st.error("SHAP explainer is unavailable for this model artifact.")
            return

        idx = st.slider("Sample Index", 0, len(X_test) - 1, 0)

        col1, col2 = st.columns(2)
        with col1:
            waterfall_path = FIGURES_DIR / "shap_waterfall.png"
            shap_waterfall_plot(explainer, X_test, idx, waterfall_path)
            if waterfall_path.exists():
                st.image(
                    str(waterfall_path), caption="SHAP Waterfall Plot"
                )
        with col2:
            force_path = FIGURES_DIR / "shap_force.html"
            shap_force_plot_html(explainer, X_test, idx, force_path)
            if force_path.exists():
                st.info(
                    "SHAP Force plot saved to `{}`. "
                    "Download from Reports if needed.".format(force_path)
                )
            else:
                st.warning("SHAP Force plot could not be generated.")

        top_features = get_top_features_shap(explainer, X_test, top_n=5)
        st.markdown(f"**Top 5 Features:** {', '.join(top_features)}")

        dep_paths = shap_dependence_plots(
            explainer, X_test, top_features, FIGURES_DIR
        )
        for p in dep_paths:
            if p.exists():
                st.image(str(p), caption=f"Dependence: {p.stem}")

    except Exception as exc:
        logger.error("SHAP explanation error: %s", exc)
        st.error(f"SHAP explanation failed: {exc}")

    st.markdown(
        "Positive SHAP values push the prediction toward failure, while "
        "negative values push it toward normal operation."
    )

    st.markdown("#### LIME Analysis")
    try:
        lime_idx = st.slider(
            "LIME Sample Index", 0, len(X_test) - 1, 0, key="lime_idx"
        )
        lime_result = lime_explain(model, X_train, X_test, idx=lime_idx)
        if lime_result:
            lime_df = pd.DataFrame(
                list(lime_result.items()),
                columns=["Feature", "Contribution"],
            )
            lime_df = lime_df.sort_values(
                "Contribution", key=abs, ascending=False
            )
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
            direction = (
                "positively"
                if top_pos["Contribution"] > 0
                else "negatively"
            )
            st.info(
                f"The top factor contributing to this model prediction was "
                f"**{top_pos['Feature']}**, which pushed the prediction "
                f"{direction} by {abs(top_pos['Contribution']):.4f}. "
                f"This is a model explanation, not a causal claim."
            )
        else:
            st.warning(
                "LIME explanation returned no contributions for this sample."
            )
    except Exception as exc:
        logger.error("LIME explanation error: %s", exc)
        st.error(f"LIME explanation failed: {exc}")


def page_performance_metrics() -> None:
    st.title("Performance Metrics")

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
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(
                card_html(
                    "Accuracy",
                    f"{metrics.get('accuracy', 0):.4f}",
                    "#00d4ff",
                ),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                card_html(
                    "Precision",
                    f"{metrics.get('precision', 0):.4f}",
                    "#00c853",
                ),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                card_html(
                    "Recall",
                    f"{metrics.get('recall', 0):.4f}",
                    "#ffab00",
                ),
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                card_html("F1", f"{metrics.get('f1', 0):.4f}", "#00d4ff"),
                unsafe_allow_html=True,
            )
        with col5:
            st.markdown(
                card_html(
                    "ROC-AUC",
                    f"{metrics.get('roc_auc', 0):.4f}",
                    "#00c853",
                ),
                unsafe_allow_html=True,
            )

    st.markdown("#### Model Comparison")
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

    st.markdown("#### Confusion Matrices")
    cm_cols = st.columns(2)
    for i, model_name in enumerate(model_names[:4]):
        cm_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
        if cm_path.exists():
            with cm_cols[i % 2]:
                st.image(
                    str(cm_path),
                    caption=f"Confusion Matrix — {model_name}",
                )

    st.markdown("#### ROC & PR Curves")
    tab1, tab2 = st.tabs(["ROC Curve", "Precision-Recall Curve"])
    with tab1:
        roc_path = RESULTS_DIR / "roc_curve.html"
        if roc_path.exists():
            with open(roc_path, "r") as f:
                roc_html = f.read()
            st.components.v1.html(roc_html, height=500)
        else:
            st.info("ROC curve HTML not found.")
    with tab2:
        pr_path = RESULTS_DIR / "pr_curve.html"
        if pr_path.exists():
            with open(pr_path, "r") as f:
                pr_html = f.read()
            st.components.v1.html(pr_html, height=500)
        else:
            st.info("Precision-recall curve HTML not found.")

    st.markdown("#### Classification Report")
    report_path = RESULTS_DIR / "classification_report.txt"
    if report_path.exists():
        with open(report_path, "r") as f:
            report_text = f.read()
        st.text(report_text)
    else:
        st.info("Classification report not found.")

    st.markdown("#### Threshold Analysis")
    thresh_path = FIGURES_DIR / "threshold_analysis.html"
    if thresh_path.exists():
        with open(thresh_path, "r") as f:
            thresh_html = f.read()
        st.components.v1.html(thresh_html, height=500)
    else:
        st.info("Threshold analysis HTML not found.")


def page_threshold_optimization() -> None:
    st.title("Threshold Optimization")

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
    xgb_versions = {
        k: v for k, v in versions.items() if v.get("model") == "xgboost"
    }
    recommended_threshold = 0.5
    if xgb_versions:
        latest_xgb = _latest_version_key(xgb_versions)
        recommended_threshold = xgb_versions[latest_xgb].get("threshold", 0.5)

    st.markdown(
        "Adjust the decision threshold and observe its effect on "
        "precision, recall, F1, and error counts."
    )
    threshold = st.slider(
        "Decision Threshold",
        0.05,
        0.95,
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

    st.markdown(
        '<div class="section-title">At This Threshold</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            card_html("Precision", f"{precision:.4f}", "#00c853"),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            card_html("Recall", f"{recall:.4f}", "#ffab00"),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            card_html("F1", f"{f1:.4f}", "#00d4ff"),
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            card_html(
                "ROC-AUC",
                f"{load_xgboost_metrics().get('roc_auc', 0):.4f}"
                if load_xgboost_metrics()
                else "N/A",
                "#ff4b4b",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Error Analysis</div>',
        unsafe_allow_html=True,
    )
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown(
            card_html("Predicted Failures", str(tp + fp), "#00d4ff"),
            unsafe_allow_html=True,
        )
    with e2:
        st.markdown(
            card_html("False Positives", str(fp), "#ffab00"),
            unsafe_allow_html=True,
        )
    with e3:
        st.markdown(
            card_html("False Negatives", str(fn), "#ff4b4b"),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Tradeoff</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Lower thresholds generally identify more potential failures "
        "but may increase false alarms. Higher thresholds reduce false "
        "alarms but may miss some failures."
    )


def page_download_reports() -> None:
    st.title("Reports & Downloads")

    st.markdown("#### Evaluation Reports")
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
        pr_path = RESULTS_DIR / "pr_curve.html"
        if pr_path.exists():
            with open(pr_path, "rb") as f:
                st.download_button(
                    label="Download Precision-Recall Curve",
                    data=f.read(),
                    file_name="pr_curve.html",
                    mime="text/html",
                )
    with col2:
        thresh_path = FIGURES_DIR / "threshold_analysis.html"
        if thresh_path.exists():
            with open(thresh_path, "rb") as f:
                st.download_button(
                    label="Download Threshold Analysis",
                    data=f.read(),
                    file_name="threshold_analysis.html",
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

    st.markdown("#### Model Artifacts")
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

    st.markdown("#### Additional Artifacts")
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

    st.markdown("#### PDF Report")
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
                    "lines": [
                        f"{k}: {v}" for k, v in latest_metrics.items()
                    ],
                },
            ]
            pdf_path = RESULTS_DIR / "prediction_report.pdf"
            generate_report(
                "Predictive Maintenance Report", sections, pdf_path
            )
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Predictive Maintenance XAI",
        layout="wide",
        page_icon="⚙️",
    )
    st.markdown(_get_css(), unsafe_allow_html=True)

    _init_session_state()

    with st.sidebar:
        st.markdown("## Predictive Maintenance")
        st.markdown("---")

        st.markdown(
            '<div class="sidebar-section">'
            '<div class="section-header">Overview</div>',
            unsafe_allow_html=True,
        )
        page_home_btn = st.button("🏠 Home", use_container_width=True)

        st.markdown(
            '<div class="sidebar-section">'
            '<div class="section-header">Data</div>',
            unsafe_allow_html=True,
        )
        page_dataset_btn = st.button(
            "📊 Dataset Overview", use_container_width=True
        )
        page_eda_btn = st.button(
            "📈 Exploratory Data Analysis", use_container_width=True
        )

        st.markdown(
            '<div class="sidebar-section">'
            '<div class="section-header">Prediction</div>',
            unsafe_allow_html=True,
        )
        page_predict_btn = st.button(
            "⚡ Predict Failure", type="primary", use_container_width=True
        )
        page_explain_btn = st.button(
            "🔎 Explain Prediction", use_container_width=True
        )

        st.markdown(
            '<div class="sidebar-section">'
            '<div class="section-header">Evaluation</div>',
            unsafe_allow_html=True,
        )
        page_metrics_btn = st.button(
            "📉 Performance Metrics", use_container_width=True
        )
        page_threshold_btn = st.button(
            "🎚️ Threshold Optimization", use_container_width=True
        )

        st.markdown(
            '<div class="sidebar-section">'
            '<div class="section-header">Reports</div>',
            unsafe_allow_html=True,
        )
        page_reports_btn = st.button(
            "📄 Reports & Downloads", use_container_width=True
        )

        st.markdown("---")
        st.markdown(
            '<div class="sidebar-section">'
            '<div class="section-header">Administration</div>',
            unsafe_allow_html=True,
        )
        page_train_btn = st.button(
            "⚙️ Model Training", use_container_width=True
        )

    if page_home_btn:
        st.session_state["page"] = "Home"
    elif page_dataset_btn:
        st.session_state["page"] = "Dataset Overview"
    elif page_eda_btn:
        st.session_state["page"] = "Exploratory Data Analysis"
    elif page_predict_btn:
        st.session_state["page"] = "Predict Failure"
    elif page_explain_btn:
        st.session_state["page"] = "Explain Prediction"
    elif page_metrics_btn:
        st.session_state["page"] = "Performance Metrics"
    elif page_threshold_btn:
        st.session_state["page"] = "Threshold Optimization"
    elif page_reports_btn:
        st.session_state["page"] = "Reports & Downloads"
    elif page_train_btn:
        st.session_state["page"] = "Model Training"

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


if __name__ == "__main__":
    main()
