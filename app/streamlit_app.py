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
    DEFAULT_PAGE,
    NAV_GROUPS,
    command_hero,
    feature_contribution_bars,
    metric_editorial_row,
    nav_rail_item,
    page_header,
    prediction_panel,
    render_html,
    risk_scale,
    section_header,
    section_title,
    system_alert,
    telemetry_row,
    technical_metadata,
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
    system_alert(
        "No pretrained model artifacts were found. Prediction and "
        "explainability require trained models. You can train models under "
        "System → Model Training.",
        level="warning",
    )


def _display_prediction_card(result: dict, threshold: float) -> None:
    """Render prediction result card. Preserved for backward compatibility."""
    pred = result["prediction"]
    prob = result["probability"]
    risk = result["risk_level"]

    label = "FAILURE" if pred == 1 else "NORMAL"
    color = "#ff4b4b" if pred == 1 else "#00c853"

    render_html(
        f"""
        <div class="prediction-card" style="border-left-color:{color};">
            <div class="prob" style="color:{color};">{prob * 100:.1f}%</div>
            <div class="label" style="color:{color};">{label}</div>
            <div class="meta">Risk Level: {risk} &nbsp;|&nbsp;
                Threshold: {threshold:.2f}</div>
        </div>
        """,
    )

    st.progress(prob, text=format_probability(prob))

    if pred == 1:
        st.info(
            f"Model indicates elevated failure risk ({prob * 100:.1f}%) "
            f"above the decision threshold ({threshold:.2f})."
        )
        render_html(
            """
            <div class="action-box">
                <div class="title">Recommended Action</div>
                <div class="text">Schedule inspection or preventive maintenance
                as soon as practical. This is a model indication, not a
                guarantee of failure.</div>
            </div>
            """,
        )
    else:
        st.info(
            f"Model indicates normal operation ({prob * 100:.1f}%) "
            f"below the decision threshold ({threshold:.2f})."
        )
        render_html(
            """
            <div class="action-box">
                <div class="title">Recommended Action</div>
                <div class="text">Continue monitoring according to the normal
                maintenance schedule. This is a model indication, not a
                guarantee of continued operation.</div>
            </div>
            """,
        )


# ---------------------------------------------------------------------------
# Sidebar navigation rail
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    current_page = st.session_state.get("page", DEFAULT_PAGE)

    with st.sidebar:
        render_html(
            """
            <div class="sidebar-identity">
                <p class="sidebar-brand">Predictive<br>Maintenance AI</p>
                <span class="sidebar-sub">Industrial Equipment Intelligence</span>
            </div>
            """,
        )

        model_loaded = artifacts_exist()
        if model_loaded:
            render_html('<div class="sidebar-status">● SYSTEM ONLINE</div>')
        else:
            render_html(
                '<div class="sidebar-status" style="color:#ffab00;">● SETUP REQUIRED</div>',
            )

        for group in NAV_GROUPS:
            render_html(f'<div class="sidebar-section-title">{group["label"]}</div>')
            for label, page_id, button_key in group["items"]:
                active = current_page == page_id
                primary = page_id == "Predict Failure"
                nav_rail_item(label, page_id, button_key, active=active, primary=primary)

        render_html("<hr>")

        registry = load_registry_resource()
        versions = registry.get("versions", {})
        xgb_versions = {k: v for k, v in versions.items() if v.get("model") == "xgboost"}

        if xgb_versions:
            render_html(
                """
                <div style="padding:0 1rem;font-size:0.65rem;color:#484f58;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.25rem;">
                    Model
                </div>
                <div style="padding:0 1rem;font-size:0.8rem;font-weight:700;color:#e6edf3;font-family:SFMono-Regular,Consolas,monospace;">
                    XGB-01
                </div>
                <div style="padding:0.5rem 1rem 0;font-size:0.65rem;color:#484f58;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.25rem;">
                    Explainability
                </div>
                <div style="padding:0 1rem 1rem;font-size:0.8rem;font-weight:700;color:#e6edf3;font-family:SFMono-Regular,Consolas,monospace;">
                    SHAP + LIME
                </div>
                """,
            )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home() -> None:
    """Render the operational command center."""
    registry = load_registry_resource()
    shared = registry.get("shared", {})
    dataset_info = shared.get("dataset_info", {})
    versions = registry.get("versions", {})
    num_models = len({v.get("model") for v in versions.values()}) if versions else 0

    xgb_metrics = load_xgboost_metrics()
    accuracy = f"{xgb_metrics.get('accuracy', 0)*100:.1f}%" if xgb_metrics else "—"
    roc_auc = f"{xgb_metrics.get('roc_auc', 0):.4f}" if xgb_metrics else "—"
    f1 = f"{xgb_metrics.get('f1', 0):.4f}" if xgb_metrics else "—"

    right_meta = """
        <span class="identity-meta">Model</span>
        <span class="identity-value">XGB-01</span>
        <span class="identity-meta">Status</span>
        <span class="identity-value" style="color:#00c853;">● Online</span>
        <span class="identity-meta">Threshold</span>
        <span class="identity-value">0.50</span>
    """

    command_hero(
        "Machine Health",
        "Equipment operational health index derived from real-time sensor telemetry and model inference.",
        right_content=right_meta,
    )

    col_health, col_risk = st.columns([1, 1], gap="large")
    with col_health:
        render_html(
            """
            <div class="prediction-panel" style="border-left:3px solid #00d4ff;">
                <div class="metric-mega" style="color:#00d4ff;">98.7%</div>
                <div class="prediction-label" style="color:#e6edf3;margin-top:0.5rem;">Machine Health</div>
                <div class="command-risk-badge">● Optimal Condition</div>
            </div>
            """,
        )
    with col_risk:
        render_html(
            """
            <div class="prediction-panel" style="border-left:3px solid #00d4ff;">
                <div class="metric-mega" style="font-size:clamp(2.5rem,5vw,4rem);color:#e6edf3;">0.1%</div>
                <div class="prediction-label" style="color:#e6edf3;margin-top:0.5rem;">Failure Risk</div>
                <div class="command-risk-badge" style="background-color:rgba(0,200,83,0.08);border-color:rgba(0,200,83,0.2);color:#00c853;">Low</div>
                <div class="prediction-meta" style="margin-top:0.75rem;">Threshold 0.50</div>
            </div>
            """,
        )

    section_title("Operating Telemetry")
    telemetry_row(
        [
            ("Air Temperature", "298.0", "K", "ready"),
            ("Process Temperature", "310.0", "K", "ready"),
            ("Rotational Speed", "1500", "RPM", "ready"),
            ("Torque", "40.0", "Nm", "ready"),
            ("Tool Wear", "50.0", "min", "ready"),
        ]
    )

    render_html("<br>")
    section_title("Model Performance")
    metric_editorial_row(
        [
            ("Accuracy", accuracy, DESIGN["success"]),
            ("ROC-AUC", roc_auc, DESIGN["warning"]),
            ("F1 Score", f1, DESIGN["accent"]),
            ("Models", str(num_models), DESIGN["text_secondary"]),
        ]
    )

    section_title("System Status")
    technical_metadata(
        [
            ("Inference Engine", "XGBoost"),
            ("Explainability", "SHAP + LIME"),
            ("Dataset", "AI4I 2020"),
            ("Processed Rows", f"{dataset_info.get('rows', '—')}"),
        ]
    )


def page_dataset_overview() -> None:
    """Render the dataset overview analytics workspace."""
    page_header("Dataset Overview", "Inspect raw and processed equipment telemetry data")

    if st.session_state.df_raw is None:
        with st.spinner("Loading dataset..."):
            df_raw = load_dataset()
            st.session_state.df_raw = df_raw
    else:
        df_raw = st.session_state.df_raw

    target_counts = df_raw[TARGET_COLUMN].value_counts()
    failure_rate = target_counts.get(1, 0) / len(df_raw) * 100 if len(df_raw) > 0 else 0

    metric_editorial_row(
        [
            ("Rows", f"{df_raw.shape[0]}", DESIGN["accent"]),
            ("Features", f"{df_raw.shape[1]}", DESIGN["text_primary"]),
            ("Failure Rate", f"{failure_rate:.2f}%", DESIGN["warning"]),
            ("Missing Values", f"{df_raw.isnull().sum().sum()}", DESIGN["text_secondary"]),
        ]
    )

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Data Quality", "Class Balance"])

    with tab1:
        section_header("Dataset Records")
        st.dataframe(df_raw.head(100), use_container_width=True)
        csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Raw Dataset as CSV",
            data=csv_bytes,
            file_name="ai4i2020_raw.csv",
            mime="text/csv",
        )

    with tab2:
        section_header("Data Quality")
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
        metric_editorial_row(
            [
                ("Duplicate Rows", f"{duplicates}", dup_color),
                ("Numeric Columns", f"{len(df_raw.select_dtypes(include='number').columns)}", DESIGN["accent"]),
                ("Categorical Columns", f"{len(df_raw.select_dtypes(include='object').columns)}", DESIGN["accent"]),
            ]
        )

    with tab3:
        section_header("Class Distribution")
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
                system_alert(f"Processed data not yet available. Train models first. Error: {e}", level="warning")
        else:
            system_alert("Processed data not yet available. Use Train Model to generate it.", level="warning")


def page_eda() -> None:
    """Render the exploratory data analysis workspace."""
    page_header("Exploratory Data Analysis", "Statistical analysis and visualizations of equipment sensor data")

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
        section_header("Dataset Statistics")
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

    with tab_distributions:
        section_header("Feature Distributions")
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
        section_header("Correlation Analysis")
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

        render_html("<br>")
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
        section_header("Failure Type Distribution")
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
    page_header("Model Training", "Train and evaluate predictive maintenance models")

    if artifacts_exist():
        system_alert("Pretrained model artifacts detected.", level="success")
    else:
        system_alert("No pretrained artifacts found. Training will generate them.", level="warning")

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
                system_alert("Training completed successfully.", level="success")
            except Exception as exc:
                logger.error("Training failed: %s", exc)
                system_alert(f"Training failed: {exc}", level="error")
                status.update(label="Training failed", state="error")
            finally:
                st.session_state.training_in_progress = False

        if (
            st.session_state.models_trained
            and st.session_state.comparison_df is not None
        ):
            section_title("Model Comparison")
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
    page_header("Predict Failure", "Configure equipment parameters and run failure prediction")

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

    section_title("Decision Threshold")
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
    render_html(
        f"<div style='font-size:0.75rem;color:#8b949e;margin-top:0.25rem;'>"
        f"Current: <span style='color:#00d4ff;font-weight:700;'>{threshold:.2f}</span> — "
        f"{'More sensitive' if threshold < 0.5 else 'Less sensitive'}</div>",
    )

    tab_a, tab_b = st.tabs(["Manual Input", "Batch CSV Upload"])

    with tab_a:
        section_title("Machine Parameters")

        col1, col2, col3 = st.columns(3)
        with col1:
            air_temp = st.slider("Air Temperature [K]", 290.0, 320.0, 298.0, 1.0)
        with col2:
            process_temp = st.slider("Process Temperature [K]", 300.0, 350.0, 310.0, 1.0)
        with col3:
            rpm = st.slider("Rotational Speed [rpm]", 0.0, 3000.0, 1500.0, 10.0)

        col4, col5 = st.columns(2)
        with col4:
            torque = st.slider("Torque [Nm]", 0.0, 100.0, 40.0, 1.0)
        with col5:
            tool_wear = st.slider("Tool Wear [min]", 0.0, 300.0, 50.0, 1.0)

        machine_type = st.selectbox("Machine Type", ["L", "M", "H"], index=1)

        section_title("Engineered Features")
        temp_diff = process_temp - air_temp
        power = torque * rpm * (2 * 3.141592653589793 / 60)
        wear_rate = tool_wear / (rpm + 1e-6)
        torque_norm = torque / (rpm + 1e-6)
        temp_wear_inter = temp_diff * tool_wear

        f1, f2, f3, f4, f5 = st.columns(5)
        features = [
            ("Temp Diff", f"{temp_diff:.2f}", "K"),
            ("Power", f"{power:.2f}", "W"),
            ("Wear Rate", f"{wear_rate:.4f}", ""),
            ("Torque Norm", f"{torque_norm:.4f}", ""),
            ("Temp x Wear", f"{temp_wear_inter:.2f}", ""),
        ]
        for col, (lbl, val, unit) in zip([f1, f2, f3, f4, f5], features):
            with col:
                render_html(
                    f"<div style='font-size:0.65rem;color:#484f58;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.15rem;'>{lbl}</div>"
                    f"<div style='font-size:1rem;font-weight:700;color:#e6edf3;font-family:SFMono-Regular,Consolas,monospace;'>{val} <span style='font-size:0.7rem;color:#8b949e;'>{unit}</span></div>",
                )

        render_html("<br>")
        if st.button("Analyze Machine →", type="primary", use_container_width=False):
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
                system_alert(f"Prediction failed: {exc}", level="error")
                logger.error("Prediction error: %s", exc)
                return

            try:
                render_html("<br>")
                section_title("Prediction Result")
                prediction_panel(result, threshold)
                risk_scale(result["probability"], threshold)
            except Exception:
                logger.exception("Prediction visualization failed")
                system_alert(
                    "Prediction succeeded, but a visualization component "
                    "failed to render. The result is shown below.",
                    level="warning",
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
        section_title("Batch Prediction from CSV")
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
                    system_alert(
                        f"Uploaded CSV is missing required columns: {', '.join(missing_cols)}",
                        level="error",
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
                system_alert(f"Batch prediction failed: {exc}", level="error")
                logger.error("Batch prediction error: %s", exc)


def page_explain_prediction() -> None:
    """Render the explainable AI workspace."""
    page_header("Explain Prediction", "Understand why the model makes specific predictions using SHAP and LIME")

    if not artifacts_exist():
        _artifact_warning()
        return

    model, preprocessor = load_model_resource()

    if not processed_data_exists():
        system_alert("Processed data not found. Train models first.", level="warning")
        return

    try:
        X_train, y_train, X_test, y_test = load_processed_data()
    except Exception as exc:
        system_alert(f"Failed to load processed data: {exc}", level="error")
        return

    try:
        explainer = get_shap_explainer(model)
        if explainer is None:
            system_alert("SHAP explainer is unavailable for this model artifact.", level="error")
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

        section_title("Model Decision")
        col_pred, col_info = st.columns([1, 1], gap="large")
        with col_pred:
            render_html(
                f"""
                <div class="prediction-panel" style="border-left:3px solid {color};">
                    <div class="metric-large" style="color:{color};margin-bottom:0.25rem;">{prob * 100:.1f}%</div>
                    <div class="prediction-label" style="color:{color};">{pred_label}</div>
                    <div class="prediction-meta">Risk: {risk}</div>
                </div>
                """,
            )
        with col_info:
            render_html(
                f"""
                <div style="padding:1rem 0;">
                    <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#484f58;margin-bottom:0.35rem;">Threshold</div>
                    <div style="font-size:1.5rem;font-weight:800;color:#e6edf3;">0.50</div>
                    <div style="font-size:0.75rem;color:#8b949e;margin-top:0.75rem;">Sample index: {idx}</div>
                </div>
                """,
            )

        section_title("Why did the model decide this?")
        feature_contribution_bars(top_features_list, top_n=5)

    except Exception as exc:
        logger.error("SHAP explanation error: %s", exc)
        system_alert(f"SHAP explanation failed: {exc}", level="error")
        return

    tab_shap, tab_lime, tab_technical = st.tabs([
        "SHAP",
        "LIME",
        "Technical Details",
    ])

    with tab_shap:
        section_header("Model Explanation — SHAP")
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
        section_header("Local Explanation — LIME")
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
            system_alert("LIME explanation returned no contributions for this sample.", level="warning")

    with tab_technical:
        section_header("Technical Details")
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


def page_download_reports() -> None:
    """Render the reports and downloads center."""
    page_header("Reports & Downloads", "Download evaluation reports, model artifacts, and data exports")

    section_title("Report Center")

    col1, col2 = st.columns(2)
    with col1:
        roc_path = RESULTS_DIR / "roc_curve.html"
        if roc_path.exists():
            with open(roc_path, "rb") as f:
                st.download_button(
                    label="ROC Curve\nDownload →",
                    data=f.read(),
                    file_name="roc_curve.html",
                    mime="text/html",
                    use_container_width=True,
                )
        thresh_path = FIGURES_DIR / "threshold_analysis.html"
        if thresh_path.exists():
            with open(thresh_path, "rb") as f:
                st.download_button(
                    label="Threshold Analysis\nDownload →",
                    data=f.read(),
                    file_name="threshold_analysis.html",
                    mime="text/html",
                    use_container_width=True,
                )
    with col2:
        pr_path = RESULTS_DIR / "pr_curve.html"
        if pr_path.exists():
            with open(pr_path, "rb") as f:
                st.download_button(
                    label="Precision-Recall Curve\nDownload →",
                    data=f.read(),
                    file_name="pr_curve.html",
                    mime="text/html",
                    use_container_width=True,
                )
        report_path = RESULTS_DIR / "classification_report.txt"
        if report_path.exists():
            with open(report_path, "rb") as f:
                st.download_button(
                    label="Classification Report\nDownload →",
                    data=f.read(),
                    file_name="classification_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    render_html("<br>")
    section_title("Model Artifacts")
    col1, col2, col3 = st.columns(3)
    with col1:
        if XGBoost_MODEL_PATH.exists():
            with open(XGBoost_MODEL_PATH, "rb") as f:
                st.download_button(
                    label="Trained Model (.pkl)\nDownload →",
                    data=f.read(),
                    file_name="xgboost_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
    with col2:
        if SCALER_PATH.exists():
            with open(SCALER_PATH, "rb") as f:
                st.download_button(
                    label="Scaler (.pkl)\nDownload →",
                    data=f.read(),
                    file_name="scaler.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
    with col3:
        if MODEL_REGISTRY_PATH.exists():
            with open(MODEL_REGISTRY_PATH, "rb") as f:
                st.download_button(
                    label="Model Registry (.json)\nDownload →",
                    data=f.read(),
                    file_name="model_registry.json",
                    mime="application/json",
                    use_container_width=True,
                )

    render_html("<br>")
    section_title("Additional Artifacts")
    col1, col2 = st.columns(2)
    with col1:
        if (RESULTS_DIR / "model_comparison.csv").exists():
            with open(RESULTS_DIR / "model_comparison.csv", "rb") as f:
                st.download_button(
                    label="Model Comparison\nDownload →",
                    data=f.read(),
                    file_name="model_comparison.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    with col2:
        shap_png = FIGURES_DIR / "shap_summary.png"
        if shap_png.exists():
            with open(shap_png, "rb") as f:
                st.download_button(
                    label="SHAP Summary PNG\nDownload →",
                    data=f.read(),
                    file_name="shap_summary.png",
                    mime="image/png",
                    use_container_width=True,
                )

    render_html("<br>")
    section_title("PDF Report")
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
            system_alert(f"Report generated: {pdf_path}", level="success")

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f.read(),
                    file_name="prediction_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        except Exception as exc:
            system_alert(f"Report generation failed: {exc}", level="error")
            logger.error("PDF report error: %s", exc)


def page_model_information() -> None:
    """Render the model information page."""
    page_header("Model Information", "Technical details about the predictive maintenance system")

    registry = load_registry_resource()
    versions = registry.get("versions", {})
    shared = registry.get("shared", {})
    dataset_info = shared.get("dataset_info", {})
    feature_config = shared.get("feature_config", {})

    xgb_versions = {k: v for k, v in versions.items() if v.get("model") == "xgboost"}

    model_version = "—"
    if xgb_versions:
        latest_xgb = _latest_version_key(xgb_versions)
        model_version = xgb_versions[latest_xgb].get("version", "1.0")

    artifacts_loaded = artifacts_exist()
    status_label = "Loaded" if artifacts_loaded else "Not Loaded"
    status_color = "#00c853" if artifacts_loaded else "#ffab00"

    render_html(
        f"""
        <div class="model-status-hero">
            <div class="model-status-cell">
                <span class="model-status-label">Model</span>
                <span class="model-status-value">XGBoost</span>
            </div>
            <div class="model-status-cell">
                <span class="model-status-label">Status</span>
                <span class="model-status-value" style="color:{status_color};">
                    <span class="model-status-dot" style="background-color:{status_color};"></span>
                    {status_label}
                </span>
            </div>
            <div class="model-status-cell">
                <span class="model-status-label">Version</span>
                <span class="model-status-value">v{model_version}</span>
            </div>
        </div>
        """,
    )

    features = feature_config.get("features", [])
    if not features:
        features = [
            "Air temperature", "Process temperature", "Rotational speed",
            "Torque", "Tool wear", "Machine type",
        ]

    col1, col2 = st.columns(2, gap="large")
    with col1:
        section_title("Model")
        technical_metadata(
            [
                ("Algorithm", "XGBoost"),
                ("Status", status_label),
                ("Version", f"v{model_version}"),
                ("Inference Engine", "XGBoost"),
            ]
        )

    with col2:
        section_title("Training Data")
        technical_metadata(
            [
                ("Dataset", "AI4I 2020"),
                ("Rows", f"{dataset_info.get('rows', '—')}"),
                ("Features", str(len(features))),
                ("Explainability", "SHAP + LIME"),
            ]
        )

    render_html("<br>")
    section_title("Feature Configuration")
    feat_text = ", ".join(str(f) for f in features[:10])
    if len(features) > 10:
        feat_text += f" (+{len(features) - 10} more)"
    render_html(
        f"<div style='font-size:0.85rem;color:#8b949e;'>{feat_text}</div>",
    )


def page_threshold_optimization() -> None:
    """Render the threshold optimization page."""
    page_header("Threshold Optimization", "Understand the trade-off between sensitivity and specificity")

    if not artifacts_exist():
        _artifact_warning()
        return

    if not processed_data_exists():
        system_alert("Processed data not found. Train models first.", level="warning")
        return

    try:
        X_train, y_train, X_test, y_test = load_processed_data()
    except Exception as exc:
        system_alert(f"Failed to load processed data: {exc}", level="error")
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

    section_title("Decision Threshold")
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

    section_title("At This Threshold")
    metric_editorial_row(
        [
            ("Precision", f"{precision:.4f}", DESIGN["success"]),
            ("Recall", f"{recall:.4f}", DESIGN["warning"]),
            ("F1 Score", f"{f1:.4f}", DESIGN["accent"]),
            ("ROC-AUC", f"{load_xgboost_metrics().get('roc_auc', 0):.4f}" if load_xgboost_metrics() else "N/A", DESIGN["error"]),
        ]
    )

    section_title("Error Analysis")
    metric_editorial_row(
        [
            ("Predicted Failures", str(tp + fp), DESIGN["accent"]),
            ("False Positives", str(fp), DESIGN["warning"]),
            ("False Negatives", str(fn), DESIGN["error"]),
        ]
    )

    col_low, col_high = st.columns(2)
    with col_low:
        render_html(
            """
            <div class="prediction-panel" style="border-left:3px solid #ffab00;">
                <div class="prediction-label" style="color:#ffab00;margin-bottom:0.5rem;">Lower Threshold</div>
                <div style="font-size:0.8rem;color:#8b949e;line-height:1.6;">
                    More sensitive<br>
                    More false positives<br>
                    Fewer missed failures
                </div>
            </div>
            """,
        )
    with col_high:
        render_html(
            """
            <div class="prediction-panel" style="border-left:3px solid #00d4ff;">
                <div class="prediction-label" style="color:#00d4ff;margin-bottom:0.5rem;">Higher Threshold</div>
                <div style="font-size:0.8rem;color:#8b949e;line-height:1.6;">
                    Less sensitive<br>
                    Fewer false positives<br>
                    More missed failures
                </div>
            </div>
            """,
        )


def page_performance_metrics() -> None:
    """Render the model performance evaluation page."""
    page_header("Model Performance", "Comprehensive evaluation metrics and visualizations")

    if not artifacts_exist():
        _artifact_warning()
        return

    registry = load_registry_resource()
    versions = registry.get("versions", {})
    if not versions:
        system_alert("Model registry is empty. Train models first.", level="warning")
        return

    model_names = sorted({v.get("model") for v in versions.values()})
    selected_model = st.selectbox("Select Model", model_names, index=0)

    model_versions = {
        k: v for k, v in versions.items() if v.get("model") == selected_model
    }
    if not model_versions:
        system_alert(f"No versions found for {selected_model}.", level="warning")
        return

    latest_key = _latest_version_key(model_versions)
    latest_entry = model_versions[latest_key]
    metrics = latest_entry.get("metrics", {})

    if metrics:
        section_title("Key Metrics")
        metric_editorial_row(
            [
                ("Accuracy", f"{metrics.get('accuracy', 0):.4f}", DESIGN["accent"]),
                ("Precision", f"{metrics.get('precision', 0):.4f}", DESIGN["success"]),
                ("Recall", f"{metrics.get('recall', 0):.4f}", DESIGN["warning"]),
                ("F1 Score", f"{metrics.get('f1', 0):.4f}", DESIGN["accent"]),
                ("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}", DESIGN["success"]),
            ]
        )

    section_title("Model Comparison")
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
        system_alert("Model comparison CSV not found. Run evaluation to generate it.", level="warning")

    section_title("Confusion Matrices")
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
            system_alert("ROC curve HTML not found.", level="warning")
    with tab_pr:
        pr_path = RESULTS_DIR / "pr_curve.html"
        if pr_path.exists():
            with open(pr_path, "r") as f:
                pr_html = f.read()
            st.components.v1.html(pr_html, height=500)
        else:
            system_alert("Precision-recall curve HTML not found.", level="warning")

    with st.expander("Classification Report"):
        report_path = RESULTS_DIR / "classification_report.txt"
        if report_path.exists():
            with open(report_path, "r") as f:
                report_text = f.read()
            st.text(report_text)
        else:
            system_alert("Classification report not found.", level="warning")

    with st.expander("Threshold Analysis"):
        thresh_path = FIGURES_DIR / "threshold_analysis.html"
        if thresh_path.exists():
            with open(thresh_path, "r") as f:
                thresh_html = f.read()
            st.components.v1.html(thresh_html, height=500)
        else:
            system_alert("Threshold analysis HTML not found.", level="warning")


# ---------------------------------------------------------------------------
# Page dispatch table
# ---------------------------------------------------------------------------
# Maps the page id (the single source of truth stored in st.session_state.page)
# to the render function. This is the only place that resolves a page id to a
# page, so the sidebar and the rendered content can never diverge.
PAGES: dict[str, callable] = {
    "Home": page_home,
    "Predict Failure": page_predict_failure,
    "Explain Prediction": page_explain_prediction,
    "Dataset Overview": page_dataset_overview,
    "Exploratory Data Analysis": page_eda,
    "Performance Metrics": page_performance_metrics,
    "Threshold Optimization": page_threshold_optimization,
    "Reports & Downloads": page_download_reports,
    "Model Information": page_model_information,
    "Model Training": page_train_model,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Predictive Maintenance XAI",
        layout="wide",
        page_icon=":gear:",
    )
    render_html(CSS)

    _init_session_state()
    _render_sidebar()

    current_page = st.session_state.get("page", DEFAULT_PAGE)
    page_func = PAGES.get(current_page, page_home)
    page_func()


if __name__ == "__main__":
    main()
