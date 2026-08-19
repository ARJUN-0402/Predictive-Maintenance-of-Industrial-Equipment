import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import f1_score

from src.config import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TARGET_COLUMN,
    XGBoost_MODEL_PATH,
)
from src.data_loader import load_dataset
from src.evaluate import (
    compute_all_metrics,
)
from src.explain import (
    get_shap_explainer,
    lime_explain,
    shap_bar_plot,
    shap_dependence_plots,
    shap_force_plot_html,
    shap_summary_plot,
    shap_waterfall_plot,
    get_top_features_shap,
)
from src.feature_engineering import engineer_features
from src.predict import batch_predict, predict
from src.preprocessing import load_processed_data
from src.train import train_all_models
from src.utils import card_html, setup_logging

logger = setup_logging("streamlit_app")


def _get_css() -> str:
    return """
<style>
    .stApp { background-color: #0e1117; }
    .stMetric, .stCard { background-color: #1c1e26; border-radius: 12px; padding: 16px; margin: 8px; }
    .stButton>button { background-color: #00d4ff; color: #0e1117; border: none; border-radius: 8px; font-weight: bold; }
    .stSlider>div>div { background-color: #00d4ff; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { color: #00d4ff; }
    .stDataFrame { background-color: #1c1e26; color: #e0e0e0; }
    .css-1d391kg { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1c1e26; color: #e0e0e0; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background-color: #00d4ff; color: #0e1117; }
</style>
"""


def _init_session_state() -> None:
    if "models_trained" not in st.session_state:
        st.session_state.models_trained = False
    if "models" not in st.session_state:
        st.session_state.models = {}
    if "metrics" not in st.session_state:
        st.session_state.metrics = {}
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None
    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = None
    if "comparison_df" not in st.session_state:
        st.session_state.comparison_df = None
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "df_processed" not in st.session_state:
        st.session_state.df_processed = None


def page_home() -> None:
    st.title("Predictive Maintenance of Industrial Equipment")
    st.markdown(
        "### Using XGBoost and Explainable AI (XAI)"
    )
    st.markdown(
        "This application predicts machine failures in industrial equipment "
        "using machine learning models with SHAP and LIME explanations."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(card_html("Models", "4", "#00d4ff"))
    with col2:
        st.markdown(card_html("Primary Model", "XGBoost", "#00c853"))
    with col3:
        st.markdown(card_html("Features", "13", "#ffab00"))
    with col4:
        st.markdown(card_html("Target", "Machine Failure", "#ff4b4b"))

    st.markdown("---")
    st.markdown("#### Architecture")
    st.markdown(
        """
        <div style="background:#1c1e26;padding:20px;border-radius:12px;">
        <pre style="color:#e0e0e0;font-family:monospace;">
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Data Loader  │───▶│ Preprocessing│───▶│Feature Eng.  │
│  (AI4I 2020)  │    │  (Pipeline)  │    │  (Derived    │
│               │    │              │    │   Features)  │
└──────────────┘    └──────────────┘    └──────────────┘
                                                      │
                                                      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Streamlit   │◀───│  Model Store │◀───│   Training   │
│   App (UI)    │    │ (XGBoost +   │    │ (4 Models)   │
│               │    │  RF, GB, LR) │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
                                                      │
                                                      ▼
                                            ┌──────────────┐
                                            │   Explain AI  │
                                            │ (SHAP + LIME) │
                                            └──────────────┘
        </pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Quick Start")
    st.markdown("1. Go to **Dataset Overview** to inspect the data")
    st.markdown("2. Go to **Train Model** to train all classifiers")
    st.markdown("3. Go to **Predict Failure** to make predictions")
    st.markdown("4. Go to **Explain Prediction** to understand model decisions")
    st.markdown("5. Go to **Performance Metrics** to evaluate models")


def page_dataset_overview() -> None:
    st.title("Dataset Overview")

    if st.session_state.df_raw is None:
        with st.spinner("Loading dataset..."):
            df_raw = load_dataset()
            st.session_state.df_raw = df_raw
    else:
        df_raw = st.session_state.df_raw

    st.markdown("#### Raw Dataset")
    st.dataframe(df_raw.head(100), use_container_width=True)
    st.markdown(f"**Shape:** {df_raw.shape}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Data Types**")
        st.dataframe(df_raw.dtypes.to_frame(name="dtype"))
    with col2:
        st.markdown("**Missing Values**")
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            st.dataframe(missing.to_frame(name="count"))
        else:
            st.write("No missing values found.")

    st.markdown("#### Class Balance")
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
    if st.session_state.df_processed is None:
        try:
            X_train, y_train, X_test, y_test = load_processed_data()
            st.session_state.df_processed = X_test.head(100)
        except Exception as e:
            st.warning(f"Processed data not yet available. Train models first. Error: {e}")
            return
    else:
        X_test = st.session_state.df_processed

    st.dataframe(X_test.head(100), use_container_width=True)

    csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Raw Dataset as CSV",
        data=csv_bytes,
        file_name="ai4i2020_raw.csv",
        mime="text/csv",
    )


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
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Correlation Heatmap")
    corr_cols = numeric_cols + ["Machine failure"]
    corr_df = df[corr_cols].corr()
    fig = px.imshow(
        corr_df,
        color_continuous_scale="RdBu_r",
        template="plotly_dark",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Box Plots by Machine Type")
    fig = px.box(
        df, x="Type", y="Process temperature [K]",
        color="Type",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Failure Type Distribution")
    failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    failure_data = df[failure_cols].melt(var_name="Failure Type", value_name="Count")
    fig = px.histogram(
        failure_data, x="Failure Type", color="Failure Type",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Torque vs RPM (colored by Failure)")
    fig = px.scatter(
        df, x="Rotational speed [rpm]", y="Torque [Nm]",
        color="Machine failure",
        color_discrete_map={0: "#00c853", 1: "#ff4b4b"},
        template="plotly_dark",
        opacity=0.5,
    )
    st.plotly_chart(fig, use_container_width=True)


def page_train_model() -> None:
    st.title("Train Model")

    if st.button("Start Training", type="primary", use_container_width=True):
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

                from src.evaluate import compare_models

                st.session_state.comparison_df = compare_models(
                    st.session_state.metrics
                )

                status.update(label="Training complete!", state="complete")
                st.success("Training completed successfully!")
            except Exception as exc:
                logger.error("Training failed: %s", exc)
                st.error(f"Training failed: {exc}")
                status.update(label="Training failed", state="error")

        if st.session_state.models_trained and st.session_state.comparison_df is not None:
            st.markdown("#### Model Comparison")
            comp_df = st.session_state.comparison_df
            st.dataframe(comp_df.style.highlight_max(subset=["roc_auc"], color="#00c853"))


def page_predict_failure() -> None:
    st.title("Predict Failure")

    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

    tab_a, tab_b = st.tabs(["Manual Input", "Batch CSV Upload"])

    with tab_a:
        st.markdown("#### Manual Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            air_temp = st.slider("Air Temperature [K]", 290.0, 320.0, 298.0, 1.0)
            process_temp = st.slider("Process Temperature [K]", 300.0, 350.0, 310.0, 1.0)
            rpm = st.slider("Rotational Speed [rpm]", 0.0, 3000.0, 1500.0, 10.0)
        with col2:
            torque = st.slider("Torque [Nm]", 0.0, 100.0, 40.0, 1.0)
            tool_wear = st.slider("Tool Wear [min]", 0.0, 300.0, 50.0, 1.0)
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"], index=1)
        with col3:
            temp_diff = process_temp - air_temp
            power = torque * rpm * (2 * 3.141592653589793 / 60)
            wear_rate = tool_wear / (rpm + 1e-6)
            torque_norm = torque / (rpm + 1e-6)
            temp_wear_inter = temp_diff * tool_wear

        st.markdown(f"**Derived Features:**")
        st.write(f"Temperature Diff: {temp_diff:.2f} K | Power: {power:.2f} W")
        st.write(f"Wear Rate: {wear_rate:.6f} | Torque Normalized: {torque_norm:.6f}")
        st.write(f"Temp-Wear Interaction: {temp_wear_inter:.2f}")

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
                _display_prediction_card(result)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                logger.error("Prediction error: %s", exc)

    with tab_b:
        st.markdown("#### Batch Prediction from CSV")
        uploaded_file = st.file_uploader(
            "Upload CSV file", type=["csv"],
            help="CSV must contain columns: Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Type"
        )
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.dataframe(df_upload.head(), use_container_width=True)

                if st.button("Run Batch Predictions", type="primary"):
                    with st.spinner("Running predictions..."):
                        temp_path = Path("data/raw/_batch_upload.csv")
                        df_upload.to_csv(temp_path, index=False)
                        results_df = batch_predict(temp_path, threshold=threshold)
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


def _display_prediction_card(result: dict) -> None:
    pred = result["prediction"]
    prob = result["probability"]
    risk = result["risk_level"]

    label = "FAILURE" if pred == 1 else "NORMAL"
    color = "#ff4b4b" if pred == 1 else "#00c853"

    st.markdown(f"""
    <div style="background:#1c1e26;border-radius:12px;padding:20px;margin:10px 0;border-left:5px solid {color};">
        <h2 style="color:{color};margin:0;">Prediction: {label}</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(card_html("Failure Probability", f"{prob*100:.1f}%", color))
    with col2:
        st.markdown(card_html("Risk Level", risk, "#ffab00" if risk == "Medium" else color))
    with col3:
        st.markdown(card_html("Threshold", f"{threshold:.2f}", "#00d4ff"))

    st.progress(prob, format=f"{prob*100:.1f}%")


def page_explain_prediction() -> None:
    st.title("Explain Prediction")

    if not st.session_state.models_trained:
        st.warning("Please train models first from the Train Model page.")
        return

    st.markdown("#### SHAP Analysis")

    try:
        model = st.session_state.models.get("xgboost")
        X_test = st.session_state.X_test
        if model is None or X_test is None:
            st.warning("Model or test data not available.")
            return

        explainer = get_shap_explainer(model)
        if explainer is None:
            st.error("Failed to create SHAP explainer.")
            return

        idx = st.slider("Sample Index", 0, len(X_test) - 1, 0)

        col1, col2 = st.columns(2)
        with col1:
            waterfall_path = Path("reports/figures/shap_waterfall.png")
            shap_waterfall_plot(explainer, X_test, idx, waterfall_path)
            if waterfall_path.exists():
                st.image(str(waterfall_path), caption="SHAP Waterfall Plot")

        with col2:
            force_path = Path("reports/figures/shap_force.html")
            shap_force_plot_html(explainer, X_test, idx, force_path)
            st.markdown(f"SHAP Force plot saved to `{force_path}`")

        top_features = get_top_features_shap(explainer, X_test, top_n=5)
        st.markdown(f"**Top 5 Features:** {', '.join(top_features)}")

        dep_paths = shap_dependence_plots(
            explainer, X_test, top_features,
            Path("reports/figures")
        )
        for p in dep_paths:
            if p.exists():
                st.image(str(p), caption=f"Dependence: {p.stem}")

    except Exception as exc:
        logger.error("SHAP explanation error: %s", exc)
        st.error(f"SHAP explanation failed: {exc}")

    st.markdown("#### LIME Analysis")
    try:
        model = st.session_state.models.get("xgboost")
        X_test = st.session_state.X_test
        if model is None or X_test is None:
            st.warning("Model or test data not available for LIME.")
        else:
            lime_result = lime_explain(model, X_test, X_test, idx=st.session_state.get("selected_idx", 0))
            if lime_result:
                lime_df = pd.DataFrame(
                    list(lime_result.items()), columns=["Feature", "Contribution"]
                )
                lime_df = lime_df.sort_values("Contribution", key=abs, ascending=False)
                fig = px.bar(
                    lime_df.head(10), x="Contribution", y="Feature",
                    orientation="h", template="plotly_dark",
                    color="Contribution",
                    color_continuous_scale="RdBu_r",
                )
                st.plotly_chart(fig, use_container_width=True)

                top_pos = lime_df.iloc[0]
                direction = "UP" if top_pos["Contribution"] > 0 else "DOWN"
                st.info(
                    f"The top factor contributing to this prediction was "
                    f"**{top_pos['Feature']}**, which pushed the risk {direction} "
                    f"by {abs(top_pos['Contribution']):.4f}."
                )
    except Exception as exc:
        logger.error("LIME explanation error: %s", exc)
        st.error(f"LIME explanation failed: {exc}")


def page_performance_metrics() -> None:
    st.title("Performance Metrics")

    if not st.session_state.models_trained:
        st.warning("Please train models first.")
        return

    model_name = st.selectbox(
        "Select Model",
        list(st.session_state.metrics.keys()),
        index=0,
    )

    metrics = st.session_state.metrics.get(model_name, {})
    if metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(card_html("Accuracy", f"{metrics.get('accuracy', 0):.4f}", "#00d4ff"))
        with col2:
            st.markdown(card_html("Precision", f"{metrics.get('precision', 0):.4f}", "#00c853"))
        with col3:
            st.markdown(card_html("Recall", f"{metrics.get('recall', 0):.4f}", "#ffab00"))
        with col4:
            st.markdown(card_html("F1", f"{metrics.get('f1', 0):.4f}", "#00d4ff"))
        with col5:
            st.markdown(card_html("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}", "#00c853"))

    y_test = st.session_state.y_test
    y_pred = st.session_state.models.get(model_name).predict(st.session_state.X_test)
    y_prob = st.session_state.models.get(model_name).predict_proba(st.session_state.X_test)[:, 1]

    cm_path = RESULTS_DIR / f"confusion_matrix_{model_name}.png"
    if cm_path.exists():
        st.image(str(cm_path), caption=f"Confusion Matrix — {model_name}")

    roc_path = RESULTS_DIR / "roc_curve.html"
    if roc_path.exists():
        with open(roc_path, "r") as f:
            roc_html = f.read()
        st.components.v1.html(roc_html, height=500)

    pr_path = RESULTS_DIR / "pr_curve.html"
    if pr_path.exists():
        with open(pr_path, "r") as f:
            pr_html = f.read()
        st.components.v1.html(pr_html, height=500)

    thresholds = np.arange(0.1, 0.95, 0.05)
    f1_scores = []
    for t in thresholds:
        y_t = (y_prob >= t).astype(int)
        f1_scores.append(f1_score(y_test, y_t, zero_division=0))

    fig = px.line(
        x=thresholds, y=f1_scores,
        labels={"x": "Threshold", "y": "F1 Score"},
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)


def page_download_reports() -> None:
    st.title("Download Reports")

    if st.button("Generate PDF Report", type="primary", use_container_width=True):
        try:
            from src.utils import generate_report

            sections = [
                {
                    "heading": "Model Summary",
                    "lines": [f"Primary Model: XGBoost", f"Random Seed: 42"],
                },
                {
                    "heading": "Metrics",
                    "lines": [
                        f"{k}: {v}" for k, v in st.session_state.metrics.get("xgboost", {}).items()
                    ],
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

    st.markdown("#### Export Data")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.models_trained and st.session_state.X_test is not None:
            model = st.session_state.models.get("xgboost")
            if model is not None:
                X_test = st.session_state.X_test
                y_prob = model.predict_proba(X_test)[:, 1]
                pred_df = pd.DataFrame({
                    "prediction": model.predict(X_test),
                    "probability": y_prob,
                })
                csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

    with col2:
        model_path = XGBoost_MODEL_PATH
        if model_path.exists():
            with open(model_path, "rb") as f:
                st.download_button(
                    label="Download Trained Model (.pkl)",
                    data=f.read(),
                    file_name="xgboost_model.pkl",
                    mime="application/octet-stream",
                )

        shap_png = Path("reports/figures/shap_summary.png")
        if shap_png.exists():
            with open(shap_png, "rb") as f:
                st.download_button(
                    label="Download SHAP Summary PNG",
                    data=f.read(),
                    file_name="shap_summary.png",
                    mime="image/png",
                )


def main() -> None:
    st.set_page_config(
        page_title="Predictive Maintenance XAI",
        layout="wide",
        page_icon="⚙️",
    )
    st.markdown(_get_css(), unsafe_allow_html=True)

    _init_session_state()

    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Dataset Overview",
            "Exploratory Data Analysis",
            "Train Model",
            "Predict Failure",
            "Explain Prediction",
            "Performance Metrics",
            "Download Reports",
        ],
    )

    if page == "Home":
        page_home()
    elif page == "Dataset Overview":
        page_dataset_overview()
    elif page == "Exploratory Data Analysis":
        page_eda()
    elif page == "Train Model":
        page_train_model()
    elif page == "Predict Failure":
        page_predict_failure()
    elif page == "Explain Prediction":
        page_explain_prediction()
    elif page == "Performance Metrics":
        page_performance_metrics()
    elif page == "Download Reports":
        page_download_reports()


if __name__ == "__main__":
    main()