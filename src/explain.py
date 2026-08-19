import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.config import RANDOM_SEED
from src.predict import load_model
from src.utils import setup_logging

logger = setup_logging("explain")


def get_shap_explainer(model, X: pd.DataFrame | None = None) -> shap.TreeExplainer:
    try:
        if X is not None:
            explainer = shap.TreeExplainer(model, X, feature_perturbation="tree_path_dependent")
        else:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        booster = model.get_booster()
        if booster.feature_names is not None:
            explainer.feature_names = list(booster.feature_names)
        return explainer
    except Exception as exc:
        logger.error("Failed to create SHAP explainer: %s", exc)
        return None


def shap_summary_plot(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    output_path: Path,
) -> Path:
    try:
        shap_values = explainer.shap_values(X, check_additivity=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = shap.summary_plot(
            shap_values, X, show=False, plot_type="dot", max_display=15
        )
        fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
        import matplotlib.pyplot as plt
        plt.close("all")
        logger.info("Saved SHAP summary plot to %s", output_path)
    except Exception as exc:
        logger.error("Failed to generate SHAP summary plot: %s", exc)
        raise
    return output_path


def shap_bar_plot(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    output_path: Path,
) -> Path:
    try:
        shap_values = explainer.shap_values(X, check_additivity=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = shap.summary_plot(
            shap_values, X, show=False, plot_type="bar", max_display=15
        )
        fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
        import matplotlib.pyplot as plt
        plt.close("all")
        logger.info("Saved SHAP bar plot to %s", output_path)
    except Exception as exc:
        logger.error("Failed to generate SHAP bar plot: %s", exc)
        raise
    return output_path


def shap_waterfall_plot(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    idx: int = 0,
    output_path: Path | None = None,
) -> Path:
    try:
        shap_values = explainer.shap_values(X, check_additivity=False)
        if output_path is None:
            output_path = Path("reports/figures/shap_waterfall.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=X.iloc[idx].values,
                feature_names=list(X.columns),
            ),
            show=False,
        )
        fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
        import matplotlib.pyplot as plt
        plt.close("all")
        logger.info("Saved SHAP waterfall plot to %s", output_path)
    except Exception as exc:
        logger.error("Failed to generate SHAP waterfall plot: %s", exc)
        raise
    return output_path


def shap_force_plot_html(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    idx: int = 0,
    output_path: Path | None = None,
) -> Path:
    try:
        shap_values = explainer.shap_values(X, check_additivity=False)
        if output_path is None:
            output_path = Path("reports/figures/shap_force.html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        force_html = shap.force_plot(
            explainer.expected_value,
            shap_values[idx],
            X.iloc[idx],
            matplotlib=False,
            show=False,
        )
        shap.save_html(str(output_path), force_html)
        logger.info("Saved SHAP force plot to %s", output_path)
    except Exception as exc:
        logger.error("Failed to generate SHAP force plot: %s", exc)
        raise
    return output_path


def shap_dependence_plots(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    top_features: list[str],
    output_dir: Path,
) -> list[Path]:
    paths = []
    try:
        shap_values = explainer.shap_values(X, check_additivity=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        for feat in top_features[:3]:
            if feat not in X.columns:
                continue
            out_path = output_dir / f"shap_dependence_{feat}.png"
            fig = shap.dependence_plot(
                feat, shap_values, X, show=False
            )
            import matplotlib.pyplot as plt
            fig.get_figure().savefig(
                str(out_path), bbox_inches="tight", dpi=150
            )
            plt.close("all")
            paths.append(out_path)
            logger.info("Saved SHAP dependence plot for %s", feat)
    except Exception as exc:
        logger.error("Failed to generate SHAP dependence plots: %s", exc)
    return paths


def lime_explain(
    model,
    X_train: pd.DataFrame,
    X_sample: pd.DataFrame,
    idx: int = 0,
    top_features: int = 10,
) -> dict[str, float]:
    try:
        from lime.lime_tabular import LimeTabularExplainer

        explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=list(X_train.columns),
            class_names=["Normal", "Failure"],
            random_state=RANDOM_SEED,
            verbose=False,
            mode="classification",
        )
        exp = explainer.explain_instance(
            X_sample.iloc[idx].values,
            model.predict_proba,
            num_features=top_features,
        )
        return {feat: contrib for feat, contrib in exp.as_list()}
    except Exception as exc:
        logger.error("LIME explanation failed: %s", exc)
        return {}


def get_top_features_shap(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    top_n: int = 5,
) -> list[str]:
    try:
        shap_values = explainer.shap_values(X, check_additivity=False)
        mean_abs = np.abs(shap_values).mean(axis=0)
        feat_importance = pd.Series(mean_abs, index=X.columns)
        return feat_importance.sort_values(ascending=False).head(top_n).index.tolist()
    except Exception as exc:
        logger.error("Failed to compute top SHAP features: %s", exc)
        return list(X.columns[:top_n])


def explain_prediction_plain_english(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    idx: int = 0,
    model=None,
    feature_name_map: dict | None = None,
) -> str:
    """Generate a plain-English explanation for a prediction.

    Maps the top SHAP features for a single prediction to readable
    sentences about what factors contribute toward or away from failure.

    Args:
        explainer: SHAP TreeExplainer instance
        X: Feature DataFrame (must match model's expected features)
        idx: Index of the sample to explain
        model: Trained model instance (optional, required for fallback)
        feature_name_map: Optional dict mapping internal feature names
            to human-readable names

    Returns:
        Plain-English explanation string
    """
    # Try to get SHAP values, gracefully handling categorical features
    shap_values = None
    try:
        shap_values = explainer.shap_values(X, check_additivity=False)
    except NotImplementedError:
        # Categorical features not supported by SHAP; use summary stats
        shap_values = None
    except Exception:
        shap_values = None

    if shap_values is None:
        # Fallback: use SHAP-based feature importances from explainer
        try:
            # Compute mean absolute SHAP values using a subsample if needed
            # Use tree_path_dependent perturbation to handle categorical features
            X_use = X if len(X) <= 100 else X.sample(n=100, random_state=42)
            shap_mean = np.abs(explainer.shap_values(X_use, check_additivity=False)).mean(axis=0)
            if isinstance(shap_mean, pd.Series):
                feat_imp = shap_mean
            else:
                feat_imp = pd.Series(shap_mean, index=list(X.columns))
            top_features = feat_imp.sort_values(ascending=False).head(3).index.tolist()
            top_vals = feat_imp.sort_values(ascending=False).head(3).values.tolist()
        except (NotImplementedError, Exception):
            return "Unable to generate explanation at this time."

        if feature_name_map:
            top_names = [feature_name_map.get(f, f) for f in top_features]
        else:
            top_names = list(top_features)

        pos_parts = [
            f"{name} has strong influence on failure prediction"
            for name in top_names
        ]
        explanation = "Based on overall feature influence, " + ". ".join(pos_parts) + "."
        return explanation

    sample_shap = shap_values[idx]
    sample_data = X.iloc[idx]

    # Get top positive and negative contributors
    positive_idx = np.argsort(sample_shap)[-3:][::-1]  # top 3 positive
    negative_idx = np.argsort(sample_shap)[:3]  # top 3 negative

    pos_features = [sample_data.index[i] for i in positive_idx]
    neg_features = [sample_data.index[i] for i in negative_idx]

    pos_values = [sample_shap[i] for i in positive_idx]
    neg_values = [sample_shap[i] for i in negative_idx]

    # Map feature names if provided
    if feature_name_map:
        pos_names = [feature_name_map.get(f, f) for f in pos_features]
        neg_names = [feature_name_map.get(f, f) for f in neg_features]
    else:
        pos_names = list(pos_features)
        neg_names = list(neg_features)

    parts = []

    if pos_values:
        pos_str = ", ".join(
            f"{name} contributed positively ({val:.4f})"
            for name, val in zip(pos_names, pos_values)
        )
        parts.append(pos_str)

    if neg_values:
        neg_str = ", ".join(
            f"{name} contributed negatively ({val:.4f})"
            for name, val in zip(neg_names, neg_values)
        )
        parts.append(neg_str)

    if not parts:
        return "Unable to generate explanation."

    explanation = "The predicted failure probability is influenced by: " + ". ".join(parts) + "."

    return explanation


__all__ = [
    "get_shap_explainer",
    "shap_summary_plot",
    "shap_bar_plot",
    "shap_waterfall_plot",
    "shap_force_plot_html",
    "shap_dependence_plots",
    "lime_explain",
    "get_top_features_shap",
    "explain_prediction_plain_english",
]