import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.config import XGBoost_MODEL_PATH
from src.predict import load_model
from src.utils import setup_logging

logger = setup_logging("explain")


def get_shap_explainer(model):
    try:
        explainer = shap.TreeExplainer(model)
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
        shap_values = explainer.shap_values(X)
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
        shap_values = explainer.shap_values(X)
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
        shap_values = explainer.shap_values(X)
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
        shap_values = explainer.shap_values(X)
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
        shap_values = explainer.shap_values(X)
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
            random_state=42,
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
        shap_values = explainer.shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        feat_importance = pd.Series(mean_abs, index=X.columns)
        return feat_importance.sort_values(ascending=False).head(top_n).index.tolist()
    except Exception as exc:
        logger.error("Failed to compute top SHAP features: %s", exc)
        return list(X.columns[:top_n])


__all__ = [
    "get_shap_explainer",
    "shap_summary_plot",
    "shap_bar_plot",
    "shap_waterfall_plot",
    "shap_force_plot_html",
    "shap_dependence_plots",
    "lime_explain",
    "get_top_features_shap",
]