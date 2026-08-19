import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import FIGURES_DIR, RESULTS_DIR
from src.utils import setup_logging

logger = setup_logging("evaluate")


def compute_all_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    return {
        "accuracy": round(
            accuracy_score(y_true, y_pred), 4
        ),
        "precision": round(
            precision_score(y_true, y_pred, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_true, y_pred, zero_division=0), 4
        ),
        "f1": round(
            f1_score(y_true, y_pred, zero_division=0), 4
        ),
        "roc_auc": round(
            roc_auc_score(y_true, y_prob), 4
        ),
    }


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    classes = ["Normal", "Failure"]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix — {model_name}",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    out_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", out_path)
    return out_path


def save_roc_curve(
    y_true: pd.Series,
    all_probs: dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    fig = go.Figure()
    for name, probs in all_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = __import__("sklearn.metrics").metrics.roc_auc_score(y_true, probs)
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} (AUC={auc:.4f})",
            )
        )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash"))
    )
    fig.update_layout(
        title="ROC Curve — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        template="plotly_dark",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info("Saved ROC curve to %s", output_path)
    return output_path


def save_pr_curve(
    y_true: pd.Series,
    all_probs: dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    fig = go.Figure()
    for name, probs in all_probs.items():
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, probs)
        fig.add_trace(
            go.Scatter(
                x=recall_vals, y=precision_vals, mode="lines",
                name=name,
            )
        )
    fig.update_layout(
        title="Precision-Recall Curve — All Models",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend=dict(x=0.01, y=0.01),
        template="plotly_dark",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info("Saved PR curve to %s", output_path)
    return output_path


def save_classification_report(
    y_true: pd.Series,
    all_preds: dict[str, np.ndarray],
    all_probs: dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for name in all_preds:
        lines.append(f"=== {name} ===")
        lines.append(
            classification_report(
                y_true,
                all_preds[name],
                target_names=["Normal", "Failure"],
            )
        )
        lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Saved classification report to %s", output_path)
    return output_path


def compare_models(
    all_metrics: dict[str, dict],
) -> pd.DataFrame:
    df = pd.DataFrame(all_metrics).T
    df = df.sort_values("roc_auc", ascending=False)
    return df


def run_evaluation(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    all_preds: dict[str, np.ndarray] = {}
    all_probs: dict[str, np.ndarray] = {}
    all_metrics: dict[str, dict] = {}

    for name, model in models.items():
        logger.info("Evaluating %s", name)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        all_preds[name] = y_pred
        all_probs[name] = y_prob
        all_metrics[name] = compute_all_metrics(y_test, y_pred, y_prob)
        save_confusion_matrix(y_test, y_pred, name)

    roc_path = RESULTS_DIR / "roc_curve.html"
    pr_path = RESULTS_DIR / "pr_curve.html"
    report_path = RESULTS_DIR / "classification_report.txt"

    save_roc_curve(y_test, all_probs, roc_path)
    save_pr_curve(y_test, all_probs, pr_path)
    save_classification_report(y_test, all_preds, all_probs, report_path)

    comparison_df = compare_models(all_metrics)
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=True)
    logger.info("Model comparison saved to %s", RESULTS_DIR / "model_comparison.csv")

    return comparison_df


__all__ = [
    "compute_all_metrics",
    "save_confusion_matrix",
    "save_roc_curve",
    "save_pr_curve",
    "save_classification_report",
    "compare_models",
    "run_evaluation",
]