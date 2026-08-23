from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, REPORTS_DIR
from src.utils import setup_logging

logger = setup_logging("evaluate")


def compute_all_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
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
    output_dir: Path = FIGURES_DIR,
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
    out_path = output_dir / f"confusion_matrix_{model_name}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
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
        auc = roc_auc_score(y_true, probs)
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
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = REPORTS_DIR,
) -> pd.DataFrame:
    all_preds: dict[str, np.ndarray] = {}
    all_probs: dict[str, np.ndarray] = {}
    all_metrics: dict[str, dict] = {}

    figures_dir = output_dir / "figures"
    results_dir = output_dir / "results"

    for name, model in models.items():
        logger.info("Evaluating %s", name)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        all_preds[name] = y_pred
        all_probs[name] = y_prob
        all_metrics[name] = compute_all_metrics(y_test, y_pred, y_prob)
        save_confusion_matrix(y_test, y_pred, name, figures_dir)

    roc_path = results_dir / "roc_curve.html"
    pr_path = results_dir / "pr_curve.html"
    report_path = results_dir / "classification_report.txt"

    save_roc_curve(y_test, all_probs, roc_path)
    save_pr_curve(y_test, all_probs, pr_path)
    save_classification_report(y_test, all_preds, all_probs, report_path)

    comparison_df = compare_models(all_metrics)
    comparison_df.to_csv(results_dir / "model_comparison.csv", index=True)
    logger.info("Model comparison saved to %s", results_dir / "model_comparison.csv")

    return comparison_df


__all__ = [
    "compute_all_metrics",
    "save_confusion_matrix",
    "save_roc_curve",
    "save_pr_curve",
    "save_classification_report",
    "compare_models",
    "run_evaluation",
    "main",
]


def main() -> int:
    """CLI entry point: train models and run full evaluation.

    Trains all models via the shared training pipeline, then evaluates them
    on the held-out test set. Generates confusion matrices, ROC/PR curves,
    a classification report, and a model comparison CSV under reports/.
    Prints the model comparison table to stdout.

    Returns:
        0 on success, 1 on failure.
    """
    try:
        from src.train import train_all_models

        logger.info("=== Training models ===")
        results = train_all_models()

        logger.info("=== Running evaluation ===")
        comparison_df = run_evaluation(
            results["models"], results["X_test"], results["y_test"]
        )

        print("\n=== Model Comparison (sorted by ROC-AUC) ===\n")
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 200,
            "display.float_format", "{:.4f}".format,
        ):
            print(comparison_df.to_string())
        print()

        logger.info("=== Evaluation artifacts saved to reports/ ===")
        return 0
    except Exception as exc:
        logger.error("Evaluation pipeline failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
