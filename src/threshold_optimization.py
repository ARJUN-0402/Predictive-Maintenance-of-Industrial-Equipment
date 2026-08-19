import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, f1_score, recall_score, precision_score

from src.config import RANDOM_SEED

logger = logging.getLogger("threshold_optimization")


def optimize_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold_range: tuple = (0.05, 0.95),
    step: float = 0.01,
) -> dict:
    """Find optimal classification threshold based on multiple criteria.

    Evaluates thresholds across a reasonable range and computes:
    - Precision, Recall, F1 for each threshold
    - False Positives and False Negatives counts

    Returns the threshold that maximizes F1, plus additional information
    for decision-making.

    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities for the positive class
        threshold_range: (min, max) range for threshold search (default 0.05 to 0.95)
        step: Step size for threshold iteration (default 0.01)

    Returns:
        dict with optimization results:
            - best_threshold: threshold maximizing F1
            - best_f1: F1 score at best threshold
            - best_precision: precision at best threshold
            - best_recall: recall at best threshold
            - best_fp: false positives at best threshold
            - best_fn: false negatives at best threshold
            - threshold_data: DataFrame with metrics per threshold
    """
    thresholds = np.arange(
        threshold_range[0], threshold_range[1] + step, step
    )

    precision_vals = []
    recall_vals = []
    f1_vals = []
    fp_vals = []
    fn_vals = []
    tp_vals = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()

        p = precision_score(y_true, y_pred, zero_division=0) if len(y_pred) > 0 else 0
        r = recall_score(y_true, y_pred, zero_division=0) if len(y_pred) > 0 else 0
        f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_pred) > 0 else 0

        precision_vals.append(p)
        recall_vals.append(r)
        f1_vals.append(f1)
        fp_vals.append(fp)
        fn_vals.append(fn)
        tp_vals.append(tp)

    results_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision_vals,
        "recall": recall_vals,
        "f1": f1_vals,
        "false_positives": fp_vals,
        "false_negatives": fn_vals,
    })

    # Find threshold that maximizes F1
    best_idx = f1_vals.index(max(f1_vals))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_vals[best_idx])
    best_precision = float(precision_vals[best_idx])
    best_recall = float(recall_vals[best_idx])
    best_fp = int(fp_vals[best_idx])
    best_fn = int(fn_vals[best_idx])

    logger.info(
        "Threshold optimization complete: best_threshold=%.4f, f1=%.4f, precision=%.4f, recall=%.4f, fp=%d, fn=%d",
        best_threshold, best_f1, best_precision, best_recall, best_fp, best_fn,
    )

# Generate precision-recall curve plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall_vals, y=precision_vals,
            mode="lines", name="Precision-Recall Curve",
            line=dict(color="#00d4ff", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[best_recall], y=[best_precision],
            mode="markers", name=f"Best Threshold ({best_threshold:.2f})",
            marker=dict(color="red", size=10),
        )
    )
    fig.update_layout(
        title="Precision-Recall Curve with Optimal Threshold",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_dark",
    )
    fig.write_html("reports/figures/precision_recall_curve.html")
    logger.info("Saved precision-recall curve to reports/figures/precision_recall_curve.html")

    # Generate threshold analysis plot
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(x=thresholds, y=f1_vals, mode="lines", name="F1 Score",
                   line=dict(color="#00d4ff", width=2)),
    )
    fig2.add_trace(
        go.Scatter(x=thresholds, y=precision_vals, mode="lines", name="Precision",
                   line=dict(color="#00c853", width=2)),
    )
    fig2.add_trace(
        go.Scatter(x=thresholds, y=recall_vals, mode="lines", name="Recall",
                   line=dict(color="#ffab00", width=2)),
    )
    fig2.add_trace(
        go.Scatter(x=thresholds, y=[fn / (fn + tp + 1e-10) * 100
                         for fn, tp in zip(fn_vals, tp_vals)],
                   mode="lines", name="False Negative %",
                   line=dict(color="red", width=2, dash="dash")),
    )
    fig2.update_layout(
        title="Threshold Analysis",
        xaxis_title="Threshold",
        yaxis_title="Percentage",
        template="plotly_dark",
        legend=dict(x=0.02, y=0.98),
    )
    fig2.write_html("reports/figures/threshold_analysis.html")
    logger.info("Saved threshold analysis to reports/figures/threshold_analysis.html")

    return {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_false_positives": best_fp,
        "best_false_negatives": best_fn,
        "threshold_data": results_df,
    }


def recommend_threshold_for_recall(
    y_true: pd.Series,
    y_prob: np.ndarray,
    minimum_recall: float = 0.80,
    threshold_range: tuple = (0.05, 0.95),
    step: float = 0.01,
) -> dict:
    """Recommend a threshold that achieves at least a minimum recall.

    In predictive maintenance, false negatives (missing a failure) can be
    costly, so we often prioritize high recall even at the expense of
    precision.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        minimum_recall: Minimum recall requirement (default 0.80)
        threshold_range: Threshold search range
        step: Step size

    Returns:
        dict with recommended threshold and associated metrics
    """
    thresholds = np.arange(
        threshold_range[0], threshold_range[1] + step, step
    )

    best_threshold = None
    best_precision = 0
    best_f1 = 0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        r = recall_score(y_true, y_pred, zero_division=0)

        if r >= minimum_recall:
            p = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if best_threshold is None or f1 > best_f1:
                best_threshold = float(t)
                best_precision = float(p)
                best_f1 = float(f1)

    if best_threshold is None:
        # If no threshold achieves the minimum recall, return the one with highest recall
        recalls = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            r = recall_score(y_true, y_pred, zero_division=0)
            recalls.append((t, r))
        recalls.sort(key=lambda x: x[1], reverse=True)
        best_threshold = float(recalls[0][0])
        best_recall = float(recalls[0][1])
        best_precision = 0.0
        best_f1 = 0.0
    else:
        y_pred = (y_prob >= best_threshold).astype(int)
        best_recall = recall_score(y_true, y_pred, zero_division=0)
        best_precision = precision_score(y_true, y_pred, zero_division=0)
        best_f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(
        "Recall-based threshold recommendation: threshold=%.4f, recall=%.4f, precision=%.4f, f1=%.4f",
        best_threshold, best_recall, best_precision, best_f1,
    )

    return {
        "threshold": best_threshold,
        "recall": best_recall,
        "precision": best_precision,
        "f1": best_f1,
        "objective": "maximize recall subject to minimum recall constraint",
    }