import logging
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics import f1_score, precision_score, recall_score

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
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

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

    best_idx = f1_vals.index(max(f1_vals))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_vals[best_idx])
    best_precision = float(precision_vals[best_idx])
    best_recall = float(recall_vals[best_idx])
    best_fp = int(fp_vals[best_idx])
    best_fn = int(fn_vals[best_idx])

    logger.info(
        "Threshold optimization complete: best_threshold=%.4f, f1=%.4f, "
        "precision=%.4f, recall=%.4f, fp=%d, fn=%d",
        best_threshold, best_f1, best_precision, best_recall, best_fp, best_fn,
    )

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

    fig2 = go.Figure()
    fnpr = [100 * fn / (fn + tp + 1e-10) for fn, tp in zip(fn_vals, tp_vals)]
    fig2.add_trace(
        go.Scatter(
            x=thresholds, y=f1_vals,
            mode="lines", name="F1 Score",
            line=dict(color="#00d4ff", width=2),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=thresholds, y=precision_vals,
            mode="lines", name="Precision",
            line=dict(color="#00c853", width=2),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=thresholds, y=recall_vals,
            mode="lines", name="Recall",
            line=dict(color="#ffab00", width=2),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=thresholds, y=fnpr,
            mode="lines", name="False Negative %",
            line=dict(color="red", width=2, dash="dash"),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=[best_threshold], y=[best_f1],
            mode="markers", name=f"Best Threshold ({best_threshold:.2f})",
            marker=dict(color="white", size=10, symbol="star"),
        )
    )
    fig2.update_layout(
        title="Threshold Analysis: F1 / Precision / Recall / FN% vs Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score / Percentage",
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
                best_precision = float(p)  # type: ignore[assignment]
                best_f1 = float(f1)  # type: ignore[assignment]

    if best_threshold is None:
        recalls = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            r = recall_score(y_true, y_pred, zero_division=0)
            recalls.append((t, r))
        recalls.sort(key=lambda x: x[1], reverse=True)
        best_threshold = float(recalls[0][0])
        best_recall = float(recalls[0][1])
        best_precision = 0.0  # type: ignore[assignment]
        best_f1 = 0.0  # type: ignore[assignment]
    else:
        y_pred = (y_prob >= best_threshold).astype(int)
        best_recall = recall_score(y_true, y_pred, zero_division=0)
        best_precision = precision_score(y_true, y_pred, zero_division=0)
        best_f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(
        "Recall-based threshold recommendation: threshold=%.4f, recall=%.4f, "
        "precision=%.4f, f1=%.4f",
        best_threshold, best_recall, best_precision, best_f1,
    )

    return {
        "threshold": best_threshold,
        "recall": best_recall,
        "precision": best_precision,
        "f1": best_f1,
        "objective": "maximize recall subject to minimum recall constraint",
    }


def main() -> int:
    """CLI entry point: train models and run threshold optimization.

    Trains all models via the shared training pipeline, obtains XGBoost
    predicted probabilities on the held-out test set, then runs both
    threshold optimization strategies. Prints the best-F1 and
    recall-oriented thresholds to stdout and generates the HTML threshold
    reports under reports/figures/.

    Returns:
        0 on success, 1 on failure.
    """
    try:
        from src.train import train_all_models

        logger.info("=== Training models ===")
        results = train_all_models()

        y_prob = results["models"]["xgboost"].predict_proba(results["X_test"])[:, 1]
        y_test = results["y_test"]

        logger.info("=== Threshold optimization (maximize F1) ===")
        result = optimize_threshold(y_test, y_prob)

        print("\n=== Threshold Optimization (maximize F1) ===\n")
        print(f"  Best threshold:      {result['best_threshold']:.4f}")
        print(f"  Best F1:             {result['best_f1']:.4f}")
        print(f"  Precision:           {result['best_precision']:.4f}")
        print(f"  Recall:              {result['best_recall']:.4f}")
        print(f"  False positives:     {result['best_false_positives']}")
        print(f"  False negatives:     {result['best_false_negatives']}")
        print()

        logger.info("=== Recall-oriented threshold (min recall >= 0.80) ===")
        rec_result = recommend_threshold_for_recall(
            y_test, y_prob, minimum_recall=0.80
        )

        print("=== Recall-Oriented Threshold (min recall >= 0.80) ===\n")
        print(f"  Recommended threshold: {rec_result['threshold']:.4f}")
        print(f"  Recall:               {rec_result['recall']:.4f}")
        print(f"  Precision:            {rec_result['precision']:.4f}")
        print(f"  F1:                   {rec_result['f1']:.4f}")
        print()

        logger.info("=== Threshold reports saved to reports/figures/ ===")
        return 0
    except Exception as exc:
        logger.error("Threshold optimization pipeline failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
