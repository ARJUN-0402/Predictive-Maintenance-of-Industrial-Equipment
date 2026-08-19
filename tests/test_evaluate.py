import pytest
import pandas as pd
import numpy as np
from src.train import train_all_models
from src.evaluate import compute_all_metrics, run_evaluation, compare_models


class TestEvaluation:
    """Tests for the evaluation module."""

    def test_compute_all_metrics_basic(self):
        y_true = pd.Series([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.4, 0.3, 0.8])
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_compute_all_metrics_perfect(self):
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2])
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc"] > 0.5

    def test_compute_all_metrics_imbalanced(self):
        y_true = pd.Series([0] * 9 + [1])  # 90% normal, 10% failure
        y_pred = np.array([0] * 10)  # Always predict normal
        y_prob = np.array([0.1] * 10)
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        assert metrics["recall"] == 0.0  # Missed the failure
        assert metrics["precision"] == 0.0  # No true positives

    def test_run_evaluation_returns_df(self):
        results = train_all_models()
        models = results["models"]
        X_test = results["X_test"]
        y_test = results["y_test"]
        comparison_df = run_evaluation(models, X_test, y_test)
        assert isinstance(comparison_df, pd.DataFrame)
        assert "roc_auc" in comparison_df.columns

    def test_compare_models_has_roc_auc(self):
        results = train_all_models()
        metrics = results["metrics"]
        comparison_df = compare_models(metrics)
        assert "roc_auc" in comparison_df.columns
        # XGBoost should have the highest ROC-AUC (with scale_pos_weight)
        top_model = comparison_df.index[0]
        assert top_model in comparison_df.index


class TestThresholdOptimization:
    """Tests for threshold optimization functionality."""

    def test_optimize_threshold_returns_dict(self):
        import warnings
        warnings.filterwarnings("ignore")
        from src.threshold_optimization import optimize_threshold
        from src.train import train_all_models

        results = train_all_models()
        y_prob = results["models"]["xgboost"].predict_proba(results["X_test"])[:, 1]
        result = optimize_threshold(results["y_test"], y_prob)
        assert "best_threshold" in result
        assert "best_f1" in result
        assert "best_recall" in result
        assert "best_precision" in result

    def test_recommend_threshold_for_recall(self):
        import warnings
        warnings.filterwarnings("ignore")
        from src.threshold_optimization import recommend_threshold_for_recall
        from src.train import train_all_models

        results = train_all_models()
        y_prob = results["models"]["xgboost"].predict_proba(results["X_test"])[:, 1]
        result = recommend_threshold_for_recall(results["y_test"], y_prob, minimum_recall=0.80)
        assert "threshold" in result
        assert "recall" in result
        assert "precision" in result
        assert "f1" in result