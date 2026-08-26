"""Tests for the CLI entry points in src.train, src.evaluate, and
src.threshold_optimization.

These tests mock train_all_models() so they run quickly and do not depend
on network access or produce large artifacts. They verify that the CLI
main() functions execute and return the expected exit codes.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_mock_model(y_pred: np.ndarray, y_prob: np.ndarray) -> MagicMock:
    """Create a mock model with predict/predict_proba returning fixed arrays."""
    model = MagicMock()
    model.predict.return_value = y_pred
    model.predict_proba.return_value = np.column_stack([1 - y_prob, y_prob])
    return model


@pytest.fixture
def mock_train_result(tmp_path: Path) -> dict:
    """Build a minimal train_all_models() result backed by synthetic data.

    Uses 200 samples and 13 feature columns to mirror the real pipeline's
    post-preprocessing shape without training any real model.
    """
    rng = np.random.default_rng(42)
    n = 200
    n_features = 13
    feature_names = [f"f{i}" for i in range(n_features)]

    X_test = pd.DataFrame(rng.standard_normal((n, n_features)), columns=feature_names)
    y_test = pd.Series(rng.integers(0, 2, size=n), name="Machine failure")

    y_prob = rng.uniform(0.0, 1.0, size=n)
    y_pred = (y_prob >= 0.5).astype(int)

    models = {
        name: _make_mock_model(y_pred, y_prob)
        for name in (
            "xgboost",
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
        )
    }

    metrics = {
        name: {
            "accuracy": 0.85,
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65,
            "roc_auc": 0.9,
        }
        for name in models
    }

    return {
        "models": models,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessor": MagicMock(),
        "dataset_info": {"rows": n, "test_size": 0.2},
        "feature_config": {"numeric_features": feature_names},
    }


# ---------------------------------------------------------------------------
# Train CLI
# ---------------------------------------------------------------------------


def test_train_main_returns_zero(mock_train_result: dict) -> None:
    with patch("src.train.train_all_models", return_value=mock_train_result):
        from src.train import main

        exit_code = main()
    assert exit_code == 0


def test_train_main_returns_one_on_failure() -> None:
    with patch(
        "src.train.train_all_models",
        side_effect=RuntimeError("boom"),
    ):
        from src.train import main

        exit_code = main()
    assert exit_code == 1


# ---------------------------------------------------------------------------
# Evaluate CLI
# ---------------------------------------------------------------------------


def test_evaluate_main_returns_zero(mock_train_result: dict) -> None:
    import src.evaluate as evaluate_mod

    with (
        patch("src.train.train_all_models", return_value=mock_train_result),
        patch("src.evaluate.run_evaluation") as mock_run_eval,
    ):
        mock_run_eval.return_value = pd.DataFrame(
            {"roc_auc": [0.9, 0.85, 0.8, 0.75]},
            index=[
                "xgboost",
                "random_forest",
                "gradient_boosting",
                "logistic_regression",
            ],
        )
        exit_code = evaluate_mod.main()

    assert exit_code == 0
    mock_run_eval.assert_called_once_with(
        mock_train_result["models"],
        mock_train_result["X_test"],
        mock_train_result["y_test"],
    )


def test_evaluate_main_returns_one_on_failure() -> None:
    with patch(
        "src.train.train_all_models",
        side_effect=RuntimeError("training failed"),
    ):
        import src.evaluate

        exit_code = src.evaluate.main()
    assert exit_code == 1


# ---------------------------------------------------------------------------
# Threshold optimization CLI
# ---------------------------------------------------------------------------


def test_threshold_main_returns_zero(mock_train_result: dict) -> None:
    with patch(
        "src.train.train_all_models",
        return_value=mock_train_result,
    ):
        from src.threshold_optimization import main

        exit_code = main()
    assert exit_code == 0


def test_threshold_main_returns_one_on_failure() -> None:
    with patch(
        "src.train.train_all_models",
        side_effect=RuntimeError("training failed"),
    ):
        import src.threshold_optimization

        exit_code = src.threshold_optimization.main()
    assert exit_code == 1
