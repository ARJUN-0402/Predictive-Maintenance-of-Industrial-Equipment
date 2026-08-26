import pandas as pd
from pathlib import Path
import pytest

from src.predict import predict, batch_predict, _classify_risk


def test_classify_risk_high() -> None:
    assert _classify_risk(0.95, threshold=0.5) == "High"
    assert _classify_risk(0.701, threshold=0.5) == "High"


def test_classify_risk_medium() -> None:
    assert _classify_risk(0.5, threshold=0.5) == "Medium"
    assert _classify_risk(0.3, threshold=0.5) == "Medium"


def test_classify_risk_low() -> None:
    assert _classify_risk(0.1, threshold=0.5) == "Low"
    assert _classify_risk(0.299, threshold=0.5) == "Low"


def test_classify_risk_not_hardcoded_low() -> None:
    levels = {_classify_risk(p, threshold=0.5) for p in [0.05, 0.4, 0.55, 0.8, 0.99]}
    assert levels == {"Low", "Medium", "High"}


def test_predict_risk_level_derived_not_hardcoded(normal_sample: dict) -> None:
    try:
        result = predict(normal_sample, threshold=0.5)
    except FileNotFoundError:
        pytest.skip("Model file not found; skipping prediction test")
    assert result["risk_level"] in ("Low", "Medium", "High")
    assert result["risk_level"] == _classify_risk(result["probability"], threshold=0.5)


def test_predict_risk_level_matches_classifier(failure_sample: dict) -> None:
    try:
        result = predict(failure_sample, threshold=0.5)
    except FileNotFoundError:
        pytest.skip("Model file not found; skipping prediction test")
    assert result["risk_level"] == _classify_risk(result["probability"], threshold=0.5)
    assert result["risk_level"] != "Low" or result["probability"] < 0.3


@pytest.fixture
def normal_sample() -> dict:
    return {
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50.0,
        "Type": "M",
    }


@pytest.fixture
def failure_sample() -> dict:
    return {
        "Air temperature [K]": 315.0,
        "Process temperature [K]": 340.0,
        "Rotational speed [rpm]": 2500.0,
        "Torque [Nm]": 80.0,
        "Tool wear [min]": 250.0,
        "Type": "H",
    }


def test_predict_returns_0_or_1(normal_sample: dict) -> None:
    try:
        result = predict(normal_sample, threshold=0.5)
        assert result["prediction"] in (
            0,
            1,
        ), f"Prediction must be 0 or 1, got {result['prediction']}"
    except FileNotFoundError:
        pytest.skip("Model file not found; skipping prediction test")


def test_predict_returns_0_or_1_failure_sample(failure_sample: dict) -> None:
    try:
        result = predict(failure_sample, threshold=0.5)
        assert result["prediction"] in (
            0,
            1,
        ), f"Prediction must be 0 or 1, got {result['prediction']}"
    except FileNotFoundError:
        pytest.skip("Model file not found; skipping prediction test")


def test_predict_probability_range(normal_sample: dict) -> None:
    try:
        result = predict(normal_sample, threshold=0.5)
        assert (
            0.0 <= result["probability"] <= 1.0
        ), f"Probability must be in [0,1], got {result['probability']}"
    except FileNotFoundError:
        pytest.skip("Model file not found; skipping prediction test")


def test_predict_risk_level(normal_sample: dict) -> None:
    try:
        result = predict(normal_sample, threshold=0.5)
        assert result["risk_level"] in (
            "Low",
            "Medium",
            "High",
        ), f"Risk level must be Low/Medium/High, got {result['risk_level']}"
    except FileNotFoundError:
        pytest.skip("Model file not found; skipping prediction test")


def test_batch_predict_with_csv(tmp_path: "Path") -> None:
    import os

    df = pd.DataFrame(
        {
            "Air temperature [K]": [298.0, 315.0],
            "Process temperature [K]": [310.0, 340.0],
            "Rotational speed [rpm]": [1500.0, 2500.0],
            "Torque [Nm]": [40.0, 80.0],
            "Tool wear [min]": [50.0, 250.0],
            "Type": ["M", "H"],
        }
    )
    csv_path = tmp_path / "test_batch.csv"
    df.to_csv(csv_path, index=False)

    try:
        results = batch_predict(csv_path)
        assert len(results) == 2
        for _, row in results.iterrows():
            assert row["prediction"] in (0, 1, -1)
            assert 0.0 <= row["probability"] <= 1.0
    except FileNotFoundError:
        pytest.skip("Model file not found; skipping batch prediction test")
    finally:
        if csv_path.exists():
            os.remove(csv_path)


def test_predict_with_custom_threshold(normal_sample: dict) -> None:
    try:
        result = predict(normal_sample, threshold=0.3)
        assert result["prediction"] in (0, 1)
    except FileNotFoundError:
        pytest.skip("Model file not found; skipping prediction test")
