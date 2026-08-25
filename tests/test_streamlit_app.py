from unittest.mock import MagicMock, patch

import pytest


def test_format_probability_normal() -> None:
    from src.utils import format_probability

    assert format_probability(0.0) == "0.0%"
    assert format_probability(0.5) == "50.0%"
    assert format_probability(0.001) == "0.1%"
    assert format_probability(0.1234) == "12.3%"
    assert format_probability(1.0) == "100.0%"


def test_format_probability_out_of_range() -> None:
    from src.utils import format_probability

    with pytest.raises(ValueError):
        format_probability(-0.1)
    with pytest.raises(ValueError):
        format_probability(1.1)


def test_display_prediction_card_no_format_keyword() -> None:
    from app.streamlit_app import _display_prediction_card

    mock_progress = MagicMock()
    mock_st = MagicMock()
    mock_st.progress = mock_progress

    with patch("app.streamlit_app.st", mock_st):
        _display_prediction_card(
            {"prediction": 0, "probability": 0.001, "risk_level": "Low"},
            0.5,
        )

    assert mock_progress.called, "st.progress should be called"
    call_kwargs = mock_progress.call_args.kwargs
    assert "format" not in call_kwargs, (
        "st.progress must not receive a 'format' keyword argument"
    )
    assert call_kwargs.get("text") == "0.1%", (
        f"Expected text='0.1%', got text={call_kwargs.get('text')!r}"
    )
    assert mock_progress.call_args.args[0] == 0.001, (
        "Progress value should be numeric probability in [0, 1]"
    )


def test_display_prediction_card_failure_format() -> None:
    from app.streamlit_app import _display_prediction_card

    mock_progress = MagicMock()
    mock_st = MagicMock()
    mock_st.progress = mock_progress

    with patch("app.streamlit_app.st", mock_st):
        _display_prediction_card(
            {"prediction": 1, "probability": 0.75, "risk_level": "High"},
            0.5,
        )

    assert mock_progress.called
    call_kwargs = mock_progress.call_args.kwargs
    assert "format" not in call_kwargs
    assert call_kwargs.get("text") == "75.0%"
    assert mock_progress.call_args.args[0] == 0.75
