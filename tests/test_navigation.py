"""Regression tests for page routing and navigation state.

These tests exercise the routing logic directly (without a full Streamlit
runtime) so that navigation regressions are caught deterministically.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.streamlit_app import PAGES
from src.ui_components import (
    DEFAULT_PAGE,
    NAV_GROUPS,
    VALID_PAGES,
    navigate_to_page,
)


def _all_items() -> list[tuple[str, str, str]]:
    return [tuple(item) for group in NAV_GROUPS for item in group["items"]]


class TestNavigationRegistry:
    """The NAV_GROUPS registry is the single source of truth for routes."""

    def test_default_page_is_valid(self) -> None:
        assert DEFAULT_PAGE in VALID_PAGES

    def test_default_page_is_home(self) -> None:
        assert DEFAULT_PAGE == "Home"

    def test_all_page_ids_are_unique(self) -> None:
        page_ids = [page_id for _, page_id, _ in _all_items()]
        assert len(page_ids) == len(set(page_ids))

    def test_all_button_keys_are_unique(self) -> None:
        keys = [key for _, _, key in _all_items()]
        assert len(keys) == len(set(keys))

    def test_button_keys_are_deterministic_strings(self) -> None:
        for _, _, key in _all_items():
            assert isinstance(key, str)
            assert key.startswith("nav_")

    def test_valid_pages_matches_registry(self) -> None:
        assert VALID_PAGES == {page_id for _, page_id, _ in _all_items()}

    def test_ten_routes_defined(self) -> None:
        assert len(_all_items()) == 10

    def test_expected_routes_present(self) -> None:
        page_ids = {page_id for _, page_id, _ in _all_items()}
        expected = {
            "Home",
            "Predict Failure",
            "Explain Prediction",
            "Dataset Overview",
            "Exploratory Data Analysis",
            "Performance Metrics",
            "Threshold Optimization",
            "Reports & Downloads",
            "Model Information",
            "Model Training",
        }
        assert page_ids == expected


class TestPageDispatch:
    """The PAGES dispatch dict must cover every valid route."""

    def test_every_valid_page_has_a_handler(self) -> None:
        for page_id in VALID_PAGES:
            assert page_id in PAGES, f"Missing handler for page: {page_id}"

    def test_no_unknown_pages_in_dispatch(self) -> None:
        for page_id in PAGES:
            assert page_id in VALID_PAGES, f"Dispatch has unknown page: {page_id}"

    def test_all_handlers_are_callable(self) -> None:
        for page_id, handler in PAGES.items():
            assert callable(handler), f"Handler for {page_id} is not callable"

    def test_default_page_handler_is_home(self) -> None:
        from app.streamlit_app import page_home

        assert PAGES[DEFAULT_PAGE] is page_home


class TestNavigateToPage:
    """navigate_to_page is the only mechanism that changes the current page."""

    def _mock_st(self) -> MagicMock:
        mock_st = MagicMock()
        mock_st.session_state = {}
        return mock_st

    def test_valid_page_updates_state_and_reruns(self) -> None:
        mock_st = self._mock_st()
        with patch("src.ui_components.st", mock_st):
            navigate_to_page("Predict Failure")

        assert mock_st.session_state["page"] == "Predict Failure"
        mock_st.rerun.assert_called_once()

    def test_selecting_each_page_sets_correct_state(self) -> None:
        for page_id in VALID_PAGES:
            mock_st = self._mock_st()
            with patch("src.ui_components.st", mock_st):
                navigate_to_page(page_id)
            assert mock_st.session_state["page"] == page_id
            mock_st.rerun.assert_called_once()

    def test_unknown_page_is_ignored(self) -> None:
        mock_st = self._mock_st()
        with patch("src.ui_components.st", mock_st):
            navigate_to_page("Nonexistent Page")

        assert "page" not in mock_st.session_state
        mock_st.rerun.assert_not_called()

    def test_empty_page_is_ignored(self) -> None:
        mock_st = self._mock_st()
        with patch("src.ui_components.st", mock_st):
            navigate_to_page("")

        assert "page" not in mock_st.session_state
        mock_st.rerun.assert_not_called()


class TestActiveStateLogic:
    """Active state must be derived from the current page — exactly one active."""

    def _active_count(self, current_page: str) -> int:
        return sum(
            1
            for group in NAV_GROUPS
            for _, page_id, _ in group["items"]
            if page_id == current_page
        )

    def test_exactly_one_active_for_each_page(self) -> None:
        for page_id in VALID_PAGES:
            assert self._active_count(page_id) == 1, (
                f"Page {page_id!r} should match exactly one nav item"
            )

    def test_default_page_has_exactly_one_active(self) -> None:
        assert self._active_count(DEFAULT_PAGE) == 1

    def test_unknown_page_has_zero_active(self) -> None:
        assert self._active_count("Does Not Exist") == 0
