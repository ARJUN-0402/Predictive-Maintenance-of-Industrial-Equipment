import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from pathlib import Path

from src.explain import (
    get_shap_explainer,
    lime_explain,
    shap_bar_plot,
    shap_dependence_plots,
    shap_force_plot_html,
    shap_summary_plot,
    shap_waterfall_plot,
)


def _make_model_and_data(n_features: int = 5, n_samples: int = 20, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.integers(0, 2, size=n_samples))
    model = xgb.XGBClassifier(
        n_estimators=10,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model, X, y


@pytest.fixture(autouse=True)
def _reset_matplotlib():
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


class TestShapWaterfallPlot:
    def test_executes_and_creates_file(self, tmp_path: Path):
        model, X, _ = _make_model_and_data()
        explainer = get_shap_explainer(model)
        assert explainer is not None

        output_path = tmp_path / "shap_waterfall.png"
        result = shap_waterfall_plot(explainer, X, idx=0, output_path=output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_default_output_path(self, tmp_path: Path, monkeypatch):
        model, X, _ = _make_model_and_data()
        explainer = get_shap_explainer(model)

        figures_dir = tmp_path / "figures"
        monkeypatch.setattr("src.explain.FIGURES_DIR", figures_dir)

        result = shap_waterfall_plot(explainer, X, idx=0)
        assert result == figures_dir / "shap_waterfall.png"
        assert result.exists()
        assert result.stat().st_size > 0


class TestShapSummaryPlot:
    def test_executes_and_creates_file(self, tmp_path: Path):
        model, X, _ = _make_model_and_data()
        explainer = get_shap_explainer(model)
        assert explainer is not None

        output_path = tmp_path / "shap_summary.png"
        result = shap_summary_plot(explainer, X, output_path=output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestShapBarPlot:
    def test_executes_and_creates_file(self, tmp_path: Path):
        model, X, _ = _make_model_and_data()
        explainer = get_shap_explainer(model)
        assert explainer is not None

        output_path = tmp_path / "shap_bar.png"
        result = shap_bar_plot(explainer, X, output_path=output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestShapDependencePlots:
    def test_executes_and_creates_files(self, tmp_path: Path):
        model, X, _ = _make_model_and_data(n_features=5)
        explainer = get_shap_explainer(model)
        assert explainer is not None

        top_features = list(X.columns)[:3]
        output_dir = tmp_path / "dependence"
        result = shap_dependence_plots(explainer, X, top_features, output_dir=output_dir)

        assert len(result) == len(top_features)
        for path in result:
            assert path.exists()
            assert path.stat().st_size > 0

    def test_skips_missing_feature(self, tmp_path: Path):
        model, X, _ = _make_model_and_data()
        explainer = get_shap_explainer(model)

        top_features = ["missing_feature"] + list(X.columns)[:2]
        output_dir = tmp_path / "dependence"
        result = shap_dependence_plots(explainer, X, top_features, output_dir=output_dir)

        assert len(result) == 2
        for path in result:
            assert path.exists()
            assert path.stat().st_size > 0


class TestShapForcePlotHtml:
    def test_executes_and_creates_file(self, tmp_path: Path):
        model, X, _ = _make_model_and_data()
        explainer = get_shap_explainer(model)
        assert explainer is not None

        output_path = tmp_path / "shap_force.html"
        result = shap_force_plot_html(explainer, X, idx=0, output_path=output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestLimeExplain:
    def test_returns_contributions(self):
        model, X, _ = _make_model_and_data()
        result = lime_explain(model, X, X, idx=0, top_features=5)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestSavePlotFigureHelper:
    def test_axes_input(self, tmp_path: Path):
        import matplotlib.pyplot as plt

        from src.explain import _save_plot_figure

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        output_path = tmp_path / "axes_test.png"
        result = _save_plot_figure(ax, output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_figure_input(self, tmp_path: Path):
        from src.explain import _save_plot_figure

        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])

        output_path = tmp_path / "figure_test.png"
        result = _save_plot_figure(fig, output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_none_input_uses_current_figure(self, tmp_path: Path):
        from src.explain import _save_plot_figure

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])

        output_path = tmp_path / "none_test.png"
        result = _save_plot_figure(None, output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0
