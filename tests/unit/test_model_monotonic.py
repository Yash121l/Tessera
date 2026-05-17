"""Monotonic constraints actually enforce monotonicity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm", reason="requires ml extra: uv sync --extra ml")


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestMonotonicConstraints:
    def test_increasing_constraint(self) -> None:
        """P(positive class) should be non-decreasing in a monotone-increasing feature."""
        from tessera.models.lightgbm_model import MetaLightGBMModel

        rng = np.random.RandomState(42)
        n = 500
        signal = np.linspace(0, 1, n)
        noise = rng.randn(n) * 0.05
        feat = pd.DataFrame({"signal": signal, "noise": noise})  # noqa: N806
        y = pd.Series((signal + noise > 0.5).astype(int))

        model = MetaLightGBMModel(
            monotonic_constraints={"signal": 1},
            seed=42,
            n_estimators=100,
            num_leaves=8,
        )
        model.fit(feat, y)

        test_feat = pd.DataFrame(
            {
                "signal": np.linspace(0, 1, 100),
                "noise": np.zeros(100),
            }
        )
        proba = model.predict_bet_size(test_feat)

        diffs = np.diff(proba)
        assert np.all(diffs >= -1e-10), f"Monotonicity violated: min diff = {diffs.min():.6f}"

    def test_decreasing_constraint(self) -> None:
        """P(positive class) should be non-increasing for a -1 constraint."""
        from tessera.models.lightgbm_model import MetaLightGBMModel

        rng = np.random.RandomState(42)
        n = 500
        signal = np.linspace(0, 1, n)
        feat = pd.DataFrame({"signal": signal, "pad": rng.randn(n) * 0.01})  # noqa: N806
        y = pd.Series((signal < 0.5).astype(int))

        model = MetaLightGBMModel(
            monotonic_constraints={"signal": -1},
            seed=42,
            n_estimators=100,
            num_leaves=8,
        )
        model.fit(feat, y)

        test_feat = pd.DataFrame(
            {
                "signal": np.linspace(0, 1, 100),
                "pad": np.zeros(100),
            }
        )
        proba = model.predict_bet_size(test_feat)

        diffs = np.diff(proba)
        assert np.all(diffs <= 1e-10), f"Monotonicity violated: max diff = {diffs.max():.6f}"

    def test_unconstrained_features_free(self) -> None:
        """Features without constraints should not be forced monotone."""
        from tessera.models.lightgbm_model import MetaLightGBMModel

        n = 500
        x = np.linspace(-2, 2, n)
        feat = pd.DataFrame({"x": x})  # noqa: N806
        y = pd.Series((np.abs(x) < 1).astype(int))

        model = MetaLightGBMModel(seed=42, n_estimators=100, num_leaves=16)
        model.fit(feat, y)

        test_feat = pd.DataFrame({"x": np.linspace(-2, 2, 100)})
        proba = model.predict_bet_size(test_feat)

        assert proba.max() > proba[0]
        assert proba.max() > proba[-1]
