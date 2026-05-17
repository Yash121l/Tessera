"""Training is deterministic given the same seed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_data(n: int = 200, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n), "c": rng.randn(n)})  # noqa: N806
    y = pd.Series(rng.choice([-1, 0, 1], size=n), name="label")
    return X, y


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestDeterminism:
    def test_primary_deterministic(self) -> None:
        from tessera.models.lightgbm_model import PrimaryLightGBMModel

        X, y = _make_data()  # noqa: N806

        m1 = PrimaryLightGBMModel(seed=99, n_estimators=30)
        m1.fit(X, y)

        m2 = PrimaryLightGBMModel(seed=99, n_estimators=30)
        m2.fit(X, y)

        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))
        np.testing.assert_allclose(m1.predict_proba(X), m2.predict_proba(X), atol=1e-12)

    def test_meta_deterministic(self) -> None:
        from tessera.models.lightgbm_model import MetaLightGBMModel

        rng = np.random.RandomState(42)
        X = pd.DataFrame({"a": rng.randn(200), "b": rng.randn(200)})  # noqa: N806
        y = pd.Series(rng.choice([0, 1], size=200))

        m1 = MetaLightGBMModel(seed=99, n_estimators=30)
        m1.fit(X, y)

        m2 = MetaLightGBMModel(seed=99, n_estimators=30)
        m2.fit(X, y)

        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))
        np.testing.assert_allclose(m1.predict_proba(X), m2.predict_proba(X), atol=1e-12)

    def test_different_seeds_differ(self) -> None:
        from tessera.models.lightgbm_model import PrimaryLightGBMModel

        X, y = _make_data()  # noqa: N806

        m1 = PrimaryLightGBMModel(seed=1, n_estimators=30, bagging_fraction=0.7, bagging_freq=1)
        m1.fit(X, y)

        m2 = PrimaryLightGBMModel(seed=9999, n_estimators=30, bagging_fraction=0.7, bagging_freq=1)
        m2.fit(X, y)

        proba1 = m1.predict_proba(X)
        proba2 = m2.predict_proba(X)
        assert not np.allclose(proba1, proba2, atol=1e-6)
