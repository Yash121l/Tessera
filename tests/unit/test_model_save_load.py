"""Round-trip save/load preserves predictions exactly (within float tolerance)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm", reason="requires ml extra: uv sync --extra ml")


def _make_data(n: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(  # noqa: N806
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "f3": rng.randn(n),
        }
    )
    y = pd.Series(rng.choice([-1, 0, 1], size=n), name="label")
    return X, y


def _make_binary_data(n: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})  # noqa: N806
    y = pd.Series(rng.choice([0, 1], size=n), name="label")
    return X, y


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestPrimaryRoundTrip:
    def test_predict_preserved(self, tmp_path: object) -> None:
        from tessera.models.lightgbm_model import PrimaryLightGBMModel

        X, y = _make_data()  # noqa: N806
        model = PrimaryLightGBMModel(seed=42, n_estimators=20)
        model.fit(X, y)

        pred_before = model.predict(X)
        proba_before = model.predict_proba(X)

        save_dir = tmp_path / "primary_test"  # type: ignore[operator]
        model.save(save_dir)

        loaded = PrimaryLightGBMModel.load(save_dir)
        pred_after = loaded.predict(X)
        proba_after = loaded.predict_proba(X)

        np.testing.assert_array_equal(pred_before, pred_after)
        np.testing.assert_allclose(proba_before, proba_after, atol=1e-10)

    def test_model_card_persisted(self, tmp_path: object) -> None:
        from tessera.models.lightgbm_model import PrimaryLightGBMModel

        X, y = _make_data()  # noqa: N806
        model = PrimaryLightGBMModel(seed=42, n_estimators=10)
        model.fit(X, y)

        save_dir = tmp_path / "card_test"  # type: ignore[operator]
        model.save(save_dir)

        loaded = PrimaryLightGBMModel.load(save_dir)
        card = loaded.get_model_card()

        assert card.model_type == "lightgbm_primary"
        assert card.features == ["f1", "f2", "f3"]
        assert "n_estimators" in card.hyperparameters


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestMetaRoundTrip:
    def test_predict_preserved(self, tmp_path: object) -> None:
        from tessera.models.lightgbm_model import MetaLightGBMModel

        X, y = _make_binary_data()  # noqa: N806
        model = MetaLightGBMModel(seed=42, n_estimators=20)
        model.fit(X, y)

        pred_before = model.predict(X)
        proba_before = model.predict_proba(X)
        bet_before = model.predict_bet_size(X)

        save_dir = tmp_path / "meta_test"  # type: ignore[operator]
        model.save(save_dir)

        loaded = MetaLightGBMModel.load(save_dir)
        pred_after = loaded.predict(X)
        proba_after = loaded.predict_proba(X)
        bet_after = loaded.predict_bet_size(X)

        np.testing.assert_array_equal(pred_before, pred_after)
        np.testing.assert_allclose(proba_before, proba_after, atol=1e-10)
        np.testing.assert_allclose(bet_before, bet_after, atol=1e-10)
