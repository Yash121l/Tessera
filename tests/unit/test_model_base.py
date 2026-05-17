"""Tests for model base utilities (deflated Sharpe, model card, registry)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


class TestDeflatedSharpe:
    def test_single_trial_returns_max(self) -> None:
        from tessera.models.base import compute_deflated_sharpe

        sr = np.array([0.5, 1.0, 0.7])
        result = compute_deflated_sharpe(sr, n_obs=100, n_trials=1)
        assert result == pytest.approx(1.0)

    def test_many_trials_penalises(self) -> None:
        from tessera.models.base import compute_deflated_sharpe

        sr = np.array([0.05, 0.08, 0.06, 0.03, 0.04, 0.07, 0.02, 0.09])
        few = compute_deflated_sharpe(sr, n_obs=50, n_trials=3)
        many = compute_deflated_sharpe(sr, n_obs=50, n_trials=1000)
        assert many < few

    def test_empty_returns_zero(self) -> None:
        from tessera.models.base import compute_deflated_sharpe

        assert compute_deflated_sharpe(np.array([]), n_obs=100, n_trials=10) == 0.0

    def test_output_bounded_zero_one(self) -> None:
        from tessera.models.base import compute_deflated_sharpe

        sr = np.array([0.3, 0.4, 0.5, 0.2, 0.1])
        result = compute_deflated_sharpe(sr, n_obs=1000, n_trials=100)
        assert 0.0 <= result <= 1.0


class TestModelCard:
    def test_serialisation_roundtrip(self) -> None:
        from tessera.models.base import CVScores, ModelCard

        card = ModelCard(
            model_name="test",
            model_type="lightgbm_primary",
            training_date="2024-01-01T00:00:00",
            git_commit="abc123",
            data_version="v1",
            features=["f1", "f2"],
            hyperparameters={"lr": 0.1},
            cv_scores=CVScores(
                mean_sharpe=0.5,
                std_sharpe=0.1,
                deflated_sharpe=0.3,
                n_trials=100,
            ),
        )
        raw = card.model_dump_json()
        restored = ModelCard.model_validate_json(raw)
        assert restored.features == card.features
        assert restored.cv_scores is not None
        assert restored.cv_scores.mean_sharpe == pytest.approx(0.5)


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestRegistry:
    def test_save_and_promote(self, tmp_path: object) -> None:
        from tessera.models.lightgbm_model import PrimaryLightGBMModel
        from tessera.models.registry import ModelRegistry

        rng = np.random.RandomState(42)
        feat = pd.DataFrame({"a": rng.randn(100), "b": rng.randn(100)})
        y = pd.Series(rng.choice([-1, 0, 1], size=100))

        model = PrimaryLightGBMModel(seed=42, n_estimators=10)
        model.fit(feat, y)

        registry = ModelRegistry(root=tmp_path)  # type: ignore[arg-type]
        path = registry.save_model(model, "primary")

        assert (path / "model.joblib").exists()
        assert (path / "model_card.json").exists()

        link = registry.promote(path)
        assert link.is_symlink()

    def test_promote_rejects_low_sharpe(self, tmp_path: object) -> None:
        from tessera.models.lightgbm_model import PrimaryLightGBMModel
        from tessera.models.registry import ModelRegistry

        rng = np.random.RandomState(42)
        feat = pd.DataFrame({"a": rng.randn(100)})
        y = pd.Series(rng.choice([-1, 0, 1], size=100))

        model = PrimaryLightGBMModel(seed=42, n_estimators=10)
        model.fit(feat, y)
        model._cv_scores = None

        registry = ModelRegistry(root=tmp_path)  # type: ignore[arg-type]
        path = registry.save_model(model, "primary")

        # No CV scores → cv_scores is None → passes (no gate)
        registry.promote(path, min_sharpe=0.0)

        # Now add low CV scores
        card_path = path / "model_card.json"
        card = json.loads(card_path.read_text())
        card["cv_scores"] = {
            "mean_sharpe": -0.5,
            "std_sharpe": 0.1,
            "deflated_sharpe": 0.0,
            "n_trials": 10,
        }
        card_path.write_text(json.dumps(card))

        with pytest.raises(ValueError, match="below promotion threshold"):
            registry.promote(path, min_sharpe=0.0)

    def test_list_versions(self, tmp_path: object) -> None:
        from tessera.models.lightgbm_model import PrimaryLightGBMModel
        from tessera.models.registry import ModelRegistry

        rng = np.random.RandomState(42)
        feat = pd.DataFrame({"a": rng.randn(100)})
        y = pd.Series(rng.choice([-1, 0, 1], size=100))

        model = PrimaryLightGBMModel(seed=42, n_estimators=10)
        model.fit(feat, y)

        registry = ModelRegistry(root=tmp_path)  # type: ignore[arg-type]
        registry.save_model(model, "primary")
        registry.save_model(model, "primary")

        versions = registry.list_versions("primary")
        assert len(versions) == 2
