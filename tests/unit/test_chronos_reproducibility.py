"""Chronos zero-shot reproducibility tests (uses a mock pipeline to avoid HF downloads)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")

_NUM_BOLT_QUANTILES = 9  # [0.1, 0.2, ..., 0.9] as in ChronosBoltPipeline


class _MockChronosBoltPipeline:
    """Deterministic mock of ChronosBoltPipeline.

    Returns mean(context) broadcast to all 9 quantile slots.
    Output shape: [batch, num_quantiles=9, prediction_length].
    """

    def predict(
        self,
        context: torch.Tensor,  # type: ignore[name-defined]
        prediction_length: int,
    ) -> torch.Tensor:  # type: ignore[name-defined]
        batch = context.shape[0]
        mean_val = context.mean(dim=-1, keepdim=True)  # [batch, 1]
        # Expand to [batch, num_quantiles, prediction_length]
        return mean_val.unsqueeze(1).expand(batch, _NUM_BOLT_QUANTILES, prediction_length)


def _patch_chronos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tessera.models.chronos_zeroshot._check_deps", lambda: None)
    monkeypatch.setattr(
        "tessera.models.chronos_zeroshot._load_pipeline",
        lambda device="cpu": _MockChronosBoltPipeline(),
    )


def _make_return_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 0.001, n)
    x = pd.DataFrame({"log_return": log_ret, "volume": rng.uniform(1, 10, n)})
    y = pd.Series(rng.choice([-1, 0, 1], size=n))
    return x, y


def test_chronos_predictions_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same input + seed must produce identical predictions."""
    _patch_chronos(monkeypatch)
    from tessera.models.chronos_zeroshot import ChronosZeroShotModel

    x_train, y_train = _make_return_data()
    x_test, _ = _make_return_data(n=80, seed=7)

    m1 = ChronosZeroShotModel(seed=42)
    m1._pipeline = _MockChronosBoltPipeline()
    m1.fit(x_train, y_train)

    m2 = ChronosZeroShotModel(seed=42)
    m2._pipeline = _MockChronosBoltPipeline()
    m2.fit(x_train, y_train)

    p1 = m1.predict(x_test)
    p2 = m2.predict(x_test)

    np.testing.assert_array_equal(
        p1, p2, err_msg="Chronos predictions are not reproducible with the same seed"
    )


def test_chronos_output_in_valid_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """All predictions must be in {-1, 0, +1}."""
    _patch_chronos(monkeypatch)
    from tessera.models.chronos_zeroshot import ChronosZeroShotModel

    x_train, y_train = _make_return_data()
    x_test, _ = _make_return_data(n=50, seed=3)

    m = ChronosZeroShotModel(seed=42)
    m._pipeline = _MockChronosBoltPipeline()
    m.fit(x_train, y_train)

    preds = m.predict(x_test)
    assert set(preds).issubset({-1, 0, 1}), f"Invalid labels: {set(preds)}"
    assert len(preds) == len(x_test)


def test_chronos_output_length_matches_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """predict() must return exactly len(X) predictions."""
    _patch_chronos(monkeypatch)
    from tessera.models.chronos_zeroshot import ChronosZeroShotModel

    x_train, y_train = _make_return_data()
    x_test, _ = _make_return_data(n=100, seed=5)

    m = ChronosZeroShotModel(seed=42)
    m._pipeline = _MockChronosBoltPipeline()
    m.fit(x_train, y_train)

    preds = m.predict(x_test)
    assert len(preds) == len(x_test), f"Expected {len(x_test)} predictions, got {len(preds)}"


def test_chronos_save_load_preserves_threshold(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Threshold calibrated at fit time must survive save/load."""
    _patch_chronos(monkeypatch)
    from tessera.models.chronos_zeroshot import ChronosZeroShotModel

    x_train, y_train = _make_return_data()

    m = ChronosZeroShotModel(seed=42)
    m._pipeline = _MockChronosBoltPipeline()
    m.fit(x_train, y_train)
    original_threshold = m._threshold

    saved = m.save(tmp_path / "chronos_test")
    m2 = ChronosZeroShotModel.load(saved)

    msg = f"Threshold changed after save/load: {original_threshold} → {m2._threshold}"
    assert m2._threshold == original_threshold, msg
