"""PatchTST determinism and parameter budget tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")


def _make_data(n: int = 200, n_features: int = 8, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.choice([-1, 0, 1], size=n))
    return x, y


def test_patchtst_predictions_deterministic_same_seed() -> None:
    """Same seed must produce identical predictions on both fits."""
    from tessera.models.patchtst import PatchTSTModel

    x_train, y_train = _make_data()

    m1 = PatchTSTModel(seed=42, max_epochs=2, batch_size=64)
    m1.fit(x_train, y_train)

    m2 = PatchTSTModel(seed=42, max_epochs=2, batch_size=64)
    m2.fit(x_train, y_train)

    x_test, _ = _make_data(n=50, seed=99)
    p1 = m1.predict(x_test)
    p2 = m2.predict(x_test)

    np.testing.assert_array_equal(
        p1,
        p2,
        err_msg="PatchTST predictions differ across two fits with the same seed",
    )


def test_patchtst_different_seeds_differ() -> None:
    """Different seeds should (almost always) produce different predictions."""
    from tessera.models.patchtst import PatchTSTModel

    x_train, y_train = _make_data()

    m1 = PatchTSTModel(seed=42, max_epochs=3, batch_size=64)
    m1.fit(x_train, y_train)

    m2 = PatchTSTModel(seed=123, max_epochs=3, batch_size=64)
    m2.fit(x_train, y_train)

    x_test, _ = _make_data(n=50, seed=99)
    p1 = m1.predict(x_test)
    p2 = m2.predict(x_test)

    assert not np.array_equal(p1, p2), "Different seeds produced identical outputs"


def test_patchtst_parameter_count_within_budget() -> None:
    """Architecture must stay under the 5M parameter budget."""
    from tessera.models.patchtst import _PatchTSTNet

    for n_features in [8, 40, 100]:
        net = _PatchTSTNet(n_features=n_features)
        n_params = net.count_parameters()
        msg = f"n_features={n_features}: {n_params:,} params exceeds 5M budget"
        assert n_params <= 5_000_000, msg


def test_patchtst_output_shapes() -> None:
    """predict() and predict_proba() must match expected output shapes."""
    from tessera.models.patchtst import PatchTSTModel

    x_train, y_train = _make_data()
    m = PatchTSTModel(seed=0, max_epochs=1, batch_size=64)
    m.fit(x_train, y_train)

    x_test, _ = _make_data(n=40, seed=7)
    preds = m.predict(x_test)
    probas = m.predict_proba(x_test)

    assert preds.shape == (40,), f"Expected (40,), got {preds.shape}"
    assert probas.shape == (40, 3), f"Expected (40, 3), got {probas.shape}"
    assert set(preds).issubset({-1, 0, 1}), f"Unexpected labels: {set(preds)}"
    np.testing.assert_allclose(probas.sum(axis=1), np.ones(40), atol=1e-5)
    assert (probas >= 0).all()


def test_patchtst_save_load_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    """Saved and loaded model must produce identical predictions."""
    from tessera.models.patchtst import PatchTSTModel

    x_train, y_train = _make_data()
    m = PatchTSTModel(seed=42, max_epochs=2, batch_size=64)
    m.fit(x_train, y_train)

    saved_path = m.save(tmp_path / "patchtst_test")
    m2 = PatchTSTModel.load(saved_path)

    x_test, _ = _make_data(n=30, seed=5)
    p1 = m.predict(x_test)
    p2 = m2.predict(x_test)

    np.testing.assert_array_equal(p1, p2, err_msg="Predictions changed after save/load roundtrip")


def test_patchtst_predict_proba_calibration() -> None:
    """The argmax of probas must agree with predict() class labels."""
    from tessera.models.patchtst import _IDX_TO_LABEL, PatchTSTModel

    x_train, y_train = _make_data()
    m = PatchTSTModel(seed=42, max_epochs=2, batch_size=64)
    m.fit(x_train, y_train)

    x_test, _ = _make_data(n=30, seed=11)
    preds = m.predict(x_test)
    probas = m.predict_proba(x_test)

    argmax_labels = np.array([_IDX_TO_LABEL[int(i)] for i in probas.argmax(axis=1)])
    np.testing.assert_array_equal(
        preds,
        argmax_labels,
        err_msg="predict() and argmax(predict_proba()) disagree",
    )
