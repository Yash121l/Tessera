"""Test CPCV split count and backtest path count."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tessera.cv.combinatorial_purged import CombinatorialPurgedKFold


def _make_labels(n: int, window: int = 5) -> pd.Series:  # type: ignore[type-arg]
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    return pd.Series([idx[min(i + window, n - 1)] for i in range(n)], index=idx)


def test_cpcv_6_2_yields_15_splits() -> None:
    """C(6, 2) = 15 splits."""
    n = 600
    t1 = _make_labels(n)
    x = np.arange(n)

    cv = CombinatorialPurgedKFold(
        n_splits=6, n_test_splits=2, samples_info_sets=t1, pct_embargo=0.0
    )

    splits = list(cv.split(x))
    assert len(splits) == 15
    assert cv.get_n_splits() == 15


def test_cpcv_6_2_yields_5_backtest_paths() -> None:
    """C(5, 1) = 5 backtest paths for (N=6, k=2)."""
    n_paths = CombinatorialPurgedKFold.get_num_backtest_paths(n_splits=6, n_test_splits=2)
    assert n_paths == 5


def test_cpcv_no_train_test_overlap() -> None:
    """CPCV must also maintain the purging invariant."""
    n = 300
    window = 10
    t1 = _make_labels(n, window)
    x = np.arange(n)

    cv = CombinatorialPurgedKFold(
        n_splits=6, n_test_splits=2, samples_info_sets=t1, pct_embargo=0.0
    )

    for split_idx, (train, test) in enumerate(cv.split(x)):
        for ti in train:
            t_t0 = t1.index[ti]
            t_t1 = t1.iloc[ti]
            for tj in test:
                test_t0 = t1.index[tj]
                test_t1 = t1.iloc[tj]
                overlaps = t_t0 <= test_t1 and t_t1 >= test_t0
                assert not overlaps, f"Split {split_idx}: train {ti} overlaps test {tj}"


def test_cpcv_train_test_disjoint() -> None:
    """Train and test indices must be disjoint in every split."""
    n = 300
    t1 = _make_labels(n)
    x = np.arange(n)

    cv = CombinatorialPurgedKFold(
        n_splits=6, n_test_splits=2, samples_info_sets=t1, pct_embargo=0.01
    )

    for train, test in cv.split(x):
        assert len(set(train) & set(test)) == 0


def test_cpcv_different_configs() -> None:
    """Verify split counts for various (N, k) configurations."""
    assert CombinatorialPurgedKFold.get_num_backtest_paths(5, 2) == 4
    assert CombinatorialPurgedKFold.get_num_backtest_paths(10, 2) == 9
    assert CombinatorialPurgedKFold.get_num_backtest_paths(6, 3) == 10
