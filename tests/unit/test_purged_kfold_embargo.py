"""Test that embargo region is excluded from training in PurgedKFold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tessera.cv.purged_kfold import PurgedKFold


def _make_non_overlapping_labels(n: int) -> pd.Series:  # type: ignore[type-arg]
    """Each event has t1 = same bar (no overlap), isolating embargo behavior."""
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    return pd.Series(idx, index=idx)


def test_embargo_excludes_samples_after_test() -> None:
    """Samples immediately after the test fold must be excluded."""
    n = 100
    t1 = _make_non_overlapping_labels(n)
    x = np.arange(n)

    pct_embargo = 0.1  # 10% = 10 samples
    cv = PurgedKFold(n_splits=5, samples_info_sets=t1, pct_embargo=pct_embargo)

    for fold_idx, (train, test) in enumerate(cv.split(x)):
        test_end = test[-1]
        embargo_end = min(test_end + 1 + int(n * pct_embargo), n)
        embargo_range = set(range(test_end + 1, embargo_end))

        # No embargoed sample should appear in training
        train_set = set(train.tolist())
        overlap = train_set & embargo_range
        assert len(overlap) == 0, f"Fold {fold_idx}: embargo samples {overlap} found in training"


def test_embargo_zero_means_no_exclusion() -> None:
    """With pct_embargo=0, only purging applies (no extra exclusion)."""
    n = 100
    t1 = _make_non_overlapping_labels(n)
    x = np.arange(n)

    cv = PurgedKFold(n_splits=5, samples_info_sets=t1, pct_embargo=0.0)

    for train, test in cv.split(x):
        # With non-overlapping labels and no embargo:
        # train should be all non-test samples
        assert len(train) + len(test) == n


def test_embargo_with_purging_combined() -> None:
    """Embargo + purging together: both exclusion mechanisms must work."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    # Windows of 5 bars
    t1 = pd.Series([idx[min(i + 5, n - 1)] for i in range(n)], index=idx)
    x = np.arange(n)

    cv = PurgedKFold(n_splits=5, samples_info_sets=t1, pct_embargo=0.05)

    for _fold_idx, (train, test) in enumerate(cv.split(x)):
        test_end = test[-1]
        embargo_size = int(n * 0.05)
        embargo_range = set(range(test_end + 1, min(test_end + 1 + embargo_size, n)))

        # Embargo samples excluded
        train_set = set(train.tolist())
        assert len(train_set & embargo_range) == 0

        # No time overlap between any train and test window
        for ti in train:
            t_t0 = t1.index[ti]
            t_t1 = t1.iloc[ti]
            for tj in test:
                test_t0 = t1.index[tj]
                test_t1 = t1.iloc[tj]
                assert not (t_t0 <= test_t1 and t_t1 >= test_t0)
