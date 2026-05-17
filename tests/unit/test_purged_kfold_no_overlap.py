"""NON-NEGOTIABLE TEST: No training label window may overlap any test label window.

This test constructs explicit [t0, t1] label windows and verifies that for
every fold, purging removes all training samples whose windows overlap the
test set's time range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tessera.cv.purged_kfold import PurgedKFold


def _make_overlapping_labels(n: int, window_size: int) -> pd.Series:  # type: ignore[type-arg]
    """Create label windows where each event spans window_size bars.

    This intentionally creates overlapping windows to test purging.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    t1_values = []
    for i in range(n):
        end_idx = min(i + window_size, n - 1)
        t1_values.append(idx[end_idx])
    return pd.Series(t1_values, index=idx)


class TestPurgedKFoldNoOverlap:
    """Every test here verifies the fundamental purging invariant."""

    def test_no_train_test_overlap_basic(self) -> None:
        """Basic case: 100 samples, window=10, 5 folds."""
        n = 100
        window_size = 10
        t1 = _make_overlapping_labels(n, window_size)
        x = np.arange(n)

        cv = PurgedKFold(n_splits=5, samples_info_sets=t1, pct_embargo=0.0)

        for fold_idx, (train, test) in enumerate(cv.split(x)):
            for train_idx in train:
                train_t0 = t1.index[train_idx]
                train_t1 = t1.iloc[train_idx]
                for test_idx in test:
                    test_t0 = t1.index[test_idx]
                    test_t1 = t1.iloc[test_idx]
                    # No overlap: train window must not intersect test window
                    overlaps = train_t0 <= test_t1 and train_t1 >= test_t0
                    assert not overlaps, (
                        f"Fold {fold_idx}: train sample {train_idx} "
                        f"[{train_t0}, {train_t1}] overlaps test sample "
                        f"{test_idx} [{test_t0}, {test_t1}]"
                    )

    def test_no_overlap_large_windows(self) -> None:
        """Large label windows (30 bars) that span multiple folds."""
        n = 200
        window_size = 30
        t1 = _make_overlapping_labels(n, window_size)
        x = np.arange(n)

        cv = PurgedKFold(n_splits=5, samples_info_sets=t1, pct_embargo=0.0)

        for fold_idx, (train, test) in enumerate(cv.split(x)):
            for train_idx in train:
                train_t0 = t1.index[train_idx]
                train_t1 = t1.iloc[train_idx]
                for test_idx in test:
                    test_t0 = t1.index[test_idx]
                    test_t1 = t1.iloc[test_idx]
                    overlaps = train_t0 <= test_t1 and train_t1 >= test_t0
                    assert not overlaps, f"Fold {fold_idx}: train={train_idx} test={test_idx}"

    def test_no_overlap_many_folds(self) -> None:
        """10-fold split with overlapping windows."""
        n = 500
        window_size = 15
        t1 = _make_overlapping_labels(n, window_size)
        x = np.arange(n)

        cv = PurgedKFold(n_splits=10, samples_info_sets=t1, pct_embargo=0.0)

        for _fold_idx, (train, test) in enumerate(cv.split(x)):
            for train_idx in train:
                train_t0 = t1.index[train_idx]
                train_t1 = t1.iloc[train_idx]
                for test_idx in test:
                    test_t0 = t1.index[test_idx]
                    test_t1 = t1.iloc[test_idx]
                    overlaps = train_t0 <= test_t1 and train_t1 >= test_t0
                    assert not overlaps

    def test_train_test_indices_are_disjoint(self) -> None:
        """Train and test indices must never overlap."""
        n = 100
        t1 = _make_overlapping_labels(n, 10)
        x = np.arange(n)

        cv = PurgedKFold(n_splits=5, samples_info_sets=t1, pct_embargo=0.0)
        for train, test in cv.split(x):
            assert len(set(train) & set(test)) == 0

    def test_purging_removes_correct_samples(self) -> None:
        """Verify that purged samples are a subset of what would normally train."""
        n = 100
        t1 = _make_overlapping_labels(n, 20)
        x = np.arange(n)

        cv_purged = PurgedKFold(n_splits=5, samples_info_sets=t1, pct_embargo=0.0)

        for train, test in cv_purged.split(x):
            # Train + test + purged should cover all indices
            all_accounted = set(train) | set(test)
            # Purged samples are the missing ones
            purged = set(range(n)) - all_accounted
            # Every purged sample must have its window overlapping test
            for p in purged:
                p_t0 = t1.index[p]
                p_t1 = t1.iloc[p]
                has_overlap = False
                for t in test:
                    t_t0 = t1.index[t]
                    t_t1 = t1.iloc[t]
                    if p_t0 <= t_t1 and p_t1 >= t_t0:
                        has_overlap = True
                        break
                assert has_overlap, f"Sample {p} was purged but doesn't overlap test"
