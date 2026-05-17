"""Property tests: purging invariant holds for arbitrary label windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from tessera.cv.purged_kfold import PurgedKFold


@st.composite
def label_windows(draw: st.DrawFn) -> tuple[int, int, pd.Series]:  # type: ignore[type-arg]
    """Generate random (n_samples, n_splits, t1) with valid label windows."""
    n = draw(st.integers(min_value=20, max_value=200))
    n_splits = draw(st.integers(min_value=2, max_value=min(10, n // 2)))
    max_window = draw(st.integers(min_value=1, max_value=max(1, n // 3)))

    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    windows = []
    for i in range(n):
        w = draw(st.integers(min_value=0, max_value=max_window))
        end = min(i + w, n - 1)
        windows.append(idx[end])

    t1 = pd.Series(windows, index=idx)
    return n, n_splits, t1


@given(data=label_windows())
@settings(max_examples=100, deadline=None)
def test_purging_invariant_always_holds(
    data: tuple[int, int, pd.Series],  # type: ignore[type-arg]
) -> None:
    """For ANY valid label windows and fold configuration, no train window
    may overlap any test window after purging.
    """
    n, n_splits, t1 = data
    x = np.arange(n)

    cv = PurgedKFold(n_splits=n_splits, samples_info_sets=t1, pct_embargo=0.0)

    for train, test in cv.split(x):
        for ti in train:
            t_t0 = t1.index[ti]
            t_t1 = t1.iloc[ti]
            for tj in test:
                test_t0 = t1.index[tj]
                test_t1 = t1.iloc[tj]
                assert not (t_t0 <= test_t1 and t_t1 >= test_t0)


@given(data=label_windows())
@settings(max_examples=50, deadline=None)
def test_train_test_always_disjoint(
    data: tuple[int, int, pd.Series],  # type: ignore[type-arg]
) -> None:
    """Train and test index sets must always be disjoint."""
    n, n_splits, t1 = data
    x = np.arange(n)

    cv = PurgedKFold(n_splits=n_splits, samples_info_sets=t1, pct_embargo=0.0)

    for train, test in cv.split(x):
        assert len(set(train) & set(test)) == 0


@given(data=label_windows())
@settings(max_examples=50, deadline=None)
def test_correct_number_of_folds(
    data: tuple[int, int, pd.Series],  # type: ignore[type-arg]
) -> None:
    """Must produce exactly n_splits folds."""
    n, n_splits, t1 = data
    x = np.arange(n)

    cv = PurgedKFold(n_splits=n_splits, samples_info_sets=t1, pct_embargo=0.0)
    folds = list(cv.split(x))
    assert len(folds) == n_splits
