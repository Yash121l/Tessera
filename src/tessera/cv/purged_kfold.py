"""Purged K-Fold cross-validation (AFML §7.4).

Standard KFold leaks information when labels have overlapping time windows.
PurgedKFold removes (purges) training samples whose label window overlaps any
test sample, and applies an additional embargo period after each test fold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """K-Fold CV with purging and embargo for financial time series.

    Args:
        n_splits: Number of folds.
        samples_info_sets: Series mapping each sample index to its label
            end time (t1). Index = sample start, value = label end.
        pct_embargo: Fraction of total samples to embargo after each test fold.
    """

    def __init__(
        self,
        n_splits: int = 5,
        samples_info_sets: pd.Series | None = None,  # type: ignore[type-arg]
        pct_embargo: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_splits = n_splits
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo

    def split(
        self,
        X: Any,  # noqa: N803
        y: Any = None,
        groups: Any = None,
    ) -> Any:
        """Generate purged train/test indices.

        Yields (train_indices, test_indices) for each fold.
        """
        if self.samples_info_sets is None:
            msg = "samples_info_sets must be provided"
            raise ValueError(msg)

        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo_size = int(n_samples * self.pct_embargo)

        fold_size = n_samples // self.n_splits
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_samples
            folds.append(indices[start:end])

        t1 = self.samples_info_sets

        for i in range(self.n_splits):
            test_indices = folds[i]
            test_start_idx = test_indices[0]
            test_end_idx = test_indices[-1]

            # Test fold time boundaries
            test_t0_min = t1.index[test_start_idx]
            test_t1_max = t1.iloc[test_start_idx : test_end_idx + 1].max()

            # Build train set: start with all non-test indices
            train_candidates = np.array([idx for idx in indices if idx not in set(test_indices)])

            # Purge: remove training samples whose label window overlaps test
            purged = set()
            for idx in train_candidates:
                sample_t0 = t1.index[idx]
                sample_t1 = t1.iloc[idx]
                # Overlap: sample window [sample_t0, sample_t1] intersects
                # test window [test_t0_min, test_t1_max]
                if sample_t0 <= test_t1_max and sample_t1 >= test_t0_min:
                    purged.add(idx)

            # Embargo: remove samples immediately after the test fold
            embargo_start = test_end_idx + 1
            embargo_end = min(embargo_start + embargo_size, n_samples)
            embargo_set = set(range(embargo_start, embargo_end))

            train_indices = np.array(
                [idx for idx in train_candidates if idx not in purged and idx not in embargo_set]
            )

            yield train_indices, test_indices

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # noqa: N803
        return self.n_splits
