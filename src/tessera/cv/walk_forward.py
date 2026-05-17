"""Walk-forward train/test splitting.

Provides expanding-window and rolling-window walk-forward splits for
time series backtesting without purging (used when label overlap is
not a concern, e.g., for non-overlapping features).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class WalkForwardSplit(BaseCrossValidator):
    """Walk-forward cross-validation splitter.

    Args:
        train_size: Number of samples in each training window.
        test_size: Number of samples in each test window.
        step_size: Number of samples to advance between splits.
        anchor: 'expanding' grows the train window; 'rolling' keeps it fixed.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        anchor: Literal["expanding", "rolling"] = "expanding",
    ) -> None:
        super().__init__()
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.anchor = anchor

    def split(
        self,
        X: Any,  # noqa: N803
        y: Any = None,
        groups: Any = None,
    ) -> Any:
        """Generate train/test index pairs."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        test_start = self.train_size
        while test_start + self.test_size <= n_samples:
            test_end = test_start + self.test_size

            if self.anchor == "expanding":
                train_indices = indices[:test_start]
            else:
                train_start = test_start - self.train_size
                train_indices = indices[train_start:test_start]

            test_indices = indices[test_start:test_end]
            yield train_indices, test_indices

            test_start += self.step_size

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # noqa: N803
        if X is None:
            return 0
        n_samples = len(X)
        n_splits = 0
        test_start = self.train_size
        while test_start + self.test_size <= n_samples:
            n_splits += 1
            test_start += self.step_size
        return n_splits
