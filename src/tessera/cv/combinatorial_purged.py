"""Combinatorial Purged K-Fold cross-validation (AFML §12).

Generates all C(n_splits, n_test_splits) train/test combinations, with
purging and embargo applied to each. Enables construction of multiple
backtest paths for deflated Sharpe ratio estimation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class CombinatorialPurgedKFold(BaseCrossValidator):
    """Combinatorial purged k-fold CV (CPCV).

    Args:
        n_splits: Total number of folds (N).
        n_test_splits: Number of folds used as test in each split (k).
        samples_info_sets: Series mapping sample index to label end time (t1).
        pct_embargo: Fraction of total samples to embargo after each test group.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        samples_info_sets: pd.Series | None = None,  # type: ignore[type-arg]
        pct_embargo: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo

    def split(
        self,
        X: Any,  # noqa: N803
        y: Any = None,
        groups: Any = None,
    ) -> Any:
        """Generate all C(N, k) purged train/test splits."""
        if self.samples_info_sets is None:
            msg = "samples_info_sets must be provided"
            raise ValueError(msg)

        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo_size = int(n_samples * self.pct_embargo)

        # Create fold boundaries
        fold_size = n_samples // self.n_splits
        folds: list[np.ndarray] = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_samples
            folds.append(indices[start:end])

        t1 = self.samples_info_sets

        for test_fold_ids in combinations(range(self.n_splits), self.n_test_splits):
            test_indices = np.concatenate([folds[i] for i in test_fold_ids])
            test_set = set(test_indices.tolist())

            # Per-group boundaries for embargo
            test_groups_end_indices = [folds[i][-1] for i in test_fold_ids]

            train_candidates = np.array([idx for idx in indices if idx not in test_set])

            # Purge: remove train samples whose label window overlaps test
            purged = set()
            for idx in train_candidates:
                sample_t0 = t1.index[idx]
                sample_t1 = t1.iloc[idx]
                # Check overlap with each test group
                for fold_id in test_fold_ids:
                    fold_indices = folds[fold_id]
                    fold_t0 = t1.index[fold_indices[0]]
                    fold_t1 = t1.iloc[fold_indices].max()
                    if sample_t0 <= fold_t1 and sample_t1 >= fold_t0:
                        purged.add(idx)
                        break

            # Embargo after each test group
            embargo_set: set[int] = set()
            for end_idx in test_groups_end_indices:
                embargo_start = end_idx + 1
                embargo_end = min(embargo_start + embargo_size, n_samples)
                embargo_set.update(range(embargo_start, embargo_end))

            train_indices = np.array(
                [idx for idx in train_candidates if idx not in purged and idx not in embargo_set]
            )

            yield train_indices, test_indices

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # noqa: N803
        from math import comb

        return comb(self.n_splits, self.n_test_splits)

    @staticmethod
    def get_num_backtest_paths(n_splits: int, n_test_splits: int) -> int:
        """Number of independent backtest paths producible by CPCV.

        Formula: N! / ((N-k)! * k!) * k / N  simplified to C(N-1, k-1).
        """
        from math import comb

        return comb(n_splits - 1, n_test_splits - 1)


def compute_backtest_paths(
    cv: CombinatorialPurgedKFold,
    returns: pd.Series,  # type: ignore[type-arg]
    predictions: dict[int, pd.Series],  # type: ignore[type-arg]
) -> pd.DataFrame:
    """Stitch out-of-sample predictions into backtest paths.

    Args:
        cv: The CombinatorialPurgedKFold instance used for splitting.
        returns: Full returns series.
        predictions: Dict mapping split index → predicted returns (OOS).

    Returns:
        DataFrame with one column per backtest path, containing the
        path's cumulative return contribution.
    """
    n_paths = cv.get_num_backtest_paths(cv.n_splits, cv.n_test_splits)

    # Assign each split's test predictions to paths
    # Each fold appears in C(N-1, k-1) splits as test
    path_returns: dict[int, list[pd.Series]] = {  # type: ignore[type-arg]
        i: [] for i in range(n_paths)
    }

    for split_id, (_train, _test) in enumerate(cv.split(np.arange(len(returns)))):
        if split_id in predictions:
            pred = predictions[split_id]
            path_id = split_id % n_paths
            path_returns[path_id].append(pred)

    result = pd.DataFrame(index=returns.index)
    for path_id, series_list in path_returns.items():
        if series_list:
            combined = pd.concat(series_list).sort_index()
            combined = combined[~combined.index.duplicated(keep="first")]
            result[f"path_{path_id}"] = combined.reindex(returns.index)
        else:
            result[f"path_{path_id}"] = np.nan

    return result
