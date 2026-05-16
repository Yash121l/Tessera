"""Feature engineering base class.

All features inherit from FeatureBase and implement the compute method.
Reference: AFML Chapter 5 — Fractionally Differentiated Features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FeatureBase(ABC):
    """Abstract base class for all feature generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this feature."""
        ...

    @property
    @abstractmethod
    def lookback(self) -> int:
        """Minimum number of bars required to compute this feature."""
        ...

    @abstractmethod
    def compute(self, df: Any) -> Any:
        """Compute feature values from an OHLCV DataFrame.

        Args:
            df: DataFrame with columns [open, high, low, close, volume].

        Returns:
            Series or DataFrame of computed feature values.
        """
        ...
