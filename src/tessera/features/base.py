"""Feature engineering base classes and pipeline.

Provides the abstract Feature class and FeaturePipeline which resolves
dependencies via topological sort, computes features in order, and caches
results per-symbol per-day to Parquet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import pandas as pd
import structlog

from tessera.config import TesseraSettings

logger = structlog.get_logger(__name__)


class Feature(ABC):
    """Abstract base class for all feature generators."""

    name: str = ""
    point_in_time_safe: bool = True
    version: str = "0.1.0"
    dependencies: list[str] = []

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        """Compute feature values from a DataFrame.

        Must only use data available at or before each row's timestamp.
        """
        ...


def _topological_sort(features: list[Feature]) -> list[Feature]:
    """Resolve feature dependencies via Kahn's algorithm."""
    name_to_feature = {f.name: f for f in features}
    in_degree: dict[str, int] = {f.name: 0 for f in features}
    dependents: dict[str, list[str]] = {f.name: [] for f in features}

    for f in features:
        for dep in f.dependencies:
            if dep in name_to_feature:
                in_degree[f.name] += 1
                dependents[dep].append(f.name)

    queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
    result: list[Feature] = []

    while queue:
        name = queue.popleft()
        result.append(name_to_feature[name])
        for dependent in dependents[name]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(features):
        computed = {f.name for f in result}
        missing = [f.name for f in features if f.name not in computed]
        msg = f"Cyclic dependency detected among features: {missing}"
        raise ValueError(msg)

    return result


class FeaturePipeline:
    """Computes features in dependency order with per-symbol per-day caching."""

    def __init__(self, features: list[Feature], cache_dir: Path | None = None) -> None:
        self.features = _topological_sort(features)
        if cache_dir is None:
            settings = TesseraSettings()
            self._cache_dir = settings.data_root / "features"
        else:
            self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, feature: Feature, symbol: str, date: str) -> Path:
        return self._cache_dir / feature.name / f"v{feature.version}" / symbol / f"{date}.parquet"

    def _read_cache(self, feature: Feature, symbol: str, date: str) -> pd.Series | None:  # type: ignore[type-arg]
        path = self._cache_path(feature, symbol, date)
        if path.exists():
            df = pd.read_parquet(path)
            if feature.name in df.columns:
                return df[feature.name]
        return None

    def _write_cache(
        self,
        feature: Feature,
        symbol: str,
        date: str,
        series: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        path = self._cache_path(feature, symbol, date)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({feature.name: series})
        df.to_parquet(path)

    def compute(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Run the full pipeline, returning a DataFrame with all features as columns."""
        result = df.copy()

        for feature in self.features:
            logger.debug("computing_feature", feature=feature.name, symbol=symbol)

            if use_cache and "event_time" in result.columns:
                dates = result["event_time"].dt.strftime("%Y-%m-%d").unique()
                if len(dates) == 1:
                    cached = self._read_cache(feature, symbol, dates[0])
                    if cached is not None and len(cached) == len(result):
                        result[feature.name] = cached.values
                        continue

            series = feature.compute(result)
            result[feature.name] = series.values

            if use_cache and "event_time" in result.columns:
                dates = result["event_time"].dt.strftime("%Y-%m-%d").unique()
                if len(dates) == 1:
                    self._write_cache(feature, symbol, dates[0], series)

        return result

    def compute_multi_day(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Compute features day-by-day for multi-day DataFrames."""
        if "event_time" not in df.columns:
            return self.compute(df, symbol, use_cache=False)

        df = df.sort_values("event_time").reset_index(drop=True)
        dates = df["event_time"].dt.date.unique()

        results: list[pd.DataFrame] = []
        for date in dates:
            mask = df["event_time"].dt.date == date
            day_df = df[mask].copy()
            day_result = self.compute(day_df, symbol, use_cache)
            results.append(day_result)

        return pd.concat(results, ignore_index=True)
