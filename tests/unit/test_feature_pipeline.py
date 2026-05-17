"""Tests for FeaturePipeline: topological sort, caching, dependency resolution."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tessera.features.base import Feature, FeaturePipeline, _topological_sort


class DummyA(Feature):
    name = "a"
    dependencies: list[str] = []
    version = "0.1.0"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        return pd.Series(df["x"] * 2, name=self.name)


class DummyB(Feature):
    name = "b"
    dependencies = ["a"]
    version = "0.1.0"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        return pd.Series(df["a"] + 1, name=self.name)


class DummyC(Feature):
    name = "c"
    dependencies = ["b"]
    version = "0.1.0"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        return pd.Series(df["b"] * 3, name=self.name)


class TestTopologicalSort:
    def test_basic_order(self) -> None:
        features = [DummyC(), DummyA(), DummyB()]
        sorted_feats = _topological_sort(features)
        names = [f.name for f in sorted_feats]
        assert names.index("a") < names.index("b")
        assert names.index("b") < names.index("c")

    def test_no_dependencies(self) -> None:
        features = [DummyA()]
        sorted_feats = _topological_sort(features)
        assert len(sorted_feats) == 1

    def test_cyclic_raises(self) -> None:
        class CyclicA(Feature):
            name = "cyc_a"
            dependencies = ["cyc_b"]
            version = "0.1.0"

            def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
                return pd.Series(dtype=float)

        class CyclicB(Feature):
            name = "cyc_b"
            dependencies = ["cyc_a"]
            version = "0.1.0"

            def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
                return pd.Series(dtype=float)

        with pytest.raises(ValueError, match="Cyclic dependency"):
            _topological_sort([CyclicA(), CyclicB()])


class TestFeaturePipeline:
    def test_compute_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FeaturePipeline([DummyC(), DummyA(), DummyB()], cache_dir=Path(tmpdir))
            df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
            result = pipeline.compute(df, use_cache=False)

            # a = x*2
            np.testing.assert_array_equal(result["a"].values, [2.0, 4.0, 6.0])
            # b = a+1
            np.testing.assert_array_equal(result["b"].values, [3.0, 5.0, 7.0])
            # c = b*3
            np.testing.assert_array_equal(result["c"].values, [9.0, 15.0, 21.0])

    def test_caching(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FeaturePipeline([DummyA()], cache_dir=Path(tmpdir))
            df = pd.DataFrame(
                {
                    "x": [1.0, 2.0, 3.0],
                    "event_time": pd.date_range("2023-01-01", periods=3, freq="1min"),
                }
            )

            # First compute should write cache
            result1 = pipeline.compute(df, symbol="TEST", use_cache=True)
            cache_path = Path(tmpdir) / "a" / "v0.1.0" / "TEST" / "2023-01-01.parquet"
            assert cache_path.exists()

            # Second compute should read from cache
            result2 = pipeline.compute(df, symbol="TEST", use_cache=True)
            np.testing.assert_array_equal(result1["a"].values, result2["a"].values)
