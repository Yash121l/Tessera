"""Test that seed_everything produces reproducible results."""

from __future__ import annotations

import numpy as np

from tessera.config import seed_everything


def test_same_seed_same_results() -> None:
    """Calling seed_everything twice with same seed must yield identical draws."""
    seed_everything(42)
    a1 = np.random.rand(10)  # noqa: NPY002

    seed_everything(42)
    a2 = np.random.rand(10)  # noqa: NPY002

    np.testing.assert_array_equal(a1, a2)


def test_different_seed_different_results() -> None:
    """Different seeds must produce different draws."""
    seed_everything(42)
    a1 = np.random.rand(10)  # noqa: NPY002

    seed_everything(99)
    a2 = np.random.rand(10)  # noqa: NPY002

    assert not np.array_equal(a1, a2)


def test_seed_everything_returns_seed() -> None:
    """seed_everything should return the seed for logging convenience."""
    result = seed_everything(123)
    assert result == 123


def test_python_random_seeded() -> None:
    """Python's random module should also be seeded."""
    import random

    seed_everything(42)
    r1 = [random.random() for _ in range(5)]

    seed_everything(42)
    r2 = [random.random() for _ in range(5)]

    assert r1 == r2
