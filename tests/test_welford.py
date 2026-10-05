"""Tests for the Welford streaming accumulator."""

import math
import statistics

import pytest

from simpy_stats.core.welford import Welford


def test_empty():
    w = Welford()
    assert w.n == 0
    assert math.isnan(w.mean)
    assert w.m2 == 0.0
    assert math.isnan(w.var_sample)
    assert math.isnan(w.var_pop)
    assert math.isnan(w.stdev_sample)


def test_single_observation():
    w = Welford()
    w.update(5.0)
    assert w.n == 1
    assert w.mean == pytest.approx(5.0)
    assert math.isnan(w.var_sample)  # n<2


def test_known_sequence():
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    w = Welford()
    for x in data:
        w.update(x)

    assert w.n == len(data)
    assert w.mean == pytest.approx(statistics.mean(data))
    assert w.var_sample == pytest.approx(statistics.variance(data), rel=1e-9)
    assert w.stdev_sample == pytest.approx(statistics.stdev(data), rel=1e-9)


def test_integers():
    data = list(range(1, 11))
    w = Welford()
    for x in data:
        w.update(x)
    assert w.mean == pytest.approx(5.5)
    assert w.stdev_sample == pytest.approx(statistics.stdev(data), rel=1e-9)


def test_population_variance():
    data = [1.0, 2.0, 3.0]
    w = Welford()
    for x in data:
        w.update(x)
    expected_pop = sum((x - statistics.mean(data)) ** 2 for x in data) / len(data)
    assert w.var_pop == pytest.approx(expected_pop)


def test_repr():
    w = Welford()
    w.update(1.0)
    assert "Welford" in repr(w)
    assert "n=1" in repr(w)


def test_large_sequence_stability():
    """Welford stays accurate for large n (no catastrophic cancellation)."""
    import random
    rng = random.Random(42)
    # Offset data by a large constant to stress numerical stability
    data = [1e8 + rng.gauss(0, 1) for _ in range(10_000)]
    w = Welford()
    for x in data:
        w.update(x)
    assert w.mean == pytest.approx(statistics.mean(data), rel=1e-6)
    assert w.stdev_sample == pytest.approx(statistics.stdev(data), rel=1e-4)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_observation_is_refused(bad):
    w = Welford()
    w.update(1.0)
    with pytest.raises(ValueError):
        w.update(bad)
    assert w.n == 1
    assert w.mean == 1.0
