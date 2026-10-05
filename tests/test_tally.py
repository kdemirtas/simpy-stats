"""Tests for Tally and Counter."""

import math
import statistics

import pytest

from simpy_stats.core.counter import Counter
from simpy_stats.core.tally import Tally

# ---------------------------------------------------------------------------
# Tally
# ---------------------------------------------------------------------------

class TestTally:
    def test_empty(self):
        t = Tally("x")
        assert t.n == 0
        assert math.isnan(t.mean)
        assert math.isnan(t.stdev)
        assert math.isinf(t.min) and t.min > 0
        assert math.isinf(t.max) and t.max < 0

    def test_single(self):
        t = Tally("x")
        t.observe(3.0)
        assert t.n == 1
        assert t.mean == pytest.approx(3.0)
        assert t.min == 3.0
        assert t.max == 3.0
        assert math.isnan(t.stdev)  # n < 2

    def test_multiple(self):
        data = [1.0, 3.0, 5.0, 7.0, 9.0]
        t = Tally("x")
        for x in data:
            t.observe(x)
        assert t.n == 5
        assert t.mean == pytest.approx(statistics.mean(data))
        assert t.stdev == pytest.approx(statistics.stdev(data), rel=1e-9)
        assert t.min == 1.0
        assert t.max == 9.0

    def test_min_max(self):
        t = Tally("x")
        for x in [5.0, -2.0, 10.0, 3.0]:
            t.observe(x)
        assert t.min == -2.0
        assert t.max == 10.0

    def test_snapshot_keys(self):
        t = Tally("wait")
        t.observe(1.0)
        t.observe(2.0)
        snap = t.snapshot()
        assert "wait.n" in snap
        assert "wait.mean" in snap
        assert "wait.stdev" in snap
        assert "wait.min" in snap
        assert "wait.max" in snap
        assert snap["wait.n"] == 2.0

    def test_snapshot_empty_nan(self):
        t = Tally("empty")
        snap = t.snapshot()
        assert math.isnan(snap["empty.min"])
        assert math.isnan(snap["empty.max"])

    def test_observe_with_timestamp(self):
        t = Tally("x")
        t.observe(4.0, t=10.0)
        assert t.n == 1
        assert t.mean == pytest.approx(4.0)

    def test_reset(self):
        t = Tally("x")
        t.observe(1.0)
        t.reset()
        assert t.n == 0

    def test_repr(self):
        t = Tally("x")
        t.observe(1.0)
        assert "Tally" in repr(t)


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------

class TestCounter:
    def test_initial_zero(self):
        c = Counter("arrivals")
        assert c.count == 0

    def test_inc_default(self):
        c = Counter("arrivals")
        c.inc()
        c.inc()
        assert c.count == 2

    def test_inc_k(self):
        c = Counter("arrivals")
        c.inc(5)
        assert c.count == 5

    def test_inc_negative_raises(self):
        c = Counter("arrivals")
        with pytest.raises(ValueError):
            c.inc(-1)

    def test_rate(self):
        c = Counter("arrivals")
        c.inc(10)
        assert c.rate(5.0) == pytest.approx(2.0)

    def test_rate_zero_duration_raises(self):
        c = Counter("arrivals")
        with pytest.raises(ValueError):
            c.rate(0.0)

    def test_snapshot(self):
        c = Counter("arrivals")
        c.inc(7)
        snap = c.snapshot()
        assert snap["arrivals.count"] == 7.0

    def test_reset(self):
        c = Counter("arrivals")
        c.inc(5)
        c.reset()
        assert c.count == 0

    def test_repr(self):
        c = Counter("arrivals")
        assert "Counter" in repr(c)


def test_an_empty_tally_reports_no_mean():
    snap = Tally("wait").snapshot()
    assert snap["wait.n"] == 0
    for key in ("wait.mean", "wait.stdev", "wait.min", "wait.max"):
        assert math.isnan(snap[key])


def test_a_nan_observation_is_refused_and_changes_nothing():
    t = Tally("wait")
    t.observe(1.0)
    with pytest.raises(ValueError):
        t.observe(math.nan)
    t.observe(3.0)
    assert (t.n, t.mean, t.min, t.max) == (2, 2.0, 1.0, 3.0)
